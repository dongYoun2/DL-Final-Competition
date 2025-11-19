"""
Main training script for self-supervised learning (IBOT/DINO)
"""
import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from tqdm import tqdm
import numpy as np

from data_loader import create_dataloader, get_transforms
from models import (
    create_ibot_model,
    create_dino_model,
    update_teacher,
    DINOLoss,
)
from utils import (
    setup_distributed,
    cleanup_distributed,
    save_checkpoint,
    load_checkpoint,
    get_cosine_schedule_with_warmup,
    AverageMeter,
)


class Trainer:
    """
    Trainer for self-supervised vision models
    """
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_rank = int(os.environ.get('RANK', 0))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))

        # Setup distributed training
        if self.world_size > 1:
            setup_distributed()
            self.device = torch.device(f'cuda:{self.local_rank}')

        # Set random seed
        torch.manual_seed(cfg.seed + self.global_rank)
        np.random.seed(cfg.seed + self.global_rank)

        # Create models
        self.create_models()

        # Create dataloaders
        self.create_dataloaders()

        # Create optimizer and scheduler
        self.create_optimizer()

        # Create loss function
        self.create_loss()

        # Setup logging
        self.setup_logging()

        # Setup checkpointing with experiment-specific directory
        base_checkpoint_dir = Path(cfg.checkpoint.save_dir)
        self.checkpoint_dir = base_checkpoint_dir / cfg.experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.epoch = 0
        self.global_step = 0

        # Auto-resume logic
        self._handle_resume()

        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if cfg.training.mixed_precision else None

    def create_models(self):
        """Create student and teacher models"""
        model_name = self.cfg.model.name

        if model_name == 'ibot':
            self.student, self.teacher = create_ibot_model(self.cfg)
        elif model_name == 'dino_v2' or model_name == 'dino_v3':
            self.student, self.teacher = create_dino_model(self.cfg)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        self.student = self.student.to(self.device)
        self.teacher = self.teacher.to(self.device)

        # Wrap with DDP
        if self.world_size > 1:
            self.student = DDP(
                self.student,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
            )

    def create_dataloaders(self):
        """Create data loaders"""
        transform = get_transforms(self.cfg)

        self.train_loader = create_dataloader(
            dataset_name=self.cfg.data.dataset_name,
            split=self.cfg.data.train_split,
            batch_size=self.cfg.training.batch_size,
            num_workers=self.cfg.training.num_workers,
            transform=transform,
            max_samples=self.cfg.data.num_samples,
            cache_dir=self.cfg.data.cache_dir,
            streaming=self.cfg.data.streaming,
            pin_memory=self.cfg.training.pin_memory,
            image_key=self.cfg.data.image_key,
            prefetch_factor=self.cfg.training.get('prefetch_factor', 4),
            persistent_workers=self.cfg.training.get('persistent_workers', True),
        )

    def create_optimizer(self):
        """Create optimizer and learning rate scheduler"""
        # Get parameters
        params = self.student.parameters()

        # Create optimizer
        if self.cfg.optimizer.name == 'adamw':
            self.optimizer = torch.optim.AdamW(
                params,
                lr=self.cfg.optimizer.lr,
                betas=self.cfg.optimizer.betas,
                eps=self.cfg.optimizer.eps,
                weight_decay=self.cfg.optimizer.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.cfg.optimizer.name}")

        # Create scheduler
        num_training_steps = len(self.train_loader) * self.cfg.training.num_epochs
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.cfg.scheduler.warmup_epochs * len(self.train_loader),
            num_training_steps=num_training_steps,
            min_lr=self.cfg.scheduler.min_lr,
        )

    def create_loss(self):
        """Create loss function"""
        model_name = self.cfg.model.name

        if model_name == 'ibot':
            model_cfg = self.cfg.model.ibot
        else:  # dino_v2 or dino_v3
            model_cfg = self.cfg.model.dino

        self.criterion = DINOLoss(
            out_dim=model_cfg.out_dim,
            ncrops=2 + model_cfg.local_crops_number,
            warmup_teacher_temp=model_cfg.warmup_teacher_temp,
            teacher_temp=model_cfg.teacher_temp,
            warmup_teacher_temp_epochs=model_cfg.warmup_teacher_temp_epochs,
            nepochs=self.cfg.training.num_epochs,
            student_temp=model_cfg.student_temp,
        ).to(self.device)

    def setup_logging(self):
        """Setup logging"""
        # Simple console logging - logs are captured by SLURM
        if self.global_rank == 0:
            print(f"Experiment: {self.cfg.experiment_name}")
            print(f"Log directory: {self.cfg.logging.log_dir}")

    def train_epoch(self):
        """Train for one epoch"""
        self.student.train()
        self.teacher.eval()

        loss_meter = AverageMeter()

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.epoch}",
            disable=self.global_rank != 0,
        )

        for batch_idx, (images, _) in enumerate(pbar):
            # Debug: print image structure on first batch
            if batch_idx == 0 and self.global_rank == 0:
                print(f"\n[DEBUG TRAIN] Images type: {type(images)}")
                if isinstance(images, list):
                    print(f"[DEBUG TRAIN] Number of crops: {len(images)}")
                    print(f"[DEBUG TRAIN] First crop shape: {images[0].shape}")
                else:
                    print(f"[DEBUG TRAIN] Images shape (not a list): {images.shape}")
                    print(f"[DEBUG TRAIN] ERROR: Expected list of crops, got tensor!")

            # Images should be a list of crops
            # If it's a tensor, something went wrong with the collate function
            if not isinstance(images, list):
                raise TypeError(f"Expected images to be a list of crop batches, got {type(images)}")

            images = [img.to(self.device, non_blocking=True) for img in images]

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.cfg.training.mixed_precision):
                # Student forward (all crops)
                student_output = self.student(images)

                # Teacher forward (only global crops)
                teacher_output = self.teacher(images[:2])

                # Compute loss
                loss = self.criterion(student_output, teacher_output, self.epoch)

            # Backward pass
            self.optimizer.zero_grad()

            if self.scaler is not None:
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.cfg.training.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.student.parameters(),
                        self.cfg.training.gradient_clip
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()

                # Gradient clipping
                if self.cfg.training.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.student.parameters(),
                        self.cfg.training.gradient_clip
                    )

                self.optimizer.step()

            # Update learning rate
            self.scheduler.step()

            # Update teacher
            model_name = self.cfg.model.name
            if model_name == 'ibot':
                momentum = self.cfg.model.ibot.momentum_teacher
            else:
                momentum = self.cfg.model.dino.momentum_teacher

            with torch.no_grad():
                m = momentum
                student_model = self.student.module if self.world_size > 1 else self.student
                update_teacher(student_model, self.teacher, m)

            # Update metrics
            loss_meter.update(loss.item())

            # Logging
            if self.global_rank == 0:
                pbar.set_postfix({
                    'loss': f'{loss_meter.avg:.4f}',
                    'lr': f'{self.scheduler.get_last_lr()[0]:.6f}',
                })

                if self.global_step % self.cfg.logging.log_frequency == 0:
                    print(f"Step {self.global_step} - Loss: {loss_meter.avg:.4f}, LR: {self.scheduler.get_last_lr()[0]:.6f}")

            self.global_step += 1

        return loss_meter.avg

    def _handle_resume(self):
        """Handle automatic resume logic"""
        resume_path = self.cfg.checkpoint.resume_from

        # Case 1: Explicit checkpoint path provided
        if resume_path and resume_path != "auto" and Path(resume_path).exists():
            if self.global_rank == 0:
                print(f"📂 Resuming from explicit checkpoint: {resume_path}")
            self.load_checkpoint(resume_path)
            return

        # Case 2: Auto-resume enabled (for SLURM requeue)
        if self.cfg.checkpoint.auto_resume or resume_path == "auto":
            latest_checkpoint = self._find_latest_checkpoint()
            if latest_checkpoint:
                if self.global_rank == 0:
                    print(f"🔄 Auto-resuming from: {latest_checkpoint}")
                self.load_checkpoint(latest_checkpoint)
                return

        # Case 3: Starting fresh
        if self.global_rank == 0:
            print(f"🆕 Starting new experiment: {self.cfg.experiment_name}")
            print(f"📁 Checkpoints will be saved to: {self.checkpoint_dir}")

    def _find_latest_checkpoint(self):
        """Find the latest checkpoint in the experiment directory"""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pth"))

        if not checkpoint_files:
            return None

        # Sort by epoch number
        def get_epoch_num(path):
            try:
                return int(path.stem.split('_')[-1])
            except:
                return -1

        latest = max(checkpoint_files, key=get_epoch_num)
        return str(latest)

    def train(self):
        """Main training loop"""
        if self.global_rank == 0:
            print(f"\n{'='*80}")
            print(f"Training Configuration")
            print(f"{'='*80}")
            print(f"Experiment: {self.cfg.experiment_name}")
            print(f"Starting epoch: {self.epoch}")
            print(f"Target epochs: {self.cfg.training.num_epochs}")
            print(f"Checkpoint dir: {self.checkpoint_dir}")
            print(f"GPUs: {self.world_size}")
            print(f"{'='*80}\n")

        for epoch in range(self.epoch, self.cfg.training.num_epochs):
            self.epoch = epoch

            # Train for one epoch
            avg_loss = self.train_epoch()

            # Save checkpoint
            if (self.global_rank == 0 and
                epoch % self.cfg.checkpoint.save_frequency == 0):
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')

            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

        # Save final checkpoint
        if self.global_rank == 0:
            self.save_checkpoint('checkpoint_final.pth')

        # Cleanup
        if self.world_size > 1:
            cleanup_distributed()

        print("Training completed!")

    def save_checkpoint(self, filename):
        """Save training checkpoint"""
        student_model = self.student.module if self.world_size > 1 else self.student

        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'student_state_dict': student_model.state_dict(),
            'teacher_state_dict': self.teacher.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': OmegaConf.to_container(self.cfg, resolve=True),
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        student_model = self.student.module if self.world_size > 1 else self.student
        student_model.load_state_dict(checkpoint['student_state_dict'])
        self.teacher.load_state_dict(checkpoint['teacher_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']

        print(f"Checkpoint loaded from {checkpoint_path}")


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    """Main entry point"""
    print("=" * 80)
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Create trainer
    trainer = Trainer(cfg)

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()




