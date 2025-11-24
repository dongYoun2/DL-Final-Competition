"""
Utility functions for training
"""
import math

import torch
from torch.optim.lr_scheduler import _LRScheduler


class CosineWithWarmupScheduler(_LRScheduler):
    """
    Cosine annealing with linear warmup that treats `min_lr` as an absolute
    floor. Prevents the LR from collapsing toward zero when `min_lr` is
    specified in absolute units (e.g., 1e-6).
    """

    def __init__(
        self,
        optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        self.num_warmup_steps = max(0, num_warmup_steps)
        self.num_training_steps = max(num_training_steps, 1)
        if self.num_warmup_steps > self.num_training_steps:
            self.num_warmup_steps = self.num_training_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        current_step = self.last_epoch

        # Warmup: linearly scale from 0 -> base_lr
        if current_step < self.num_warmup_steps:
            warmup_progress = float(current_step + 1) / max(1, self.num_warmup_steps)
            return [base_lr * warmup_progress for base_lr in self.base_lrs]

        if self.num_training_steps == self.num_warmup_steps:
            return [self.min_lr for _ in self.base_lrs]

        progress = float(current_step - self.num_warmup_steps) / max(
            1, self.num_training_steps - self.num_warmup_steps
        )
        progress = min(max(progress, 0.0), 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

        return [
            self.min_lr + (base_lr - self.min_lr) * cosine_decay
            for base_lr in self.base_lrs
        ]


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    num_cycles=0.5,  # kept for backward compatibility
    min_lr=0.0,
):
    """
    Create a cosine scheduler with linear warmup that decays from the optimizer's
    initial LR down to `min_lr`.
    """
    return CosineWithWarmupScheduler(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr=min_lr,
    )


class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res




