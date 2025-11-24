"""
Utility functions for training
"""
import math

import torch
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


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


@torch.no_grad()
def extract_features(model, dataloader, device):
    """
    Assumes the dataloader returns a batch of images, labels, and filenames
    """
    model.eval()
    all_features = []
    all_labels = []
    all_filenames = []

    for batch in tqdm(dataloader, desc="Extracting features"):
        assert len(batch) == 3, "Dataloader must return a batch of images, labels, and filenames"
        images, labels, filenames = batch
        images = images.to(device)
        features = model(images)

        # Use [CLS] token
        cls_features = features[:, 0]  # Shape: [B, D]

        all_features.append(cls_features.cpu())
        all_labels.append(labels.cpu())
        all_filenames.extend(filenames)

    all_features = torch.cat(all_features, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    return all_features, all_labels, all_filenames



class KNNClassifier:
    def __init__(self, k: int = 20):
        self.knn = KNeighborsClassifier(n_neighbors=k, metric='cosine', n_jobs=-1)


    def normalize_features(self, features):
        return features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)


    def train(self, features, labels):
        # Normalize features
        features = self.normalize_features(features)
        self.knn.fit(features, labels)


    def predict(self, features):
        # Normalize features
        features = self.normalize_features(features)
        return self.knn.predict(features)


    def evaluate(self, features, labels):
        pred = self.predict(features)
        return accuracy_score(labels, pred)

