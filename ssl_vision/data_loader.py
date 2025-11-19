"""
Data loading utilities using HuggingFace datasets
"""
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Optional, List, Tuple
import random


class MultiCropTransform:
    """
    Multi-crop augmentation for self-supervised learning (IBOT/DINO)
    """
    def __init__(
        self,
        global_crops_scale: Tuple[float, float],
        local_crops_scale: Tuple[float, float],
        local_crops_number: int,
        global_crops_number: int = 2,
        size: int = 224,
        color_jitter: float = 0.4,
        grayscale_prob: float = 0.2,
        gaussian_blur_prob: float = 0.5,
        solarization_prob: float = 0.2,
    ):
        self.global_crops_number = global_crops_number
        self.local_crops_number = local_crops_number

        # Normalization (ImageNet stats)
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Global crop transformation
        self.global_transfo = transforms.Compose([
            transforms.RandomResizedCrop(
                size,
                scale=global_crops_scale,
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(
                    brightness=color_jitter,
                    contrast=color_jitter,
                    saturation=color_jitter,
                    hue=color_jitter/4
                )],
                p=0.8
            ),
            transforms.RandomGrayscale(p=grayscale_prob),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))],
                p=gaussian_blur_prob
            ),
            transforms.ToTensor(),
            normalize,
        ])

        # Add solarization to second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(
                size,
                scale=global_crops_scale,
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(
                    brightness=color_jitter,
                    contrast=color_jitter,
                    saturation=color_jitter,
                    hue=color_jitter/4
                )],
                p=0.8
            ),
            transforms.RandomGrayscale(p=grayscale_prob),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))],
                p=gaussian_blur_prob
            ),
            transforms.RandomSolarize(threshold=128, p=solarization_prob),
            transforms.ToTensor(),
            normalize,
        ])

        # Local crop transformation
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(
                size,
                scale=local_crops_scale,
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(
                    brightness=color_jitter,
                    contrast=color_jitter,
                    saturation=color_jitter,
                    hue=color_jitter/4
                )],
                p=0.8
            ),
            transforms.RandomGrayscale(p=grayscale_prob),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))],
                p=gaussian_blur_prob
            ),
            transforms.ToTensor(),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        # Global crops
        crops.append(self.global_transfo(image))
        crops.append(self.global_transfo2(image))
        # Local crops
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


class HuggingFaceImageDataset(Dataset):
    """
    Wrapper for HuggingFace datasets with custom transformations
    """
    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        transform=None,
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
        streaming: bool = False,
        image_key: str = "image",
    ):
        self.transform = transform
        self.image_key = image_key

        print(f"Loading dataset: {dataset_name}, split: {split}")

        # Load dataset from HuggingFace
        self.dataset = load_dataset(
            dataset_name,
            split=split,
            cache_dir=cache_dir,
            streaming=streaming,
        )

        # Limit number of samples if specified
        if max_samples is not None and not streaming:
            indices = list(range(min(max_samples, len(self.dataset))))
            self.dataset = self.dataset.select(indices)


        print(f"Dataset loaded with {len(self.dataset) if not streaming else 'streaming'} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Get image
        image = item[self.image_key]

        # Convert to PIL Image if necessary
        if not isinstance(image, Image.Image):
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")

        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply transformation
        if self.transform:
            image = self.transform(image)

            # Debug first sample only
            if not hasattr(HuggingFaceImageDataset, '_debug_printed') and idx == 0:
                print(f"\n[DEBUG DATASET] After transform:")
                print(f"[DEBUG DATASET] Type: {type(image)}")
                if isinstance(image, list):
                    print(f"[DEBUG DATASET] List length: {len(image)}")
                    if len(image) > 0 and hasattr(image[0], 'shape'):
                        print(f"[DEBUG DATASET] First element shape: {image[0].shape}")
                else:
                    print(f"[DEBUG DATASET] Shape: {image.shape if hasattr(image, 'shape') else 'no shape'}")
                HuggingFaceImageDataset._debug_printed = True

        # Get label if available (for evaluation), otherwise return index
        if 'label' in item:
            label = item['label']
        elif 'fine_label' in item:  # CIFAR-100 uses 'fine_label'
            label = item['fine_label']
        else:
            label = idx  # Use index as pseudo-label for unsupervised training

        return image, label


def collate_fn(batch):
    """
    Custom collate function for multi-crop batches.
    Handles the case where each sample returns a list of crops.
    """
    try:
        # Debug: print batch structure (only first time)
        if not hasattr(collate_fn, '_debug_printed'):
            print(f"\n[DEBUG COLLATE] Function called")
            print(f"[DEBUG COLLATE] Batch length: {len(batch)}")
            print(f"[DEBUG COLLATE] First item type: {type(batch[0])}")
            print(f"[DEBUG COLLATE] First item length: {len(batch[0]) if isinstance(batch[0], (list, tuple)) else 'N/A'}")
            print(f"[DEBUG COLLATE] First item[0] type: {type(batch[0][0])}")

            if isinstance(batch[0][0], list):
                print(f"[DEBUG COLLATE] First item[0] IS a list with {len(batch[0][0])} elements")
                if len(batch[0][0]) > 0:
                    print(f"[DEBUG COLLATE] First crop type: {type(batch[0][0][0])}")
                    if hasattr(batch[0][0][0], 'shape'):
                        print(f"[DEBUG COLLATE] First crop shape: {batch[0][0][0].shape}")
            else:
                print(f"[DEBUG COLLATE] First item[0] is NOT a list")
                if hasattr(batch[0][0], 'shape'):
                    print(f"[DEBUG COLLATE] First item[0] shape: {batch[0][0].shape}")
                else:
                    print(f"[DEBUG COLLATE] First item[0] value: {batch[0][0]}")

            collate_fn._debug_printed = True

        # Check if the first image is a list of crops or a single tensor
        first_img = batch[0][0]

        if isinstance(first_img, list):
            # Multi-crop case: transpose to get list of crop batches
            # batch is a list of (crops_list, label) tuples
            # We want to convert it to (list_of_crop_batches, labels_batch)

            num_crops = len(first_img)
            crops_batched = []

            for i in range(num_crops):
                # Stack the i-th crop from all samples in the batch
                crop_batch = torch.stack([item[0][i] for item in batch])
                crops_batched.append(crop_batch)

            labels = torch.tensor([item[1] for item in batch])

            if not hasattr(collate_fn, '_debug_printed2'):
                print(f"[DEBUG COLLATE] Multi-crop case: Returning {len(crops_batched)} crop batches")
                print(f"[DEBUG COLLATE] Each crop batch shape: {crops_batched[0].shape}")
                collate_fn._debug_printed2 = True

            return crops_batched, labels
        else:
            # Single image case: wrap in list to maintain consistency
            # This ensures train.py always receives a list
            images = [torch.stack([item[0] for item in batch])]
            labels = torch.tensor([item[1] for item in batch])

            if not hasattr(collate_fn, '_debug_printed2'):
                print(f"[DEBUG COLLATE] Single image case: wrapping in list")
                print(f"[DEBUG COLLATE] Returning list with 1 element of shape {images[0].shape}")
                collate_fn._debug_printed2 = True

            return images, labels

    except Exception as e:
        print(f"[ERROR COLLATE] Exception in collate_fn: {e}")
        print(f"[ERROR COLLATE] Batch type: {type(batch)}")
        print(f"[ERROR COLLATE] Batch length: {len(batch) if hasattr(batch, '__len__') else 'N/A'}")
        if len(batch) > 0:
            print(f"[ERROR COLLATE] First item: {batch[0]}")
        raise


def create_dataloader(
    dataset_name: str,
    split: str,
    batch_size: int,
    num_workers: int,
    transform,
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    streaming: bool = False,
    pin_memory: bool = True,
    image_key: str = "image",
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for self-supervised learning with optimizations
    """
    dataset = HuggingFaceImageDataset(
        dataset_name=dataset_name,
        split=split,
        transform=transform,
        max_samples=max_samples,
        cache_dir=cache_dir,
        streaming=streaming,
        image_key=image_key,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if not streaming else False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=collate_fn,  # Use custom collate function
        prefetch_factor=prefetch_factor,  # Prefetch more batches
        persistent_workers=persistent_workers,  # Keep workers alive between epochs
        multiprocessing_context='fork',  # Faster worker startup
    )

    return dataloader


def get_transforms(cfg):
    """
    Create transforms based on model configuration
    """
    model_name = cfg.model.name

    if model_name in ['ibot', 'dino_v2', 'dino_v3']:
        # Multi-crop transformation for self-supervised learning
        transform = MultiCropTransform(
            global_crops_scale=tuple(cfg.model[cfg.model.name.replace('_v2', '').replace('_v3', '')].global_crops_scale),
            local_crops_scale=tuple(cfg.model[cfg.model.name.replace('_v2', '').replace('_v3', '')].local_crops_scale),
            local_crops_number=cfg.model[cfg.model.name.replace('_v2', '').replace('_v3', '')].local_crops_number,
            size=cfg.model.vit.img_size,
            color_jitter=cfg.data.augmentation.color_jitter,
            grayscale_prob=cfg.data.augmentation.grayscale_prob,
            gaussian_blur_prob=cfg.data.augmentation.gaussian_blur_prob,
            solarization_prob=cfg.data.augmentation.solarization_prob,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return transform




