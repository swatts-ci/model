"""
DataModule for the Chesapeake Bay dataset for segmentation tasks.

This implementation provides a structured way to handle the data loading and
preprocessing required for training and validating a segmentation model.

Dataset citation:
Robinson C, Hou L, Malkin K, Soobitsky R, Czawlytko J, Dilkina B, Jojic N.
Large Scale High-Resolution Land Cover Mapping with Multi-Resolution Data.
Proceedings of the 2019 Conference on Computer Vision and Pattern Recognition
(CVPR 2019).

Dataset URL: https://lila.science/datasets/chesapeakelandcover
"""

import re
from pathlib import Path

import lightning as L
import numpy as np
import torch
import yaml
from box import Box
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from torchvision.tv_tensors import Mask
from PIL import Image


class ChesapeakeDataset(Dataset):
    """
    Dataset class for the Chesapeake Bay segmentation dataset.

    Args:
        chip_dir (str): Directory containing the image chips.
        label_dir (str): Directory containing the labels.
        metadata (Box): Metadata for normalization and other dataset-specific details.
        platform (str): Platform identifier used in metadata.
    """

    def __init__(self, chip_dir, label_dir, metadata, platform):
        self.chip_dir = Path(chip_dir)
        self.label_dir = Path(label_dir)
        self.metadata = metadata
        self.transform = self.create_transforms(
            mean=list(metadata[platform].bands.mean.values()),
            std=list(metadata[platform].bands.std.values()),
        )

        # Load chip and label file names
        self.chips = [chip_path.name for chip_path in self.chip_dir.glob("*.npy")]
        self.labels = [re.sub("_naip-new_", "_lc_", chip) for chip in self.chips]

    def create_transforms(self, mean, std):
        """
        Create normalization transforms.

        Args:
            mean (list): Mean values for normalization.
            std (list): Standard deviation values for normalization.

        Returns:
            torchvision.transforms.Compose: A composition of transforms.
        """
        return v2.Compose(
            [
                v2.Normalize(mean=mean, std=std),
            ],
        )

    def __len__(self):
        return len(self.chips)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary containing the image, label, and additional information.
        """
        chip_name = self.chip_dir / self.chips[idx]
        label_name = self.label_dir / self.labels[idx]

        chip = np.load(chip_name).astype(np.float32)
        label = np.load(label_name)

        # Remap labels to match desired classes
        label_mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 15: 6}
        remapped_label = np.vectorize(label_mapping.get)(label)

        sample = {
            "pixels": self.transform(torch.from_numpy(chip)),
            "label": torch.from_numpy(remapped_label[0]),
            "time": torch.zeros(4),  # Placeholder for time information
            "latlon": torch.zeros(4),  # Placeholder for latlon information
        }
        return sample
    

class DeepGlobeDataset(Dataset):
    train_transforms = [
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomAffine(degrees=(-30, 30), scale=(0.8, 1.2)),
    ]

    all_transforms = [
        v2.ToImage(),
        v2.ToDtype(torch.float64),
    ]

    def __init__(self, chip_dir, label_dir, metadata, platform):
        self.chip_dir = chip_dir
        self.label_dir = label_dir

        norm_transform = [
            v2.Normalize(
                mean=list(metadata[platform].bands.mean.values()), 
                std=list(metadata[platform].bands.std.values()),
            )
        ]

        if "train" in chip_dir:
            self.transform = v2.Compose(
                self.all_transforms + self.train_transforms + norm_transform
            )
        elif "val" in chip_dir:
            self.transform = v2.Compose(self.all_transforms + norm_transform)
        else:
            raise ValueError("Unknown dataset split")
        
        self.filenames = [
            f.stem for f in (Path(chip_dir)).glob("*")
        ]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        f = self.filenames[index]

        image = Image.open(str(Path(self.chip_dir) / (f + ".png")))
        mask = Mask(
            Image.open(
                str(Path(self.label_dir) / (f + ".png"))
            ).convert("1")
        )

        if self.transform:
            image, mask = self.transform(image, mask)

        sample = {
            "pixels": image,
            "label": mask.float().squeeze(0),
            "time": torch.zeros(4),  # Placeholder for time information
            "latlon": torch.zeros(4),  # Placeholder for latlon information
        }
        return sample


class ChesapeakeDataModule(L.LightningDataModule):
    """
    DataModule class for the Chesapeake Bay dataset.

    Args:
        train_chip_dir (str): Directory containing training image chips.
        train_label_dir (str): Directory containing training labels.
        val_chip_dir (str): Directory containing validation image chips.
        val_label_dir (str): Directory containing validation labels.
        metadata_path (str): Path to the metadata file.
        batch_size (int): Batch size for data loading.
        num_workers (int): Number of workers for data loading.
        platform (str): Platform identifier used in metadata.
    """

    def __init__(  # noqa: PLR0913
        self,
        train_chip_dir,
        train_label_dir,
        val_chip_dir,
        val_label_dir,
        metadata_path,
        batch_size,
        num_workers,
        platform,
    ):
        super().__init__()
        self.train_chip_dir = train_chip_dir
        self.train_label_dir = train_label_dir
        self.val_chip_dir = val_chip_dir
        self.val_label_dir = val_label_dir
        self.metadata = Box(yaml.safe_load(open(metadata_path)))
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.platform = platform

    def setup(self, stage=None):
        """
        Setup datasets for training and validation.

        Args:
            stage (str): Stage identifier ('fit' or 'test').
        """
        if stage in {"fit", None}:
            self.trn_ds = DeepGlobeDataset(
                self.train_chip_dir,
                self.train_label_dir,
                self.metadata,
                self.platform,
            )
            self.val_ds = DeepGlobeDataset(
                self.val_chip_dir,
                self.val_label_dir,
                self.metadata,
                self.platform,
            )

    def train_dataloader(self):
        """
        Create DataLoader for training data.

        Returns:
            DataLoader: DataLoader for training dataset.
        """
        return DataLoader(
            self.trn_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self, shuffle=False):
        """
        Create DataLoader for validation data.

        Returns:
            DataLoader: DataLoader for validation dataset.
        """
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
        )
