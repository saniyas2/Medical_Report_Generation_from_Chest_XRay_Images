import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import logging
from pathlib import Path
from typing import Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from sklearn.model_selection import train_test_split

class ChestXrayDataset(Dataset):
    def __init__(
        self,
        data_frame: pd.DataFrame,
        transform: Optional[A.Compose] = None,
        is_training: bool = True
    ):
        """
        Initialize the dataset
        """
        self.data = data_frame
        self.transform = transform or self._get_default_transforms(is_training)
        self.is_training = is_training

        # Clean and preprocess the dataset
        self._preprocess_dataset()

    def _preprocess_dataset(self):
        """Clean and preprocess the dataset"""
        initial_len = len(self.data)

        # Convert image paths to Path objects and check existence
        self.data['image_path'] = self.data['image_path'].apply(lambda x: Path(x))
        valid_paths = self.data['image_path'].apply(lambda x: x.exists())

        # Remove rows with missing values or empty impressions
        valid_data = (
            self.data['findings'].notna() &
            (self.data['findings'].str.strip().str.len() > 0) &
            valid_paths
        )

        self.data = self.data[valid_data].reset_index(drop=True)

        # Now, try to open each image and remove samples where images cannot be opened
        invalid_indices = []
        for idx in range(len(self.data)):
            img_path = self.data.iloc[idx]['image_path']
            image = cv2.imread(str(img_path))
            if image is None:
                logging.error(f"Error reading image {img_path}")
                invalid_indices.append(idx)

        if invalid_indices:
            self.data = self.data.drop(index=invalid_indices).reset_index(drop=True)
            logging.warning(f"Removed {len(invalid_indices)} samples due to image loading errors")

        # Log removed entries
        removed = initial_len - len(self.data)
        if removed > 0:
            logging.warning(f"Removed {removed} invalid entries from dataset")

        if len(self.data) == 0:
            raise ValueError("No valid samples remaining after preprocessing")

        logging.info(f"Final dataset size: {len(self.data)} samples")

    def _get_default_transforms(self, is_training: bool) -> A.Compose:
        """Get default transforms based on training/validation mode"""
        if is_training:
            return A.Compose([
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.OneOf([
                    A.GaussNoise(p=1),
                    A.GaussianBlur(p=1),
                ], p=0.3),
                A.OneOf([
                    A.OpticalDistortion(p=1),
                    A.GridDistortion(p=1),
                ], p=0.3),
                A.OneOf([
                    A.RandomBrightnessContrast(p=1),
                    A.RandomGamma(p=1),
                ], p=0.3),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(224, 224),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """Get a sample from the dataset"""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image path and impression
        img_path = self.data.iloc[idx]['image_path']
        impression = self.data.iloc[idx]['findings']

        # Load and process image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']

        return image, impression

def get_dataloaders(
    csv_path: str,
    batch_size: int = 8,
    train_split: float = 0.85,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders"""
    try:
        # Read data
        df = pd.read_csv(csv_path)
        logging.info(f"Loaded dataset with {len(df)} samples")

        # Split into train and validation
        train_df, val_df = train_test_split(
            df,
            train_size=train_split,
            random_state=seed,
            shuffle=True
        )

        # Create datasets
        train_dataset = ChestXrayDataset(train_df, is_training=True)
        val_dataset = ChestXrayDataset(val_df, is_training=False)

        if len(train_dataset) == 0 or len(val_dataset) == 0:
            raise ValueError("Empty dataset after preprocessing")

        logging.info(f"Created train dataset with {len(train_dataset)} samples")
        logging.info(f"Created validation dataset with {len(val_dataset)} samples")

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        return train_loader, val_loader

    except Exception as e:
        logging.error(f"Error creating dataloaders: {str(e)}")
        raise

def get_sample_batch(dataloader: DataLoader) -> Tuple[torch.Tensor, list]:
    """Get a sample batch from dataloader for testing"""
    try:
        images, impressions = next(iter(dataloader))
        logging.info(f"Sample batch shapes - Images: {images.shape}")
        return images, impressions
    except Exception as e:
        logging.error(f"Error getting sample batch: {str(e)}")
        raise