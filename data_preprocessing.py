# data_processing.py

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
import logging
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from transformers import AutoTokenizer
from typing import Optional, List, Tuple
from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

# Define findings columns at the module level
findings_columns = [
    'Enlarged Cardiomediastinum',
    'Cardiomegaly',
    'Lung Opacity',
    'Lung Lesion',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Pleural Other',
    'Fracture',
    'Support Devices',
    'No Finding'
]

class ChestXrayDataset(Dataset):
    def __init__(
            self,
            data_frame: pd.DataFrame,
            transform: Optional[A.Compose] = None,
            is_training: bool = True,
            max_length: int = 512
    ):
        self.data = data_frame
        self.transform = transform or self._get_default_transforms(is_training)
        self.is_training = is_training
        self.max_length = max_length

        # Medical terms and abbreviations mapping
        self.medical_abbreviations = {
            'ap': 'anteroposterior',
            'pa': 'posteroanterior',
            'lat': 'lateral',
            'bilat': 'bilateral',
            'w/': 'with',
            'w/o': 'without',
            'vs': 'versus',
            'etc': 'etcetera',
            'aka': 'also known as',
            'cf': 'compare',
            're': 'regarding',
            'esp': 'especially',
        }

        self.findings_columns = findings_columns

        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/biogpt')
        self._preprocess_dataset()

    def _clean_text(self, text: str) -> str:
        """Enhanced text cleaning function"""
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Replace medical abbreviations
        for abbr, full in self.medical_abbreviations.items():
            text = re.sub(r'\b' + abbr + r'\b', full, text)

        # Remove special characters but keep necessary punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,;:()/\-]', '', text)

        # Standardize spacing
        text = re.sub(r'\s+', ' ', text)

        # Standardize sentence endings
        text = re.sub(r'\.+', '.', text)

        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,;:])', r'\1', text)

        return text.strip()

    def _preprocess_dataset(self):
        """Enhanced dataset preprocessing"""
        initial_len = len(self.data)

        # Convert image paths to Path objects
        self.data['image_path'] = self.data['image_path'].apply(lambda x: Path(x))

        # Clean and preprocess text fields (actual report)
        self.data['findings_text'] = self.data['findings'].apply(self._clean_text)

        # Remove invalid entries
        valid_data = (
            self.data['findings_text'].notna() &
            (self.data['findings_text'].str.strip().str.len() > 0) &
            self.data['image_path'].apply(lambda x: x.exists())
        )

        self.data = self.data[valid_data].reset_index(drop=True)

        # Log preprocessing results
        removed = initial_len - len(self.data)
        if removed > 0:
            logging.warning(f"Removed {removed} invalid entries from dataset")

        if len(self.data) == 0:
            raise ValueError("No valid samples remaining after preprocessing")

        logging.info(f"Final dataset size: {len(self.data)} samples")

        # Process findings labels (structured findings)
        self.data['findings_list'] = self.data.apply(self._get_findings_list, axis=1)

    def _get_findings_list(self, row):
        findings_list = []
        for col in self.findings_columns:
            if col in row and row[col] == 1:
                if col != 'No Finding':
                    findings_list.append(col)
                else:
                    # If 'No Finding' is present, ignore other findings
                    findings_list = ['No Findings']
                    break
        return findings_list

    def _get_default_transforms(self, is_training: bool) -> A.Compose:
        """Enhanced image transformations"""
        if is_training:
            return A.Compose([
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=1),
                    A.GaussianBlur(blur_limit=(3, 7), p=1),
                    A.MedianBlur(blur_limit=5, p=1)
                ], p=0.3),
                A.OneOf([
                    A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1),
                    A.GridDistortion(num_steps=5, distort_limit=0.05, p=1),
                    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1)
                ], p=0.3),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
                    A.RandomGamma(gamma_limit=(80, 120), p=1),
                    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1)
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

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get sample data
        img_path = self.data.iloc[idx]['image_path']
        findings_text = self.data.iloc[idx]['findings_text']
        findings_list = self.data.iloc[idx]['findings_list']

        # Load and process image
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Image not found or cannot be opened: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']

        return image, findings_text, findings_list  # Return findings_text for alignment

def custom_collate_fn(batch):
    """Enhanced collate function with padding and attention masks"""
    images = torch.stack([item[0] for item in batch])
    findings_texts = [item[1] for item in batch]  # Actual findings (for alignment)
    findings_lists = [item[2] for item in batch]  # Pathology findings list (for prompt)

    return images, findings_texts, findings_lists

def get_dataloaders(
        csv_with_image_paths: str,
        csv_with_labels: str,
        batch_size: int = 8,
        train_split: float = 0.85,
        num_workers: int = 4,
        seed: int = 42,
        collate_fn=custom_collate_fn
) -> Tuple[DataLoader, DataLoader]:
    """Enhanced dataloader creation with stratification"""
    try:
        # Read data
        df_images = pd.read_csv(csv_with_image_paths)
        df_labels = pd.read_csv(csv_with_labels)

        # Merge datasets on 'image_id'
        df = pd.merge(df_images, df_labels, on='image_id', how='inner')
        logging.info(f"Merged dataset has {len(df)} samples")

        # Create a stratification column based on the number of findings
        df['num_findings'] = df[findings_columns].sum(axis=1)
        df['strat_column'] = pd.qcut(df['num_findings'], q=5, labels=False, duplicates='drop')

        # Stratified split
        train_df, val_df = train_test_split(
            df,
            train_size=train_split,
            random_state=seed,
            shuffle=True,
            stratify=df['strat_column']
        )

        # Create datasets
        train_dataset = ChestXrayDataset(train_df, is_training=True)
        val_dataset = ChestXrayDataset(val_df, is_training=False)

        if len(train_dataset) == 0 or len(val_dataset) == 0:
            raise ValueError("Empty dataset after preprocessing")

        logging.info(f"Created train dataset with {len(train_dataset)} samples")
        logging.info(f"Created validation dataset with {len(val_dataset)} samples")

        # Create dataloaders with automatic batching
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
            persistent_workers=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=True
        )

        return train_loader, val_loader

    except Exception as e:
        logging.error(f"Error creating dataloaders: {str(e)}")
        raise
