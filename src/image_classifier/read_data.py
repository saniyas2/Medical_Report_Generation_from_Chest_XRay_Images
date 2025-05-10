import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os


# class ChestXrayDataSet(Dataset):
#     def __init__(self, data_dir, csv_file, transform=None):
#         """
#         Args:
#             data_dir: path to image directory
#             csv_file: path to the CSV file containing images and labels
#             transform: optional transform to be applied on a sample
#         """
#         self.data = pd.read_csv(csv_file)
#         self.data_dir = data_dir
#         self.transform = transform
#
#         # Define the label columns (all except 'image_id' and 'Report Impression')
#         self.label_cols = [
#             'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
#             'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
#             'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
#             'Pleural Other', 'Fracture', 'Support Devices', 'No Finding'
#         ]
#
#         # Fill NaN values with 0 in label columns
#         self.data[self.label_cols] = self.data[self.label_cols].fillna(0)
#
#     def __getitem__(self, index):
#         """
#         Args:
#             index: the index of item
#         Returns:
#             image and its labels
#         """
#         # Get image path and labels
#         image_id = self.data.iloc[index]['image_id']
#         image_path = os.path.join(self.data_dir, image_id)
#
#         # Load image
#         image = Image.open(image_path).convert('RGB')
#
#         # Get labels
#         labels = self.data.iloc[index][self.label_cols].values.astype(float)
#
#         # Apply transforms
#         if self.transform is not None:
#             image = self.transform(image)
#
#         return image, torch.FloatTensor(labels)
#
#     def __len__(self):
#         return len(self.data)

import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
import numpy as np


class ChestXrayDataSet(Dataset):
    def __init__(self, data_dir, csv_file, transform=None):
        """
        Args:
            data_dir: path to image directory
            csv_file: path to the CSV file with annotations
            transform: optional transform to be applied on images
        """
        self.data = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.transform = transform

        # Define the label columns (excluding 'image_id' and 'Report Impression')
        self.label_cols = [
            'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
            'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
            'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
            'Pleural Other', 'Fracture', 'Support Devices', 'No Finding'
        ]

        # Convert NaN to 0 and ensure binary values
        self.data[self.label_cols] = self.data[self.label_cols].fillna(0)

        # Normalize and clean the labels
        for col in self.label_cols:
            # Convert to numeric, replacing any non-numeric values with 0
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce').fillna(0)

            # Ensure binary values (0 or 1)
            self.data[col] = self.data[col].apply(lambda x: 1 if x > 0 else 0)

        # Verify data
        self._verify_data()

    def _verify_data(self):
        """Verify that all labels are properly normalized"""
        for col in self.label_cols:
            min_val = self.data[col].min()
            max_val = self.data[col].max()
            if min_val < 0 or max_val > 1:
                raise ValueError(f"Labels in column {col} are not normalized "
                                 f"(min: {min_val}, max: {max_val})")

            # Check for any non-binary values
            unique_vals = self.data[col].unique()
            if not all(x in [0, 1] for x in unique_vals):
                raise ValueError(f"Non-binary values found in column {col}: {unique_vals}")

    def __getitem__(self, index):
        """Get image and labels for a given index"""
        # Get image path
        img_id = self.data.iloc[index]['image_id']
        img_path = os.path.join(self.data_dir, img_id)

        # Load and process image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise IOError(f"Error loading image {img_path}: {e}")

        if self.transform:
            image = self.transform(image)

        # Get labels and ensure they're properly formatted
        labels = self.data.iloc[index][self.label_cols].values.astype(np.float32)

        # Final verification of label values
        if not ((labels >= 0) & (labels <= 1)).all():
            raise ValueError(f"Invalid label values found: {labels}")

        return image, torch.FloatTensor(labels)

    def __len__(self):
        return len(self.data)


# Example usage
def test_dataset(data_dir, csv_file):
    """Test the dataset implementation"""
    import torchvision.transforms as transforms

    # Basic transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create and test dataset
    try:
        dataset = ChestXrayDataSet(data_dir, csv_file, transform)
        print(f"Dataset created successfully with {len(dataset)} samples")

        # Test first item
        image, labels = dataset[0]
        print(f"\nFirst item test:")
        print(f"Image shape: {image.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Labels range: [{labels.min():.1f}, {labels.max():.1f}]")
        print(f"Unique label values: {torch.unique(labels).tolist()}")

        return True

    except Exception as e:
        print(f"Error testing dataset: {e}")
        return False


# Example usage:
if __name__ == '__main__':
    # Example of how to use the dataset
    import torchvision.transforms as transforms

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create dataset
    dataset = ChestXrayDataSet(
        data_dir='path/to/images',
        csv_file='path/to/your/labels.csv',
        transform=transform
    )

    # Test the dataset
    image, labels = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Labels: {labels}")