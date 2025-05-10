import pandas as pd
from sklearn.model_selection import train_test_split
import os


def split_dataset(
        csv_path: str,
        output_dir: str,
        train_size: float = 0.7,
        valid_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int = 42
):
    """
    Split dataset into train, validation, and test sets

    Args:
        csv_path: Path to CSV file containing image paths
        output_dir: Directory to save split CSVs
        train_size: Proportion of data for training
        valid_size: Proportion of data for validation
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # First split: separate test set
    train_valid_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state
    )

    # Second split: separate train and validation sets
    relative_valid_size = valid_size / (train_size + valid_size)
    train_df, valid_df = train_test_split(
        train_valid_df,
        test_size=relative_valid_size,
        random_state=random_state
    )

    # Save CSV files
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    valid_df.to_csv(os.path.join(output_dir, 'valid.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    # Print split statistics
    print("\nDataset split statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Training samples: {len(train_df)} ({len(train_df) / len(df) * 100:.1f}%)")
    print(f"Validation samples: {len(valid_df)} ({len(valid_df) / len(df) * 100:.1f}%)")
    print(f"Test samples: {len(test_df)} ({len(test_df) / len(df) * 100:.1f}%)")

    return train_df, valid_df, test_df


# Example usage
if __name__ == "__main__":
    # Example paths
    csv_path = "../datasets/labeled_reports_with_images.csv"  # Your CSV file with image paths
    output_dir = "split_data"  # Where to save the splits

    # Split the dataset
    train_df, valid_df, test_df = split_dataset(
        csv_path=csv_path,
        output_dir=output_dir
    )