"""
Data preprocessing module for XAI Chest X-ray Analysis Project.

This module handles data loading, preprocessing, augmentation, and dataset creation
for the NIH Chest X-ray dataset with support for explainability techniques.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets
from PIL import Image
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import logging
from loguru import logger

from .config import Config, DISEASE_CLASSES


class ChestXrayDataset(Dataset):
    """
    Custom dataset class for NIH Chest X-ray dataset.
    
    This dataset handles loading and preprocessing of chest X-ray images
    with support for multi-label classification and explainability.
    """
    
    def __init__(
        self,
        data_dir: str,
        metadata_file: str,
        transform: Optional[transforms.Compose] = None,
        target_transform: Optional[transforms.Compose] = None,
        mode: str = "train",
        config: Config = None
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing the images
            metadata_file: Path to the metadata CSV file
            transform: Image transformations
            target_transform: Target transformations
            mode: Dataset mode ('train', 'val', 'test')
            config: Configuration object
        """
        self.data_dir = Path(data_dir)
        self.metadata_file = metadata_file
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        self.config = config or Config()
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Create image paths and labels
        self.image_paths, self.labels = self._prepare_data()
        
        logger.info(f"Loaded {len(self.image_paths)} images for {mode} mode")
        logger.info(f"Label distribution: {self._get_label_distribution()}")
    
    def _load_metadata(self) -> pd.DataFrame:
        """Load and preprocess metadata."""
        metadata_path = self.data_dir / self.metadata_file
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        # Load metadata
        df = pd.read_csv(metadata_path)
        
        # Ensure required columns exist
        required_columns = ['Image Index'] + DISEASE_CLASSES
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Convert labels to numeric
        for disease in DISEASE_CLASSES:
            df[disease] = df[disease].astype(int)
        
        return df
    
    def _prepare_data(self) -> Tuple[List[str], np.ndarray]:
        """Prepare image paths and labels."""
        image_paths = []
        labels = []
        
        for _, row in self.metadata.iterrows():
            image_path = self.data_dir / "images" / row['Image Index']
            
            if image_path.exists():
                image_paths.append(str(image_path))
                
                # Extract labels for all diseases
                disease_labels = [row[disease] for disease in DISEASE_CLASSES]
                labels.append(disease_labels)
        
        return image_paths, np.array(labels)
    
    def _get_label_distribution(self) -> Dict[str, int]:
        """Get distribution of labels."""
        label_counts = {}
        for i, disease in enumerate(DISEASE_CLASSES):
            label_counts[disease] = int(np.sum(self.labels[:, i]))
        return label_counts
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample."""
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Get labels
        labels = self.labels[idx]
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            labels = self.target_transform(labels)
        
        return image, torch.FloatTensor(labels)


class DataPreprocessor:
    """
    Data preprocessing pipeline for chest X-ray analysis.
    
    This class handles all aspects of data preprocessing including:
    - Data loading and validation
    - Image preprocessing and augmentation
    - Dataset splitting and balancing
    - DataLoader creation
    """
    
    def __init__(self, config: Config = None):
        """
        Initialize the data preprocessor.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.setup_logging()
        
        # Create transforms
        self.train_transform = self._create_train_transform()
        self.val_transform = self._create_val_transform()
        self.test_transform = self._create_test_transform()
        
        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Class weights for imbalanced data
        self.class_weights = None
    
    def setup_logging(self):
        """Setup logging configuration."""
        logger.add(
            "logs/data_preprocessing.log",
            rotation="10 MB",
            retention="7 days",
            level="INFO"
        )
    
    def _create_train_transform(self) -> transforms.Compose:
        """Create training transforms with augmentation."""
        return transforms.Compose([
            transforms.Resize(self.config.data.image_size),
            transforms.RandomHorizontalFlip(p=self.config.data.horizontal_flip_prob),
            transforms.RandomRotation(self.config.data.rotation_range),
            transforms.ColorJitter(
                brightness=self.config.data.brightness_factor,
                contrast=self.config.data.contrast_factor
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.data.normalization_mean,
                std=self.config.data.normalization_std
            )
        ])
    
    def _create_val_transform(self) -> transforms.Compose:
        """Create validation transforms."""
        return transforms.Compose([
            transforms.Resize(self.config.data.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.data.normalization_mean,
                std=self.config.data.normalization_std
            )
        ])
    
    def _create_test_transform(self) -> transforms.Compose:
        """Create test transforms."""
        return self._create_val_transform()
    
    def _create_albumentations_transform(self, is_training: bool = True) -> A.Compose:
        """Create Albumentations transforms for advanced augmentation."""
        if is_training:
            return A.Compose([
                A.Resize(*self.config.data.image_size),
                A.HorizontalFlip(p=self.config.data.horizontal_flip_prob),
                A.Rotate(limit=self.config.data.rotation_range, p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=self.config.data.brightness_factor,
                    contrast_limit=self.config.data.contrast_factor,
                    p=0.5
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
                A.Normalize(
                    mean=self.config.data.normalization_mean,
                    std=self.config.data.normalization_std
                ),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(*self.config.data.image_size),
                A.Normalize(
                    mean=self.config.data.normalization_mean,
                    std=self.config.data.normalization_std
                ),
                ToTensorV2()
            ])
    
    def load_data(self, data_dir: str, metadata_file: str) -> None:
        """
        Load the complete dataset.
        
        Args:
            data_dir: Directory containing the dataset
            metadata_file: Path to metadata file
        """
        logger.info(f"Loading data from {data_dir}")
        
        # Load full dataset
        full_dataset = ChestXrayDataset(
            data_dir=data_dir,
            metadata_file=metadata_file,
            transform=self.train_transform,
            config=self.config
        )
        
        # Split dataset
        self._split_dataset(full_dataset)
        
        # Calculate class weights
        self._calculate_class_weights()
        
        logger.info("Data loading completed successfully")
    
    def _split_dataset(self, full_dataset: ChestXrayDataset) -> None:
        """Split dataset into train, validation, and test sets."""
        total_size = len(full_dataset)
        train_size = int(self.config.data.train_split * total_size)
        val_size = int(self.config.data.val_split * total_size)
        test_size = total_size - train_size - val_size
        
        logger.info(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")
        
        # Split dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.config.data.random_seed)
        )
        
        # Apply appropriate transforms
        self.val_dataset.dataset.transform = self.val_transform
        self.test_dataset.dataset.transform = self.test_transform
    
    def _calculate_class_weights(self) -> None:
        """Calculate class weights for imbalanced data."""
        if self.train_dataset is None:
            logger.warning("Train dataset not loaded. Cannot calculate class weights.")
            return
        
        # Extract labels from training set
        train_labels = []
        for idx in self.train_dataset.indices:
            labels = self.train_dataset.dataset.labels[idx]
            train_labels.append(labels)
        
        train_labels = np.array(train_labels)
        
        # Calculate class weights
        class_weights = []
        for i in range(len(DISEASE_CLASSES)):
            weights = compute_class_weight(
                'balanced',
                classes=np.unique(train_labels[:, i]),
                y=train_labels[:, i]
            )
            class_weights.append(weights[1])  # Weight for positive class
        
        self.class_weights = torch.FloatTensor(class_weights)
        
        logger.info(f"Class weights calculated: {dict(zip(DISEASE_CLASSES, class_weights))}")
    
    def apply_smote_balancing(self) -> None:
        """Apply SMOTE for class balancing."""
        if self.train_dataset is None:
            logger.warning("Train dataset not loaded. Cannot apply SMOTE.")
            return
        
        logger.info("Applying SMOTE balancing...")
        
        # Extract features and labels
        train_features = []
        train_labels = []
        
        for idx in self.train_dataset.indices:
            # Load and preprocess image
            image_path = self.train_dataset.dataset.image_paths[idx]
            image = Image.open(image_path).convert('RGB')
            image = self.val_transform(image)  # Use validation transform for consistency
            
            train_features.append(image.numpy().flatten())
            train_labels.append(self.train_dataset.dataset.labels[idx])
        
        train_features = np.array(train_features)
        train_labels = np.array(train_labels)
        
        # Apply SMOTE for each class
        smote = SMOTE(
            k_neighbors=self.config.data.smote_k_neighbors,
            random_state=self.config.data.random_seed
        )
        
        # Apply SMOTE to each disease class
        balanced_features = []
        balanced_labels = []
        
        for i, disease in enumerate(DISEASE_CLASSES):
            if np.sum(train_labels[:, i]) > 0:  # Only apply if positive samples exist
                # Reshape features for SMOTE
                features_reshaped = train_features.reshape(len(train_features), -1)
                
                # Apply SMOTE
                features_balanced, labels_balanced = smote.fit_resample(
                    features_reshaped,
                    train_labels[:, i]
                )
                
                # Add balanced samples
                for j in range(len(features_balanced)):
                    if labels_balanced[j] == 1:  # Only add synthetic positive samples
                        balanced_features.append(features_balanced[j])
                        balanced_labels.append(train_labels[j])
        
        logger.info(f"SMOTE balancing completed. Added {len(balanced_features)} synthetic samples.")
    
    def create_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create DataLoaders for training, validation, and test sets.
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        if any(dataset is None for dataset in [self.train_dataset, self.val_dataset, self.test_dataset]):
            raise ValueError("Datasets not loaded. Call load_data() first.")
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.model.batch_size,
            shuffle=True,
            num_workers=self.config.training.num_workers,
            pin_memory=self.config.training.pin_memory
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.model.batch_size,
            shuffle=False,
            num_workers=self.config.training.num_workers,
            pin_memory=self.config.training.pin_memory
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.model.batch_size,
            shuffle=False,
            num_workers=self.config.training.num_workers,
            pin_memory=self.config.training.pin_memory
        )
        
        logger.info(f"Created DataLoaders: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    
    def get_dataset_info(self) -> Dict:
        """Get information about the datasets."""
        info = {
            'train_size': len(self.train_dataset) if self.train_dataset else 0,
            'val_size': len(self.val_dataset) if self.val_dataset else 0,
            'test_size': len(self.test_dataset) if self.test_dataset else 0,
            'num_classes': len(DISEASE_CLASSES),
            'class_names': DISEASE_CLASSES,
            'image_size': self.config.data.image_size,
            'class_weights': self.class_weights.tolist() if self.class_weights is not None else None
        }
        
        return info
    
    def save_processed_data(self, output_dir: str) -> None:
        """Save processed data information."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save dataset info
        dataset_info = self.get_dataset_info()
        info_df = pd.DataFrame([dataset_info])
        info_df.to_csv(output_path / "dataset_info.csv", index=False)
        
        # Save class weights
        if self.class_weights is not None:
            weights_df = pd.DataFrame({
                'disease': DISEASE_CLASSES,
                'weight': self.class_weights.numpy()
            })
            weights_df.to_csv(output_path / "class_weights.csv", index=False)
        
        logger.info(f"Processed data information saved to {output_path}")


def create_sample_data_for_testing():
    """Create sample data for testing purposes."""
    # This function creates a small sample dataset for testing
    # In a real scenario, you would download the actual NIH dataset
    
    sample_dir = Path("data/raw/sample_nih_chest_xray")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample metadata
    sample_data = {
        'Image Index': [f'sample_{i:04d}.jpg' for i in range(100)],
        'Finding Labels': ['Normal'] * 50 + ['Atelectasis'] * 30 + ['Pneumonia'] * 20,
        'Patient ID': [f'patient_{i:03d}' for i in range(100)],
        'Patient Age': np.random.randint(20, 80, 100),
        'Patient Gender': np.random.choice(['M', 'F'], 100),
        'View Position': np.random.choice(['PA', 'AP'], 100)
    }
    
    # Add disease columns
    for disease in DISEASE_CLASSES:
        sample_data[disease] = np.random.choice([0, 1], 100, p=[0.8, 0.2])
    
    # Create sample metadata file
    metadata_df = pd.DataFrame(sample_data)
    metadata_df.to_csv(sample_dir / "sample_data_entries.csv", index=False)
    
    logger.info(f"Sample data created at {sample_dir}")
    return str(sample_dir), "sample_data_entries.csv"


if __name__ == "__main__":
    # Test the data preprocessing pipeline
    config = Config()
    preprocessor = DataPreprocessor(config)
    
    # Create sample data for testing
    data_dir, metadata_file = create_sample_data_for_testing()
    
    # Load data
    preprocessor.load_data(data_dir, metadata_file)
    
    # Create data loaders
    train_loader, val_loader, test_loader = preprocessor.create_data_loaders()
    
    # Print dataset information
    info = preprocessor.get_dataset_info()
    print("Dataset Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\nData preprocessing pipeline test completed successfully!") 