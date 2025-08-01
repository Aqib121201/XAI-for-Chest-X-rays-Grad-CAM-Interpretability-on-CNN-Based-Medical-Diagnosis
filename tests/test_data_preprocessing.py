"""
Unit tests for data preprocessing module.
"""

import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from src.config import Config
from src.data_preprocessing import DataPreprocessor, ChestXrayDataset, create_sample_data_for_testing


class TestDataPreprocessing:
    """Test class for data preprocessing functionality."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return Config()
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_data(self, temp_data_dir):
        """Create sample data for testing."""
        return create_sample_data_for_testing()
    
    def test_config_initialization(self, config):
        """Test configuration initialization."""
        assert config is not None
        assert config.data.image_size == (224, 224)
        assert config.model.num_classes == 14
        assert len(config.data.normalization_mean) == 3
        assert len(config.data.normalization_std) == 3
    
    def test_data_preprocessor_initialization(self, config):
        """Test data preprocessor initialization."""
        preprocessor = DataPreprocessor(config)
        assert preprocessor is not None
        assert preprocessor.config == config
        assert preprocessor.train_transform is not None
        assert preprocessor.val_transform is not None
        assert preprocessor.test_transform is not None
    
    def test_create_sample_data(self, temp_data_dir):
        """Test sample data creation."""
        data_dir, metadata_file = create_sample_data_for_testing()
        
        assert Path(data_dir).exists()
        assert Path(data_dir) / metadata_file
        
        # Check metadata file
        metadata_path = Path(data_dir) / metadata_file
        df = pd.read_csv(metadata_path)
        
        assert len(df) > 0
        assert 'Image Index' in df.columns
        assert all(disease in df.columns for disease in [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax'
        ])
    
    def test_dataset_creation(self, config, sample_data):
        """Test dataset creation."""
        data_dir, metadata_file = sample_data
        
        # Create dataset
        dataset = ChestXrayDataset(
            data_dir=data_dir,
            metadata_file=metadata_file,
            transform=preprocessor.val_transform,
            config=config
        )
        
        assert len(dataset) > 0
        assert dataset.image_paths is not None
        assert dataset.labels is not None
        assert len(dataset.image_paths) == len(dataset.labels)
    
    def test_data_transforms(self, config):
        """Test data transformation pipelines."""
        preprocessor = DataPreprocessor(config)
        
        # Test train transform
        assert preprocessor.train_transform is not None
        
        # Test validation transform
        assert preprocessor.val_transform is not None
        
        # Test test transform
        assert preprocessor.test_transform is not None
    
    def test_data_splitting(self, config, sample_data):
        """Test data splitting functionality."""
        data_dir, metadata_file = sample_data
        preprocessor = DataPreprocessor(config)
        
        # Load data
        preprocessor.load_data(data_dir, metadata_file)
        
        # Check that datasets were created
        assert preprocessor.train_dataset is not None
        assert preprocessor.val_dataset is not None
        assert preprocessor.test_dataset is not None
        
        # Check split sizes
        total_size = len(preprocessor.train_dataset) + len(preprocessor.val_dataset) + len(preprocessor.test_dataset)
        assert total_size > 0
    
    def test_class_weights_calculation(self, config, sample_data):
        """Test class weights calculation."""
        data_dir, metadata_file = sample_data
        preprocessor = DataPreprocessor(config)
        
        # Load data
        preprocessor.load_data(data_dir, metadata_file)
        
        # Check class weights
        if preprocessor.class_weights is not None:
            assert preprocessor.class_weights.shape[0] == config.model.num_classes
            assert torch.all(preprocessor.class_weights >= 0)
    
    def test_data_loader_creation(self, config, sample_data):
        """Test data loader creation."""
        data_dir, metadata_file = sample_data
        preprocessor = DataPreprocessor(config)
        
        # Load data
        preprocessor.load_data(data_dir, metadata_file)
        
        # Create data loaders
        train_loader, val_loader, test_loader = preprocessor.create_data_loaders()
        
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None
        
        # Check batch size
        assert train_loader.batch_size == config.model.batch_size
        assert val_loader.batch_size == config.model.batch_size
        assert test_loader.batch_size == config.model.batch_size
    
    def test_dataset_info(self, config, sample_data):
        """Test dataset information retrieval."""
        data_dir, metadata_file = sample_data
        preprocessor = DataPreprocessor(config)
        
        # Load data
        preprocessor.load_data(data_dir, metadata_file)
        
        # Get dataset info
        info = preprocessor.get_dataset_info()
        
        assert 'train_size' in info
        assert 'val_size' in info
        assert 'test_size' in info
        assert 'num_classes' in info
        assert 'class_names' in info
        assert 'image_size' in info
        
        assert info['num_classes'] == config.model.num_classes
        assert info['image_size'] == config.data.image_size
        assert len(info['class_names']) == config.model.num_classes
    
    def test_data_saving(self, config, sample_data, temp_data_dir):
        """Test processed data saving."""
        data_dir, metadata_file = sample_data
        preprocessor = DataPreprocessor(config)
        
        # Load data
        preprocessor.load_data(data_dir, metadata_file)
        
        # Save processed data
        output_dir = Path(temp_data_dir) / "processed"
        preprocessor.save_processed_data(str(output_dir))
        
        # Check that files were created
        assert (output_dir / "dataset_info.csv").exists()
        if preprocessor.class_weights is not None:
            assert (output_dir / "class_weights.csv").exists()


class TestChestXrayDataset:
    """Test class for ChestXrayDataset."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return Config()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return create_sample_data_for_testing()
    
    def test_dataset_initialization(self, config, sample_data):
        """Test dataset initialization."""
        data_dir, metadata_file = sample_data
        
        dataset = ChestXrayDataset(
            data_dir=data_dir,
            metadata_file=metadata_file,
            config=config
        )
        
        assert dataset is not None
        assert dataset.data_dir == Path(data_dir)
        assert dataset.metadata_file == metadata_file
        assert dataset.config == config
    
    def test_metadata_loading(self, config, sample_data):
        """Test metadata loading."""
        data_dir, metadata_file = sample_data
        
        dataset = ChestXrayDataset(
            data_dir=data_dir,
            metadata_file=metadata_file,
            config=config
        )
        
        assert dataset.metadata is not None
        assert len(dataset.metadata) > 0
        
        # Check required columns
        required_columns = ['Image Index'] + [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
            'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]
        
        for col in required_columns:
            assert col in dataset.metadata.columns
    
    def test_data_preparation(self, config, sample_data):
        """Test data preparation."""
        data_dir, metadata_file = sample_data
        
        dataset = ChestXrayDataset(
            data_dir=data_dir,
            metadata_file=metadata_file,
            config=config
        )
        
        assert dataset.image_paths is not None
        assert dataset.labels is not None
        assert len(dataset.image_paths) == len(dataset.labels)
        assert len(dataset.image_paths) > 0
    
    def test_label_distribution(self, config, sample_data):
        """Test label distribution calculation."""
        data_dir, metadata_file = sample_data
        
        dataset = ChestXrayDataset(
            data_dir=data_dir,
            metadata_file=metadata_file,
            config=config
        )
        
        distribution = dataset._get_label_distribution()
        
        assert distribution is not None
        assert len(distribution) == config.model.num_classes
        
        for disease, count in distribution.items():
            assert count >= 0
            assert isinstance(count, int)
    
    def test_dataset_length(self, config, sample_data):
        """Test dataset length."""
        data_dir, metadata_file = sample_data
        
        dataset = ChestXrayDataset(
            data_dir=data_dir,
            metadata_file=metadata_file,
            config=config
        )
        
        assert len(dataset) > 0
        assert len(dataset) == len(dataset.image_paths)
    
    def test_dataset_getitem(self, config, sample_data):
        """Test dataset item retrieval."""
        data_dir, metadata_file = sample_data
        
        dataset = ChestXrayDataset(
            data_dir=data_dir,
            metadata_file=metadata_file,
            config=config
        )
        
        if len(dataset) > 0:
            image, labels = dataset[0]
            
            assert isinstance(image, torch.Tensor)
            assert isinstance(labels, torch.Tensor)
            assert image.shape[0] == 3  # RGB channels
            assert labels.shape[0] == config.model.num_classes


if __name__ == "__main__":
    pytest.main([__file__]) 