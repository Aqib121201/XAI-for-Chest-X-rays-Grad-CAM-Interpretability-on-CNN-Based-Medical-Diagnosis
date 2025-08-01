"""
Unit tests for model training module.
"""

import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from src.config import Config
from src.model_training import ModelTrainer
from src.model_utils import create_model, create_loss_function, create_optimizer, MultiLabelMetrics
from src.data_preprocessing import create_sample_data_for_testing


class TestModelTraining:
    """Test class for model training functionality."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return Config()
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return create_sample_data_for_testing()
    
    def test_config_validation(self, config):
        """Test configuration validation."""
        # Test valid configuration
        assert config.validate() is True
        
        # Test invalid configuration (should raise error)
        config.data.train_split = 0.8
        config.data.val_split = 0.3
        config.data.test_split = 0.1
        
        with pytest.raises(ValueError):
            config.validate()
    
    def test_device_detection(self, config):
        """Test device detection."""
        device = config.get_device()
        assert device in ['cpu', 'cuda', 'mps']
    
    def test_model_trainer_initialization(self, config):
        """Test model trainer initialization."""
        trainer = ModelTrainer(config)
        
        assert trainer is not None
        assert trainer.config == config
        assert trainer.device == config.get_device()
        assert trainer.model is None
        assert trainer.optimizer is None
        assert trainer.scheduler is None
        assert trainer.criterion is None
        assert trainer.metrics is not None
        assert isinstance(trainer.metrics, MultiLabelMetrics)
    
    def test_model_creation(self, config):
        """Test model creation."""
        model = create_model(
            model_name='resnet50',
            num_classes=config.model.num_classes,
            pretrained=False
        )
        
        assert model is not None
        assert hasattr(model, 'forward')
        
        # Test forward pass
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        output = model(input_tensor)
        
        assert output.shape == (batch_size, config.model.num_classes)
    
    def test_loss_function_creation(self, config):
        """Test loss function creation."""
        # Test focal loss
        focal_loss = create_loss_function('focal_loss', alpha=1.0, gamma=2.0)
        assert focal_loss is not None
        
        # Test BCE loss
        bce_loss = create_loss_function('bce')
        assert bce_loss is not None
        
        # Test weighted BCE loss
        class_weights = torch.ones(config.model.num_classes)
        weighted_bce = create_loss_function('weighted_bce', class_weights=class_weights)
        assert weighted_bce is not None
    
    def test_optimizer_creation(self, config):
        """Test optimizer creation."""
        model = create_model('resnet50', num_classes=config.model.num_classes, pretrained=False)
        
        # Test Adam optimizer
        adam_optimizer = create_optimizer(model, 'adam', learning_rate=1e-4)
        assert adam_optimizer is not None
        assert isinstance(adam_optimizer, torch.optim.Adam)
        
        # Test SGD optimizer
        sgd_optimizer = create_optimizer(model, 'sgd', learning_rate=1e-3)
        assert sgd_optimizer is not None
        assert isinstance(sgd_optimizer, torch.optim.SGD)
    
    def test_metrics_initialization(self, config):
        """Test metrics initialization."""
        metrics = MultiLabelMetrics()
        
        assert metrics is not None
        assert metrics.class_names == config.model.num_classes
        assert len(metrics.class_names) == 14
    
    def test_metrics_update_and_compute(self, config):
        """Test metrics update and computation."""
        metrics = MultiLabelMetrics()
        
        # Create sample data
        batch_size = 4
        num_classes = config.model.num_classes
        
        outputs = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, 2, (batch_size, num_classes)).float()
        
        # Update metrics
        metrics.update(outputs, targets)
        
        # Compute metrics
        computed_metrics = metrics.compute()
        
        assert computed_metrics is not None
        assert 'accuracy' in computed_metrics
        assert 'precision_macro' in computed_metrics
        assert 'recall_macro' in computed_metrics
        assert 'f1_macro' in computed_metrics
        assert 'auroc_mean' in computed_metrics
        
        # Check metric values are reasonable
        assert 0 <= computed_metrics['accuracy'] <= 1
        assert 0 <= computed_metrics['precision_macro'] <= 1
        assert 0 <= computed_metrics['recall_macro'] <= 1
        assert 0 <= computed_metrics['f1_macro'] <= 1
        assert 0 <= computed_metrics['auroc_mean'] <= 1
    
    def test_data_loading(self, config, sample_data):
        """Test data loading functionality."""
        data_dir, metadata_file = sample_data
        trainer = ModelTrainer(config)
        
        # Load data
        trainer.load_data(data_dir, metadata_file)
        
        # Check that data loaders were created
        assert trainer.train_loader is not None
        assert trainer.val_loader is not None
        assert trainer.test_loader is not None
        
        # Check dataset info
        assert trainer.dataset_info is not None
        assert 'train_size' in trainer.dataset_info
        assert 'val_size' in trainer.dataset_info
        assert 'test_size' in trainer.dataset_info
    
    def test_model_setup(self, config):
        """Test model setup functionality."""
        trainer = ModelTrainer(config)
        
        # Setup model
        trainer.setup_model()
        
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
        assert trainer.criterion is not None
        
        # Check model is on correct device
        assert next(trainer.model.parameters()).device.type == trainer.device
    
    def test_training_epoch(self, config, sample_data):
        """Test training epoch functionality."""
        data_dir, metadata_file = sample_data
        trainer = ModelTrainer(config)
        
        # Load data and setup model
        trainer.load_data(data_dir, metadata_file)
        trainer.setup_model()
        
        # Run one training epoch
        train_metrics = trainer.train_epoch()
        
        assert train_metrics is not None
        assert 'loss' in train_metrics
        assert 'accuracy' in train_metrics
        assert 'auroc_mean' in train_metrics
        
        # Check metric values are reasonable
        assert train_metrics['loss'] >= 0
        assert 0 <= train_metrics['accuracy'] <= 1
        assert 0 <= train_metrics['auroc_mean'] <= 1
    
    def test_validation_epoch(self, config, sample_data):
        """Test validation epoch functionality."""
        data_dir, metadata_file = sample_data
        trainer = ModelTrainer(config)
        
        # Load data and setup model
        trainer.load_data(data_dir, metadata_file)
        trainer.setup_model()
        
        # Run one validation epoch
        val_metrics = trainer.validate_epoch()
        
        assert val_metrics is not None
        assert 'loss' in val_metrics
        assert 'accuracy' in val_metrics
        assert 'auroc_mean' in val_metrics
        
        # Check metric values are reasonable
        assert val_metrics['loss'] >= 0
        assert 0 <= val_metrics['accuracy'] <= 1
        assert 0 <= val_metrics['auroc_mean'] <= 1
    
    def test_model_evaluation(self, config, sample_data):
        """Test model evaluation functionality."""
        data_dir, metadata_file = sample_data
        trainer = ModelTrainer(config)
        
        # Load data and setup model
        trainer.load_data(data_dir, metadata_file)
        trainer.setup_model()
        
        # Evaluate model
        results = trainer.evaluate()
        
        assert results is not None
        assert 'metrics' in results
        assert 'confusion_matrices' in results
        assert 'predictions' in results
        assert 'targets' in results
        assert 'probabilities' in results
        
        # Check metrics
        metrics = results['metrics']
        assert 'accuracy' in metrics
        assert 'precision_macro' in metrics
        assert 'recall_macro' in metrics
        assert 'f1_macro' in metrics
        assert 'auroc_mean' in metrics
    
    def test_checkpoint_saving_and_loading(self, config, temp_dir):
        """Test checkpoint saving and loading."""
        trainer = ModelTrainer(config)
        trainer.setup_model()
        
        # Create sample metrics
        sample_metrics = {
            'accuracy': 0.85,
            'auroc_mean': 0.89,
            'loss': 0.15
        }
        
        # Save checkpoint
        checkpoint_path = Path(temp_dir) / "test_checkpoint.pth"
        trainer.save_checkpoint(str(checkpoint_path), metrics=sample_metrics)
        
        assert checkpoint_path.exists()
        
        # Load checkpoint
        trainer.load_checkpoint(str(checkpoint_path))
        
        assert trainer.current_epoch >= 0
    
    def test_training_history(self, config):
        """Test training history tracking."""
        trainer = ModelTrainer(config)
        
        # Check initial history
        assert 'train_loss' in trainer.training_history
        assert 'val_loss' in trainer.training_history
        assert 'train_accuracy' in trainer.training_history
        assert 'val_accuracy' in trainer.training_history
        assert 'train_auroc' in trainer.training_history
        assert 'val_auroc' in trainer.training_history
        assert 'learning_rate' in trainer.training_history
        
        # Check initial values are empty lists
        assert len(trainer.training_history['train_loss']) == 0
        assert len(trainer.training_history['val_loss']) == 0
    
    def test_results_saving(self, config, sample_data, temp_dir):
        """Test results saving functionality."""
        data_dir, metadata_file = sample_data
        trainer = ModelTrainer(config)
        
        # Load data and setup model
        trainer.load_data(data_dir, metadata_file)
        trainer.setup_model()
        
        # Create sample results
        sample_results = {
            'metrics': {
                'accuracy': 0.85,
                'auroc_mean': 0.89,
                'precision_macro': 0.82,
                'recall_macro': 0.84,
                'f1_macro': 0.83
            },
            'confusion_matrices': {},
            'predictions': np.random.rand(10, 14),
            'targets': np.random.randint(0, 2, (10, 14)),
            'probabilities': np.random.rand(10, 14)
        }
        
        # Save results
        output_dir = Path(temp_dir) / "results"
        trainer.save_results(sample_results, str(output_dir))
        
        # Check that files were created
        assert (output_dir / "test_metrics.csv").exists()
        assert (output_dir / "training_history.csv").exists()
        assert (output_dir / "config.yaml").exists()
        assert (output_dir / "dataset_info.csv").exists()


class TestModelUtils:
    """Test class for model utilities."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return Config()
    
    def test_focal_loss(self, config):
        """Test focal loss computation."""
        from src.model_utils import FocalLoss
        
        focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
        
        # Create sample data
        batch_size = 4
        num_classes = config.model.num_classes
        
        inputs = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, 2, (batch_size, num_classes)).float()
        
        # Compute loss
        loss = focal_loss(inputs, targets)
        
        assert loss is not None
        assert loss.item() >= 0
        assert isinstance(loss, torch.Tensor)
    
    def test_weighted_bce_loss(self, config):
        """Test weighted BCE loss computation."""
        from src.model_utils import WeightedBCELoss
        
        class_weights = torch.ones(config.model.num_classes)
        weighted_bce = WeightedBCELoss(pos_weight=class_weights)
        
        # Create sample data
        batch_size = 4
        num_classes = config.model.num_classes
        
        inputs = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, 2, (batch_size, num_classes)).float()
        
        # Compute loss
        loss = weighted_bce(inputs, targets)
        
        assert loss is not None
        assert loss.item() >= 0
        assert isinstance(loss, torch.Tensor)
    
    def test_model_architectures(self, config):
        """Test different model architectures."""
        # Test ResNet50
        resnet50 = create_model('resnet50', num_classes=config.model.num_classes, pretrained=False)
        assert resnet50 is not None
        
        # Test DenseNet121
        densenet121 = create_model('densenet121', num_classes=config.model.num_classes, pretrained=False)
        assert densenet121 is not None
        
        # Test ResNet101
        resnet101 = create_model('resnet101', num_classes=config.model.num_classes, pretrained=False)
        assert resnet101 is not None
        
        # Test invalid architecture
        with pytest.raises(ValueError):
            create_model('invalid_model', num_classes=config.model.num_classes, pretrained=False)
    
    def test_loss_functions(self, config):
        """Test different loss functions."""
        # Test BCE
        bce = create_loss_function('bce')
        assert bce is not None
        
        # Test focal loss
        focal = create_loss_function('focal_loss', alpha=1.0, gamma=2.0)
        assert focal is not None
        
        # Test weighted BCE
        class_weights = torch.ones(config.model.num_classes)
        weighted_bce = create_loss_function('weighted_bce', class_weights=class_weights)
        assert weighted_bce is not None
        
        # Test invalid loss function
        with pytest.raises(ValueError):
            create_loss_function('invalid_loss')
    
    def test_optimizers(self, config):
        """Test different optimizers."""
        model = create_model('resnet50', num_classes=config.model.num_classes, pretrained=False)
        
        # Test Adam
        adam = create_optimizer(model, 'adam', learning_rate=1e-4)
        assert adam is not None
        assert isinstance(adam, torch.optim.Adam)
        
        # Test SGD
        sgd = create_optimizer(model, 'sgd', learning_rate=1e-3)
        assert sgd is not None
        assert isinstance(sgd, torch.optim.SGD)
        
        # Test AdamW
        adamw = create_optimizer(model, 'adamw', learning_rate=1e-4)
        assert adamw is not None
        assert isinstance(adamw, torch.optim.AdamW)
        
        # Test invalid optimizer
        with pytest.raises(ValueError):
            create_optimizer(model, 'invalid_optimizer')


if __name__ == "__main__":
    pytest.main([__file__]) 