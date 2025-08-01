"""
Model training module for XAI Chest X-ray Analysis Project.

This module provides comprehensive training functionality including:
- Training loops with validation
- Model checkpointing and early stopping
- Learning rate scheduling
- Performance monitoring and logging
- Cross-validation support
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import time
import json
from loguru import logger
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

from .config import Config, DISEASE_CLASSES
from .data_preprocessing import DataPreprocessor
from .model_utils import (
    create_model, create_loss_function, create_optimizer, create_scheduler,
    MultiLabelMetrics, save_model, load_model, plot_training_history
)
from .explainability import ExplainabilityEngine


class ModelTrainer:
    """
    Comprehensive model trainer for chest X-ray classification.
    
    This class handles the complete training pipeline including data loading,
    model training, validation, and evaluation with support for explainability.
    """
    
    def __init__(self, config: Config = None):
        """
        Initialize model trainer.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.device = self.config.get_device()
        
        # Initialize components
        self.data_preprocessor = DataPreprocessor(self.config)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.metrics = MultiLabelMetrics()
        
        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.training_history = {
            'train_loss': [], 'val_loss': [],
            'train_accuracy': [], 'val_accuracy': [],
            'train_auroc': [], 'val_auroc': [],
            'learning_rate': []
        }
        
        # Setup logging
        self.setup_logging()
        
        logger.info(f"Model trainer initialized on device: {self.device}")
    
    def setup_logging(self):
        """Setup logging configuration."""
        logger.add(
            "logs/model_training.log",
            rotation="10 MB",
            retention="7 days",
            level="INFO"
        )
    
    def load_data(self, data_dir: str, metadata_file: str):
        """
        Load and preprocess data.
        
        Args:
            data_dir: Directory containing the dataset
            metadata_file: Path to metadata file
        """
        logger.info("Loading and preprocessing data...")
        
        # Load data
        self.data_preprocessor.load_data(data_dir, metadata_file)
        
        # Apply SMOTE if configured
        if self.config.data.use_smote:
            self.data_preprocessor.apply_smote_balancing()
        
        # Create data loaders
        self.train_loader, self.val_loader, self.test_loader = \
            self.data_preprocessor.create_data_loaders()
        
        # Get dataset info
        self.dataset_info = self.data_preprocessor.get_dataset_info()
        
        logger.info(f"Data loaded successfully: {self.dataset_info}")
    
    def setup_model(self, model_name: str = None):
        """
        Setup model, optimizer, loss function, and scheduler.
        
        Args:
            model_name: Name of the model architecture
        """
        model_name = model_name or self.config.model.model_name
        
        logger.info(f"Setting up model: {model_name}")
        
        # Create model
        self.model = create_model(
            model_name=model_name,
            num_classes=self.config.model.num_classes,
            pretrained=self.config.model.pretrained,
            dropout_rate=self.config.model.dropout_rate
        )
        
        # Move model to device
        self.model.to(self.device)
        
        # Create loss function
        if self.config.model.loss_function == 'weighted_bce':
            if self.data_preprocessor.class_weights is not None:
                class_weights = self.data_preprocessor.class_weights.to(self.device)
                self.criterion = create_loss_function(
                    'weighted_bce', class_weights=class_weights
                )
            else:
                logger.warning("Class weights not available, using focal loss")
                self.criterion = create_loss_function(
                    'focal_loss',
                    alpha=self.config.model.focal_alpha,
                    gamma=self.config.model.focal_gamma
                )
        else:
            self.criterion = create_loss_function(
                self.config.model.loss_function,
                alpha=self.config.model.focal_alpha,
                gamma=self.config.model.focal_gamma
            )
        
        # Create optimizer
        self.optimizer = create_optimizer(
            self.model,
            self.config.model.optimizer,
            learning_rate=self.config.model.learning_rate,
            weight_decay=self.config.model.weight_decay,
            adam_beta1=self.config.model.adam_beta1,
            adam_beta2=self.config.model.adam_beta2,
            adam_epsilon=self.config.model.adam_epsilon
        )
        
        # Create scheduler
        self.scheduler = create_scheduler(
            self.optimizer,
            self.config.model.lr_scheduler,
            lr_patience=self.config.model.lr_patience,
            lr_factor=self.config.model.lr_factor,
            lr_min=self.config.model.lr_min
        )
        
        logger.info("Model setup completed")
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        epoch_loss = 0.0
        self.metrics.reset()
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            # Move data to device
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            self.metrics.update(outputs, targets)
            
            # Log progress
            if batch_idx % self.config.training.log_interval == 0:
                logger.info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )
        
        # Calculate epoch metrics
        epoch_metrics = self.metrics.compute()
        epoch_metrics['loss'] = epoch_loss / len(self.train_loader)
        
        return epoch_metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Returns:
            Dictionary containing validation metrics
        """
        self.model.eval()
        epoch_loss = 0.0
        self.metrics.reset()
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                # Move data to device
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                # Update metrics
                epoch_loss += loss.item()
                self.metrics.update(outputs, targets)
        
        # Calculate epoch metrics
        epoch_metrics = self.metrics.compute()
        epoch_metrics['loss'] = epoch_loss / len(self.val_loader)
        
        return epoch_metrics
    
    def train(self, num_epochs: int = None, resume_from_checkpoint: str = None):
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train
            resume_from_checkpoint: Path to checkpoint to resume from
        """
        num_epochs = num_epochs or self.config.model.num_epochs
        
        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        # Early stopping
        patience_counter = 0
        best_metric = 0.0
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate_epoch()
            
            # Update learning rate
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['loss'])
            else:
                self.scheduler.step()
            
            # Update training history
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['train_accuracy'].append(train_metrics['accuracy'])
            self.training_history['val_accuracy'].append(val_metrics['accuracy'])
            self.training_history['train_auroc'].append(train_metrics['auroc_mean'])
            self.training_history['val_auroc'].append(val_metrics['auroc_mean'])
            self.training_history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Log epoch results
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, "
                f"Train AUROC: {train_metrics['auroc_mean']:.4f}, "
                f"Val AUROC: {val_metrics['auroc_mean']:.4f}"
            )
            
            # Save best model
            current_metric = val_metrics[self.config.evaluation.primary_metric]
            if current_metric > best_metric:
                best_metric = current_metric
                patience_counter = 0
                
                # Save best model
                if self.config.model.save_best_model:
                    self.save_checkpoint(
                        f"models/best_{self.config.model.model_name}.pth",
                        is_best=True,
                        metrics=val_metrics
                    )
            else:
                patience_counter += 1
            
            # Save checkpoint periodically
            if (epoch + 1) % self.config.training.save_interval == 0:
                self.save_checkpoint(
                    f"models/checkpoint_epoch_{epoch + 1}.pth",
                    metrics=val_metrics
                )
            
            # Early stopping
            if patience_counter >= self.config.model.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Save final model
        if self.config.model.save_last_model:
            self.save_checkpoint(
                f"models/final_{self.config.model.model_name}.pth",
                metrics=val_metrics
            )
        
        logger.info("Training completed")
    
    def evaluate(self, test_loader: DataLoader = None) -> Dict[str, Any]:
        """
        Evaluate the model on test set.
        
        Args:
            test_loader: Test data loader (uses self.test_loader if None)
            
        Returns:
            Dictionary containing evaluation results
        """
        if test_loader is None:
            test_loader = self.test_loader
        
        logger.info("Evaluating model on test set...")
        
        self.model.eval()
        self.metrics.reset()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, targets in test_loader:
                # Move data to device
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                probabilities = torch.sigmoid(outputs)
                predictions = (probabilities > 0.5).float()
                
                # Update metrics
                self.metrics.update(outputs, targets)
                
                # Store results
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                all_probabilities.append(probabilities.cpu().numpy())
        
        # Calculate metrics
        test_metrics = self.metrics.compute()
        
        # Get confusion matrices
        confusion_matrices = self.metrics.get_confusion_matrices()
        
        # Prepare results
        results = {
            'metrics': test_metrics,
            'confusion_matrices': confusion_matrices,
            'predictions': np.concatenate(all_predictions, axis=0),
            'targets': np.concatenate(all_targets, axis=0),
            'probabilities': np.concatenate(all_probabilities, axis=0)
        }
        
        # Log results
        logger.info("Test Results:")
        for metric, value in test_metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {metric}: {value:.4f}")
        
        return results
    
    def cross_validate(self, n_folds: int = 5) -> Dict[str, List[float]]:
        """
        Perform cross-validation.
        
        Args:
            n_folds: Number of folds
            
        Returns:
            Dictionary containing cross-validation results
        """
        if not self.config.evaluation.use_cross_validation:
            logger.warning("Cross-validation not enabled in config")
            return {}
        
        logger.info(f"Starting {n_folds}-fold cross-validation...")
        
        # Get full dataset
        full_dataset = self.data_preprocessor.train_dataset.dataset
        
        # Create cross-validation splits
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.config.data.random_seed)
        
        # Get labels for stratification (use first class for simplicity)
        labels = full_dataset.labels[:, 0]
        
        cv_results = {
            'fold_metrics': [],
            'mean_metrics': {},
            'std_metrics': {}
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(full_dataset)), labels)):
            logger.info(f"Training fold {fold + 1}/{n_folds}")
            
            # Create fold-specific datasets
            train_subset = torch.utils.data.Subset(full_dataset, train_idx)
            val_subset = torch.utils.data.Subset(full_dataset, val_idx)
            
            # Create data loaders
            train_loader = DataLoader(
                train_subset,
                batch_size=self.config.model.batch_size,
                shuffle=True,
                num_workers=self.config.training.num_workers,
                pin_memory=self.config.training.pin_memory
            )
            
            val_loader = DataLoader(
                val_subset,
                batch_size=self.config.model.batch_size,
                shuffle=False,
                num_workers=self.config.training.num_workers,
                pin_memory=self.config.training.pin_memory
            )
            
            # Setup model for this fold
            self.setup_model()
            
            # Train for a few epochs (shorter for CV)
            self.train(num_epochs=10)
            
            # Evaluate
            fold_results = self.evaluate(val_loader)
            cv_results['fold_metrics'].append(fold_results['metrics'])
            
            logger.info(f"Fold {fold + 1} completed")
        
        # Calculate mean and std across folds
        metrics_df = pd.DataFrame(cv_results['fold_metrics'])
        
        for metric in metrics_df.columns:
            cv_results['mean_metrics'][metric] = metrics_df[metric].mean()
            cv_results['std_metrics'][metric] = metrics_df[metric].std()
        
        logger.info("Cross-validation completed")
        logger.info("Mean metrics across folds:")
        for metric, value in cv_results['mean_metrics'].items():
            std = cv_results['std_metrics'][metric]
            logger.info(f"  {metric}: {value:.4f} ± {std:.4f}")
        
        return cv_results
    
    def save_checkpoint(self, filepath: str, is_best: bool = False, metrics: Dict = None):
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            is_best: Whether this is the best model so far
            metrics: Metrics to save with checkpoint
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        save_model(
            self.model,
            filepath,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            metrics=metrics
        )
        
        if is_best:
            logger.info(f"Best model saved to {filepath}")
        else:
            logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to checkpoint file
        """
        if self.model is None:
            self.setup_model()
        
        checkpoint_info = load_model(
            self.model,
            filepath,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device
        )
        
        self.current_epoch = checkpoint_info.get('epoch', 0)
        logger.info(f"Checkpoint loaded from {filepath}")
    
    def generate_explanations(self, num_samples: int = 10) -> Dict[str, Any]:
        """
        Generate explanations for sample predictions.
        
        Args:
            num_samples: Number of samples to explain
            
        Returns:
            Dictionary containing explanations
        """
        logger.info("Generating explanations...")
        
        # Initialize explainability engine
        explainability_engine = ExplainabilityEngine(self.model, self.config)
        
        # Get sample data
        sample_images = []
        sample_labels = []
        
        for i, (images, labels) in enumerate(self.test_loader):
            if len(sample_images) >= num_samples:
                break
            
            sample_images.extend(images[:num_samples - len(sample_images)])
            sample_labels.extend(labels[:num_samples - len(sample_labels)])
        
        sample_images = torch.stack(sample_images).to(self.device)
        sample_labels = torch.stack(sample_labels).to(self.device)
        
        # Generate explanations
        explanations = explainability_engine.generate_gradcam_explanations(sample_images)
        
        # Save visualizations
        if self.config.explainability.save_visualizations:
            viz_path = f"{self.config.explainability.visualization_path}/explanations.png"
            explainability_engine.visualize_explanations(
                explanations, save_path=viz_path, num_images=num_samples
            )
        
        # Cleanup
        explainability_engine.cleanup()
        
        logger.info("Explanations generated successfully")
        return explanations
    
    def plot_training_history(self, save_path: str = None):
        """
        Plot training history.
        
        Args:
            save_path: Path to save the plot
        """
        if save_path is None:
            save_path = f"{self.config.explainability.visualization_path}/training_history.png"
        
        plot_training_history(self.training_history, save_path)
    
    def save_results(self, results: Dict[str, Any], output_dir: str = "results"):
        """
        Save training and evaluation results.
        
        Args:
            results: Results to save
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        if 'metrics' in results:
            metrics_df = pd.DataFrame([results['metrics']])
            metrics_df.to_csv(output_path / "test_metrics.csv", index=False)
        
        # Save training history
        history_df = pd.DataFrame(self.training_history)
        history_df.to_csv(output_path / "training_history.csv", index=False)
        
        # Save configuration
        self.config.save_to_file(output_path / "config.yaml")
        
        # Save dataset info
        dataset_info_df = pd.DataFrame([self.dataset_info])
        dataset_info_df.to_csv(output_path / "dataset_info.csv", index=False)
        
        logger.info(f"Results saved to {output_path}")


def train_model_from_scratch(config_file: str = None, data_dir: str = None, 
                           metadata_file: str = None):
    """
    Complete training pipeline from scratch.
    
    Args:
        config_file: Path to configuration file
        data_dir: Directory containing the dataset
        metadata_file: Path to metadata file
    """
    # Load configuration
    config = Config(config_file) if config_file else Config()
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    
    # Load data
    if data_dir and metadata_file:
        trainer.load_data(data_dir, metadata_file)
    else:
        # Use default paths from config
        trainer.load_data(
            config.data.nih_dataset_path,
            config.data.nih_metadata_file
        )
    
    # Setup model
    trainer.setup_model()
    
    # Train model
    trainer.train()
    
    # Evaluate model
    results = trainer.evaluate()
    
    # Generate explanations
    explanations = trainer.generate_explanations()
    
    # Plot training history
    trainer.plot_training_history()
    
    # Save results
    trainer.save_results(results)
    
    logger.info("Training pipeline completed successfully")
    
    return trainer, results


if __name__ == "__main__":
    # Test the training pipeline
    print("Testing model training pipeline...")
    
    # Create a simple test
    config = Config()
    trainer = ModelTrainer(config)
    
    # Test model setup
    trainer.setup_model()
    print("Model setup completed successfully!")
    
    print("Model training module test completed successfully!") 