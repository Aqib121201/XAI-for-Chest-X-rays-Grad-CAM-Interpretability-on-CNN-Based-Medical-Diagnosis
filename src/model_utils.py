"""
Model utilities for XAI Chest X-ray Analysis Project.

This module contains custom loss functions, evaluation metrics, model architectures,
and utility functions for training and evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

from .config import Config, DISEASE_CLASSES


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in multi-label classification.
    
    Focal Loss reduces the relative loss for well-classified examples and
    puts more focus on hard, misclassified examples.
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class (default: 1.0)
            gamma: Focusing parameter (default: 2.0)
            reduction: Reduction method ('none', 'mean', 'sum')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal Loss.
        
        Args:
            inputs: Predicted logits (N, C)
            targets: Ground truth labels (N, C)
            
        Returns:
            Focal loss value
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)
        
        # Calculate binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calculate focal loss
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        focal_loss = alpha_weight * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross Entropy Loss for handling class imbalance.
    """
    
    def __init__(self, pos_weight: torch.Tensor, reduction: str = 'mean'):
        """
        Initialize Weighted BCE Loss.
        
        Args:
            pos_weight: Weight for positive class (C,)
            reduction: Reduction method ('none', 'mean', 'sum')
        """
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Weighted BCE Loss.
        
        Args:
            inputs: Predicted logits (N, C)
            targets: Ground truth labels (N, C)
            
        Returns:
            Weighted BCE loss value
        """
        return F.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self.pos_weight, reduction=self.reduction
        )


class MultiLabelMetrics:
    """
    Multi-label classification metrics calculator.
    
    This class provides comprehensive evaluation metrics for multi-label
    classification tasks, including per-class and overall metrics.
    """
    
    def __init__(self, class_names: List[str] = None):
        """
        Initialize metrics calculator.
        
        Args:
            class_names: List of class names
        """
        self.class_names = class_names or DISEASE_CLASSES
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
        self.probabilities = []
    
    def update(self, outputs: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with batch predictions.
        
        Args:
            outputs: Model outputs (logits)
            targets: Ground truth labels
        """
        # Convert to numpy
        probs = torch.sigmoid(outputs).cpu().numpy()
        preds = (probs > 0.5).astype(int)
        targets_np = targets.cpu().numpy()
        
        self.probabilities.append(probs)
        self.predictions.append(preds)
        self.targets.append(targets_np)
    
    def compute(self) -> Dict[str, Union[float, List[float]]]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary containing all computed metrics
        """
        if not self.predictions:
            raise ValueError("No predictions available. Call update() first.")
        
        # Concatenate all batches
        all_preds = np.concatenate(self.predictions, axis=0)
        all_targets = np.concatenate(self.targets, axis=0)
        all_probs = np.concatenate(self.probabilities, axis=0)
        
        metrics = {}
        
        # Overall metrics
        metrics['accuracy'] = accuracy_score(all_targets.flatten(), all_preds.flatten())
        metrics['precision_macro'] = precision_score(all_targets, all_preds, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(all_targets, all_preds, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        metrics['precision_micro'] = precision_score(all_targets, all_preds, average='micro', zero_division=0)
        metrics['recall_micro'] = recall_score(all_targets, all_preds, average='micro', zero_division=0)
        metrics['f1_micro'] = f1_score(all_targets, all_preds, average='micro', zero_division=0)
        
        # AUROC for each class
        auroc_scores = []
        for i in range(all_targets.shape[1]):
            try:
                auroc = roc_auc_score(all_targets[:, i], all_probs[:, i])
                auroc_scores.append(auroc)
            except ValueError:
                auroc_scores.append(0.0)
        
        metrics['auroc_per_class'] = auroc_scores
        metrics['auroc_mean'] = np.mean(auroc_scores)
        
        # Per-class metrics
        per_class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            per_class_metrics[class_name] = {
                'precision': precision_score(all_targets[:, i], all_preds[:, i], zero_division=0),
                'recall': recall_score(all_targets[:, i], all_preds[:, i], zero_division=0),
                'f1': f1_score(all_targets[:, i], all_preds[:, i], zero_division=0),
                'auroc': auroc_scores[i]
            }
        
        metrics['per_class'] = per_class_metrics
        
        return metrics
    
    def get_confusion_matrices(self) -> Dict[str, np.ndarray]:
        """Get confusion matrices for each class."""
        if not self.predictions:
            raise ValueError("No predictions available. Call update() first.")
        
        all_preds = np.concatenate(self.predictions, axis=0)
        all_targets = np.concatenate(self.targets, axis=0)
        
        confusion_matrices = {}
        for i, class_name in enumerate(self.class_names):
            cm = confusion_matrix(all_targets[:, i], all_preds[:, i])
            confusion_matrices[class_name] = cm
        
        return confusion_matrices


class ResNet50MultiLabel(nn.Module):
    """
    ResNet50 model adapted for multi-label chest X-ray classification.
    
    This model uses transfer learning with a pre-trained ResNet50 backbone
    and custom final layers for multi-label classification.
    """
    
    def __init__(self, num_classes: int = 14, pretrained: bool = True, dropout_rate: float = 0.5):
        """
        Initialize ResNet50 model.
        
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pre-trained weights
            dropout_rate: Dropout rate for regularization
        """
        super(ResNet50MultiLabel, self).__init__()
        
        # Load pre-trained ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Remove the final classification layer
        num_features = self.backbone.fc.in_features
        
        # Create custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Replace the final layer
        self.backbone.fc = self.classifier
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Output logits (B, num_classes)
        """
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the backbone.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Feature tensor (B, num_features)
        """
        # Remove the classifier
        backbone_without_fc = nn.Sequential(*list(self.backbone.children())[:-1])
        features = backbone_without_fc(x)
        return features.view(features.size(0), -1)


class DenseNet121MultiLabel(nn.Module):
    """
    DenseNet121 model adapted for multi-label chest X-ray classification.
    """
    
    def __init__(self, num_classes: int = 14, pretrained: bool = True, dropout_rate: float = 0.5):
        """
        Initialize DenseNet121 model.
        
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pre-trained weights
            dropout_rate: Dropout rate for regularization
        """
        super(DenseNet121MultiLabel, self).__init__()
        
        # Load pre-trained DenseNet121
        self.backbone = models.densenet121(pretrained=pretrained)
        
        # Remove the final classification layer
        num_features = self.backbone.classifier.in_features
        
        # Create custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Replace the final layer
        self.backbone.classifier = self.classifier
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Output logits (B, num_classes)
        """
        return self.backbone(x)


def create_model(model_name: str, num_classes: int = 14, pretrained: bool = True, **kwargs) -> nn.Module:
    """
    Factory function to create different model architectures.
    
    Args:
        model_name: Name of the model architecture
        num_classes: Number of output classes
        pretrained: Whether to use pre-trained weights
        **kwargs: Additional arguments for model initialization
        
    Returns:
        Initialized model
    """
    model_name = model_name.lower()
    
    if model_name == 'resnet50':
        return ResNet50MultiLabel(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'densenet121':
        return DenseNet121MultiLabel(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'resnet101':
        model = models.resnet101(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(kwargs.get('dropout_rate', 0.5)),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(kwargs.get('dropout_rate', 0.5)),
            nn.Linear(512, num_classes)
        )
        return model
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")


def create_loss_function(loss_name: str, class_weights: torch.Tensor = None, **kwargs) -> nn.Module:
    """
    Factory function to create loss functions.
    
    Args:
        loss_name: Name of the loss function
        class_weights: Class weights for weighted loss
        **kwargs: Additional arguments for loss initialization
        
    Returns:
        Initialized loss function
    """
    loss_name = loss_name.lower()
    
    if loss_name == 'bce':
        if class_weights is not None:
            return WeightedBCELoss(pos_weight=class_weights)
        else:
            return nn.BCEWithLogitsLoss()
    elif loss_name == 'focal_loss':
        return FocalLoss(
            alpha=kwargs.get('alpha', 1.0),
            gamma=kwargs.get('gamma', 2.0)
        )
    elif loss_name == 'weighted_bce':
        if class_weights is None:
            raise ValueError("Class weights required for weighted BCE loss")
        return WeightedBCELoss(pos_weight=class_weights)
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")


def create_optimizer(model: nn.Module, optimizer_name: str, **kwargs) -> torch.optim.Optimizer:
    """
    Factory function to create optimizers.
    
    Args:
        model: Model to optimize
        optimizer_name: Name of the optimizer
        **kwargs: Additional arguments for optimizer initialization
        
    Returns:
        Initialized optimizer
    """
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=kwargs.get('learning_rate', 1e-4),
            betas=(kwargs.get('adam_beta1', 0.9), kwargs.get('adam_beta2', 0.999)),
            eps=kwargs.get('adam_epsilon', 1e-8),
            weight_decay=kwargs.get('weight_decay', 0)
        )
    elif optimizer_name == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=kwargs.get('learning_rate', 1e-3),
            momentum=kwargs.get('momentum', 0.9),
            weight_decay=kwargs.get('weight_decay', 0)
        )
    elif optimizer_name == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=kwargs.get('learning_rate', 1e-4),
            betas=(kwargs.get('adam_beta1', 0.9), kwargs.get('adam_beta2', 0.999)),
            eps=kwargs.get('adam_epsilon', 1e-8),
            weight_decay=kwargs.get('weight_decay', 1e-4)
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def create_scheduler(optimizer: torch.optim.Optimizer, scheduler_name: str, **kwargs) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Factory function to create learning rate schedulers.
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_name: Name of the scheduler
        **kwargs: Additional arguments for scheduler initialization
        
    Returns:
        Initialized scheduler
    """
    scheduler_name = scheduler_name.lower()
    
    if scheduler_name == 'reduce_lr_on_plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=kwargs.get('lr_factor', 0.5),
            patience=kwargs.get('lr_patience', 10),
            min_lr=kwargs.get('lr_min', 1e-7),
            verbose=True
        )
    elif scheduler_name == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 30),
            gamma=kwargs.get('gamma', 0.1)
        )
    elif scheduler_name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 100),
            eta_min=kwargs.get('eta_min', 1e-7)
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def save_model(model: nn.Module, filepath: str, optimizer: torch.optim.Optimizer = None, 
               scheduler: torch.optim.lr_scheduler._LRScheduler = None, epoch: int = None, 
               metrics: Dict = None):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        filepath: Path to save the model
        optimizer: Optimizer state (optional)
        scheduler: Scheduler state (optional)
        epoch: Current epoch (optional)
        metrics: Training metrics (optional)
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'metrics': metrics
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, filepath)
    logger.info(f"Model saved to {filepath}")


def load_model(model: nn.Module, filepath: str, optimizer: torch.optim.Optimizer = None,
               scheduler: torch.optim.lr_scheduler._LRScheduler = None, device: str = 'cpu'):
    """
    Load model checkpoint.
    
    Args:
        model: Model to load weights into
        filepath: Path to the checkpoint file
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to load the model on
        
    Returns:
        Dictionary containing loaded information
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    result = {
        'epoch': checkpoint.get('epoch'),
        'metrics': checkpoint.get('metrics')
    }
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        result['optimizer'] = optimizer
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        result['scheduler'] = scheduler
    
    logger.info(f"Model loaded from {filepath}")
    return result


def plot_training_history(history: Dict[str, List[float]], save_path: str = None):
    """
    Plot training history.
    
    Args:
        history: Dictionary containing training metrics
        save_path: Path to save the plot (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    if 'train_loss' in history and 'val_loss' in history:
        axes[0, 0].plot(history['train_loss'], label='Train Loss')
        axes[0, 0].plot(history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
    
    # Accuracy plot
    if 'train_accuracy' in history and 'val_accuracy' in history:
        axes[0, 1].plot(history['train_accuracy'], label='Train Accuracy')
        axes[0, 1].plot(history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # AUROC plot
    if 'train_auroc' in history and 'val_auroc' in history:
        axes[1, 0].plot(history['train_auroc'], label='Train AUROC')
        axes[1, 0].plot(history['val_auroc'], label='Validation AUROC')
        axes[1, 0].set_title('Training and Validation AUROC')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUROC')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Learning rate plot
    if 'learning_rate' in history:
        axes[1, 1].plot(history['learning_rate'])
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Test model utilities
    config = Config()
    
    # Test model creation
    model = create_model('resnet50', num_classes=14, pretrained=False)
    print(f"Model created: {type(model).__name__}")
    
    # Test loss function creation
    loss_fn = create_loss_function('focal_loss', alpha=1.0, gamma=2.0)
    print(f"Loss function created: {type(loss_fn).__name__}")
    
    # Test optimizer creation
    optimizer = create_optimizer(model, 'adam', learning_rate=1e-4)
    print(f"Optimizer created: {type(optimizer).__name__}")
    
    # Test scheduler creation
    scheduler = create_scheduler(optimizer, 'reduce_lr_on_plateau')
    print(f"Scheduler created: {type(scheduler).__name__}")
    
    # Test metrics
    metrics = MultiLabelMetrics()
    print(f"Metrics created: {type(metrics).__name__}")
    
    print("\nModel utilities test completed successfully!") 