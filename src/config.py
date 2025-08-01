"""
Configuration file for XAI Chest X-ray Analysis Project

This module contains all configuration parameters, hyperparameters, and paths
used throughout the project for reproducibility and easy modification.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import yaml
from dataclasses import dataclass


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    
    # Dataset paths
    raw_data_path: str = "data/raw"
    processed_data_path: str = "data/processed"
    external_data_path: str = "data/external"
    
    # NIH Chest X-ray dataset specific
    nih_dataset_path: str = "data/raw/nih_chest_xray"
    nih_metadata_file: str = "data_entries.csv"
    nih_labels_file: str = "labels.csv"
    
    # Image processing
    image_size: Tuple[int, int] = (224, 224)
    image_channels: int = 3
    normalization_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalization_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # Data splitting
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42
    
    # Data augmentation
    rotation_range: int = 15
    horizontal_flip_prob: float = 0.5
    brightness_factor: float = 0.2
    contrast_factor: float = 0.2
    
    # Class balancing
    use_smote: bool = True
    smote_k_neighbors: int = 5
    class_weights: Dict[str, float] = None


@dataclass
class ModelConfig:
    """Configuration for model architecture and training."""
    
    # Model architecture
    model_name: str = "resnet50"
    pretrained: bool = True
    num_classes: int = 14
    dropout_rate: float = 0.5
    
    # Training parameters
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    
    # Optimizer
    optimizer: str = "adam"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Loss function
    loss_function: str = "focal_loss"
    focal_alpha: float = 1.0
    focal_gamma: float = 2.0
    
    # Learning rate scheduling
    lr_scheduler: str = "reduce_lr_on_plateau"
    lr_patience: int = 10
    lr_factor: float = 0.5
    lr_min: float = 1e-7
    
    # Early stopping
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-4
    
    # Model saving
    save_best_model: bool = True
    save_last_model: bool = True
    model_save_path: str = "models"


@dataclass
class ExplainabilityConfig:
    """Configuration for explainability techniques."""
    
    # Grad-CAM settings
    gradcam_target_layer: str = "layer4.2.conv3"
    gradcam_alpha: float = 0.4
    
    # SHAP settings
    shap_background_size: int = 1000
    shap_nsamples: int = 100
    shap_l1_reg: str = "auto"
    
    # Visualization settings
    heatmap_colormap: str = "jet"
    heatmap_alpha: float = 0.6
    save_visualizations: bool = True
    visualization_path: str = "visualizations"
    
    # Explanation evaluation
    evaluate_with_radiologist_annotations: bool = False
    annotation_path: str = "data/external/radiologist_annotations"


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    
    # Hardware settings
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    num_workers: int = 4
    pin_memory: bool = True
    
    # Logging and monitoring
    log_interval: int = 10
    save_interval: int = 5
    tensorboard_logging: bool = True
    wandb_logging: bool = False
    wandb_project: str = "xai-chest-xray"
    
    # Validation
    validation_interval: int = 1
    validation_metrics: List[str] = None
    
    # Checkpointing
    checkpoint_path: str = "checkpoints"
    resume_from_checkpoint: bool = False
    checkpoint_file: str = None


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    
    # Metrics
    primary_metric: str = "auroc"
    secondary_metrics: List[str] = None
    
    # Cross-validation
    use_cross_validation: bool = True
    cv_folds: int = 5
    cv_strategy: str = "stratified"
    
    # Statistical testing
    perform_statistical_tests: bool = True
    confidence_level: float = 0.95
    
    # Results saving
    save_results: bool = True
    results_path: str = "results"


class Config:
    """Main configuration class that combines all sub-configurations."""
    
    def __init__(self, config_file: str = None):
        """
        Initialize configuration.
        
        Args:
            config_file: Path to YAML configuration file
        """
        self.data = DataConfig()
        self.model = ModelConfig()
        self.explainability = ExplainabilityConfig()
        self.training = TrainingConfig()
        self.evaluation = EvaluationConfig()
        
        # Set default secondary metrics
        if self.evaluation.secondary_metrics is None:
            self.evaluation.secondary_metrics = [
                "accuracy", "precision", "recall", "f1_score"
            ]
        
        # Set default validation metrics
        if self.training.validation_metrics is None:
            self.training.validation_metrics = [
                "loss", "accuracy", "auroc", "precision", "recall"
            ]
        
        # Load from file if provided
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
    
    def load_from_file(self, config_file: str):
        """Load configuration from YAML file."""
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Update configurations
        for section, values in config_dict.items():
            if hasattr(self, section):
                section_config = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
    
    def save_to_file(self, config_file: str):
        """Save configuration to YAML file."""
        config_dict = {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'explainability': self.explainability.__dict__,
            'training': self.training.__dict__,
            'evaluation': self.evaluation.__dict__
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def get_device(self) -> str:
        """Get the appropriate device for training."""
        if self.training.device == "auto":
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return self.training.device
    
    def create_directories(self):
        """Create necessary directories."""
        directories = [
            self.data.raw_data_path,
            self.data.processed_data_path,
            self.data.external_data_path,
            self.model.model_save_path,
            self.explainability.visualization_path,
            self.training.checkpoint_path,
            self.evaluation.results_path,
            "logs",
            "notebooks",
            "tests"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        # Check data splits sum to 1
        if abs(self.data.train_split + self.data.val_split + self.data.test_split - 1.0) > 1e-6:
            raise ValueError("Data splits must sum to 1.0")
        
        # Check learning rate is positive
        if self.model.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        # Check batch size is positive
        if self.model.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        # Check image size is valid
        if self.data.image_size[0] <= 0 or self.data.image_size[1] <= 0:
            raise ValueError("Image size must be positive")
        
        return True


# Default configuration instance
config = Config()

# Disease classes for NIH Chest X-ray dataset
DISEASE_CLASSES = [
    'Atelectasis',
    'Cardiomegaly', 
    'Effusion',
    'Infiltration',
    'Mass',
    'Nodule',
    'Pneumonia',
    'Pneumothorax',
    'Consolidation',
    'Edema',
    'Emphysema',
    'Fibrosis',
    'Pleural_Thickening',
    'Hernia'
]

# Color mapping for visualization
DISEASE_COLORS = {
    'Atelectasis': '#FF6B6B',
    'Cardiomegaly': '#4ECDC4',
    'Effusion': '#45B7D1',
    'Infiltration': '#96CEB4',
    'Mass': '#FFEAA7',
    'Nodule': '#DDA0DD',
    'Pneumonia': '#98D8C8',
    'Pneumothorax': '#F7DC6F',
    'Consolidation': '#BB8FCE',
    'Edema': '#85C1E9',
    'Emphysema': '#F8C471',
    'Fibrosis': '#82E0AA',
    'Pleural_Thickening': '#F1948A',
    'Hernia': '#85C1E9'
}

if __name__ == "__main__":
    # Test configuration
    config = Config()
    config.create_directories()
    config.validate()
    print("Configuration validated successfully!")
    print(f"Device: {config.get_device()}")
    print(f"Model: {config.model.model_name}")
    print(f"Classes: {config.model.num_classes}") 