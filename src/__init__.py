"""
XAI for Chest X-rays: Grad-CAM Interpretability on CNN-Based Medical Diagnosis

This package provides a comprehensive framework for explainable AI in medical imaging,
specifically focused on chest X-ray analysis using ResNet50 with Grad-CAM and SHAP.

Author: [Your Name]
Date: 2024
"""

__version__ = "1.0.0"
__author__ = "[Your Name]"
__email__ = "[your.email@institution.edu]"

from .config import Config
from .data_preprocessing import DataPreprocessor
from .model_training import ModelTrainer
from .model_utils import ModelUtils
from .explainability import ExplainabilityEngine

__all__ = [
    "Config",
    "DataPreprocessor", 
    "ModelTrainer",
    "ModelUtils",
    "ExplainabilityEngine"
] 