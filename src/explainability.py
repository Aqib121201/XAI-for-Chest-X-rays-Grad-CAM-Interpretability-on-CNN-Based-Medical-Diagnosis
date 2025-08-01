"""
Explainability module for XAI Chest X-ray Analysis Project.

This module implements various explainability techniques including:
- Grad-CAM (Gradient-weighted Class Activation Mapping)
- SHAP (SHapley Additive exPlanations)
- LIME (Local Interpretable Model-agnostic Explanations)
- Feature importance analysis
- Visualization utilities
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import shap
from captum.attr import (
    GradientShap, IntegratedGradients, Occlusion, 
    GuidedGradCam, LayerGradCam, LayerAttribution
)
from captum.attr import visualization as viz
import lime
from lime import lime_image
from loguru import logger
import pandas as pd
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

from .config import Config, DISEASE_CLASSES, DISEASE_COLORS
from .model_utils import create_model, MultiLabelMetrics


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping implementation.
    
    Grad-CAM provides visual explanations for CNN predictions by highlighting
    the regions in the input image that are most important for the model's decision.
    """
    
    def __init__(self, model: torch.nn.Module, target_layer: str = "layer4.2.conv3"):
        """
        Initialize Grad-CAM.
        
        Args:
            model: Trained model
            target_layer: Target layer for gradient computation
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Find the target layer
        target_module = None
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                target_module = module
                break
        
        if target_module is None:
            raise ValueError(f"Target layer '{self.target_layer}' not found in model")
        
        # Register hooks
        self.forward_handle = target_module.register_forward_hook(forward_hook)
        self.backward_handle = target_module.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_image: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Generate Grad-CAM for a specific class.
        
        Args:
            input_image: Input image tensor (1, C, H, W)
            class_idx: Target class index
            
        Returns:
            Grad-CAM heatmap
        """
        # Forward pass
        self.model.zero_grad()
        output = self.model(input_image)
        
        # Backward pass for target class
        if output.dim() == 1:
            output[class_idx].backward()
        else:
            output[0, class_idx].backward()
        
        # Get gradients and activations
        gradients = self.gradients.detach().cpu().numpy()
        activations = self.activations.detach().cpu().numpy()
        
        # Calculate weights
        weights = np.mean(gradients, axis=(2, 3))
        
        # Generate CAM
        cam = np.sum(weights[:, :, np.newaxis, np.newaxis] * activations, axis=1)
        cam = np.maximum(cam, 0)  # ReLU
        
        # Normalize
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        
        return cam[0]
    
    def generate_cam_batch(self, input_images: torch.Tensor, class_indices: List[int]) -> List[np.ndarray]:
        """
        Generate Grad-CAM for multiple images and classes.
        
        Args:
            input_images: Batch of input images (B, C, H, W)
            class_indices: List of target class indices
            
        Returns:
            List of Grad-CAM heatmaps
        """
        cams = []
        
        for i, (image, class_idx) in enumerate(zip(input_images, class_indices)):
            cam = self.generate_cam(image.unsqueeze(0), class_idx)
            cams.append(cam)
        
        return cams
    
    def overlay_heatmap(self, image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.6) -> np.ndarray:
        """
        Overlay heatmap on original image.
        
        Args:
            image: Original image (H, W, C)
            heatmap: Grad-CAM heatmap (H, W)
            alpha: Transparency factor
            
        Returns:
            Overlaid image
        """
        # Resize heatmap to match image size
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlaid = alpha * heatmap_colored + (1 - alpha) * image
        overlaid = np.clip(overlaid, 0, 255).astype(np.uint8)
        
        return overlaid
    
    def cleanup(self):
        """Remove hooks."""
        if hasattr(self, 'forward_handle'):
            self.forward_handle.remove()
        if hasattr(self, 'backward_handle'):
            self.backward_handle.remove()


class SHAPExplainer:
    """
    SHAP (SHapley Additive exPlanations) implementation for deep learning models.
    
    SHAP provides both local and global explanations for model predictions
    by computing the contribution of each feature to the prediction.
    """
    
    def __init__(self, model: torch.nn.Module, background_data: torch.Tensor = None):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model
            background_data: Background dataset for SHAP computation
        """
        self.model = model
        self.background_data = background_data
        self.explainer = None
        
        # Initialize explainer
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Initialize SHAP explainer."""
        if self.background_data is None:
            # Create dummy background data
            self.background_data = torch.randn(100, 3, 224, 224)
        
        # Use DeepExplainer for neural networks
        self.explainer = shap.DeepExplainer(self.model, self.background_data)
    
    def explain_instance(self, input_image: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        Explain a single instance.
        
        Args:
            input_image: Input image tensor (1, C, H, W)
            class_idx: Target class index (if None, use predicted class)
            
        Returns:
            SHAP values
        """
        # Get prediction if class_idx not provided
        if class_idx is None:
            with torch.no_grad():
                output = self.model(input_image)
                class_idx = torch.argmax(output).item()
        
        # Compute SHAP values
        shap_values = self.explainer.shap_values(input_image)
        
        # Return SHAP values for target class
        if isinstance(shap_values, list):
            return shap_values[class_idx]
        else:
            return shap_values
    
    def explain_batch(self, input_images: torch.Tensor, class_indices: List[int] = None) -> List[np.ndarray]:
        """
        Explain multiple instances.
        
        Args:
            input_images: Batch of input images (B, C, H, W)
            class_indices: List of target class indices
            
        Returns:
            List of SHAP values
        """
        if class_indices is None:
            with torch.no_grad():
                outputs = self.model(input_images)
                class_indices = torch.argmax(outputs, dim=1).tolist()
        
        shap_values_list = []
        
        for i, (image, class_idx) in enumerate(zip(input_images, class_indices)):
            shap_values = self.explain_instance(image.unsqueeze(0), class_idx)
            shap_values_list.append(shap_values)
        
        return shap_values_list
    
    def get_feature_importance(self, input_images: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        Get feature importance scores.
        
        Args:
            input_images: Input images (B, C, H, W)
            class_idx: Target class index
            
        Returns:
            Feature importance scores
        """
        shap_values = self.explain_batch(input_images, [class_idx] * len(input_images))
        
        # Average SHAP values across batch
        avg_shap = np.mean(np.abs(shap_values), axis=0)
        
        return avg_shap


class LIMEExplainer:
    """
    LIME (Local Interpretable Model-agnostic Explanations) implementation.
    
    LIME provides local explanations by approximating the model's behavior
    around a specific prediction using a simpler, interpretable model.
    """
    
    def __init__(self, model: torch.nn.Module, class_names: List[str] = None):
        """
        Initialize LIME explainer.
        
        Args:
            model: Trained model
            class_names: List of class names
        """
        self.model = model
        self.class_names = class_names or DISEASE_CLASSES
        
        # Initialize LIME explainer
        self.explainer = lime_image.LimeImageExplainer()
    
    def explain_instance(self, input_image: np.ndarray, class_idx: int = None, 
                        num_samples: int = 1000, hide_color: int = 0) -> lime.explanation.Explanation:
        """
        Explain a single instance using LIME.
        
        Args:
            input_image: Input image (H, W, C)
            class_idx: Target class index
            num_samples: Number of samples for explanation
            hide_color: Color to use for hiding parts of the image
            
        Returns:
            LIME explanation
        """
        def model_predict(images):
            """Wrapper function for model prediction."""
            # Preprocess images
            processed_images = []
            for img in images:
                # Convert to tensor and normalize
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0)
                
                # Apply normalization
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_tensor = (img_tensor - mean) / std
                
                processed_images.append(img_tensor)
            
            # Batch predict
            batch = torch.cat(processed_images, dim=0)
            with torch.no_grad():
                outputs = self.model(batch)
                probs = torch.sigmoid(outputs)
            
            return probs.cpu().numpy()
        
        # Generate explanation
        explanation = self.explainer.explain_instance(
            input_image,
            model_predict,
            top_labels=len(self.class_names),
            hide_color=hide_color,
            num_samples=num_samples
        )
        
        return explanation
    
    def get_explanation_mask(self, explanation: lime.explanation.Explanation, 
                           class_idx: int, num_features: int = 10) -> np.ndarray:
        """
        Get explanation mask for visualization.
        
        Args:
            explanation: LIME explanation
            class_idx: Target class index
            num_features: Number of top features to include
            
        Returns:
            Explanation mask
        """
        # Get top features for the class
        top_features = explanation.local_exp[class_idx][:num_features]
        
        # Create mask
        mask = np.zeros(explanation.segments.shape)
        for feature_idx, weight in top_features:
            mask[explanation.segments == feature_idx] = weight
        
        return mask


class ExplainabilityEngine:
    """
    Main explainability engine that combines multiple XAI techniques.
    
    This class provides a unified interface for generating explanations
    using different XAI methods and visualizing the results.
    """
    
    def __init__(self, model: torch.nn.Module, config: Config = None):
        """
        Initialize explainability engine.
        
        Args:
            model: Trained model
            config: Configuration object
        """
        self.model = model
        self.config = config or Config()
        self.device = self.config.get_device()
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize explainers
        self.gradcam = GradCAM(model, self.config.explainability.gradcam_target_layer)
        self.shap_explainer = None  # Initialize when needed
        self.lime_explainer = LIMEExplainer(model)
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration."""
        logger.add(
            "logs/explainability.log",
            rotation="10 MB",
            retention="7 days",
            level="INFO"
        )
    
    def initialize_shap(self, background_data: torch.Tensor = None):
        """Initialize SHAP explainer with background data."""
        if background_data is None:
            # Create random background data
            background_data = torch.randn(
                self.config.explainability.shap_background_size, 
                3, *self.config.data.image_size
            ).to(self.device)
        
        self.shap_explainer = SHAPExplainer(self.model, background_data)
        logger.info("SHAP explainer initialized")
    
    def generate_gradcam_explanations(self, images: torch.Tensor, 
                                    class_indices: List[int] = None) -> Dict[str, np.ndarray]:
        """
        Generate Grad-CAM explanations for multiple images.
        
        Args:
            images: Batch of images (B, C, H, W)
            class_indices: List of target class indices
            
        Returns:
            Dictionary containing Grad-CAM heatmaps and overlaid images
        """
        if class_indices is None:
            # Use predicted classes
            with torch.no_grad():
                outputs = self.model(images)
                class_indices = torch.argmax(outputs, dim=1).tolist()
        
        # Generate Grad-CAM heatmaps
        heatmaps = self.gradcam.generate_cam_batch(images, class_indices)
        
        # Convert images to numpy for visualization
        images_np = images.cpu().numpy().transpose(0, 2, 3, 1)
        
        # Denormalize images
        mean = np.array(self.config.data.normalization_mean)
        std = np.array(self.config.data.normalization_std)
        images_denorm = images_np * std + mean
        images_denorm = np.clip(images_denorm * 255, 0, 255).astype(np.uint8)
        
        # Create overlaid images
        overlaid_images = []
        for i, (image, heatmap) in enumerate(zip(images_denorm, heatmaps)):
            overlaid = self.gradcam.overlay_heatmap(
                image, heatmap, self.config.explainability.gradcam_alpha
            )
            overlaid_images.append(overlaid)
        
        return {
            'heatmaps': heatmaps,
            'overlaid_images': overlaid_images,
            'original_images': images_denorm,
            'class_indices': class_indices
        }
    
    def generate_shap_explanations(self, images: torch.Tensor, 
                                 class_indices: List[int] = None) -> Dict[str, np.ndarray]:
        """
        Generate SHAP explanations for multiple images.
        
        Args:
            images: Batch of images (B, C, H, W)
            class_indices: List of target class indices
            
        Returns:
            Dictionary containing SHAP values
        """
        if self.shap_explainer is None:
            self.initialize_shap()
        
        if class_indices is None:
            # Use predicted classes
            with torch.no_grad():
                outputs = self.model(images)
                class_indices = torch.argmax(outputs, dim=1).tolist()
        
        # Generate SHAP explanations
        shap_values = self.shap_explainer.explain_batch(images, class_indices)
        
        return {
            'shap_values': shap_values,
            'class_indices': class_indices
        }
    
    def generate_lime_explanations(self, images: np.ndarray, 
                                 class_indices: List[int] = None) -> Dict[str, Any]:
        """
        Generate LIME explanations for multiple images.
        
        Args:
            images: Batch of images (B, H, W, C) in uint8 format
            class_indices: List of target class indices
            
        Returns:
            Dictionary containing LIME explanations
        """
        if class_indices is None:
            # Convert images to tensor for prediction
            images_tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
            images_tensor = images_tensor.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(images_tensor)
                class_indices = torch.argmax(outputs, dim=1).tolist()
        
        # Generate LIME explanations
        explanations = []
        masks = []
        
        for i, (image, class_idx) in enumerate(zip(images, class_indices)):
            explanation = self.lime_explainer.explain_instance(image, class_idx)
            mask = self.lime_explainer.get_explanation_mask(explanation, class_idx)
            
            explanations.append(explanation)
            masks.append(mask)
        
        return {
            'explanations': explanations,
            'masks': masks,
            'class_indices': class_indices
        }
    
    def visualize_explanations(self, explanations: Dict[str, Any], 
                             save_path: str = None, num_images: int = 5) -> None:
        """
        Visualize explanations from different XAI methods.
        
        Args:
            explanations: Dictionary containing explanations
            save_path: Path to save visualizations
            num_images: Number of images to visualize
        """
        num_images = min(num_images, len(explanations.get('original_images', [])))
        
        fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))
        if num_images == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_images):
            # Original image
            axes[i, 0].imshow(explanations['original_images'][i])
            axes[i, 0].set_title(f"Original Image\nClass: {DISEASE_CLASSES[explanations['class_indices'][i]]}")
            axes[i, 0].axis('off')
            
            # Grad-CAM heatmap
            axes[i, 1].imshow(explanations['heatmaps'][i], cmap='jet')
            axes[i, 1].set_title("Grad-CAM Heatmap")
            axes[i, 1].axis('off')
            
            # Overlaid image
            axes[i, 2].imshow(explanations['overlaid_images'][i])
            axes[i, 2].set_title("Grad-CAM Overlay")
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Explanations visualization saved to {save_path}")
        
        plt.show()
    
    def compare_explanations(self, image: torch.Tensor, class_idx: int, 
                           save_path: str = None) -> None:
        """
        Compare explanations from different XAI methods for a single image.
        
        Args:
            image: Input image (1, C, H, W)
            class_idx: Target class index
            save_path: Path to save comparison
        """
        # Generate explanations
        gradcam_result = self.generate_gradcam_explanations(image, [class_idx])
        
        if self.shap_explainer is not None:
            shap_result = self.generate_shap_explanations(image, [class_idx])
        
        # Convert image for LIME
        image_np = image.cpu().numpy().transpose(0, 2, 3, 1)
        mean = np.array(self.config.data.normalization_mean)
        std = np.array(self.config.data.normalization_std)
        image_denorm = image_np[0] * std + mean
        image_denorm = np.clip(image_denorm * 255, 0, 255).astype(np.uint8)
        
        lime_result = self.generate_lime_explanations(
            image_denorm[np.newaxis, ...], [class_idx]
        )
        
        # Create comparison visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original image
        axes[0, 0].imshow(gradcam_result['original_images'][0])
        axes[0, 0].set_title(f"Original Image\nClass: {DISEASE_CLASSES[class_idx]}")
        axes[0, 0].axis('off')
        
        # Grad-CAM
        axes[0, 1].imshow(gradcam_result['heatmaps'][0], cmap='jet')
        axes[0, 1].set_title("Grad-CAM")
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(gradcam_result['overlaid_images'][0])
        axes[0, 2].set_title("Grad-CAM Overlay")
        axes[0, 2].axis('off')
        
        # LIME
        axes[1, 0].imshow(lime_result['masks'][0], cmap='RdBu')
        axes[1, 0].set_title("LIME Mask")
        axes[1, 0].axis('off')
        
        # SHAP (if available)
        if self.shap_explainer is not None:
            shap_values = shap_result['shap_values'][0]
            axes[1, 1].imshow(np.abs(shap_values).mean(axis=0), cmap='viridis')
            axes[1, 1].set_title("SHAP Values")
            axes[1, 1].axis('off')
        
        # Combined view
        axes[1, 2].imshow(gradcam_result['original_images'][0])
        axes[1, 2].imshow(lime_result['masks'][0], alpha=0.6, cmap='RdBu')
        axes[1, 2].set_title("Combined View")
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Explanation comparison saved to {save_path}")
        
        plt.show()
    
    def evaluate_explanations(self, test_loader, num_samples: int = 100) -> Dict[str, float]:
        """
        Evaluate the quality of explanations.
        
        Args:
            test_loader: Test data loader
            num_samples: Number of samples to evaluate
            
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Evaluating explanation quality...")
        
        # Generate explanations for test samples
        explanations = []
        predictions = []
        targets = []
        
        for i, (images, labels) in enumerate(test_loader):
            if i >= num_samples // test_loader.batch_size:
                break
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(images)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
            
            # Generate Grad-CAM explanations
            gradcam_result = self.generate_gradcam_explanations(images)
            
            explanations.extend(gradcam_result['heatmaps'])
            predictions.extend(preds.cpu().numpy())
            targets.extend(labels.cpu().numpy())
        
        # Calculate evaluation metrics
        metrics = {}
        
        # Average heatmap intensity
        avg_intensity = np.mean([np.mean(heatmap) for heatmap in explanations])
        metrics['avg_heatmap_intensity'] = avg_intensity
        
        # Heatmap focus (entropy)
        heatmap_entropies = []
        for heatmap in explanations:
            # Normalize heatmap
            heatmap_norm = heatmap / (np.sum(heatmap) + 1e-8)
            # Calculate entropy
            entropy = -np.sum(heatmap_norm * np.log(heatmap_norm + 1e-8))
            heatmap_entropies.append(entropy)
        
        metrics['avg_heatmap_entropy'] = np.mean(heatmap_entropies)
        
        # Prediction confidence correlation
        confidences = []
        for pred in predictions:
            confidence = np.max(pred)
            confidences.append(confidence)
        
        correlation = np.corrcoef(avg_intensity, confidences)[0, 1]
        metrics['confidence_intensity_correlation'] = correlation
        
        logger.info(f"Explanation evaluation completed: {metrics}")
        
        return metrics
    
    def cleanup(self):
        """Clean up resources."""
        self.gradcam.cleanup()
        logger.info("Explainability engine cleaned up")


def create_sample_explanations():
    """Create sample explanations for testing."""
    # This function creates sample explanations for demonstration
    # In a real scenario, you would use actual trained models and data
    
    # Create a dummy model
    model = create_model('resnet50', num_classes=14, pretrained=False)
    
    # Create sample images
    sample_images = torch.randn(5, 3, 224, 224)
    
    # Initialize explainability engine
    config = Config()
    engine = ExplainabilityEngine(model, config)
    
    # Generate explanations
    explanations = engine.generate_gradcam_explanations(sample_images)
    
    # Visualize
    engine.visualize_explanations(explanations, num_images=3)
    
    # Cleanup
    engine.cleanup()
    
    logger.info("Sample explanations created successfully")


if __name__ == "__main__":
    # Test explainability module
    create_sample_explanations()
    print("Explainability module test completed successfully!") 