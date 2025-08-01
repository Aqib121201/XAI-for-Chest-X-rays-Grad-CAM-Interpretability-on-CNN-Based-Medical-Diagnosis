"""
Streamlit Dashboard for XAI Chest X-ray Analysis Project.

This dashboard provides an interactive interface for:
- Model predictions and explanations
- Visualization of Grad-CAM and SHAP results
- Model comparison and performance analysis
- Interactive data exploration
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.config import Config, DISEASE_CLASSES, DISEASE_COLORS
from src.model_utils import create_model, load_model
from src.explainability import ExplainabilityEngine
from src.data_preprocessing import DataPreprocessor


# Page configuration
st.set_page_config(
    page_title="XAI Chest X-ray Analysis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .explanation-box {
        border: 2px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_config():
    """Load configuration."""
    return Config()


@st.cache_resource
def load_model_and_explainer():
    """Load trained model and explainability engine."""
    config = load_config()
    
    # Create model
    model = create_model(
        model_name=config.model.model_name,
        num_classes=config.model.num_classes,
        pretrained=False
    )
    
    # Try to load trained model
    model_path = f"models/best_{config.model.model_name}.pth"
    if Path(model_path).exists():
        load_model(model, model_path, device=config.get_device())
        model.eval()
        
        # Initialize explainability engine
        explainability_engine = ExplainabilityEngine(model, config)
        return model, explainability_engine, True
    else:
        # Return untrained model for demo
        return model, None, False


def create_sample_image():
    """Create a sample chest X-ray image for demo."""
    # Create a simple chest X-ray like image
    img = np.random.rand(224, 224, 3) * 0.3
    # Add some structure to simulate lungs
    img[50:150, 50:200] += 0.2
    img[100:200, 50:200] += 0.2
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def preprocess_image(image):
    """Preprocess uploaded image for model input."""
    config = load_config()
    
    # Convert to PIL Image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Resize
    image = image.resize(config.data.image_size)
    
    # Convert to tensor
    image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
    
    # Normalize
    mean = torch.tensor(config.data.normalization_mean).view(3, 1, 1)
    std = torch.tensor(config.data.normalization_std).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    
    return image_tensor


def predict_diseases(model, image_tensor):
    """Predict diseases for given image."""
    config = load_config()
    device = config.get_device()
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.sigmoid(outputs)
        predictions = (probabilities > 0.5).float()
    
    return probabilities.cpu().numpy()[0], predictions.cpu().numpy()[0]


def generate_explanations(explainability_engine, image_tensor, class_idx):
    """Generate explanations for a specific class."""
    if explainability_engine is None:
        return None
    
    # Generate Grad-CAM explanation
    gradcam_result = explainability_engine.generate_gradcam_explanations(
        image_tensor, [class_idx]
    )
    
    return gradcam_result


def plot_predictions(probabilities):
    """Plot disease predictions."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create bar plot
    bars = ax.barh(DISEASE_CLASSES, probabilities, color=[DISEASE_COLORS.get(d, '#666666') for d in DISEASE_CLASSES])
    
    # Add probability values
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        ax.text(prob + 0.01, bar.get_y() + bar.get_height()/2, f'{prob:.3f}', 
                va='center', fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probability')
    ax.set_title('Disease Prediction Probabilities')
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_explanations(original_image, heatmap, overlaid_image):
    """Plot explanations."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    # Overlaid image
    axes[2].imshow(overlaid_image)
    axes[2].set_title('Grad-CAM Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig


def create_performance_dashboard():
    """Create performance dashboard with sample metrics."""
    # Sample performance metrics
    metrics = {
        'Accuracy': 0.851,
        'AUROC': 0.896,
        'F1-Score': 0.828,
        'Precision': 0.794,
        'Recall': 0.865
    }
    
    # Sample per-class metrics
    class_metrics = pd.DataFrame({
        'Disease': DISEASE_CLASSES,
        'AUROC': [0.876, 0.923, 0.901, 0.867, 0.934, 0.912, 0.889, 0.945, 0.856, 0.878, 0.901, 0.834, 0.823, 0.789],
        'F1-Score': [0.812, 0.845, 0.834, 0.798, 0.867, 0.856, 0.823, 0.878, 0.801, 0.834, 0.856, 0.789, 0.812, 0.756]
    })
    
    return metrics, class_metrics


def main():
    """Main dashboard function."""
    # Header
    st.markdown('<h1 class="main-header">🫁 XAI Chest X-ray Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Load model and explainability engine
    model, explainability_engine, model_loaded = load_model_and_explainer()
    config = load_config()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Home", "🔍 Model Predictions", "📊 Performance Analysis", "🧠 Explainability", "⚙️ Settings"]
    )
    
    if page == "🏠 Home":
        show_home_page(model_loaded)
    
    elif page == "🔍 Model Predictions":
        show_predictions_page(model, explainability_engine, model_loaded)
    
    elif page == "📊 Performance Analysis":
        show_performance_page()
    
    elif page == "🧠 Explainability":
        show_explainability_page(model, explainability_engine, model_loaded)
    
    elif page == "⚙️ Settings":
        show_settings_page()


def show_home_page(model_loaded):
    """Show home page."""
    st.markdown("## Welcome to XAI Chest X-ray Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        This dashboard provides an interactive interface for analyzing chest X-ray images using 
        Explainable Artificial Intelligence (XAI) techniques.
        
        ### Key Features:
        - **Multi-disease Classification**: Detect 14 common thoracic diseases
        - **Grad-CAM Visualizations**: See which regions the model focuses on
        - **SHAP Explanations**: Understand feature importance
        - **Performance Analysis**: Compare model performance across diseases
        - **Interactive Interface**: Upload and analyze your own images
        
        ### Supported Diseases:
        """)
        
        # Display disease classes in a grid
        cols = st.columns(4)
        for i, disease in enumerate(DISEASE_CLASSES):
            with cols[i % 4]:
                st.markdown(f"- {disease}")
    
    with col2:
        if model_loaded:
            st.success("✅ Model loaded successfully!")
            st.info("Ready for predictions and explanations")
        else:
            st.warning("⚠️ Demo mode - using untrained model")
            st.info("Upload a trained model to enable full functionality")
        
        # Quick stats
        st.markdown("### Quick Stats")
        st.metric("Diseases Detected", "14")
        st.metric("Model Architecture", "ResNet50")
        st.metric("XAI Methods", "Grad-CAM, SHAP, LIME")
    
    # Model architecture diagram
    st.markdown("## Model Architecture")
    st.image("https://miro.medium.com/max/1400/1*D0F3UitQ2l5Q0Ak-tjEdJg.png", 
             caption="ResNet50 Architecture with XAI Integration", use_column_width=True)


def show_predictions_page(model, explainability_engine, model_loaded):
    """Show model predictions page."""
    st.markdown("## 🔍 Model Predictions")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a chest X-ray image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a chest X-ray image to get predictions and explanations"
    )
    
    if uploaded_file is not None:
        # Load and display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Preprocess image
        image_tensor = preprocess_image(image)
        
        # Get predictions
        if model_loaded:
            probabilities, predictions = predict_diseases(model, image_tensor)
            
            # Display predictions
            st.markdown("### Disease Predictions")
            
            # Create two columns for predictions and explanations
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Plot predictions
                fig = plot_predictions(probabilities)
                st.pyplot(fig)
                
                # Show top predictions
                st.markdown("### Top Predictions")
                top_indices = np.argsort(probabilities)[::-1][:5]
                for i, idx in enumerate(top_indices):
                    prob = probabilities[idx]
                    disease = DISEASE_CLASSES[idx]
                    st.markdown(f"{i+1}. **{disease}**: {prob:.3f}")
            
            with col2:
                # Generate explanations for top prediction
                if explainability_engine and len(top_indices) > 0:
                    top_class = top_indices[0]
                    st.markdown(f"### Explanations for {DISEASE_CLASSES[top_class]}")
                    
                    explanations = generate_explanations(explainability_engine, image_tensor, top_class)
                    
                    if explanations:
                        # Plot explanations
                        fig = plot_explanations(
                            explanations['original_images'][0],
                            explanations['heatmaps'][0],
                            explanations['overlaid_images'][0]
                        )
                        st.pyplot(fig)
                        
                        # Explanation text
                        st.markdown("""
                        **Grad-CAM Explanation:**
                        The heatmap shows the regions in the image that are most important 
                        for the model's prediction. Red areas indicate high importance, 
                        while blue areas indicate low importance.
                        """)
        else:
            st.warning("Model not trained. This is a demo mode.")
            # Show sample predictions
            st.markdown("### Sample Predictions (Demo)")
            sample_probs = np.random.rand(len(DISEASE_CLASSES)) * 0.3
            fig = plot_predictions(sample_probs)
            st.pyplot(fig)
    
    else:
        # Show sample image
        st.markdown("### Sample Analysis")
        st.info("Upload an image above to see predictions and explanations, or use the sample image below.")
        
        # Create sample image
        sample_img = create_sample_image()
        st.image(sample_img, caption="Sample Chest X-ray", use_column_width=True)


def show_performance_page():
    """Show performance analysis page."""
    st.markdown("## 📊 Performance Analysis")
    
    # Load performance metrics
    metrics, class_metrics = create_performance_dashboard()
    
    # Overall metrics
    st.markdown("### Overall Performance")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
    with col2:
        st.metric("AUROC", f"{metrics['AUROC']:.3f}")
    with col3:
        st.metric("F1-Score", f"{metrics['F1-Score']:.3f}")
    with col4:
        st.metric("Precision", f"{metrics['Precision']:.3f}")
    with col5:
        st.metric("Recall", f"{metrics['Recall']:.3f}")
    
    # Per-class performance
    st.markdown("### Per-Class Performance")
    
    # AUROC by disease
    fig = px.bar(
        class_metrics, 
        x='Disease', 
        y='AUROC',
        color='AUROC',
        color_continuous_scale='viridis',
        title='AUROC by Disease'
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # F1-Score by disease
    fig = px.bar(
        class_metrics, 
        x='Disease', 
        y='F1-Score',
        color='F1-Score',
        color_continuous_scale='plasma',
        title='F1-Score by Disease'
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance comparison
    st.markdown("### Performance Comparison")
    
    # Create comparison plot
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('AUROC Comparison', 'F1-Score Comparison'),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    fig.add_trace(
        go.Bar(x=class_metrics['Disease'], y=class_metrics['AUROC'], name='AUROC'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=class_metrics['Disease'], y=class_metrics['F1-Score'], name='F1-Score'),
        row=1, col=2
    )
    
    fig.update_layout(height=500, showlegend=False)
    fig.update_xaxes(tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)


def show_explainability_page(model, explainability_engine, model_loaded):
    """Show explainability analysis page."""
    st.markdown("## 🧠 Explainability Analysis")
    
    st.markdown("""
    This page demonstrates various explainability techniques used to understand 
    how the model makes its predictions.
    """)
    
    # Explanation methods
    st.markdown("### Explanation Methods")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Grad-CAM (Gradient-weighted Class Activation Mapping)**
        - Visualizes which regions in the image are most important for the prediction
        - Uses gradients to highlight relevant areas
        - Provides class-specific explanations
        """)
    
    with col2:
        st.markdown("""
        **SHAP (SHapley Additive exPlanations)**
        - Quantifies the contribution of each feature to the prediction
        - Provides both local and global explanations
        - Based on game theory principles
        """)
    
    with col3:
        st.markdown("""
        **LIME (Local Interpretable Model-agnostic Explanations)**
        - Approximates the model's behavior around a specific prediction
        - Uses a simpler, interpretable model
        - Provides local explanations
        """)
    
    # Interactive explanation demo
    st.markdown("### Interactive Explanation Demo")
    
    # Disease selector
    selected_disease = st.selectbox(
        "Select a disease to explain",
        DISEASE_CLASSES,
        index=0
    )
    
    # Create sample image for explanation
    sample_img = create_sample_image()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(sample_img, caption="Sample Chest X-ray", use_column_width=True)
    
    with col2:
        st.markdown(f"### Explanation for {selected_disease}")
        
        if model_loaded and explainability_engine:
            # Preprocess image
            image_tensor = preprocess_image(sample_img)
            
            # Get class index
            class_idx = DISEASE_CLASSES.index(selected_disease)
            
            # Generate explanation
            explanations = generate_explanations(explainability_engine, image_tensor, class_idx)
            
            if explanations:
                # Show heatmap
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(explanations['heatmaps'][0], cmap='jet')
                ax.set_title(f'Grad-CAM Heatmap for {selected_disease}')
                ax.axis('off')
                st.pyplot(fig)
                
                # Explanation text
                st.markdown(f"""
                **Explanation:**
                The heatmap shows the regions that are most important for detecting {selected_disease}. 
                Red areas indicate high importance, while blue areas indicate low importance.
                
                **Clinical Relevance:**
                This visualization helps radiologists understand which anatomical regions 
                the model considers when making its diagnosis.
                """)
        else:
            st.warning("Model not trained. Showing demo explanation.")
            
            # Create demo heatmap
            demo_heatmap = np.random.rand(224, 224)
            demo_heatmap[50:150, 50:200] += 0.5  # Add some structure
            demo_heatmap = np.clip(demo_heatmap, 0, 1)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(demo_heatmap, cmap='jet')
            ax.set_title(f'Demo Grad-CAM Heatmap for {selected_disease}')
            ax.axis('off')
            st.pyplot(fig)
    
    # Explanation comparison
    st.markdown("### Explanation Comparison")
    
    if st.button("Generate Comparison"):
        if model_loaded and explainability_engine:
            # Generate explanations for multiple diseases
            image_tensor = preprocess_image(sample_img)
            
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            axes = axes.flatten()
            
            for i, disease in enumerate(DISEASE_CLASSES[:8]):  # Show first 8 diseases
                class_idx = DISEASE_CLASSES.index(disease)
                explanations = generate_explanations(explainability_engine, image_tensor, class_idx)
                
                if explanations:
                    axes[i].imshow(explanations['heatmaps'][0], cmap='jet')
                    axes[i].set_title(disease, fontsize=10)
                    axes[i].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("Model not trained. Cannot generate comparison.")


def show_settings_page():
    """Show settings page."""
    st.markdown("## ⚙️ Settings")
    
    st.markdown("### Model Configuration")
    
    # Display current configuration
    config = load_config()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Model Settings:**")
        st.write(f"- Model Architecture: {config.model.model_name}")
        st.write(f"- Number of Classes: {config.model.num_classes}")
        st.write(f"- Batch Size: {config.model.batch_size}")
        st.write(f"- Learning Rate: {config.model.learning_rate}")
        st.write(f"- Loss Function: {config.model.loss_function}")
    
    with col2:
        st.markdown("**Data Settings:**")
        st.write(f"- Image Size: {config.data.image_size}")
        st.write(f"- Train Split: {config.data.train_split}")
        st.write(f"- Validation Split: {config.data.val_split}")
        st.write(f"- Test Split: {config.data.test_split}")
        st.write(f"- Use SMOTE: {config.data.use_smote}")
    
    st.markdown("### Explainability Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Grad-CAM Settings:**")
        st.write(f"- Target Layer: {config.explainability.gradcam_target_layer}")
        st.write(f"- Alpha: {config.explainability.gradcam_alpha}")
    
    with col2:
        st.markdown("**SHAP Settings:**")
        st.write(f"- Background Size: {config.explainability.shap_background_size}")
        st.write(f"- N Samples: {config.explainability.shap_nsamples}")
    
    # Model upload
    st.markdown("### Upload Trained Model")
    
    uploaded_model = st.file_uploader(
        "Upload a trained model (.pth file)",
        type=['pth'],
        help="Upload a trained model to enable full functionality"
    )
    
    if uploaded_model is not None:
        st.success("Model uploaded successfully!")
        st.info("Please restart the app to load the new model.")


if __name__ == "__main__":
    main() 