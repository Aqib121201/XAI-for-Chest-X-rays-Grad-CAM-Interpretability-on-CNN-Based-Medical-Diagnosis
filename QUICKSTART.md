# Quick Start Guide - XAI Chest X-ray Analysis Project

This guide will help you get up and running with the XAI Chest X-ray Analysis project in minutes.

## 🚀 Quick Setup (5 minutes)

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd XAI-for-Chest-X-rays-Grad-CAM-Interpretability-on-CNN-Based-Medical-Diagnosis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Test Installation

```bash
# Run quick test to verify everything works
make quick-test
```

### 3. Run Demo

```bash
# Run the complete pipeline in test mode (uses sample data)
python run_pipeline.py --mode test
```

## 📊 What You Get

After running the demo, you'll have:

- ✅ **Trained Model**: ResNet50 trained on chest X-ray data
- ✅ **Explanations**: Grad-CAM heatmaps and SHAP visualizations
- ✅ **Dashboard**: Interactive Streamlit app
- ✅ **Results**: Performance metrics and visualizations

## 🎯 Key Features

### 🔍 **Multi-Disease Detection**
- Detects 14 common thoracic diseases
- Multi-label classification
- Handles class imbalance with SMOTE and focal loss

### 🧠 **Explainable AI**
- **Grad-CAM**: Visual heatmaps showing model focus areas
- **SHAP**: Feature importance analysis
- **LIME**: Local interpretable explanations

### 📈 **Interactive Dashboard**
- Upload your own chest X-ray images
- Get instant predictions and explanations
- Compare different XAI methods
- Performance analysis and visualizations

## 🛠️ Usage Examples

### Run Complete Pipeline

```bash
# Full training and evaluation
python run_pipeline.py --mode full

# With custom configuration
python run_pipeline.py --config configs/sample_config.yaml --mode full
```

### Start Interactive Dashboard

```bash
# Launch Streamlit dashboard
streamlit run app/app.py

# Or use make command
make run-dashboard
```

### Run Individual Components

```bash
# Data preprocessing only
python run_pipeline.py --mode preprocess

# Training only
python run_pipeline.py --mode train

# Explainability analysis only
python run_pipeline.py --mode explain
```

### Docker Deployment

```bash
# Build Docker image
make docker-build

# Run with GPU support
make docker-run-gpu

# Run with CPU only
make docker-run
```

## 📁 Project Structure

```
XAI-for-Chest-X-rays-Grad-CAM-Interpretability-on-CNN-Based-Medical-Diagnosis/
├── 📁 src/                   # Core source code
│   ├── config.py            # Configuration management
│   ├── data_preprocessing.py # Data loading and preprocessing
│   ├── model_training.py    # Training pipeline
│   ├── model_utils.py       # Model utilities and metrics
│   └── explainability.py    # XAI implementations
├── 📁 app/                  # Streamlit dashboard
│   └── app.py              # Interactive web interface
├── 📁 notebooks/           # Jupyter notebooks for EDA
├── 📁 tests/               # Unit tests
├── 📁 docker/              # Docker configuration
├── 📁 configs/             # Configuration files
├── run_pipeline.py         # Main execution script
├── requirements.txt        # Python dependencies
├── Makefile               # Project management commands
└── README.md              # Detailed documentation
```

## 🎮 Interactive Dashboard Features

### Home Page
- Project overview and key features
- Quick stats and model information
- Architecture visualization

### Model Predictions
- Upload chest X-ray images
- Get disease predictions with confidence scores
- View Grad-CAM explanations
- Compare multiple XAI methods

### Performance Analysis
- Overall model metrics
- Per-disease performance breakdown
- Interactive visualizations
- Cross-validation results

### Explainability Analysis
- Interactive explanation generation
- Method comparison (Grad-CAM vs SHAP vs LIME)
- Clinical relevance insights
- Explanation quality metrics

## 🔧 Customization

### Configuration

Copy and modify the sample configuration:

```bash
cp configs/sample_config.yaml configs/my_config.yaml
# Edit my_config.yaml with your settings
python run_pipeline.py --config configs/my_config.yaml
```

### Key Parameters to Tune

```yaml
model:
  model_name: "resnet50"        # Try: resnet101, densenet121
  batch_size: 32               # Adjust based on GPU memory
  learning_rate: 0.0001        # Learning rate
  num_epochs: 100              # Training epochs

data:
  use_smote: true              # Class balancing
  rotation_range: 15           # Data augmentation
  horizontal_flip_prob: 0.5    # Data augmentation

explainability:
  gradcam_target_layer: "layer4.2.conv3"  # Target layer for Grad-CAM
  shap_background_size: 1000   # SHAP background samples
```

## 🧪 Testing

```bash
# Run all tests
make test

# Run tests with coverage
make test-cov

# Run linting
make lint

# Format code
make format
```

## 📊 Expected Results

### Performance Metrics
- **Accuracy**: ~85%
- **AUROC**: ~89%
- **F1-Score**: ~83%

### Supported Diseases
1. Atelectasis
2. Cardiomegaly
3. Effusion
4. Infiltration
5. Mass
6. Nodule
7. Pneumonia
8. Pneumothorax
9. Consolidation
10. Edema
11. Emphysema
12. Fibrosis
13. Pleural Thickening
14. Hernia

## 🚨 Troubleshooting

### Common Issues

**CUDA out of memory:**
```bash
# Reduce batch size in config
model:
  batch_size: 16  # or 8
```

**Missing dependencies:**
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

**Model not loading:**
```bash
# Check if model file exists
ls models/
# If not, run training first
python run_pipeline.py --mode train
```

### Getting Help

1. Check the logs: `tail -f logs/pipeline.log`
2. Run tests: `make test`
3. Check project status: `make status`
4. Review the full README.md for detailed documentation

## 🎯 Next Steps

1. **Download Real Data**: Get the NIH Chest X-ray dataset
2. **Customize Configuration**: Modify parameters for your use case
3. **Train Full Model**: Run complete training pipeline
4. **Deploy**: Use Docker for production deployment
5. **Extend**: Add new XAI methods or model architectures

## 📚 Additional Resources

- [Full Documentation](README.md)
- [Configuration Guide](configs/sample_config.yaml)
- [API Reference](src/)
- [Test Suite](tests/)
- [Docker Guide](docker/)

