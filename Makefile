# Makefile for XAI Chest X-ray Analysis Project

.PHONY: help install test clean run-dashboard run-pipeline docker-build docker-run

# Default target
help:
	@echo "XAI Chest X-ray Analysis Project - Available Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  install          Install dependencies"
	@echo "  install-dev      Install development dependencies"
	@echo "  setup-env        Create virtual environment and install dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  test             Run all tests"
	@echo "  test-cov         Run tests with coverage"
	@echo "  lint             Run linting checks"
	@echo "  format           Format code with black"
	@echo ""
	@echo "Running:"
	@echo "  run-dashboard    Start Streamlit dashboard"
	@echo "  run-pipeline     Run the complete pipeline"
	@echo "  run-test         Run pipeline in test mode"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build     Build Docker image"
	@echo "  docker-run       Run Docker container"
	@echo ""
	@echo "Data:"
	@echo "  download-data    Download NIH Chest X-ray dataset"
	@echo "  preprocess       Run data preprocessing only"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean            Clean generated files"
	@echo "  clean-all        Clean all generated files and caches"

# Setup commands
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8 mypy

setup-env:
	python -m venv venv
	@echo "Virtual environment created. Activate it with:"
	@echo "source venv/bin/activate  # On Unix/macOS"
	@echo "venv\\Scripts\\activate     # On Windows"
	@echo "Then run: make install"

# Testing commands
test:
	python -m pytest tests/ -v

test-cov:
	python -m pytest tests/ --cov=src --cov-report=html --cov-report=term

lint:
	flake8 src/ tests/ --max-line-length=88 --ignore=E203,W503
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/ --line-length=88
	isort src/ tests/

# Running commands
run-dashboard:
	streamlit run app/app.py

run-pipeline:
	python run_pipeline.py --mode full

run-test:
	python run_pipeline.py --mode test

# Docker commands
docker-build:
	docker build -f docker/Dockerfile -t xai-chest-xray .

docker-run:
	docker run -p 8501:8501 --name xai-chest-xray-app xai-chest-xray

docker-run-gpu:
	docker run --gpus all -p 8501:8501 --name xai-chest-xray-app xai-chest-xray

# Data commands
download-data:
	@echo "Please download the NIH Chest X-ray dataset from:"
	@echo "https://www.kaggle.com/datasets/nih-chest-xrays/data"
	@echo "and place it in the data/raw/ directory"

preprocess:
	python run_pipeline.py --mode preprocess

# Cleanup commands
clean:
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -rf tests/__pycache__/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf logs/*.log
	rm -rf visualizations/*.png
	rm -rf visualizations/*.jpg
	rm -rf results/*.csv
	rm -rf results/*.json

clean-all: clean
	rm -rf venv/
	rm -rf .venv/
	rm -rf models/*.pth
	rm -rf checkpoints/
	rm -rf data/processed/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

# Development commands
dev-setup: install-dev
	@echo "Development environment setup complete!"
	@echo "Run 'make test' to verify installation"

quick-test:
	python -c "from src.config import Config; print('Config loaded successfully')"
	python -c "from src.data_preprocessing import DataPreprocessor; print('DataPreprocessor imported successfully')"
	python -c "from src.model_utils import create_model; print('Model utilities imported successfully')"
	python -c "from src.explainability import ExplainabilityEngine; print('Explainability imported successfully')"
	@echo "All modules imported successfully!"

# Documentation
docs:
	@echo "Generating documentation..."
	@echo "Please install sphinx and run:"
	@echo "sphinx-apidoc -o docs/source src/"
	@echo "cd docs && make html"

# Project status
status:
	@echo "=== Project Status ==="
	@echo "Python version: $(shell python --version)"
	@echo "PyTorch version: $(shell python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
	@echo "CUDA available: $(shell python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Unknown')"
	@echo "Dependencies installed: $(shell pip list | grep -E '(torch|streamlit|shap)' | wc -l) packages"
	@echo "Test files: $(shell find tests/ -name "*.py" | wc -l) files"
	@echo "Source files: $(shell find src/ -name "*.py" | wc -l) files"

# CI/CD helpers
ci-install:
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8 mypy

ci-test:
	python -m pytest tests/ --cov=src --cov-report=xml --cov-report=term-missing
	black --check src/ tests/
	flake8 src/ tests/ --max-line-length=88 --ignore=E203,W503
	mypy src/ --ignore-missing-imports

# Environment-specific commands
prod-setup:
	pip install -r requirements.txt
	python run_pipeline.py --mode full

staging-setup:
	pip install -r requirements.txt
	python run_pipeline.py --mode test

# Monitoring and logging
logs:
	tail -f logs/pipeline.log

logs-all:
	tail -f logs/*.log

# Backup and restore
backup:
	tar -czf backup-$(shell date +%Y%m%d-%H%M%S).tar.gz \
		--exclude='venv' \
		--exclude='__pycache__' \
		--exclude='.git' \
		--exclude='data/raw' \
		--exclude='models/*.pth' \
		.

# Performance testing
benchmark:
	python -c "
import time
import torch
from src.config import Config
from src.model_utils import create_model

config = Config()
model = create_model('resnet50', num_classes=14, pretrained=False)
input_tensor = torch.randn(1, 3, 224, 224)

# Warm up
for _ in range(10):
    _ = model(input_tensor)

# Benchmark
start_time = time.time()
for _ in range(100):
    _ = model(input_tensor)
end_time = time.time()

avg_time = (end_time - start_time) / 100
print(f'Average inference time: {avg_time*1000:.2f} ms')
print(f'Throughput: {1/avg_time:.1f} images/second')
" 