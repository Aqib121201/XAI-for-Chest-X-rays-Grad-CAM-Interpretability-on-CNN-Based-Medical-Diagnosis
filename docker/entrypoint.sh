#!/bin/bash

# Entrypoint script for XAI Chest X-ray Analysis Docker container

set -e

echo "Starting XAI Chest X-ray Analysis Container..."

# Function to check if CUDA is available
check_cuda() {
    if command -v nvidia-smi &> /dev/null; then
        echo "CUDA is available:"
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
    else
        echo "CUDA not available, using CPU"
    fi
}

# Function to check Python environment
check_python() {
    echo "Python version: $(python --version)"
    echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
    echo "CUDA available in PyTorch: $(python -c 'import torch; print(torch.cuda.is_available())')"
}

# Function to validate project structure
validate_project() {
    echo "Validating project structure..."
    
    required_dirs=("src" "app" "data" "models" "visualizations" "logs")
    for dir in "${required_dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            echo "Warning: Required directory $dir not found"
        else
            echo "✓ Found directory: $dir"
        fi
    done
    
    required_files=("requirements.txt" "run_pipeline.py" "app/app.py")
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            echo "Warning: Required file $file not found"
        else
            echo "✓ Found file: $file"
        fi
    done
}

# Function to setup environment
setup_environment() {
    echo "Setting up environment..."
    
    # Create directories if they don't exist
    mkdir -p data/raw data/processed data/external \
           models visualizations logs results \
           notebooks tests
    
    # Set permissions
    chmod +x run_pipeline.py
    
    echo "Environment setup completed"
}

# Function to run tests
run_tests() {
    if [ "$RUN_TESTS" = "true" ]; then
        echo "Running tests..."
        python -m pytest tests/ -v
        echo "Tests completed"
    fi
}

# Function to start the application
start_app() {
    case "$APP_MODE" in
        "streamlit")
            echo "Starting Streamlit dashboard..."
            exec streamlit run app/app.py \
                --server.port=8501 \
                --server.address=0.0.0.0 \
                --server.headless=true \
                --browser.gatherUsageStats=false
            ;;
        "pipeline")
            echo "Running pipeline..."
            exec python run_pipeline.py "$@"
            ;;
        "jupyter")
            echo "Starting Jupyter notebook..."
            exec jupyter notebook \
                --ip=0.0.0.0 \
                --port=8888 \
                --no-browser \
                --allow-root \
                --NotebookApp.token='' \
                --NotebookApp.password=''
            ;;
        *)
            echo "Unknown APP_MODE: $APP_MODE"
            echo "Available modes: streamlit, pipeline, jupyter"
            exit 1
            ;;
    esac
}

# Main execution
main() {
    echo "=========================================="
    echo "XAI Chest X-ray Analysis Container"
    echo "=========================================="
    
    # Check system
    check_cuda
    check_python
    
    # Validate project
    validate_project
    
    # Setup environment
    setup_environment
    
    # Run tests if requested
    run_tests
    
    # Start application
    start_app "$@"
}

# Handle signals
trap 'echo "Received signal, shutting down..."; exit 0' SIGTERM SIGINT

# Run main function
main "$@" 