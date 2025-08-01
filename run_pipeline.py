#!/usr/bin/env python3
"""
Main pipeline script for XAI Chest X-ray Analysis Project.

This script orchestrates the complete workflow including:
- Data preprocessing
- Model training
- Evaluation
- Explainability analysis
- Results generation

Usage:
    python run_pipeline.py [--config CONFIG_FILE] [--mode MODE] [--data_dir DATA_DIR]
"""

import argparse
import sys
import os
from pathlib import Path
import logging
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import Config
from src.data_preprocessing import DataPreprocessor, create_sample_data_for_testing
from src.model_training import ModelTrainer, train_model_from_scratch
from src.explainability import ExplainabilityEngine, create_sample_explanations
from src.model_utils import create_model, load_model


def setup_logging():
    """Setup logging configuration."""
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Add file handler
    logger.add(
        "logs/pipeline.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="XAI Chest X-ray Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline with default settings
  python run_pipeline.py

  # Run with custom configuration
  python run_pipeline.py --config configs/custom_config.yaml

  # Run only data preprocessing
  python run_pipeline.py --mode preprocess

  # Run only training
  python run_pipeline.py --mode train

  # Run only explainability analysis
  python run_pipeline.py --mode explain

  # Run with custom data directory
  python run_pipeline.py --data_dir /path/to/data --metadata_file data_entries.csv
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (YAML)"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "preprocess", "train", "evaluate", "explain", "test"],
        default="full",
        help="Pipeline mode to run"
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing the dataset"
    )
    
    parser.add_argument(
        "--metadata_file",
        type=str,
        default=None,
        help="Path to metadata file"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to pre-trained model for evaluation/explainability"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (overrides config)"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def run_data_preprocessing(config: Config, data_dir: str = None, metadata_file: str = None):
    """Run data preprocessing pipeline."""
    logger.info("=" * 60)
    logger.info("STEP 1: DATA PREPROCESSING")
    logger.info("=" * 60)
    
    try:
        # Initialize data preprocessor
        preprocessor = DataPreprocessor(config)
        
        # Use provided paths or defaults
        if data_dir is None:
            data_dir = config.data.nih_dataset_path
        if metadata_file is None:
            metadata_file = config.data.nih_metadata_file
        
        # Check if data exists, if not create sample data for testing
        if not Path(data_dir).exists():
            logger.warning(f"Data directory {data_dir} not found. Creating sample data for testing.")
            data_dir, metadata_file = create_sample_data_for_testing()
        
        # Load and preprocess data
        preprocessor.load_data(data_dir, metadata_file)
        
        # Apply SMOTE if configured
        if config.data.use_smote:
            logger.info("Applying SMOTE for class balancing...")
            preprocessor.apply_smote_balancing()
        
        # Create data loaders
        train_loader, val_loader, test_loader = preprocessor.create_data_loaders()
        
        # Get dataset info
        dataset_info = preprocessor.get_dataset_info()
        
        # Save processed data info
        preprocessor.save_processed_data("data/processed")
        
        logger.info("Data preprocessing completed successfully!")
        logger.info(f"Dataset info: {dataset_info}")
        
        return preprocessor, train_loader, val_loader, test_loader
        
    except Exception as e:
        logger.error(f"Data preprocessing failed: {str(e)}")
        raise


def run_model_training(config: Config, train_loader, val_loader, test_loader, 
                      num_epochs: int = None, model_path: str = None):
    """Run model training pipeline."""
    logger.info("=" * 60)
    logger.info("STEP 2: MODEL TRAINING")
    logger.info("=" * 60)
    
    try:
        # Initialize model trainer
        trainer = ModelTrainer(config)
        
        # Override config parameters if provided
        if num_epochs is not None:
            config.model.num_epochs = num_epochs
        if args.batch_size is not None:
            config.model.batch_size = args.batch_size
        if args.learning_rate is not None:
            config.model.learning_rate = args.learning_rate
        
        # Setup model
        trainer.setup_model()
        
        # Load pre-trained model if provided
        if model_path and Path(model_path).exists():
            logger.info(f"Loading pre-trained model from {model_path}")
            trainer.load_checkpoint(model_path)
        
        # Train model
        trainer.train()
        
        # Evaluate model
        logger.info("Evaluating trained model...")
        results = trainer.evaluate(test_loader)
        
        # Generate explanations
        logger.info("Generating explanations...")
        explanations = trainer.generate_explanations(num_samples=10)
        
        # Plot training history
        trainer.plot_training_history()
        
        # Save results
        trainer.save_results(results, args.output_dir)
        
        logger.info("Model training completed successfully!")
        
        return trainer, results, explanations
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise


def run_evaluation(config: Config, model_path: str, test_loader):
    """Run model evaluation pipeline."""
    logger.info("=" * 60)
    logger.info("STEP 3: MODEL EVALUATION")
    logger.info("=" * 60)
    
    try:
        # Create model
        model = create_model(
            model_name=config.model.model_name,
            num_classes=config.model.num_classes,
            pretrained=False
        )
        
        # Load trained model
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint_info = load_model(model, model_path, device=config.get_device())
        
        # Initialize trainer for evaluation
        trainer = ModelTrainer(config)
        trainer.model = model
        trainer.test_loader = test_loader
        
        # Evaluate model
        results = trainer.evaluate(test_loader)
        
        # Save evaluation results
        trainer.save_results(results, args.output_dir)
        
        logger.info("Model evaluation completed successfully!")
        
        return results
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {str(e)}")
        raise


def run_explainability_analysis(config: Config, model_path: str, test_loader):
    """Run explainability analysis pipeline."""
    logger.info("=" * 60)
    logger.info("STEP 4: EXPLAINABILITY ANALYSIS")
    logger.info("=" * 60)
    
    try:
        # Create model
        model = create_model(
            model_name=config.model.model_name,
            num_classes=config.model.num_classes,
            pretrained=False
        )
        
        # Load trained model
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint_info = load_model(model, model_path, device=config.get_device())
        
        # Initialize explainability engine
        explainability_engine = ExplainabilityEngine(model, config)
        
        # Get sample data for explanations
        sample_images = []
        sample_labels = []
        
        for i, (images, labels) in enumerate(test_loader):
            if len(sample_images) >= 10:  # Generate explanations for 10 samples
                break
            sample_images.extend(images[:10 - len(sample_images)])
            sample_labels.extend(labels[:10 - len(sample_labels)])
        
        sample_images = torch.stack(sample_images).to(config.get_device())
        
        # Generate Grad-CAM explanations
        logger.info("Generating Grad-CAM explanations...")
        gradcam_explanations = explainability_engine.generate_gradcam_explanations(sample_images)
        
        # Initialize SHAP explainer
        logger.info("Initializing SHAP explainer...")
        explainability_engine.initialize_shap()
        
        # Generate SHAP explanations
        logger.info("Generating SHAP explanations...")
        shap_explanations = explainability_engine.generate_shap_explanations(sample_images)
        
        # Visualize explanations
        logger.info("Creating explanation visualizations...")
        explainability_engine.visualize_explanations(
            gradcam_explanations,
            save_path=f"{config.explainability.visualization_path}/gradcam_explanations.png"
        )
        
        # Compare explanations for a single image
        logger.info("Creating comparison visualizations...")
        explainability_engine.compare_explanations(
            sample_images[0:1],
            class_idx=0,
            save_path=f"{config.explainability.visualization_path}/explanation_comparison.png"
        )
        
        # Evaluate explanation quality
        logger.info("Evaluating explanation quality...")
        explanation_metrics = explainability_engine.evaluate_explanations(test_loader)
        
        # Save explanation results
        import json
        with open(f"{args.output_dir}/explanation_metrics.json", 'w') as f:
            json.dump(explanation_metrics, f, indent=2)
        
        # Cleanup
        explainability_engine.cleanup()
        
        logger.info("Explainability analysis completed successfully!")
        
        return {
            'gradcam': gradcam_explanations,
            'shap': shap_explanations,
            'metrics': explanation_metrics
        }
        
    except Exception as e:
        logger.error(f"Explainability analysis failed: {str(e)}")
        raise


def run_test_mode(config: Config):
    """Run test mode with sample data."""
    logger.info("=" * 60)
    logger.info("TEST MODE: Running with sample data")
    logger.info("=" * 60)
    
    try:
        # Create sample data
        logger.info("Creating sample data...")
        data_dir, metadata_file = create_sample_data_for_testing()
        
        # Run preprocessing
        preprocessor, train_loader, val_loader, test_loader = run_data_preprocessing(
            config, data_dir, metadata_file
        )
        
        # Run training (shorter for testing)
        config.model.num_epochs = 2  # Just 2 epochs for testing
        trainer, results, explanations = run_model_training(
            config, train_loader, val_loader, test_loader
        )
        
        # Run explainability analysis
        model_path = f"models/best_{config.model.model_name}.pth"
        explainability_results = run_explainability_analysis(config, model_path, test_loader)
        
        logger.info("Test mode completed successfully!")
        
        return {
            'preprocessor': preprocessor,
            'trainer': trainer,
            'results': results,
            'explanations': explanations,
            'explainability_results': explainability_results
        }
        
    except Exception as e:
        logger.error(f"Test mode failed: {str(e)}")
        raise


def main():
    """Main pipeline function."""
    global args
    args = parse_arguments()
    
    # Setup logging
    setup_logging()
    
    if args.verbose:
        logger.add(sys.stderr, level="DEBUG")
    
    logger.info("Starting XAI Chest X-ray Analysis Pipeline")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Load configuration
        config = Config(args.config)
        config.create_directories()
        config.validate()
        
        # Override config parameters if provided
        if args.batch_size is not None:
            config.model.batch_size = args.batch_size
        if args.learning_rate is not None:
            config.model.learning_rate = args.learning_rate
        if args.num_epochs is not None:
            config.model.num_epochs = args.num_epochs
        
        # Run pipeline based on mode
        if args.mode == "test":
            results = run_test_mode(config)
            
        elif args.mode == "preprocess":
            preprocessor, train_loader, val_loader, test_loader = run_data_preprocessing(
                config, args.data_dir, args.metadata_file
            )
            results = {'preprocessor': preprocessor}
            
        elif args.mode == "train":
            # Load data first
            preprocessor, train_loader, val_loader, test_loader = run_data_preprocessing(
                config, args.data_dir, args.metadata_file
            )
            
            # Train model
            trainer, training_results, explanations = run_model_training(
                config, train_loader, val_loader, test_loader, args.num_epochs, args.model_path
            )
            results = {
                'trainer': trainer,
                'results': training_results,
                'explanations': explanations
            }
            
        elif args.mode == "evaluate":
            # Load data first
            preprocessor, train_loader, val_loader, test_loader = run_data_preprocessing(
                config, args.data_dir, args.metadata_file
            )
            
            # Evaluate model
            if args.model_path is None:
                args.model_path = f"models/best_{config.model.model_name}.pth"
            
            evaluation_results = run_evaluation(config, args.model_path, test_loader)
            results = {'evaluation_results': evaluation_results}
            
        elif args.mode == "explain":
            # Load data first
            preprocessor, train_loader, val_loader, test_loader = run_data_preprocessing(
                config, args.data_dir, args.metadata_file
            )
            
            # Run explainability analysis
            if args.model_path is None:
                args.model_path = f"models/best_{config.model.model_name}.pth"
            
            explainability_results = run_explainability_analysis(config, args.model_path, test_loader)
            results = {'explainability_results': explainability_results}
            
        else:  # full mode
            # Run complete pipeline
            preprocessor, train_loader, val_loader, test_loader = run_data_preprocessing(
                config, args.data_dir, args.metadata_file
            )
            
            trainer, training_results, explanations = run_model_training(
                config, train_loader, val_loader, test_loader, args.num_epochs, args.model_path
            )
            
            # Run explainability analysis
            model_path = f"models/best_{config.model.model_name}.pth"
            explainability_results = run_explainability_analysis(config, model_path, test_loader)
            
            results = {
                'preprocessor': preprocessor,
                'trainer': trainer,
                'results': training_results,
                'explanations': explanations,
                'explainability_results': explainability_results
            }
        
        # Print summary
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
        if 'results' in results:
            logger.info("Training Results:")
            for metric, value in results['results']['metrics'].items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {metric}: {value:.4f}")
        
        if 'explainability_results' in results:
            logger.info("Explanation Quality Metrics:")
            for metric, value in results['explainability_results']['metrics'].items():
                logger.info(f"  {metric}: {value:.4f}")
        
        logger.info(f"Results saved to: {args.output_dir}")
        logger.info(f"Visualizations saved to: {config.explainability.visualization_path}")
        logger.info(f"Models saved to: {config.model.model_save_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main() 