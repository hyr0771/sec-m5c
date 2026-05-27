#!/usr/bin/env python3
"""
Training script for the three-stage RNA sequence analysis model.
Supports stage-wise training, parameter configuration, and model checkpoint management.
"""

import argparse
import sys
import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config import Config, get_config
from training_framework import ProgressiveTrainer
from logger import get_logger


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train RNA sequence analysis model with progressive stages",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--data-file", 
        type=str, 
        default=None,
        help="Path to sequence data file (FASTA, CSV, or Excel). If not provided, will try to read from config."
    )
    
    # Stage control
    parser.add_argument(
        "--stage", 
        type=int, 
        choices=[1, 2], 
        default=None,
        help="Run specific stage only (1: sequence, 2: structure). If not specified, runs both stages."
    )
    
    parser.add_argument(
        "--stages", 
        type=str, 
        default="1,2",
        help="Comma-separated list of stages to run (e.g., '1,2')"
    )
    
    # Configuration
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="Path to configuration file. If not provided, uses default config."
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=None,
        help="Output directory for models, logs, and checkpoints (default: use paths from --config)"
    )
    
    # Model paths for stage control
    parser.add_argument(
        "--stage1-model", 
        type=str, 
        default=None,
        help="Path to stage 1 model checkpoint (for stages 2 and 3)"
    )
    
    parser.add_argument(
        "--stage2-model", 
        type=str, 
        default=None,
        help="Path to stage 2 model checkpoint (for stage 3)"
    )
    
    # Structure data
    parser.add_argument(
        "--structure-file", 
        type=str, 
        default=None,
        help="Path to structure data file (optional)"
    )
    
    # Training parameters (with factor-style naming)
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=None,
        help="Training batch size"
    )
    
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=None,
        help="Learning rate for all stages"
    )
    
    parser.add_argument(
        "--stage1-lr", 
        type=float, 
        default=None,
        help="Learning rate for stage 1"
    )
    
    parser.add_argument(
        "--stage2-lr", 
        type=float, 
        default=None,
        help="Learning rate for stage 2"
    )
    
    # stage3-lr removed
    
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=None,
        help="Number of epochs for all stages"
    )
    
    parser.add_argument(
        "--stage1-epochs", 
        type=int, 
        default=None,
        help="Number of epochs for stage 1"
    )
    
    parser.add_argument(
        "--stage2-epochs", 
        type=int, 
        default=None,
        help="Number of epochs for stage 2"
    )
    
    # stage3-epochs removed
    
    # Factor-style parameters
    parser.add_argument(
        "--factor", 
        type=float, 
        default=None,
        help="Global scaling factor for learning rates and other parameters"
    )
    
    parser.add_argument(
        "--dropout-factor", 
        type=float, 
        default=None,
        help="Scaling factor for dropout rates"
    )
    
    parser.add_argument(
        "--hidden-factor", 
        type=float, 
        default=None,
        help="Scaling factor for hidden dimensions"
    )
    
    # Device and performance
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for training"
    )
    
    parser.add_argument(
        "--num-workers", 
        type=int, 
        default=None,
        help="Number of data loading workers"
    )
    
    # Logging and checkpointing
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    parser.add_argument(
        "--save-interval", 
        type=int, 
        default=None,
        help="Save checkpoint every N epochs"
    )
    
    parser.add_argument(
        "--no-save", 
        action="store_true",
        help="Don't save model checkpoints"
    )
    
    # Early stopping
    parser.add_argument(
        "--early-stopping-patience", 
        type=int, 
        default=None,
        help="Early stopping patience (epochs)"
    )
    
    # Experiment settings
    parser.add_argument(
        "--experiment-name", 
        type=str, 
        default=None,
        help="Name for this experiment (used in logging and output paths)"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None,
        help="Random seed for reproducibility"
    )
    
    # Validation and testing
    parser.add_argument(
        "--no-validation", 
        action="store_true",
        help="Skip validation during training"
    )
    
    parser.add_argument(
        "--test-only", 
        action="store_true",
        help="Only run final evaluation on test set"
    )
    
    # Stage 3 specific parameters removed
    
    return parser.parse_args()


def update_config_from_args(config: Config, args: argparse.Namespace) -> Config:
    """Update configuration with command line arguments"""
    
    # Apply factor-based scaling first
    if args.factor is not None:
        # Scale learning rates
        if hasattr(config.training, 'stage1_lr'):
            config.training.stage1_lr *= args.factor
        if hasattr(config.training, 'stage2_lr'):
            config.training.stage2_lr *= args.factor
    
    if args.dropout_factor is not None:
        # Scale dropout rates
        if hasattr(config.model, 'dropout_prob'):
            config.model.dropout_prob *= args.dropout_factor
    
    if args.hidden_factor is not None:
        # Scale hidden dimensions
        if hasattr(config.model, 'hidden_size'):
            config.model.hidden_size = int(config.model.hidden_size * args.hidden_factor)
        if hasattr(config.model, 'intermediate_size'):
            config.model.intermediate_size = int(config.model.intermediate_size * args.hidden_factor)
    
    # Apply direct parameter overrides
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    
    if args.learning_rate is not None:
        config.training.stage1_lr = args.learning_rate
        config.training.stage2_lr = args.learning_rate
    
    if args.stage1_lr is not None:
        config.training.stage1_lr = args.stage1_lr
    if args.stage2_lr is not None:
        config.training.stage2_lr = args.stage2_lr
    # stage3 lr removed
    
    if args.epochs is not None:
        config.training.stage1_epochs = args.epochs
        config.training.stage2_epochs = args.epochs
    
    if args.stage1_epochs is not None:
        config.training.stage1_epochs = args.stage1_epochs
    if args.stage2_epochs is not None:
        config.training.stage2_epochs = args.stage2_epochs
    # stage3 epochs removed
    
    if args.num_workers is not None:
        config.data.num_workers = args.num_workers
    
    if args.early_stopping_patience is not None:
        config.experiment.early_stopping_patience = args.early_stopping_patience
    
    if args.device != "auto":
        config.experiment.device = args.device
    
    if args.experiment_name is not None:
        config.experiment.experiment_name = args.experiment_name
    
    if args.seed is not None:
        config.experiment.seed = args.seed
    
    # Update output directories
    if args.output_dir:
        config.experiment.output_dir = args.output_dir
        config.experiment.log_dir = os.path.join(args.output_dir, "logs")
        config.experiment.checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    
    # Stage 3 param updates removed
    
    return config


def setup_model_paths(trainer: ProgressiveTrainer, args: argparse.Namespace) -> None:
    """Setup model paths for stage-specific training"""
    
    # Set up paths for loading previous stage models
    if args.stage1_model and os.path.exists(args.stage1_model):
        trainer.stage1_model_path = args.stage1_model
    else:
        trainer.stage1_model_path = trainer.config.get_model_save_path("stage1", "best")
    
    if args.stage2_model and os.path.exists(args.stage2_model):
        trainer.stage2_model_path = args.stage2_model
    else:
        trainer.stage2_model_path = trainer.config.get_model_save_path("stage2", "best")


def run_single_stage(trainer: ProgressiveTrainer, stage: int, args: argparse.Namespace) -> Dict[str, Any]:
    """Run a single training stage"""
    
    logger = trainer.logger
    
    # Data should already be set up by run_multiple_stages or individual stage calls
    # if trainer.train_loader is None:
    #     trainer.setup_data(args.data_file, args.structure_file)
    
    if stage == 1:
        logger.info("Running Stage 1: Sequence Pretraining")
        trainer.setup_stage1_model()
        return {"stage1": trainer.train_stage1()}
    
    elif stage == 2:
        logger.info("Running Stage 2: Structure Enhancement")
        
        # Load stage 1 model if needed
        if trainer.sequence_model is None:
            if hasattr(trainer, 'stage1_model_path') and os.path.exists(trainer.stage1_model_path):
                logger.info(f"Loading stage 1 model from: {trainer.stage1_model_path}")
            else:
                logger.info("Stage 1 model not found, training stage 1 first...")
                trainer.setup_stage1_model()
                trainer.train_stage1()
        
        trainer.setup_stage2_model()
        return {"stage2": trainer.train_stage2()}
    
    # stage 3 branch removed
    
    else:
        raise ValueError(f"Invalid stage: {stage}")


def run_multiple_stages(trainer: ProgressiveTrainer, stages: list, args: argparse.Namespace) -> Dict[str, Any]:
    """Run multiple training stages"""
    
    results = {}
    
    # Don't setup data once - let each stage setup its own data
    # trainer.setup_data(args.data_file, args.structure_file)
    
    for stage in stages:
        # Setup data for specific stage
        trainer.setup_data_for_stage(stage)
        stage_results = run_single_stage(trainer, stage, args)
        results.update(stage_results)
    
    # Final evaluation if stage 3 was included
    if 3 in stages and not args.test_only:
        trainer.logger.info("Running final evaluation...")
        final_metrics = trainer._final_evaluation()
        results['final_evaluation'] = final_metrics
    
    return results


def main():
    """Main training function"""
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting RNA sequence analysis model training")
    logger.info(f"Arguments: {vars(args)}")
    
    
    try:
        # Load and update configuration
        raw_config = None
        if args.config and os.path.exists(args.config):
            config = get_config(args.config)
            logger.info(f"Loaded configuration from: {args.config}")
            try:
                with open(args.config, 'r') as f:
                    raw_config = json.load(f)
            except Exception as e:
                logging.getLogger(__name__).warning(f"Could not parse raw config JSON: {e}")
        else:
            config = Config()
            logger.info("Using default configuration")
        
        # Update config with command line arguments (CLI should override config)
        config = update_config_from_args(config, args)
        
        # Resolve data and structure files from CLI or config
        def _pick_stage(a: argparse.Namespace) -> int:
            if a.stage is not None:
                return a.stage
            try:
                stages_list = [int(s.strip()) for s in a.stages.split(',')]
                for s in stages_list:
                    if s in [1, 2]:
                        return s
            except Exception:
                pass
            return 1
        
        def _parse_stages(a: argparse.Namespace) -> list:
            if a.stage is not None:
                return [a.stage]
            try:
                stages_list = [int(s.strip()) for s in a.stages.split(',')]
                stages_list = [s for s in stages_list if s in [1, 2]]
                return stages_list or [1, 2]
            except Exception:
                return [1, 2]
        
        if not getattr(args, 'data_file', None):
            selected_stage = _pick_stage(args)
            seq_path = None
            if raw_config and isinstance(raw_config, dict) and 'data' in raw_config:
                data_section = raw_config['data'] or {}
                stage_key = {1: 'stage1_data_path', 2: 'stage2_data_path'}.get(selected_stage)
                if stage_key and data_section.get(stage_key):
                    seq_path = data_section.get(stage_key)
                if not seq_path and data_section.get('sequence_data_path'):
                    seq_path = data_section.get('sequence_data_path')
            if not seq_path and hasattr(config, 'data') and hasattr(config.data, 'sequence_data_path'):
                seq_path = getattr(config.data, 'sequence_data_path', None)
            args.data_file = seq_path
        
        stages_planned = _parse_stages(args)
        need_structure = any(s in (2,) for s in stages_planned)
        if not getattr(args, 'structure_file', None):
            if need_structure:
                struct_path = None
                if raw_config and isinstance(raw_config, dict) and 'data' in raw_config:
                    struct_path = (raw_config['data'] or {}).get('structure_data_path')
                if not struct_path and hasattr(config, 'data') and hasattr(config.data, 'structure_data_path'):
                    struct_path = getattr(config.data, 'structure_data_path', None)
                args.structure_file = struct_path
            else:
                args.structure_file = None
        
        # Validate input files after resolving from config
        if not args.data_file or not os.path.exists(args.data_file):
            logger.error(f"Data file not found or not provided. Provide via --data-file or in config. Got: {args.data_file}")
            sys.exit(1)
        if args.structure_file and not os.path.exists(args.structure_file):
            logger.error(f"Structure file not found: {args.structure_file}")
            sys.exit(1)
        
        # Create output directories based on resolved configuration
        os.makedirs(config.experiment.output_dir, exist_ok=True)
        os.makedirs(config.experiment.log_dir, exist_ok=True)
        os.makedirs(config.experiment.checkpoint_dir, exist_ok=True)
        os.makedirs(config.experiment.model_save_dir, exist_ok=True)
        
        # Save updated config
        config_save_path = os.path.join(config.experiment.output_dir, "training_config.json")
        config.save_to_file(config_save_path)
        logger.info(f"Configuration saved to: {config_save_path}")
        
        # Initialize trainer
        trainer = ProgressiveTrainer(config_save_path)
        setup_model_paths(trainer, args)
        
        # Determine which stages to run
        if args.stage is not None:
            # Single stage
            stages = [args.stage]
            logger.info(f"Running single stage: {args.stage}")
        else:
            # Multiple stages from --stages argument
            try:
                stages = [int(s.strip()) for s in args.stages.split(',')]
                stages = [s for s in stages if s in [1, 2]]  # Filter valid stages
                if not stages:
                    stages = [1, 2]  # Default to both stages
            except:
                stages = [1, 2]  # Default to both stages
            
            logger.info(f"Running stages: {stages}")
        
        # Run training
        if args.test_only:
            logger.info("Running test-only evaluation...")
            # Evaluate using Stage 2 (preferred) or Stage 1 model
            trainer.setup_data(args.data_file, args.structure_file)
            results = {"final_evaluation": trainer._final_evaluation()}
        else:
            # Normal training
            if len(stages) == 1:
                # Setup data for single stage
                trainer.setup_data_for_stage(stages[0])
                results = run_single_stage(trainer, stages[0], args)
            else:
                results = run_multiple_stages(trainer, stages, args)
        
        # Log final results
        logger.info("Training completed successfully!")
        logger.info("Final Results:")
        for key, value in results.items():
            if isinstance(value, dict):
                logger.info(f"  {key}:")
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (int, float)):
                        logger.info(f"    {subkey}: {subvalue:.4f}")
                    else:
                        logger.info(f"    {subkey}: {subvalue}")
            else:
                logger.info(f"  {key}: {value}")
        
        # Save final results
        results_path = os.path.join(config.experiment.output_dir, "training_results.json")
        with open(results_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy(obj):
                if hasattr(obj, 'item'):
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(v) for v in obj]
                else:
                    return obj
            
            json.dump(convert_numpy(results), f, indent=2)
        
        logger.info(f"Results saved to: {results_path}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
