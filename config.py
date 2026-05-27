"""
Configuration management system for RNA sequence analysis model.
Handles all configuration parameters for the three-stage training pipeline.
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path


@dataclass
class DataConfig:
    """Data processing configuration"""
    # Dataset paths
    sequence_data_path: str = "data/sequences.fasta"
    structure_data_path: str = "data/structures.pdb"
    
    # Stage-specific data paths
    stage1_data_path: str = "data/sequences.fasta"
    stage2_data_path: str = "data/sequences.fasta"
    
    # Sequence processing - optimized for actual data (sequences of length 41)
    max_sequence_length: int = 48   # Maximum allowed sequence length (optimized for memory)
    min_sequence_length: int = 10   # Minimum required sequence length
    vocab_size: int = 8  # A, T, G, C, N, [PAD], [MASK], [CLS]
    
    # Structure processing
    max_structure_nodes: int = 512
    structure_feature_dim: int = 128
    
    # Data splits
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Batch processing (optimized for RTX 4090)
    batch_size: int = 128  # 增加batch size提高GPU利用率
    num_workers: int = 8   # 增加数据加载进程数


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Common parameters
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    intermediate_size: int = 3072
    dropout_prob: float = 0.1
    
    # Sequence model (Stage 1)
    seq_vocab_size: int = 8
    seq_max_position_embeddings: int = 48   # Optimized for actual sequence length (41) + buffer
    
    # Structure model (Stage 2)
    structure_hidden_size: int = 256
    structure_num_layers: int = 6
    
    # Fusion for Stage 2 sequence-structure integration
    fusion_hidden_size: int = 512
    fusion_num_heads: int = 8


@dataclass
class TrainingConfig:
    """Training configuration for stages 1 and 2"""
    # Stage 1: Sequence pretraining (optimized for faster training)
    stage1_epochs: int = 50
    stage1_lr: float = 5e-4  # 提高学习率5倍
    stage1_warmup_steps: int = 200  # 减少warmup步数
    stage1_weight_decay: float = 0.01
    
    # Stage 2: Structure enhancement
    stage2_epochs: int = 30
    stage2_lr: float = 5e-5
    stage2_warmup_steps: int = 500
    stage2_weight_decay: float = 0.01
    stage2_freeze_sequence: bool = False  # Whether to freeze sequence encoder
    
    # Stage 3 removed

    # Common training parameters
    gradient_clip_norm: float = 1.0
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    
    # Task weights for multi-task learning - Stage 1
    stage1_mlm_weight: float = 0.7
    stage1_classification_weight: float = 0.3
    
    # Task weights for multi-task learning - Stage 2
    stage2_mlm_weight: float = 0.6
    stage2_classification_weight: float = 0.3
    stage2_structure_weight: float = 0.1
    
    # Backward compatibility (deprecated)
    mlm_weight: float = 0.7
    classification_weight: float = 0.3
    structure_weight: float = 0.1


@dataclass
class ExperimentConfig:
    """Experiment and output configuration"""
    # Experiment tracking
    experiment_name: str = "dna_sequence_analysis"
    run_id: Optional[str] = None
    
    # Output directories
    output_dir: str = "outputs"
    model_save_dir: str = "models"
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    
    # Model saving
    save_best_only: bool = True
    save_total_limit: int = 3
    
    # Evaluation
    eval_metric: str = "accuracy"
    early_stopping_patience: int = 5
    
    # Hardware
    device: str = "auto"  # auto, cpu, cuda
    mixed_precision: bool = True
    
    # Random seed
    seed: int = 42


class Config:
    """Main configuration class that combines all configuration components"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration from file or defaults"""
        self.data = DataConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.experiment = ExperimentConfig()
        
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
    
    def load_from_file(self, config_path: str) -> None:
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Update each section if present in file
        if 'data' in config_dict:
            self._update_dataclass(self.data, config_dict['data'])
        if 'model' in config_dict:
            self._update_dataclass(self.model, config_dict['model'])
        if 'training' in config_dict:
            self._update_dataclass(self.training, config_dict['training'])
        if 'experiment' in config_dict:
            self._update_dataclass(self.experiment, config_dict['experiment'])
    
    def save_to_file(self, config_path: str) -> None:
        """Save current configuration to JSON file"""
        config_dict = {
            'data': asdict(self.data),
            'model': asdict(self.model),
            'training': asdict(self.training),
            'experiment': asdict(self.experiment)
        }
        
        # Create directory if it doesn't exist
        config_dir = os.path.dirname(config_path)
        if config_dir:  # Only create directory if dirname is not empty
            os.makedirs(config_dir, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def _update_dataclass(self, dataclass_obj: Any, update_dict: Dict) -> None:
        """Update dataclass fields from dictionary"""
        for key, value in update_dict.items():
            if hasattr(dataclass_obj, key):
                setattr(dataclass_obj, key, value)
    
    def create_directories(self) -> None:
        """Create necessary directories for the experiment"""
        dirs = [
            self.experiment.output_dir,
            self.experiment.model_save_dir,
            self.experiment.log_dir,
            self.experiment.checkpoint_dir,
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def get_model_save_path(self, stage: str, epoch: Optional[int] = None) -> str:
        """Get model save path for specific stage"""
        if epoch is not None:
            filename = f"{stage}_model_epoch_{epoch}.pt"
        else:
            filename = f"{stage}_model_best.pt"
        
        return os.path.join(self.experiment.model_save_dir, filename)
    
    def get_checkpoint_path(self, stage: str, step: int) -> str:
        """Get checkpoint path for specific stage and step"""
        filename = f"{stage}_checkpoint_step_{step}.pt"
        return os.path.join(self.experiment.checkpoint_dir, filename)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        warnings = []
        
        # Validate data ratios
        total_ratio = self.data.train_ratio + self.data.val_ratio + self.data.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            errors.append(f"Data split ratios sum to {total_ratio}, should be 1.0")
        
        # Validate sequence length
        if self.data.max_sequence_length <= self.data.min_sequence_length:
            errors.append("max_sequence_length must be greater than min_sequence_length")
        
        # Validate model dimensions
        if self.model.hidden_size % self.model.num_attention_heads != 0:
            errors.append("hidden_size must be divisible by num_attention_heads")
        
        # Validate learning rates
        if self.training.stage1_lr <= 0 or self.training.stage2_lr <= 0:
            errors.append("All learning rates must be positive")
        
        # Validate structure data path (warning only, not error)
        if self.data.structure_data_path and not os.path.exists(self.data.structure_data_path):
            warnings.append(f"Structure data path does not exist: {self.data.structure_data_path}")
            warnings.append("Stage 2 training may be affected. Will fallback to sequence-only if needed.")
        
        # Log warnings
        if warnings:
            import logging
            logger = logging.getLogger(__name__)
            for warning in warnings:
                logger.warning(warning)
        
        return errors
    
    def __str__(self) -> str:
        """String representation of configuration"""
        return f"""Configuration Summary:
Data: batch_size={self.data.batch_size}, max_seq_len={self.data.max_sequence_length}
Model: hidden_size={self.model.hidden_size}, num_layers={self.model.num_hidden_layers}
Training: stage1_lr={self.training.stage1_lr}, stage2_lr={self.training.stage2_lr}
Experiment: {self.experiment.experiment_name}, device={self.experiment.device}
"""


# Global configuration instance
_global_config = None


def get_config(config_path: Optional[str] = None) -> Config:
    """Get global configuration instance"""
    global _global_config
    if _global_config is None:
        _global_config = Config(config_path)
    return _global_config


def set_config(config: Config) -> None:
    """Set global configuration instance"""
    global _global_config
    _global_config = config


if __name__ == "__main__":
    # Example usage and testing
    config = Config()
    
    # Validate configuration
    errors = config.validate()
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Configuration is valid!")
    
    # Print configuration summary
    print(config)
    
    # Save example configuration
    config.save_to_file("example_config.json")
    print("Example configuration saved to example_config.json")
