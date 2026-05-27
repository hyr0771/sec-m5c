"""
Integrated training framework for the three-stage RNA sequence analysis model.
Combines training logic, loss computation, evaluation metrics, and progressive training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, Any, List
import numpy as np
from tqdm import tqdm
import time
import os
import math
import json
from collections import defaultdict

from config import get_config
from logger import get_logger
from data_pipeline import DataProcessor
from models import SequencePretrainingModel, StructureEnhancedModel


class ProgressiveTrainer:
    """
    Progressive trainer for the three-stage RNA sequence analysis pipeline.
    Handles training orchestration, evaluation, and model management.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config(config_path)
        self.logger = get_logger(
            self.config.experiment.log_dir,
            "INFO"
        )
        
        # Training state
        self.current_stage = None
        self.current_epoch = 0
        self.global_step = 0
        self.best_metrics = {}
        
        # Models
        self.sequence_model = None
        self.structure_model = None
        
        # Model paths for stage control
        self.stage1_model_path = None
        self.stage2_model_path = None
        
        # Data
        self.data_processor = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        self.loss_tracker = defaultdict(list)
        
        # Create output directories
        self.config.create_directories()
    
    def setup_data(self, sequence_file: str, structure_file: Optional[str] = None) -> None:
        """Setup data processing and loaders"""
        self.logger.info("Setting up data processing...")
        
        self.data_processor = DataProcessor()
        
        # Prepare datasets
        train_dataset, val_dataset, test_dataset = self.data_processor.prepare_datasets(
            sequence_file, structure_file
        )
        
        # Create data loaders
        self.train_loader, self.val_loader, self.test_loader = self.data_processor.create_dataloaders(
            train_dataset, val_dataset, test_dataset
        )
        
        # Log data statistics
        stats = self.data_processor.get_stats()
        self.logger.info(f"Data loaded successfully:")
        for key, value in stats.items():
            self.logger.info(f"  {key}: {value}")
    
    def setup_data_for_stage(self, stage: int) -> None:
        """Setup data for specific training stage"""
        if stage == 1:
            # Stage 1: Use stage1_data_path, no structure data
            sequence_file = self.config.data.stage1_data_path
            structure_file = None
            self.logger.info(f"Setting up Stage 1 data: {sequence_file}")
        elif stage == 2:
            # Stage 2: Use stage2_data_path with structure data
            sequence_file = self.config.data.stage2_data_path
            structure_file = self.config.data.structure_data_path
            self.logger.info(f"Setting up Stage 2 data: {sequence_file} + {structure_file}")
        else:
            raise ValueError(f"Unsupported stage: {stage}")
        
        self.setup_data(sequence_file, structure_file)
    
    def setup_stage1_model(self) -> None:
        """Setup sequence pretraining model"""
        self.logger.info("Setting up Stage 1: Sequence Pretraining Model")
        
        self.sequence_model = SequencePretrainingModel()
        
        # Log model info
        num_params, model_size_mb = self.sequence_model.get_model_size()
        self.logger.log_model_info("SequencePretrainingModel", num_params, model_size_mb)
        
        # Move to device
        device = self._get_device()
        self.sequence_model = self.sequence_model.to(device)
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.sequence_model.parameters(),
            lr=self.config.training.stage1_lr,
            weight_decay=self.config.training.stage1_weight_decay
        )
        
        # Setup scheduler - Cosine Annealing with Warmup
        total_steps = len(self.train_loader) * self.config.training.stage1_epochs
        warmup_steps = self.config.training.stage1_warmup_steps
        
        # Linear warmup followed by cosine annealing
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return step / warmup_steps
            else:
                # Cosine annealing after warmup
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1.0 + math.cos(math.pi * progress))
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def setup_stage2_model(self) -> None:
        """Setup structure-enhanced model"""
        self.logger.info("Setting up Stage 2: Structure Enhancement Model")
        
        # Check if we need to load stage 1 model first
        stage1_model_path = None
        if hasattr(self, 'stage1_model_path') and self.stage1_model_path:
            stage1_model_path = self.stage1_model_path
        else:
            # Find stage 1 model
            default_stage1_path = "models/stage1_model_epoch_best.pt"
            if os.path.exists(default_stage1_path):
                stage1_model_path = default_stage1_path
                self.logger.info(f"Using default stage 1 model: {stage1_model_path}")
            else:
                raise RuntimeError("Stage 1 model not found. Please train stage 1 first or specify stage1_model path.")
        
        # If sequence_model is not initialized, we're starting fresh with stage 2
        stage1_save_path = self.config.get_model_save_path("stage1")
        if self.sequence_model is not None:
            # Save current sequence model
            self.sequence_model.save_pretrained(stage1_save_path)
            self.logger.info(f"Stage 1 model saved to {stage1_save_path}")
        else:
            # Use the existing stage 1 model directly
            stage1_save_path = stage1_model_path
            self.logger.info(f"Using existing stage 1 model: {stage1_save_path}")
        
        # Initialize stage 2 model
        self.structure_model = StructureEnhancedModel(
            sequence_model_path=stage1_save_path,
            freeze_sequence_encoder=self.config.training.stage2_freeze_sequence
        )
        
        # Log model info
        num_params, model_size_mb = self.structure_model.get_model_size()
        trainable_params, trainable_size_mb = self.structure_model.get_trainable_parameters()
        self.logger.log_model_info("StructureEnhancedModel", num_params, model_size_mb)
        self.logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_size_mb:.2f} MB)")
        
        # Move to device
        device = self._get_device()
        self.structure_model = self.structure_model.to(device)
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.structure_model.parameters(),
            lr=self.config.training.stage2_lr,
            weight_decay=self.config.training.stage2_weight_decay
        )
        
        # Setup scheduler - Cosine Annealing with Warmup
        total_steps = len(self.train_loader) * self.config.training.stage2_epochs
        warmup_steps = self.config.training.stage2_warmup_steps
        
        # Linear warmup followed by cosine annealing
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return step / warmup_steps
            else:
                # Cosine annealing after warmup
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1.0 + math.cos(math.pi * progress))
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    
    
    def train_stage1(self) -> Dict[str, float]:
        """Train sequence pretraining model"""
        self.logger.set_stage("stage1")
        self.current_stage = "stage1"
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.training.stage1_epochs):
            self.logger.set_epoch(epoch)
            self.current_epoch = epoch
            
            # Training
            train_metrics = self._train_epoch_stage1()
            
            # Validation
            val_metrics = self._validate_stage1()
            
            # Log epoch results
            self.logger.info(f"Epoch {epoch} - Train Loss: {train_metrics['loss']:.4f}, "
                           f"Val Loss: {val_metrics['loss']:.4f}, "
                           f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < best_loss:
                best_loss = val_metrics['loss']
                patience_counter = 0
                best_model_path = self.config.get_model_save_path("stage1", "best")
                self.sequence_model.save_pretrained(best_model_path)
                self.logger.info(f"New best model saved: {best_model_path}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config.experiment.early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        return {'best_loss': best_loss}
    
    def train_stage2(self) -> Dict[str, float]:
        """Train structure-enhanced model"""
        self.logger.set_stage("stage2")
        self.current_stage = "stage2"
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.training.stage2_epochs):
            self.logger.set_epoch(epoch)
            self.current_epoch = epoch
            
            # Training
            train_metrics = self._train_epoch_stage2()
            
            # Validation
            val_metrics = self._validate_stage2()
            
            # Log epoch results
            self.logger.info(f"Epoch {epoch} - Train Loss: {train_metrics['loss']:.4f}, "
                           f"Val Loss: {val_metrics['loss']:.4f}, "
                           f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < best_loss:
                best_loss = val_metrics['loss']
                patience_counter = 0
                best_model_path = self.config.get_model_save_path("stage2", "best")
                self.structure_model.save_pretrained(best_model_path)
                self.logger.info(f"New best model saved: {best_model_path}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config.experiment.early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        return {'best_loss': best_loss}
    
    
    
    def _train_epoch_stage1(self) -> Dict[str, float]:
        """Train one epoch for stage 1"""
        self.sequence_model.train()
        
        total_loss = 0.0
        total_mlm_loss = 0.0
        total_cls_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Stage 1 Epoch {self.current_epoch}")
        
        for batch in progress_bar:
            start_time = time.time()
            
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            outputs = self.sequence_model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                masked_input_ids=batch.get('masked_input_ids'),
                mlm_labels=batch.get('mlm_labels'),
                classification_labels=batch.get('labels')
            )
            
            loss = outputs['loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.sequence_model.parameters(),
                self.config.training.gradient_clip_norm
            )
            
            self.optimizer.step()
            # Update scheduler every step
            self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            if 'mlm_loss' in outputs:
                total_mlm_loss += outputs['mlm_loss'].item()
            if 'classification_loss' in outputs:
                total_cls_loss += outputs['classification_loss'].item()
            
            num_batches += 1
            self.global_step += 1
            
            # Log step time
            step_time = time.time() - start_time
            self.logger.log_step_time(step_time)
            self.logger.increment_step()
            
            # Log metrics
            if self.global_step % self.config.training.logging_steps == 0:
                self.logger.log_metric("train_loss", loss.item())
                if 'mlm_loss' in outputs:
                    self.logger.log_metric("train_mlm_loss", outputs['mlm_loss'].item())
                if 'classification_loss' in outputs:
                    self.logger.log_metric("train_cls_loss", outputs['classification_loss'].item())
                self.logger.log_metric("learning_rate", self.optimizer.param_groups[0]['lr'])
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        return {
            'loss': total_loss / num_batches,
            'mlm_loss': total_mlm_loss / num_batches,
            'cls_loss': total_cls_loss / num_batches
        }
    
    def _train_epoch_stage2(self) -> Dict[str, float]:
        """Train one epoch for stage 2"""
        self.structure_model.train()
        
        total_loss = 0.0
        total_mlm_loss = 0.0
        total_cls_loss = 0.0
        total_struct_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Stage 2 Epoch {self.current_epoch}")
        
        for batch in progress_bar:
            start_time = time.time()
            
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            outputs = self.structure_model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                structure_features=batch.get('structure_features'),
                structure_adjacency=batch.get('structure_adjacency'),
                masked_input_ids=batch.get('masked_input_ids'),
                mlm_labels=batch.get('mlm_labels'),
                classification_labels=batch.get('labels'),
                structure_labels=batch.get('structure_labels'),
            )
            
            loss = outputs['loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.structure_model.parameters() if p.requires_grad],
                self.config.training.gradient_clip_norm
            )
            
            self.optimizer.step()
            # Update scheduler every step
            self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            if 'mlm_loss' in outputs:
                total_mlm_loss += outputs['mlm_loss'].item()
            if 'classification_loss' in outputs:
                total_cls_loss += outputs['classification_loss'].item()
            if 'structure_loss' in outputs:
                total_struct_loss += outputs['structure_loss'].item()
            
            num_batches += 1
            self.global_step += 1
            
            # Log step time
            step_time = time.time() - start_time
            self.logger.log_step_time(step_time)
            self.logger.increment_step()
            
            # Log metrics
            if self.global_step % self.config.training.logging_steps == 0:
                self.logger.log_metric("train_loss", loss.item())
                if 'mlm_loss' in outputs:
                    self.logger.log_metric("train_mlm_loss", outputs['mlm_loss'].item())
                if 'classification_loss' in outputs:
                    self.logger.log_metric("train_cls_loss", outputs['classification_loss'].item())
                if 'structure_loss' in outputs:
                    self.logger.log_metric("train_struct_loss", outputs['structure_loss'].item())
                self.logger.log_metric("learning_rate", self.optimizer.param_groups[0]['lr'])
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}",
                'has_struct': outputs.get('has_structure', False)
            })
        
        return {
            'loss': total_loss / num_batches,
            'mlm_loss': total_mlm_loss / num_batches,
            'cls_loss': total_cls_loss / num_batches,
            'struct_loss': total_struct_loss / num_batches
        }
    
    
    
    def _validate_stage1(self) -> Dict[str, float]:
        """Validate stage 1 model"""
        return self._validate_model(self.sequence_model, "stage1")
    
    def _validate_stage2(self) -> Dict[str, float]:
        """Validate stage 2 model"""
        return self._validate_model(self.structure_model, "stage2")
    
    
    
    def _validate_model(self, model: nn.Module, stage: str) -> Dict[str, float]:
        """Generic model validation for stages 1 and 2"""
        model.eval()
        
        # Handle empty validation set
        if len(self.val_loader) == 0:
            return {'loss': 0.0, 'accuracy': 0.0}
        
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._move_batch_to_device(batch)
                
                if stage == "stage1":
                    outputs = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        classification_labels=batch.get('labels')
                    )
                elif stage == "stage2":
                    outputs = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        structure_features=batch.get('structure_features'),
                        structure_adjacency=batch.get('structure_adjacency'),
                        classification_labels=batch.get('labels'),
                        structure_labels=batch.get('structure_labels'),
                    )
                else:
                    raise ValueError(f"Unsupported stage for validation: {stage}")
                
                if 'loss' in outputs:
                    total_loss += outputs['loss'].item()
                
                # Compute accuracy
                if 'classification_logits' in outputs and 'labels' in batch:
                    predictions = torch.argmax(outputs['classification_logits'], dim=-1)
                    correct_predictions += (predictions == batch['labels']).sum().item()
                    total_predictions += batch['labels'].size(0)
        
        metrics = {
            'loss': total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0.0,
            'accuracy': correct_predictions / total_predictions if total_predictions > 0 else 0.0
        }
        
        return metrics
    
    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch to appropriate device"""
        device = self._get_device()
        
        moved_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved_batch[key] = value.to(device)
            else:
                moved_batch[key] = value
        
        return moved_batch
    
    def _get_device(self) -> torch.device:
        """Get training device"""
        if self.config.experiment.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            return torch.device(self.config.experiment.device)
    
    
    
    def train_single_stage(self, stage: int, sequence_file: str, structure_file: Optional[str] = None) -> Dict[str, Any]:
        """Train a single stage"""
        self.logger.info(f"Starting single stage training: Stage {stage}")
        
        # Setup data for the specific stage
        self.setup_data_for_stage(stage)
        
        results = {}
        
        if stage == 1:
            self.logger.info("=" * 50)
            self.logger.info("STAGE 1: SEQUENCE PRETRAINING")
            self.logger.info("=" * 50)
            
            self.setup_stage1_model()
            stage1_results = self.train_stage1()
            results['stage1'] = stage1_results
            
        elif stage == 2:
            self.logger.info("=" * 50)
            self.logger.info("STAGE 2: STRUCTURE ENHANCEMENT")
            self.logger.info("=" * 50)
            
            # Load stage 1 model if needed
            if self.stage1_model_path and os.path.exists(self.stage1_model_path):
                self.logger.info(f"Using stage 1 model from: {self.stage1_model_path}")
            elif os.path.exists(self.config.get_model_save_path("stage1", "best")):
                self.stage1_model_path = self.config.get_model_save_path("stage1", "best")
                self.logger.info(f"Found stage 1 model at: {self.stage1_model_path}")
            else:
                self.logger.info("Stage 1 model not found. Training stage 1 first...")
                self.setup_stage1_model()
                stage1_results = self.train_stage1()
                results['stage1'] = stage1_results
            
            self.setup_stage2_model()
            stage2_results = self.train_stage2()
            results['stage2'] = stage2_results
            
        else:
            raise ValueError(f"Invalid stage: {stage}. Must be 1 or 2.")
        
        return results
    
    def train_multiple_stages(self, stages: List[int], sequence_file: str, structure_file: Optional[str] = None) -> Dict[str, Any]:
        """Train multiple stages in sequence"""
        self.logger.info(f"Starting multi-stage training: Stages {stages}")
        
        # Setup data once
        self.setup_data(sequence_file, structure_file)
        
        results = {}
        
        for stage in sorted(stages):
            stage_results = self.train_single_stage(stage, sequence_file, structure_file)
            results.update(stage_results)
        
        # Final evaluation if stage 2 was included
        if 2 in stages:
            self.logger.info("=" * 50)
            self.logger.info("FINAL EVALUATION")
            self.logger.info("=" * 50)
            final_metrics = self._final_evaluation()
            results['final_evaluation'] = final_metrics
        
        return results
    
    def train_all_stages(self, sequence_file: str, structure_file: Optional[str] = None) -> Dict[str, Any]:
        """Train stages 1 and 2 progressively"""
        return self.train_multiple_stages([1, 2], sequence_file, structure_file)
    
    def _final_evaluation(self) -> Dict[str, Any]:
        """Perform final evaluation on test set using Stage 2 model if available, otherwise Stage 1."""
        self.logger.info("Performing final evaluation on test set...")
        device = self._get_device()
        
        model = None

        def _resolve_checkpoint(stage: str, trainer_path_attr: str) -> Optional[str]:
            """Prefer CLI/set trainer path, then epoch-best naming, then legacy *_model_best.pt."""
            p = getattr(self, trainer_path_attr, None)
            if p and os.path.exists(p):
                return p
            epoch_best = self.config.get_model_save_path(stage, "best")
            if os.path.exists(epoch_best):
                return epoch_best
            legacy = self.config.get_model_save_path(stage)
            if os.path.exists(legacy):
                return legacy
            return None

        # Prefer stage 2 (explicit path from training setup, then default save locations)
        stage2_path = _resolve_checkpoint("stage2", "stage2_model_path")
        if stage2_path:
            try:
                self.logger.info(f"Loading Stage 2 model from: {stage2_path}")
                from models import StructureEnhancedModel
                model = StructureEnhancedModel.from_pretrained(stage2_path)
            except Exception:
                model = None

        if model is None:
            stage1_path = _resolve_checkpoint("stage1", "stage1_model_path")
            if stage1_path:
                self.logger.info(f"Loading Stage 1 model from: {stage1_path}")
                from models import SequencePretrainingModel
                try:
                    model = SequencePretrainingModel.from_pretrained(stage1_path)
                except Exception:
                    model = None
            if model is None:
                self.logger.error("No trained model found for final evaluation.")
                return {}
        
        model = model.to(device)
        model.eval()
        
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in self.test_loader:
                batch = self._move_batch_to_device(batch)
                
                if isinstance(model, StructureEnhancedModel):
                    outputs = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        structure_features=batch.get('structure_features'),
                        structure_adjacency=batch.get('structure_adjacency'),
                        classification_labels=batch.get('labels'),
                        structure_labels=batch.get('structure_labels'),
                    )
                else:
                    outputs = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        classification_labels=batch.get('labels')
                    )
                
                if 'loss' in outputs:
                    total_loss += outputs['loss'].item()
                
                if 'classification_logits' in outputs and 'labels' in batch:
                    predictions = torch.argmax(outputs['classification_logits'], dim=-1)
                    correct_predictions += (predictions == batch['labels']).sum().item()
                    total_predictions += batch['labels'].size(0)
        
        final_metrics = {
            'test_loss': total_loss / len(self.test_loader) if len(self.test_loader) > 0 else 0.0,
            'test_accuracy': correct_predictions / total_predictions if total_predictions > 0 else 0.0,
        }
        
        self.logger.info("Final Test Results:")
        for key, value in final_metrics.items():
            self.logger.info(f"  {key}: {value:.4f}")
        
        return final_metrics
    
    


if __name__ == "__main__":
    # Test the training framework
    from config import Config
    
    print("Testing Training Framework...")
    
    # Create a minimal config for testing
    config = Config()
    config.training.stage1_epochs = 2
    config.training.stage2_epochs = 2
    config.data.batch_size = 2  # Smaller batch size
    config.data.max_sequence_length = 256  # Shorter sequences
    config.data.num_workers = 0
    config.training.logging_steps = 5
    config.experiment.early_stopping_patience = 10
    config.experiment.device = "cpu"  # Force CPU for testing
    
    # Smaller model configuration
    config.model.hidden_size = 128
    config.model.num_hidden_layers = 4
    config.model.num_attention_heads = 4
    config.model.intermediate_size = 256
    
    # Save config and initialize trainer
    config.save_to_file("test_config.json")
    trainer = ProgressiveTrainer("test_config.json")
    
    # Test data setup (without structure data to avoid collate issues)
    print("Testing data setup...")
    trainer.setup_data("dummy.fasta")  # No structure file
    print("Data setup completed!")
    
    # Test stage 1 setup
    print("Testing stage 1 setup...")
    trainer.setup_stage1_model()
    print("Stage 1 setup completed!")
    
    # Test one training step
    print("Testing training step...")
    trainer.sequence_model.train()
    
    for batch in trainer.train_loader:
        batch = trainer._move_batch_to_device(batch)
        outputs = trainer.sequence_model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            masked_input_ids=batch.get('masked_input_ids'),
            mlm_labels=batch.get('mlm_labels'),
            classification_labels=batch.get('labels')
        )
        
        print(f"Training step successful! Loss: {outputs['loss'].item():.4f}")
        break
    
    print("Training framework test completed!")
