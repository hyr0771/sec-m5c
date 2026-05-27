"""
Integrated models for the three-stage RNA sequence analysis pipeline.
Contains all model implementations: sequence, structure, and fusion models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import Dict, Optional, Tuple, Any, Union
import math

from config import get_config
from components import (
    SequencePooler, ClassificationHead, MLMHead,
    StructureEncoder, CrossAttentionFusion, TransformerBlock
)


# ================================
# STAGE 1: SEQUENCE PRETRAINING MODEL
# ================================

class SequencePretrainingModel(nn.Module):
    """
    Stage 1: RNA sequence pretraining model with BERT-like architecture.
    Supports masked language modeling and sequence classification.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        super().__init__()
        self.config = get_config(config_path)
        
        # Embeddings
        self.token_embeddings = nn.Embedding(
            self.config.model.seq_vocab_size, 
            self.config.model.hidden_size
        )
        self.position_embeddings = nn.Embedding(
            self.config.model.seq_max_position_embeddings, 
            self.config.model.hidden_size
        )
        self.layer_norm = nn.LayerNorm(self.config.model.hidden_size)
        self.dropout = nn.Dropout(self.config.model.dropout_prob)
        
        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=self.config.model.hidden_size,
            nhead=self.config.model.num_attention_heads,
            dim_feedforward=self.config.model.intermediate_size,
            dropout=self.config.model.dropout_prob,
            batch_first=True
        )
        self.encoder = TransformerEncoder(encoder_layer, self.config.model.num_hidden_layers)
        
        # Task-specific heads
        self.pooler = SequencePooler(self.config.model.hidden_size, pooling_type="cls")
        self.mlm_head = MLMHead(self.config.model.hidden_size, self.config.model.seq_vocab_size)
        self.classification_head = ClassificationHead(
            self.config.model.hidden_size, 2, self.config.model.dropout_prob
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                masked_input_ids: Optional[torch.Tensor] = None,
                mlm_labels: Optional[torch.Tensor] = None,
                classification_labels: Optional[torch.Tensor] = None,
                return_dict: bool = True) -> Dict[str, torch.Tensor]:
        
        batch_size, seq_len = input_ids.shape
        
        # Use masked inputs for MLM if provided
        actual_input_ids = masked_input_ids if masked_input_ids is not None else input_ids
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(actual_input_ids)
        
        # Embeddings
        token_embeds = self.token_embeddings(actual_input_ids)
        
        # Position embeddings
        position_ids = torch.arange(seq_len, device=actual_input_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embeddings(position_ids)
        
        # Combine embeddings
        embeddings = token_embeds + position_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # Transformer encoding  
        # For PyTorch TransformerEncoder, we need to invert the mask (True = ignore)
        src_key_padding_mask = (attention_mask == 0)
        hidden_states = self.encoder(embeddings, src_key_padding_mask=src_key_padding_mask)
        
        # Get pooled representation
        pooled_output = self.pooler(hidden_states, attention_mask)
        
        # Prepare outputs
        outputs = {
            'last_hidden_state': hidden_states,
            'pooled_output': pooled_output
        }
        
        # Compute losses and predictions
        total_loss = 0.0
        num_tasks = 0
        
        # MLM task
        if mlm_labels is not None:
            mlm_logits = self.mlm_head(hidden_states)
            outputs['mlm_logits'] = mlm_logits
            
            mlm_loss = F.cross_entropy(
                mlm_logits.view(-1, self.config.model.seq_vocab_size),
                mlm_labels.view(-1),
                ignore_index=-100
            )
            outputs['mlm_loss'] = mlm_loss
            total_loss += self.config.training.stage1_mlm_weight * mlm_loss
            num_tasks += 1
        
        # Classification task
        if classification_labels is not None:
            classification_logits = self.classification_head(pooled_output)
            outputs['classification_logits'] = classification_logits
            
            classification_loss = F.cross_entropy(classification_logits, classification_labels)
            outputs['classification_loss'] = classification_loss
            total_loss += self.config.training.stage1_classification_weight * classification_loss
            num_tasks += 1
        
        # Total loss - do not average by number of tasks as weights are already normalized
        if num_tasks > 0:
            outputs['loss'] = total_loss
        
        return outputs
    
    def encode_sequence(self, input_ids: torch.Tensor, 
                       attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode sequences and return hidden states"""
        with torch.no_grad():
            outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            return outputs['last_hidden_state']
    
    def get_sequence_embedding(self, input_ids: torch.Tensor, 
                              attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get pooled sequence embedding"""
        with torch.no_grad():
            outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            return outputs['pooled_output']
    
    def classify_sequence(self, input_ids: torch.Tensor, 
                         attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Classify sequences"""
        with torch.no_grad():
            outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            pooled_output = outputs['pooled_output']
            logits = self.classification_head(pooled_output)
            return F.softmax(logits, dim=-1)
    
    def save_pretrained(self, save_path: str) -> None:
        """Save model state dict"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, save_path)
    
    @classmethod
    def from_pretrained(cls, load_path: str) -> 'SequencePretrainingModel':
        """Load model from saved state dict"""
        checkpoint = torch.load(load_path, map_location='cpu', weights_only=False)
        model = cls()
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def get_model_size(self) -> Tuple[int, float]:
        """Get model size in parameters and MB"""
        num_params = sum(p.numel() for p in self.parameters())
        model_size_mb = num_params * 4 / (1024 * 1024)
        return num_params, model_size_mb


# ================================
# STAGE 2: STRUCTURE ENHANCED MODEL
# ================================

class StructureEnhancedModel(nn.Module):
    """
    Stage 2: Structure enhancement model that combines sequence and structure information.
    Uses pretrained sequence encoder and adds structure processing capabilities.
    """
    
    def __init__(self, sequence_model_path: Optional[str] = None, 
                 config_path: Optional[str] = None, freeze_sequence_encoder: bool = False):
        super().__init__()
        self.config = get_config(config_path)
        self.freeze_sequence_encoder = freeze_sequence_encoder
        
        # Load pretrained sequence model
        if sequence_model_path:
            self.sequence_encoder = SequencePretrainingModel.from_pretrained(sequence_model_path)
        else:
            self.sequence_encoder = SequencePretrainingModel(config_path)
        
        # Freeze sequence encoder if specified
        if freeze_sequence_encoder:
            for param in self.sequence_encoder.parameters():
                param.requires_grad = False
        
        # Structure encoder
        self.structure_encoder = StructureEncoder(
            node_feature_dim=self.config.data.structure_feature_dim,
            hidden_dim=self.config.model.structure_hidden_size,
            num_layers=self.config.model.structure_num_layers,
            dropout=self.config.model.dropout_prob
        )
        
        # Cross-attention fusion
        self.fusion_layer = CrossAttentionFusion(
            seq_dim=self.config.model.hidden_size,
            struct_dim=self.config.model.structure_hidden_size,
            hidden_dim=self.config.model.fusion_hidden_size,
            num_heads=self.config.model.fusion_num_heads,
            dropout=self.config.model.dropout_prob
        )
        
        # Additional transformer layers for enhanced representation
        self.enhancement_layers = nn.ModuleList([
            TransformerBlock(
                d_model=self.config.model.fusion_hidden_size,
                num_heads=self.config.model.fusion_num_heads,
                d_ff=self.config.model.fusion_hidden_size * 2,
                dropout=self.config.model.dropout_prob
            )
            for _ in range(2)  # 2 additional layers for fusion enhancement
        ])
        
        # Task-specific heads
        self.pooler = SequencePooler(self.config.model.fusion_hidden_size, pooling_type="cls")
        self.mlm_head = MLMHead(self.config.model.fusion_hidden_size, self.config.model.seq_vocab_size)
        self.classification_head = ClassificationHead(
            self.config.model.fusion_hidden_size, 2, self.config.model.dropout_prob
        )
        
        # Structure-specific prediction head
        self.structure_prediction_head = nn.Sequential(
            nn.Linear(self.config.model.fusion_hidden_size, self.config.model.structure_hidden_size),
            nn.ReLU(),
            nn.Dropout(self.config.model.dropout_prob),
            nn.Linear(self.config.model.structure_hidden_size, self.config.data.structure_feature_dim)
        )
        
        # Define fixed fallback projection once to avoid recreating per-batch
        self.fallback_projection = nn.Linear(
            self.config.model.hidden_size,
            self.config.model.fusion_hidden_size
        )
        
        # Initialize new components
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights for new components"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                structure_features: Optional[torch.Tensor] = None,
                structure_adjacency: Optional[torch.Tensor] = None,
                masked_input_ids: Optional[torch.Tensor] = None,
                mlm_labels: Optional[torch.Tensor] = None,
                classification_labels: Optional[torch.Tensor] = None,
                structure_labels: Optional[torch.Tensor] = None,
                return_dict: bool = True) -> Dict[str, torch.Tensor]:
        
        batch_size = input_ids.shape[0]
        
        # Get sequence representations from pretrained encoder
        if self.freeze_sequence_encoder:
            with torch.no_grad():
                sequence_outputs = self.sequence_encoder(
                    input_ids=masked_input_ids if masked_input_ids is not None else input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
        else:
            sequence_outputs = self.sequence_encoder(
                input_ids=masked_input_ids if masked_input_ids is not None else input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
        
        sequence_hidden_states = sequence_outputs['last_hidden_state']
        
        # Process structure information if available
        if (structure_features is not None and structure_adjacency is not None):
            # Encode structure
            # Ensure batch sizes align; otherwise, skip fusion safely
            if structure_features.size(0) != batch_size or structure_adjacency.size(0) != batch_size:
                structure_hidden_states = None
            else:
                structure_hidden_states = self.structure_encoder(
                    structure_features, structure_adjacency
                )
            
            # Fuse sequence and structure representations
            if structure_hidden_states is not None:
                fused_states = self.fusion_layer(
                    sequence_hidden_states, 
                    structure_hidden_states,
                    seq_mask=attention_mask
                )
            else:
                # Fallback to sequence-only path if structure batch mismatched
                fused_states = self.fallback_projection(sequence_hidden_states)
            
            # Apply enhancement layers
            for layer in self.enhancement_layers:
                fused_states = layer(fused_states, attention_mask.unsqueeze(1).unsqueeze(2))
            
            final_hidden_states = fused_states
            has_structure = structure_hidden_states is not None
            
        else:
            # No structure available, use sequence-only representations
            # Project sequence representations to fusion dimension
            final_hidden_states = self.fallback_projection(sequence_hidden_states)
            
            # Apply enhancement layers
            for layer in self.enhancement_layers:
                final_hidden_states = layer(final_hidden_states, attention_mask.unsqueeze(1).unsqueeze(2))
            
            has_structure = False
        
        # Prepare outputs
        outputs = {
            'last_hidden_state': final_hidden_states,
            'sequence_hidden_state': sequence_hidden_states,
            'has_structure': has_structure
        }
        
        if has_structure:
            outputs['structure_hidden_state'] = structure_hidden_states
        
        # Get pooled representation
        pooled_output = self.pooler(final_hidden_states, attention_mask)
        outputs['pooled_output'] = pooled_output
        
        # Compute losses and predictions
        total_loss = 0.0
        num_tasks = 0
        
        # MLM task
        if mlm_labels is not None:
            mlm_logits = self.mlm_head(final_hidden_states)
            outputs['mlm_logits'] = mlm_logits
            
            mlm_loss = F.cross_entropy(
                mlm_logits.view(-1, self.config.model.seq_vocab_size),
                mlm_labels.view(-1),
                ignore_index=-100
            )
            outputs['mlm_loss'] = mlm_loss
            total_loss += self.config.training.stage2_mlm_weight * mlm_loss
            num_tasks += 1
        
        # Classification task
        if classification_labels is not None:
            classification_logits = self.classification_head(pooled_output)
            outputs['classification_logits'] = classification_logits
            
            classification_loss = F.cross_entropy(classification_logits, classification_labels)
            outputs['classification_loss'] = classification_loss
            total_loss += self.config.training.stage2_classification_weight * classification_loss
            num_tasks += 1
        
        # Structure reconstruction: L_struct = (1/n) sum_i || phi(H_final,i) - H_g,i^(0) ||^2
        if structure_labels is not None and has_structure:
            struct_predictions = self.structure_prediction_head(final_hidden_states)
            # Skip [CLS] at index 0; H_g^(0) is defined per nucleotide position
            pred = struct_predictions[:, 1:, :]
            target = structure_labels
            align_len = min(pred.size(1), target.size(1))
            if align_len > 0:
                aligned_pred = pred[:, :align_len, :]
                aligned_target = target[:, :align_len, :]
                outputs['structure_predictions'] = aligned_pred
                structure_loss = F.mse_loss(aligned_pred, aligned_target)
                outputs['structure_loss'] = structure_loss
                total_loss += self.config.training.stage2_structure_weight * structure_loss
                num_tasks += 1
        
        # Total loss - do not average by number of tasks as weights are already normalized
        if num_tasks > 0:
            outputs['loss'] = total_loss
        
        return outputs
    
    def save_pretrained(self, save_path: str) -> None:
        """Save model state dict"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'freeze_sequence_encoder': self.freeze_sequence_encoder
        }, save_path)
    
    @classmethod
    def from_pretrained(cls, load_path: str) -> 'StructureEnhancedModel':
        """Load model from saved state dict"""
        checkpoint = torch.load(load_path, map_location='cpu', weights_only=False)
        
        model = cls(
            sequence_model_path=None,
            freeze_sequence_encoder=checkpoint.get('freeze_sequence_encoder', False)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def get_model_size(self) -> Tuple[int, float]:
        """Get model size in parameters and MB"""
        num_params = sum(p.numel() for p in self.parameters())
        model_size_mb = num_params * 4 / (1024 * 1024)
        return num_params, model_size_mb
    
    def get_trainable_parameters(self) -> Tuple[int, float]:
        """Get number of trainable parameters"""
        num_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        trainable_size_mb = num_trainable * 4 / (1024 * 1024)
        return num_trainable, trainable_size_mb


# [Stage 3 AdaptiveFusionModel removed]
