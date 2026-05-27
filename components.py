"""
Common model components for RNA sequence analysis.
Provides reusable building blocks for all model stages.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import Optional, Tuple, Dict, Any
import numpy as np

from config import get_config


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence inputs"""
    
    def __init__(self, d_model: int, max_len: int = 48, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)
        # Validate batch sizes to avoid silent broadcasting or invalid reshapes
        if key.size(0) != batch_size or value.size(0) != batch_size:
            raise RuntimeError(
                f"MultiHeadAttention batch mismatch: Q batch={batch_size}, "
                f"K batch={key.size(0)}, V batch={value.size(0)}"
            )
        
        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attn_output, attn_weights = self._attention(Q, K, V, mask)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear transformation
        output = self.w_o(attn_output)
        
        return output, attn_weights
    
    def _attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                   mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        
        return attn_output, attn_weights


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Single transformer encoder block"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class RNAEmbedding(nn.Module):
    """RNA sequence embedding layer"""
    
    def __init__(self, vocab_size: int, d_model: int, max_length: int = 48, 
                 dropout: float = 0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = PositionalEncoding(d_model, max_length, dropout)
        self.d_model = d_model
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len]
        Returns:
            embeddings: [batch_size, seq_len, d_model]
        """
        # Token embeddings
        embeddings = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        
        # Add positional encoding
        embeddings = embeddings.transpose(0, 1)  # [seq_len, batch_size, d_model]
        embeddings = self.position_encoding(embeddings)
        embeddings = embeddings.transpose(0, 1)  # [batch_size, seq_len, d_model]
        
        return embeddings


class StructureEncoder(nn.Module):
    """Graph-based encoder for protein structure data"""
    
    def __init__(self, node_feature_dim: int, hidden_dim: int, num_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.num_layers = num_layers
        
        # Node feature transformation
        self.node_proj = nn.Linear(node_feature_dim, hidden_dim)
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: [batch_size, num_nodes, feature_dim]
            adjacency: [batch_size, num_nodes, num_nodes]
        Returns:
            encoded_features: [batch_size, num_nodes, hidden_dim]
        """
        # Project node features
        x = self.node_proj(node_features)
        
        # Apply graph convolution layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x, adjacency)
        
        # Output projection
        x = self.output_proj(x)
        
        return x


class GraphConvLayer(nn.Module):
    """Graph convolution layer for structure encoding"""
    
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_nodes, in_dim]
            adjacency: [batch_size, num_nodes, num_nodes]
        """
        # Normalize adjacency matrix
        degree = adjacency.sum(dim=-1, keepdim=True) + 1e-6
        norm_adj = adjacency / degree
        
        # Apply graph convolution
        x = self.linear(x)
        x = torch.bmm(norm_adj, x)  # Aggregate neighbor features
        x = self.norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        return x


class CrossAttentionFusion(nn.Module):
    """Cross-attention mechanism for fusing sequence and structure features"""
    
    def __init__(self, seq_dim: int, struct_dim: int, hidden_dim: int, 
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.seq_proj = nn.Linear(seq_dim, hidden_dim)
        self.struct_proj = nn.Linear(struct_dim, hidden_dim)
        
        # Cross-attention layers
        self.seq_to_struct_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.struct_to_seq_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        
        # Fusion layers
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.fusion_ff = FeedForward(hidden_dim, hidden_dim * 2, dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, seq_features: torch.Tensor, struct_features: torch.Tensor,
                seq_mask: Optional[torch.Tensor] = None,
                struct_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            seq_features: [batch_size, seq_len, seq_dim]
            struct_features: [batch_size, num_nodes, struct_dim]
        Returns:
            fused_features: [batch_size, seq_len, hidden_dim]
        """
        # Project to common dimension
        seq_proj = self.seq_proj(seq_features)
        struct_proj = self.struct_proj(struct_features)
        
        # Cross-attention: sequence attends to structure
        seq_enhanced, _ = self.seq_to_struct_attn(
            seq_proj, struct_proj, struct_proj, struct_mask
        )
        
        # Residual connection
        seq_enhanced = seq_proj + self.dropout(seq_enhanced)
        seq_enhanced = self.fusion_norm(seq_enhanced)
        
        # Feed-forward
        fused = self.fusion_ff(seq_enhanced)
        fused = seq_enhanced + self.dropout(fused)
        
        return fused


class MLMHead(nn.Module):
    """Masked Language Modeling prediction head"""
    
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class ClassificationHead(nn.Module):
    """Classification prediction head"""
    
    def __init__(self, hidden_size: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class SequencePooler(nn.Module):
    """Pooling layer for sequence representations"""
    
    def __init__(self, hidden_size: int, pooling_type: str = "cls"):
        super().__init__()
        self.pooling_type = pooling_type
        if pooling_type == "attention":
            self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
        Returns:
            pooled_output: [batch_size, hidden_size]
        """
        if self.pooling_type == "cls":
            # Use [CLS] token representation
            pooled_output = hidden_states[:, 0]
        
        elif self.pooling_type == "mean":
            # Mean pooling over sequence length
            if attention_mask is not None:
                # Mask out padding tokens
                hidden_states = hidden_states * attention_mask.unsqueeze(-1)
                pooled_output = hidden_states.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            else:
                pooled_output = hidden_states.mean(dim=1)
        
        elif self.pooling_type == "max":
            # Max pooling over sequence length
            if attention_mask is not None:
                hidden_states = hidden_states.masked_fill(
                    attention_mask.unsqueeze(-1) == 0, -float('inf')
                )
            pooled_output = hidden_states.max(dim=1)[0]
        
        elif self.pooling_type == "attention":
            # Attention-based pooling
            attn_weights = self.attention(hidden_states).squeeze(-1)
            if attention_mask is not None:
                attn_weights = attn_weights.masked_fill(attention_mask == 0, -float('inf'))
            attn_weights = F.softmax(attn_weights, dim=-1)
            pooled_output = (hidden_states * attn_weights.unsqueeze(-1)).sum(dim=1)
        
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")
        
        return pooled_output


if __name__ == "__main__":
    # Test components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test RNA embedding
    print("Testing RNA embedding...")
    vocab_size = 8
    d_model = 256
    seq_len = 100
    batch_size = 4
    
    embedding = RNAEmbedding(vocab_size, d_model).to(device)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    embeddings = embedding(input_ids)
    print(f"Embedding output shape: {embeddings.shape}")
    
    # Test transformer block
    print("\nTesting transformer block...")
    transformer_block = TransformerBlock(d_model, num_heads=8, d_ff=512).to(device)
    output = transformer_block(embeddings)
    print(f"Transformer block output shape: {output.shape}")
    
    # Test structure encoder
    print("\nTesting structure encoder...")
    num_nodes = 50
    node_features = torch.randn(batch_size, num_nodes, 128).to(device)
    adjacency = torch.rand(batch_size, num_nodes, num_nodes).to(device)
    adjacency = (adjacency > 0.7).float()  # Sparse adjacency
    
    struct_encoder = StructureEncoder(128, 256).to(device)
    struct_output = struct_encoder(node_features, adjacency)
    print(f"Structure encoder output shape: {struct_output.shape}")
    
    # Test cross-attention fusion
    print("\nTesting cross-attention fusion...")
    fusion = CrossAttentionFusion(d_model, 256, 512).to(device)
    fused_output = fusion(embeddings, struct_output)
    print(f"Fusion output shape: {fused_output.shape}")
    
    # Test pooling
    print("\nTesting pooling...")
    pooler = SequencePooler(512, "attention").to(device)
    pooled = pooler(fused_output)
    print(f"Pooled output shape: {pooled.shape}")
    # Test prediction heads
    print("\nTesting prediction heads...")
    mlm_head = MLMHead(512, vocab_size).to(device)
    mlm_logits = mlm_head(fused_output)
    print(f"MLM logits shape: {mlm_logits.shape}")
    
    cls_head = ClassificationHead(512, 2).to(device)
    cls_logits = cls_head(pooled)
    print(f"Classification logits shape: {cls_logits.shape}")
    
    print("\nAll component tests passed!")
