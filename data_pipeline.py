"""
Integrated data processing pipeline for RNA sequence analysis.
Combines data loading, preprocessing, tokenization, and batch creation.
"""

import os
import re
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional, Any, Union
from pathlib import Path
import random
from collections import defaultdict
from dataclasses import dataclass
import pickle
import pandas as pd

from config import get_config


@dataclass
class SequenceData:
    """Container for sequence data"""
    id: str
    sequence: str
    label: Optional[str] = None
    structure: Optional[Any] = None
    metadata: Optional[Dict] = None


@dataclass
class StructureData:
    """Container for structure data (2D features and adjacency only)"""
    id: str
    features: np.ndarray     # Shape: (num_residues, feature_dim)
    adjacency: np.ndarray    # Shape: (num_residues, num_residues)
    sequence_id: str


class RNATokenizer:
    """Tokenizer for RNA sequences with special tokens"""
    
    def __init__(self, vocab_size: int = 8):
        # Base vocabulary for RNA
        self.base_vocab = ['A', 'U', 'G', 'C', 'N']
        
        # Special tokens
        self.special_tokens = ['[PAD]', '[MASK]', '[CLS]']
        
        # Complete vocabulary
        self.vocab = self.base_vocab + self.special_tokens
        
        # Token mappings
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}
        
        # Special token IDs
        self.pad_token_id = self.token_to_id['[PAD]']
        self.mask_token_id = self.token_to_id['[MASK]']
        self.cls_token_id = self.token_to_id['[CLS]']
        
        self.vocab_size = len(self.vocab)
    
    def encode(self, sequence: str, max_length: Optional[int] = None, 
               add_special_tokens: bool = True) -> List[int]:
        """Encode RNA sequence to token IDs"""
        # Clean and uppercase sequence
        sequence = sequence.upper().strip()
        # Convert T to U for RNA, then replace unknown bases with N
        sequence = sequence.replace('T', 'U')  
        sequence = re.sub(r'[^AUGCN]', 'N', sequence)  # Replace unknown bases with N
        
        # Tokenize
        tokens = list(sequence)
        
        # Add special tokens
        if add_special_tokens:
            tokens = ['[CLS]'] + tokens
        
        # Convert to IDs
        token_ids = [self.token_to_id.get(token, self.token_to_id['N']) for token in tokens]
        
        # Truncate or pad
        if max_length:
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            else:
                token_ids.extend([self.pad_token_id] * (max_length - len(token_ids)))
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to sequence"""
        tokens = [self.id_to_token[token_id] for token_id in token_ids]
        
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in self.special_tokens]
        
        return ''.join(tokens)
    
    def mask_sequence(self, token_ids: List[int], mask_prob: float = 0.15) -> Tuple[List[int], List[int]]:
        """Create masked sequence for MLM task"""
        masked_ids = token_ids.copy()
        labels = [-100] * len(token_ids)  # -100 is ignore index for loss
        
        for i, token_id in enumerate(token_ids):
            # Skip special tokens
            if token_id in [self.pad_token_id, self.cls_token_id]:
                continue
            
            if random.random() < mask_prob:
                labels[i] = token_id  # Original token for loss calculation
                
                # 80% mask, 10% random, 10% unchanged
                rand = random.random()
                if rand < 0.8:
                    masked_ids[i] = self.mask_token_id
                elif rand < 0.9:
                    # Random token from base vocabulary (excluding special tokens)
                    base_token_ids = [self.token_to_id[t] for t in self.base_vocab]
                    masked_ids[i] = random.choice(base_token_ids)
                # else: keep original token (10%)
        
        return masked_ids, labels


class SequenceDataset(Dataset):
    """Dataset for RNA sequences with optional structure data"""
    
    def __init__(self, sequences: List[SequenceData], tokenizer: RNATokenizer, 
                 max_length: int = 48, include_structure: bool = False,
                 mlm_probability: float = 0.15):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_structure = include_structure
        self.mlm_probability = mlm_probability
        
        # Index sequences by ID for quick lookup
        self.seq_index = {seq.id: seq for seq in sequences}
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq_data = self.sequences[idx]
        
        # Tokenize sequence
        input_ids = self.tokenizer.encode(
            seq_data.sequence, 
            max_length=self.max_length,
            add_special_tokens=True
        )
        
        # Create attention mask
        attention_mask = [1 if token_id != self.tokenizer.pad_token_id else 0 
                         for token_id in input_ids]
        
        # Prepare output
        output = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'sequence_id': seq_data.id
        }
        
        # Add MLM data if needed
        if self.mlm_probability > 0:
            masked_ids, mlm_labels = self.tokenizer.mask_sequence(
                input_ids, self.mlm_probability
            )
            output['masked_input_ids'] = torch.tensor(masked_ids, dtype=torch.long)
            output['mlm_labels'] = torch.tensor(mlm_labels, dtype=torch.long)
        
        # Add classification label if available
        if seq_data.label is not None:
            output['labels'] = torch.tensor(int(seq_data.label), dtype=torch.long)
        
        # Add structure data if needed and available
        if self.include_structure and seq_data.structure is not None:
            struct_data = seq_data.structure
            struct_features = torch.tensor(struct_data.features, dtype=torch.float)
            output.update({
                'structure_features': struct_features,
                'structure_adjacency': torch.tensor(struct_data.adjacency, dtype=torch.float),
                # H_g^(0): ground-truth structural features for L_struct reconstruction
                'structure_labels': struct_features.clone(),
            })
        
        return output


class DataProcessor:
    """Main data processing class"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config(config_path)
        self.tokenizer = RNATokenizer(vocab_size=self.config.data.vocab_size)
        # Seed all RNGs for reproducibility
        try:
            seed = int(getattr(self.config.experiment, 'seed', 42))
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
        
        # Data storage
        self.sequences = []
        self.structures = {}  # sequence_id -> StructureData
        
        # Statistics
        self.stats = {
            'num_sequences': 0,
            'num_structures': 0,
            'sequence_lengths': [],
            'label_distribution': defaultdict(int)
        }
    
    def load_fasta_file(self, file_path: str) -> List[SequenceData]:
        """Load sequences from FASTA file"""
        sequences = []
        current_id = None
        current_seq = []
        
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('>'):
                        # Save previous sequence
                        if current_id and current_seq:
                            sequences.append(SequenceData(
                                id=current_id,
                                sequence=''.join(current_seq)
                            ))
                        
                        # Start new sequence
                        current_id = line[1:].split()[0]  # Remove '>' and take first part
                        current_seq = []
                    elif line:
                        current_seq.append(line)
                
                # Save last sequence
                if current_id and current_seq:
                    sequences.append(SequenceData(
                        id=current_id,
                        sequence=''.join(current_seq)
                    ))
        
        except FileNotFoundError:
            print(f"Warning: FASTA file {file_path} not found. Using synthetic data.")
            # Generate synthetic sequences for testing
            sequences = self._generate_synthetic_sequences(100)
        
        # Filter by length
        filtered_sequences = []
        for seq in sequences:
            seq_len = len(seq.sequence)
            if (self.config.data.min_sequence_length <= seq_len <= 
                self.config.data.max_sequence_length):
                filtered_sequences.append(seq)
                self.stats['sequence_lengths'].append(seq_len)
        
        self.stats['num_sequences'] = len(filtered_sequences)
        print(f"Loaded {len(filtered_sequences)} sequences from {file_path}")
        
        return filtered_sequences
    
    def load_xlsx_file(self, file_path: str, sequence_column: str = 'sequence', 
                       label_column: str = 'label') -> List[SequenceData]:
        """Load sequences from Excel (xlsx) file"""
        sequences = []
        
        try:
            # Read Excel file
            df = pd.read_excel(file_path)
            
            # Check if required columns exist
            if sequence_column not in df.columns:
                raise ValueError(f"Column '{sequence_column}' not found in Excel file. Available columns: {list(df.columns)}")
            
            # Process each row
            for idx, row in df.iterrows():
                sequence = str(row[sequence_column]).upper().strip()
                
                # Extract label if column exists
                label = None
                if label_column in df.columns and pd.notna(row[label_column]):
                    label = row[label_column]
                
                # Create sequence ID
                seq_id = f"seq_{idx:06d}"
                if label is not None:
                    seq_id += f"_label_{label}"
                
                # Sanitize sequence: replace non-ATGCU characters with N (tokenizer will handle T->U)
                sanitized = re.sub(r'[^ATGCU]', 'N', sequence)
                sequences.append(SequenceData(
                    id=seq_id,
                    sequence=sanitized,
                    label=label
                ))
        
        except FileNotFoundError:
            print(f"Error: Excel file {file_path} not found.")
            return []
        except Exception as e:
            print(f"Error reading Excel file {file_path}: {str(e)}")
            return []
        
        # Filter by length
        filtered_sequences = []
        for seq in sequences:
            seq_len = len(seq.sequence)
            if (self.config.data.min_sequence_length <= seq_len <= 
                self.config.data.max_sequence_length):
                filtered_sequences.append(seq)
                self.stats['sequence_lengths'].append(seq_len)
                if seq.label is not None:
                    self.stats['label_distribution'][seq.label] += 1
        
        self.stats['num_sequences'] = len(filtered_sequences)
        print(f"Loaded {len(filtered_sequences)} sequences from {file_path}")
        
        return filtered_sequences
    
    def load_structure_file(self, file_path: str) -> Dict[str, StructureData]:
        """Load structure data from file (Excel with dot-bracket notation)"""
        structures = {}
        
        try:
            import os
            expanded_path = os.path.expanduser(file_path)
            
            if not os.path.exists(expanded_path):
                print(f"Warning: Structure file {expanded_path} not found. Using synthetic structures.")
                structures = self._generate_synthetic_structures(50)
                self.stats['num_structures'] = len(structures)
                return structures
            
            print(f"Loading structure data from {expanded_path}")
            
            # Load Excel file
            if expanded_path.endswith('.xlsx') or expanded_path.endswith('.xls'):
                df = pd.read_excel(expanded_path)
                
                # Check if required columns exist
                if 'structure' not in df.columns:
                    print(f"Warning: No 'structure' column found in {expanded_path}. Using synthetic structures.")
                    structures = self._generate_synthetic_structures(50)
                else:
                    # Process each row
                    for idx, row in df.iterrows():
                        # Generate ID to match sequence loading format
                        seq_id = f"seq_{idx:06d}"
                        if 'label' in df.columns and pd.notna(row['label']):
                            seq_id += f"_label_{row['label']}"
                        
                        structure_str = row['structure']
                        if pd.isna(structure_str) or structure_str == '':
                            continue
                            
                        # Convert dot-bracket notation to structure data
                        structure_data = self._parse_dot_bracket_structure(seq_id, structure_str)
                        if structure_data:
                            structures[seq_id] = structure_data
                    
                    print(f"Successfully loaded {len(structures)} structures from Excel file")
            else:
                print(f"Warning: Unsupported file format for {expanded_path}. Using synthetic structures.")
                structures = self._generate_synthetic_structures(50)
        
        except Exception as e:
            print(f"Error loading structure file {file_path}: {e}")
            print("Using synthetic structures instead.")
            structures = self._generate_synthetic_structures(50)
        
        self.stats['num_structures'] = len(structures)
        print(f"Total structures available: {len(structures)}")
        
        return structures
    
    def _parse_dot_bracket_structure(self, seq_id: str, structure_str: str) -> Optional[StructureData]:
        """Parse dot-bracket notation into StructureData"""
        try:
            length = len(structure_str)
            
            # Create adjacency matrix based on base pairing
            adjacency = np.zeros((length, length))
            
            # Parse base pairs from dot-bracket notation
            stack = []
            for i, char in enumerate(structure_str):
                if char == '(':
                    stack.append(i)
                elif char == ')' and stack:
                    j = stack.pop()
                    # Mark base pair in adjacency matrix
                    adjacency[i, j] = 1
                    adjacency[j, i] = 1
            
            # Create features to match expected feature dimension (128)
            feature_dim = 128  # Match config.data.structure_feature_dim
            features = np.zeros((length, feature_dim))
            
            # Basic structural features in first 4 dimensions
            for i, char in enumerate(structure_str):
                if char == '(':
                    features[i, 0] = 1  # Opening bracket
                elif char == ')':
                    features[i, 1] = 1  # Closing bracket
                elif char == '.':
                    features[i, 2] = 1  # Unpaired
                else:
                    features[i, 3] = 1  # Other
            
            # Fill remaining dimensions with structural context features
            # This is a simplified approach - in practice you'd use more sophisticated features
            for i in range(length):
                # Position encoding (normalized)
                features[i, 4] = i / length
                features[i, 5] = (length - i) / length
                
                # Local structure context (window around current position)
                window_size = 5
                start_idx = max(0, i - window_size)
                end_idx = min(length, i + window_size + 1)
                
                # Count different structure types in local window
                window_str = structure_str[start_idx:end_idx]
                features[i, 6] = window_str.count('(') / len(window_str)
                features[i, 7] = window_str.count(')') / len(window_str)
                features[i, 8] = window_str.count('.') / len(window_str)
                
                # Random features for remaining dimensions (placeholder)
                # In practice, these would be meaningful structural features
                if feature_dim > 9:
                    features[i, 9:] = np.random.normal(0, 0.1, feature_dim - 9)
            
            return StructureData(
                id=seq_id,
                features=features,
                adjacency=adjacency,
                sequence_id=seq_id
            )
            
        except Exception as e:
            print(f"Error parsing structure for {seq_id}: {e}")
            return None
    
    def _generate_synthetic_sequences(self, num_sequences: int) -> List[SequenceData]:
        """Generate synthetic RNA sequences for testing"""
        sequences = []
        bases = ['A', 'U', 'G', 'C']
        
        for i in range(num_sequences):
            # Random length between min and max
            length = random.randint(
                self.config.data.min_sequence_length,
                min(self.config.data.max_sequence_length, 500)  # Cap for testing
            )
            
            # Generate random sequence
            sequence = ''.join(random.choices(bases, k=length))
            
            # Random binary label
            label = str(random.randint(0, 1))
            
            sequences.append(SequenceData(
                id=f"synthetic_{i:04d}",
                sequence=sequence,
                label=label
            ))
        
        return sequences
    
    def _generate_synthetic_structures(self, num_structures: int) -> Dict[str, StructureData]:
        """Generate synthetic structure data for testing"""
        structures = {}
        
        for i in range(num_structures):
            seq_id = f"synthetic_{i:04d}"
            num_residues = random.randint(50, 200)
            
            # Random features
            features = np.random.randn(num_residues, self.config.data.structure_feature_dim)
            
            # Random adjacency matrix (symmetric)
            adjacency = np.random.rand(num_residues, num_residues)
            adjacency = (adjacency + adjacency.T) / 2  # Make symmetric
            adjacency = (adjacency > 0.8).astype(float)  # Sparse connections
            
            structures[seq_id] = StructureData(
                id=f"struct_{i:04d}",
                features=features,
                adjacency=adjacency,
                sequence_id=seq_id
            )
        
        return structures
    
    def prepare_datasets(self, sequence_file: str, structure_file: Optional[str] = None, **kwargs) -> Tuple[Dataset, Dataset, Dataset]:
        """Prepare train, validation, and test datasets"""
        # Load data using universal loader
        self.sequences = self.load_data(sequence_file, **kwargs)
        
        if structure_file:
            self.structures = self.load_structure_file(structure_file)
            # Link structures to sequences
            for seq in self.sequences:
                if seq.id in self.structures:
                    seq.structure = self.structures[seq.id]
        
        # Split data with grouping by raw sequence string to avoid duplicates across splits
        # Build groups: raw sequence string -> list[SequenceData]
        sequence_groups: Dict[str, List[SequenceData]] = defaultdict(list)
        for seq in self.sequences:
            sequence_groups[seq.sequence].append(seq)
        unique_keys = list(sequence_groups.keys())
        random.shuffle(unique_keys)
        
        n_total = len(self.sequences)
        target_train = int(n_total * self.config.data.train_ratio)
        target_val = int(n_total * self.config.data.val_ratio)
        
        train_sequences: List[SequenceData] = []
        val_sequences: List[SequenceData] = []
        test_sequences: List[SequenceData] = []
        
        count_train = 0
        count_val = 0
        
        for key in unique_keys:
            group = sequence_groups[key]
            # Assign whole group to a split to prevent cross-split duplicates
            if count_train + len(group) <= target_train:
                train_sequences.extend(group)
                count_train += len(group)
            elif count_val + len(group) <= target_val:
                val_sequences.extend(group)
                count_val += len(group)
            else:
                test_sequences.extend(group)
        
        # Create datasets
        train_dataset = SequenceDataset(
            train_sequences, 
            self.tokenizer,
            max_length=self.config.data.max_sequence_length,
            include_structure=(structure_file is not None),
            mlm_probability=0.15
        )
        
        val_dataset = SequenceDataset(
            val_sequences,
            self.tokenizer,
            max_length=self.config.data.max_sequence_length,
            include_structure=(structure_file is not None),
            mlm_probability=0.0  # No masking for validation
        )
        
        test_dataset = SequenceDataset(
            test_sequences,
            self.tokenizer,
            max_length=self.config.data.max_sequence_length,
            include_structure=(structure_file is not None),
            mlm_probability=0.0  # No masking for testing
        )
        
        print(f"Dataset splits: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def _collate_fn(self, batch):
        """Custom collate function to handle variable-length sequences"""
        keys = batch[0].keys()
        collated = {}
        
        for key in keys:
            if key == 'sequence_id':
                collated[key] = [item[key] for item in batch]
            elif key.startswith('structure_'):
                # Handle structure data that might not be present in all samples
                structure_data = [item[key] for item in batch if key in item]
                if structure_data:
                    collated[key] = torch.stack(structure_data)
            else:
                collated[key] = torch.stack([item[key] for item in batch])
        
        return collated
    
    def create_sequence_dataset(self, sequences: List[SequenceData]) -> Dataset:
        """Create a dataset from sequence data"""
        return SequenceDataset(sequences, self.tokenizer, self.config.data.max_sequence_length)
    
    def create_dataloader(self, dataset: Dataset, batch_size: Optional[int] = None, 
                         shuffle: bool = False) -> DataLoader:
        """Create a single dataloader"""
        batch_size = batch_size or self.config.data.batch_size
        
        # Ensure dataloader workers are seeded deterministically
        def _seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        generator = torch.Generator()
        generator.manual_seed(int(getattr(self.config.experiment, 'seed', 42)))
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn,
            num_workers=self.config.data.num_workers,
            pin_memory=True if self.config.experiment.device == "cuda" else False,
            worker_init_fn=_seed_worker,
            generator=generator
        )
    
    def create_dataloaders(self, train_dataset: Dataset, val_dataset: Dataset, 
                          test_dataset: Dataset) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create data loaders for training"""
        
        def collate_fn(batch):
            """Custom collate function to handle variable-length sequences"""
            keys = batch[0].keys()
            collated = {}
            
            for key in keys:
                if key == 'sequence_id':
                    collated[key] = [item[key] for item in batch]
                elif key.startswith('structure_'):
                    # Handle structure data that might not be present in all samples
                    structure_data = [item[key] for item in batch if key in item]
                    if structure_data:
                        collated[key] = torch.stack(structure_data)
                else:
                    collated[key] = torch.stack([item[key] for item in batch])
            
            return collated
        
        # Shared worker seeding and generator for all loaders
        def _seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        generator = torch.Generator()
        generator.manual_seed(int(getattr(self.config.experiment, 'seed', 42)))
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            worker_init_fn=_seed_worker,
            generator=generator
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            worker_init_fn=_seed_worker,
            generator=generator
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            worker_init_fn=_seed_worker,
            generator=generator
        )
        
        return train_loader, val_loader, test_loader
    
    def save_tokenizer(self, save_path: str) -> None:
        """Save tokenizer to file"""
        with open(save_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
    
    def load_tokenizer(self, load_path: str) -> None:
        """Load tokenizer from file"""
        with open(load_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
    
    def load_data(self, file_path: str, **kwargs) -> List[SequenceData]:
        """通用数据加载函数，自动检测文件格式
        
        支持的格式:
        - .fasta, .fa: FASTA格式
        - .csv: CSV格式 (sequence,label)
        - .xlsx, .xls: Excel格式
        
        Args:
            file_path: 数据文件路径
            **kwargs: 格式特定的参数
                - sequence_column: Excel/CSV中序列列名 (默认: 'sequence')
                - label_column: Excel/CSV中标签列名 (默认: 'label')
        
        Returns:
            List[SequenceData]: 加载的序列数据
        """
        file_path = Path(file_path)
        file_extension = file_path.suffix.lower()
        
        print(f"检测到文件格式: {file_extension}")
        
        if file_extension in ['.fasta', '.fa']:
            return self.load_fasta_file(str(file_path))
        elif file_extension == '.csv':
            # CSV文件处理
            try:
                df = pd.read_csv(file_path)
                sequence_column = kwargs.get('sequence_column', 'sequence')
                label_column = kwargs.get('label_column', 'label')
                
                sequences = []
                for idx, row in df.iterrows():
                    sequence = str(row[sequence_column]).upper().strip()
                    label = None
                    if label_column in df.columns and pd.notna(row[label_column]):
                        label = row[label_column]
                    
                    seq_id = f"seq_{idx:06d}"
                    if label is not None:
                        seq_id += f"_label_{label}"
                    
                    # Sanitize sequence: replace non-ATGCU characters with N (tokenizer will handle T->U)
                    sanitized = re.sub(r'[^ATGCU]', 'N', sequence)
                    sequences.append(SequenceData(
                        id=seq_id,
                        sequence=sanitized,
                        label=label
                    ))
                
                # Filter by length
                filtered_sequences = []
                for seq in sequences:
                    seq_len = len(seq.sequence)
                    if (self.config.data.min_sequence_length <= seq_len <= 
                        self.config.data.max_sequence_length):
                        filtered_sequences.append(seq)
                        self.stats['sequence_lengths'].append(seq_len)
                        if seq.label is not None:
                            self.stats['label_distribution'][seq.label] += 1
                
                self.stats['num_sequences'] = len(filtered_sequences)
                print(f"从CSV加载了 {len(filtered_sequences)} 条序列")
                return filtered_sequences
                
            except Exception as e:
                print(f"加载CSV文件出错: {str(e)}")
                return []
                
        elif file_extension in ['.xlsx', '.xls']:
            sequence_column = kwargs.get('sequence_column', 'sequence')
            label_column = kwargs.get('label_column', 'label')
            return self.load_xlsx_file(str(file_path), sequence_column, label_column)
        else:
            raise ValueError(f"不支持的文件格式: {file_extension}. 支持的格式: .fasta, .fa, .csv, .xlsx, .xls")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get data processing statistics"""
        stats = self.stats.copy()
        if self.stats['sequence_lengths']:
            stats.update({
                'avg_sequence_length': np.mean(self.stats['sequence_lengths']),
                'median_sequence_length': np.median(self.stats['sequence_lengths']),
                'min_sequence_length': np.min(self.stats['sequence_lengths']),
                'max_sequence_length': np.max(self.stats['sequence_lengths'])
            })
            # Remove the detailed sequence_lengths list to avoid cluttering logs
            # Only keep the summary statistics
            del stats['sequence_lengths']
        return stats
    
    def create_dataloaders(self, train_dataset, val_dataset, test_dataset):
        """Create data loaders for training, validation, and testing"""
        from torch.utils.data import DataLoader
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the data pipeline
    from config import Config
    
    # Create test configuration
    config = Config()
    config.data.min_sequence_length = 50
    config.data.max_sequence_length = 512
    config.data.batch_size = 4
    config.data.num_workers = 0  # Disable multiprocessing for testing
    
    # Initialize processor
    processor = DataProcessor()
    
    # Test tokenizer
    print("Testing tokenizer...")
    test_seq = "AUGCGAUCGAUCG"  # RNA sequence with U instead of T
    encoded = processor.tokenizer.encode(test_seq)
    decoded = processor.tokenizer.decode(encoded)
    print(f"Original: {test_seq}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    
    # Test masking
    masked, labels = processor.tokenizer.mask_sequence(encoded, mask_prob=0.3)
    print(f"Masked: {masked}")
    print(f"Labels: {labels}")
    
    # Test dataset creation (using synthetic data)
    print("\nTesting dataset creation...")
    train_ds, val_ds, test_ds = processor.prepare_datasets("dummy.fasta", "dummy.pdb")
    
    # Test data loaders
    print("\nTesting data loaders...")
    train_loader, val_loader, test_loader = processor.create_dataloaders(train_ds, val_ds, test_ds)
    
    # Test a batch
    for batch in train_loader:
        print("Batch keys:", list(batch.keys()))
        print("Input shape:", batch['input_ids'].shape)
        print("Attention mask shape:", batch['attention_mask'].shape)
        if 'mlm_labels' in batch:
            print("MLM labels shape:", batch['mlm_labels'].shape)
        break
    
    # Print statistics
    print("\nData statistics:")
    stats = processor.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("Data pipeline test completed!")
