import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.preprocessing import LabelBinarizer
import dgeb
from tqdm import tqdm
import wandb
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
import os
from dataclasses import dataclass, field
from typing import List, Optional, Literal
import time
import collections
from datasets import load_dataset
import pandas as pd
from Bio.ExPASy import Enzyme


# Original architecture: Two-head model for whole EC numbers
class TwoHeadFineTuner(nn.Module):
    def __init__(self, base_model, num_classes, layer_idx=-1, dropout_rate=0.1):
        super().__init__()
        self.base = base_model
        self.layer_idx = layer_idx
        self.embed_dim = base_model.embed_dim
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Classification heads - both have the same number of output classes
        self.main_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.embed_dim // 2, num_classes)
        )
        
        self.aux_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.embed_dim // 2, num_classes)
        )
    
    def forward(self, sequences):
        # Get embeddings from base model
        embeds = self.base.encode(sequences)  # [batch_size, num_layers, embed_dim]
        
        # Ensure embeddings are PyTorch tensors
        if not isinstance(embeds, torch.Tensor):
            embeds = torch.tensor(embeds, dtype=torch.float32).to(self.main_head[0].weight.device)
            
        # Choose specific layer output (use -1 for last layer)
        layer_embeds = embeds[:, self.layer_idx, :]  # [batch_size, embed_dim]
        
        # Apply dropout
        layer_embeds = self.dropout(layer_embeds)
        
        # Get predictions from both heads
        main_logits = self.main_head(layer_embeds)
        aux_logits = self.aux_head(layer_embeds)
        
        return main_logits, aux_logits

# Hierarchical architecture: Multi-head model for EC levels
class HierarchicalFineTuner(nn.Module):
    def __init__(self, base_model, num_classes_per_level, layer_idx=-1, dropout_rate=0.1):
        super().__init__()
        self.base = base_model
        self.layer_idx = layer_idx
        self.embed_dim = base_model.embed_dim
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Create separate classification heads for each EC level
        self.main_heads = nn.ModuleList()
        self.aux_heads = nn.ModuleList()
        
        for level, num_classes in enumerate(num_classes_per_level):
            # Main task heads
            self.main_heads.append(nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.embed_dim // 2, num_classes)
            ))
            
            # Auxiliary task heads
            self.aux_heads.append(nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.embed_dim // 2, num_classes)
            ))
    
    def forward(self, sequences):
        # Get embeddings from base model
        embeds = self.base.encode(sequences)  # [batch_size, num_layers, embed_dim]
        
        # Ensure embeddings are PyTorch tensors
        if not isinstance(embeds, torch.Tensor):
            embeds = torch.tensor(embeds, dtype=torch.float32).to(self.main_heads[0][0].weight.device)
            
        # Choose specific layer output (use -1 for last layer)
        layer_embeds = embeds[:, self.layer_idx, :]  # [batch_size, embed_dim]
        
        # Apply dropout
        layer_embeds = self.dropout(layer_embeds)
        
        # Get predictions from all heads
        main_logits = [head(layer_embeds) for head in self.main_heads]
        aux_logits = [head(layer_embeds) for head in self.aux_heads]
        
        return main_logits, aux_logits