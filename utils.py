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

# Original approach
def train_original(model, dataloader, optimizer, criterion, alpha=0.7, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model.train()
    total_loss = 0
    main_preds = []
    main_labels = []
    
    for batch in tqdm(dataloader, desc="Training"):
        # Get data
        sequences = batch['sequence']
        main_label = batch['main_label'].to(device)
        aux_label = batch['aux_label'].to(device)
        
        # Forward pass
        main_logits, aux_logits = model(sequences)
        
        # Compute combined loss
        main_loss = criterion(main_logits, main_label)
        aux_loss = criterion(aux_logits, aux_label)
        combined_loss = alpha * main_loss + (1 - alpha) * aux_loss
        
        # Backward pass and optimization
        optimizer.zero_grad()
        combined_loss.backward()
        optimizer.step()
        
        # Track statistics
        total_loss += combined_loss.item()
        
        # Store predictions for F1 score calculation
        _, main_pred = torch.max(main_logits, 1)
        main_preds.extend(main_pred.cpu().numpy())
        main_labels.extend(main_label.cpu().numpy())
    
    # Calculate F1 score for main task (macro)
    main_f1 = f1_score(main_labels, main_preds, average='macro', zero_division=0)
    
    return total_loss / len(dataloader), main_f1

# Hierarchical approach
def train_hierarchical(model, dataloader, optimizer, criterion, level_weights, alpha=0.7, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model.train()
    total_loss = 0
    
    # Create counters for correct predictions and total samples for each level
    correct_preds = [0 for _ in range(4)]
    total_samples = 0
    
    # For F1 calculation
    all_preds = [[] for _ in range(4)]
    all_labels = [[] for _ in range(4)]
    
    for batch in tqdm(dataloader, desc="Training"):
        # Get data
        sequences = batch['sequence']
        main_labels = [label.to(device) for label in batch['main_labels']]
        aux_labels = [label.to(device) for label in batch['aux_labels']]
        
        # Forward pass
        main_logits, aux_logits = model(sequences)
        
        # Compute weighted loss across all levels
        main_losses = [criterion(main_logits[i], main_labels[i]) for i in range(4)]
        aux_losses = [criterion(aux_logits[i], aux_labels[i]) for i in range(4)]
        
        # Weight the losses by EC level importance
        weighted_main_loss = sum(loss * weight for loss, weight in zip(main_losses, level_weights))
        weighted_aux_loss = sum(loss * weight for loss, weight in zip(aux_losses, level_weights))
        
        # Combined loss with alpha weighting between main and auxiliary tasks
        combined_loss = alpha * weighted_main_loss + (1 - alpha) * weighted_aux_loss
        
        # Backward pass and optimization
        optimizer.zero_grad()
        combined_loss.backward()
        optimizer.step()
        
        # Track statistics
        total_loss += combined_loss.item()
        batch_size = main_labels[0].size(0)
        total_samples += batch_size
        
        # Calculate accuracy and collect predictions for each level
        for level in range(4):
            _, level_preds = torch.max(main_logits[level], 1)
            level_correct = (level_preds == main_labels[level])
            correct_preds[level] += level_correct.sum().item()
            
            # Collect predictions for F1 calculation
            all_preds[level].extend(level_preds.cpu().numpy())
            all_labels[level].extend(main_labels[level].cpu().numpy())
    
    # Calculate average accuracy for each level
    level_accs = [correct / total_samples for correct in correct_preds]
    
    # Calculate F1 scores for each level using macro average
    level_f1_scores = [
        f1_score(all_labels[level], all_preds[level], average='macro', zero_division=0)
        for level in range(4)
    ]
    
    return total_loss / len(dataloader), level_accs, level_f1_scores

# Evaluation functions

# Original approach
def evaluate_original(model, dataloader, criterion, alpha=0.7, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model.eval()
    total_loss = 0
    main_preds = []
    main_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Get data
            sequences = batch['sequence']
            main_label = batch['main_label'].to(device)
            
            # Check if aux_label exists in this batch
            if 'aux_label' in batch:
                aux_label = batch['aux_label'].to(device)
            else:
                # For test set, use main_label as aux_label
                aux_label = main_label
            
            # Forward pass
            main_logits, aux_logits = model(sequences)
            
            # Compute loss
            main_loss = criterion(main_logits, main_label)
            
            if 'aux_label' in batch:
                aux_loss = criterion(aux_logits, aux_label)
                combined_loss = alpha * main_loss + (1 - alpha) * aux_loss
            else:
                # Just use main loss if there's no aux label
                combined_loss = main_loss
            
            # Track statistics
            total_loss += combined_loss.item()
            
            # Store predictions for F1 score calculation
            _, main_pred = torch.max(main_logits, 1)
            main_preds.extend(main_pred.cpu().numpy())
            main_labels.extend(main_label.cpu().numpy())
    
    # Calculate F1 score (macro)
    main_f1 = f1_score(main_labels, main_preds, average='macro', zero_division=0)
    
    return total_loss / len(dataloader), main_f1

# Hierarchical approach
def evaluate_hierarchical(model, dataloader, criterion, level_weights, alpha=0.7, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model.eval()
    total_loss = 0
    
    # Create counters for correct predictions and total samples for each level
    correct_preds = [0 for _ in range(4)]
    total_samples = 0
    
    # For exact match calculation
    all_correct = 0
    
    # For F1 score calculation
    all_preds = [[] for _ in range(4)]
    all_labels = [[] for _ in range(4)]
    
    # For whole-label evaluation
    whole_pred_labels = []
    whole_true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Get data
            sequences = batch['sequence']
            main_labels = [label.to(device) for label in batch['main_labels']]
            
            # Forward pass
            main_logits, _ = model(sequences)
            
            # Compute weighted loss across all levels for main task only
            main_losses = [criterion(main_logits[i], main_labels[i]) for i in range(4)]
            weighted_main_loss = sum(loss * weight for loss, weight in zip(main_losses, level_weights))
            
            # Track statistics
            total_loss += weighted_main_loss.item()
            batch_size = main_labels[0].size(0)
            total_samples += batch_size
            
            # For each sample in the batch
            batch_correct = torch.ones(batch_size, dtype=torch.bool, device=device)
            
            # Process each sample in the batch
            for i in range(batch_size):
                # Construct whole EC label for evaluation
                full_pred_ec = []
                full_true_ec = []
                
                # Calculate accuracy and collect predictions for each level
                for level in range(4):
                    _, level_preds = torch.max(main_logits[level], 1)
                    level_correct = (level_preds == main_labels[level])
                    correct_preds[level] += level_correct.sum().item()
                    
                    # Track which samples are correct at all levels for exact match
                    batch_correct = batch_correct & level_correct
                    
                    # Store predictions and labels for level-wise F1 score
                    all_preds[level].extend(level_preds.cpu().numpy())
                    all_labels[level].extend(main_labels[level].cpu().numpy())
                    
                    # Collect parts for whole EC label evaluation
                    full_pred_ec.append(level_preds[i].item())
                    full_true_ec.append(main_labels[level][i].item())
                
                # Add complete EC number predictions and truths for whole-label evaluation
                whole_pred_labels.append(tuple(full_pred_ec))
                whole_true_labels.append(tuple(full_true_ec))
            
            # Count exact matches (correct at all levels)
            all_correct += batch_correct.sum().item()
    
    # Calculate average accuracy for each level
    level_accs = [correct / total_samples for correct in correct_preds]
    
    # Calculate F1 scores for each level using macro average
    level_f1_scores = [
        f1_score(all_labels[level], all_preds[level], average='macro', zero_division=0)
        for level in range(4)
    ]
    
    # Calculate exact match accuracy
    exact_match = all_correct / total_samples
    
    # Calculate whole-label F1 score (treating each complete EC number as a class)
    # First convert tuples to strings to avoid numeric comparison issues
    str_pred_labels = ['.'.join(map(str, label)) for label in whole_pred_labels]
    str_true_labels = ['.'.join(map(str, label)) for label in whole_true_labels]
    
    # Use LabelBinarizer for whole-label F1 calculation
    lb = LabelBinarizer()
    # Find all unique labels
    all_unique_labels = list(set(str_pred_labels + str_true_labels))
    lb.fit(all_unique_labels)
    
    # Transform to binary matrices
    bin_pred = lb.transform(str_pred_labels)
    bin_true = lb.transform(str_true_labels)
    
    # If there's only one class, handle special case
    if len(all_unique_labels) == 1:
        whole_f1 = accuracy_score(bin_true, bin_pred)  # For single class, F1 = accuracy
    else:
        whole_f1 = f1_score(bin_true, bin_pred, average='macro', zero_division=0)
    
    return total_loss / len(dataloader), level_f1_scores, exact_match, whole_f1

# Test-only evaluation functions (just F1 score)
def evaluate_test_f1_original(model, dataloader, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model.eval()
    main_preds = []
    main_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            # Get data
            sequences = batch['sequence']
            main_label = batch['main_label'].to(device)
            
            # Forward pass (only need main_logits)
            main_logits, _ = model(sequences)
            
            # Store predictions for F1 score calculation
            _, main_pred = torch.max(main_logits, 1)
            main_preds.extend(main_pred.cpu().numpy())
            main_labels.extend(main_label.cpu().numpy())
    
    # Calculate F1 score (macro)
    main_f1 = f1_score(main_labels, main_preds, average='macro', zero_division=0)
    return main_f1

def create_unique_filename(cfg, prefix=""):
    # Extract model name - just the last part after the last slash
    model_short_name = cfg.model.model_name.split('/')[-1]
    
    # Create a unique name including key parameters
    unique_name = f"{prefix}_{model_short_name}_{cfg.model.architecture}_{cfg.model.fine_tuning_mode}_lr{cfg.training.learning_rate}_alpha{cfg.training.alpha}"
    
    # Include run ID/timestamp for absolute uniqueness
    timestamp = wandb.run.id if wandb.run else time.strftime("%Y%m%d_%H%M%S")
    
    return f"{unique_name}_{timestamp}.pt"
