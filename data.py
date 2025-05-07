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

# Load the dataset
def load_ec_dataset():
    dataset = load_dataset("tattabio/ec_classification")
    print("Dataset loaded successfully:")
    print(dataset)
    return dataset

def parse_enzyme_dat(enzyme_file="enzyme.dat"):
    """Parses enzyme.dat to create a mapping from UniProt AC to EC numbers."""
    ec_map = collections.defaultdict(list)
    records = Enzyme.parse(open(enzyme_file))
    for record in records:
        ec_number = record.get('ID', None)
        if not ec_number:
            continue
        # DR lines contain cross-references, format is
        dr_records = record.get('DR',)
        for accession, _ in dr_records:
            # We are interested in UniProtKB/Swiss-Prot accessions
            if len(accession) > 0 and accession.isalnum():
                 ec_map[accession].append(ec_number)
    return dict(ec_map)

def process_blast_results(blast_file="blast_results.tsv", uniprot_to_ec=None):
    column_names = ['qseqid', 'sseqid', 'pident', 'length', 'mismatch', 'gapopen',
                'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore']

    # Load the TSV file into a pandas DataFrame
    blast_results_df = pd.read_csv(
        blast_file,
        sep='\t',
        header=None,
        names=column_names
    )
    print(f"Loaded {len(blast_results_df)} BLAST/DIAMOND hits.")
    
    # Ensure numeric types are correct
    blast_results_df['evalue'] = pd.to_numeric(blast_results_df['evalue'], errors='coerce')
    blast_results_df['bitscore'] = pd.to_numeric(blast_results_df['bitscore'], errors='coerce')
    blast_results_df['pident'] = pd.to_numeric(blast_results_df['pident'], errors='coerce')

    # Drop rows where conversion failed
    blast_results_df.dropna(subset=['evalue', 'bitscore', 'pident'], inplace=True)

    # Keep only necessary columns for mapping
    relevant_hits = blast_results_df[['qseqid', 'sseqid', 'evalue', 'bitscore', 'pident']].copy()
    print(f"Kept {len(relevant_hits)} hits after potential filtering.")

    query_to_aux_ec = collections.defaultdict(set)  # Using set to store unique ECs

    # Process BLAST hits to get auxiliary EC numbers
    for _, row in relevant_hits.iterrows():
        query_id = row['qseqid']
        subject_id = row['sseqid']
        
        # Handle potential prefixes like 'sp|' or 'tr|' if present
        if '|' in subject_id:
            try:
                accession = subject_id.split('|')[1]
            except IndexError:
                accession = subject_id  # Fallback if parsing fails
        else:
            accession = subject_id

        if uniprot_to_ec and accession in uniprot_to_ec:
            associated_ecs = uniprot_to_ec[accession]
            query_to_aux_ec[query_id].update(associated_ecs)  # Add all ECs from this hit

    # Convert sets to lists for easier handling later
    query_to_aux_ec_list = {query: list(ecs) for query, ecs in query_to_aux_ec.items()}

    print(f"Generated auxiliary EC lists for {len(query_to_aux_ec_list)} query proteins.")
    return query_to_aux_ec_list

def add_auxiliary_labels(dataset, query_to_aux_ec_list, id_column='Entry'):
    """Function to add auxiliary EC list to a dataset."""
    def process_example(example):
        protein_id = example[id_column]
        # Get the auxiliary ECs, default to empty list if no hits found
        example['auxiliary_ec'] = query_to_aux_ec_list.get(protein_id, [])
        return example

    # Apply the transformation
    dataset_with_aux = dataset.map(process_example)
    
    # Handle auxiliary EC extraction
    def transform_auxiliary_ec(example):
        if len(example['auxiliary_ec']) >= 1:
            # If auxiliary_ec has only one element, use that element
            example['auxiliary_ec'] = example['auxiliary_ec'][0]
        else:
            # If auxiliary_ec is empty, use main label
            example['auxiliary_ec'] = example['Label']
        return example

    # Apply the transformation to train set
    dataset_with_aux['train'] = dataset_with_aux['train'].map(transform_auxiliary_ec)
    
    print("Auxiliary labels added to the dataset:")
    print(dataset_with_aux)
    return dataset_with_aux

# Unified Dataset class that can handle both approaches
class UnifiedProteinDataset(Dataset):
    def __init__(self, dataset, architecture="hierarchical", label_mappings=None):
        self.dataset = dataset
        self.architecture = architecture  # "original" or "hierarchical"
        
        if architecture == "original":
            # For original approach, we need a single mapping for whole EC numbers
            if not label_mappings:
                self.label_to_idx = self._create_whole_label_mapping()
            else:
                self.label_to_idx = label_mappings
        else:
            # For hierarchical approach, we need separate mappings for each level
            if not label_mappings:
                self.label_mappings = self._create_hierarchical_label_mapping()
            else:
                self.label_mappings = label_mappings
    
    def _create_whole_label_mapping(self):
        # Get unique EC labels from main and auxiliary labels
        all_labels = set()
        
        # Process train set
        for item in self.dataset:
            all_labels.add(item['Label'])
            if 'auxiliary_ec' in item and item['auxiliary_ec']:
                all_labels.add(item['auxiliary_ec'])
        
        # Create mapping
        label_to_idx = {label: idx for idx, label in enumerate(sorted(all_labels))}
        print(f"Total unique EC classes (whole): {len(label_to_idx)}")
        return label_to_idx
    
    def _create_hierarchical_label_mapping(self):
        # Create separate mappings for each EC level
        level_mappings = [{} for _ in range(4)]
        
        # Process all items
        for item in self.dataset:
            # Process main label
            ec_parts = self._split_ec_number(item['Label'])
            
            # Add each part to its respective mapping
            for level, part in enumerate(ec_parts):
                if part not in level_mappings[level]:
                    level_mappings[level][part] = len(level_mappings[level])
            
            # Process auxiliary label if available
            if 'auxiliary_ec' in item and item['auxiliary_ec']:
                aux_ec_parts = self._split_ec_number(item['auxiliary_ec'])
                
                # Add each part to its respective mapping
                for level, part in enumerate(aux_ec_parts):
                    if part not in level_mappings[level]:
                        level_mappings[level][part] = len(level_mappings[level])
        
        # Print statistics
        for level, mapping in enumerate(level_mappings):
            print(f"EC Level {level+1}: {len(mapping)} unique values")
        
        return level_mappings
    
    def _split_ec_number(self, ec_number):
        """Split EC number into its 4 hierarchical components."""
        parts = ec_number.split('.')
        # Ensure we have exactly 4 parts, pad with "0" if necessary
        while len(parts) < 4:
            parts.append("0")
        return parts[:4]  # Take only first 4 parts if there are more
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        sequence = item['Sequence']
        
        if self.architecture == "original":
            # Original approach: single label
            main_label_idx = self.label_to_idx[item['Label']]
            
            # Get auxiliary label
            if 'auxiliary_ec' in item and item['auxiliary_ec'] in self.label_to_idx:
                aux_label_idx = self.label_to_idx[item['auxiliary_ec']]
            else:
                # If aux label doesn't exist, use main label
                aux_label_idx = main_label_idx
            
            return {
                'sequence': sequence,
                'main_label': torch.tensor(main_label_idx, dtype=torch.long),
                'aux_label': torch.tensor(aux_label_idx, dtype=torch.long)
            }
        else:
            # Hierarchical approach: split into levels
            # Split the EC numbers into hierarchical components
            main_ec_parts = self._split_ec_number(item['Label'])
            
            # Convert each part to its corresponding index
            main_label_indices = [
                self.label_mappings[level][part] 
                for level, part in enumerate(main_ec_parts)
            ]
            
            # Same for auxiliary EC if available
            if 'auxiliary_ec' in item and item['auxiliary_ec']:
                aux_ec_parts = self._split_ec_number(item['auxiliary_ec'])
                aux_label_indices = [
                    self.label_mappings[level][part] 
                    for level, part in enumerate(aux_ec_parts)
                ]
            else:
                # If no auxiliary EC, use main EC
                aux_label_indices = main_label_indices
            
            return {
                'sequence': sequence,
                'main_labels': [torch.tensor(idx, dtype=torch.long) for idx in main_label_indices],
                'aux_labels': [torch.tensor(idx, dtype=torch.long) for idx in aux_label_indices]
            }

# Test Dataset - only for original architecture
class TestProteinDataset(Dataset):
    def __init__(self, dataset, label_to_idx):
        self.dataset = dataset
        self.label_to_idx = label_to_idx
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        sequence = item['Sequence']
        
        # Get main label
        if item['Label'] in self.label_to_idx:
            main_label_idx = self.label_to_idx[item['Label']]
        else:
            # Handle unknown labels
            main_label_idx = 0  # Assume 0 is a valid class
            
        return {
            'sequence': sequence,
            'main_label': torch.tensor(main_label_idx, dtype=torch.long)
        }
