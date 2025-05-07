# EC-BLAST: Hierarchical Enzyme Classification with Foundation Models

A framework for enzyme classification that leverages protein language models and BLAST search results as auxiliary labels.

## BLAST Process Overview

To generate auxiliary EC labels for each protein sequence:

```bash
# 1. Download Swiss-Prot database
wget https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz
gunzip uniprot_sprot.fasta.gz

# 2. Download enzyme.dat for EC mapping
wget https://ftp.expasy.org/databases/enzyme/enzyme.dat

# 3. Create BLAST database
makeblastdb -in uniprot_sprot.fasta -dbtype prot -out swiss_prot_db -parse_seqids

# 4. Run BLAST search
blastp -query query_sequences.fasta \
       -db swiss_prot_db \
       -out blast_results.tsv \
       -outfmt "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore" \
       -evalue 1e-5 \
       -max_target_seqs 10 \
       -num_threads 8
```

These BLAST results provide evolutionary information to enhance the neural model's predictions.

## Overview
This repo implements two classification architectures:

1. Original: Treats EC numbers as flat class labels
2. Hierarchical: Models each level of EC numbers (e.g., 1.14.14.18) with separate classifiers

## Features

- Two-task learning framework with BLAST-derived auxiliary labels
- Level-weighted loss for hierarchical EC classification
- Comprehensive evaluation metrics including level-wise and whole-label F1
- Weights & Biases integration for experiment tracking

# Supported Models
Any protein language model available through the dgeb library can be used:

- ESM2 family (t6_8M_UR50D, t12_35M_UR50D, t30_150M_UR50D, t33_650M_UR50D)
- ESM1b
- ProtT5
- Other models supported by dgeb

## Requirements
```bash
torch
dgeb
wandb
hydra-core
omegaconf
pandas
biopython
scikit-learn
tqdm
datasets
```

## Usage
```bash
# Run with default configuration
python -m ec_blast.main

# Run with custom configuration
python -m ec_blast.main model.architecture=hierarchical training.level_weights=[0.5,0.3,0.1,0.1]

# Run for multiple parameters using sbatch
sbatch run_all.sh
```