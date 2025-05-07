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
from ec_blast.utils import (
    train_hierarchical,
    evaluate_hierarchical,
    train_original,
    evaluate_original,
    evaluate_test_f1_original,
    create_unique_filename,
)
from ec_blast.data import (
    UnifiedProteinDataset,
    TestProteinDataset,
    load_ec_dataset,
    parse_enzyme_dat,
    process_blast_results,
    add_auxiliary_labels,
)
from ec_blast.networks import TwoHeadFineTuner, HierarchicalFineTuner


# Config dataclasses for Hydra
@dataclass
class ModelConfig:
    model_name: str = "facebook/esm2_t6_8M_UR50D"
    freeze_base: bool = True
    layer_idx: int = -1
    dropout_rate: float = 0.1
    fine_tuning_mode: str = "classification_only"
    architecture: str = "hierarchical"  


@dataclass
class TrainingConfig:
    batch_size: int = 16
    num_epochs: int = 10
    learning_rate: float = 5e-4
    alpha: float = 0.7  
    val_split: float = 0.2  
    seed: int = 42
    level_weights: List[float] = field(default_factory=lambda: [0.4, 0.3, 0.2, 0.1])


@dataclass
class MainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb_project: str = "enzyme-classification"
    wandb_entity: Optional[str] = None 
    experiment_name: str = "enzyme-classifier"
    save_dir: str = "outputs"
    wandb_mode: str = "online"


cs = ConfigStore.instance()
cs.store(name="config", node=MainConfig)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Initialize wandb
    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=cfg.experiment_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=cfg.wandb_mode,
    )

    # Load dataset
    dataset = load_ec_dataset()

    # Process BLAST results and add auxiliary labels
    uniprot_to_ec = parse_enzyme_dat("enzyme.dat")
    query_to_aux_ec_list = process_blast_results("blast_results.tsv", uniprot_to_ec)
    dataset_with_aux = add_auxiliary_labels(dataset, query_to_aux_ec_list)

    # Choose architecture
    architecture = cfg.model.architecture
    print(f"Using architecture: {architecture}")

    if architecture == "original":
        # Original approach: single classifier for whole EC numbers
        # Get label mapping
        full_train_dataset = UnifiedProteinDataset(
            dataset_with_aux["train"], architecture="original"
        )
        label_to_idx = full_train_dataset.label_to_idx

        # Calculate split sizes
        train_size = int((1 - cfg.training.val_split) * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size

        # Split the dataset
        train_dataset, val_dataset = random_split(
            full_train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(cfg.training.seed),
        )

        # Create test dataset
        test_dataset = TestProteinDataset(dataset_with_aux["test"], label_to_idx)

        print(f"Training set size: {len(train_dataset)}")
        print(f"Validation set size: {len(val_dataset)}")
        print(f"Test set size: {len(test_dataset)}")
        print(f"Number of classes: {len(label_to_idx)}")

        # Log dataset sizes to wandb
        wandb.log(
            {
                "train_size": len(train_dataset),
                "val_size": len(val_dataset),
                "test_size": len(test_dataset),
                "num_classes": len(label_to_idx),
            }
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size=cfg.training.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=cfg.training.batch_size)

        # Load the base model
        model = dgeb.get_model(cfg.model.model_name)

        # Create the fine-tuner model
        fine_tuner = TwoHeadFineTuner(
            base_model=model,
            num_classes=len(label_to_idx),
            layer_idx=cfg.model.layer_idx,
            dropout_rate=cfg.model.dropout_rate,
        )

    else:  # "hierarchical"
        # Hierarchical approach: separate classifiers for each EC level
        # Get label mappings
        full_train_dataset = UnifiedProteinDataset(
            dataset_with_aux["train"], architecture="hierarchical"
        )
        label_mappings = full_train_dataset.label_mappings

        # Calculate split sizes
        train_size = int((1 - cfg.training.val_split) * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size

        # Split the dataset
        train_dataset, val_dataset = random_split(
            full_train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(cfg.training.seed),
        )

        # Create test dataset
        test_dataset = UnifiedProteinDataset(
            dataset_with_aux["test"],
            architecture="hierarchical",
            label_mappings=label_mappings,
        )

        print(f"Training set size: {len(train_dataset)}")
        print(f"Validation set size: {len(val_dataset)}")
        print(f"Test set size: {len(test_dataset)}")

        # Extract number of classes for each level
        num_classes_per_level = [len(mapping) for mapping in label_mappings]
        print(f"Number of classes per level: {num_classes_per_level}")

        # Log dataset sizes to wandb
        wandb.log(
            {
                "train_size": len(train_dataset),
                "val_size": len(val_dataset),
                "test_size": len(test_dataset),
                "num_classes_level1": num_classes_per_level[0],
                "num_classes_level2": num_classes_per_level[1],
                "num_classes_level3": num_classes_per_level[2],
                "num_classes_level4": num_classes_per_level[3],
            }
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size=cfg.training.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=cfg.training.batch_size)

        # Load the base model
        model = dgeb.get_model(cfg.model.model_name)

        # Create the fine-tuner model
        fine_tuner = HierarchicalFineTuner(
            base_model=model,
            num_classes_per_level=num_classes_per_level,
            layer_idx=cfg.model.layer_idx,
            dropout_rate=cfg.model.dropout_rate,
        )

    # Configure fine-tuning mode
    if cfg.model.fine_tuning_mode == "classification_only":
        # Freeze the pre-trained model parameters
        for param in fine_tuner.base.encoder.parameters():
            param.requires_grad = False
        print("Fine-tuning mode: Classification heads only")
    else:
        print("Fine-tuning mode: Full model (base + classification heads)")

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    fine_tuner = fine_tuner.to(device)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Configure optimizer based on architecture and fine-tuning mode
    if architecture == "original":
        if cfg.model.fine_tuning_mode == "classification_only":
            # Only train the classification heads
            optimizer = optim.Adam(
                [
                    {"params": fine_tuner.main_head.parameters()},
                    {"params": fine_tuner.aux_head.parameters()},
                ],
                lr=cfg.training.learning_rate,
            )
        else:
            # Train all parameters with different learning rates
            optimizer = optim.Adam(
                [
                    {
                        "params": fine_tuner.base.encoder.parameters(),
                        "lr": cfg.training.learning_rate * 0.1,
                    },  # Lower LR for base
                    {"params": fine_tuner.main_head.parameters()},
                    {"params": fine_tuner.aux_head.parameters()},
                ],
                lr=cfg.training.learning_rate,
            )
    else:  # "hierarchical"
        if cfg.model.fine_tuning_mode == "classification_only":
            # Only train the classification heads
            params = []
            for heads in [fine_tuner.main_heads, fine_tuner.aux_heads]:
                for head in heads:
                    params.extend(list(head.parameters()))
            optimizer = optim.Adam(params, lr=cfg.training.learning_rate)
        else:
            # Train all parameters with different learning rates
            params = [
                {
                    "params": fine_tuner.base.encoder.parameters(),
                    "lr": cfg.training.learning_rate * 0.1,
                }
            ]
            for heads in [fine_tuner.main_heads, fine_tuner.aux_heads]:
                for head in heads:
                    params.append({"params": head.parameters()})
            optimizer = optim.Adam(params, lr=cfg.training.learning_rate)

    # Training loop
    if architecture == "original":
        best_val_f1 = -1

        for epoch in range(cfg.training.num_epochs):
            print(f"Epoch {epoch+1}/{cfg.training.num_epochs}")

            # Train
            train_loss, train_f1 = train_original(
                fine_tuner,
                train_loader,
                optimizer,
                criterion,
                cfg.training.alpha,
                device,
            )
            print(f"Train Loss: {train_loss:.4f}, Train Macro F1: {train_f1:.4f}")

            # Evaluate on validation set
            val_loss, val_f1 = evaluate_original(
                fine_tuner, val_loader, criterion, cfg.training.alpha, device
            )
            print(f"Validation Loss: {val_loss:.4f}, Validation Macro F1: {val_f1:.4f}")
            print("-" * 50)

            # Log metrics to wandb
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_f1": train_f1,
                    "val_loss": val_loss,
                    "val_f1": val_f1,
                }
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_filename = create_unique_filename(cfg, prefix="best")
                model_path = os.path.join(
                    hydra.utils.get_original_cwd(), cfg.save_dir, best_model_filename
                )
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(fine_tuner.state_dict(), model_path)
                print(f"New best model saved as: {best_model_filename}")

                # Store the filename in wandb for later reference
                wandb.run.summary["best_model_filename"] = best_model_filename
                wandb.run.summary["best_val_f1"] = best_val_f1
                wandb.run.summary["best_epoch"] = epoch + 1

        print("\nEvaluating on test set with best model...")

        # Load the best model
        best_model_filename = wandb.run.summary["best_model_filename"]
        best_model_path = os.path.join(
            hydra.utils.get_original_cwd(), cfg.save_dir, best_model_filename
        )

        # Create a new model instance
        best_model = TwoHeadFineTuner(
            base_model=model,  # Use the same base model
            num_classes=len(label_to_idx),
            layer_idx=cfg.model.layer_idx,
            dropout_rate=cfg.model.dropout_rate,
        )

        # Load the saved weights
        best_model.load_state_dict(torch.load(best_model_path))
        best_model = best_model.to(device)

        # Evaluate with the best model
        test_f1 = evaluate_test_f1_original(best_model, test_loader, device)
        print(f"Test Macro F1 Score (best model): {test_f1:.4f}")

        # Log test metrics to wandb
        wandb.log(
            {
                "test_f1_best_model": test_f1,
            }
        )
        wandb.run.summary["test_f1_best_model"] = test_f1

    else:  # "hierarchical"
        best_val_exact_match = -1

        for epoch in range(cfg.training.num_epochs):
            print(f"Epoch {epoch+1}/{cfg.training.num_epochs}")

            # Train
            train_loss, train_level_accs, train_level_f1 = train_hierarchical(
                fine_tuner,
                train_loader,
                optimizer,
                criterion,
                cfg.training.level_weights,
                cfg.training.alpha,
                device,
            )
            print(f"Train Loss: {train_loss:.4f}")
            for level, (acc, f1) in enumerate(zip(train_level_accs, train_level_f1)):
                print(
                    f"Train Level {level+1} - Accuracy: {acc:.4f}, Macro F1: {f1:.4f}"
                )

            # Evaluate on validation set
            val_loss, val_level_f1, val_exact_match, val_whole_f1 = (
                evaluate_hierarchical(
                    fine_tuner,
                    val_loader,
                    criterion,
                    cfg.training.level_weights,
                    cfg.training.alpha,
                    device,
                )
            )
            print(f"Validation Loss: {val_loss:.4f}")
            for level, f1 in enumerate(val_level_f1):
                print(f"Validation Level {level+1} Macro F1: {f1:.4f}")
            print(f"Validation Exact Match: {val_exact_match:.4f}")
            print(f"Validation Whole-Label Macro F1: {val_whole_f1:.4f}")
            print("-" * 50)

            # Log metrics to wandb
            wandb_log = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_exact_match": val_exact_match,
                "val_whole_f1": val_whole_f1,
            }

            # Add level-specific metrics
            for level, acc in enumerate(train_level_accs):
                wandb_log[f"train_level{level+1}_acc"] = acc
                wandb_log[f"train_level{level+1}_f1"] = train_level_f1[level]

            for level, f1 in enumerate(val_level_f1):
                wandb_log[f"val_level{level+1}_f1"] = f1

            wandb.log(wandb_log)

            # Save best model based on exact match
            if val_exact_match > best_val_exact_match:
                best_val_exact_match = val_exact_match
                best_model_filename = create_unique_filename(cfg, prefix="best")
                model_path = os.path.join(
                    hydra.utils.get_original_cwd(), cfg.save_dir, best_model_filename
                )
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(fine_tuner.state_dict(), model_path)
                print(f"New best model saved as: {best_model_filename}")

                # Store the filename in wandb for later reference
                wandb.run.summary["best_model_filename"] = best_model_filename
                wandb.run.summary["best_val_exact_match"] = best_val_exact_match
                wandb.run.summary["best_epoch"] = epoch + 1

        print("\nEvaluating on test set with best model...")

        # Load the best model
        best_model_filename = wandb.run.summary["best_model_filename"]
        best_model_path = os.path.join(
            hydra.utils.get_original_cwd(), cfg.save_dir, best_model_filename
        )

        # Create a new model instance
        best_model = HierarchicalFineTuner(
            base_model=model,  # Use the same base model
            num_classes_per_level=[len(mapping) for mapping in label_mappings],
            layer_idx=cfg.model.layer_idx,
            dropout_rate=cfg.model.dropout_rate,
        )

        # Load the saved weights
        best_model.load_state_dict(torch.load(best_model_path))
        best_model = best_model.to(device)

        # Evaluate with the best model
        test_loss, test_level_f1, test_exact_match, test_whole_f1 = (
            evaluate_hierarchical(
                best_model,
                test_loader,
                criterion,
                cfg.training.level_weights,
                cfg.training.alpha,
                device,
            )
        )

        print(f"Test Loss: {test_loss:.4f}")
        for level, f1 in enumerate(test_level_f1):
            print(f"Test Level {level+1} Macro F1: {f1:.4f}")
        print(f"Test Exact Match: {test_exact_match:.4f}")
        print(f"Test Whole-Label Macro F1: {test_whole_f1:.4f}")

        # Log test metrics to wandb
        test_log = {
            "test_loss": test_loss,
            "test_exact_match": test_exact_match,
            "test_whole_f1": test_whole_f1,
        }

        for level, f1 in enumerate(test_level_f1):
            test_log[f"test_level{level+1}_f1"] = f1

        wandb.log(test_log)
        wandb.run.summary.update(test_log)

    # Save final model
    final_model_filename = create_unique_filename(cfg, prefix="final")
    final_model_path = os.path.join(
        hydra.utils.get_original_cwd(), cfg.save_dir, final_model_filename
    )
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    torch.save(fine_tuner.state_dict(), final_model_path)
    print(f"Final model saved as: {final_model_filename}")

    # Also log the final filename to wandb
    wandb.run.summary["final_model_filename"] = final_model_filename

    # Close wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
