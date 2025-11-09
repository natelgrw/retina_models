#!/usr/bin/env python3
"""
train_ReTiNA_MLP1.py

Author: natelgrw
Created: 11/09/2025

Trains a PyTorch MLP model on the ReTiNA dataset with cross-validation 
on scaffold, method, and cluster splits using ReTINA encoder features.
"""

import sys
import json
import math
import random
import warnings
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from scipy import stats
from pathlib import Path

warnings.filterwarnings('ignore')


# ===== Configuration ===== #


RANDOM_SEED = 42

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
PERFORMANCE_DIR = PROJECT_ROOT / "performance" / "ReTiNA_MLP1"
ENCODING_DIR = SCRIPT_DIR / "encoding"

sys.path.insert(0, str(ENCODING_DIR))
from encoding.retina_encoder import (
    load_compound_descriptors,
    encode_solvents,
    normalize_gradient,
    encode_column,
    SOLVENTS_ORDER,
    ADDITIVES_ORDER
)

HYPERPARAMETERS = {
    "hidden_layers": [1024, 512, 256, 128, 64],
    "dropout_rate": 0.3,
    "use_batch_norm": True,
    "use_residual": True,
    "learning_rate": 0.001,
    "batch_size": 128,
    "max_epochs": 500,
    "early_stopping_patience": 50,
    "early_stopping_min_delta": 1e-4,
    "lr_scheduler_patience": 20,
    "lr_scheduler_factor": 0.5,
    "lr_warmup_epochs": 5,
    "weight_decay": 1e-5,
    "grad_clip_norm": 1.0,
    "optimizer": "AdamW",
    "random_state": RANDOM_SEED
}

SPLIT_DIRS = {
    "scaffold": DATA_DIR / "scaffold_split",
    "method": DATA_DIR / "method_split",
    "cluster": DATA_DIR / "cluster_split"
}

SPLIT_FILES = {
    "scaffold": ["fold_1.csv", "fold_2.csv", "fold_3.csv", "fold_4.csv", "fold_5.csv"],
    "method": ["methods_1.csv", "methods_2.csv", "methods_3.csv", "methods_4.csv", "methods_5.csv"],
    "cluster": ["cluster_1.csv", "cluster_2.csv", "cluster_3.csv", "cluster_4.csv", "cluster_5.csv"]
}


# ===== Utilities ===== #


def set_seed(seed: int):
    """
    Sets the seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sanitize_feature_name(name: str, existing: set) -> str:
    """
    Sanitizes a feature name to be a valid Python variable name.
    """
    sanitized = ''.join(ch if ch.isalnum() or ch == '_' else '_' for ch in name)
    sanitized = '_'.join(filter(None, sanitized.split('_')))
    if not sanitized:
        sanitized = "feature"
    base = sanitized
    counter = 1
    while sanitized in existing:
        counter += 1
        sanitized = f"{base}_{counter}"
    existing.add(sanitized)
    return sanitized


def build_feature_names(descriptor_cols: List[str]) -> List[str]:
    """
    Builds a list of feature names matching the encoding order.
    """
    feature_names: List[str] = []
    seen: set = set()

    def add(raw: str):
        """
        Adds a feature name to the list.
        """
        feature_names.append(sanitize_feature_name(raw, seen))

    for col in descriptor_cols:
        add(f"comp_{col}")

    for phase in ['A', 'B']:
        for solvent in SOLVENTS_ORDER:
            add(f"solv_{solvent}_{phase}_pct")
        for additive in ADDITIVES_ORDER:
            add_short = additive[:20] if len(additive) > 20 else additive
            add(f"add_{add_short}_{phase}_M")

    for i in range(100):
        add(f"grad_t{i:03d}")
    add("grad_total_time")

    for name in ['col_RP', 'col_HI', 'col_diam_mm', 'col_len_mm', 'col_part_um']:
        add(name)

    add('flow_rate_mL_min')
    add('temp_C')

    return feature_names


def encode_single_sample(row, comp_descriptors) -> np.ndarray:
    """
    Encodes a single data sample into a feature vector.
    """
    features = []

    compound = row['compound']
    if compound in comp_descriptors:
        comp_features = comp_descriptors[compound]
        features.extend(comp_features)
    else:
        features.extend([0.0] * len(next(iter(comp_descriptors.values()))))

    solvent_features = encode_solvents(row['solvents'])
    features.extend(solvent_features)

    gradient_features, gradient_total_time = normalize_gradient(row['gradient'])
    features.extend(gradient_features)
    features.append(gradient_total_time)

    column_features = encode_column(row['column'])
    features.extend(column_features)

    features.append(float(row['flow_rate']))
    features.append(float(row['temp']))

    return np.array(features, dtype=np.float32)


def prepare_data(df: pd.DataFrame, comp_descriptors) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Prepares features and target for training.
    """
    X_list = []
    valid_indices = []

    for idx, row in df.iterrows():
        try:
            features = encode_single_sample(row, comp_descriptors)
            X_list.append(features)
            valid_indices.append(idx)
        except Exception:
            continue

    X = np.vstack(X_list)
    y = df.loc[valid_indices, 'rt'].values.astype(np.float32)

    return X, y, df.loc[valid_indices]


def compute_permutation_importance(model: nn.Module,
                                   X: np.ndarray,
                                   y: np.ndarray,
                                   feature_names: List[str],
                                   grad_total_idx: int,
                                   device: torch.device,
                                   rng: np.random.Generator,
                                   baseline_preds: np.ndarray) -> List[Tuple[str, float]]:
    """
    Computes permutation importance using the evaluation set.
    """
    model.eval()
    baseline_mse = mean_squared_error(y, baseline_preds)
    importances: List[Tuple[str, float]] = []

    for idx, name in enumerate(feature_names):
        shuffled = X.copy()
        column = X[:, idx].copy()
        shuffled[:, idx] = column[rng.permutation(column.shape[0])]

        shuffled_tensor = torch.from_numpy(shuffled.astype(np.float32)).to(device)
        with torch.no_grad():
            shuffled_preds = model(shuffled_tensor).cpu().numpy()

        shuffled_method_lengths = np.maximum(shuffled[:, grad_total_idx], 0.0)
        shuffled_preds = np.clip(shuffled_preds, 0.0, shuffled_method_lengths)

        perm_mse = mean_squared_error(y, shuffled_preds)
        importances.append((name, float(perm_mse - baseline_mse)))

    return importances


def calculate_metrics(y_true, y_pred):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    try:
        pearson_r, _ = stats.pearsonr(y_true, y_pred)
    except ValueError:
        pearson_r = 0.0

    errors = np.abs(y_true - y_pred)
    within_5s = (errors <= 5).mean() * 100
    within_10s = (errors <= 10).mean() * 100
    within_30s = (errors <= 30).mean() * 100

    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Pearson_r': pearson_r,
        'Within_5s': within_5s,
        'Within_10s': within_10s,
        'Within_30s': within_30s,
        'N_test': len(y_true)
    }


# ===== Model Definition ===== #


class ResidualBlock(nn.Module):
    """
    Defines a residual block for the RetinaMLP model.
    """

    def __init__(self, in_features: int, out_features: int, dropout: float, use_batch_norm: bool, use_residual: bool):
        super().__init__()
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        self.in_features = in_features
        self.out_features = out_features

        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features) if use_batch_norm else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        if use_residual:
            if in_features != out_features:
                self.residual = nn.Linear(in_features, out_features)
            else:
                self.residual = nn.Identity()
        else:
            self.residual = None

    def forward(self, x):
        """
        Performs a forward pass through the residual block.
        """
        out = self.linear(x)
        if self.bn is not None:
            out = self.bn(out)
        out = F.relu(out)
        if self.dropout is not None:
            out = self.dropout(out)

        if self.residual is not None:
            res = self.residual(x)
            out = out + res

        return out


class RetinaMLP(nn.Module):
    """
    Defines the RetinaMLP model.
    """
    def __init__(self, input_dim: int, hidden_layers: List[int], dropout_rate: float,
                 use_batch_norm: bool, use_residual: bool):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(
                ResidualBlock(
                    in_features=prev_dim,
                    out_features=hidden_dim,
                    dropout=dropout_rate,
                    use_batch_norm=use_batch_norm,
                    use_residual=use_residual
                )
            )
            prev_dim = hidden_dim

        self.layers = nn.ModuleList(layers)
        self.output = nn.Linear(prev_dim, 1)

    def forward(self, x):
        """
        Performs a forward pass through the RetinaMLP model.
        """
        for layer in self.layers:
            x = layer(x)
        return self.output(x).squeeze(-1)


# ===== Training Helper Functions ===== #


def create_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True) -> DataLoader:
    """
    Creates a DataLoader for the training data.
    """
    tensor_x = torch.from_numpy(X)
    tensor_y = torch.from_numpy(y)
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_one_epoch(model: nn.Module, dataloader: DataLoader, criterion, optimizer, device, grad_clip_norm):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        running_loss += loss.item() * batch_x.size(0)

    return running_loss / len(dataloader.dataset)


def evaluate(model: nn.Module, dataloader: DataLoader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    preds = []
    targets = []
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            running_loss += loss.item() * batch_x.size(0)
            preds.append(outputs.cpu().numpy())
            targets.append(batch_y.cpu().numpy())

    avg_loss = running_loss / len(dataloader.dataset)
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return avg_loss, preds, targets


def warmup_lr(optimizer, base_lr, epoch, warmup_epochs):
    """
    Warms up the learning rate.
    """
    if epoch < warmup_epochs and warmup_epochs > 0:
        warmup_factor = (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = base_lr * warmup_factor
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = base_lr


# ===== Training Pipeline ===== #


def train_single_split(split_type: str, holdout_id: int, comp_descriptors,
                       feature_names: List[str], grad_total_idx: int,
                       device: torch.device):
    """
    Trains the model on a single train/test split.
    """
    print(f"\n{'='*70}")
    print(f"Training: {split_type.capitalize()} Split - Holdout {holdout_id}")
    print(f"{'='*70}")

    split_dir = SPLIT_DIRS[split_type]
    split_files = SPLIT_FILES[split_type]

    folds = []
    for i, filename in enumerate(split_files):
        fold_path = split_dir / filename
        fold_df = pd.read_csv(fold_path)
        fold_df['fold_id'] = i + 1
        folds.append(fold_df)

    test_df = folds[holdout_id - 1]
    train_dfs = [folds[i] for i in range(5) if i != holdout_id - 1]
    train_df = pd.concat(train_dfs, ignore_index=True)

    print(f"  Train size: {len(train_df):,}")
    print(f"  Test size: {len(test_df):,}")

    X_train_all, y_train_all, train_merged = prepare_data(train_df, comp_descriptors)
    X_test, y_test, test_merged = prepare_data(test_df, comp_descriptors)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all,
        y_train_all,
        test_size=0.1,
        random_state=RANDOM_SEED,
        shuffle=True
    )

    train_loader = create_dataloader(X_train, y_train, HYPERPARAMETERS["batch_size"], shuffle=True)
    val_loader = create_dataloader(X_val, y_val, HYPERPARAMETERS["batch_size"], shuffle=False)
    test_loader = create_dataloader(X_test, y_test, HYPERPARAMETERS["batch_size"], shuffle=False)

    input_dim = X_train.shape[1]
    model = RetinaMLP(
        input_dim=input_dim,
        hidden_layers=HYPERPARAMETERS["hidden_layers"],
        dropout_rate=HYPERPARAMETERS["dropout_rate"],
        use_batch_norm=HYPERPARAMETERS["use_batch_norm"],
        use_residual=HYPERPARAMETERS["use_residual"]
    ).to(device)

    if HYPERPARAMETERS["optimizer"].lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=HYPERPARAMETERS["learning_rate"],
            weight_decay=HYPERPARAMETERS["weight_decay"]
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=HYPERPARAMETERS["learning_rate"],
            weight_decay=HYPERPARAMETERS["weight_decay"]
        )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=HYPERPARAMETERS["lr_scheduler_factor"],
        patience=HYPERPARAMETERS["lr_scheduler_patience"],
        threshold=HYPERPARAMETERS["early_stopping_min_delta"],
        threshold_mode='abs'
    )

    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    base_lr = HYPERPARAMETERS["learning_rate"]

    for epoch in range(HYPERPARAMETERS["max_epochs"]):
        warmup_lr(optimizer, base_lr, epoch, HYPERPARAMETERS["lr_warmup_epochs"])

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            HYPERPARAMETERS["grad_clip_norm"]
        )

        val_loss, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        improved = val_loss + HYPERPARAMETERS["early_stopping_min_delta"] < best_val_loss

        if improved:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if epochs_no_improve >= HYPERPARAMETERS["early_stopping_patience"]:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    test_loss, test_preds, _ = evaluate(model, test_loader, criterion, device)

    method_lengths_test = np.maximum(X_test[:, grad_total_idx], 0.0)
    test_preds = np.clip(test_preds, 0.0, method_lengths_test)

    metrics = calculate_metrics(y_test, test_preds)

    print(f"  Test Results:")
    print(f"    RMSE: {metrics['RMSE']:.3f}s")
    print(f"    MAE: {metrics['MAE']:.3f}s")
    print(f"    R²: {metrics['R2']:.4f}")
    print(f"    Pearson r: {metrics['Pearson_r']:.4f}")
    print(f"    Within 10s: {metrics['Within_10s']:.1f}%")

    predictions_df = test_merged[['compound']].copy()
    predictions_df['split_type'] = split_type
    predictions_df['holdout_id'] = holdout_id
    predictions_df['rt_true'] = y_test
    predictions_df['rt_pred'] = test_preds
    predictions_df['method_number'] = test_merged['method_number'].values
    predictions_df['source'] = test_merged['source'].values

    return {
        'split_type': split_type,
        'holdout_id': holdout_id,
        'metrics': metrics,
        'predictions': predictions_df,
        'model_state': best_model_state
    }


def train_all_splits(device: torch.device):
    """
    Trains the model on all splits and collects results.
    """
    print("\n" + "="*70)
    print("STARTING CROSS-VALIDATION TRAINING")
    print("="*70)

    print("\nLoading compound descriptors...")
    comp_descriptors, descriptor_cols = load_compound_descriptors()
    feature_names = build_feature_names(descriptor_cols)
    grad_total_idx = feature_names.index('grad_total_time')

    all_results = []
    all_predictions = []

    for split_type in ['scaffold', 'method', 'cluster']:
        for holdout_id in range(1, 6):
            result = train_single_split(
                split_type,
                holdout_id,
                comp_descriptors,
                feature_names,
                grad_total_idx,
                device
            )

            all_results.append({
                'Split_Type': split_type,
                'Holdout': holdout_id,
                **result['metrics']
            })

            all_predictions.append(result['predictions'])

    return all_results, all_predictions, comp_descriptors, feature_names, grad_total_idx


def create_summary_table(all_results: List[Dict]) -> pd.DataFrame:
    """
    Creates a summary table with averages.
    """
    results_df = pd.DataFrame(all_results)
    summary_rows = []

    for split_type in ['scaffold', 'method', 'cluster']:
        split_results = results_df[results_df['Split_Type'] == split_type]
        avg_row = {
            'Split_Type': f"{split_type.capitalize()} (avg)",
            'Holdout': '—',
            'RMSE': split_results['RMSE'].mean(),
            'MAE': split_results['MAE'].mean(),
            'R2': split_results['R2'].mean(),
            'Pearson_r': split_results['Pearson_r'].mean(),
            'Within_5s': split_results['Within_5s'].mean(),
            'Within_10s': split_results['Within_10s'].mean(),
            'Within_30s': split_results['Within_30s'].mean(),
            'N_test': split_results['N_test'].sum()
        }
        summary_rows.append(avg_row)

    overall_avg = {
        'Split_Type': 'Overall (avg)',
        'Holdout': '—',
        'RMSE': results_df['RMSE'].mean(),
        'MAE': results_df['MAE'].mean(),
        'R2': results_df['R2'].mean(),
        'Pearson_r': results_df['Pearson_r'].mean(),
        'Within_5s': results_df['Within_5s'].mean(),
        'Within_10s': results_df['Within_10s'].mean(),
        'Within_30s': results_df['Within_30s'].mean(),
        'N_test': results_df['N_test'].sum()
    }
    summary_rows.append(overall_avg)

    summary_df = pd.concat([results_df, pd.DataFrame(summary_rows)], ignore_index=True)
    return summary_df


def train_final_model(comp_descriptors, feature_names, grad_total_idx, device):
    """
    Trains the final model on the entire dataset.
    """
    print("\n" + "="*70)
    print("TRAINING FINAL MODEL ON ENTIRE DATASET")
    print("="*70)

    full_data_path = DATA_DIR / "retina_dataset.csv"
    df = pd.read_csv(full_data_path)
    print(f"  Full dataset size: {len(df):,}")

    X, y, _ = prepare_data(df, comp_descriptors)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.1,
        random_state=RANDOM_SEED,
        shuffle=True
    )

    train_loader = create_dataloader(X_train, y_train, HYPERPARAMETERS["batch_size"], shuffle=True)
    val_loader = create_dataloader(X_val, y_val, HYPERPARAMETERS["batch_size"], shuffle=False)
    full_loader = create_dataloader(X, y, HYPERPARAMETERS["batch_size"], shuffle=False)

    input_dim = X.shape[1]
    model = RetinaMLP(
        input_dim=input_dim,
        hidden_layers=HYPERPARAMETERS["hidden_layers"],
        dropout_rate=HYPERPARAMETERS["dropout_rate"],
        use_batch_norm=HYPERPARAMETERS["use_batch_norm"],
        use_residual=HYPERPARAMETERS["use_residual"]
    ).to(device)

    if HYPERPARAMETERS["optimizer"].lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=HYPERPARAMETERS["learning_rate"],
            weight_decay=HYPERPARAMETERS["weight_decay"]
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=HYPERPARAMETERS["learning_rate"],
            weight_decay=HYPERPARAMETERS["weight_decay"]
        )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=HYPERPARAMETERS["lr_scheduler_factor"],
        patience=HYPERPARAMETERS["lr_scheduler_patience"],
        threshold=HYPERPARAMETERS["early_stopping_min_delta"],
        threshold_mode='abs'
    )

    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    base_lr = HYPERPARAMETERS["learning_rate"]

    for epoch in range(HYPERPARAMETERS["max_epochs"]):
        warmup_lr(optimizer, base_lr, epoch, HYPERPARAMETERS["lr_warmup_epochs"])

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            HYPERPARAMETERS["grad_clip_norm"]
        )

        val_loss, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        improved = val_loss + HYPERPARAMETERS["early_stopping_min_delta"] < best_val_loss

        if improved:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if epochs_no_improve >= HYPERPARAMETERS["early_stopping_patience"]:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    _, preds, _ = evaluate(model, full_loader, criterion, device)
    method_lengths = np.maximum(X[:, grad_total_idx], 0.0)
    preds = np.clip(preds, 0.0, method_lengths)

    metrics = calculate_metrics(y, preds)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "ReTiNA_MLP1/ReTiNA_MLP1.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\n  Model saved to: {model_path}")

    return model, metrics


def save_outputs(summary_df, all_predictions, final_metrics, n_features):
    """
    Saves the training outputs.
    """
    print("\n" + "="*70)
    print("SAVING OUTPUTS")
    print("="*70)

    PERFORMANCE_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = PERFORMANCE_DIR / "summary_table.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved summary table: {summary_path}")

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    predictions_path = PERFORMANCE_DIR / "all_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"  Saved all predictions: {predictions_path}")
    print(f"    Total predictions: {len(predictions_df):,}")

    json_summary = {
        'model': 'ReTiNA_MLP1',
        'model_type': 'PyTorch MLP',
        'hyperparameters': HYPERPARAMETERS,
        'n_features': n_features,
        'n_samples': len(all_predictions[0]),
        'hardware': {
            'compute_type': 'CPU',
            'cpu': {
                'model': 'Apple M3',
                'architecture': 'ARM64 (Apple Silicon)',
                'performance_cores': 4,
                'efficiency_cores': 4,
                'total_cores': 8
            },
            'ram_gb': 18
        },
        'cross_validation': {
            'scaffold': {},
            'method': {},
            'cluster': {},
            'overall': {}
        }
    }

    for split_key in ['Scaffold (avg)', 'Method (avg)', 'Cluster (avg)', 'Overall (avg)']:
        if split_key in summary_df['Split_Type'].values:
            row = summary_df[summary_df['Split_Type'] == split_key].iloc[0]
            split_lower = split_key.split()[0].lower()
            json_summary['cross_validation'][split_lower] = {
                'RMSE': float(row['RMSE']),
                'MAE': float(row['MAE']),
                'R2': float(row['R2']),
                'Pearson_r': float(row['Pearson_r']),
                'Within_10s': float(row['Within_10s'])
            }

    json_path = PERFORMANCE_DIR / "summary.json"
    with open(json_path, 'w') as f:
        json.dump(json_summary, f, indent=2)
    print(f"  Saved JSON summary: {json_path}")

def print_final_summary(summary_df: pd.DataFrame):
    """
    Prints the final summary table.
    """
    print("\n" + "="*70)
    print("FINAL SUMMARY TABLE")
    print("="*70)

    display_df = summary_df.copy()

    for col in ['RMSE', 'MAE', 'R2', 'Pearson_r']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.3f}" if isinstance(x, (int, float, np.floating)) else x
            )

    if 'Within_10s' in display_df.columns:
        display_df['Within_10s'] = display_df['Within_10s'].apply(
            lambda x: f"{x:.1f}%" if isinstance(x, (int, float, np.floating)) else x
        )

    print(display_df[['Split_Type', 'Holdout', 'RMSE', 'MAE', 'R2', 'Within_10s']].to_string(index=False))
    print("="*70)


# ===== Main ===== #


def main():
    """
    Main execution function.
    """
    set_seed(RANDOM_SEED)

    print("\n" + "="*70)
    print("ReTiNA_MLP1 TRAINING PIPELINE")
    print("="*70)
    print("Hyperparameters:")
    for key, value in HYPERPARAMETERS.items():
        print(f"  {key}: {value}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    all_results, all_predictions, comp_descriptors, feature_names, grad_total_idx = train_all_splits(device)

    summary_df = create_summary_table(all_results)

    _, final_metrics = train_final_model(comp_descriptors, feature_names, grad_total_idx, device)

    save_outputs(summary_df, all_predictions, final_metrics, n_features=len(feature_names))

    print_final_summary(summary_df)

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"All outputs saved to: {PERFORMANCE_DIR}")
    print(f"Final model saved to: {MODELS_DIR / 'ReTiNA_MLP1/ReTiNA_MLP1.pt'}")


if __name__ == "__main__":
    main()


