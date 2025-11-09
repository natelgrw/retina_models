#!/usr/bin/env python3
"""
train_ReTINA_XGB1.py

Trains a XGBoost model on the ReTiNA dataset with cross-validation on scaffold, 
method, and cluster splits using ReTINA encoder features.
"""

import os
import sys
import json
import warnings
import re
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path

warnings.filterwarnings('ignore')


# ===== Configuration ===== #


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
PERFORMANCE_DIR = PROJECT_ROOT / "performance" / "ReTINA_XGB1"
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
    "n_estimators": 1000,
    "max_depth": 9,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "device": "cpu",
    "random_state": RANDOM_SEED,
    "verbosity": 1
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

GRAD_TOTAL_TIME_FEATURE = "grad_total_time"


# ===== Helper Functions ===== #


def _sanitize_feature_name(name: str, existing: set) -> str:
    """
    Sanitizes a feature name to be a valid Python variable name.
    """
    sanitized = re.sub(r"[^0-9A-Za-z_]", "_", name)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    if not sanitized:
        sanitized = "feature"
    base = sanitized
    counter = 1
    while sanitized in existing:
        counter += 1
        sanitized = f"{base}_{counter}"
    existing.add(sanitized)
    return sanitized


def build_feature_names(descriptor_cols):
    """
    Constructs feature names matching the encoding order.
    """
    feature_names = []
    seen = set()

    def add(raw_name: str):
        feature_names.append(_sanitize_feature_name(raw_name, seen))

    # compound descriptors
    for col in descriptor_cols:
        add(f"comp_{col}")

    # solvents
    for phase in ['A', 'B']:
        for solvent in SOLVENTS_ORDER:
            add(f"solv_{solvent}_{phase}_pct")
        for additive in ADDITIVES_ORDER:
            add_short = additive[:20] if len(additive) > 20 else additive
            add(f"add_{add_short}_{phase}_M")

    # gradient
    for i in range(100):
        add(f"grad_t{i:03d}")
    add(GRAD_TOTAL_TIME_FEATURE)

    # column
    for name in ['col_RP', 'col_HI', 'col_diam_mm', 'col_len_mm', 'col_part_um']:
        add(name)

    # flow rate and temperature
    add('flow_rate_mL_min')
    add('temp_C')

    return feature_names

def encode_single_sample(row, comp_descriptors):
    """
    Encodes a single data sample into a feature vector.
    """
    features = []
    
    # compound descriptors (156 features)
    if row['compound'] in comp_descriptors:
        comp_features = comp_descriptors[row['compound']]
        features.extend(comp_features)
    else:
        features.extend([0.0] * 156)
    
    # solvents (28 features)
    solvent_features = encode_solvents(row['solvents'])
    features.extend(solvent_features)
    
    # gradient profile and duration (101 features)
    gradient_features, gradient_total_time = normalize_gradient(row['gradient'])
    features.extend(gradient_features)
    features.append(gradient_total_time)
    
    # column (5 features)
    column_features = encode_column(row['column'])
    features.extend(column_features)
    
    # flow rate (1 feature)
    features.append(float(row['flow_rate']))
    
    # temperature (1 feature)
    features.append(float(row['temp']))
    
    return np.array(features)


def prepare_data(df, comp_descriptors):
    """
    Prepares features and target for training.
    """
    print(f"    Encoding {len(df):,} samples...")
    
    X_list = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        try:
            features = encode_single_sample(row, comp_descriptors)
            X_list.append(features)
            valid_indices.append(idx)
        except Exception as e:
            continue
    
    X = np.array(X_list)
    y = df.loc[valid_indices, 'rt'].values
    
    print(f"Successfully encoded: {len(X):,} samples with {X.shape[1]} features")
    
    return X, y, df.loc[valid_indices]


def calculate_metrics(y_true, y_pred):
    """
    Calculates regression metrics.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson_r, _ = stats.pearsonr(y_true, y_pred)
    
    errors = np.abs(y_true - y_pred)
    within_5s = (errors <= 5).mean() * 100
    within_10s = (errors <= 10).mean() * 100
    within_30s = (errors <= 30).mean() * 100
    
    n_test = len(y_true)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Pearson_r': pearson_r,
        'Within_5s': within_5s,
        'Within_10s': within_10s,
        'Within_30s': within_30s,
        'N_test': n_test
    }


def train_single_split(split_type, holdout_id, comp_descriptors, feature_names, grad_total_idx):
    """
    Trains a model on a single train/test split.
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
    
    X_train, y_train, train_merged = prepare_data(train_df, comp_descriptors)
    X_test, y_test, test_merged = prepare_data(test_df, comp_descriptors)
    
    # training model
    print(f"  Training XGBoost model...")
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)
    
    evallist = [(dtrain, 'train'), (dtest, 'test')]
    
    model = xgb.train(
        HYPERPARAMETERS,
        dtrain,
        num_boost_round=HYPERPARAMETERS['n_estimators'],
        evals=evallist,
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    y_pred = model.predict(dtest)

    method_lengths_test = np.maximum(X_test[:, grad_total_idx], 0.0)
    y_pred = np.clip(y_pred, 0.0, method_lengths_test)
    
    metrics = calculate_metrics(y_test, y_pred)
    
    print(f"  Results:")
    print(f"    RMSE: {metrics['RMSE']:.3f}s")
    print(f"    MAE: {metrics['MAE']:.3f}s")
    print(f"    R²: {metrics['R2']:.4f}")
    print(f"    Pearson r: {metrics['Pearson_r']:.4f}")
    print(f"    Within 10s: {metrics['Within_10s']:.1f}%")
    
    predictions_df = test_merged[['compound']].copy()
    predictions_df['split_type'] = split_type
    predictions_df['holdout_id'] = holdout_id
    predictions_df['rt_true'] = y_test
    predictions_df['rt_pred'] = y_pred
    predictions_df['method_number'] = test_merged['method_number'].values
    predictions_df['source'] = test_merged['source'].values
    
    importance_gain = model.get_score(importance_type='gain')
    sorted_importance = sorted(importance_gain.items(), key=lambda item: item[1], reverse=True)
    top_importance = [
        {
            'feature': feature,
            'importance': float(score),
            'split_type': split_type,
            'holdout_id': holdout_id,
            'rank': rank + 1
        }
        for rank, (feature, score) in enumerate(sorted_importance[:10])
    ]

    return {
        'split_type': split_type,
        'holdout_id': holdout_id,
        'metrics': metrics,
        'predictions': predictions_df,
        'model': model
    }


def train_all_splits():
    """
    Trains models on all splits and collects results.
    """
    print("\n" + "="*70)
    print("STARTING CROSS-VALIDATION TRAINING")
    print("="*70)
    
    print("\nLoading compound descriptors...")
    comp_descriptors, descriptor_cols = load_compound_descriptors()
    feature_names = build_feature_names(descriptor_cols)
    grad_total_idx = feature_names.index(GRAD_TOTAL_TIME_FEATURE)
    
    all_results = []
    all_predictions = []
    
    for split_type in ['scaffold', 'method', 'cluster']:
        for holdout_id in range(1, 6):
            result = train_single_split(
                split_type,
                holdout_id,
                comp_descriptors,
                feature_names,
                grad_total_idx
            )
            
            all_results.append({
                'Split_Type': split_type,
                'Holdout': holdout_id,
                **result['metrics']
            })
            
            all_predictions.append(result['predictions'])

    return all_results, all_predictions, comp_descriptors, feature_names, grad_total_idx


def create_summary_table(all_results):
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


def train_final_model(comp_descriptors, feature_names, grad_total_idx):
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
    
    print(f"  Training final XGBoost model...")
    dtrain = xgb.DMatrix(X, label=y, feature_names=feature_names)
    
    model = xgb.train(
        HYPERPARAMETERS,
        dtrain,
        num_boost_round=HYPERPARAMETERS['n_estimators'],
        evals=[(dtrain, 'train')],
        verbose_eval=False
    )
    
    y_pred = model.predict(dtrain)
    method_lengths_train = np.maximum(X[:, grad_total_idx], 0.0)
    y_pred = np.clip(y_pred, 0.0, method_lengths_train)
    metrics = calculate_metrics(y, y_pred)
    
    print(f"  Training performance:")
    print(f"    RMSE: {metrics['RMSE']:.3f}s")
    print(f"    MAE: {metrics['MAE']:.3f}s")
    print(f"    R²: {metrics['R2']:.4f}")
    print(f"    Pearson r: {metrics['Pearson_r']:.4f}")
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "ReTiNA_XGB1/ReTiNA_XGB1.json"
    model.save_model(str(model_path))
    print(f"\n  Model saved to: {model_path}")
    
    return model, metrics


def save_outputs(summary_df, all_predictions, final_metrics):
    """
    Saves all output files.
    """
    print("\n" + "="*70)
    print("SAVING OUTPUTS")
    print("="*70)
    
    PERFORMANCE_DIR.mkdir(parents=True, exist_ok=True)
    
    summary_path = PERFORMANCE_DIR / "summary_table.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary table: {summary_path}")
    
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    predictions_path = PERFORMANCE_DIR / "all_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Saved all predictions: {predictions_path}")
    print(f"Total predictions: {len(predictions_df):,}")
    
    json_summary = {
        'model': 'ReTiNA_XGB1',
        'model_type': 'XGBoost',
        'hyperparameters': HYPERPARAMETERS,
        'n_features': 292,
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
            'scaffold': {
                'RMSE': float(summary_df[summary_df['Split_Type'] == 'Scaffold (avg)']['RMSE'].values[0]),
                'MAE': float(summary_df[summary_df['Split_Type'] == 'Scaffold (avg)']['MAE'].values[0]),
                'R2': float(summary_df[summary_df['Split_Type'] == 'Scaffold (avg)']['R2'].values[0]),
                'Pearson_r': float(summary_df[summary_df['Split_Type'] == 'Scaffold (avg)']['Pearson_r'].values[0]),
                'Within_10s': float(summary_df[summary_df['Split_Type'] == 'Scaffold (avg)']['Within_10s'].values[0])
            },
            'method': {
                'RMSE': float(summary_df[summary_df['Split_Type'] == 'Method (avg)']['RMSE'].values[0]),
                'MAE': float(summary_df[summary_df['Split_Type'] == 'Method (avg)']['MAE'].values[0]),
                'R2': float(summary_df[summary_df['Split_Type'] == 'Method (avg)']['R2'].values[0]),
                'Pearson_r': float(summary_df[summary_df['Split_Type'] == 'Method (avg)']['Pearson_r'].values[0]),
                'Within_10s': float(summary_df[summary_df['Split_Type'] == 'Method (avg)']['Within_10s'].values[0])
            },
            'cluster': {
                'RMSE': float(summary_df[summary_df['Split_Type'] == 'Cluster (avg)']['RMSE'].values[0]),
                'MAE': float(summary_df[summary_df['Split_Type'] == 'Cluster (avg)']['MAE'].values[0]),
                'R2': float(summary_df[summary_df['Split_Type'] == 'Cluster (avg)']['R2'].values[0]),
                'Pearson_r': float(summary_df[summary_df['Split_Type'] == 'Cluster (avg)']['Pearson_r'].values[0]),
                'Within_10s': float(summary_df[summary_df['Split_Type'] == 'Cluster (avg)']['Within_10s'].values[0])
            },
            'overall': {
                'RMSE': float(summary_df[summary_df['Split_Type'] == 'Overall (avg)']['RMSE'].values[0]),
                'MAE': float(summary_df[summary_df['Split_Type'] == 'Overall (avg)']['MAE'].values[0]),
                'R2': float(summary_df[summary_df['Split_Type'] == 'Overall (avg)']['R2'].values[0]),
                'Pearson_r': float(summary_df[summary_df['Split_Type'] == 'Overall (avg)']['Pearson_r'].values[0]),
                'Within_10s': float(summary_df[summary_df['Split_Type'] == 'Overall (avg)']['Within_10s'].values[0])
            }
        }
    }
    
    json_path = PERFORMANCE_DIR / "summary.json"
    with open(json_path, 'w') as f:
        json.dump(json_summary, f, indent=2)
    print(f"Saved JSON summary: {json_path}")


def print_final_summary(summary_df):
    """
    Prints final summary table to console.
    """
    print("\n" + "="*70)
    print("FINAL SUMMARY TABLE")
    print("="*70)
    
    display_df = summary_df.copy()
    display_df['RMSE'] = display_df['RMSE'].apply(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x)
    display_df['MAE'] = display_df['MAE'].apply(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x)
    display_df['R2'] = display_df['R2'].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
    display_df['Pearson_r'] = display_df['Pearson_r'].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
    display_df['Within_10s'] = display_df['Within_10s'].apply(lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else x)
    
    print(display_df[['Split_Type', 'Holdout', 'RMSE', 'MAE', 'R2', 'Within_10s']].to_string(index=False))
    print("="*70)


def main():
    """
    Main training pipeline.
    """
    print("\n" + "="*70)
    print("ReTiNA_XGB1 TRAINING PIPELINE")
    print("="*70)
    print(f"Hyperparameters:")
    for key, value in HYPERPARAMETERS.items():
        print(f"  {key}: {value}")
    
    all_results, all_predictions, comp_descriptors, feature_names, grad_total_idx = train_all_splits()
    
    summary_df = create_summary_table(all_results)
    
    final_model, final_metrics = train_final_model(comp_descriptors, feature_names, grad_total_idx)
    
    save_outputs(summary_df, all_predictions, final_metrics)
    
    print_final_summary(summary_df)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"All outputs saved to: {PERFORMANCE_DIR}")
    print(f"Final model saved to: {MODELS_DIR / 'ReTiNA_XGB1/ReTiNA_XGB1.json'}")


if __name__ == "__main__":
    main()
