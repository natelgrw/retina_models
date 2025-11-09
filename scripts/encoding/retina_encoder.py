#!/usr/bin/env python3
"""
retina_encoder.py

Author: natelgrw
Created: 11/09/2025

Encodes retina_dataset.csv into feature vectors for ML modeling.

Feature breakdown:
- Compound: 156 features
- Solvents: 28 features
- Gradient profile: 100 features
- Gradient duration: 1 features
- Column: 5 features
- Flow rate: 1 features
- Temperature: 1 features
"""

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from ast import literal_eval
from pathlib import Path
from tqdm import tqdm


# ===== Configuration ===== #


PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
COMPOUNDS_DIR = DATA_DIR / "compounds"
SOLVENTS_DIR = DATA_DIR / "solvents"

DATASET_CSV = DATA_DIR / "retina_dataset.csv"
COMP_DESC_CSV = COMPOUNDS_DIR / "comp_descriptors.csv"
SOLV_SMI = SOLVENTS_DIR / "solv.smi"
ADDITIVES_TXT = DATA_DIR / "unique_additives.txt"

OUTPUT_CSV = DATA_DIR / "retina_dataset_encoded.csv"

SOLVENTS_ORDER = ['O', 'CC#N', 'CO', 'CC(O)C', 'CC(C)O', 'CC(=O)C']  # mol1-6

ADDITIVES_ORDER = [
    'C(=O)(C(F)(F)F)O',
    'C(=O)C',
    'C(=O)O',
    'C(=O)O.[NH4+]',
    'C(=O)[O-].[NH4+]',
    'C(CN(CC(=O)O)CC(=O)O)N(CC(=O)O)CC(=O)O',
    'CC(=O)O',
    'CC(=O)[O-].[NH4+]'
]

COLUMN_TYPES = ['RP', 'HI']


# ===== Helper Functions ===== #


def load_compound_descriptors():
    """
    Load compound descriptors (excluding Morgan fingerprints).
    """
    print("Loading compound descriptors...")
    df = pd.read_csv(COMP_DESC_CSV)
    
    descriptor_cols = [col for col in df.columns 
                      if col != 'smiles' and not col.lower().startswith('morgan')]
    
    print(f"  Using {len(descriptor_cols)} descriptors per compound")
    print(f"  Total compounds: {len(df):,}")
    
    descriptors_dict = {}
    for _, row in df.iterrows():
        descriptors_dict[row['smiles']] = row[descriptor_cols].values
    
    return descriptors_dict, descriptor_cols


def encode_solvents(solvents_str):
    """
    Encode solvents into 28 features.
    
    Returns:
        12 features: 6 solvents x 2 phases (A, B) x percentage
        16 features: 8 additives x 2 phases (A, B) x molarity
    """
    try:
        solvents_dict = literal_eval(solvents_str)
    except:
        return np.zeros(28)
    
    features = []
    
    for phase in ['A', 'B']:
        solv_percentages = np.zeros(6)
        additive_molarities = np.zeros(8)
        
        if phase in solvents_dict and isinstance(solvents_dict[phase], list):
            phase_data = solvents_dict[phase]
            
            # solvent percentages
            if len(phase_data) > 0 and isinstance(phase_data[0], dict):
                solvent_dict = phase_data[0]
                for solvent_smiles, percentage in solvent_dict.items():
                    if solvent_smiles in SOLVENTS_ORDER:
                        idx = SOLVENTS_ORDER.index(solvent_smiles)
                        solv_percentages[idx] = float(percentage)
            
            # additive molarities
            if len(phase_data) > 1 and isinstance(phase_data[1], dict):
                additive_dict = phase_data[1]
                for additive_smiles, molarity in additive_dict.items():
                    if additive_smiles in ADDITIVES_ORDER:
                        idx = ADDITIVES_ORDER.index(additive_smiles)
                        additive_molarities[idx] = float(molarity)
        
        # adding to features
        features.extend(solv_percentages)
        features.extend(additive_molarities)
    
    return np.array(features)


def normalize_gradient(gradient_str, n_points=100):
    """
    Encodes the gradient profile and duration (in seconds).
    
    Steps:
    1. Parse gradient list of tuples (time, %B)
    2. Normalize time to [0, 1]
    3. Interpolate to 100 uniform points
    
    Returns:
        gradient_vector (np.ndarray): 100-dimensional vector of %B values
        total_time_seconds (float): total method duration in seconds
    """
    try:
        gradient_list = literal_eval(gradient_str)
    except Exception:
        return np.zeros(n_points), 0.0

    if not gradient_list or len(gradient_list) < 2:
        return np.zeros(n_points), 0.0

    # extracting times and % B values
    try:
        times = np.array([float(point[0]) for point in gradient_list], dtype=float)
        percent_b = np.array([float(point[1]) for point in gradient_list], dtype=float)
    except (TypeError, ValueError):
        return np.zeros(n_points), 0.0

    try:
        total_time_minutes = float(gradient_list[-1][0])
    except (TypeError, ValueError, IndexError):
        total_time_minutes = 0.0

    if total_time_minutes <= 0:
        return np.zeros(n_points), 0.0

    # normalizing times to [0, 1]
    times_normalized = times / total_time_minutes

    interp_func = interp1d(times_normalized, percent_b,
                           kind='linear',
                           bounds_error=False,
                           fill_value=(percent_b[0], percent_b[-1]))

    # sampling at 100 uniform points
    t_uniform = np.linspace(0, 1, n_points)
    gradient_vector = interp_func(t_uniform)

    total_time_seconds = total_time_minutes * 60.0

    return gradient_vector, total_time_seconds


def encode_column(column_str):
    """
    Encodes the column into 5 features.
    
    Returns:
        2 features: one-hot encoding for type (RP, HI)
        3 features: diameter (mm), length (mm), particle size (µm)
    """
    try:
        column_tuple = literal_eval(column_str)
    except:
        return np.zeros(5)
    
    if not isinstance(column_tuple, tuple) or len(column_tuple) != 4:
        return np.zeros(5)
    
    features = []
    
    # one-hot encoding column type
    col_type = column_tuple[0]
    features.append(1.0 if col_type == 'RP' else 0.0)
    features.append(1.0 if col_type == 'HI' else 0.0)
    
    features.append(float(column_tuple[1]))
    features.append(float(column_tuple[2]))
    features.append(float(column_tuple[3]))
    
    return np.array(features)


def encode_dataset():
    """
    Encodes the dataset into feature vectors.
    """
    print("=" * 80)
    print(" " * 25 + "RETINA ENCODER")
    print("=" * 80)
    
    comp_descriptors, descriptor_cols = load_compound_descriptors()
    
    print(f"\nLoading dataset from: {DATASET_CSV}")
    df = pd.read_csv(DATASET_CSV)
    print(f"  Loaded {len(df):,} datapoints")
    
    encoded_data = []
    skipped_count = 0
    
    print(f"\nEncoding features...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Encoding"):
        try:
            features = []
            
            # compound descriptors (156 features)
            compound = row['compound']
            if compound in comp_descriptors:
                comp_features = comp_descriptors[compound]
                features.extend(comp_features)
            else:
                skipped_count += 1
                continue
            
            # solvents (28 features)
            solvent_features = encode_solvents(row['solvents'])
            features.extend(solvent_features)
            
            # gradient and duration (101 features)
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
            
            rt = float(row['rt'])
            
            encoded_data.append({
                'features': features,
                'rt': rt,
                'compound': compound,
                'source': row['source'],
                'method_number': row['method_number']
            })
            
        except Exception as e:
            print(f"\n  Warning: Error encoding row {idx}: {e}")
            skipped_count += 1
    
    print(f"\n\nEncoding complete!")
    print(f"  Successfully encoded: {len(encoded_data):,} datapoints")
    print(f"  Skipped: {skipped_count:,} datapoints")
    
    feature_names = []
    
    feature_names.extend([f"comp_{col}" for col in descriptor_cols])
    
    for phase in ['A', 'B']:
        for solvent in SOLVENTS_ORDER:
            feature_names.append(f"solv_{solvent}_{phase}_pct")
        for additive in ADDITIVES_ORDER:
            add_short = additive[:20] if len(additive) > 20 else additive
            feature_names.append(f"add_{add_short}_{phase}_M")
    
    feature_names.extend([f"grad_t{i:03d}" for i in range(100)])
    feature_names.append("grad_total_time")
    
    feature_names.extend(['col_RP', 'col_HI', 'col_diam_mm', 'col_len_mm', 'col_part_um'])
    
    feature_names.extend(['flow_rate_mL_min', 'temp_C'])
    
    print(f"\n  Total features per datapoint: {len(feature_names)}")
    
    print(f"\nCreating encoded DataFrame...")
    
    X = np.array([d['features'] for d in encoded_data])
    y = np.array([d['rt'] for d in encoded_data])
    
    df_encoded = pd.DataFrame(X, columns=feature_names)
    
    df_encoded['rt'] = y
    df_encoded['compound'] = [d['compound'] for d in encoded_data]
    df_encoded['source'] = [d['source'] for d in encoded_data]
    df_encoded['method_number'] = [d['method_number'] for d in encoded_data]
    
    print(f"\nSaving encoded dataset to: {OUTPUT_CSV}")
    df_encoded.to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved {len(df_encoded):,} rows × {len(df_encoded.columns):,} columns")
    
    print("\n" + "=" * 80)
    print("ENCODING SUMMARY")
    print("=" * 80)
    print(f"Feature breakdown:")
    print(f"  Compound descriptors:  156 features")
    print(f"  Solvents:              28 features (12 solvents + 16 additives)")
    print(f"  Gradient profile:      100 features")
    print(f"  Gradient duration:     1 feature")
    print(f"  Column:                5 features")
    print(f"  Flow rate:             1 feature")
    print(f"  Temperature:           1 feature")
    print(f"  Total:                 {len(feature_names)} features")
    print(f"\nTarget variable: rt (retention time)")
    print(f"Metadata: compound, source, method_number")
    print("=" * 80)
    
    return df_encoded


# ===== Main ===== #


if __name__ == "__main__":
    encoded_df = encode_dataset()
    print(f"\nEncoding complete! Dataset saved to:")
    print(f"  {OUTPUT_CSV}")

