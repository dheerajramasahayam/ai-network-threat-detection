"""
preprocessing.py
----------------
Data preprocessing pipeline for the CICIDS2017 network intrusion dataset.
Handles feature engineering, normalization, encoding, label quality auditing,
SMOTE balancing, and train/test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Canonical 41 feature names (lowercase, underscore-separated).
# Used for both CICIDS2017 and the multi-dataset combined.csv.
FEATURE_COLUMNS = [
    'flow_duration', 'total_fwd_packets', 'total_bwd_packets',
    'total_len_fwd_packets', 'total_len_bwd_packets',
    'fwd_packet_len_max', 'fwd_packet_len_min', 'fwd_packet_len_mean',
    'bwd_packet_len_max', 'bwd_packet_len_min', 'bwd_packet_len_mean',
    'flow_bytes_s', 'flow_packets_s', 'flow_iat_mean', 'flow_iat_std',
    'fwd_iat_total', 'bwd_iat_total', 'fwd_psh_flags', 'bwd_psh_flags',
    'fwd_header_len', 'bwd_header_len', 'fwd_packets_s', 'bwd_packets_s',
    'min_packet_len', 'max_packet_len', 'packet_len_mean',
    'packet_len_std', 'packet_len_variance', 'fin_flag_cnt',
    'syn_flag_cnt', 'rst_flag_cnt', 'psh_flag_cnt', 'ack_flag_cnt',
    'urg_flag_cnt', 'avg_packet_size', 'avg_fwd_seg_size',
    'avg_bwd_seg_size', 'init_win_bytes_fwd', 'init_win_bytes_bwd',
    'act_data_pkt_fwd', 'min_seg_size_fwd',
]
LABEL_COLUMN = 'Label'

# CICIDS2017 original mixed-case column names → canonical names
_CICIDS_COL_MAP = {
    'Flow Duration':                'flow_duration',
    'Total Fwd Packets':            'total_fwd_packets',
    'Total Backward Packets':       'total_bwd_packets',
    'Total Length of Fwd Packets':  'total_len_fwd_packets',
    'Total Length of Bwd Packets':  'total_len_bwd_packets',
    'Fwd Packet Length Max':        'fwd_packet_len_max',
    'Fwd Packet Length Min':        'fwd_packet_len_min',
    'Fwd Packet Length Mean':       'fwd_packet_len_mean',
    'Bwd Packet Length Max':        'bwd_packet_len_max',
    'Bwd Packet Length Min':        'bwd_packet_len_min',
    'Bwd Packet Length Mean':       'bwd_packet_len_mean',
    'Flow Bytes/s':                 'flow_bytes_s',
    'Flow Packets/s':               'flow_packets_s',
    'Flow IAT Mean':                'flow_iat_mean',
    'Flow IAT Std':                 'flow_iat_std',
    'Fwd IAT Total':                'fwd_iat_total',
    'Bwd IAT Total':                'bwd_iat_total',
    'Fwd PSH Flags':                'fwd_psh_flags',
    'Bwd PSH Flags':                'bwd_psh_flags',
    'Fwd Header Length':            'fwd_header_len',
    'Bwd Header Length':            'bwd_header_len',
    'Fwd Packets/s':                'fwd_packets_s',
    'Bwd Packets/s':                'bwd_packets_s',
    'Min Packet Length':            'min_packet_len',
    'Max Packet Length':            'max_packet_len',
    'Packet Length Mean':           'packet_len_mean',
    'Packet Length Std':            'packet_len_std',
    'Packet Length Variance':       'packet_len_variance',
    'FIN Flag Count':               'fin_flag_cnt',
    'SYN Flag Count':               'syn_flag_cnt',
    'RST Flag Count':               'rst_flag_cnt',
    'PSH Flag Count':               'psh_flag_cnt',
    'ACK Flag Count':               'ack_flag_cnt',
    'URG Flag Count':               'urg_flag_cnt',
    'Average Packet Size':          'avg_packet_size',
    'Avg Fwd Segment Size':         'avg_fwd_seg_size',
    'Avg Bwd Segment Size':         'avg_bwd_seg_size',
    'Init_Win_bytes_forward':       'init_win_bytes_fwd',
    'Init_Win_bytes_backward':      'init_win_bytes_bwd',
    'act_data_pkt_fwd':             'act_data_pkt_fwd',
    'min_seg_size_forward':         'min_seg_size_fwd',
    'Label':                        'Label',
}

# CSE-CIC-IDS2018 compact column names → canonical names
_CSE_CIC_IDS2018_COL_MAP = {
    'Flow Duration':           'flow_duration',
    'Tot Fwd Pkts':            'total_fwd_packets',
    'Tot Bwd Pkts':            'total_bwd_packets',
    'TotLen Fwd Pkts':         'total_len_fwd_packets',
    'TotLen Bwd Pkts':         'total_len_bwd_packets',
    'Fwd Pkt Len Max':         'fwd_packet_len_max',
    'Fwd Pkt Len Min':         'fwd_packet_len_min',
    'Fwd Pkt Len Mean':        'fwd_packet_len_mean',
    'Bwd Pkt Len Max':         'bwd_packet_len_max',
    'Bwd Pkt Len Min':         'bwd_packet_len_min',
    'Bwd Pkt Len Mean':        'bwd_packet_len_mean',
    'Flow Byts/s':             'flow_bytes_s',
    'Flow Pkts/s':             'flow_packets_s',
    'Flow IAT Mean':           'flow_iat_mean',
    'Flow IAT Std':            'flow_iat_std',
    'Fwd IAT Tot':             'fwd_iat_total',
    'Bwd IAT Tot':             'bwd_iat_total',
    'Fwd PSH Flags':           'fwd_psh_flags',
    'Bwd PSH Flags':           'bwd_psh_flags',
    'Fwd Header Len':          'fwd_header_len',
    'Bwd Header Len':          'bwd_header_len',
    'Fwd Pkts/s':              'fwd_packets_s',
    'Bwd Pkts/s':              'bwd_packets_s',
    'Pkt Len Min':             'min_packet_len',
    'Pkt Len Max':             'max_packet_len',
    'Pkt Len Mean':            'packet_len_mean',
    'Pkt Len Std':             'packet_len_std',
    'Pkt Len Var':             'packet_len_variance',
    'FIN Flag Cnt':            'fin_flag_cnt',
    'SYN Flag Cnt':            'syn_flag_cnt',
    'RST Flag Cnt':            'rst_flag_cnt',
    'PSH Flag Cnt':            'psh_flag_cnt',
    'ACK Flag Cnt':            'ack_flag_cnt',
    'URG Flag Cnt':            'urg_flag_cnt',
    'Pkt Size Avg':            'avg_packet_size',
    'Fwd Seg Size Avg':        'avg_fwd_seg_size',
    'Bwd Seg Size Avg':        'avg_bwd_seg_size',
    'Init Fwd Win Byts':       'init_win_bytes_fwd',
    'Init Bwd Win Byts':       'init_win_bytes_bwd',
    'Fwd Act Data Pkts':       'act_data_pkt_fwd',
    'Fwd Seg Size Min':        'min_seg_size_fwd',
    'Label':                   'Label',
}


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load a NIDS CSV dataset (CICIDS2017 or combined.csv).
    Automatically renames CICIDS2017 mixed-case columns to the canonical
    lowercase-underscore schema so both file formats share one pipeline.
    """
    logger.info(f"Loading dataset from {path}")
    df = pd.read_csv(path, low_memory=False)
    # Strip whitespace from column names (CICIDS2017 has trailing spaces)
    df.columns = df.columns.str.strip()
    # Rename CICIDS2017-style columns → canonical names (no-op for combined.csv)
    rename_map = {k: v for k, v in _CICIDS_COL_MAP.items() if k in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)
        logger.info(f"Normalized {len(rename_map)} CICIDS2017 column names to canonical form.")
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove infinite values and NaNs."""
    logger.info("Cleaning data (inf/NaN removal)...")
    df = df.replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna()
    logger.info(f"Removed {before - len(df):,} rows with NaN/inf values")
    return df


def audit_labels(df: pd.DataFrame, contamination: float = 0.01) -> dict:
    """
    Label Quality Audit — addresses CICIDS2017's known mislabeling problem.

    Uses Isolation Forest to detect statistical outliers within each class.
    Rows that are outliers in their own class distribution are likely mislabeled.
    Research shows up to 50% mislabeling in some CICIDS2017 attack categories.

    Parameters
    ----------
    df            : DataFrame with 'Label_encoded' column already present
    contamination : expected fraction of mislabeled samples (default 1 %)

    Returns
    -------
    dict with keys:
        flagged_indices  - row indices suspected of mislabeling
        flag_rate_pct    - percentage of dataset flagged
        per_class_flags  - {class_label: count_flagged}
    """
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError:
        logger.warning("scikit-learn IsolationForest not available, skipping label audit.")
        return {}

    logger.info(f"Running label quality audit (contamination={contamination}) ...")
    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    flagged = []
    per_class = {}

    for cls in df['Label_encoded'].unique():
        subset = df[df['Label_encoded'] == cls]
        if len(subset) < 50:
            continue
        iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        preds = iso.fit_predict(subset[available])
        class_flagged = subset.index[preds == -1].tolist()
        per_class[int(cls)] = len(class_flagged)
        flagged.extend(class_flagged)

    flag_rate = round(len(flagged) / len(df) * 100, 2)
    logger.info(
        f"Label audit: flagged {len(flagged):,} rows ({flag_rate}% of dataset) "
        f"as potentially mislabeled — per class: {per_class}"
    )
    return {
        'flagged_indices': flagged,
        'flag_rate_pct':   flag_rate,
        'per_class_flags': per_class,
    }


def encode_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    """
    Encode the Label column to binary (0 = BENIGN, 1 = ATTACK).
    Handles both string labels ('BENIGN'/'ATTACK') and pre-encoded numeric labels (0/1).
    """
    logger.info("Encoding labels...")
    le = LabelEncoder()
    df = df.copy()
    col = df[LABEL_COLUMN]

    if pd.api.types.is_numeric_dtype(col):
        # Already numeric (e.g. combined.csv) — just ensure 0/1
        df['Label_encoded'] = (col.fillna(0).astype(int) != 0).astype(int)
    else:
        # String labels — compare to 'BENIGN'
        df['Label_encoded'] = (col.astype(str).str.strip().str.upper() != 'BENIGN').astype(int)

    label_distribution = df['Label_encoded'].value_counts()
    logger.info(f"Label distribution:\n{label_distribution}")
    return df, le


def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Select available feature columns and the encoded label.

    Handles three naming conventions:
      - CICIDS2017 mixed-case with spaces: 'Flow Duration'
      - combined.csv canonical lowercase with underscores: 'flow_duration'
      - Direct match: 'act_data_pkt_fwd'
    """
    def _normalize(s: str) -> str:
        return s.lower().replace(' ', '_').replace('/', '_').replace('-', '_')

    # Build normalize(col) → actual_col_name map for the DataFrame
    col_norm_map = {_normalize(c): c for c in df.columns}

    selected = []
    for feat in FEATURE_COLUMNS:
        if feat in df.columns:
            selected.append(feat)
        else:
            norm = _normalize(feat)
            if norm in col_norm_map:
                selected.append(col_norm_map[norm])

    missing = len(FEATURE_COLUMNS) - len(selected)
    if missing:
        logger.warning(f"Missing {missing} expected feature columns — they will be skipped.")
    X = df[selected]
    y = df['Label_encoded']
    logger.info(f"Selected {len(selected)} features.")
    return X, y


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit StandardScaler on training data and transform both splits."""
    logger.info("Scaling features with StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def balance_classes(X: np.ndarray, y, strategy: str = 'smote') -> tuple[np.ndarray, np.ndarray]:
    """
    Balance classes via SMOTE, undersampling, or oversampling.

    Strategies
    ----------
    'smote'       : Synthetic Minority Oversampling Technique (best accuracy)
    'undersample' : Random undersample majority to match minority
    'oversample'  : Random oversample minority to match majority

    SMOTE generates synthetic attack samples that address CICIDS2017's severe
    class imbalance without information loss, enabling ~1-2% accuracy gain
    over random undersampling.
    """
    logger.info(f"Balancing classes using: {strategy} ...")
    y = np.asarray(y)

    if strategy == 'smote':
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            X_bal, y_bal = smote.fit_resample(X, y)
            logger.info(f"SMOTE balanced dataset: {X_bal.shape[0]:,} samples")
            return X_bal, y_bal
        except ImportError:
            logger.warning("imbalanced-learn not installed; falling back to 'undersample'.")
            strategy = 'undersample'

    df_bal = pd.DataFrame(X)
    df_bal['_label'] = y
    majority = df_bal[df_bal['_label'] == 0]
    minority = df_bal[df_bal['_label'] == 1]

    if strategy == 'undersample':
        majority_sampled = resample(majority, replace=False,
                                    n_samples=len(minority), random_state=42)
        balanced = pd.concat([majority_sampled, minority])
    else:  # oversample
        minority_sampled = resample(minority, replace=True,
                                    n_samples=len(majority), random_state=42)
        balanced = pd.concat([majority, minority_sampled])

    balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    y_bal = balanced['_label'].values
    X_bal = balanced.drop('_label', axis=1).values
    logger.info(f"Balanced dataset size: {len(X_bal):,} samples")
    return X_bal, y_bal


def preprocess(
    dataset_path: str,
    test_size: float = 0.2,
    balance: bool = True,
    balance_strategy: str = 'smote',
    run_label_audit: bool = False,
    drop_flagged: bool = False,
    label_col_override: str | None = None,
) -> dict:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    dataset_path      : path to cicids2017.csv or combined.csv
    test_size         : fraction for test split (default 0.2)
    balance           : whether to apply class balancing
    balance_strategy  : 'smote' | 'undersample' | 'oversample'
    run_label_audit   : run IsolationForest label quality check
    drop_flagged      : remove audit-flagged rows before training
    label_col_override: use a different column as the label (e.g. 'label'
                        for combined.csv instead of 'Label')

    Returns a dict with keys:
        X_train, X_test, y_train, y_test, scaler, feature_names, audit_result
    """
    global LABEL_COLUMN
    orig_label_col = LABEL_COLUMN
    if label_col_override:
        LABEL_COLUMN = label_col_override

    df = load_dataset(dataset_path)

    # Drop non-feature metadata columns that combined.csv may include
    for drop_col in ['_source']:
        if drop_col in df.columns:
            df = df.drop(columns=[drop_col])

    df = clean_data(df)
    df, le = encode_labels(df)

    audit_result = {}
    if run_label_audit:
        audit_result = audit_labels(df)
        if drop_flagged and audit_result.get('flagged_indices'):
            before = len(df)
            df = df.drop(index=audit_result['flagged_indices'], errors='ignore')
            logger.info(f"Dropped {before - len(df):,} flagged rows after label audit.")

    X, y = select_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    logger.info(f"Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")

    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)

    if balance:
        X_train_s, y_train = balance_classes(X_train_s, y_train, strategy=balance_strategy)

    # Restore global
    LABEL_COLUMN = orig_label_col

    return {
        'X_train':       X_train_s,
        'X_test':        X_test_s,
        'y_train':       y_train if isinstance(y_train, np.ndarray) else y_train.values,
        'y_test':        y_test.values,
        'scaler':        scaler,
        'feature_names': list(X.columns),
        'audit_result':  audit_result,
    }


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = preprocess(
        os.path.join(BASE_DIR, 'dataset', 'cicids2017.csv'),
        balance_strategy='smote',
        run_label_audit=True,
    )
    print("Preprocessing complete.")
    print(f"  X_train shape : {data['X_train'].shape}")
    print(f"  X_test  shape : {data['X_test'].shape}")
    if data['audit_result']:
        print(f"  Label audit   : {data['audit_result']['flag_rate_pct']}% flagged")
