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

# Features used from CICIDS2017 (subset of most informative columns)
FEATURE_COLUMNS = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
    'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std',
    'Fwd IAT Total', 'Bwd IAT Total', 'Fwd PSH Flags', 'Bwd PSH Flags',
    'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
    'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
    'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
    'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
    'URG Flag Count', 'Average Packet Size', 'Avg Fwd Segment Size',
    'Avg Bwd Segment Size', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
    'act_data_pkt_fwd', 'min_seg_size_forward',
]
LABEL_COLUMN = 'Label'


def load_dataset(path: str) -> pd.DataFrame:
    """Load the CICIDS2017 CSV dataset."""
    logger.info(f"Loading dataset from {path}")
    df = pd.read_csv(path, low_memory=False)
    # Strip whitespace from column names (CICIDS2017 has trailing spaces)
    df.columns = df.columns.str.strip()
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
    Encode the Label column to binary (0 = BENIGN, 1 = ATTACK)
    and return the dataframe with a numeric label column.
    """
    logger.info("Encoding labels...")
    le = LabelEncoder()
    df = df.copy()
    df['Label_encoded'] = (df[LABEL_COLUMN].str.strip().str.upper() != 'BENIGN').astype(int)
    label_distribution = df['Label_encoded'].value_counts()
    logger.info(f"Label distribution:\n{label_distribution}")
    return df, le


def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select available feature columns and the encoded label."""
    available = [col for col in FEATURE_COLUMNS if col in df.columns]
    missing = set(FEATURE_COLUMNS) - set(available)
    if missing:
        logger.warning(f"Missing {len(missing)} expected feature columns — they will be skipped.")
    X = df[available]
    y = df['Label_encoded']
    logger.info(f"Selected {len(available)} features.")
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
) -> dict:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    dataset_path      : path to cicids2017.csv
    test_size         : fraction for test split (default 0.2)
    balance           : whether to apply class balancing
    balance_strategy  : 'smote' | 'undersample' | 'oversample'
    run_label_audit   : run IsolationForest label quality check
    drop_flagged      : remove audit-flagged rows before training

    Returns a dict with keys:
        X_train, X_test, y_train, y_test, scaler, feature_names, audit_result
    """
    df = load_dataset(dataset_path)
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
