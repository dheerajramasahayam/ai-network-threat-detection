from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.preprocessing import FEATURE_COLUMNS, _CICIDS_COL_MAP

RANDOM_STATE = 42
FLOAT32_CLIP = np.finfo(np.float32).max / 1024.0

NSL_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty_level",
]


@dataclass
class CanonicalDatasetSplit:
    name: str
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    preprocessor: Pipeline
    feature_names: list[str]
    train_rows: int
    test_rows: int


def _signed_log1p(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))


def _make_preprocessor() -> Pipeline:
    return Pipeline(
        [
            ("signed_log", FunctionTransformer(_signed_log1p, validate=False)),
            ("scaler", StandardScaler()),
        ]
    )


def _sanitize_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return cleaned.clip(lower=-FLOAT32_CLIP, upper=FLOAT32_CLIP)


def _frame_to_array(df: pd.DataFrame) -> np.ndarray:
    cleaned = _sanitize_numeric_frame(df)
    values = cleaned.to_numpy(dtype=np.float64, copy=True)
    return np.nan_to_num(
        values,
        nan=0.0,
        posinf=FLOAT32_CLIP,
        neginf=-FLOAT32_CLIP,
        copy=False,
    )


def _to_canonical(df: pd.DataFrame, label: pd.Series, source: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for column in FEATURE_COLUMNS:
        values = df[column] if column in df.columns else pd.Series(0.0, index=df.index)
        out[column] = pd.to_numeric(values, errors="coerce").fillna(0.0)
    out[FEATURE_COLUMNS] = _sanitize_numeric_frame(out[FEATURE_COLUMNS])
    out["label"] = label.astype(int).values
    out["_source"] = source
    return out


def _load_cicids_canonical(path: str | Path) -> pd.DataFrame:
    usecols = list(_CICIDS_COL_MAP.keys())
    df = pd.read_csv(path, usecols=lambda c: c.strip() in usecols, low_memory=False)
    df.columns = df.columns.str.strip()
    rename_map = {key: value for key, value in _CICIDS_COL_MAP.items() if key in df.columns}
    df = df.rename(columns=rename_map)
    label = (df["Label"].astype(str).str.strip().str.upper() != "BENIGN").astype(int)
    return _to_canonical(df, label, "cicids2017")


def iter_cicids_canonical_chunks(path: str | Path, chunksize: int = 100_000):
    usecols = list(_CICIDS_COL_MAP.keys())
    reader = pd.read_csv(
        path,
        usecols=lambda c: c.strip() in usecols,
        low_memory=False,
        chunksize=chunksize,
    )
    for chunk in reader:
        chunk.columns = chunk.columns.str.strip()
        rename_map = {key: value for key, value in _CICIDS_COL_MAP.items() if key in chunk.columns}
        chunk = chunk.rename(columns=rename_map)
        label = (chunk["Label"].astype(str).str.strip().str.upper() != "BENIGN").astype(int)
        yield _to_canonical(chunk, label, "cicids2017")


def _load_unsw_canonical(path: str | Path) -> pd.DataFrame:
    columns = [
        "dur", "spkts", "dpkts", "sbytes", "dbytes", "sloss", "dloss", "sload",
        "dload", "sinpkt", "dinpkt", "sjit", "djit", "swin", "dwin", "stcpb",
        "dtcpb", "smean", "dmean", "trans_depth", "label",
    ]
    df = pd.read_csv(path, usecols=lambda c: c.strip().lower() in columns, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    col_map = {
        "dur": "flow_duration",
        "spkts": "total_fwd_packets",
        "dpkts": "total_bwd_packets",
        "sbytes": "total_len_fwd_packets",
        "dbytes": "total_len_bwd_packets",
        "sloss": "fwd_psh_flags",
        "dloss": "bwd_psh_flags",
        "sload": "flow_bytes_s",
        "dload": "bwd_packets_s",
        "sinpkt": "flow_iat_mean",
        "dinpkt": "flow_iat_std",
        "sjit": "packet_len_std",
        "djit": "packet_len_variance",
        "swin": "init_win_bytes_fwd",
        "dwin": "init_win_bytes_bwd",
        "stcpb": "fwd_iat_total",
        "dtcpb": "bwd_iat_total",
        "smean": "packet_len_mean",
        "dmean": "avg_packet_size",
        "trans_depth": "act_data_pkt_fwd",
    }
    df = df.rename(columns={key: value for key, value in col_map.items() if key in df.columns})
    label = (df["label"].astype(str).str.strip() != "0").astype(int)
    return _to_canonical(df, label, "unsw_nb15")


def _load_nsl_kdd_canonical(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=NSL_COLUMNS, low_memory=False)
    label = (df["label"].astype(str).str.strip().str.lower() != "normal").astype(int)
    col_map = {
        "duration": "flow_duration",
        "src_bytes": "total_len_fwd_packets",
        "dst_bytes": "total_len_bwd_packets",
        "count": "total_fwd_packets",
        "srv_count": "total_bwd_packets",
        "hot": "fwd_psh_flags",
        "num_failed_logins": "syn_flag_cnt",
        "num_compromised": "rst_flag_cnt",
        "num_root": "ack_flag_cnt",
        "num_shells": "fin_flag_cnt",
        "serror_rate": "flow_iat_mean",
        "rerror_rate": "flow_iat_std",
        "same_srv_rate": "fwd_iat_total",
        "diff_srv_rate": "bwd_iat_total",
        "dst_host_same_srv_rate": "packet_len_mean",
        "dst_host_diff_srv_rate": "packet_len_std",
        "dst_host_serror_rate": "packet_len_variance",
        "dst_host_count": "flow_bytes_s",
        "dst_host_srv_count": "flow_packets_s",
        "wrong_fragment": "fwd_header_len",
        "urgent": "bwd_header_len",
    }
    df = df.rename(columns={key: value for key, value in col_map.items() if key in df.columns})
    return _to_canonical(df, label, "nsl_kdd")


def _split_train_validation(train_df: pd.DataFrame, validation_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_part, val_part = train_test_split(
        train_df,
        test_size=validation_size,
        random_state=random_state,
        stratify=train_df["label"],
    )
    return train_part.reset_index(drop=True), val_part.reset_index(drop=True)


def _build_numeric_split(name: str, train_df: pd.DataFrame, test_df: pd.DataFrame, validation_size: float = 0.15) -> CanonicalDatasetSplit:
    train_part, val_part = _split_train_validation(train_df, validation_size, RANDOM_STATE)
    preprocessor = _make_preprocessor()

    X_train = preprocessor.fit_transform(_frame_to_array(train_part[FEATURE_COLUMNS]))
    X_val = preprocessor.transform(_frame_to_array(val_part[FEATURE_COLUMNS]))
    X_test = preprocessor.transform(_frame_to_array(test_df[FEATURE_COLUMNS]))

    return CanonicalDatasetSplit(
        name=name,
        train_df=train_part,
        val_df=val_part,
        test_df=test_df.reset_index(drop=True),
        X_train=X_train.astype(np.float32),
        X_val=X_val.astype(np.float32),
        X_test=X_test.astype(np.float32),
        y_train=train_part["label"].to_numpy(dtype=np.int32),
        y_val=val_part["label"].to_numpy(dtype=np.int32),
        y_test=test_df["label"].to_numpy(dtype=np.int32),
        preprocessor=preprocessor,
        feature_names=list(FEATURE_COLUMNS),
        train_rows=int(len(train_df)),
        test_rows=int(len(test_df)),
    )


def prepare_official_unsw_split(dataset_dir: str | Path) -> CanonicalDatasetSplit:
    base = Path(dataset_dir)
    train_df = _load_unsw_canonical(base / "UNSW_NB15_training-set.csv")
    test_df = _load_unsw_canonical(base / "UNSW_NB15_testing-set.csv")
    return _build_numeric_split("UNSW-NB15 Official", train_df, test_df)


def prepare_official_nsl_split(dataset_dir: str | Path) -> CanonicalDatasetSplit:
    base = Path(dataset_dir)
    train_df = _load_nsl_kdd_canonical(base / "KDDTrain+.txt")
    test_df = _load_nsl_kdd_canonical(base / "KDDTest+.txt")
    return _build_numeric_split("NSL-KDD Official", train_df, test_df)


def prepare_external_cicids(dataset_path: str | Path, sample_size: int | None = None) -> pd.DataFrame:
    df = _load_cicids_canonical(dataset_path)
    if sample_size and sample_size < len(df):
        attack = df[df["label"] == 1]
        benign = df[df["label"] == 0]
        attack_n = int(round(sample_size * len(attack) / len(df)))
        benign_n = max(1, sample_size - attack_n)
        df = pd.concat(
            [
                attack.sample(n=min(len(attack), attack_n), random_state=RANDOM_STATE),
                benign.sample(n=min(len(benign), benign_n), random_state=RANDOM_STATE),
            ],
            ignore_index=True,
        ).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    return df


def prepare_joint_unsw_nsl_to_cicids(
    unsw_dir: str | Path,
    nsl_dir: str | Path,
    cicids_path: str | Path,
    validation_size: float = 0.15,
    cicids_sample_size: int | None = 200000,
) -> CanonicalDatasetSplit:
    unsw_train = _load_unsw_canonical(Path(unsw_dir) / "UNSW_NB15_training-set.csv")
    nsl_train = _load_nsl_kdd_canonical(Path(nsl_dir) / "KDDTrain+.txt")
    train_df = pd.concat([unsw_train, nsl_train], ignore_index=True)
    test_df = prepare_external_cicids(cicids_path, sample_size=cicids_sample_size)
    return _build_numeric_split("UNSW+NSL -> CICIDS2017", train_df, test_df, validation_size=validation_size)


def dataset_summary(split: CanonicalDatasetSplit) -> dict:
    return {
        "name": split.name,
        "train_rows": split.train_rows,
        "test_rows": split.test_rows,
        "train_distribution": split.train_df["label"].value_counts().sort_index().to_dict(),
        "test_distribution": split.test_df["label"].value_counts().sort_index().to_dict(),
        "feature_count": len(split.feature_names),
        "feature_names": split.feature_names,
    }


def load_official_unsw_frames(dataset_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = Path(dataset_dir)
    train_df = _load_unsw_canonical(base / "UNSW_NB15_training-set.csv")
    test_df = _load_unsw_canonical(base / "UNSW_NB15_testing-set.csv")
    return train_df, test_df
