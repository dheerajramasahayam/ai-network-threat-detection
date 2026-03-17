from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "flow_duration",
    "source_packets",
    "destination_packets",
    "source_bytes",
    "destination_bytes",
    "total_packets",
    "total_bytes",
    "avg_packet_size",
    "source_packet_size_mean",
    "destination_packet_size_mean",
    "bytes_per_second",
    "packets_per_second",
    "source_bytes_per_second",
    "destination_bytes_per_second",
    "source_inter_packet_time",
    "destination_inter_packet_time",
    "tcp_round_trip_time",
    "synack_latency",
    "ack_latency",
    "source_ttl",
    "destination_ttl",
    "source_loss",
    "destination_loss",
    "service_connection_count",
    "destination_connection_count",
    "source_destination_flow_count",
    "source_port_reuse_count",
    "destination_port_reuse_count",
    "same_ip_port_flag",
    "state_ttl_count",
]

CATEGORICAL_FEATURES = ["proto", "service", "state"]


@dataclass
class ResearchDataset:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    class_names: list[str]
    preprocessor: ColumnTransformer
    label_encoder: LabelEncoder
    numeric_features: list[str]
    categorical_features: list[str]


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    safe_denominator = denominator.replace(0, np.nan)
    values = numerator / safe_denominator
    return values.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _stratified_cap(df: pd.DataFrame, label_col: str, max_rows: int | None, random_state: int) -> pd.DataFrame:
    if not max_rows or max_rows >= len(df):
        return df.copy()

    fractions = df[label_col].value_counts(normalize=True)
    counts = df[label_col].value_counts()
    pieces: list[pd.DataFrame] = []
    labels = list(fractions.index)
    minimum_per_class = 4
    reserved = {label: min(minimum_per_class, int(counts[label])) for label in labels}
    remaining_budget = max_rows - sum(reserved.values())
    allocated = 0

    if remaining_budget < 0:
        return df.groupby(label_col, group_keys=False).apply(
            lambda frame: frame.sample(n=1, random_state=random_state)
        ).reset_index(drop=True)

    for idx, label in enumerate(labels):
        subset = df[df[label_col] == label]
        base_take = reserved[label]
        if idx == len(labels) - 1:
            extra_take = max(0, remaining_budget - allocated)
        else:
            extra_take = int(round(fractions[label] * remaining_budget))
        take = max(base_take, min(len(subset), base_take + extra_take))
        allocated += max(0, take - base_take)
        pieces.append(subset.sample(n=take, random_state=random_state))

    sampled = pd.concat(pieces, ignore_index=True)
    if len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=random_state)
    return sampled.reset_index(drop=True)


def load_unsw_nb15(dataset_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_dir = Path(dataset_dir)
    train_path = base_dir / "UNSW_NB15_training-set.csv"
    test_path = base_dir / "UNSW_NB15_testing-set.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df["split"] = "train"
    test_df["split"] = "test"
    return train_df, test_df


def engineer_unsw_features(df: pd.DataFrame) -> pd.DataFrame:
    engineered = df.copy()
    engineered["attack_cat"] = engineered["attack_cat"].fillna("Normal").astype(str).str.strip()

    total_packets = engineered["spkts"] + engineered["dpkts"]
    total_bytes = engineered["sbytes"] + engineered["dbytes"]

    engineered["flow_duration"] = engineered["dur"]
    engineered["source_packets"] = engineered["spkts"]
    engineered["destination_packets"] = engineered["dpkts"]
    engineered["source_bytes"] = engineered["sbytes"]
    engineered["destination_bytes"] = engineered["dbytes"]
    engineered["total_packets"] = total_packets
    engineered["total_bytes"] = total_bytes
    engineered["avg_packet_size"] = _safe_divide(total_bytes, total_packets)
    engineered["source_packet_size_mean"] = engineered["smean"]
    engineered["destination_packet_size_mean"] = engineered["dmean"]
    engineered["bytes_per_second"] = _safe_divide(total_bytes, engineered["dur"])
    engineered["packets_per_second"] = _safe_divide(total_packets, engineered["dur"])
    engineered["source_bytes_per_second"] = _safe_divide(engineered["sbytes"], engineered["dur"])
    engineered["destination_bytes_per_second"] = _safe_divide(engineered["dbytes"], engineered["dur"])
    engineered["source_inter_packet_time"] = engineered["sinpkt"]
    engineered["destination_inter_packet_time"] = engineered["dinpkt"]
    engineered["tcp_round_trip_time"] = engineered["tcprtt"]
    engineered["synack_latency"] = engineered["synack"]
    engineered["ack_latency"] = engineered["ackdat"]
    engineered["source_ttl"] = engineered["sttl"]
    engineered["destination_ttl"] = engineered["dttl"]
    engineered["source_loss"] = engineered["sloss"]
    engineered["destination_loss"] = engineered["dloss"]
    engineered["service_connection_count"] = engineered["ct_srv_src"]
    engineered["destination_connection_count"] = engineered["ct_dst_ltm"]
    engineered["source_destination_flow_count"] = engineered["ct_dst_src_ltm"]
    engineered["source_port_reuse_count"] = engineered["ct_src_dport_ltm"]
    engineered["destination_port_reuse_count"] = engineered["ct_dst_sport_ltm"]
    engineered["same_ip_port_flag"] = engineered["is_sm_ips_ports"]
    engineered["state_ttl_count"] = engineered["ct_state_ttl"]

    engineered[CATEGORICAL_FEATURES] = engineered[CATEGORICAL_FEATURES].fillna("unknown").astype(str)
    engineered[NUMERIC_FEATURES] = (
        engineered[NUMERIC_FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32)
    )
    return engineered


def build_feature_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_frame = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    labels = df["label"].astype(int).copy()
    return feature_frame, labels


def _feature_names(preprocessor: ColumnTransformer) -> list[str]:
    feature_names = preprocessor.get_feature_names_out()
    return [name.replace("num__", "").replace("cat__", "") for name in feature_names]


def prepare_research_dataset(
    dataset_dir: str | Path,
    max_train_rows: int | None = 50000,
    max_test_rows: int | None = 25000,
    validation_size: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> ResearchDataset:
    train_df, test_df = load_unsw_nb15(dataset_dir)
    train_df = engineer_unsw_features(train_df)
    test_df = engineer_unsw_features(test_df)

    train_df = _stratified_cap(train_df, "attack_cat", max_train_rows, random_state)
    test_df = _stratified_cap(test_df, "attack_cat", max_test_rows, random_state)

    train_features, train_labels = build_feature_table(train_df)
    test_features, test_labels = build_feature_table(test_df)

    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(["Benign", "Attack"], dtype=object)

    y_train_full = train_labels.to_numpy(dtype=np.int32)
    y_test = test_labels.to_numpy(dtype=np.int32)

    label_counts = pd.Series(y_train_full).value_counts()
    stratify_labels = y_train_full if int(label_counts.min()) >= 2 else None

    X_train_frame, X_val_frame, y_train, y_val = train_test_split(
        train_features,
        y_train_full,
        test_size=validation_size,
        random_state=random_state,
        stratify=stratify_labels,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    X_train = preprocessor.fit_transform(X_train_frame).astype(np.float32)
    X_val = preprocessor.transform(X_val_frame).astype(np.float32)
    X_test = preprocessor.transform(test_features).astype(np.float32)

    return ResearchDataset(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train.astype(np.int32),
        y_val=y_val.astype(np.int32),
        y_test=y_test.astype(np.int32),
        feature_names=_feature_names(preprocessor),
        class_names=["Benign", "Attack"],
        preprocessor=preprocessor,
        label_encoder=label_encoder,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )


def summarize_unsw_nb15(dataset_dir: str | Path) -> dict:
    train_df, test_df = load_unsw_nb15(dataset_dir)
    combined = pd.concat([train_df, test_df], ignore_index=True)
    class_counts = combined["attack_cat"].fillna("Normal").astype(str).str.strip().value_counts().sort_index()

    return {
        "dataset_name": "UNSW-NB15",
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "total_rows": int(len(combined)),
        "num_classes": int(class_counts.shape[0]),
        "classes": class_counts.to_dict(),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }
