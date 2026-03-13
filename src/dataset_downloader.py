"""
dataset_downloader.py
---------------------
Downloads, normalizes, and merges public NIDS datasets from Kaggle
into a unified training CSV compatible with our CICIDS2017 schema.

Datasets pulled
---------------
1. UNSW-NB15      — 2.5M flows, 9 attack types (Backdoor, Worm, Shellcode…)
2. NSL-KDD        — 125K flows, 4 attack categories (R2L, U2R, DoS, Probe)
3. NF-ToN-IoT v2  — IoT-specific attacks (2026-relevant threat landscape)

All are normalized to a common 41-feature schema matching CICIDS2017,
then merged with the existing cicids2017.csv to create combined.csv.

Usage
-----
    export KAGGLE_API_TOKEN=<your_token>
    python3 src/dataset_downloader.py

    # Or supply token explicitly:
    python3 src/dataset_downloader.py --token KGAT_xxx
"""

from __future__ import annotations

import os
import sys
import zipfile
import argparse
import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
DATASET_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── 41 canonical features shared across datasets (CICIDS2017 schema) ────────
CANONICAL_FEATURES = [
    "flow_duration", "total_fwd_packets", "total_bwd_packets",
    "total_len_fwd_packets", "total_len_bwd_packets",
    "fwd_packet_len_max", "fwd_packet_len_min", "fwd_packet_len_mean",
    "bwd_packet_len_max", "bwd_packet_len_min", "bwd_packet_len_mean",
    "flow_bytes_s", "flow_packets_s", "flow_iat_mean", "flow_iat_std",
    "fwd_iat_total", "bwd_iat_total", "fwd_psh_flags", "bwd_psh_flags",
    "fwd_header_len", "bwd_header_len", "fwd_packets_s", "bwd_packets_s",
    "min_packet_len", "max_packet_len", "packet_len_mean",
    "packet_len_std", "packet_len_variance", "fin_flag_cnt",
    "syn_flag_cnt", "rst_flag_cnt", "psh_flag_cnt", "ack_flag_cnt",
    "urg_flag_cnt", "avg_packet_size", "avg_fwd_seg_size",
    "avg_bwd_seg_size", "init_win_bytes_fwd", "init_win_bytes_bwd",
    "act_data_pkt_fwd", "min_seg_size_fwd",
]
LABEL_COL = "label"          # 0 = benign, 1 = attack


# ──────────────────────────────────────────────────────────────────────────────
# Kaggle helpers
# ──────────────────────────────────────────────────────────────────────────────

def _set_kaggle_token(token: str | None):
    """Configure Kaggle credentials from token string or env var."""
    token = token or os.environ.get("KAGGLE_API_TOKEN", "")
    if not token:
        raise ValueError(
            "Set KAGGLE_API_TOKEN env var or pass --token. "
            "Get yours at kaggle.com → Account → API."
        )
    os.environ["KAGGLE_API_TOKEN"] = token
    logger.info("Kaggle credentials configured.")


def _kaggle_download(dataset_slug: str, dest_dir: Path) -> Path:
    """
    Download a Kaggle dataset zip to dest_dir and return the path.
    Uses `kaggle` CLI so the KAGGLE_API_TOKEN env var is respected.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading Kaggle dataset: {dataset_slug} → {dest_dir}")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset_slug,
         "-p", str(dest_dir), "--unzip"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Kaggle download failed for {dataset_slug}:\n"
            f"{result.stdout}\n{result.stderr}"
        )
    logger.info(f"Download complete: {dataset_slug}")
    return dest_dir


# ──────────────────────────────────────────────────────────────────────────────
# Dataset-specific normalizers
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_cicids2017(path: Path) -> pd.DataFrame:
    """Load and normalize the already-merged CICIDS2017 CSV."""
    logger.info(f"Loading CICIDS2017 ({path}) …")
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()

    col_map = {
        "Flow Duration":                "flow_duration",
        "Total Fwd Packets":            "total_fwd_packets",
        "Total Backward Packets":       "total_bwd_packets",
        "Total Length of Fwd Packets":  "total_len_fwd_packets",
        "Total Length of Bwd Packets":  "total_len_bwd_packets",
        "Fwd Packet Length Max":        "fwd_packet_len_max",
        "Fwd Packet Length Min":        "fwd_packet_len_min",
        "Fwd Packet Length Mean":       "fwd_packet_len_mean",
        "Bwd Packet Length Max":        "bwd_packet_len_max",
        "Bwd Packet Length Min":        "bwd_packet_len_min",
        "Bwd Packet Length Mean":       "bwd_packet_len_mean",
        "Flow Bytes/s":                 "flow_bytes_s",
        "Flow Packets/s":               "flow_packets_s",
        "Flow IAT Mean":                "flow_iat_mean",
        "Flow IAT Std":                 "flow_iat_std",
        "Fwd IAT Total":                "fwd_iat_total",
        "Bwd IAT Total":                "bwd_iat_total",
        "Fwd PSH Flags":                "fwd_psh_flags",
        "Bwd PSH Flags":                "bwd_psh_flags",
        "Fwd Header Length":            "fwd_header_len",
        "Bwd Header Length":            "bwd_header_len",
        "Fwd Packets/s":                "fwd_packets_s",
        "Bwd Packets/s":                "bwd_packets_s",
        "Min Packet Length":            "min_packet_len",
        "Max Packet Length":            "max_packet_len",
        "Packet Length Mean":           "packet_len_mean",
        "Packet Length Std":            "packet_len_std",
        "Packet Length Variance":       "packet_len_variance",
        "FIN Flag Count":               "fin_flag_cnt",
        "SYN Flag Count":               "syn_flag_cnt",
        "RST Flag Count":               "rst_flag_cnt",
        "PSH Flag Count":               "psh_flag_cnt",
        "ACK Flag Count":               "ack_flag_cnt",
        "URG Flag Count":               "urg_flag_cnt",
        "Average Packet Size":          "avg_packet_size",
        "Avg Fwd Segment Size":         "avg_fwd_seg_size",
        "Avg Bwd Segment Size":         "avg_bwd_seg_size",
        "Init_Win_bytes_forward":       "init_win_bytes_fwd",
        "Init_Win_bytes_backward":      "init_win_bytes_bwd",
        "act_data_pkt_fwd":             "act_data_pkt_fwd",
        "min_seg_size_forward":         "min_seg_size_fwd",
        "Label":                        LABEL_COL,
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    df[LABEL_COL] = (df[LABEL_COL].astype(str).str.strip().str.upper() != "BENIGN").astype(int)
    df["_source"] = "cicids2017"
    logger.info(f"  CICIDS2017: {len(df):,} rows loaded")
    return df


def _normalize_unsw_nb15(raw_dir: Path) -> pd.DataFrame:
    """
    Normalize UNSW-NB15 to canonical schema.
    Feature mapping: UNSW uses 'dur', 'spkts', 'dpkts' etc.
    """
    logger.info("Normalizing UNSW-NB15 …")
    csvs = sorted(raw_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSVs found in {raw_dir}")

    dfs = []
    for f in csvs:
        if "features" in f.name.lower():
            continue  # skip the feature-description file
        try:
            df = pd.read_csv(f, low_memory=False)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"  Skipping {f.name}: {e}")

    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.strip().str.lower()

    # UNSW-NB15 column → canonical
    # Best-effort mapping to CIC-style features
    col_map = {
        "dur":        "flow_duration",
        "spkts":      "total_fwd_packets",
        "dpkts":      "total_bwd_packets",
        "sbytes":     "total_len_fwd_packets",
        "dbytes":     "total_len_bwd_packets",
        "sloss":      "fwd_psh_flags",
        "dloss":      "bwd_psh_flags",
        "sload":      "flow_bytes_s",
        "dload":      "bwd_packets_s",
        "sinpkt":     "flow_iat_mean",
        "dinpkt":     "flow_iat_std",
        "sjit":       "packet_len_std",
        "djit":       "packet_len_variance",
        "swin":       "init_win_bytes_fwd",
        "dwin":       "init_win_bytes_bwd",
        "stcpb":      "fwd_iat_total",
        "dtcpb":      "bwd_iat_total",
        "smean":      "packet_len_mean",
        "dmean":      "avg_packet_size",
        "trans_depth": "act_data_pkt_fwd",
        "res_bdy_len": "min_seg_size_fwd",
        "label":       LABEL_COL,
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Binary label: 0 = normal, 1+ = attack
    if LABEL_COL in df.columns:
        df[LABEL_COL] = (df[LABEL_COL].astype(str).str.strip() != "0").astype(int)

    df["_source"] = "unsw_nb15"
    logger.info(f"  UNSW-NB15: {len(df):,} rows")
    return df


def _normalize_nsl_kdd(raw_dir: Path) -> pd.DataFrame:
    """
    Normalize NSL-KDD to canonical schema.
    NSL-KDD uses TCP handshake / connection-level features.
    """
    logger.info("Normalizing NSL-KDD …")

    train_file = next(raw_dir.glob("*Train*.txt"), None) or \
                 next(raw_dir.glob("*train*.csv"), None) or \
                 next(raw_dir.glob("KDDTrain*.txt"), None)
    test_file  = next(raw_dir.glob("*Test*.txt"), None) or \
                 next(raw_dir.glob("KDDTest*.txt"), None)

    kdd_cols = [
        "duration","protocol_type","service","flag","src_bytes","dst_bytes",
        "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
        "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
        "num_shells","num_access_files","num_outbound_cmds","is_host_login",
        "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
        "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
        "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
        "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
        "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty_level",
    ]
    dfs = []
    for f in [train_file, test_file]:
        if f and f.exists():
            df = pd.read_csv(f, header=None, names=kdd_cols[:len(kdd_cols)], low_memory=False)
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No NSL-KDD txt/csv found in {raw_dir}")

    df = pd.concat(dfs, ignore_index=True)

    # Binary label  (normal = 0, any attack = 1)
    df[LABEL_COL] = (df["label"].astype(str).str.strip().str.lower() != "normal").astype(int)

    # Map KDD numeric features → canonical schema (best-effort)
    col_map = {
        "duration":           "flow_duration",
        "src_bytes":          "total_len_fwd_packets",
        "dst_bytes":          "total_len_bwd_packets",
        "count":              "total_fwd_packets",
        "srv_count":          "total_bwd_packets",
        "hot":                "fwd_psh_flags",
        "num_failed_logins":  "syn_flag_cnt",
        "num_compromised":    "rst_flag_cnt",
        "num_root":           "ack_flag_cnt",
        "num_shells":         "fin_flag_cnt",
        "serror_rate":        "flow_iat_mean",
        "rerror_rate":        "flow_iat_std",
        "same_srv_rate":      "fwd_iat_total",
        "diff_srv_rate":      "bwd_iat_total",
        "dst_host_same_srv_rate":   "packet_len_mean",
        "dst_host_diff_srv_rate":   "packet_len_std",
        "dst_host_serror_rate":     "packet_len_variance",
        "dst_host_count":           "flow_bytes_s",
        "dst_host_srv_count":       "flow_packets_s",
        "wrong_fragment":           "fwd_header_len",
        "urgent":                   "bwd_header_len",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    df["_source"] = "nsl_kdd"
    logger.info(f"  NSL-KDD: {len(df):,} rows")
    return df


def _normalize_nf_ton_iot(raw_dir: Path) -> pd.DataFrame:
    """
    Normalize NF-ToN-IoT v2 (NetFlow features) to canonical schema.
    """
    logger.info("Normalizing NF-ToN-IoT v2 …")
    csvs = sorted(raw_dir.glob("*.csv")) + sorted(raw_dir.glob("*.parquet"))
    if not csvs:
        raise FileNotFoundError(f"No data files in {raw_dir}")

    dfs = []
    for f in csvs[:5]:  # up to 5 files
        try:
            if f.suffix == ".parquet":
                chunk = pd.read_parquet(f)
            else:
                chunk = pd.read_csv(f, low_memory=False)
            dfs.append(chunk)
            logger.info(f"  {f.name}: {len(chunk):,} rows")
        except Exception as e:
            logger.warning(f"  Skipping {f.name}: {e}")

    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.strip().str.upper()

    # NF-ToN-IoT → canonical (NetFlow field names)
    col_map = {
        "IN_PKTS":      "total_fwd_packets",
        "OUT_PKTS":     "total_bwd_packets",
        "IN_BYTES":     "total_len_fwd_packets",
        "OUT_BYTES":    "total_len_bwd_packets",
        "FLOW_DURATION_MILLISECONDS": "flow_duration",
        "TCP_FLAGS":    "psh_flag_cnt",
        "L4_SRC_PORT":  "fwd_header_len",
        "L4_DST_PORT":  "bwd_header_len",
        "PROTOCOL":     "fwd_psh_flags",
        "LABEL":        LABEL_COL,
        "Attack":       LABEL_COL,
        "attack":       LABEL_COL,
        "Label":        LABEL_COL,
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    if LABEL_COL in df.columns:
        df[LABEL_COL] = (df[LABEL_COL].astype(str).str.strip() != "0").astype(int)

    df["_source"] = "nf_ton_iot"
    logger.info(f"  NF-ToN-IoT: {len(df):,} rows")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Merge & clean
# ──────────────────────────────────────────────────────────────────────────────

def _to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Project any DataFrame down to the canonical feature set + label."""
    out = pd.DataFrame(index=df.index)
    for col in CANONICAL_FEATURES:
        out[col] = pd.to_numeric(df.get(col, np.nan), errors="coerce")
    out[LABEL_COL] = df[LABEL_COL].values if LABEL_COL in df.columns else 0
    out["_source"] = df.get("_source", "unknown")
    return out


def build_combined(
    cicids_path: Path,
    raw_dirs: dict[str, Path],
    out_path: Path,
    max_rows_per_source: int = 500_000,
) -> pd.DataFrame:
    """
    Merge CICIDS2017 + all extra datasets into one unified CSV.

    Parameters
    ----------
    cicids_path          : path to existing cicids2017.csv
    raw_dirs             : {dataset_name: directory_path}
    out_path             : where to write combined.csv
    max_rows_per_source  : cap per extra dataset to avoid imbalance
    """
    parts: list[pd.DataFrame] = []

    # 1. CICIDS2017 (all rows)
    parts.append(_to_canonical(_normalize_cicids2017(cicids_path)))

    # 2. Extra datasets
    normalizers = {
        "unsw_nb15":  _normalize_unsw_nb15,
        "nsl_kdd":    _normalize_nsl_kdd,
        "nf_ton_iot": _normalize_nf_ton_iot,
    }
    for name, norm_fn in normalizers.items():
        raw = raw_dirs.get(name)
        if raw and raw.exists():
            try:
                df = _to_canonical(norm_fn(raw))
                # Sample if too large
                if len(df) > max_rows_per_source:
                    df = df.sample(n=max_rows_per_source, random_state=42)
                parts.append(df)
                logger.info(f"  Added {name}: {len(df):,} rows sampled")
            except Exception as e:
                logger.warning(f"  {name} skipped: {e}")
        else:
            logger.warning(f"  {name}: directory not found → skipping")

    # 3. Concatenate
    combined = pd.concat(parts, ignore_index=True)

    # 4. Clean — replace inf/NaN with column median
    combined = combined.replace([np.inf, -np.inf], np.nan)
    for col in CANONICAL_FEATURES:
        if col in combined.columns:
            med = combined[col].median()
            combined[col] = combined[col].fillna(med if pd.notna(med) else 0)

    logger.info(f"\n{'='*55}")
    logger.info(f"Combined dataset: {len(combined):,} rows")
    logger.info(f"Label distribution:\n{combined[LABEL_COL].value_counts().to_string()}")
    logger.info(f"Source distribution:\n{combined['_source'].value_counts().to_string()}")

    combined.to_csv(out_path, index=False)
    sz = out_path.stat().st_size // 1024 // 1024
    logger.info(f"Saved combined.csv → {out_path} ({sz} MB)")
    return combined


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download Kaggle NIDS datasets and merge with CICIDS2017"
    )
    parser.add_argument("--token", default=None,
                        help="Kaggle API token (or set KAGGLE_API_TOKEN env var)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip Kaggle download (use existing raw dirs)")
    parser.add_argument("--max-rows", type=int, default=500_000,
                        help="Max rows per extra dataset (default 500K)")
    args = parser.parse_args()

    _set_kaggle_token(args.token)

    raw_base = DATASET_DIR / "raw"
    raw_base.mkdir(exist_ok=True)

    datasets = {
        "unsw_nb15":  ("mrwellsdavid/unsw-nb15",   raw_base / "unsw_nb15"),
        "nsl_kdd":    ("hassan06/nslkdd",           raw_base / "nsl_kdd"),
        "nf_ton_iot": ("dhoogla/nftoniotv2",        raw_base / "nf_ton_iot"),
    }

    if not args.skip_download:
        for name, (slug, dest) in datasets.items():
            try:
                _kaggle_download(slug, dest)
            except Exception as e:
                logger.warning(f"Download failed for {name}: {e}  — will skip")
    else:
        logger.info("Skipping downloads (--skip-download set).")

    raw_dirs = {name: dest for name, (_, dest) in datasets.items()}

    cicids_path = DATASET_DIR / "cicids2017.csv"
    if not cicids_path.exists():
        raise FileNotFoundError(
            f"cicids2017.csv not found at {cicids_path}. "
            "Run the CICIDS2017 merge first."
        )

    out_path = DATASET_DIR / "combined.csv"
    combined = build_combined(cicids_path, raw_dirs, out_path, args.max_rows)
    print(f"\n✅  Combined dataset ready: {len(combined):,} rows → {out_path}")
    print("   Next: python3 src/training.py --dataset dataset/combined.csv")


if __name__ == "__main__":
    main()
