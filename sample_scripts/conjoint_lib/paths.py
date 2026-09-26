"""data/・results/ の場所と、results/ 配下のファイル名を一意に決める。

conjoint_lib 自身の `__file__` から repo root を逆算するため、notebook
（cwd=sample_scripts/）・CLIスクリプト（実行ディレクトリ不問）のどちらから
importしても同じパスになる（旧実装は notebook が `Path.cwd().parent`、
スクリプトが `Path(__file__).resolve().parent.parent` と別々に解決していた）。
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"


def camera_long_csv(d: Path = RESULTS_DIR) -> Path:
    return Path(d) / "camera_long.csv"


def camera_raw_long_csv(d: Path = RESULTS_DIR) -> Path:
    return Path(d) / "camera_raw_long.csv"


def persearch_long_csv(d: Path = RESULTS_DIR) -> Path:
    return Path(d) / "persearch_camera_long.csv"


def persearch_raw_long_csv(d: Path = RESULTS_DIR) -> Path:
    return Path(d) / "persearch_camera_raw_long.csv"


def raw_long_csv(label: str, d: Path = RESULTS_DIR) -> Path:
    """label: 'camera' または 'persearch'。"""
    if label == "camera":
        return camera_raw_long_csv(d)
    if label == "persearch":
        return persearch_raw_long_csv(d)
    raise ValueError(f"unknown label: {label!r}")


def _hb_stem(label: str, tag: str | None) -> str:
    return f"hb_diag_{label}" if tag is None else f"hb_diag_{label}_{tag}"


def hb_trace_nc(label: str, d: Path = RESULTS_DIR, tag: str | None = None) -> Path:
    return Path(d) / f"{_hb_stem(label, tag)}_trace.nc"


def hb_summary_csv(label: str, d: Path = RESULTS_DIR, tag: str | None = None) -> Path:
    return Path(d) / f"{_hb_stem(label, tag)}_summary.csv"


def hb_importance_csv(label: str, d: Path = RESULTS_DIR, tag: str | None = None) -> Path:
    return Path(d) / f"{_hb_stem(label, tag)}_importance.csv"


def hb_meta_json(label: str, d: Path = RESULTS_DIR, tag: str | None = None) -> Path:
    return Path(d) / f"{_hb_stem(label, tag)}_meta.json"


def counting_csv(label: str, model: str, d: Path = RESULTS_DIR) -> Path:
    """label: 'camera' または 'persearch_camera'。model: 'main5' または 'full'。"""
    return Path(d) / f"counting_{label}_{model}.csv"


def counting_importance_csv(label: str, model: str, d: Path = RESULTS_DIR) -> Path:
    return Path(d) / f"counting_{label}_{model}_importance.csv"


def predictive_summary_csv(target: str, d: Path = RESULTS_DIR) -> Path:
    """target: 'real' または 'pseudo'。"""
    return Path(d) / f"predictive_check_{target}_summary.csv"


def predictive_tauscan_csv(target: str, d: Path = RESULTS_DIR) -> Path:
    return Path(d) / f"predictive_check_{target}_tauscan.csv"
