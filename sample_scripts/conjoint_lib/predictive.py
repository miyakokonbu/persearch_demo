"""hb_diag で推定した beta_bar が、選択行動そのものをどれだけ予測できるかを検証する。

符号の一致率・Spearman順位相関は「疑似データの係数が実データの係数とどれだけ一致するか」を
見る指標だが、ここでは「係数がズレていても、実際の選択行動の予測には使えるか」という、
より実務に近い問いを検証する。real-on-real（理論上の上限）・pseudo-on-real（本題）・
uniform（何もしない場合の下限）の3通りで、対数尤度（確率としての較正）とtop-1的中率
（複数候補のうちどれが選ばれるかの的中率）を比較する。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import schema
from .hb import build_design
from .paths import RESULTS_DIR, hb_summary_csv, predictive_summary_csv, predictive_tauscan_csv
from .prepare import load_raw_long

# direction（予測対象データセット） -> hb_diag側のラベル
HB_LABEL = {"real": "camera", "pseudo": "persearch"}


def load_beta_bar(summary_csv: Path, feature_names: list[str] = schema.RAW_COLUMNS) -> np.ndarray:
    if not summary_csv.exists():
        raise FileNotFoundError(f"{summary_csv} が見つかりません。hb_diag.ipynb を先に実行してください。")
    summary = pd.read_csv(summary_csv, encoding="utf-8-sig")
    coefs = dict(zip(summary["param"], summary["mean_beta_bar"]))
    missing = [f for f in feature_names if f not in coefs]
    if missing:
        raise ValueError(f"{summary_csv}: missing params {missing}")
    return np.array([coefs[f] for f in feature_names])


def evaluate(X: np.ndarray, choice_idx: np.ndarray, beta: np.ndarray, tau: float = 1.0) -> tuple[float, float]:
    """(avg_log_lik_per_choice, top1_hit_rate) を返す。tau は beta に掛ける温度（縮小率）。"""
    V = (X @ beta) * tau  # (n_resp, n_task, n_alt)
    V = V - V.max(axis=-1, keepdims=True)
    expV = np.exp(V)
    P = expV / expV.sum(axis=-1, keepdims=True)

    n_alt = P.shape[-1]
    flat_P = P.reshape(-1, n_alt)
    flat_choice = choice_idx.reshape(-1)

    chosen_p = flat_P[np.arange(len(flat_choice)), flat_choice]
    avg_ll = float(np.log(np.clip(chosen_p, 1e-12, None)).mean())

    pred = flat_P.argmax(axis=-1)
    hit_rate = float((pred == flat_choice).mean())
    return avg_ll, hit_rate


def _require_direction(direction: str) -> str:
    if direction not in HB_LABEL:
        raise ValueError(f"unknown direction: {direction!r} (expected 'real' or 'pseudo')")
    return "pseudo" if direction == "real" else "real"


def _load_design(direction: str, in_dir: Path | None):
    label = HB_LABEL[direction]
    df = load_raw_long(label, in_dir)
    return build_design(df)


def run_predictive_check(
    direction: str,
    *,
    in_dir: Path | None = None,
    out_dir: Path | None = None,
    tau: float = 1.0,
) -> pd.DataFrame:
    other = _require_direction(direction)
    in_dir = Path(in_dir) if in_dir is not None else RESULTS_DIR
    out_dir = Path(out_dir) if out_dir is not None else RESULTS_DIR

    design = _load_design(direction, in_dir)
    beta_oracle = load_beta_bar(hb_summary_csv(HB_LABEL[direction], in_dir))
    beta_cross = load_beta_bar(hb_summary_csv(HB_LABEL[other], in_dir))

    n_alt = design.X.shape[2]
    uniform_ll = float(np.log(1.0 / n_alt))

    ll_oracle, hit_oracle = evaluate(design.X, design.choice_idx, beta_oracle, tau=1.0)
    ll_cross, hit_cross = evaluate(design.X, design.choice_idx, beta_cross, tau=tau)

    rows = [
        {
            "model": f"{direction}-on-{direction}", "source": direction, "target": direction,
            "tau": 1.0, "avg_log_lik": ll_oracle, "top1_hit_rate": hit_oracle,
        },
        {
            "model": f"{other}-on-{direction}", "source": other, "target": direction,
            "tau": tau, "avg_log_lik": ll_cross, "top1_hit_rate": hit_cross,
        },
        {
            "model": "uniform", "source": None, "target": direction,
            "tau": None, "avg_log_lik": uniform_ll, "top1_hit_rate": 1.0 / n_alt,
        },
    ]
    result = pd.DataFrame(rows)
    result["n_resp"] = design.X.shape[0]
    result["n_task"] = design.X.shape[1]
    result["n_alt"] = n_alt

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = predictive_summary_csv(direction, out_dir)
    result.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"=== {direction} の実際の選択（{design.X.shape[0]}人 x {design.X.shape[1]}課題）に対する予測性能 ===")
    print(f"[{direction}-on-{direction} (理論上の上限)] avg_log_lik={ll_oracle:.4f}  top1_hit_rate={hit_oracle * 100:.1f}%")
    tau_note = "" if tau == 1.0 else f" (tau={tau})"
    print(f"[{other}-on-{direction}{tau_note} (本題)] avg_log_lik={ll_cross:.4f}  top1_hit_rate={hit_cross * 100:.1f}%")
    print(f"[uniform baseline] avg_log_lik={uniform_ll:.4f}  top1_hit_rate={100 / n_alt:.1f}%")
    print(f"wrote {out_path}")

    return result


def run_tau_scan(
    direction: str,
    start: float,
    stop: float,
    step: float,
    *,
    in_dir: Path | None = None,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    if step <= 0:
        raise ValueError("step must be > 0")
    if stop < start:
        raise ValueError("stop must be >= start")

    other = _require_direction(direction)
    in_dir = Path(in_dir) if in_dir is not None else RESULTS_DIR
    out_dir = Path(out_dir) if out_dir is not None else RESULTS_DIR

    design = _load_design(direction, in_dir)
    beta_cross = load_beta_bar(hb_summary_csv(HB_LABEL[other], in_dir))

    taus = np.arange(start, stop + step / 2, step)
    rows = []
    best = None
    for tau in taus:
        ll, hit = evaluate(design.X, design.choice_idx, beta_cross, tau=float(tau))
        rows.append({"tau": round(float(tau), 4), "avg_log_lik": ll, "top1_hit_rate": hit})
        print(f"  tau={tau:.2f}  avg_log_lik={ll:.4f}  top1_hit_rate={hit * 100:.1f}%")
        if best is None or ll > best[1]:
            best = (tau, ll, hit)

    result = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = predictive_tauscan_csv(direction, out_dir)
    result.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[best tau] {best[0]:.2f} (avg_log_lik={best[1]:.4f}, top1_hit_rate={best[2] * 100:.1f}%)")
    print(f"wrote {out_path}")

    return result
