"""階層ベイズMNL（対角共分散、PyMC）。

raw符号化された long DataFrame（brand 4列ダミー・2値属性の0/1・price を連続値、
「どれも選ばない」は全列0の行、といった符号化）に、対角共分散の階層構造
`beta_i ~ Normal(beta_bar, diag(sigma^2))` を適用する。非中心化パラメータ化
（`z ~ Normal(0,1)` を `sigma_beta` と組み合わせる）で収束を安定させる。

`build_design`/`fit_hb_diag`/`summarize` は raw符号化の列名（`feature_names`）を
引数で受け取るだけで属性の意味を知らないため、そのままどの調査にも使える。
`importance_from_summary`だけは属性重要度の算出方法（フルダミー/基準水準コード/連続値）
を知る必要があるため `study.StudyConfig` を受け取る。
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from . import schema
from .paths import RESULTS_DIR, hb_importance_csv, hb_meta_json, hb_summary_csv, hb_trace_nc
from .prepare import load_raw_long
from .study import StudyConfig

DEFAULT_PRIORS = {"beta_bar_scale": 5.0, "sigma_beta_scale": 2.5}
DEFAULT_SEED = 42


@dataclass
class DesignData:
    X: np.ndarray  # (n_resp, n_task, n_alt, K)
    choice_idx: np.ndarray  # (n_resp, n_task)
    resp_ids: list[int]
    task_ids: list[int]
    option_labels: list[str]


def build_design(df: pd.DataFrame, feature_names: list[str] = schema.CAMERA_STUDY.raw_columns()) -> DesignData:
    """long DataFrame から (n_resp, n_task, n_alt, K) の設計配列を作る。

    camera・persearch とも n_alt=5（A〜D + none）。データセット間で水準数が違っても
    対応できるよう、実データから option_label のユニーク数を都度読み取る。
    """
    resp_ids = sorted(df["respondent_id"].unique().tolist())
    task_ids = sorted(df["task_index"].unique().tolist())
    n_resp, n_task = len(resp_ids), len(task_ids)
    resp_pos = {r: i for i, r in enumerate(resp_ids)}
    task_pos = {t: i for i, t in enumerate(task_ids)}

    option_labels = sorted(df["option_label"].unique().tolist(), key=lambda x: (x == "none", x))
    n_alt = len(option_labels)
    opt_pos = {o: i for i, o in enumerate(option_labels)}

    df_sorted = df.sort_values(["respondent_id", "task_index"])

    chosen_per_task = df_sorted.groupby(["respondent_id", "task_index"])["chosen"].sum()
    if not (chosen_per_task == 1).all():
        raise ValueError("each (respondent, task) must have exactly one chosen row")

    K = len(feature_names)
    X = np.zeros((n_resp, n_task, n_alt, K), dtype=np.float64)
    choice_idx = np.zeros((n_resp, n_task), dtype=np.int64)

    r_arr = df_sorted["respondent_id"].map(resp_pos).to_numpy()
    t_arr = df_sorted["task_index"].map(task_pos).to_numpy()
    a_arr = df_sorted["option_label"].map(opt_pos).to_numpy()
    feat_arr = df_sorted[feature_names].to_numpy(dtype=np.float64)
    chosen_arr = df_sorted["chosen"].to_numpy()

    X[r_arr, t_arr, a_arr, :] = feat_arr
    chosen_mask = chosen_arr == 1
    choice_idx[r_arr[chosen_mask], t_arr[chosen_mask]] = a_arr[chosen_mask]

    return DesignData(X=X, choice_idx=choice_idx, resp_ids=resp_ids, task_ids=task_ids, option_labels=option_labels)


def fit_hb_diag(
    X: np.ndarray,
    choice_idx: np.ndarray,
    *,
    feature_names: list[str] = schema.CAMERA_STUDY.raw_columns(),
    beta_bar_scale: float = DEFAULT_PRIORS["beta_bar_scale"],
    sigma_beta_scale: float = DEFAULT_PRIORS["sigma_beta_scale"],
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 2,
    target_accept: float = 0.9,
    seed: int = DEFAULT_SEED,
    progressbar: bool = True,
) -> az.InferenceData:
    n_resp, n_task, n_alt, K = X.shape
    coords = {"resp": np.arange(n_resp), "param": feature_names, "task": np.arange(n_task)}
    with pm.Model(coords=coords) as model:
        X_data = pm.Data("X_data", X)

        beta_bar = pm.Normal("beta_bar", mu=0.0, sigma=beta_bar_scale, dims="param")
        sigma_beta = pm.HalfNormal("sigma_beta", sigma=sigma_beta_scale, dims="param")

        z = pm.Normal("z", mu=0.0, sigma=1.0, dims=("resp", "param"))
        beta_i = pm.Deterministic("beta_i", beta_bar + z * sigma_beta, dims=("resp", "param"))

        V = pt.einsum("rtak,rk->rta", X_data, beta_i)
        pm.Categorical("choice", logit_p=V, observed=choice_idx, dims=("resp", "task"))

        idata = pm.sample(
            draws=draws, tune=tune, chains=chains,
            target_accept=target_accept, random_seed=seed, progressbar=progressbar,
        )
    return idata


_PARAM_INDEX_RE = re.compile(r"\[(.+)\]$")


def _extract_param_names(index: pd.Index) -> list[str]:
    names = []
    for label in index:
        m = _PARAM_INDEX_RE.search(str(label))
        names.append(m.group(1) if m else str(label))
    return names


def summarize(idata: az.InferenceData, feature_names: list[str] = schema.CAMERA_STUDY.raw_columns()) -> pd.DataFrame:
    """beta_bar・sigma_beta の事後要約を feature_names の順で返す。

    旧実装は `az.summary()` の行順が coords 順と一致する前提で `param` 列を位置で
    上書きしていた。ここではインデックス名（`beta_bar[canon]` 等）をパースしてから
    `feature_names` で reindex するため、行順の実装依存を排除している。
    """
    bb = az.summary(idata, var_names=["beta_bar"], ci_prob=0.94, ci_kind="hdi", round_to="none")
    sb = az.summary(idata, var_names=["sigma_beta"], ci_prob=0.94, ci_kind="hdi", round_to="none")

    bb = bb.reset_index().rename(columns={"index": "raw_index"})
    sb = sb.reset_index().rename(columns={"index": "raw_index"})
    bb["param"] = _extract_param_names(bb["raw_index"])
    sb["param"] = _extract_param_names(sb["raw_index"])

    if set(bb["param"]) != set(feature_names) or set(sb["param"]) != set(feature_names):
        raise ValueError("summary param names do not match feature_names")

    bb = bb.set_index("param").loc[feature_names].reset_index().drop(columns="raw_index")
    sb = sb.set_index("param").loc[feature_names].reset_index().drop(columns="raw_index")
    return bb.merge(sb, on="param", suffixes=("_beta_bar", "_sigma_beta"))


def continuous_ranges(df: pd.DataFrame, config: StudyConfig = schema.CAMERA_STUDY) -> dict[str, float]:
    """連続属性ごとの値の範囲（max-min）。`importance_from_summary`のスケーリングに使う。"""
    ranges: dict[str, float] = {}
    for attr in config.continuous_attributes():
        vals = df.loc[df["is_none"] == 0, attr.continuous_column]
        ranges[attr.continuous_column] = float(vals.max() - vals.min())
    return ranges


def importance_from_summary(
    summary: pd.DataFrame,
    ranges: dict[str, float],
    config: StudyConfig = schema.CAMERA_STUDY,
) -> pd.DataFrame:
    """属性重要度（beta_bar から算出）。

    カテゴリ属性がフルダミー符号化（水準数ぶんの列、brandのような）なら列間の
    最大値-最小値、基準水準コード（1列のみ、2値属性のような）ならその係数の絶対値、
    連続属性（price のような）なら係数の絶対値×値の範囲、を重要度の素点とする。
    水準数・属性数が調査によって変わっても同じ計算式でよいのは、raw_columns() が
    1個の属性は「絶対値」、複数個の属性は「レンジ」という2パターンしかないため。
    """
    coefs = dict(zip(summary["param"], summary["mean_beta_bar"]))
    raw_importance: dict[str, float] = {}
    for attr in config.attributes:
        cols = attr.raw_columns()
        if attr.is_continuous:
            raw_importance[attr.name] = abs(coefs[cols[0]]) * ranges.get(cols[0], 1.0)
        elif len(cols) == 1:
            raw_importance[attr.name] = abs(coefs[cols[0]])
        else:
            vals = [coefs[c] for c in cols]
            raw_importance[attr.name] = max(vals) - min(vals)

    total = sum(raw_importance.values())
    imp = pd.Series(raw_importance) / total
    return imp.sort_values(ascending=False).rename("importance").reset_index().rename(columns={"index": "attr"})


def convergence_headline(summary: pd.DataFrame) -> dict[str, float]:
    """beta_bar・sigma_beta 全パラメータを合わせた最大r_hat・最小ess_bulk。

    旧実装の `run()` は beta_bar のみで算出しており、README/記事が使っている
    「beta_bar・sigma_beta 全体」の定義とズレていた。
    """
    r_hat_cols = [c for c in summary.columns if c.startswith("r_hat_")]
    ess_cols = [c for c in summary.columns if c.startswith("ess_bulk_")]
    return {
        "max_r_hat": float(summary[r_hat_cols].to_numpy().max()),
        "min_ess_bulk": float(summary[ess_cols].to_numpy().min()),
    }


def prior_tag(sigma_beta_scale: float, beta_bar_scale: float) -> str:
    """デフォルトと異なる事前分布スケールのみをファイル名に含める。

    旧実装は常に `sigmascale...` を名乗っており、beta_bar_scale だけ振った場合に
    誤ったファイル名になっていた。
    """
    parts = []
    if sigma_beta_scale != DEFAULT_PRIORS["sigma_beta_scale"]:
        parts.append(f"sigmascale{str(sigma_beta_scale).replace('.', 'p')}")
    if beta_bar_scale != DEFAULT_PRIORS["beta_bar_scale"]:
        parts.append(f"betascale{str(beta_bar_scale).replace('.', 'p')}")
    if not parts:
        raise ValueError("prior_tag() was called with default priors; pass an explicit tag instead")
    return "_".join(parts)


def run_hb(
    label: str,
    *,
    out_dir: Path | None = None,
    in_dir: Path | None = None,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 2,
    target_accept: float = 0.9,
    seed: int = DEFAULT_SEED,
    n_resp_limit: int | None = None,
    beta_bar_scale: float = DEFAULT_PRIORS["beta_bar_scale"],
    sigma_beta_scale: float = DEFAULT_PRIORS["sigma_beta_scale"],
    tag: str | None = None,
    save_trace: bool = True,
    save_importance: bool = True,
    progressbar: bool = True,
    config: StudyConfig = schema.CAMERA_STUDY,
) -> dict[str, Path]:
    """notebook（hb_diag.ipynb）とCLI（prior_sensitivity.py）が共有する実行本体。

    非デフォルトの事前分布で `tag` を省略すると、本番ファイル（`hb_diag_{label}_*`）を
    誤って上書きしてしまうため明示的にエラーにする。
    """
    is_default_priors = (
        beta_bar_scale == DEFAULT_PRIORS["beta_bar_scale"]
        and sigma_beta_scale == DEFAULT_PRIORS["sigma_beta_scale"]
    )
    if tag is None and not is_default_priors:
        raise ValueError(
            "non-default priors require an explicit tag (see prior_tag()) "
            "so the official hb_diag_{label}_* files are never overwritten"
        )

    out_dir = Path(out_dir) if out_dir is not None else RESULTS_DIR
    df = load_raw_long(label, in_dir, config=config)
    if n_resp_limit is not None:
        keep_ids = sorted(df["respondent_id"].unique())[:n_resp_limit]
        df = df[df["respondent_id"].isin(keep_ids)]

    ranges = continuous_ranges(df, config)
    feature_names = config.raw_columns()

    t0 = time.time()
    design = build_design(df, feature_names)
    print(f"[{label}] design matrix: X.shape={design.X.shape}, build={time.time() - t0:.1f}s")

    t0 = time.time()
    idata = fit_hb_diag(
        design.X, design.choice_idx,
        feature_names=feature_names,
        beta_bar_scale=beta_bar_scale, sigma_beta_scale=sigma_beta_scale,
        draws=draws, tune=tune, chains=chains, target_accept=target_accept,
        seed=seed, progressbar=progressbar,
    )
    elapsed = time.time() - t0
    print(f"[{label}] sampling done in {elapsed:.1f}s")

    summary = summarize(idata, feature_names)
    summary["elapsed_sec"] = elapsed

    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    if save_trace:
        trace_path = hb_trace_nc(label, out_dir, tag)
        idata.to_netcdf(trace_path)
        written["trace"] = trace_path

    summary_path = hb_summary_csv(label, out_dir, tag)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    written["summary"] = summary_path

    if save_importance:
        imp = importance_from_summary(summary, ranges, config)
        imp_path = hb_importance_csv(label, out_dir, tag)
        imp.to_csv(imp_path, index=False, encoding="utf-8-sig")
        written["importance"] = imp_path

    headline = convergence_headline(summary)
    meta = {
        "label": label, "tag": tag, "seed": seed,
        "draws": draws, "tune": tune, "chains": chains, "target_accept": target_accept,
        "beta_bar_scale": beta_bar_scale, "sigma_beta_scale": sigma_beta_scale,
        "n_resp": int(df["respondent_id"].nunique()), "elapsed_sec": elapsed,
        "max_r_hat": headline["max_r_hat"], "min_ess_bulk": headline["min_ess_bulk"],
    }
    meta_path = hb_meta_json(label, out_dir, tag)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    written["meta"] = meta_path

    print(summary[[
        "param", "mean_beta_bar", "r_hat_beta_bar", "ess_bulk_beta_bar",
        "mean_sigma_beta", "r_hat_sigma_beta",
    ]].to_string(index=False))
    print(f"max r_hat (beta_bar+sigma_beta): {headline['max_r_hat']}, min ess_bulk: {headline['min_ess_bulk']}")

    return written
