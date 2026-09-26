"""カウンティング法（選択率の単純集計）による効用値・属性重要度の推定。

水準ごとの選択率（選択回数 ÷ 提示回数）の対数を効用の代用値とみなす簡易法。
理論的な弱点（同一選択セット内でしか `ln P_i − ln P_j = V_i − V_j` が成り立たない
こと）は Qiita 記事側で説明する。ここではその弱点込みで「手早く算出できる簡易法」
として実装する。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import schema
from .paths import (
    RESULTS_DIR,
    camera_long_csv,
    counting_csv,
    counting_importance_csv,
    persearch_long_csv,
)


def _price_level_order(df: pd.DataFrame, price_col: str) -> list[float]:
    return sorted(df[price_col].dropna().unique().tolist())


def counting_utilities(
    df: pd.DataFrame,
    attr_cols: list[str],
    price_col: str | None = None,
) -> pd.DataFrame:
    """属性×水準ごとの提示回数・選択回数・選択率・効用値(代用値)を計算する。

    Args:
        df: long 形式 DataFrame（1行=1課題×1選択肢、is_none/chosen 列を持つ）
        attr_cols: 集計するカテゴリ属性の列名（price を除く）
        price_col: 価格属性の列名（数値）。None なら価格は集計しない

    Returns:
        columns: attr, level, shown, chosen, rate, log_rate, utility(基準水準=0),
                 is_reference
    """
    shown_df = df[df["is_none"] == 0]

    rows: list[dict] = []
    all_attrs = list(attr_cols)
    level_orders: dict[str, list] = {a: schema.ATTRIBUTE_LEVEL_ORDER[a] for a in attr_cols}
    if price_col is not None:
        all_attrs = all_attrs + [price_col]
        level_orders[price_col] = _price_level_order(df, price_col)

    for attr in all_attrs:
        levels = level_orders[attr]
        for level_idx, level in enumerate(levels):
            mask = shown_df[attr] == level
            shown = int(mask.sum())
            chosen = int(shown_df.loc[mask, "chosen"].sum())
            rate = chosen / shown if shown > 0 else float("nan")
            rows.append(
                {
                    "attr": attr,
                    "level": level,
                    "shown": shown,
                    "chosen": chosen,
                    "rate": rate,
                    "log_rate": np.log(rate) if rate > 0 else float("-inf"),
                    "is_reference": level_idx == 0,
                }
            )

    out = pd.DataFrame(rows)

    # 属性ごとに基準水準(先頭)の log_rate を引いて効用の代用値(0始まり)にする
    out["utility"] = np.nan
    for attr in all_attrs:
        sub = out["attr"] == attr
        ref_log_rate = out.loc[sub & out["is_reference"], "log_rate"].iloc[0]
        out.loc[sub, "utility"] = out.loc[sub, "log_rate"] - ref_log_rate

    return out


def attribute_importance(counting_result: pd.DataFrame) -> pd.DataFrame:
    """属性重要度（属性内 utility の幅を全属性の幅の合計で正規化）を計算する。"""
    width = counting_result.groupby("attr")["utility"].agg(lambda s: s.max() - s.min())
    importance = (width / width.sum()).sort_values(ascending=False)
    return importance.rename("importance").reset_index()


def run_counting_for_dataset(
    long_csv: Path,
    label: str,
    *,
    out_dir: Path | None = None,
    price_col: str = "price_usd100",
) -> list[Path]:
    out_dir = Path(out_dir) if out_dir is not None else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(long_csv)

    written: list[Path] = []
    for model_name, attrs in [("main5", schema.MAIN_ATTRS), ("full", schema.FULL_ATTRS)]:
        if model_name == "full" and not set(schema.FULL_ATTRS) <= set(df.columns):
            missing = sorted(set(schema.FULL_ATTRS) - set(df.columns))
            print(f"skip full: {long_csv} is missing columns {missing}")
            continue

        result = counting_utilities(df, attrs, price_col=price_col)
        importance = attribute_importance(result)

        out_path = counting_csv(label, model_name, out_dir)
        result.to_csv(out_path, index=False, encoding="utf-8-sig")
        imp_path = counting_importance_csv(label, model_name, out_dir)
        importance.to_csv(imp_path, index=False, encoding="utf-8-sig")
        written += [out_path, imp_path]

        print(f"=== {label} / {model_name} ===")
        print(result.to_string(index=False))
        print(importance.to_string(index=False))
        print()

    return written


def run_counting_all(*, in_dir: Path | None = None, out_dir: Path | None = None) -> list[Path]:
    in_dir = Path(in_dir) if in_dir is not None else RESULTS_DIR
    written: list[Path] = []

    camera_csv = camera_long_csv(in_dir)
    if camera_csv.exists():
        written += run_counting_for_dataset(camera_csv, "camera", out_dir=out_dir)
    else:
        print(f"skip: {camera_csv} not found (run prepare_camera.ipynb first)")

    persearch_csv = persearch_long_csv(in_dir)
    if persearch_csv.exists():
        written += run_counting_for_dataset(persearch_csv, "persearch_camera", out_dir=out_dir)
    else:
        print(f"skip: {persearch_csv} not found (run prepare_persearch_camera.ipynb first)")

    return written
