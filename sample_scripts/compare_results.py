"""2つの results/ 出力（hb_diag summary/importance か、単純なCSV）を比較する検証ツール。

リファクタリング・バグ修正の前後で数値が変わっていないことを確認するために作成した
（`results/` 配下は Qiita 記事が直接引用しているデータなので、上書き前に必ず比較する）。

使い方:
    # 単純なCSV同士（prepare_*/counting の出力など）。全列を許容誤差なしで比較する
    .venv/Scripts/python sample_scripts/compare_results.py OLD.csv NEW.csv --kind csv

    # hb_diag の summary.csv 同士。elapsed_sec は無視し、列ごとに許容誤差を変える
    .venv/Scripts/python sample_scripts/compare_results.py OLD_summary.csv NEW_summary.csv --kind hb
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# hb summary の列プレフィックスごとの許容誤差（絶対値）。r_hat/ess_bulk は列名にプレフィックスを含む。
_HB_TOLERANCES = [
    ("r_hat_", 0.001),
    ("ess_bulk_", 5.0),
    ("ess_tail_", 5.0),
    ("mcse_", None),  # 参考情報のみ。差分は報告するが不一致とはしない
    ("mean_", 5e-4),
    ("sd_", 5e-4),
    ("hdi_", 5e-4),
]
_HB_IGNORE_COLS = {"elapsed_sec"}


def _tolerance_for_column(col: str) -> float | None:
    for prefix, tol in _HB_TOLERANCES:
        if col.startswith(prefix):
            return tol
    return 0.0  # 未知の列（param など）は完全一致を要求


def compare_csv(old: pd.DataFrame, new: pd.DataFrame, *, ignore_cols: set[str] = frozenset(), key_col: str | None = None) -> list[str]:
    diffs: list[str] = []

    old_cols = [c for c in old.columns if c not in ignore_cols]
    new_cols = [c for c in new.columns if c not in ignore_cols]
    if old_cols != new_cols:
        diffs.append(f"columns differ: old={old_cols} new={new_cols}")
        return diffs

    if key_col is not None:
        old = old.set_index(key_col)
        new = new.set_index(key_col)
        if set(old.index) != set(new.index):
            diffs.append(f"{key_col} values differ: old={sorted(old.index)} new={sorted(new.index)}")
            return diffs
        new = new.loc[old.index]
    else:
        if len(old) != len(new):
            diffs.append(f"row count differs: old={len(old)} new={len(new)}")
            return diffs

    for col in old_cols:
        if key_col is not None and col == key_col:
            continue
        tol = _tolerance_for_column(col)
        old_vals = old[col]
        new_vals = new[col]
        if pd.api.types.is_numeric_dtype(old_vals) and pd.api.types.is_numeric_dtype(new_vals):
            if tol is None:
                continue  # informational-only column (mcse)
            delta = (old_vals.astype(float) - new_vals.astype(float)).abs()
            bad = delta[delta > tol]
            if not bad.empty:
                for idx in bad.index:
                    diffs.append(f"{col}[{idx}]: old={old_vals[idx]} new={new_vals[idx]} delta={bad[idx]:.6g} (tol={tol})")
        else:
            mismatch = old_vals.astype(str) != new_vals.astype(str)
            for idx in old_vals.index[mismatch]:
                diffs.append(f"{col}[{idx}]: old={old_vals[idx]!r} new={new_vals[idx]!r}")

    return diffs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--kind", choices=["csv", "hb"], default="csv")
    args = parser.parse_args()

    if not args.old.exists():
        print(f"old file not found: {args.old}", file=sys.stderr)
        return 2
    if not args.new.exists():
        print(f"new file not found: {args.new}", file=sys.stderr)
        return 2

    old = pd.read_csv(args.old, encoding="utf-8-sig")
    new = pd.read_csv(args.new, encoding="utf-8-sig")

    if args.kind == "hb":
        diffs = compare_csv(old, new, ignore_cols=_HB_IGNORE_COLS, key_col="param")
    else:
        diffs = compare_csv(old, new)

    if not diffs:
        print(f"MATCH: {args.old} == {args.new} (within tolerance)")
        return 0

    print(f"MISMATCH: {args.old} vs {args.new} ({len(diffs)} differences)")
    for d in diffs:
        print(f"  {d}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
