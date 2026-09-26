"""camera 実データ・PerSearch 疑似データの読み込み・変換・検証。"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import rdata

from . import schema
from .paths import (
    DATA_DIR,
    RESULTS_DIR,
    camera_long_csv,
    camera_raw_long_csv,
    persearch_long_csv,
    persearch_raw_long_csv,
)

_PRICE_RE = re.compile(r"[\d,]+(?:\.\d+)?")


# ---------------------------------------------------------------------------
# camera (実データ)
# ---------------------------------------------------------------------------

def load_camera_rdata(path: Path) -> list[dict]:
    parsed = rdata.parser.parse_file(str(path))
    converted = rdata.conversion.convert(parsed)
    return converted["camera"]


def build_camera_long_df(camera: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for resp_idx, resp in enumerate(camera):
        respondent_id = resp_idx + 1
        y = np.asarray(resp["y"]).astype(int)  # 長さ16、1〜5
        X = np.asarray(resp["X"].values).astype(float)  # (80, 10)

        if X.shape != (schema.CAMERA_N_TASKS * schema.CAMERA_N_ALTS, 10):
            raise ValueError(f"respondent {respondent_id}: unexpected X shape {X.shape}")
        if len(y) != schema.CAMERA_N_TASKS:
            raise ValueError(f"respondent {respondent_id}: unexpected y length {len(y)}")

        for task_index in range(schema.CAMERA_N_TASKS):
            block = X[task_index * schema.CAMERA_N_ALTS : (task_index + 1) * schema.CAMERA_N_ALTS]
            chosen_option_number = int(y[task_index])  # 1〜5

            for alt_idx in range(schema.CAMERA_N_ALTS):
                row_vals = block[alt_idx]
                is_none = alt_idx == schema.CAMERA_N_ALTS - 1  # 5番目(index4)が none 行
                option_number = alt_idx + 1
                option_label = "none" if is_none else schema.OPTION_LABELS[alt_idx]

                if is_none:
                    brand = None
                    features = {attr: None for attr in schema.BINARY_ATTRS}
                    price_usd100 = None
                else:
                    brand_vals = row_vals[:4]
                    if brand_vals.sum() != 1:
                        raise ValueError(
                            f"respondent {respondent_id} task {task_index} alt {alt_idx}: "
                            f"brand dummies do not sum to 1 ({brand_vals})"
                        )
                    brand = schema.BRANDS[int(np.argmax(brand_vals))]
                    features = {}
                    for i, attr in enumerate(schema.BINARY_ATTRS):
                        flag = int(row_vals[4 + i])
                        features[attr] = schema.BINARY_LEVELS[attr][flag]
                    price_usd100 = float(row_vals[9])

                rows.append(
                    {
                        "respondent_id": respondent_id,
                        "task_index": task_index,
                        "option_label": option_label,
                        "is_none": int(is_none),
                        "chosen": int(option_number == chosen_option_number),
                        "brand": brand,
                        **features,
                        "price_usd100": price_usd100,
                    }
                )
    return pd.DataFrame(rows)


def _validate_task_structure(df: pd.DataFrame, *, n_alt: int | None = None) -> None:
    """1課題ごとに: chosen行がちょうど1つ・none行がちょうど1つ・全課題で行数が揃っていること。"""
    chosen_per_task = df.groupby(["respondent_id", "task_index"])["chosen"].sum()
    if not (chosen_per_task == 1).all():
        raise AssertionError("each task must have exactly one chosen row")

    none_per_task = df.groupby(["respondent_id", "task_index"])["is_none"].sum()
    if not (none_per_task == 1).all():
        raise AssertionError("each task must have exactly one none row")

    rows_per_task = df.groupby(["respondent_id", "task_index"]).size()
    if n_alt is not None:
        if not (rows_per_task == n_alt).all():
            raise AssertionError(f"each task must have exactly {n_alt} rows")
    elif rows_per_task.nunique() != 1:
        raise AssertionError("tasks do not all have the same number of rows")


def validate_camera_long(df: pd.DataFrame) -> None:
    n_resp = df["respondent_id"].nunique()
    if n_resp != schema.CAMERA_N_RESP:
        raise AssertionError(f"respondent count mismatch: {n_resp}")
    expected_rows = schema.CAMERA_N_RESP * schema.CAMERA_N_TASKS * schema.CAMERA_N_ALTS
    if len(df) != expected_rows:
        raise AssertionError(f"row count mismatch: {len(df)}")
    _validate_task_structure(df, n_alt=schema.CAMERA_N_ALTS)


def camera_long_to_raw(df: pd.DataFrame) -> pd.DataFrame:
    """`camera_long.csv`（カテゴリ文字列の符号化）を raw 0/1・ドル価格の符号化に変換する。"""
    out = df[schema.LONG_KEY_COLS].copy()
    is_product = df["is_none"] == 0

    for brand in schema.BRANDS:
        out[brand] = ((df["brand"] == brand) & is_product).astype(float)

    for attr in schema.BINARY_ATTRS:
        ref, treat = schema.BINARY_LEVELS[attr]
        value = df[attr]
        unknown = is_product & ~value.isin([ref, treat])
        if unknown.any():
            raise ValueError(f"{attr}: 未知の水準があります: {sorted(value[unknown].dropna().unique())}")
        out[attr] = ((value == treat) & is_product).astype(float)

    out["price_usd100"] = df["price_usd100"].where(is_product, 0.0).fillna(0.0)

    return out[schema.LONG_KEY_COLS + schema.RAW_COLUMNS]


# ---------------------------------------------------------------------------
# PerSearch (疑似データ)
# ---------------------------------------------------------------------------

def parse_price_usd100(raw: str | float) -> float:
    """`"$129"`のような表記を100ドル単位の連続値(1.29)に変換する。パースできない値は例外にする

    （旧実装は不正値を黙って`0.0`にしており、データ破損に気づけなかった）。
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        raise ValueError("price is missing")
    m = _PRICE_RE.search(str(raw).replace(",", ""))
    if not m:
        raise ValueError(f"price could not be parsed: {raw!r}")
    return float(m.group(0)) / 100.0


def load_persearch_sheet(path: Path) -> pd.DataFrame:
    """PerSearchの「回答データ（Long形式）」シートを読み、構造列を内部名に正規化する。

    PerSearch側の出力形式が変わり、構造列（UUID・課題番号・選択肢・どれも選ばない・
    選択された）の列名が変わることがあった（属性の水準ラベルは変わっていない）。
    想定した列名が見つからない場合は、原因がわかるように具体的なエラーにする。
    """
    df = pd.read_excel(path, sheet_name=schema.PERSEARCH_SHEET_NAME)

    missing = [c for c in schema.PERSEARCH_COLUMN_RENAME if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} の「{schema.PERSEARCH_SHEET_NAME}」シートに想定した列がありません: {missing}。"
            f"PerSearchの出力形式が変わっていないか確認し、変わっていれば "
            f"conjoint_lib.schema.PERSEARCH_COLUMN_RENAME を更新してください。"
            f"（実際の列: {df.columns.tolist()}）"
        )
    df = df.rename(columns=schema.PERSEARCH_COLUMN_RENAME)

    # 課題番号の開始値がPerSearch側の出力仕様（0始まり/1始まり等）に依存しないよう、
    # 回答者ごとの最小値を引いて0始まりに揃える（camera_long.csv の task_index と同じ規約）。
    df["task_index"] = df["task_index"] - df.groupby("uuid")["task_index"].transform("min")

    return df


def assign_respondent_ids(df: pd.DataFrame) -> pd.Series:
    """uuid（ペルソナID）を出現順の連番 respondent_id に変換する。"""
    uuid_order = df["uuid"].drop_duplicates().tolist()
    uuid_to_id = {u: i + 1 for i, u in enumerate(uuid_order)}
    return df["uuid"].map(uuid_to_id)


def map_strict(series: pd.Series, mapping: dict, *, col: str, mask: pd.Series) -> pd.Series:
    """`mask`が立っている行だけmappingで変換し、変換できない値があれば例外にする。

    未知のラベルを黙って0/NaN埋めしていた旧実装のバグ修正（過去に「動画撮影」列が
    ラベル文言のズレに気づかず全行0になった実インシデントの原因）。
    """
    mapped = series.map(mapping)
    unmapped = mask & mapped.isna()
    if unmapped.any():
        bad_vals = sorted(series[unmapped].unique())
        raise ValueError(f"{col}: マッピングされていない値があります: {bad_vals}")
    return mapped


def build_persearch_raw_long_df(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["respondent_id"] = assign_respondent_ids(df)
    is_product = df["is_none"] == 0

    map_strict(
        df[schema.PERSEARCH_BRAND_COL], schema.PERSEARCH_BRAND_MAP,
        col=schema.PERSEARCH_BRAND_COL, mask=is_product,
    )
    ja_by_brand = {en: ja for ja, en in schema.PERSEARCH_BRAND_MAP.items()}
    for brand in schema.BRANDS:
        df[brand] = ((df[schema.PERSEARCH_BRAND_COL] == ja_by_brand[brand]) & is_product).astype(float)

    for ja_col, raw_col in schema.PERSEARCH_ATTR_COLS.items():
        mapped = map_strict(df[ja_col], schema.PERSEARCH_BINARY_MAPS[ja_col], col=ja_col, mask=is_product)
        df[raw_col] = mapped.where(is_product, 0.0).fillna(0.0).astype(float)

    df["price_usd100"] = [
        parse_price_usd100(v) if is_prod else 0.0
        for v, is_prod in zip(df[schema.PERSEARCH_PRICE_COL], is_product)
    ]

    out = df[schema.LONG_KEY_COLS + schema.RAW_COLUMNS]
    return out.sort_values(["respondent_id", "task_index"]).reset_index(drop=True)


def build_persearch_labeled_long_df(raw: pd.DataFrame) -> pd.DataFrame:
    """camera_long.csv と同じカテゴリ文字列表記の long DataFrame を作る（counting.py 用）。"""
    df = raw.copy()
    df["respondent_id"] = assign_respondent_ids(df)
    is_product = df["is_none"] == 0

    brand_mapped = map_strict(
        df[schema.PERSEARCH_BRAND_COL], schema.PERSEARCH_BRAND_MAP,
        col=schema.PERSEARCH_BRAND_COL, mask=is_product,
    )
    df["brand"] = brand_mapped.where(is_product)

    for ja_col, attr in schema.PERSEARCH_ATTR_COLS.items():
        mapped = map_strict(df[ja_col], schema.PERSEARCH_LABEL_MAPS[ja_col], col=ja_col, mask=is_product)
        df[attr] = mapped.where(is_product)

    df["price_usd100"] = [
        parse_price_usd100(v) if is_prod else None
        for v, is_prod in zip(df[schema.PERSEARCH_PRICE_COL], is_product)
    ]

    keep_cols = schema.LONG_KEY_COLS + ["brand"] + schema.BINARY_ATTRS + ["price_usd100"]
    return df[keep_cols].sort_values(["respondent_id", "task_index"]).reset_index(drop=True)


def validate_raw_long(df: pd.DataFrame, *, expect_n_alt: int | None = None) -> None:
    _validate_task_structure(df, n_alt=expect_n_alt)

    brand_sum = df.loc[df["is_none"] == 0, schema.BRANDS].sum(axis=1)
    if not (brand_sum == 1).all():
        raise AssertionError("each product row must have exactly one brand flag set")

    prices = df.loc[df["is_none"] == 0, "price_usd100"].round(2)
    bad_prices = set(prices.unique()) - set(schema.EXPECTED_PRICES_USD100)
    if bad_prices:
        raise AssertionError(f"unexpected price levels: {sorted(bad_prices)}")


def validate_labeled_long(df: pd.DataFrame) -> None:
    is_product = df["is_none"] == 0

    if not df.loc[is_product, "brand"].isin(schema.BRANDS).all():
        raise AssertionError("unexpected brand level in labeled long df")

    for attr in schema.BINARY_ATTRS:
        levels = set(schema.BINARY_LEVELS[attr])
        if not df.loc[is_product, attr].isin(levels).all():
            raise AssertionError(f"unexpected level for {attr}")

    none_cols = ["brand", *schema.BINARY_ATTRS, "price_usd100"]
    if not df.loc[~is_product, none_cols].isna().all().all():
        raise AssertionError("none rows must be all-null")


# ---------------------------------------------------------------------------
# オーケストレータ（notebook・CLIスクリプト共通の実行本体）
# ---------------------------------------------------------------------------

def run_prepare_camera(*, data_dir: Path | None = None, out_dir: Path | None = None) -> dict[str, Path]:
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    out_dir = Path(out_dir) if out_dir is not None else RESULTS_DIR

    rdata_path = data_dir / "camera.RData"
    if not rdata_path.exists():
        raise FileNotFoundError(f"{rdata_path} が見つかりません。README の手順で取得してください。")

    camera = load_camera_rdata(rdata_path)
    df = build_camera_long_df(camera)
    validate_camera_long(df)

    out_dir.mkdir(parents=True, exist_ok=True)
    long_path = camera_long_csv(out_dir)
    df.to_csv(long_path, index=False, encoding="utf-8-sig")

    raw_df = camera_long_to_raw(df)
    raw_path = camera_raw_long_csv(out_dir)
    raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")

    print(f"wrote {len(df)} rows to {long_path}")
    print(f"wrote {len(raw_df)} rows to {raw_path}")
    print(f"respondents: {df['respondent_id'].nunique()}, tasks/respondent: {schema.CAMERA_N_TASKS}")
    print("none choice rate:", df.loc[df["is_none"] == 1, "chosen"].mean())
    print("price levels (USD/100):", sorted(df["price_usd100"].dropna().unique()))

    return {"long": long_path, "raw_long": raw_path}


def run_prepare_persearch(*, xlsx_path: Path | None = None, out_dir: Path | None = None) -> dict[str, Path]:
    xlsx_path = Path(xlsx_path) if xlsx_path is not None else (DATA_DIR / "persearch_camera.xlsx")
    out_dir = Path(out_dir) if out_dir is not None else RESULTS_DIR

    if not xlsx_path.exists():
        raise FileNotFoundError(f"{xlsx_path} が見つかりません。")

    raw_sheet = load_persearch_sheet(xlsx_path)

    raw_df = build_persearch_raw_long_df(raw_sheet)
    validate_raw_long(raw_df)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = persearch_raw_long_csv(out_dir)
    raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")

    n_resp = raw_df["respondent_id"].nunique()
    n_tasks = raw_df["task_index"].nunique()
    n_alt = len(raw_df) // (n_resp * n_tasks)
    print(f"wrote {len(raw_df)} rows to {raw_path}")
    print(f"respondents: {n_resp}, tasks/respondent: {n_tasks}, alts/task: {n_alt}")
    print("none choice rate:", raw_df.loc[raw_df["is_none"] == 1, "chosen"].mean())
    print("price levels (USD/100):", sorted(raw_df["price_usd100"].unique()))

    labeled_df = build_persearch_labeled_long_df(raw_sheet)
    validate_labeled_long(labeled_df)
    labeled_path = persearch_long_csv(out_dir)
    labeled_df.to_csv(labeled_path, index=False, encoding="utf-8-sig")
    print(f"wrote {len(labeled_df)} rows to {labeled_path}")

    return {"raw_long": raw_path, "long": labeled_path}


def load_raw_long(label: str, d: Path | None = None) -> pd.DataFrame:
    """`{label}_raw_long.csv`を読む。camera側はキャッシュが無ければ`camera_long.csv`から作る。"""
    if label not in ("camera", "persearch"):
        raise ValueError(f"unknown label: {label!r}")
    d = Path(d) if d is not None else RESULTS_DIR

    if label == "persearch":
        path = persearch_raw_long_csv(d)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} が見つかりません。prepare_persearch_camera.ipynb（または regen_persearch_long.py）を先に実行してください。"
            )
        return pd.read_csv(path)

    raw_path = camera_raw_long_csv(d)
    if raw_path.exists():
        return pd.read_csv(raw_path)

    long_path = camera_long_csv(d)
    if not long_path.exists():
        raise FileNotFoundError(f"{long_path} が見つかりません。prepare_camera.ipynb を先に実行してください。")
    raw_df = camera_long_to_raw(pd.read_csv(long_path))
    raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    return raw_df
