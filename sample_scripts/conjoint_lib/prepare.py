"""camera 実データ・PerSearch 疑似データの読み込み・変換・検証。

PerSearchシートの取り込み・raw符号化変換・検証は`study.StudyConfig`を受け取る形に
なっており、camera調査以外の選択型コンジョイント調査にも適用できる（既定値は
`schema.CAMERA_STUDY`）。camera実データの取り込み（bayesm::cameraのRData解析）は
このデータセット固有の物理レイアウトに依存するため、調査非依存の対象にはしていない。
"""
from __future__ import annotations

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
from .study import StudyConfig


# ---------------------------------------------------------------------------
# camera (実データ) -- bayesm::camera 固有の取り込み
# ---------------------------------------------------------------------------

def load_camera_rdata(path: Path) -> list[dict]:
    parsed = rdata.parser.parse_file(str(path))
    converted = rdata.conversion.convert(parsed)
    return converted["camera"]


def build_camera_long_df(camera: list[dict]) -> pd.DataFrame:
    """bayesm::camera の raw X 行列（brand4列+2値5列+price、camera調査固定のレイアウト）を
    CAMERA_STUDY の labeled long スキーマに変換する。"""
    study = schema.CAMERA_STUDY
    binary_attrs = [a.name for a in study.categorical_attributes() if a.name != "brand"]
    brand_labels = study.attribute("brand").level_labels()
    binary_levels = {a.name: (a.level_labels()[0], a.level_labels()[1]) for a in study.categorical_attributes() if a.name != "brand"}

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
                    features = {attr: None for attr in binary_attrs}
                    price_usd100 = None
                else:
                    brand_vals = row_vals[:4]
                    if brand_vals.sum() != 1:
                        raise ValueError(
                            f"respondent {respondent_id} task {task_index} alt {alt_idx}: "
                            f"brand dummies do not sum to 1 ({brand_vals})"
                        )
                    brand = brand_labels[int(np.argmax(brand_vals))]
                    features = {}
                    for i, attr in enumerate(binary_attrs):
                        flag = int(row_vals[4 + i])
                        features[attr] = binary_levels[attr][flag]
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


def validate_camera_long(df: pd.DataFrame) -> None:
    n_resp = df["respondent_id"].nunique()
    if n_resp != schema.CAMERA_N_RESP:
        raise AssertionError(f"respondent count mismatch: {n_resp}")
    expected_rows = schema.CAMERA_N_RESP * schema.CAMERA_N_TASKS * schema.CAMERA_N_ALTS
    if len(df) != expected_rows:
        raise AssertionError(f"row count mismatch: {len(df)}")
    _validate_task_structure(df, n_alt=schema.CAMERA_N_ALTS)


# ---------------------------------------------------------------------------
# 調査非依存の共通ロジック（StudyConfig 経由でどの調査にも適用できる）
# ---------------------------------------------------------------------------

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


def labeled_to_raw(df: pd.DataFrame, config: StudyConfig = schema.CAMERA_STUDY) -> pd.DataFrame:
    """カテゴリ文字列で符号化された labeled long DataFrame を raw 0/1・連続値の符号化に変換する。

    `camera_long.csv`（bayesm::camera由来）・PerSearchの labeled long のどちらにも使う。
    """
    out = df[schema.LONG_KEY_COLS].copy()
    is_product = df["is_none"] == 0

    for attr in config.categorical_attributes():
        value = df[attr.name]
        unknown = is_product & ~value.isin(attr.level_labels())
        if unknown.any():
            raise ValueError(f"{attr.name}: 未知の水準があります: {sorted(value[unknown].dropna().unique())}")
        for level in attr.levels:
            if level.raw_column is None:
                continue
            out[level.raw_column] = ((value == level.label) & is_product).astype(float)

    for attr in config.continuous_attributes():
        out[attr.continuous_column] = df[attr.continuous_column].where(is_product, 0.0).fillna(0.0)

    return out[schema.LONG_KEY_COLS + config.raw_columns()]


# 後方互換名。camera_long.csv の変換にこれまで通り使う。
camera_long_to_raw = labeled_to_raw


def load_persearch_sheet(path: Path, config: StudyConfig = schema.CAMERA_STUDY) -> pd.DataFrame:
    """PerSearchの回答データシートを読み、構造列を内部名に正規化する。

    PerSearch側の出力形式が変わり、構造列（UUID・課題番号・選択肢・どれも選ばない・
    選択された）の列名が変わることがあった（属性の水準ラベルは変わっていない）。
    想定した列名が見つからない場合は、原因がわかるように具体的なエラーにする。
    """
    df = pd.read_excel(path, sheet_name=config.persearch_sheet_name)

    missing = [c for c in schema.PERSEARCH_COLUMN_RENAME if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} の「{config.persearch_sheet_name}」シートに想定した列がありません: {missing}。"
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


def build_persearch_raw_long_df(raw: pd.DataFrame, config: StudyConfig = schema.CAMERA_STUDY) -> pd.DataFrame:
    df = raw.copy()
    df["respondent_id"] = assign_respondent_ids(df)
    is_product = df["is_none"] == 0

    for attr in config.categorical_attributes():
        # 未知の水準文字列がないか検証してから、水準ごとのダミー列を作る
        map_strict(df[attr.ja_column], attr.ja_level_map(), col=attr.ja_column, mask=is_product)
        for level in attr.levels:
            if level.raw_column is None:
                continue
            in_level = df[attr.ja_column].isin(level.ja_labels)
            df[level.raw_column] = (in_level & is_product).astype(float)

    for attr in config.continuous_attributes():
        df[attr.continuous_column] = [
            attr.parser(v) if is_prod else 0.0
            for v, is_prod in zip(df[attr.ja_column], is_product)
        ]

    out = df[schema.LONG_KEY_COLS + config.raw_columns()]
    return out.sort_values(["respondent_id", "task_index"]).reset_index(drop=True)


def build_persearch_labeled_long_df(raw: pd.DataFrame, config: StudyConfig = schema.CAMERA_STUDY) -> pd.DataFrame:
    """camera_long.csv と同じカテゴリ文字列表記の long DataFrame を作る（counting.py 用）。"""
    df = raw.copy()
    df["respondent_id"] = assign_respondent_ids(df)
    is_product = df["is_none"] == 0

    for attr in config.categorical_attributes():
        mapped = map_strict(df[attr.ja_column], attr.ja_level_map(), col=attr.ja_column, mask=is_product)
        df[attr.name] = mapped.where(is_product)

    for attr in config.continuous_attributes():
        df[attr.continuous_column] = [
            attr.parser(v) if is_prod else None
            for v, is_prod in zip(df[attr.ja_column], is_product)
        ]

    keep_cols = (
        schema.LONG_KEY_COLS
        + [a.name for a in config.categorical_attributes()]
        + [a.continuous_column for a in config.continuous_attributes()]
    )
    return df[keep_cols].sort_values(["respondent_id", "task_index"]).reset_index(drop=True)


def validate_raw_long(df: pd.DataFrame, *, expect_n_alt: int | None = None, config: StudyConfig = schema.CAMERA_STUDY) -> None:
    _validate_task_structure(df, n_alt=expect_n_alt)

    for attr in config.categorical_attributes():
        cols = attr.raw_columns()
        if len(cols) <= 1:
            # 基準水準コード(1列のみ)の属性は0/1どちらも正常値なので「合計が1」のチェックはできない
            continue
        level_sum = df.loc[df["is_none"] == 0, cols].sum(axis=1)
        if not (level_sum == 1).all():
            raise AssertionError(f"each product row must have exactly one {attr.name} flag set")

    for attr in config.continuous_attributes():
        if attr.expected_values is None:
            continue
        values = df.loc[df["is_none"] == 0, attr.continuous_column].round(2)
        bad = set(values.unique()) - set(attr.expected_values)
        if bad:
            raise AssertionError(f"unexpected {attr.name} levels: {sorted(bad)}")


def validate_labeled_long(df: pd.DataFrame, config: StudyConfig = schema.CAMERA_STUDY) -> None:
    is_product = df["is_none"] == 0

    for attr in config.categorical_attributes():
        if not df.loc[is_product, attr.name].isin(attr.level_labels()).all():
            raise AssertionError(f"unexpected level for {attr.name}")

    none_cols = (
        [a.name for a in config.categorical_attributes()]
        + [a.continuous_column for a in config.continuous_attributes()]
    )
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

    raw_df = labeled_to_raw(df)
    raw_path = camera_raw_long_csv(out_dir)
    raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")

    print(f"wrote {len(df)} rows to {long_path}")
    print(f"wrote {len(raw_df)} rows to {raw_path}")
    print(f"respondents: {df['respondent_id'].nunique()}, tasks/respondent: {schema.CAMERA_N_TASKS}")
    print("none choice rate:", df.loc[df["is_none"] == 1, "chosen"].mean())
    print("price levels (USD/100):", sorted(df["price_usd100"].dropna().unique()))

    return {"long": long_path, "raw_long": raw_path}


def run_prepare_persearch(
    *,
    xlsx_path: Path | None = None,
    out_dir: Path | None = None,
    config: StudyConfig = schema.CAMERA_STUDY,
) -> dict[str, Path]:
    xlsx_path = Path(xlsx_path) if xlsx_path is not None else (DATA_DIR / "persearch_camera.xlsx")
    out_dir = Path(out_dir) if out_dir is not None else RESULTS_DIR

    if not xlsx_path.exists():
        raise FileNotFoundError(f"{xlsx_path} が見つかりません。")

    raw_sheet = load_persearch_sheet(xlsx_path, config=config)

    raw_df = build_persearch_raw_long_df(raw_sheet, config=config)
    validate_raw_long(raw_df, config=config)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = persearch_raw_long_csv(out_dir)
    raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")

    n_resp = raw_df["respondent_id"].nunique()
    n_tasks = raw_df["task_index"].nunique()
    n_alt = len(raw_df) // (n_resp * n_tasks)
    print(f"wrote {len(raw_df)} rows to {raw_path}")
    print(f"respondents: {n_resp}, tasks/respondent: {n_tasks}, alts/task: {n_alt}")
    print("none choice rate:", raw_df.loc[raw_df["is_none"] == 1, "chosen"].mean())
    for attr in config.continuous_attributes():
        print(f"{attr.name} levels:", sorted(raw_df[attr.continuous_column].unique()))

    labeled_df = build_persearch_labeled_long_df(raw_sheet, config=config)
    validate_labeled_long(labeled_df, config=config)
    labeled_path = persearch_long_csv(out_dir)
    labeled_df.to_csv(labeled_path, index=False, encoding="utf-8-sig")
    print(f"wrote {len(labeled_df)} rows to {labeled_path}")

    return {"raw_long": raw_path, "long": labeled_path}


def load_raw_long(label: str, d: Path | None = None, config: StudyConfig = schema.CAMERA_STUDY) -> pd.DataFrame:
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
    raw_df = labeled_to_raw(pd.read_csv(long_path), config=config)
    raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    return raw_df
