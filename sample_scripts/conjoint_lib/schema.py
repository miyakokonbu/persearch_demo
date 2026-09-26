"""camera調査固有の設定と、調査に依存しないプラットフォーム定数。

属性・水準・raw符号化列といった調査固有の構成は`CAMERA_STUDY`（`study.StudyConfig`の
インスタンス）にまとめてある。別の調査に対応するには、この`CAMERA_STUDY`と同じ形で
新しい`StudyConfig`を作り、`conjoint_lib.prepare`/`counting`/`hb`/`predictive`の各関数に
`config=`として渡す（詳細は README.md）。
"""
from __future__ import annotations

import re

import pandas as pd

from .study import Attribute, Level, StudyConfig

LONG_KEY_COLS = ["respondent_id", "task_index", "option_label", "is_none", "chosen"]

# PerSearchの出力列名(構造列) -> このパッケージ内部での列名(LONG_KEY_COLS に揃える)。
# 属性列(ブランド・画素数等)の列名・水準ラベルは調査ごとに変わるため CAMERA_STUDY 側で持つ。
# こちらはPerSearchのExport形式（プラットフォーム側の仕様）なので調査によらず共通と想定する。
PERSEARCH_COLUMN_RENAME = {
    "UUID": "uuid",
    "課題番号": "task_index",
    "選択肢": "option_label",
    "どれも選ばない": "is_none",
    "選択された": "chosen",
}

# --- camera (実データ、bayesm::camera) の取り込み固有の定数 ---
# bayesm::camera の取り込み(load_camera_rdata/build_camera_long_df)はこのデータセット
# 固有の物理レイアウト（Xの列順など）に依存しており、他の調査では使わないため
# StudyConfigには含めていない。
CAMERA_N_RESP = 332
CAMERA_N_TASKS = 16
CAMERA_N_ALTS = 5  # A/B/C/D + none
OPTION_LABELS = ["A", "B", "C", "D"]  # none は別扱い

_PRICE_RE = re.compile(r"[\d,]+(?:\.\d+)?")


def parse_price_usd100(raw: object) -> float:
    """`"$129"`のような表記を100ドル単位の連続値(1.29)に変換する。パースできない値は例外にする。"""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        raise ValueError("price is missing")
    m = _PRICE_RE.search(str(raw).replace(",", ""))
    if not m:
        raise ValueError(f"price could not be parsed: {raw!r}")
    return float(m.group(0)) / 100.0


# --- camera調査の属性構成 ---
# 属性の宣言順が raw符号化(feature_names)・labeled long CSV の列順になる。
CAMERA_STUDY = StudyConfig(
    attributes=(
        Attribute(
            name="brand",
            ja_column="ブランド",
            levels=(
                Level(label="canon", ja_labels=("キヤノン",), raw_column="canon"),
                Level(label="sony", ja_labels=("ソニー",), raw_column="sony"),
                Level(label="nikon", ja_labels=("ニコン",), raw_column="nikon"),
                Level(label="panasonic", ja_labels=("パナソニック",), raw_column="panasonic"),
            ),
        ),
        Attribute(
            name="pixels",
            ja_column="画素数",
            levels=(
                Level(label="低画素", ja_labels=("標準画素数",)),
                Level(label="高画素", ja_labels=("高画素数",), raw_column="pixels"),
            ),
        ),
        Attribute(
            name="zoom",
            ja_column="ズーム",
            levels=(
                Level(label="標準ズーム", ja_labels=("標準ズーム",)),
                Level(label="高倍率ズーム", ja_labels=("高倍率ズーム",), raw_column="zoom"),
            ),
        ),
        Attribute(
            name="video",
            ja_column="動画撮影",
            levels=(
                Level(label="動画機能なし", ja_labels=("動画非対応",)),
                Level(label="動画機能あり", ja_labels=("動画対応",), raw_column="video"),
            ),
        ),
        Attribute(
            name="swivel",
            ja_column="可動式モニタ",
            levels=(
                Level(label="回転式液晶なし", ja_labels=("可動式モニタなし",)),
                Level(label="回転式液晶あり", ja_labels=("可動式モニタあり",), raw_column="swivel"),
            ),
        ),
        Attribute(
            name="wifi",
            ja_column="Wi-Fi",
            levels=(
                Level(label="Wi-Fiなし", ja_labels=("Wi-Fi非対応",)),
                Level(label="Wi-Fiあり", ja_labels=("Wi-Fi対応",), raw_column="wifi"),
            ),
        ),
        Attribute(
            name="price",
            ja_column="価格",
            continuous_column="price_usd100",
            parser=parse_price_usd100,
            expected_values=(0.79, 1.29, 1.79, 2.29, 2.79),
        ),
    ),
    persearch_sheet_name="回答データ（Long形式）",
    attribute_groups={
        "main5": ("brand", "pixels", "zoom", "wifi"),  # 5属性モデル(価格込み)。persearchの疑似データと直接比較する主分析
        "full": ("brand", "pixels", "zoom", "video", "swivel", "wifi"),  # 全属性モデル。頑健性チェック用
    },
)
