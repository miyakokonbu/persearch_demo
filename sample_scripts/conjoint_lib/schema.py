"""属性・水準・ラベル対応など、パイプライン全体で共有する定数。ロジックは持たない。"""
from __future__ import annotations

BRANDS = ["canon", "sony", "nikon", "panasonic"]
BINARY_ATTRS = ["pixels", "zoom", "video", "swivel", "wifi"]
RAW_COLUMNS = BRANDS + BINARY_ATTRS + ["price_usd100"]

LONG_KEY_COLS = ["respondent_id", "task_index", "option_label", "is_none", "chosen"]

# 属性 -> (基準水準, 上位水準) のカテゴリ文字列
BINARY_LEVELS: dict[str, tuple[str, str]] = {
    "pixels": ("低画素", "高画素"),
    "zoom": ("標準ズーム", "高倍率ズーム"),
    "video": ("動画機能なし", "動画機能あり"),
    "swivel": ("回転式液晶なし", "回転式液晶あり"),
    "wifi": ("Wi-Fiなし", "Wi-Fiあり"),
}

ATTRIBUTE_LEVEL_ORDER: dict[str, list[str]] = {
    "brand": BRANDS,
    **{attr: list(levels) for attr, levels in BINARY_LEVELS.items()},
}

# カウンティング法(counting.py)で使う属性グループ
MAIN_ATTRS = ["brand", "pixels", "zoom", "wifi"]
# 全属性モデル。camera・persearch のどちらも video/swivel を持つため両方で生成される。
FULL_ATTRS = ["brand", "pixels", "zoom", "video", "swivel", "wifi"]

# --- camera (実データ) ---
CAMERA_N_RESP = 332
CAMERA_N_TASKS = 16
CAMERA_N_ALTS = 5  # A/B/C/D + none
OPTION_LABELS = ["A", "B", "C", "D"]  # none は別扱い

# --- PerSearch (疑似データ) の日本語ラベル対応 ---
PERSEARCH_SHEET_NAME = "回答データ（Long形式）"
PERSEARCH_PRICE_COL = "価格"
PERSEARCH_BRAND_COL = "ブランド"

# PerSearch側の出力列名(構造列) -> このパッケージ内部での列名(LONG_KEY_COLS に揃える)。
# 属性列(ブランド・画素数等)は日本語のまま扱うためここには含めない。
PERSEARCH_COLUMN_RENAME = {
    "UUID": "uuid",
    "課題番号": "task_index",
    "選択肢": "option_label",
    "どれも選ばない": "is_none",
    "選択された": "chosen",
}

PERSEARCH_BRAND_MAP = {"キヤノン": "canon", "ソニー": "sony", "ニコン": "nikon", "パナソニック": "panasonic"}

# 属性の日本語列名 -> raw 0/1 列名(英語)
PERSEARCH_ATTR_COLS = {
    "画素数": "pixels",
    "ズーム": "zoom",
    "動画撮影": "video",
    "可動式モニタ": "swivel",
    "Wi-Fi": "wifi",
}

# 属性の日本語列名 -> {日本語水準: 0/1}
PERSEARCH_BINARY_MAPS: dict[str, dict[str, int]] = {
    "画素数": {"標準画素数": 0, "高画素数": 1},
    "ズーム": {"標準ズーム": 0, "高倍率ズーム": 1},
    "動画撮影": {"動画非対応": 0, "動画対応": 1},
    "可動式モニタ": {"可動式モニタなし": 0, "可動式モニタあり": 1},
    "Wi-Fi": {"Wi-Fi非対応": 0, "Wi-Fi対応": 1},
}


def _derive_persearch_label_maps() -> dict[str, dict[str, str]]:
    # PERSEARCH_BINARY_MAPS (0/1) と BINARY_LEVELS (基準/上位のカテゴリ文字列) から導出する。
    # 手書きの対応表をもう一つ持つと、PerSearch側の表記が変わったときに片方だけ更新されて
    # 2つの出力(raw / labeled)が無言でズレる恐れがあるため、常にこちらから計算する。
    out: dict[str, dict[str, str]] = {}
    for ja_col, en_col in PERSEARCH_ATTR_COLS.items():
        ref, treat = BINARY_LEVELS[en_col]
        out[ja_col] = {
            ja_level: (treat if value == 1 else ref)
            for ja_level, value in PERSEARCH_BINARY_MAPS[ja_col].items()
        }
    return out


# 属性の日本語列名 -> {日本語水準: camera_long.csv と同じカテゴリ文字列}
PERSEARCH_LABEL_MAPS: dict[str, dict[str, str]] = _derive_persearch_label_maps()

EXPECTED_PRICES_USD100 = (0.79, 1.29, 1.79, 2.29, 2.79)
