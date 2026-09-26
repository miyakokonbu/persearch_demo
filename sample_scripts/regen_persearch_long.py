"""prepare_persearch_camera.ipynb 相当の処理を notebook を開かず CLI から実行する。

`data/persearch_camera.xlsx` を PerSearch で再生成・差し替えたときに、notebook を開かず
コマンド一発で `results/persearch_camera_raw_long.csv` / `persearch_camera_long.csv` を
再生成するための補助スクリプト（Windows のコンソール既定エンコーディング(cp932)では
notebook 内の日本語文字列をそのまま実行できないため、UTF-8 前提の CLI 版として用意している）。

使い方:
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 .venv/Scripts/python sample_scripts/regen_persearch_long.py

注意: 日本語ラベルの対応表は `conjoint_lib.schema` にあり、PerSearch の出力文言が
変わるとズレて `ValueError` になる（意図的に検出するようにしてある。過去に「動画撮影」列で
実際の値が「動画非対応/動画対応」なのに「非対応/対応」と誤って決め打ちしており、
気づかずに実行すると該当列が全行0に埋められてしまう事故があったため）。
xlsx を差し替えたときは、実行前に `df[col].unique()` で実際のラベル文言を確認すること。
"""
from __future__ import annotations

import argparse

from conjoint_lib.prepare import run_prepare_persearch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xlsx", default=None, help="既定: data/persearch_camera.xlsx")
    parser.add_argument("--out-dir", default=None, help="既定: results/")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_prepare_persearch(xlsx_path=args.xlsx, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
