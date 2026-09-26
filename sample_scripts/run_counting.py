"""counting.ipynb 相当の処理を notebook を開かず CLI から実行する。

`results/camera_long.csv` / `results/persearch_camera_long.csv` からカウンティング法
（選択率の単純集計）で効用値・属性重要度を算出し、`counting_{camera,persearch_camera}_
{main5,full}(_importance).csv` を再生成する。notebook を開かず、疑似データ（xlsx）を
差し替えたあとの再集計をコマンド一発で行うための補助スクリプト。

使い方:
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 .venv/Scripts/python sample_scripts/run_counting.py

前提: `results/persearch_camera_long.csv` が最新であること
（xlsx を差し替えた場合は先に `regen_persearch_long.py` を実行する）。
"""
from __future__ import annotations

import argparse

from conjoint_lib.counting import run_counting_all


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in-dir", default=None, help="既定: results/")
    parser.add_argument("--out-dir", default=None, help="既定: results/")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_counting_all(in_dir=args.in_dir, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
