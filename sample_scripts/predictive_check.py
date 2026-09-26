"""hb_diag.ipynb で推定した beta_bar が、選択行動そのものをどれだけ予測できるかを検証する。

符号の一致率・Spearman順位相関は「疑似データの係数が実データの係数とどれだけ一致するか」を
見る指標だが、ここでは「係数がズレていても、実際の選択行動の予測には使えるか」という、
より実務に近い問いを検証する。real-on-real（理論上の上限）・pseudo-on-real（本題）・
uniform（何もしない場合の下限）の3通りで、対数尤度（確率としての較正）とtop-1的中率
（複数候補のうちどれが選ばれるかの的中率）を比較する。

--tau オプションで beta_bar に掛ける温度（temperature）を指定すると、
「疑似データの係数を一律に縮小すると較正が改善するか」を確認できる
（top-1的中率は softmax の argmax が符号のスケールに依存しないため変化しない）。

使い方:
    # 実データに対する予測性能（前提: hb_diag.ipynb で両データセットの推定が済んでいること）
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 .venv/Scripts/python sample_scripts/predictive_check.py

    # 疑似データに対する予測性能（逆方向）も見る
    .venv/Scripts/python sample_scripts/predictive_check.py --direction pseudo

    # beta_bar を0.3倍に縮小した場合の較正を確認する
    .venv/Scripts/python sample_scripts/predictive_check.py --tau 0.3

    # tau を範囲指定して最適値を探す（結果は results/predictive_check_{direction}_tauscan.csv に保存）
    .venv/Scripts/python sample_scripts/predictive_check.py --tau-scan 0.1 1.2 0.1

通常モード（--tau-scan なし）の3行比較は results/predictive_check_{direction}_summary.csv
にも保存する。
"""
from __future__ import annotations

import argparse

from conjoint_lib.predictive import run_predictive_check, run_tau_scan


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--direction", choices=["real", "pseudo"], default="real",
                        help="どちらのデータセットの実際の選択を予測対象にするか（既定: real）")
    parser.add_argument("--tau", type=float, default=1.0,
                        help="評価対象モデルの beta_bar に掛ける温度（既定: 1.0 = 補正なし）")
    parser.add_argument("--tau-scan", nargs=3, type=float, metavar=("START", "STOP", "STEP"),
                        help="tau を START から STOP まで STEP 刻みで走査し、結果を CSV に保存する")
    parser.add_argument("--in-dir", default=None, help="既定: results/")
    parser.add_argument("--out-dir", default=None, help="既定: results/")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.tau_scan:
        start, stop, step = args.tau_scan
        run_tau_scan(args.direction, start, stop, step, in_dir=args.in_dir, out_dir=args.out_dir)
    else:
        run_predictive_check(args.direction, tau=args.tau, in_dir=args.in_dir, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
