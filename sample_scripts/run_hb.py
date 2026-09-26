"""hb_diag.ipynb 相当の階層ベイズMNL推定を notebook を開かず CLI から実行する。

`hb_diag.ipynb` の「実行設定」セルを書き換えて実行する代わりに使う。xlsx を差し替えた後の
再推定や、リファクタリング前後で数値が変わっていないかの検証（scratch ディレクトリへ
出力して `compare_results.py` で比較する）に使う想定（Windows のコンソール既定エンコー
ディング(cp932)では日本語文字列がそのまま実行できないため、UTF-8 前提のCLI版として
用意している）。

使い方:
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 .venv/Scripts/python sample_scripts/run_hb.py persearch
    .venv/Scripts/python sample_scripts/run_hb.py camera --seed 42 --out-dir /tmp/scratch
    .venv/Scripts/python sample_scripts/run_hb.py persearch --draws 50 --tune 50 --chains 1 --n-resp-limit 20
"""
from __future__ import annotations

import argparse

from conjoint_lib.hb import DEFAULT_PRIORS, DEFAULT_SEED, run_hb


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("label", choices=["camera", "persearch"])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument("--n-resp-limit", type=int, default=None, help="動作確認だけしたい場合に回答者数を絞る")
    parser.add_argument("--beta-bar-scale", type=float, default=DEFAULT_PRIORS["beta_bar_scale"])
    parser.add_argument("--sigma-beta-scale", type=float, default=DEFAULT_PRIORS["sigma_beta_scale"])
    parser.add_argument("--tag", default=None, help="非デフォルトの事前分布を使う場合は必須")
    parser.add_argument("--in-dir", default=None, help="raw_long.csv の読み込み元（既定: results/）")
    parser.add_argument("--out-dir", default=None, help="出力先（既定: results/）。検証時はscratchディレクトリを指定する")
    parser.add_argument("--no-trace", action="store_true", help="trace.nc を保存しない")
    parser.add_argument("--no-progressbar", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_hb(
        args.label,
        out_dir=args.out_dir,
        in_dir=args.in_dir,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        target_accept=args.target_accept,
        seed=args.seed,
        n_resp_limit=args.n_resp_limit,
        beta_bar_scale=args.beta_bar_scale,
        sigma_beta_scale=args.sigma_beta_scale,
        tag=args.tag,
        save_trace=not args.no_trace,
        progressbar=not args.no_progressbar,
    )


if __name__ == "__main__":
    main()
