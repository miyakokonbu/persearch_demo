"""hb_diag.ipynb の事前分布(prior)を差し替えて再実行するための prior sensitivity check ツール。

「実データ/疑似データの sigma_beta 比（個人間異質性の過小評価の度合い）」や
「beta_bar の Spearman 順位相関の低さ」が、事前分布の選び方に依存して動くアーティファクト
なのか、データに由来する頑健な結果なのかを切り分けるために作成した。

hb_diag.ipynb のモデル定義（`conjoint_lib.hb`）をそのまま流用し、sigma_beta / beta_bar
それぞれの事前分布のスケールだけを引数化している。ベースライン(sigma_beta=2.5,
beta_bar=5.0)は既存の `results/hb_diag_*_summary.csv`（notebook の実行結果）をそのまま
比較対象にできる。

CLIとして使う場合:
    # sigma_beta の HalfNormal スケールを振る（beta_bar は既定値 5.0 で固定）
    .venv/Scripts/python sample_scripts/prior_sensitivity.py camera sigma 1.0
    .venv/Scripts/python sample_scripts/prior_sensitivity.py persearch sigma 5.0

    # beta_bar の Normal スケールを振る（sigma_beta は既定値 2.5 で固定）
    .venv/Scripts/python sample_scripts/prior_sensitivity.py camera beta 2.0
    .venv/Scripts/python sample_scripts/prior_sensitivity.py persearch beta 10.0

出力は `results/hb_diag_{label}_sigmascale{scale}_summary.csv` /
`results/hb_diag_{label}_betascale{scale}_summary.csv` に保存される
（`hb_diag.ipynb` 本来の出力 `hb_diag_{label}_summary.csv` は上書きしない。
`conjoint_lib.hb.run_hb` が非デフォルト事前分布に対して tag を必須にしているため、
上書きしようとすると `ValueError` になる）。

`conjoint_lib.hb.run_hb()` を直接呼び出せば、任意の `tag` を指定して結果を保存できる
（xlsx を差し替えて疑似データそのものを更新した場合の再推定など）。trace は保存しない
（`--out-dir` で出力先を変えれば `results/` を汚さずに試せる）。
"""
from __future__ import annotations

import argparse

from conjoint_lib.hb import DEFAULT_PRIORS, DEFAULT_SEED, prior_tag, run_hb


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="hb_diag.ipynb の事前分布(prior)を差し替えて再実行する prior sensitivity check。",
    )
    parser.add_argument("label", choices=["camera", "persearch"], help="実データ(camera) / 疑似データ(persearch)")
    parser.add_argument("mode", choices=["sigma", "beta"], help="sigma: sigma_beta の事前分布を振る / beta: beta_bar の事前分布を振る")
    parser.add_argument("scale", type=float, help="HalfNormal(sigma) または Normal(0, scale) のスケール値")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", default=None, help="既定: results/")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.mode == "sigma":
        sigma_beta_scale, beta_bar_scale = args.scale, DEFAULT_PRIORS["beta_bar_scale"]
    else:
        sigma_beta_scale, beta_bar_scale = DEFAULT_PRIORS["sigma_beta_scale"], args.scale

    tag = prior_tag(sigma_beta_scale, beta_bar_scale)
    written = run_hb(
        args.label,
        sigma_beta_scale=sigma_beta_scale,
        beta_bar_scale=beta_bar_scale,
        seed=args.seed,
        tag=tag,
        out_dir=args.out_dir,
        save_trace=False,
        progressbar=False,
    )
    print(f"wrote {written['summary']}")


if __name__ == "__main__":
    main()
