# sample_scripts

`camera` 実データ（R `bayesm` パッケージ）と PerSearch で再現した疑似データを比較する分析コード。
Qiita 記事「生成AIで作った疑似データでコンジョイント分析やってみた」シリーズで使用。

4本の Jupyter notebook（`.ipynb`）で構成する。データの読み込み・変換・カウンティング法・
階層ベイズMNL・予測性能チェックのロジックは `sample_scripts/conjoint_lib/` パッケージに
まとまっており、notebook・CLIスクリプトの両方がそこから import する（notebook・スクリプトが
同じロジックを別々に持って挙動がズレる、という問題を避けるため）。`conjoint_lib` は
`data/` `results/` の場所を自分の `__file__` から解決するため、notebook（カレントディレクトリ
= `sample_scripts/`）・スクリプト（実行ディレクトリ不問）のどちらから使っても同じ結果になる。

| notebook | 役割 |
|---|---|
| `prepare_camera.ipynb` | 実データ（`camera.RData`）→ `camera_long.csv` / `camera_raw_long.csv` |
| `prepare_persearch_camera.ipynb` | 疑似データ（`persearch_camera.xlsx`）→ `persearch_camera_*_long.csv` |
| `counting.ipynb` | カウンティング法（選択率の単純集計）で効用値・属性重要度を推定 |
| `hb_diag.ipynb` | 階層ベイズ MNL（対角共分散、PyMC）で同じ比較をより精緻に行う |

上から順に実行する（後段は前段の出力 CSV を読む）。

このほか、上記 notebook のロジックを notebook を開かず CLI から実行できるようにした
補助スクリプトが6本ある（Windows のコンソール既定エンコーディング(cp932)では notebook 内の
日本語文字列をそのまま実行できないため、`PYTHONUTF8=1 PYTHONIOENCODING=utf-8` を付けて
実行する前提の CLI 版として用意している）。

| script | 役割 |
|---|---|
| `regen_persearch_long.py` | `prepare_persearch_camera.ipynb` 相当。`persearch_camera.xlsx` を差し替えたときの再生成用 |
| `run_counting.py` | `counting.ipynb` 相当。カウンティング法の結果を一括再生成する |
| `run_hb.py` | `hb_diag.ipynb` 相当。階層ベイズMNLの推定を notebook を開かず実行する |
| `prior_sensitivity.py` | `hb_diag.ipynb` の事前分布(prior)を差し替えて再実行する prior sensitivity check 用 |
| `predictive_check.py` | 一方のデータセットで推定した beta_bar で、もう一方の実際の選択を予測できるかを検証する |
| `compare_results.py` | 2つの結果CSV（`results/` 配下）を許容誤差つきで比較する検証ツール。リファクタリングやバグ修正の前後で数値が変わっていないかを確認するために使う |

`regen_persearch_long.py`・`run_counting.py` 以外は `--help` で使い方を確認できる
（`regen_persearch_long.py`・`run_counting.py` は引数を取らないため、ファイル冒頭の docstring を参照）。

## セットアップ

**Python 3.12 が必要**（`pymc`/`pytensor` は本記事執筆時点で Python 3.13/3.14 向けのビルド済みホイールを配布していないため、システムの既定が新しいバージョンの場合は 3.12 を明示的に指定すること）。

### uv を使う場合（推奨）

```
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt jupyter ipykernel
```

### venv + pip を使う場合

```
python3.12 -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt jupyter ipykernel
```

`requirements.txt` には動作確認済みのバージョンを固定してある。バージョンを緩めて
再現できなくなった場合は、まずこのファイルの値に戻して切り分けること。
`jupyter`/`ipykernel` は notebook 実行専用で `requirements.txt` には含めていない
（このリポジトリの分析コード自体が依存するライブラリではないため）。

### notebook の起動

```
.venv/Scripts/jupyter lab
```

Windows の Git Bash 経由でコマンドラインから実行する場合（`jupyter nbconvert --execute` 等）は、
日本語出力・PyMC の進捗バーの文字化けを避けるため以下の環境変数を付ける。

```
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 .venv/Scripts/jupyter nbconvert --to notebook --execute --inplace <notebook>.ipynb
```

## 実データの準備

```
curl -sL -o ../data/camera.RData https://raw.githubusercontent.com/cran/bayesm/master/data/camera.RData
```

その後 `prepare_camera.ipynb` を先頭から実行する。

→ `../results/camera_long.csv`（332人 × 16課題 × 5選択肢 = 26,560行）を生成する。

## 疑似データの準備

PerSearch で `camera` と同じ7属性・価格5水準（$79〜$279）・4案＋どれも選ばないの調査を
作成・実行し、ダウンロードした `responses.xlsx` を `../data/persearch_camera.xlsx` に置く
（このリポジトリにはサンプルとして同梱済み。camera と同数の332体のペルソナ・32課題、
年齢層は20〜50代に絞ったペルソナサンプルを使用）。

xlsx を差し替えた場合は、`prepare_persearch_camera.ipynb` の代わりに
`regen_persearch_long.py`（同じロジックの CLI 版）を実行してもよい。

`prepare_persearch_camera.ipynb` を先頭から実行する。

→ `../results/persearch_camera_raw_long.csv`（camera と同じ raw 0/1・ドル価格符号化）と
`../results/persearch_camera_long.csv`（camera と同じカテゴリ文字列符号化、`counting.ipynb` 用）を生成する。

## カウンティング法（選択率の単純集計）

`counting.ipynb` を先頭から実行する。

`../results/counting_{camera,persearch_camera}_{main5,full}.csv` と
`_importance.csv` を出力する。

## 階層ベイズ MNL（対角共分散、PyMC）

camera の生の符号化（brand 4列ダミー・pixels/zoom/video/swivel/wifi の0/1・price を
100ドル単位の連続値）のまま、`beta_i ~ Normal(beta_bar, diag(sigma^2))` の階層構造を推定する。

`hb_diag.ipynb` の「実行設定」セルで `DATASET` を `"camera"` または `"persearch"` に切り替えて、
以降のセルを実行する（1回の実行で1データセット分。両方比較するには2回実行が必要）。

draws=1000, tune=1000, chains=2（既定値）で、camera 側が約2.4分、persearch 側が
約4.7分（12コア環境、C コンパイラ未導入・pytensor の numpy フォールバックのままの実測値。
課題数・収束の難しさにも左右されるため目安として捉えること）。
動作確認だけしたい場合は「実行設定」セルの `N_RESP_LIMIT` を 20 程度に絞ると速く終わる。

出力:
- `../results/hb_diag_{label}_trace.nc`（InferenceData 全体）
- `../results/hb_diag_{label}_summary.csv`（beta_bar・sigma_beta の事後平均・94%HDI・r_hat・ess_bulk）
- `../results/hb_diag_{label}_importance.csv`（属性重要度）

`r_hat` が1.01未満・`ess_bulk` が400超であることが理想。回答者数を camera と揃えた
現在のデータ（persearch 側は20〜50代に絞ったペルソナ）では chains=2・draws=1000 でも
camera 側 r_hat=1.005/ess_bulk=314、persearch 側 r_hat=1.022/ess_bulk=154 まで収束する
（回答者数が少ない条件や persearch 側のデータ構成によってはもっと不安定になりやすいので
注意）。収束を上げたい場合は `draws`・`chains`（chains=4 が既定の推奨値）を増やす。

## 事前分布の頑健性チェック（prior_sensitivity.py）

`beta_bar`・`sigma_beta` の事前分布のスケールを変えても、記事で報告している
「Spearman 順位相関の低さ」「sigma_beta 比（個人間異質性の過小評価）」がほぼ変化しないことを
確認済み（sigma_beta: HalfNormal(1.0/2.5/5.0)、beta_bar: Normal(0, 2.0/5.0/10.0) で検証）。
再現する場合は `prior_sensitivity.py --help` を参照。

## 実データに対する予測性能チェック（predictive_check.py）

係数（beta_bar）の一致度とは別に、「一方のデータセットで推定したモデルで、もう一方の
実際の選択をどれだけ予測できるか」を対数尤度・選択的中率で検証するスクリプト。

記事で報告している結果（persearch の beta_bar で camera の実際の選択を予測）:

| モデル | avg log-lik/choice | 選択的中率 |
|---|---:|---:|
| real-on-real（理論上の上限） | -1.396 | 47.9% |
| pseudo-on-real（本題） | -1.819 | 44.3% |
| pseudo-on-real（tau=0.3 で縮小） | -1.295 | 44.3% |
| uniform（何もしない場合） | -1.609 | 20.0% |

選択的中率は符号のスケールに依存しないため tau を変えても変化しない一方、
対数尤度は tau=0.3 前後まで beta_bar を縮小すると大きく改善する（= 疑似データの
モデルは自信過剰になりやすい）。ただしこの最適な tau は実データ側の当てはまりから
逆算した値であり、実データなしに正しい補正量を決める方法までは示せていない。

再現する場合は `predictive_check.py --help` を参照。通常モード（上表の3行比較）は
`../results/predictive_check_{direction}_summary.csv` に保存される。tau を範囲指定して
最適値を探す場合は `--tau-scan START STOP STEP` を使う
（結果は `../results/predictive_check_{direction}_tauscan.csv` に保存される）。

## 別の調査に適用する場合

`conjoint_lib`のロジック自体は camera 調査（ブランド4水準・2値属性5つ・価格）の属性構成に
決め打ちされておらず、`conjoint_lib.study.StudyConfig` を差し替えるだけで、属性数・水準数・
連続属性の有無が違う別の選択型コンジョイント調査（PerSearchのLong形式シート）にも
適用できる。camera調査自体は `conjoint_lib.schema.CAMERA_STUDY` として定義されており、
これが各関数の既定値になっている。

新しい調査を定義するには、`StudyConfig`・`Attribute`・`Level`（`conjoint_lib/study.py`）を
使って属性構成を書く。

```python
from conjoint_lib.study import Attribute, Level, StudyConfig

MY_STUDY = StudyConfig(
    attributes=(
        # 3水準以上のカテゴリ属性: 全水準に raw_column を与えるとフルダミー符号化になる
        # （camera調査のブランドと同じ扱い。基準水準を置かない）
        Attribute(
            name="color", ja_column="カラー",
            levels=(
                Level(label="black", ja_labels=("黒",), raw_column="color_black"),
                Level(label="white", ja_labels=("白",), raw_column="color_white"),
                Level(label="red", ja_labels=("赤",), raw_column="color_red"),
            ),
        ),
        # 2水準のカテゴリ属性: 先頭を基準水準(raw_columnなし)にすると1列の0/1ダミーになる
        Attribute(
            name="warranty", ja_column="保証",
            levels=(
                Level(label="1年保証", ja_labels=("1年",)),
                Level(label="3年保証", ja_labels=("3年",), raw_column="warranty_3y"),
            ),
        ),
        # 連続属性: parser で文字列を数値に変換する（価格以外にも使える）
        Attribute(
            name="capacity", ja_column="容量",
            continuous_column="capacity_gb",
            parser=lambda s: float(s),
        ),
    ),
    # カウンティング法のグルーピング（省略するとカテゴリ属性すべてを1グループにする）
    attribute_groups={"all": ("color", "warranty")},
)
```

あとは `conjoint_lib.prepare`/`counting`/`hb`/`predictive` の各関数に `config=MY_STUDY` を渡す。

```python
from conjoint_lib.prepare import run_prepare_persearch
from conjoint_lib.counting import run_counting_for_dataset
from conjoint_lib.hb import run_hb

run_prepare_persearch(xlsx_path=..., out_dir=..., config=MY_STUDY)
run_counting_for_dataset(long_csv=..., label="my_study", config=MY_STUDY)
run_hb("my_study", config=MY_STUDY, ...)  # feature_names・属性重要度の算出方法もMY_STUDYから決まる
```

**このまま使えない・注意が必要な点:**

- `sample_scripts/conjoint_lib/paths.py` の結果ファイル名（`counting_camera_*`・
  `hb_diag_{camera,persearch}_*` 等）は camera/persearch という2データセット比較の命名を
  決め打ちしている。別の調査を同じ `results/` に混在させる場合は、`paths.py` を参考に
  別の命名のヘルパーを足すか、`out_dir`/`in_dir` を調査ごとに分ける。
- `StudyConfig.price_attribute()`（カウンティング法・属性重要度で価格として扱う連続属性）は
  現状1つまでしか対応していない。連続属性を複数使う調査では明示的に `ValueError` になる。
- camera実データの取り込み（`prepare.load_camera_rdata`/`build_camera_long_df`）は
  `bayesm::camera` の物理レイアウト固有の処理で、`StudyConfig` を渡しても切り替わらない
  （他の調査では実データ側もPerSearchのLong形式シートから作る前提）。
