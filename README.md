# persearch_demo

Qiita記事「生成AIで作った疑似データでコンジョイント分析やってみた」シリーズのための検証用リポジトリ。
生成AIによるペルソナ調査ツール **PerSearch** で作った疑似回答データが、実データを使ったコンジョイント分析
（属性重要度の推定）とどの程度近い結果を再現できるかを比較する。

比較対象:

- **実データ**: R の `bayesm` パッケージに収録されている `camera` データセット（デジタルカメラのコンジョイント調査、332人 × 16課題 × 5選択肢）
- **疑似データ**: PerSearch で `camera` と同一の調査設計（7属性・価格5水準）を組み、332体のペルソナに32課題を回答させて生成した疑似回答データ

分析手法として、カウンティング法（選択率の単純集計）と階層ベイズ MNL（対角共分散、PyMC）の2種類で属性重要度を推定し、
実データと疑似データの結果を比較する。

## リポジトリ構成

```
persearch_demo/
├── data/            分析の入力データ
├── results/         notebook の出力（中間CSV・推定結果）
├── sample_scripts/  分析コード（Jupyter notebook）
└── README.md        このファイル
```

## 同梱データ（data/）

| ファイル | 内容 |
|---|---|
| `camera.RData` | 実データ。R の `bayesm` パッケージ収録の `camera` データセット。ライセンス上の理由でリポジトリには同梱せず `.gitignore` で除外している。取得手順は後述 |
| `persearch_camera.xlsx` | 疑似データ。PerSearch で `camera` と同じ7属性・価格5水準（$79〜$279）の調査を作成・実行した結果。camera と同数の332体のペルソナが32課題に回答している（サンプルとして同梱済み） |

## 出力データ（results/）

`sample_scripts/` 配下の各 notebook を実行すると生成される中間データ・分析結果。

| ファイル | 内容 |
|---|---|
| `camera_long.csv` / `camera_raw_long.csv` | `camera` を long 形式に変換したデータ（`prepare_camera.ipynb` の出力） |
| `persearch_camera_long.csv` / `persearch_camera_raw_long.csv` | `persearch_camera.xlsx` を long 形式に変換したデータ（`prepare_persearch_camera.ipynb` の出力） |
| `counting_{camera,persearch_camera}_{main5,full}.csv` / 同 `_importance.csv` | カウンティング法（選択率の単純集計）による効用値・属性重要度（`counting.ipynb` の出力） |
| `hb_diag_{camera,persearch}_trace.nc` / `_summary.csv` / `_importance.csv` | 階層ベイズ MNL の事後サンプル一式・要約統計（事後平均・94%HDI・r_hat・ess_bulk）・属性重要度（`hb_diag.ipynb` の出力） |

いずれも同梱データ（`data/`）から notebook を実行すれば再生成できるため、`results/` 配下のファイルは
すべて `.gitignore` で除外しており、リポジトリには含めていない。

## 分析環境のセットアップ

**Python 3.12 が必要**（`pymc`/`pytensor` が本記事執筆時点で Python 3.13/3.14 向けのビルド済みホイールを配布していないため、
システムの既定が新しいバージョンの場合は 3.12 を明示的に指定すること）。

### uv を使う場合（推奨）

```
cd sample_scripts
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt jupyter ipykernel
```

### venv + pip を使う場合

```
cd sample_scripts
python3.12 -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt jupyter ipykernel
```

依存バージョンは `sample_scripts/requirements.txt` に動作確認済みの値を固定してある。

## 分析コードの動かし方

分析コードは `sample_scripts/` 配下の4本の Jupyter notebook で構成する。各 notebook は `sample_scripts/` に、
データ・出力は1つ上の階層の `data/` `results/` に置く前提でパスを組んでいる。上から順に実行する
（後段は前段の出力 CSV を読む）。

| notebook | 役割 |
|---|---|
| `prepare_camera.ipynb` | 実データ（`camera.RData`）→ `results/camera_long.csv` |
| `prepare_persearch_camera.ipynb` | 疑似データ（`persearch_camera.xlsx`）→ `results/persearch_camera_*_long.csv` |
| `counting.ipynb` | カウンティング法（選択率の単純集計）で効用値・属性重要度を推定 |
| `hb_diag.ipynb` | 階層ベイズ MNL（対角共分散、PyMC）で同じ比較をより精緻に行う |

1. 実データを取得する。

   ```
   curl -sL -o data/camera.RData https://raw.githubusercontent.com/cran/bayesm/master/data/camera.RData
   ```

2. notebook を起動する。

   ```
   sample_scripts/.venv/Scripts/jupyter lab
   ```

3. 上表の順に、各 notebook を先頭セルから実行する（`hb_diag.ipynb` は「実行設定」セルで
   `DATASET` を `"camera"` / `"persearch"` に切り替えて2回実行する）。

セットアップの詳細、各 notebook の入出力・実行時間・収束診断の目安などは
[`sample_scripts/README.md`](sample_scripts/README.md) を参照。
