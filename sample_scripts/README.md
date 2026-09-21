# sample_scripts

`camera` 実データ（R `bayesm` パッケージ）と PerSearch で再現した疑似データを比較する分析コード。
Qiita 記事「生成AIで作った疑似データでコンジョイント分析やってみた」シリーズで使用。

4本の Jupyter notebook（`.ipynb`）で構成する。各 notebook は `sample_scripts/` に、
データ・出力は1つ上の階層の `data/` `results/` に置く前提でパスを組んでいる
（`Path.cwd().parent`。Jupyter のカレントディレクトリは notebook 自身の場所になる）。

| notebook | 役割 |
|---|---|
| `prepare_camera.ipynb` | 実データ（`camera.RData`）→ `camera_long.csv` |
| `prepare_persearch_camera.ipynb` | 疑似データ（`persearch_camera.xlsx`）→ `persearch_camera_*_long.csv` |
| `counting.ipynb` | カウンティング法（選択率の単純集計）で効用値・属性重要度を推定 |
| `hb_diag.ipynb` | 階層ベイズ MNL（対角共分散、PyMC）で同じ比較をより精緻に行う |

上から順に実行する（後段は前段の出力 CSV を読む）。

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
（このリポジトリにはサンプルとして同梱済み。camera と同数の332体のペルソナ・32課題）。

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
現在のデータでは chains=2・draws=1000 でも camera 側 r_hat=1.005/ess_bulk=314、
persearch 側 r_hat=1.011/ess_bulk=169 まで収束する（回答者数が少ない条件ではもっと
不安定になりやすいので注意）。収束を上げたい場合は `draws`・`chains`（chains=4 が
既定の推奨値）を増やす。
