"""persearch_demo の分析パイプライン共有ロジック。

data/ の読み込み・変換（`prepare`）、カウンティング法（`counting`）、
階層ベイズMNL（`hb`）、予測性能チェック（`predictive`）、パス解決（`paths`）、
定数（`schema`）に分かれている。notebook・CLIスクリプトの両方から
`import conjoint_lib` で使う想定（パッケージのインストールは不要。
notebookは `sample_scripts/` をカレントディレクトリとして開き、
スクリプトは自分のディレクトリが `sys.path[0]` になるため、どちらからでも
このパッケージが見える）。
"""
