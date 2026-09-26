"""選択型コンジョイント調査の属性構成を表す設定オブジェクト。

camera調査は「ブランド4水準・2値属性5つ・価格」という固定の構成だったため、
これまでの実装では属性名・水準・raw符号化の列名がすべてモジュールレベルの
定数として決め打ちされていた。属性数・水準数・連続変数の有無は調査ごとに
変わるため、それらを`StudyConfig`一つにまとめ、`conjoint_lib.prepare`/
`counting`/`hb`/`predictive`の各関数が`config`引数として受け取れるようにする。
これにより、別の選択型コンジョイント調査にも同じロジック（PerSearchシートの
取り込み・カウンティング法・階層ベイズMNL・予測性能チェック）を適用できる。

camera調査自体は`conjoint_lib.schema.CAMERA_STUDY`にこの型のインスタンスとして
定義されており、既定値として使われる。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Level:
    """カテゴリ属性の1水準。"""

    label: str
    """labeled long CSV（camera_long.csv 相当）で使うカテゴリ文字列。"""

    ja_labels: tuple[str, ...] = ()
    """PerSearch側の出力でこの水準に対応する日本語文字列（表記ゆれ対応で複数指定可）。"""

    raw_column: str | None = None
    """raw符号化での列名。Noneならこの水準を基準水準として扱い、専用の列は作らない
    （2水準の属性を1列の0/1ダミーにする、という従来の符号化と等価になる）。
    全水準に raw_column を与えれば、水準数ぶんの列を持つフルダミー符号化になる
    （bayesm::camera のブランド4列のような符号化）。"""


@dataclass(frozen=True)
class Attribute:
    """調査の1属性（カテゴリ属性 または 連続属性）。"""

    name: str
    """属性の内部名。labeled long CSV の列名、属性重要度の行名になる。"""

    ja_column: str
    """PerSearchシート側の列名（日本語）。"""

    levels: tuple[Level, ...] = ()
    """カテゴリ属性の水準一覧（宣言順）。連続属性なら空にする。"""

    continuous_column: str | None = None
    """連続属性のraw符号化での列名（例: "price_usd100"）。カテゴリ属性ならNone。"""

    parser: Callable[[object], float] | None = None
    """連続属性の値をraw列の数値に変換する関数（例: "$129" -> 1.29）。連続属性は必須。"""

    expected_values: tuple[float, ...] | None = None
    """連続属性の想定値一覧（バリデーション用）。Noneなら値のチェックをしない。"""

    def __post_init__(self) -> None:
        if self.is_continuous and self.levels:
            raise ValueError(f"attribute {self.name!r}: continuous attributes cannot have levels")
        if not self.is_continuous and not self.levels:
            raise ValueError(f"attribute {self.name!r}: categorical attributes need at least one level")
        if self.is_continuous and self.parser is None:
            raise ValueError(f"attribute {self.name!r}: continuous attributes need a parser")

    @property
    def is_continuous(self) -> bool:
        return self.continuous_column is not None

    def raw_columns(self) -> list[str]:
        """このプロパティの並び順が、raw符号化・階層ベイズの feature_names の並びになる。"""
        if self.is_continuous:
            return [self.continuous_column]
        return [lvl.raw_column for lvl in self.levels if lvl.raw_column is not None]

    def level_labels(self) -> list[str]:
        return [lvl.label for lvl in self.levels]

    def ja_level_map(self) -> dict[str, str]:
        """PerSearch側の日本語水準文字列 -> labeled表記(label) の対応表（基準水準も含む）。"""
        out: dict[str, str] = {}
        for lvl in self.levels:
            for ja in lvl.ja_labels:
                out[ja] = lvl.label
        return out


@dataclass(frozen=True)
class StudyConfig:
    """1つの選択型コンジョイント調査（属性構成）を表す。"""

    attributes: tuple[Attribute, ...]
    persearch_sheet_name: str = "回答データ（Long形式）"
    attribute_groups: dict[str, tuple[str, ...]] | None = None
    """カウンティング法で使う属性のグルーピング（例: {"main5": (...), "full": (...)}）。
    Noneならカテゴリ属性すべてを1グループ("all")として扱う。"""

    def attribute(self, name: str) -> Attribute:
        for attr in self.attributes:
            if attr.name == name:
                return attr
        raise KeyError(f"unknown attribute: {name!r}")

    def categorical_attributes(self) -> list[Attribute]:
        return [a for a in self.attributes if not a.is_continuous]

    def continuous_attributes(self) -> list[Attribute]:
        return [a for a in self.attributes if a.is_continuous]

    def price_attribute(self) -> Attribute | None:
        """カウンティング法・属性重要度で価格として扱う連続属性。現状は1つまで対応する。"""
        cont = self.continuous_attributes()
        if not cont:
            return None
        if len(cont) > 1:
            raise ValueError(
                "price_attribute() only supports a single continuous attribute for now "
                f"(got {[a.name for a in cont]})"
            )
        return cont[0]

    def raw_columns(self) -> list[str]:
        cols: list[str] = []
        for attr in self.attributes:
            cols += attr.raw_columns()
        return cols

    def attribute_level_order(self) -> dict[str, list[str]]:
        return {a.name: a.level_labels() for a in self.categorical_attributes()}

    def groups(self) -> dict[str, list[str]]:
        if self.attribute_groups is not None:
            return {name: list(attrs) for name, attrs in self.attribute_groups.items()}
        return {"all": [a.name for a in self.categorical_attributes()]}
