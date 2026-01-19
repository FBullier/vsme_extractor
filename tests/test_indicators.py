"""Tests for indicator loading and filtering.

Focus: deterministic logic in [`vsme_extractor.indicators.get_indicators()`](vsme_extractor/indicators.py:20).

We intentionally avoid any network calls and only validate CSV parsing + environment-driven filtering.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vsme_extractor.indicators import get_indicators


def _write_csv(path: Path) -> None:
    """Write a minimal indicator CSV to `path` for unit tests."""
    # Minimal semicolon-separated CSV compatible with vsme_extractor.indicators.get_indicators()
    path.write_text(
        """;Code indicateur;Thématique;Métrique;Unité / Détail;Mots clés;Keywords;code_vsme;defaut
0;A1;Theme;Metric A1-1;u;;;;A1_1;1
1;A1;Theme;Metric A1-2;u;;;;A1_2;0
2;B10;Theme;Metric B10-1;u;;;;B10_1;1
""",
        encoding="utf-8",
    )


def test_get_indicators_default_filter_defaut_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When no explicit list is provided, default behaviour is `defaut == 1` filtering."""
    csv_path = tmp_path / "indicators.csv"
    _write_csv(csv_path)

    monkeypatch.delenv("VSME_CODE_VSME_LIST", raising=False)

    rows = get_indicators(csv_path)
    assert [r["code_vsme"] for r in rows] == ["A1_1", "B10_1"]


def test_get_indicators_filter_by_code_vsme_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When VSME_CODE_VSME_LIST is set, it should override `defaut` filtering."""
    csv_path = tmp_path / "indicators.csv"
    _write_csv(csv_path)

    # Should keep exactly those codes, even if defaut=0.
    monkeypatch.setenv("VSME_CODE_VSME_LIST", "A1_2, B10_1")

    rows = get_indicators(csv_path)
    assert sorted(r["code_vsme"] for r in rows) == ["A1_2", "B10_1"]


def test_get_indicators_apply_env_filter_false_returns_all(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`apply_env_filter=False` should bypass all `.env`-based filtering."""
    csv_path = tmp_path / "indicators.csv"
    _write_csv(csv_path)

    monkeypatch.setenv("VSME_CODE_VSME_LIST", "A1_1")

    rows = get_indicators(csv_path, apply_env_filter=False)
    assert sorted(r["code_vsme"] for r in rows) == ["A1_1", "A1_2", "B10_1"]


@pytest.mark.parametrize(
    "codes_raw",
    [
        "A1_1 B10_1",  # spaces
        "A1_1,B10_1",  # commas
        "A1_1;B10_1",  # semicolons
        "  A1_1 ;  B10_1  ",  # extra spaces
    ],
)
def test_get_indicators_code_list_parsing_separators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, codes_raw: str
) -> None:
    """The list parser accepts common separators: spaces, commas, semicolons."""
    csv_path = tmp_path / "indicators.csv"
    _write_csv(csv_path)

    monkeypatch.setenv("VSME_CODE_VSME_LIST", codes_raw)
    rows = get_indicators(csv_path)
    assert sorted(r["code_vsme"] for r in rows) == ["A1_1", "B10_1"]


def test_get_indicators_when_code_vsme_missing_does_not_filter_by_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If the CSV lacks `code_vsme`, we keep rows unfiltered (and do not crash)."""
    # If VSME_CODE_VSME_LIST is set but the CSV has no `code_vsme` column,
    # we should not crash and should return unfiltered rows.
    csv_path = tmp_path / "indicators_no_code_vsme.csv"
    csv_path.write_text(
        """;Code indicateur;Thématique;Métrique;Unité / Détail;Mots clés;Keywords;defaut
0;A1;Theme;Metric A1-1;u;;;;1
1;A1;Theme;Metric A1-2;u;;;;0
2;B10;Theme;Metric B10-1;u;;;;1
""",
        encoding="utf-8",
    )

    monkeypatch.setenv("VSME_CODE_VSME_LIST", "A1_1")
    rows = get_indicators(csv_path)
    assert len(rows) == 3


def test_get_indicators_when_defaut_missing_and_no_list_returns_all(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If neither list nor `defaut` column exist, no filtering can apply."""
    csv_path = tmp_path / "indicators_no_defaut.csv"
    csv_path.write_text(
        """;Code indicateur;Thématique;Métrique;Unité / Détail;Mots clés;Keywords;code_vsme
0;A1;Theme;Metric A1-1;u;;;;A1_1
1;A1;Theme;Metric A1-2;u;;;;A1_2
2;B10;Theme;Metric B10-1;u;;;;B10_1
""",
        encoding="utf-8",
    )

    monkeypatch.delenv("VSME_CODE_VSME_LIST", raising=False)
    rows = get_indicators(csv_path)
    assert len(rows) == 3


@pytest.mark.parametrize("defaut_value", ["1", " 1 ", 1])
def test_get_indicators_defaut_accepts_string_or_int(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defaut_value) -> None:
    """`defaut` values may be typed as string or int depending on CSV parsing."""
    csv_path = tmp_path / "indicators_defaut_types.csv"
    # Write with pandas-like loose typing behavior (string/int). For simplicity use CSV text.
    csv_path.write_text(
        """;Code indicateur;Thématique;Métrique;Unité / Détail;Mots clés;Keywords;code_vsme;defaut
0;A1;Theme;Metric A1-1;u;;;;A1_1;{dv}
1;A1;Theme;Metric A1-2;u;;;;A1_2;0
""".format(dv=defaut_value),
        encoding="utf-8",
    )

    monkeypatch.delenv("VSME_CODE_VSME_LIST", raising=False)
    rows = get_indicators(csv_path)
    assert [r["code_vsme"] for r in rows] == ["A1_1"]
