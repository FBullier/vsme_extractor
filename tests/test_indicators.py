from __future__ import annotations

from pathlib import Path

import pytest

from vsme_extractor.indicators import get_indicators


def _write_csv(path: Path) -> None:
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
    csv_path = tmp_path / "indicators.csv"
    _write_csv(csv_path)

    monkeypatch.delenv("VSME_CODE_VSME_LIST", raising=False)

    rows = get_indicators(csv_path)
    assert [r["code_vsme"] for r in rows] == ["A1_1", "B10_1"]


def test_get_indicators_filter_by_code_vsme_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = tmp_path / "indicators.csv"
    _write_csv(csv_path)

    # Should keep exactly those codes, even if defaut=0.
    monkeypatch.setenv("VSME_CODE_VSME_LIST", "A1_2, B10_1")

    rows = get_indicators(csv_path)
    assert sorted(r["code_vsme"] for r in rows) == ["A1_2", "B10_1"]


def test_get_indicators_apply_env_filter_false_returns_all(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = tmp_path / "indicators.csv"
    _write_csv(csv_path)

    monkeypatch.setenv("VSME_CODE_VSME_LIST", "A1_1")

    rows = get_indicators(csv_path, apply_env_filter=False)
    assert sorted(r["code_vsme"] for r in rows) == ["A1_1", "A1_2", "B10_1"]

