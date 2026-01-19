"""Tests for stats computation.

Focus: deterministic logic in [`vsme_extractor.stats.count_filled_indicators()`](vsme_extractor/stats.py:25)
and [`vsme_extractor.stats._is_filled()`](vsme_extractor/stats.py:10).

We create temporary `.vsme.xlsx` files and validate counting + sorting behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import pandas as pd

from vsme_extractor.stats import _is_filled, count_filled_indicators


def test_count_filled_indicators_counts_and_sorts(tmp_path: Path) -> None:
    """Basic scenario: counts filled values and applies natural sort B1 before B10."""
    # Build 2 minimal .vsme.xlsx files.
    df1 = pd.DataFrame(
        [
            {"Code indicateur": "B1", "Thématique": "T", "Métrique": "M1", "Valeur": "123"},
            {"Code indicateur": "B10", "Thématique": "T", "Métrique": "M10", "Valeur": "NA"},
        ]
    )
    df2 = pd.DataFrame(
        [
            {"Code indicateur": "B1", "Thématique": "T", "Métrique": "M1", "Valeur": ""},
            {"Code indicateur": "B10", "Thématique": "T", "Métrique": "M10", "Valeur": "999"},
        ]
    )

    p1 = tmp_path / "a.vsme.xlsx"
    p2 = tmp_path / "b.vsme.xlsx"
    df1.to_excel(p1, index=False)
    df2.to_excel(p2, index=False)

    out = count_filled_indicators(tmp_path)
    # 2 rows (B1/M1) and (B10/M10)
    assert set(out["Code indicateur"].tolist()) == {"B1", "B10"}

    # B1/M1: appears in 2 files, filled in 1 file ("123" only)
    row_b1 = out[out["Code indicateur"] == "B1"].iloc[0]
    assert row_b1["Fichiers contenant la métrique"] == 2
    assert row_b1["Occurrences renseignées"] == 1

    # Natural sort by numeric part: B1 should come before B10
    assert out["Code indicateur"].tolist()[0] == "B1"


def test_count_filled_indicators_raises_when_empty_dir(tmp_path: Path) -> None:
    """An empty directory should raise (no input files to aggregate)."""
    # No *.vsme.xlsx files
    with pytest.raises(FileNotFoundError):
        count_filled_indicators(tmp_path)


def test_count_filled_indicators_skips_files_with_missing_columns(tmp_path: Path) -> None:
    """Files with missing columns are skipped; valid files are still processed."""
    # File missing required columns should be skipped and not crash.
    bad = pd.DataFrame([{"foo": 1}])
    bad_path = tmp_path / "bad.vsme.xlsx"
    bad.to_excel(bad_path, index=False)

    good = pd.DataFrame(
        [{"Code indicateur": "B2", "Thématique": "T", "Métrique": "M", "Valeur": "1"}]
    )
    good_path = tmp_path / "good.vsme.xlsx"
    good.to_excel(good_path, index=False)

    out = count_filled_indicators(tmp_path)
    assert out["Code indicateur"].tolist() == ["B2"]


@pytest.mark.parametrize(
    "value,expected",
    [
        ("123", True),
        (" 123 ", True),
        ("0", True),
        ("", False),
        ("   ", False),
        ("NA", False),
        ("n/a", False),
        ("None", False),
        ("null", False),
        ("NaN", False),
        (None, False),
        (float("nan"), False),
        (0, True),
        (1.23, True),
    ],
)
def test_is_filled_variants(value, expected: bool) -> None:
    """`_is_filled` should treat NA-ish strings and blank values as not filled."""
    assert _is_filled(value) is expected
