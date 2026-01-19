from __future__ import annotations

from pathlib import Path

import pandas as pd

from vsme_extractor.stats import count_filled_indicators


def test_count_filled_indicators_counts_and_sorts(tmp_path: Path) -> None:
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

