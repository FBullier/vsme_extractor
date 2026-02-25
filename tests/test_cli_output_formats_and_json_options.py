from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import vsme_extractor.cli as cli


@dataclass
class _FakeStats:
    total_indicators: int = 1
    indicators_llm_queried: int = 1
    indicators_value_found: int = 1
    total_input_tokens: int = 10
    total_output_tokens: int = 5
    total_cost_eur: float = 0.001


class _FakeExtractor:
    def __init__(self, *args, **kwargs):  # noqa: D401
        pass

    def extract_from_pdf(self, pdf_path: str):
        df = pd.DataFrame(
            [
                {
                    "Code indicateur": "B3_1",
                    "Thématique": "T",
                    "Métrique": "M",
                    "Pages candidates": [1, 2],
                    "Pages conservées": [2],
                    "Retrieval par page": [{"page": 2, "score": 1}],
                    "Valeur": "123",
                    "Unité extraite": "u",
                    "Paragraphe source": "p",
                }
            ]
        )
        return df, _FakeStats()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_cli_output_format_json_strips_retrieval_details_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    # Avoid loading a real .env
    monkeypatch.setattr(cli, "find_dotenv", lambda *args, **kwargs: "")
    monkeypatch.setattr(cli, "load_dotenv", lambda *args, **kwargs: False)

    # Avoid network
    monkeypatch.setattr(cli, "VSMExtractor", _FakeExtractor)

    # Deterministic RSE mapping
    monkeypatch.setattr(
        cli,
        "_load_rse_mapping",
        lambda _p: {
            "B3_1": {
                "matched_rse_code": "B3-30",
                "matched_rse_champs_id": "estimation_emissions_GES",
                "matched_rse_colonne_id": "scope_1",
            }
        },
    )

    cli.main(
        [
            str(pdf),
            "--output-format",
            "json",
            "--json-include-status",
            "--no-log-stdout",
        ]
    )

    out = pdf.with_suffix(".vsme.json")
    assert out.exists()

    payload = _read_json(out)
    assert payload.get("pdf") == str(pdf)
    assert "status" in payload

    results = payload.get("results")
    assert isinstance(results, list) and len(results) == 1
    row = results[0]

    # Retrieval details should be stripped by default.
    assert "Pages candidates" not in row
    assert "Pages conservées" not in row
    assert "Retrieval par page" not in row

    # RSE fields must be present.
    assert row.get("matched_rse_code") == "B3-30"
    assert row.get("matched_rse_champs_id") == "estimation_emissions_GES"
    assert row.get("matched_rse_colonne_id") == "scope_1"


def test_cli_json_can_include_retrieval_details_with_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(cli, "find_dotenv", lambda *args, **kwargs: "")
    monkeypatch.setattr(cli, "load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setattr(cli, "VSMExtractor", _FakeExtractor)
    monkeypatch.setattr(cli, "_load_rse_mapping", lambda _p: {})

    cli.main(
        [
            str(pdf),
            "--output-format",
            "json",
            "--json-include-status",
            "--json-include-retrieval-details",
            "--no-log-stdout",
        ]
    )

    out = pdf.with_suffix(".vsme.json")
    payload = _read_json(out)
    row = payload["results"][0]

    assert "Pages candidates" in row
    assert "Pages conservées" in row
    assert "Retrieval par page" in row


def test_cli_json_no_status_flag_removes_status_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(cli, "find_dotenv", lambda *args, **kwargs: "")
    monkeypatch.setattr(cli, "load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setattr(cli, "VSMExtractor", _FakeExtractor)
    monkeypatch.setattr(cli, "_load_rse_mapping", lambda _p: {})

    cli.main(
        [str(pdf), "--output-format", "json", "--json-no-status", "--no-log-stdout"]
    )

    out = pdf.with_suffix(".vsme.json")
    payload = _read_json(out)
    assert "status" not in payload


def test_cli_output_format_xlsx_writes_excel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(cli, "find_dotenv", lambda *args, **kwargs: "")
    monkeypatch.setattr(cli, "load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setattr(cli, "VSMExtractor", _FakeExtractor)

    cli.main([str(pdf), "--output-format", "xlsx", "--no-log-stdout"])
    out = pdf.with_suffix(".vsme.xlsx")
    assert out.exists()
