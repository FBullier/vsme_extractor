"""Tests des options de listing de la CLI.

Focus : comportement déterministe de [`vsme_extractor.cli.main()`](vsme_extractor/cli.py:87)
pour `--list-current-indicators` et `--list-all-indicators`.

On monkeypatch le chargement dotenv et le chargement d'indicateurs pour éviter toute dépendance
à un fichier `.env` local.
"""

from __future__ import annotations

import pytest

import vsme_extractor.cli as cli


def test_cli_list_all_indicators_is_naturally_sorted(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Le listing doit utiliser un tri naturel (B9 avant B10, etc.)."""
    # Évite de lire un `.env` réel pendant les tests.
    monkeypatch.setattr(cli, "find_dotenv", lambda *args, **kwargs: "")
    monkeypatch.setattr(cli, "load_dotenv", lambda *args, **kwargs: False)

    # Provide an unsorted list; CLI should output A1_1, B2_1, B9_1, B10_1.
    def fake_get_indicators(*, apply_env_filter: bool = True):  # type: ignore[no-untyped-def]
        """Stub: renvoie une liste d'indicateurs pour tester le tri CLI."""
        return [
            {"code_vsme": "B10_1", "Code indicateur": "B10", "Métrique": "m"},
            {"code_vsme": "B9_1", "Code indicateur": "B9", "Métrique": "m"},
            {"code_vsme": "A1_1", "Code indicateur": "A1", "Métrique": "m"},
            {"code_vsme": "B2_1", "Code indicateur": "B2", "Métrique": "m"},
        ]

    monkeypatch.setattr(cli, "get_indicators", fake_get_indicators)

    cli.main(["--list-all-indicators", "--no-log-stdout"])
    out = capsys.readouterr().out.strip().splitlines()

    assert out[0].startswith("code_vsme")
    codes = [line.split("\t")[0] for line in out[1:]]
    assert codes == ["A1_1", "B2_1", "B9_1", "B10_1"]


def test_cli_list_current_indicators_passes_apply_env_filter_true(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """`--list-current-indicators` doit demander le filtrage env à `get_indicators()`."""
    monkeypatch.setattr(cli, "find_dotenv", lambda *args, **kwargs: "")
    monkeypatch.setattr(cli, "load_dotenv", lambda *args, **kwargs: False)

    seen = {"apply_env_filter": None}

    def fake_get_indicators(*, apply_env_filter: bool = True):  # type: ignore[no-untyped-def]
        """Stub: permet de capturer la valeur de `apply_env_filter` utilisée par la CLI."""
        seen["apply_env_filter"] = apply_env_filter
        return [{"code_vsme": "B1_1", "Code indicateur": "B1", "Métrique": "m"}]

    monkeypatch.setattr(cli, "get_indicators", fake_get_indicators)

    cli.main(["--list-current-indicators", "--no-log-stdout"])
    _ = capsys.readouterr().out
    assert seen["apply_env_filter"] is True


def test_cli_mutually_exclusive_list_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Les deux options de listing sont exclusives (argparse doit terminer)."""
    monkeypatch.setattr(cli, "find_dotenv", lambda *args, **kwargs: "")
    monkeypatch.setattr(cli, "load_dotenv", lambda *args, **kwargs: False)

    with pytest.raises(SystemExit):
        cli.main(["--list-current-indicators", "--list-all-indicators", "--no-log-stdout"])


@pytest.mark.parametrize(
    "codes,expected",
    [
        (["B10_1", "B9_1"], ["B9_1", "B10_1"]),
        (["A10_1", "A2_1"], ["A2_1", "A10_1"]),
        (["C8_10", "C8_2", "C8_1"], ["C8_1", "C8_2", "C8_10"]),
        (["AA2_1", "A2_1"], ["A2_1", "AA2_1"]),
    ],
)
def test_cli_natural_sort_various_codes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], codes: list[str], expected: list[str]
) -> None:
    """Le tri naturel doit fonctionner sur plusieurs patterns : multi-lettres, suffixes, etc."""
    monkeypatch.setattr(cli, "find_dotenv", lambda *args, **kwargs: "")
    monkeypatch.setattr(cli, "load_dotenv", lambda *args, **kwargs: False)

    def fake_get_indicators(*, apply_env_filter: bool = True):  # type: ignore[no-untyped-def]
        """Stub: renvoie les codes demandés pour tester l'ordre de tri."""
        return [{"code_vsme": c, "Code indicateur": c.split("_")[0], "Métrique": "m"} for c in codes]

    monkeypatch.setattr(cli, "get_indicators", fake_get_indicators)
    cli.main(["--list-all-indicators", "--no-log-stdout"])
    out = capsys.readouterr().out.strip().splitlines()[1:]
    got = [line.split("\t")[0] for line in out]
    assert got == expected
