from __future__ import annotations

import pytest

import vsme_extractor.cli as cli


def test_cli_list_all_indicators_is_naturally_sorted(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    # Avoid reading any real .env during tests.
    monkeypatch.setattr(cli, "find_dotenv", lambda *args, **kwargs: "")
    monkeypatch.setattr(cli, "load_dotenv", lambda *args, **kwargs: False)

    # Provide an unsorted list; CLI should output A1_1, B2_1, B9_1, B10_1.
    def fake_get_indicators(*, apply_env_filter: bool = True):  # type: ignore[no-untyped-def]
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

