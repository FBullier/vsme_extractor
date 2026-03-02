"""Example: extract indicators from a PDF and write a `.vsme.xlsx` output.

This example calls the CLI programmatically, so the behavior matches `vsme-extract`.

Run:
  .venv/bin/python examples/example_cli_extract_pdf_xlsx.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv


def main() -> None:
    # If running from the git repo, prefer the local package. If the script is copied
    # elsewhere, this does nothing and `import vsme_extractor` will use the installed package.
    repo_root = Path(__file__).resolve().parents[1]
    if (repo_root / "vsme_extractor").exists() and str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Load .env if present, but DO NOT override explicit environment variables.
    dotenv_path = find_dotenv(usecwd=True)
    load_dotenv(dotenv_path, override=False)

    # Set an existing PDF path (edit this)
    pdf_path = Path("/your_path/your_file.pdf")
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")

    # Restrict to a small set of indicators for faster testing (edit as needed)
    codes = "B3_1,B3_2"

    from vsme_extractor.cli import main as cli_main  # noqa: E402

    cli_main([str(pdf_path), "--no-log-stdout", "--output-format", "xlsx", "--codes", codes])

    out_path = pdf_path.with_suffix(".vsme.xlsx")
    print("Export:", out_path)


if __name__ == "__main__":
    main()
