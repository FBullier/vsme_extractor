"""Example: extract indicators from a PDF and write a `.vsme.json` output.

This example reuses the CLI implementation so the JSON schema matches `vsme-extract`:
- `pdf`, `results`, `stats` (+ optional `status`)
- RSE enrichment fields (if mapping table exists)
- Optional retrieval details inclusion

Run:
  .venv/bin/python examples/example_cli_extract_pdf_json.py
"""

from __future__ import annotations

import os
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

    # Optionally restrict to a small set of indicators for faster testing
    codes = "B3_1,B3_2,B7_1,B1_1"

    # Optional: keep retrieval debug fields in each indicator row
    # os.environ["VSME_OUTPUT_JSON_INCLUDE_RETRIEVAL_DETAILS"] = "1"
    include_retrieval_details = (
        os.getenv("VSME_OUTPUT_JSON_INCLUDE_RETRIEVAL_DETAILS") or "0"
    ).strip().lower() in {"1", "true", "yes", "y", "on"}

    # Call the CLI main programmatically (same behavior as `vsme-extract`)
    from vsme_extractor.cli import main as cli_main  # noqa: E402

    args = [
        str(pdf_path),
        "--no-log-stdout",
        "--output-format",
        "json",
        "--json-include-status",
        "--codes",
        codes,
    ]
    if include_retrieval_details:
        args.append("--json-include-retrieval-details")
    cli_main(args)

    out_path = pdf_path.with_suffix(".vsme.json")
    print("Export:", out_path)


if __name__ == "__main__":
    main()
