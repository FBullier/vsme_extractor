from __future__ import annotations

"""
Exemple d’utilisation de la librairie `vsme_extractor` (ce fichier n'est pas la CLI).

Après installation du package (par ex. `pip install .`), vous pouvez exécuter ce fichier :
    python main.py

La CLI installable est fournie par :
    vsme-extract  (entrypoint : vsme_extractor.cli:main)

Le logging (optionnel) peut être activé via des variables d’environnement :
- SME_LOG_LEVEL=INFO|DEBUG|WARNING|ERROR
- VSME_LOG_FILE=/chemin/vers/vsme.log
- VSME_LOG_STDOUT=1|0 (true/false, yes/no, on/off)
"""

from pathlib import Path

from dotenv import load_dotenv

from vsme_extractor import VSMExtractor
from vsme_extractor.logging_utils import configure_logging_from_env


def main() -> None:
    # Charge .env si présent (SCW_API_KEY, etc.)
    load_dotenv()

    # Optionnel : configure le logging uniquement si des variables sont définies
    configure_logging_from_env()

    pdf_path = Path("chemin/vers/rapport.pdf")  # <-- à modifier

    extractor = VSMExtractor(
        top_k_snippets=6,
        temperature=0.2,
        max_tokens=512,
    )

    df, stats = extractor.extract_from_pdf(str(pdf_path))
    out_path = pdf_path.with_suffix(".vsme.xlsx")
    df.to_excel(out_path, index=False)

    print("Export :", out_path)
    print("Tokens entrée :", stats.total_input_tokens)
    print("Tokens sortie :", stats.total_output_tokens)
    print("Coût total    :", f"{stats.total_cost_eur:.4f} €")


if __name__ == "__main__":
    main()
