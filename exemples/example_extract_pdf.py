from __future__ import annotations

"""
Exemple : utiliser la librairie `vsme_extractor` depuis votre propre script.

Prérequis :
- Définir SCW_API_KEY (et optionnellement SCW_BASE_URL / SCW_MODEL_NAME) dans l’environnement ou un fichier .env.
- (Optionnel) Activer le logging via les variables d’environnement :
    SME_LOG_LEVEL=INFO|DEBUG|WARNING|ERROR
    VSME_LOG_FILE=/chemin/vers/vsme.log
    VSME_LOG_STDOUT=1|0 (true/false, yes/no, on/off)

Exécution :
    python exemples/example_extract_pdf.py
"""

from pathlib import Path
import sys

from dotenv import load_dotenv

# Garantit l'import de la version du repo local lors de l'exécution via `python exemples/...`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import vsme_extractor  # noqa: E402
from vsme_extractor import VSMExtractor  # noqa: E402
from vsme_extractor.logging_utils import configure_logging_from_env  # noqa: E402


def main() -> None:
    load_dotenv()
    configure_logging_from_env()

    # Audit : affiche quelle version de vsme_extractor est utilisée (repo local vs site-packages)
    print("vsme_extractor chargé depuis :", Path(vsme_extractor.__file__).resolve())

    pdf_path = Path("./data/test/sanofi.pdf")  # <-- à modifier
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF introuvable : {pdf_path}\nÉditez ce fichier et définissez `pdf_path` vers un PDF existant."
        )

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