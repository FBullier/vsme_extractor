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

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import sys
import traceback
from datetime import datetime, timezone

from dotenv import find_dotenv, load_dotenv

# Garantit l'import de la version du repo local lors de l'exécution via `python exemples/...`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import vsme_extractor  # noqa: E402
from vsme_extractor import VSMExtractor  # noqa: E402
from vsme_extractor.logging_utils import configure_logging_from_env  # noqa: E402


logger = logging.getLogger(__name__)


def main() -> None:
    """Exemple : extraction d'un PDF en utilisant la librairie (hors CLI)."""
    # Charge `.env` depuis le répertoire courant vers les parents, et force l'override.
    # Cela évite le cas où une variable déjà exportée (parfois vide) masque la valeur du `.env`.
    dotenv_path = find_dotenv(usecwd=True)
    dotenv_loaded = load_dotenv(dotenv_path, override=True)
    configure_logging_from_env()

    print("dotenv | path=", dotenv_path or None, "| loaded=", dotenv_loaded)

    # L'init de `VSMExtractor` peut faire un healthcheck LLM (réseau + quota) :
    # - utile en prod pour diagnostiquer l'environnement,
    # - mais gênant dans un exemple si le provider est en quota.
    # Ici on le désactive par défaut (sans surcharger une valeur explicitement définie).
    os.environ.setdefault("VSME_LLM_HEALTHCHECK", "0")
    os.environ.setdefault("VSME_LLM_HEALTHCHECK_STRICT", "0")

    # Audit : affiche quelle version de vsme_extractor est utilisée (repo local vs site-packages)
    print("vsme_extractor chargé depuis :", Path(vsme_extractor.__file__).resolve())

    # Default to a repo-provided PDF if available, otherwise edit this path.
    pdf_path = Path("./data/test/nexans.pdf")
    if not pdf_path.exists():
        pdf_path = Path("./data/test/sanofi.pdf")  # <-- à modifier si besoin
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF introuvable : {pdf_path}\nÉditez ce fichier et définissez `pdf_path` vers un PDF existant."
        )

    extractor = None
    stage = "init"
    try:
        extractor = VSMExtractor(
            top_k_snippets=6,
            temperature=0.2,
            max_tokens=512,
        )

        stage = "extract"
        df, stats = extractor.extract_from_pdf(str(pdf_path))
    except Exception as e:
        # Note: un `try/except` empêche l'exception de remonter jusqu'à Python,
        # mais n'empêche pas *les logs* (ni les stacktraces) si du code appelle
        # `logger.exception(...)`. Ici on loggue sans stacktrace pour éviter le bruit
        # sur la console ; le détail complet est dans le fichier `.vsme.error.json`.
        logger.error(
            "Example failed | stage=%s | pdf=%s | error=%s: %s",
            stage,
            str(pdf_path),
            e.__class__.__name__,
            str(e),
        )

        # Et écrit aussi un fichier d'erreur à côté du PDF, utile en batch ou pour support.
        # (Ne pas inclure de secrets comme SCW_API_KEY.)
        err_payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "pdf": str(pdf_path),
            "stage": stage,
            "error": {"type": e.__class__.__name__, "message": str(e)},
            "traceback": traceback.format_exc(),
            "llm": {
                "base_url": getattr(
                    getattr(extractor, "config", None), "base_url", None
                ),
                "model": getattr(getattr(extractor, "config", None), "model", None),
                "api_protocol": getattr(
                    getattr(extractor, "config", None), "api_protocol", None
                ),
                "invoke_mode": getattr(
                    getattr(extractor, "config", None), "invoke_mode", None
                ),
            },
            "params": {
                "top_k_snippets": getattr(extractor, "top_k_snippets", None),
                "temperature": getattr(extractor, "temperature", None),
                "max_tokens": getattr(extractor, "max_tokens", None),
                "retrieval_method": getattr(extractor, "retrieval_method", None),
            },
        }
        err_path = pdf_path.with_suffix(".vsme.error.json")
        err_path.write_text(
            json.dumps(err_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"Extraction FAILED: {e.__class__.__name__}: {e}\nError report: {err_path}",
            file=sys.stderr,
        )
        raise SystemExit(2)

    out_path = pdf_path.with_suffix(".vsme.xlsx")
    df.to_excel(out_path, index=False)

    print("Export :", out_path)
    print("Tokens entrée :", stats.total_input_tokens)
    print("Tokens sortie :", stats.total_output_tokens)
    print("Coût total    :", f"{stats.total_cost_eur:.4f} €")


if __name__ == "__main__":
    main()
