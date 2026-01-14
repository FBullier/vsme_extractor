from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from . import VSMExtractor
from .logging_utils import configure_logging, parse_env_bool
from .stats import count_filled_indicators


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vsme-extract",
        description="Extraction automatique d'indicateurs VSME depuis des rapports PDF.",
    )
    parser.add_argument(
        "target",
        nargs="?",
        help="Chemin vers un fichier PDF ou un dossier contenant des PDFs.",
    )
    parser.add_argument(
        "--count",
        dest="count_dir",
        help="Dossier contenant des fichiers .vsme.xlsx (pour calculer la complétude).",
    )

    # Logging defaults can come from environment variables
    # - SME_LOG_LEVEL: DEBUG/INFO/WARNING/ERROR
    # - VSME_LOG_FILE: path
    # - VSME_LOG_STDOUT: 1/0, true/false, yes/no, on/off
    env_level = os.getenv("SME_LOG_LEVEL") or "INFO"
    env_file = os.getenv("VSME_LOG_FILE") or None
    env_stdout = parse_env_bool(os.getenv("VSME_LOG_STDOUT"))

    parser.add_argument(
        "--log-level",
        default=env_level,
        help="Niveau de logs (DEBUG, INFO, WARNING, ERROR). Peut aussi être défini via SME_LOG_LEVEL.",
    )
    parser.add_argument(
        "--log-file",
        default=env_file,
        help="Chemin d'un fichier log (optionnel). Peut aussi être défini via VSME_LOG_FILE.",
    )
    stdout_group = parser.add_mutually_exclusive_group()
    stdout_group.add_argument(
        "--log-stdout",
        dest="log_stdout",
        action="store_true",
        help="Activer les logs sur stdout (par défaut). Peut aussi être défini via VSME_LOG_STDOUT.",
    )
    stdout_group.add_argument(
        "--no-log-stdout",
        dest="log_stdout",
        action="store_false",
        help="Désactiver les logs sur stdout. Peut aussi être défini via VSME_LOG_STDOUT.",
    )
    parser.set_defaults(log_stdout=(env_stdout if env_stdout is not None else True))

    return parser


def main(argv: list[str] | None = None) -> None:
    """Entrée CLI installable (console_scripts)."""
    load_dotenv()

    parser = build_parser()
    ns = parser.parse_args(argv)

    configure_logging(
        level=ns.log_level,
        log_to_stdout=ns.log_stdout,
        log_file=ns.log_file,
        reset_handlers=True,
    )
    logger = logging.getLogger(__name__)

    # ===============================
    # CHECK OPTION --count
    # ===============================
    if ns.count_dir:
        results_dir = Path(ns.count_dir)
        if not results_dir.exists():
            logger.error("Erreur : dossier introuvable : %s", results_dir)
            raise SystemExit(1)

        logger.info("=== Comptage de complétude dans : %s ===", results_dir)
        df_stats = count_filled_indicators(results_dir)

        out_path = results_dir / "stats_completude.xlsx"
        df_stats.to_excel(out_path, index=False)

        logger.info("Résultats → %s", out_path)
        logger.info("\n%s", df_stats.to_string(index=False))
        return

    # ===============================
    # MODE EXTRACTION
    # ===============================
    if not ns.target:
        logger.error("Argument manquant: target (fichier.pdf ou dossier/)")
        logger.info("\n%s", parser.format_help())
        raise SystemExit(1)

    target = Path(ns.target)
    if not target.exists():
        logger.error("Erreur : chemin introuvable : %s", target)
        raise SystemExit(1)

    extractor = VSMExtractor()

    # -------- Un fichier PDF --------
    if target.is_file() and target.suffix.lower() == ".pdf":
        logger.info("PDF: %s", target)
        logger.info("=== Extraction (single) : %s ===", target.name)

        df, stats = extractor.extract_from_pdf(str(target))
        out_path = target.with_suffix(".vsme.xlsx")
        df.to_excel(out_path, index=False)

        logger.info("Export: %s", out_path)
        logger.info("Tokens input : %s", stats.total_input_tokens)
        logger.info("Tokens output: %s", stats.total_output_tokens)
        logger.info("Coût total   : %.4f €", stats.total_cost_eur)
        logger.info("=== Extraction terminée : %s ===", target.name)
        return

    # -------- Un dossier --------
    if target.is_dir():
        pdfs = list(target.glob("*.pdf"))
        if not pdfs:
            logger.warning("Aucun PDF trouvé dans %s", target)
            return

        for pdf in pdfs:
            logger.info("PDF: %s", pdf)
            logger.info("=== Extraction : %s ===", pdf.name)
            df, stats = extractor.extract_from_pdf(str(pdf))
            out_path = pdf.with_suffix(".vsme.xlsx")
            df.to_excel(out_path, index=False)

            logger.info("Export: %s", out_path)
            logger.info("Tokens input : %s", stats.total_input_tokens)
            logger.info("Tokens output: %s", stats.total_output_tokens)
            logger.info("Coût total   : %.4f €", stats.total_cost_eur)
            logger.info("=== Extraction terminée : %s ===", pdf.name)

        return


if __name__ == "__main__":
    main()