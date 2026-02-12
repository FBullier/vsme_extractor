from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import re

from dotenv import find_dotenv, load_dotenv

from . import VSMExtractor
from .indicators import get_indicators
from .logging_utils import configure_logging, parse_env_bool
from .stats import count_filled_indicators


def build_parser() -> argparse.ArgumentParser:
    """Construit le parser CLI (args + defaults issus de l'environnement)."""
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

    # Retrieval method (selection des extraits/pages avant appel LLM)
    env_retrieval = (os.getenv("VSME_RETRIEVAL_METHOD") or "count").strip()
    parser.add_argument(
        "--retrieval-method",
        dest="retrieval_method",
        default=env_retrieval,
        choices=["count", "count_score", "bm25", "bm25_souple"],
        help=(
            "Méthode de sélection des extraits/pages avant appel LLM. "
            "Par défaut: count. Peut aussi être définie via VSME_RETRIEVAL_METHOD."
        ),
    )

    parser.add_argument(
        "--codes",
        dest="code_vsme_list",
        default=None,
        help=(
            "Liste de codes `code_vsme` à extraire (surcharge VSME_CODE_VSME_LIST si fourni). "
            "Séparateurs acceptés : virgule, point-virgule, espaces. "
            "Exemple: --codes B3_1,B3_2,C1_1"
        ),
    )
    list_group = parser.add_mutually_exclusive_group()
    list_group.add_argument(
        "--list-current-indicators",
        action="store_true",
        help=(
            "Affiche la liste des indicateurs actuellement utilisés par l’extracteur "
            "(après application des variables d’environnement), sous la forme: code_vsme + métrique, "
            "triés par code_vsme."
        ),
    )
    list_group.add_argument(
        "--list-all-indicators",
        action="store_true",
        help=(
            "Affiche la liste complète des indicateurs du CSV (sans filtre `.env`), sous la forme: "
            "code_vsme + métrique, triés par code_vsme."
        ),
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
    # Charge `.env` (en recherchant depuis le répertoire courant vers les parents)
    # et force l'override.
    # Cela évite les cas surprenants où une variable déjà exportée (parfois vide)
    # masque la valeur définie dans `.env`.
    dotenv_path = find_dotenv(usecwd=True)
    dotenv_loaded = load_dotenv(dotenv_path, override=True)

    parser = build_parser()
    ns = parser.parse_args(argv)

    # Option CLI : la liste de codes fournie surcharge la variable d'environnement.
    # Cela permet d'utiliser un `.env` générique tout en forçant une sélection ponctuelle.
    if ns.code_vsme_list is not None and str(ns.code_vsme_list).strip() != "":
        os.environ["VSME_CODE_VSME_LIST"] = str(ns.code_vsme_list).strip()

    configure_logging(
        level=ns.log_level,
        log_to_stdout=ns.log_stdout,
        log_file=ns.log_file,
        reset_handlers=True,
    )
    logger = logging.getLogger(__name__)

    # Diagnostic : confirme quel `.env` a été trouvé/chargé et l'état des variables
    # de filtrage des indicateurs.
    # (Ne jamais logger de secrets.)
    logger.info(
        "dotenv | found=%s | loaded=%s | cwd=%s",
        dotenv_path or None,
        dotenv_loaded,
        str(Path.cwd()),
    )
    logger.info(
        "env | SCW_API_KEY_set=%s | VSM_INDICATORS_PATH=%s | VSME_CODE_VSME_LIST=%s",
        bool(os.getenv("SCW_API_KEY")),
        (os.getenv("VSM_INDICATORS_PATH") or "").strip() or None,
        (os.getenv("VSME_CODE_VSME_LIST") or "").strip() or None,
    )

    # ===============================
    # LIST INDICATORS
    # ===============================
    if ns.list_current_indicators or ns.list_all_indicators:
        indicators = get_indicators(apply_env_filter=bool(ns.list_current_indicators))
        items = []
        for row in indicators:
            code_vsme = str(row.get("code_vsme") or "").strip()
            metric = str(row.get("Métrique") or row.get("Metrique") or "").strip()
            code_ind = str(row.get("Code indicateur") or "").strip()
            items.append((code_vsme, metric, code_ind))

        def _code_sort_key(code: str) -> tuple:
            """Clé de tri pour des codes du type B1, B10, C8_1.

            Ordre : lettre(s), puis partie numérique (int), puis suffixe optionnel `_indice` (int).
            """
            s = (code or "").strip()
            m = re.match(r"^([A-Za-z]+)(\d+)(?:_(\d+))?$", s)
            if not m:
                # Met les codes non conformes à la fin (fallback lexical)
                return ("~", 10**9, 10**9, s)
            letters = m.group(1).upper()
            num = int(m.group(2))
            idx = int(m.group(3)) if m.group(3) is not None else 0
            return (letters, num, idx, s)

        def _row_sort_key(t: tuple[str, str, str]) -> tuple:
            """Clé de tri globale pour une ligne (priorité au code_vsme)."""
            code_vsme, metric, code_ind = t
            # Préfère `code_vsme` si présent ; sinon fallback sur `Code indicateur`.
            primary = code_vsme or code_ind
            return (
                _code_sort_key(primary),
                _code_sort_key(code_vsme),
                _code_sort_key(code_ind),
                metric,
            )

        items.sort(key=_row_sort_key)

        print("code_vsme\tCode indicateur\tMétrique")
        for code_vsme, metric, code_ind in items:
            print(f"{code_vsme}\t{code_ind}\t{metric}")
        return

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

    extractor = VSMExtractor(retrieval_method=ns.retrieval_method)

    # Format de sortie contrôlé via .env
    # - xlsx (défaut) : export Excel
    # - json : export JSON
    output_format = (os.getenv("VSME_OUTPUT_FORMAT") or "xlsx").strip().lower()
    if output_format not in {"xlsx", "json"}:
        output_format = "xlsx"

    include_json_status = (
        parse_env_bool(os.getenv("VSME_OUTPUT_JSON_INCLUDE_STATUS"))
        if output_format == "json"
        else False
    )

    def _missing_requested_codes(codes_raw: str | None) -> list[str] | None:
        """Retourne la liste (ordonnée) des codes demandés absents du référentiel indicateurs."""
        if codes_raw is None:
            return None
        codes_raw = str(codes_raw).strip()
        if codes_raw == "":
            return None

        requested = [c.strip() for c in re.split(r"[\s,;]+", codes_raw) if c.strip()]
        if not requested:
            return None

        try:
            all_rows = get_indicators(apply_env_filter=False)
            available = {
                str(r.get("code_vsme") or "").strip() for r in all_rows if r.get("code_vsme")
            }
        except Exception:
            return None

        missing = [c for c in requested if c not in available]
        return missing or []

    # -------- Un fichier PDF --------
    if target.is_file() and target.suffix.lower() == ".pdf":
        logger.info("PDF: %s", target)
        logger.info("=== Extraction (single) : %s ===", target.name)

        # Trace l'état des filtres réellement appliqués (utile pour interpréter un JSON partiel).
        code_list_effective = (os.getenv("VSME_CODE_VSME_LIST") or "").strip() or None
        indicators_path_effective = (os.getenv("VSM_INDICATORS_PATH") or "").strip() or None
        missing_codes = _missing_requested_codes(code_list_effective)

        try:
            df, stats = extractor.extract_from_pdf(str(target))
            extraction_error = None
            completed = True
        except Exception as e:
            df = None
            stats = None
            extraction_error = {
                "type": e.__class__.__name__,
                "message": str(e),
            }
            completed = False
        if output_format == "json":
            out_path = target.with_suffix(".vsme.json")
            payload: dict[str, object] = {"pdf": str(target)}
            if include_json_status:
                payload["status"] = {
                    "completed": completed,
                    "error": extraction_error,
                    "filters": {
                        "VSME_CODE_VSME_LIST": code_list_effective,
                        "VSM_INDICATORS_PATH": indicators_path_effective,
                    },
                    "missing_codes": missing_codes,
                }
            payload["results"] = (
                [] if df is None else df.to_dict(orient="records")
            )
            payload["stats"] = (
                None
                if stats is None
                else {
                    "total_input_tokens": stats.total_input_tokens,
                    "total_output_tokens": stats.total_output_tokens,
                    "total_cost_eur": stats.total_cost_eur,
                }
            )
            out_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        else:
            if df is None or stats is None:
                logger.error(
                    "Extraction échouée, aucun export Excel produit | error=%s",
                    extraction_error,
                )
                raise SystemExit(2)
            out_path = target.with_suffix(".vsme.xlsx")
            df.to_excel(out_path, index=False)

        logger.info("Export: %s", out_path)
        if stats is not None:
            logger.info("Tokens input : %s", stats.total_input_tokens)
            logger.info("Tokens output: %s", stats.total_output_tokens)
            logger.info("Coût total   : %.4f €", stats.total_cost_eur)
            logger.info("=== Extraction terminée : %s ===", target.name)
        else:
            logger.error("=== Extraction interrompue : %s ===", target.name)
            raise SystemExit(2)
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
            code_list_effective = (os.getenv("VSME_CODE_VSME_LIST") or "").strip() or None
            indicators_path_effective = (os.getenv("VSM_INDICATORS_PATH") or "").strip() or None
            missing_codes = _missing_requested_codes(code_list_effective)

            try:
                df, stats = extractor.extract_from_pdf(str(pdf))
                extraction_error = None
                completed = True
            except Exception as e:
                df = None
                stats = None
                extraction_error = {
                    "type": e.__class__.__name__,
                    "message": str(e),
                }
                completed = False
            if output_format == "json":
                out_path = pdf.with_suffix(".vsme.json")
                payload: dict[str, object] = {"pdf": str(pdf)}
                if include_json_status:
                    payload["status"] = {
                        "completed": completed,
                        "error": extraction_error,
                        "filters": {
                            "VSME_CODE_VSME_LIST": code_list_effective,
                            "VSM_INDICATORS_PATH": indicators_path_effective,
                        },
                        "missing_codes": missing_codes,
                    }
                payload["results"] = (
                    [] if df is None else df.to_dict(orient="records")
                )
                payload["stats"] = (
                    None
                    if stats is None
                    else {
                        "total_input_tokens": stats.total_input_tokens,
                        "total_output_tokens": stats.total_output_tokens,
                        "total_cost_eur": stats.total_cost_eur,
                    }
                )
                out_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            else:
                if df is None or stats is None:
                    logger.error(
                        "Extraction échouée, aucun export Excel produit | pdf=%s | error=%s",
                        pdf,
                        extraction_error,
                    )
                    continue
                out_path = pdf.with_suffix(".vsme.xlsx")
                df.to_excel(out_path, index=False)

            logger.info("Export: %s", out_path)
            if stats is not None:
                logger.info("Tokens input : %s", stats.total_input_tokens)
                logger.info("Tokens output: %s", stats.total_output_tokens)
                logger.info("Coût total   : %.4f €", stats.total_cost_eur)
                logger.info("=== Extraction terminée : %s ===", pdf.name)
            else:
                logger.error("=== Extraction interrompue : %s ===", pdf.name)

        return


if __name__ == "__main__":
    main()
