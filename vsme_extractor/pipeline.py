from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Dict, Literal, Tuple

import pandas as pd
from langdetect import LangDetectException, detect
from langdetect.detector_factory import DetectorFactory


from .config import (
    EURO_COST_PER_MILLION_INPUT,
    EURO_COST_PER_MILLION_OUTPUT,
    load_llm_config,
)
from .extraction import extract_value_for_metric
from .indicators import get_indicators
from .llm_client import LLM
from .pdf_loader import load_pdf
from .retrieval import find_relevant_snippets


logger = logging.getLogger(__name__)


@dataclass
class ExtractionStats:
    total_input_tokens: int
    total_output_tokens: int
    total_cost_eur: float


# Garantit que langdetect est déterministe d'une exécution à l'autre
DetectorFactory.seed = 0


def detect_document_language(page_texts: list[str]) -> str:
    """
    Détecte la langue principale du document à partir des premières pages.

    Retourne :
        - 'fr' pour français,
        - 'en' pour anglais,
        - ou un code ISO à 2 lettres renvoyé par langdetect.
    """
    # Concatène les premières pages pour obtenir un échantillon représentatif
    sample = " ".join(page_texts[:5])  # ~5 pages suffisent généralement
    sample = sample.strip()

    if len(sample) < 50:
        # Pas assez de texte : par défaut on considère "en"
        return "en"

    try:
        lang = detect(sample)
        return lang  # ex. 'fr', 'en', 'de', ...
    except LangDetectException:
        return "en"


class VSMExtractor:
    def __init__(
        self,
        top_k_snippets: int = 6,
        temperature: float = 0.2,
        max_tokens: int = 512,
        retrieval_method: Literal["count", "bm25"] = "count",
    ):
        self.config = load_llm_config()
        self.llm = LLM(self.config)

        # Audit : petit “healthcheck” pour tracer dans les logs que le LLM répond.
        # On garde ça minimal pour limiter coût/latence (requiert tout de même un accès réseau).
        try:
            health = self.llm.invoke(
                question="Réponds uniquement: OK",
                temperature=0.0,
                max_tokens=4,
                system_prompt="Réponds uniquement: OK",
            )
            ok_text = (health.get("text", "") or "").strip()
            logger.info(
                "LLM opérationnel | model=%s | base_url=%s | response=%s",
                self.config.model,
                self.config.base_url,
                ok_text,
            )
        except Exception:
            logger.exception("LLM check failed | model=%s | base_url=%s", self.config.model, self.config.base_url)
            raise

        self.top_k_snippets = top_k_snippets
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retrieval_method = retrieval_method

        # Cache de traduction des mots-clés : (lang, keywords) -> translated_keywords
        self._keywords_translation_cache: Dict[tuple[str, str], str] = {}

    def extract_from_pdf(self, pdf_path: str) -> Tuple[pd.DataFrame, ExtractionStats]:
        t_total0 = time.perf_counter()
        logger.info("START extract_from_pdf | pdf_path=%s | retrieval_method=%s", pdf_path, self.retrieval_method)

        t0 = time.perf_counter()
        pages, page_texts, full_text = load_pdf(pdf_path)
        t_load_pdf = time.perf_counter() - t0
        logger.info(
            "Step load_pdf | duration_s=%.3f | pages=%s | chars_full_text=%s",
            t_load_pdf,
            len(pages),
            len(full_text),
        )

        # Charge la liste des indicateurs à chercher
        t0 = time.perf_counter()
        indicateurs = get_indicators()
        t_indicators = time.perf_counter() - t0
        logger.info("Step get_indicators | duration_s=%.3f | count=%s", t_indicators, len(indicateurs))

        # Détecte la langue du document
        t0 = time.perf_counter()
        lang = detect_document_language(page_texts)
        t_lang = time.perf_counter() - t0
        logger.info("Step detect_language | duration_s=%.3f | lang=%s", t_lang, lang)

        if lang == "fr":
            keywords_tag = "Mots clés"
        else:
            keywords_tag = "Keywords"

        results = []
        tot_in_tokens = 0
        tot_out_tokens = 0

        t0 = time.perf_counter()
        # Boucle sur chaque indicateur
        for idx, row in enumerate(indicateurs, start=1):
            ind_t0 = time.perf_counter()

            # Récupère le “profil” de l’indicateur
            metric = row["Métrique"]
            unite = row["Unité / Détail"]
            keywords = row.get(keywords_tag, "")

            code = row.get("Code indicateur", "NA")
            logger.info(
                "Indicator %s/%s | code=%s | metric=%s | unit=%s",
                idx,
                len(indicateurs),
                code,
                metric,
                unite,
            )
            logger.debug("Indicator details | code=%s | metric=%s | unit=%s", code, metric, unite)

            # Traduit les mots-clés si besoin (avec cache par (lang, keywords))
            if lang not in ["en", "fr"] and keywords:
                cache_key = (lang, str(keywords))
                translated = self._keywords_translation_cache.get(cache_key)
                if translated is None:
                    logger.debug("Keywords translation | cache=miss | lang=%s | keywords=%s", lang, keywords)
                    promt_conv_lg = (
                        f"Traduit les mots clés de la chaine suivante en langue ISO {lang} : {keywords}\n"
                        "Répond au format : mot1,mot2,mot3,.. \n"
                        "N'ajoute aucune autre information ni remarques.\n"
                        "Si tu ne connais pas la langue, renvoie une chaine vide."
                    )
                    reponse = self.llm.invoke(promt_conv_lg, max_tokens=128)
                    translated = (reponse.get("text", "") or "").strip()
                    # Met aussi en cache une chaîne vide pour éviter de rappeler le LLM
                    self._keywords_translation_cache[cache_key] = translated
                else:
                    logger.debug("Keywords translation | cache=hit | lang=%s", lang)

                if translated:
                    keywords = translated

            # Sélectionne les extraits/pages les plus pertinents via les mots-clés
            ctx_selected = find_relevant_snippets(
                query=str(keywords),
                page_texts=page_texts,
                k=self.top_k_snippets,
                method=self.retrieval_method,
            )
            logger.debug(
                "Retrieval | snippets=%s | method=%s",
                len(ctx_selected),
                self.retrieval_method,
            )

            # Sinon, fallback : utiliser le début du document
            if not ctx_selected:
                logger.debug("Fallback retrieval | utilisation de l'en-tête du document")
                ctx_selected = [full_text[:10000]]

            # Contexte réellement envoyé au LLM (limité à 6 extraits côté prompt)
            ctx_used = list(ctx_selected[:6])
            context_joined = "\n\n---\n\n".join(ctx_used)
            chars_context = len(context_joined)
            est_tokens_context = max(1, int(round(chars_context / 4)))  # heuristique ~4 chars/token

            logger.info(
                "Indicator context %s/%s | code=%s | snippets_selected=%s | snippets_used=%s | chars_context=%s | est_tokens_context=%s",
                idx,
                len(indicateurs),
                code,
                len(ctx_selected),
                len(ctx_used),
                chars_context,
                est_tokens_context,
            )

            # Extrait la valeur de l’indicateur à partir du contexte sélectionné
            data, in_tokens, out_tokens, _ = extract_value_for_metric(
                llm=self.llm,
                metric=metric,
                unite=unite,
                contexte_snippets=ctx_used,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            logger.debug(
                "LLM extract | in_tokens=%s | out_tokens=%s | valeur=%s",
                in_tokens,
                out_tokens,
                data.get("valeur"),
            )

            tot_in_tokens += in_tokens
            tot_out_tokens += out_tokens

            ind_dt = time.perf_counter() - ind_t0
            logger.info(
                "Indicator done %s/%s | code=%s | duration_s=%.3f | valeur=%s | unit=%s",
                idx,
                len(indicateurs),
                code,
                ind_dt,
                data.get("valeur"),
                data.get("unité"),
            )

            results.append(
                {
                    "Code indicateur": row["Code indicateur"],
                    "Thématique":row["Thématique"],
                    "Métrique":row["Métrique"],
                    "Valeur": data["valeur"],
                    "Unité extraite": data["unité"],
                    "Paragraphe source": data["paragraphe"],
                }
            )

        t_loop = time.perf_counter() - t0
        logger.info("Step loop_indicators | duration_s=%.3f | indicators=%s", t_loop, len(indicateurs))

        # Calcule le coût total estimé (à partir des tokens)
        total_cost = (
            EURO_COST_PER_MILLION_INPUT * tot_in_tokens
            + EURO_COST_PER_MILLION_OUTPUT * tot_out_tokens
        ) / 1e6

        # Statistiques à retourner
        stats = ExtractionStats(
            total_input_tokens=tot_in_tokens,
            total_output_tokens=tot_out_tokens,
            total_cost_eur=total_cost,
        )

        t_total = time.perf_counter() - t_total0
        logger.info(
            "END extract_from_pdf | duration_s=%.3f | input_tokens=%s | output_tokens=%s | cost_eur=%.6f",
            t_total,
            stats.total_input_tokens,
            stats.total_output_tokens,
            stats.total_cost_eur,
        )

        # Indicateurs avec leur valeur (NA = non trouvé)
        df = pd.DataFrame(results)

        return df, stats
