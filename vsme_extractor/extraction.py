from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Sequence, Tuple

from .llm_client import LLM, _estimate_tokens
from .prompts import EXTRACTION_SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)


def extract_value_for_metric(
    llm: LLM,
    metric: str,
    unite: str,
    contexte_snippets: Sequence[str],
    temperature: float = 0.2,
    max_tokens: int = 512,
    *,
    json_repair: bool = True,
) -> Tuple[Dict[str, Any], int, int, str]:
    """Extrait une valeur (et unité/paragraphe) pour une métrique via le LLM.

    Retourne un tuple :
    - dict normalisé {"valeur", "unité", "paragraphe"}
    - prompt_tokens (int)
    - completion_tokens (int)
    - trace du prompt (str)
    """
    context_joined = "\n\n---\n\n".join(contexte_snippets[:6])
    user_prompt = build_user_prompt(metric, unite)

    # IMPORTANT: do not duplicate the system prompt.
    # The system prompt is passed via `system_prompt=...` and the user prompt goes in `question`.
    prompt_trace = (
        f"SYSTEM:\n{EXTRACTION_SYSTEM_PROMPT}"
        f"\n\nCONTEXT:\n{context_joined}"
        f"\n\nUSER:\n{user_prompt}"
    )

    logger.debug("START extract_value_for_metric | metric=%s | unit=%s", metric, unite)
    raw = llm.invoke(
        question=user_prompt.strip(),
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
        context=context_joined,
    )

    text = raw.get("text", "") or ""
    in_tokens = int(raw.get("prompt_tokens") or 0)
    out_tokens = int(raw.get("completion_tokens") or 0)

    # Debug: allow verifying whether high output_tokens come from genuinely long text
    # or from provider-reported usage. Does not log the content.
    try:
        logger.debug(
            "LLM response stats | metric=%s | used_api_usage=%s | chars_out=%s | est_out_tokens=%s | reported_out_tokens=%s",
            metric,
            bool(raw.get("used_api_usage")),
            len(text),
            _estimate_tokens(text),
            out_tokens,
        )
    except Exception:
        pass

    def _normalize(data: Dict[str, Any]) -> Dict[str, str]:
        """Normalise la sortie modèle vers les 3 champs attendus."""
        valeur = str(data.get("valeur", "NA")).strip()
        unite_ret = str(data.get("unité", data.get("unite", "NA"))).strip()
        paragraphe = str(data.get("paragraphe", "NA")).strip()
        return {
            "valeur": valeur or "NA",
            "unité": unite_ret or "NA",
            "paragraphe": paragraphe or "NA",
        }

    try:
        data = json.loads(text)
        logger.debug("JSON parse | status=ok")
        return (
            _normalize(data),
            in_tokens,
            out_tokens,
            prompt_trace,
        )
    except Exception:
        logger.debug("JSON parse | status=failed | attempting fallback")

        # 1) Best-effort regex extraction for the 3 fields
        def _extract(pattern: str) -> str | None:
            """Extrait un champ via regex (best-effort)."""
            m = re.search(pattern, text, re.S | re.I)
            return m.group(1).strip() if m else None

        valeur = _extract(r'"valeur"\s*:\s*"([^"]*?)"') or _extract(
            r"'valeur'\s*:\s*'([^']*?)'"
        )
        unite_ret = (
            _extract(r'"unité"\s*:\s*"([^"]*?)"')
            or _extract(r'"unite"\s*:\s*"([^"]*?)"')
            or _extract(r"'unité'\s*:\s*'([^']*?)'")
            or _extract(r"'unite'\s*:\s*'([^']*?)'")
        )
        paragraphe = _extract(r'"paragraphe"\s*:\s*"([^"]*?)"') or _extract(
            r"'paragraphe'\s*:\s*'([^']*?)'"
        )

        extracted = {
            "valeur": (valeur or "NA"),
            "unité": (unite_ret or "NA"),
            "paragraphe": (paragraphe or "NA"),
        }

        # 2) Optional single repair pass: ask the model to output strict JSON only
        # On évite un appel "repair" si la réponse modèle est vide.
        if (
            json_repair
            and text.strip() != ""
            and ("NA" in extracted.values() or "{" in text or "}" in text)
        ):
            logger.debug("JSON repair | status=attempt")
            repair_system = "Tu es un assistant de correction de JSON. Tu réponds uniquement en JSON valide."
            repair_user = (
                "Le texte suivant est censé être un JSON avec les champs "
                '"valeur", "unité", "paragraphe". '
                "Corrige-le et renvoie UNIQUEMENT un JSON valide au format:\n"
                '{"valeur":"...","unité":"...","paragraphe":"..."}\n\n'
                f"TEXTE:\n{text}"
            )
            raw2 = llm.invoke(
                question=repair_user,
                temperature=0.0,
                max_tokens=256,
                system_prompt=repair_system,
            )
            text2 = raw2.get("text", "") or ""
            in_tokens += int(raw2.get("prompt_tokens") or 0)
            out_tokens += int(raw2.get("completion_tokens") or 0)
            prompt_trace = prompt_trace + "\n\nREPAIR:\n" + repair_user

            try:
                data2 = json.loads(text2)
                logger.debug("JSON repair | status=ok")
                return (
                    _normalize(data2),
                    in_tokens,
                    out_tokens,
                    prompt_trace,
                )
            except Exception:
                logger.debug("JSON repair | status=failed")
                pass

        logger.debug(
            "END extract_value_for_metric | status=fallback | valeur=%s",
            extracted.get("valeur"),
        )
        return (
            _normalize(extracted),
            in_tokens,
            out_tokens,
            prompt_trace,
        )
