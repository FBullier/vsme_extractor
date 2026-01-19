from __future__ import annotations

from collections.abc import Mapping
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional, Dict

from openai import OpenAI
from tiktoken import get_encoding

from .config import (
    LLMConfig,
    EURO_COST_PER_MILLION_INPUT,
    EURO_COST_PER_MILLION_OUTPUT,
)
from .logging_utils import SizedTimedRotatingFileHandler


PROMPTS_LOGGER = logging.getLogger("vsme_extractor.prompts")
_PROMPTS_LOGGER_CONFIGURED = False


def _env_bool(name: str) -> bool:
    """
    Parse une variable d’environnement booléenne : 1/0, true/false, yes/no, on/off.

    Par défaut : False si la variable n’existe pas ou si la valeur est non reconnue.
    """
    v = os.getenv(name)
    if v is None:
        return False
    s = v.strip().lower()
    return s in {"1", "true", "yes", "y", "on"}


def _format_prompt_block(*, model: str, base_url: str, system: str, user: str) -> str:
    """Formate un bloc de prompt (system+user) pour audit/logging."""
    # Sépare clairement les blocs pour faciliter la relecture/audit
    return (
        "\n"
        "==================== LLM PROMPT (BEGIN) ====================\n"
        f"model: {model}\n"
        f"base_url: {base_url}\n"
        "-------------------- SYSTEM --------------------\n"
        f"{system}\n"
        "--------------------- USER ---------------------\n"
        f"{user}\n"
        "===================== LLM PROMPT (END) =====================\n"
    )


def _configure_prompts_logger() -> bool:
    """
    Configure les sorties de logging des prompts (séparées du logger applicatif global).

    Variables d’environnement :
      - VSME_PROMPT_FILE : chemin vers un fichier de trace des prompts (optionnel)
      - VSME_PROMPT_STDOUT : 1/0, true/false, yes/no, on/off

    Comportement :
      - Si aucune destination n’est activée, le logging des prompts est désactivé.
      - Les prompts ne propagent jamais vers le root logger (ils n’apparaissent donc PAS dans VSME_LOG_FILE).
      - Si un fichier est configuré, il est soumis à la même rotation que les logs applicatifs :
        rotation quotidienne + rotation à ~10 Mo, avec rétention 7 jours.
    """
    global _PROMPTS_LOGGER_CONFIGURED

    prompt_file = (os.getenv("VSME_PROMPT_FILE") or "").strip() or None
    prompt_stdout = _env_bool("VSME_PROMPT_STDOUT")

    enabled = bool(prompt_file) or prompt_stdout
    if not enabled:
        return False

    if _PROMPTS_LOGGER_CONFIGURED:
        return True

    PROMPTS_LOGGER.setLevel(logging.INFO)
    PROMPTS_LOGGER.propagate = (
        False  # critique : ne doit pas "fuiter" vers les handlers du root
    )

    # Garantit l'idempotence si appelé plusieurs fois
    for h in list(PROMPTS_LOGGER.handlers):
        PROMPTS_LOGGER.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    if prompt_stdout:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        PROMPTS_LOGGER.addHandler(sh)

    if prompt_file:
        path = Path(prompt_file)
        path.parent.mkdir(parents=True, exist_ok=True)

        fh = SizedTimedRotatingFileHandler(
            path,
            when="midnight",
            interval=1,
            utc=False,
            max_bytes=10_000_000,  # < 10 Mo
            retention_days=7,
        )
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        PROMPTS_LOGGER.addHandler(fh)

    _PROMPTS_LOGGER_CONFIGURED = True
    return True


def _safe_log_prompt(*, model: str, base_url: str, system: str, user: str) -> None:
    """Loggue le prompt si (et seulement si) la config d'audit des prompts est activée."""
    if not _configure_prompts_logger():
        return
    PROMPTS_LOGGER.info(
        "%s",
        _format_prompt_block(model=model, base_url=base_url, system=system, user=user),
    )


def _safe_get_usage(resp: Any) -> Optional[Dict[str, int]]:
    """Récupère les compteurs de tokens depuis une réponse provider (best-effort)."""
    usage = getattr(resp, "usage", None)
    if usage is not None:
        if hasattr(usage, "prompt_tokens"):
            return dict(
                prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
            )
        if isinstance(usage, Mapping):
            return dict(
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                completion_tokens=int(usage.get("completion_tokens") or 0),
                total_tokens=int(usage.get("total_tokens") or 0),
            )
    try:
        data = resp.model_dump()
        if "usage" in data:
            u = data["usage"]
            return dict(
                prompt_tokens=int(u.get("prompt_tokens") or 0),
                completion_tokens=int(u.get("completion_tokens") or 0),
                total_tokens=int(u.get("total_tokens") or 0),
            )
    except Exception:
        pass
    return None


def _estimate_tokens(text: str) -> int:
    """Estime le nombre de tokens pour un texte (tiktoken si dispo, sinon heuristique)."""
    try:
        enc = get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, int(round(len(text) / 4)))


def _usage_from_obj(u):
    """Normalise un objet/dict 'usage' vers un dict standard."""
    if not u:
        return None
    if hasattr(u, "prompt_tokens"):
        return {
            "prompt_tokens": int(getattr(u, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(u, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(u, "total_tokens", 0) or 0),
        }
    if isinstance(u, dict):
        return {
            "prompt_tokens": int(u.get("prompt_tokens") or 0),
            "completion_tokens": int(u.get("completion_tokens") or 0),
            "total_tokens": int(u.get("total_tokens") or 0),
        }
    return None


class LLM:
    """Client LLM OpenAI-compatible avec estimation de tokens et coût."""

    def __init__(self, config: LLMConfig):
        """Initialise le client OpenAI-compatible avec la config fournie."""
        self.config = config
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
        )

    def invoke(
        self,
        question: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
        system_prompt: Optional[str] = None,
    ):
        """Effectue un appel non-stream au LLM et renvoie texte + métriques de tokens/coût."""

        extra_body = {"reasoning": {"effort": "low"}}  # ou "medium"/"high"
        system = system_prompt or self.config.system_prompt

        _safe_log_prompt(
            model=self.config.model,
            base_url=self.config.base_url,
            system=system,
            user=question,
        )

        resp = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.1,
            presence_penalty=0,
            stream=False,
            response_format={"type": "text"},
            extra_body=extra_body,  # spécifique au provider
        )

        choice0 = resp.choices[0]
        msg = getattr(choice0, "message", None) or {}
        if isinstance(msg, Mapping):
            text = msg.get("content") or msg.get("text") or ""
        else:
            text = getattr(msg, "content", "") or ""

        if not text:
            try:
                data = resp.model_dump()
                text = data["choices"][0]["message"]["content"]
            except Exception:
                text = str(resp)

        usage = _safe_get_usage(resp)
        if usage:
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)
            total_tokens = int(
                usage.get("total_tokens") or (prompt_tokens + completion_tokens)
            )
            used_api_usage = True
        else:
            prompt_tokens = _estimate_tokens(f"{system}\n{question}")
            completion_tokens = _estimate_tokens(text)
            total_tokens = prompt_tokens + completion_tokens
            used_api_usage = False

        cost_prompt_eur = (prompt_tokens / 1_000_000) * EURO_COST_PER_MILLION_INPUT
        cost_completion_eur = (
            completion_tokens / 1_000_000
        ) * EURO_COST_PER_MILLION_OUTPUT

        return {
            "text": text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost_eur": {
                "input": cost_prompt_eur,
                "output": cost_completion_eur,
                "total": cost_prompt_eur + cost_completion_eur,
            },
            "used_api_usage": used_api_usage,
        }

    # Variante streaming : écrit les tokens au fil de l'eau (optionnel) et retourne les métriques d'usage
    def invoke_stream(
        self,
        question: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
        system_prompt: Optional[str] = None,
        print_tokens=True,
    ):
        """
        Stream la réponse sur stdout au fur et à mesure et retourne :

        {
          text,
          prompt_tokens, completion_tokens, total_tokens,
          cost_eur: {input, output, total},
          used_api_usage (bool),
          finish_reason
        }
        """
        parts = []
        usage_final = None
        finish_reason = None

        extra_body = {"reasoning": {"effort": "low"}}  # ou "medium"/"high"
        system = system_prompt or self.config.system_prompt

        _safe_log_prompt(
            model=self.config.model,
            base_url=self.config.base_url,
            system=system,
            user=question,
        )

        stream = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.1,
            presence_penalty=0,
            stream=True,
            stream_options={
                "include_usage": True
            },  # si supporté, ajoute un dernier chunk contenant l'usage
            extra_body=extra_body,  # spécifique au provider
        )

        for chunk in stream:
            # 1) Tokens texte
            try:
                choice = chunk.choices[0]
            except Exception:
                choice = None

            if choice is not None:
                delta = getattr(choice, "delta", None)
                token = getattr(delta, "content", None) if delta is not None else None

                # Certains proxies renvoient des dicts
                if token is None and isinstance(delta, dict):
                    token = delta.get("content")

                if token:
                    if print_tokens:
                        sys.stdout.write(token)
                        sys.stdout.flush()
                    parts.append(token)

                # récupère finish_reason lorsqu'il apparaît (souvent dans le dernier chunk)
                fr = getattr(choice, "finish_reason", None)
                if fr is None and isinstance(choice, dict):
                    fr = choice.get("finish_reason")
                if fr:
                    finish_reason = fr

            # 2) Usage (si le proxy l'émet dans un chunk final)
            u = getattr(chunk, "usage", None)
            if u is None and isinstance(chunk, dict):
                u = chunk.get("usage")
            u_norm = _usage_from_obj(u)
            if u_norm:
                usage_final = u_norm

        if print_tokens:
            sys.stdout.write("\n")
            sys.stdout.flush()

        text = "".join(parts)

        # Calcule les tokens (préfère l'usage exact si disponible)
        if usage_final:
            prompt_tokens = int(usage_final["prompt_tokens"] or 0)
            completion_tokens = int(usage_final["completion_tokens"] or 0)
            total_tokens = int(
                usage_final.get("total_tokens", prompt_tokens + completion_tokens)
            )

            used_api_usage = True

        else:
            prompt_tokens = _estimate_tokens(f"{system}\n{question}")
            completion_tokens = _estimate_tokens(text)
            total_tokens = prompt_tokens + completion_tokens
            used_api_usage = False

        # Calcul des coûts
        cost_prompt_eur = (prompt_tokens / 1_000_000) * EURO_COST_PER_MILLION_INPUT
        cost_completion_eur = (
            completion_tokens / 1_000_000
        ) * EURO_COST_PER_MILLION_OUTPUT

        return {
            "text": text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost_eur": {
                "input": cost_prompt_eur,
                "output": cost_completion_eur,
                "total": cost_prompt_eur + cost_completion_eur,
            },
            "used_api_usage": used_api_usage,
            "finish_reason": finish_reason,
        }
