from __future__ import annotations
import os
from dataclasses import dataclass

# Coûts, éventuellement overridables par .env
EURO_COST_PER_MILLION_INPUT = float(os.getenv("VSM_INPUT_COST_EUR", 0.15))
EURO_COST_PER_MILLION_OUTPUT = float(os.getenv("VSM_OUTPUT_COST_EUR", 0.60))


@dataclass
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    system_prompt: str = "You are a helpful assistant"


def load_llm_config() -> LLMConfig:
    api_key = os.getenv("SCW_API_KEY")
    if not api_key:
        raise RuntimeError("SCW_API_KEY manquant. Définis-le dans ton environnement ou ton .env.")

    base_url = os.getenv(
        "SCW_BASE_URL",
        "https://api.scaleway.ai/06f1a171-1eef-4d8b-aed5-b78189d17335/v1",
    )
    model = os.getenv("SCW_MODEL_NAME", "gpt-oss-120b")

    return LLMConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
    )
