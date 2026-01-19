from __future__ import annotations

import logging
import os
import re
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List

import chardet
import pandas as pd

logger = logging.getLogger(__name__)

PACKAGE_DATA_MODULE = "vsme_extractor"
PACKAGE_CSV_NAME = "indicateurs_vsme.csv"
DEFAULT_INDICATORS_PATH = Path(__file__).parent / "data" / PACKAGE_CSV_NAME


def get_indicators(
    path: str | Path | None = None,
    *,
    apply_env_filter: bool = True,
) -> List[Dict[str, Any]]:
    if path is None:
        env_path = os.getenv("VSM_INDICATORS_PATH")

        if env_path:
            env_path = Path(env_path).expanduser().resolve()
            if env_path.exists():
                path = env_path
            else:
                logger.warning(
                    "La variable d’environnement VSM_INDICATORS_PATH est définie mais le fichier n’existe pas : %s. Utilisation du CSV packagé en fallback.",
                    env_path,
                )
                resource = resources.files(PACKAGE_DATA_MODULE) / "data" / PACKAGE_CSV_NAME
                with resources.as_file(resource) as p:
                    path = Path(p)
        else:
            resource = resources.files(PACKAGE_DATA_MODULE) / "data" / PACKAGE_CSV_NAME
            with resources.as_file(resource) as p:
                path = Path(p)

    def detect_encoding(p: str | Path):
        with open(p, "rb") as f:
            raw = f.read()
        return chardet.detect(raw)

    info = detect_encoding(path)

    df = pd.read_csv(
        path,
        sep=";",
        encoding=info.get("encoding"),
        on_bad_lines="skip",
    )

    if apply_env_filter:
        # Optional filtering by `code_vsme` via .env
        # - If VSME_CODE_VSME_LIST is set and non-empty: keep only these `code_vsme`
        # - Else (missing/empty): keep only rows where `defaut` == 1
        codes_raw = (os.getenv("VSME_CODE_VSME_LIST") or "").strip()
        if codes_raw:
            if "code_vsme" not in df.columns:
                logger.warning(
                    "VSME_CODE_VSME_LIST est défini mais la colonne 'code_vsme' est absente du CSV (%s). Aucun filtrage appliqué.",
                    path,
                )
            else:
                codes = [c.strip() for c in re.split(r"[\s,;]+", codes_raw) if c.strip()]
                df = df[df["code_vsme"].astype(str).isin(codes)]
        else:
            if "defaut" in df.columns:
                df = df[df["defaut"].astype(str).str.strip() == "1"]
            else:
                logger.warning(
                    "VSME_CODE_VSME_LIST est vide/absent et la colonne 'defaut' est absente du CSV (%s). Aucun filtrage appliqué.",
                    path,
                )

    return df.to_dict(orient="records")
