from __future__ import annotations

import logging
import os
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List

import chardet
import pandas as pd

logger = logging.getLogger(__name__)

PACKAGE_DATA_MODULE = "vsme_extractor.data"
PACKAGE_CSV_NAME = "indicateurs_sup_b1_b11.csv"
DEFAULT_INDICATORS_PATH = Path(__file__).parent / "data" / PACKAGE_CSV_NAME


def get_indicators(path: str | Path | None = None) -> List[Dict[str, Any]]:
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
                resource = resources.files(PACKAGE_DATA_MODULE) / PACKAGE_CSV_NAME
                with resources.as_file(resource) as p:
                    path = Path(p)
        else:
            resource = resources.files(PACKAGE_DATA_MODULE) / PACKAGE_CSV_NAME
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

    return df.to_dict(orient="records")
