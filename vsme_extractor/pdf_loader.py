from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

from langchain_community.document_loaders import PyPDFLoader

logger = logging.getLogger(__name__)


def load_pdf(pdf_path: str | Path) -> Tuple[list, List[str], str]:
    """Charge un PDF et retourne (pages, textes_par_page, texte_complet).

    Utilise `PyPDFLoader` (langchain-community) et concatène les pages.
    """
    pdf_path = Path(pdf_path)

    logger.info("START load_pdf | path=%s", pdf_path)

    try:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load_and_split()
        page_texts = [p.page_content for p in pages]
        full_text = "\n\n".join(page_texts)

        logger.info(
            "END load_pdf | pages=%s | chars_full_text=%s",
            len(pages),
            len(full_text),
        )
        return pages, page_texts, full_text
    except Exception:
        logger.exception("FAILED load_pdf | path=%s", pdf_path)
        raise
