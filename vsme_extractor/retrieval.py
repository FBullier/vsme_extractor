from __future__ import annotations

import math
import re
from typing import List, Literal


def _tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"\W+", (text or "").lower()) if len(t) > 2]


def find_relevant_snippets(
    query: str,
    page_texts: List[str],
    k: int = 6,
    method: Literal["count", "bm25"] = "count",
) -> List[str]:
    """
    Retourne les k meilleurs extraits/pages les plus pertinents pour `query`.

    Arguments :
        query: Chaîne de recherche (souvent des mots-clés).
        page_texts: Liste des textes de pages (un élément par page).
        k: Nombre d'extraits à retourner.
        method:
          - "count": matching lexical en comptant les occurrences (comportement historique).
          - "bm25": score BM25 simple (sans dépendances supplémentaires).

    Notes :
        - Les deux méthodes sont purement lexicales et fonctionnent mieux avec un texte “propre”.
        - Les pages avec un score <= 0 sont exclues.
    """
    tokens = _tokenize(query)
    if not tokens or not page_texts:
        return []

    if method == "count":
        scores = []
        for i, txt in enumerate(page_texts):
            t = (txt or "").lower()
            score = sum(t.count(tok) for tok in tokens)
            if score > 0:
                scores.append((score, i, txt))
        scores.sort(reverse=True, key=lambda x: x[0])
        return [s[2] for s in scores[:k]]

    if method == "bm25":
        # Paramètres BM25
        k1 = 1.5
        b = 0.75

        # Pré-calcul : fréquences par document (TF) et fréquences documentaires (DF)
        doc_tfs: list[dict[str, int]] = []
        doc_lens: list[int] = []
        df: dict[str, int] = {}

        for txt in page_texts:
            toks = _tokenize(txt)
            doc_lens.append(len(toks) or 1)
            tf: dict[str, int] = {}
            for tok in toks:
                tf[tok] = tf.get(tok, 0) + 1
            doc_tfs.append(tf)

            # fréquence documentaire
            for tok in set(toks):
                df[tok] = df.get(tok, 0) + 1

        n_docs = len(page_texts)
        avgdl = (sum(doc_lens) / n_docs) if n_docs else 1.0

        # L'IDF n'est nécessaire que pour les tokens de la requête
        def idf(term: str) -> float:
            n_q = df.get(term, 0)
            # Lissage “style BM25+” pour éviter des valeurs négatives sur des termes très fréquents
            return math.log(1.0 + (n_docs - n_q + 0.5) / (n_q + 0.5))

        scores = []
        for i, (txt, tf, dl) in enumerate(zip(page_texts, doc_tfs, doc_lens)):
            score = 0.0
            for term in tokens:
                f = tf.get(term, 0)
                if f <= 0:
                    continue
                denom = f + k1 * (1.0 - b + b * (dl / avgdl))
                score += idf(term) * (f * (k1 + 1.0)) / denom
            if score > 0:
                scores.append((score, i, txt))

        scores.sort(reverse=True, key=lambda x: x[0])
        return [s[2] for s in scores[:k]]

    raise ValueError(f"Méthode inconnue : {method!r}. Attendu : 'count' ou 'bm25'.")
