from __future__ import annotations

import math
import re
import unicodedata
from typing import Callable, List, Literal, Sequence


def _tokenize(text: str) -> list[str]:
    """Tokenise un texte en mots (min 3 caractères), en minuscules."""
    return [t for t in re.split(r"\W+", (text or "").lower()) if len(t) > 2]


_SUBSCRIPT_DIGITS = str.maketrans(
    {
        "₀": "0",
        "₁": "1",
        "₂": "2",
        "₃": "3",
        "₄": "4",
        "₅": "5",
        "₆": "6",
        "₇": "7",
        "₈": "8",
        "₉": "9",
    }
)


def _normalize_text_soft(text: str) -> str:
    """Normalisation légère pour améliorer la robustesse (Unicode/OCR/tirets).

    Objectif : se rapprocher de la tolérance du matching substring (mode `count`)
    tout en restant dans un cadre "token-based" compatible BM25.
    """
    t = text or ""

    # Normalisation Unicode (ex. compatibilité) + conversion des chiffres en indice.
    t = unicodedata.normalize("NFKC", t).translate(_SUBSCRIPT_DIGITS)

    # Supprime les césures de fin de ligne (classique OCR/PDF) : "emis-\nsions" -> "emissions".
    t = re.sub(r"([A-Za-z])\-\s*\n\s*([A-Za-z])", r"\1\2", t)

    # Harmonise les tirets (– — −) en '-'.
    t = t.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")

    # Aplatis les retours ligne.
    t = re.sub(r"\s+", " ", t)
    return t.strip().lower()


def _strip_accents(text: str) -> str:
    """Retire les diacritiques (é -> e) pour tolérer les variations OCR."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _tokenize_soft(text: str) -> list[str]:
    """Tokenisation plus permissive que [`_tokenize()`](vsme_extractor/retrieval.py:8).

    Principes :
    - normalisation Unicode/OCR
    - gestion des tirets ("scope-1" ~ "scope 1" ~ "scope1")
    - retrait des accents (tolérance OCR)

    Notes :
    - On garde des tokens de longueur >= 2 (utile pour "co2", "s1" etc.).
    - On génère des variantes sans tiret pour retrouver du signal que `count` capte en substring.
    """
    base = _normalize_text_soft(text)
    base = _strip_accents(base)

    # 1) version avec tirets remplacés par espace (tokenisation classique)
    v1 = base.replace("-", " ")
    toks1 = re.findall(r"[a-z0-9]{2,}", v1)

    # 2) version avec tirets supprimés (pour matcher "scope-1" vs "scope1")
    v2 = base.replace("-", "")
    toks2 = re.findall(r"[a-z0-9]{2,}", v2)

    # Dé-duplication tout en gardant un ordre stable (pour debug).
    seen: set[str] = set()
    out: list[str] = []
    for tok in [*toks1, *toks2]:
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out


def _count_scores(query: str, page_texts: List[str]) -> list[tuple[int, int, str]]:
    """Scores `count` (substring) : retourne [(score, i, txt)] trié décroissant."""
    tokens = _tokenize(query)
    if not tokens or not page_texts:
        return []

    scores: list[tuple[int, int, str]] = []
    for i, txt in enumerate(page_texts):
        t = (txt or "").lower()
        score = sum(t.count(tok) for tok in tokens)
        if score > 0:
            scores.append((score, i, txt))

    scores.sort(reverse=True, key=lambda x: x[0])
    return scores


def _bm25_scores(
    *,
    query: str,
    page_texts: List[str],
    tokenize: Callable[[str], list[str]],
    candidate_indices: set[int] | None = None,
) -> list[tuple[float, int, str]]:
    """Scores BM25 : retourne [(score, i, txt)] trié décroissant.

    `candidate_indices` (optionnel) permet de ne scorer que certaines pages,
    tout en calculant DF/avgdl sur l'ensemble (meilleure stabilité).
    """
    tokens = tokenize(query)
    if not tokens or not page_texts:
        return []

    # Paramètres BM25
    k1 = 1.5
    b = 0.75

    # Pré-calcul : fréquences par document (TF) et fréquences documentaires (DF)
    doc_tfs: list[dict[str, int]] = []
    doc_lens: list[int] = []
    df: dict[str, int] = {}

    for txt in page_texts:
        toks = tokenize(txt)
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
        """Calcule un IDF BM25 lissé pour un terme."""
        n_q = df.get(term, 0)
        # Lissage “style BM25+” pour éviter des valeurs négatives sur des termes très fréquents
        return math.log(1.0 + (n_docs - n_q + 0.5) / (n_q + 0.5))

    query_terms = list(dict.fromkeys(tokens))

    scores: list[tuple[float, int, str]] = []
    for i, (txt, tf, dl) in enumerate(zip(page_texts, doc_tfs, doc_lens)):
        if candidate_indices is not None and i not in candidate_indices:
            continue
        score = 0.0
        for term in query_terms:
            f = tf.get(term, 0)
            if f <= 0:
                continue
            denom = f + k1 * (1.0 - b + b * (dl / avgdl))
            score += idf(term) * (f * (k1 + 1.0)) / denom
        if score > 0:
            scores.append((score, i, txt))

    scores.sort(reverse=True, key=lambda x: x[0])
    return scores


def find_relevant_snippets(
    query: str,
    page_texts: List[str],
    k: int = 6,
    method: Literal["count", "count_score", "bm25", "bm25_souple"] = "count",
    candidates_k: int = 24,
    min_relative_score: float | None = None,
    min_coverage: float | None = None,
    min_density_rel: float | None = None,
    min_terms_found: int | None = None,
    must_have_terms: Sequence[str] | None = None,
) -> List[str]:
    """
    Retourne les k meilleurs extraits/pages les plus pertinents pour `query`.

    Arguments :
        query: Chaîne de recherche (souvent des mots-clés).
        page_texts: Liste des textes de pages (un élément par page).
        k: Nombre d'extraits à retourner.
        method:
          - "count": matching lexical en comptant les occurrences (comportement historique).
          - "count_score": 2-étapes : candidates via `count` puis filtrage par score relatif + coverage
            (+ densité relative) pour éliminer des pages peu informatives, sans BM25.
          - "bm25": score BM25 simple (sans dépendances supplémentaires).
          - "bm25_souple": 2-étapes : candidates via `count` puis scoring BM25 avec tokenisation
            plus tolérante (Unicode/OCR/tirets). Utile pour filtrer les pages peu informatives.

    Notes :
        - Les deux méthodes sont purement lexicales et fonctionnent mieux avec un texte “propre”.
        - Les pages avec un score <= 0 sont exclues.
    """
    if method == "count":
        scores = _count_scores(query, page_texts)
        return [s[2] for s in scores[:k]]

    if method == "count_score":
        # Étape 1 : candidates via `count` (robuste car substring)
        count_scores = _count_scores(query, page_texts)
        if not count_scores:
            return []

        n = max(k, int(candidates_k))
        candidates = count_scores[:n]

        # Seuils par défaut : plutôt restrictifs (objectif = filtrer les pages peu informatives).
        rel_thr = 0.40 if min_relative_score is None else float(min_relative_score)
        cov_thr = 0.50 if min_coverage is None else float(min_coverage)
        dens_thr = 0.35 if min_density_rel is None else float(min_density_rel)

        q_terms = list(dict.fromkeys(_tokenize(query)))
        q_set = set(q_terms)

        # Seuil absolu : impose un minimum de mots-clés distincts trouvés.
        # (sinon `coverage` peut être trompeur quand la requête a peu de tokens)
        if min_terms_found is None:
            min_terms_abs = 2 if len(q_set) < 6 else 3
        else:
            min_terms_abs = max(1, int(min_terms_found))

        max_score = max(s for s, _, _ in candidates) if candidates else 0

        # Pré-calc densité max sur candidates (normalisation relative, robuste aux docs variés)
        densities: dict[int, float] = {}
        max_density = 0.0
        for score, i, txt in candidates:
            # approx longueur en tokens (même tokeniseur strict que query)
            dl = len(_tokenize(txt)) or 1
            d = float(score) / float(dl)
            densities[i] = d
            if d > max_density:
                max_density = d

        kept: list[tuple[int, int, str]] = []
        rejected: list[tuple[int, int, str]] = []
        for score, i, txt in candidates:
            rel = (float(score) / float(max_score)) if max_score > 0 else 0.0
            t = (txt or "").lower()
            terms_found = sum(1 for term in q_set if term and term in t) if q_set else 0
            cov = (terms_found / len(q_set)) if q_set else 0.0
            dens_rel = (densities.get(i, 0.0) / max_density) if max_density > 0 else 0.0

            if (
                rel >= rel_thr
                and cov >= cov_thr
                and dens_rel >= dens_thr
                and terms_found >= min_terms_abs
            ):
                kept.append((score, i, txt))
            else:
                rejected.append((score, i, txt))

        # Étape "rescue" : si des pages contiennent des must-have, on peut les ré-intégrer
        # même si elles échouent les seuils, afin d'éviter des faux négatifs.
        #must_terms = [
        #    str(t).strip().lower()
        #    for t in (must_have_terms or [])
        #    if str(t).strip() != ""
        #]

        #if must_terms and len(kept) < k:
        #    rescued: list[tuple[int, int, str]] = []
        #    kept_indices = {i for _, i, _ in kept}
        #    for score, i, txt in rejected:
        #        if i in kept_indices:
        #            continue
        #        page_lower = (txt or "").lower()
        #        if any(mt in page_lower for mt in must_terms):
        #            rescued.append((score, i, txt))

            # On conserve l'ordre `count` (score décroissant) sur les rescued.
        #    rescued.sort(reverse=True, key=lambda x: x[0])
        #    kept.extend(rescued)

        # Si rien ne passe les seuils (et rien n'est rescué), on renvoie une liste vide : NA.
        #if not kept:
        #    return []

        # On conserve l'ordre du `count` (score décroissant) et on tronque à k.
        kept.sort(reverse=True, key=lambda x: x[0])
        return [s[2] for s in kept[:k]]

    if method == "bm25":
        scores = _bm25_scores(query=query, page_texts=page_texts, tokenize=_tokenize)
        return [s[2] for s in scores[:k]]

    if method == "bm25_souple":
        # Étape 1 : candidates via `count` (robuste car substring)
        count_scores = _count_scores(query, page_texts)
        if not count_scores:
            return []

        # Top-N candidates (au moins k, borné)
        n = max(k, int(candidates_k))
        candidate_indices = {i for _, i, _ in count_scores[:n]}

        # Étape 2 : scoring BM25 "souple" sur les candidates (DF/avgdl calculés sur tout le doc)
        bm25_scores = _bm25_scores(
            query=query,
            page_texts=page_texts,
            tokenize=_tokenize_soft,
            candidate_indices=candidate_indices,
        )
        if not bm25_scores:
            return []

        # Optionnel : filtre par score relatif et couverture (utile pour éliminer des pages peu informatives)
        # Par défaut, on active des seuils modestes pour rendre la méthode exploitable sans configuration.
        rel_thr = 0.75 if min_relative_score is None else float(min_relative_score)
        cov_thr = 0.50 if min_coverage is None else float(min_coverage)

        # Couverture: fraction des termes de requête trouvés dans la page.
        q_terms = list(dict.fromkeys(_tokenize_soft(query)))
        q_set = set(q_terms)

        # Pour calculer coverage, on retokenise les candidates seulement.
        coverage_by_idx: dict[int, float] = {}
        for idx in candidate_indices:
            page_terms = set(_tokenize_soft(page_texts[idx]))
            if not q_set:
                coverage_by_idx[idx] = 0.0
            else:
                coverage_by_idx[idx] = len(q_set.intersection(page_terms)) / len(q_set)

        max_score = bm25_scores[0][0]
        filtered: list[tuple[float, int, str]] = []
        for score, i, txt in bm25_scores:
            rel = (score / max_score) if max_score > 0 else 0.0
            cov = coverage_by_idx.get(i, 0.0)
            if rel >= rel_thr and cov >= cov_thr:
                filtered.append((score, i, txt))

        # Si rien ne passe les seuils, on renvoie une liste vide : cela permet au pipeline
        # de décider explicitement "NA" plutôt que de tomber sur un fallback.
        if not filtered:
            return []

        return [s[2] for s in filtered[:k]]

    raise ValueError(
        f"Méthode inconnue : {method!r}. Attendu : 'count', 'count_score', 'bm25' ou 'bm25_souple'."
    )
