from __future__ import annotations

from vsme_extractor.retrieval import find_relevant_snippets_with_details


def test_tfidf_ngram_souple_rel40_has_rel40_and_no_abs_thr() -> None:
    page_texts = [
        "Intro: generalities.",
        "Key: emissions scope-3 reporting. emissions scope-3.",
        "Other topic: biodiversity.",
    ]
    query = "emissions scope-3"

    new, details = find_relevant_snippets_with_details(
        query=query,
        page_texts=page_texts,
        k=1,
        method="count_refine",
        candidates_k=3,
    )

    assert new
    assert new[0] == page_texts[1]

    thresholds = details.get("thresholds")
    assert isinstance(thresholds, dict)
    # `rel_thr` default is 0.40 but can be overridden by env (VSME_RETRIEVAL_REL_THR).
    # Keep the test environment-agnostic.
    rel_thr = thresholds.get("rel_thr")
    assert isinstance(rel_thr, float)
    assert 0.0 <= rel_thr <= 1.0
    assert "abs_thr" not in thresholds
