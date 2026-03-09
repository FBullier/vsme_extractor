from __future__ import annotations

from vsme_extractor.retrieval import find_relevant_snippets_with_details


def test_tfidf_ngram_souple_has_thresholds_in_details() -> None:
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
    # `rel_thr` default can be overridden by env (VSME_RETRIEVAL_REL_THR).
    # Keep the test environment-agnostic.
    rel_thr = thresholds.get("rel_thr")
    assert isinstance(rel_thr, float)
    assert 0.0 <= rel_thr <= 1.0
    # `abs_thr` default can be overridden by env (VSME_RETRIEVAL_ABS_THR).
    abs_thr = thresholds.get("abs_thr")
    assert isinstance(abs_thr, float)
    assert abs_thr >= 0.0
