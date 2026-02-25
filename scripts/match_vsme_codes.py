"""Match legacy `code_vsme` (CSV) with new VSME form codes (JSON).

Inputs
------
- JSON directory: `data/new_vsme_codes/*.json`
  Each JSON file is a dict where keys are codes (e.g. "B1-24-a") and values
  contain a "description" (and often a "titre").

- CSV file: `vsme_extractor/data/indicateurs_vsme.csv`
  Must contain columns `code_vsme` and `Métrique`.

Output
------
- `correspondance_vsme_codes.csv`

The matching is done by text similarity between `Métrique` (CSV) and
`titre + description` (JSON), with a lightweight TF‑IDF cosine similarity over
word n‑grams.

Usage
-----
  .venv/bin/python scripts/match_vsme_codes.py \
    --json-dir data/new_vsme_codes \
    --csv vsme_extractor/data/indicateurs_vsme.csv \
    --output correspondance_vsme_codes.csv
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import chardet
import pandas as pd


@dataclass(frozen=True)
class NewCodeDoc:
    code: str
    group: str
    titre: str
    description: str
    champs: tuple[tuple[str, str], ...]  # (id, label)
    colonnes: tuple[tuple[str, str], ...]  # (id, label)
    source_json: str
    tf: Counter[str]
    norm: float


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


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
    t = text or ""
    t = unicodedata.normalize("NFKC", t).translate(_SUBSCRIPT_DIGITS)
    # Remove end-of-line hyphenation
    t = re.sub(r"([A-Za-z])\-\s*\n\s*([A-Za-z])", r"\1\2", t)
    # Harmonize dashes
    t = t.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    t = re.sub(r"\s+", " ", t)
    t = _strip_accents(t)
    return t.strip().lower()


def _tokenize_soft(text: str) -> list[str]:
    base = _normalize_text_soft(text)
    # hyphen as space
    v1 = base.replace("-", " ")
    # Keep alpha tokens (len>=2) and numeric tokens (any length) to preserve signals like "scope 2".
    toks1 = re.findall(r"(?:[a-z]{2,}|[0-9]+)", v1)
    # hyphen removed (scope-1 vs scope1)
    v2 = base.replace("-", "")
    toks2 = re.findall(r"(?:[a-z]{2,}|[0-9]+)", v2)

    seen: set[str] = set()
    out: list[str] = []
    for tok in [*toks1, *toks2]:
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out


def _word_ngrams(tokens: list[str], ngram_range: tuple[int, int] = (1, 3)) -> list[str]:
    if not tokens:
        return []
    n_min, n_max = ngram_range
    n_min = max(1, int(n_min))
    n_max = max(n_min, int(n_max))

    out: list[str] = []
    n_toks = len(tokens)
    for n in range(n_min, n_max + 1):
        if n > n_toks:
            break
        for i in range(0, n_toks - n + 1):
            out.append(" ".join(tokens[i : i + n]))
    return out


def _detect_encoding(path: Path) -> str | None:
    raw = path.read_bytes()
    info = chardet.detect(raw)
    enc = info.get("encoding")
    return str(enc) if enc else None


def _iter_json_files(directory: Path) -> Iterable[Path]:
    yield from sorted(directory.glob("*.json"))


def _code_group_from_vsme(code_vsme: str) -> str:
    # Example: B1_3 -> B1 ; C9_1 -> C9
    c = (code_vsme or "").strip()
    if "_" in c:
        return c.split("_", 1)[0]
    return c


def _code_group_from_new(code: str) -> str:
    # Example: B1-24-e-ii -> B1 ; C6-33-a -> C6
    c = (code or "").strip()
    if "-" in c:
        return c.split("-", 1)[0]
    return c


def load_new_codes(json_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for p in _iter_json_files(json_dir):
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue
        for code, obj in data.items():
            if not isinstance(obj, dict):
                continue
            desc = str(obj.get("description") or "").strip()
            titre = str(obj.get("titre") or "").strip()

            champs = obj.get("champs")
            champs_list = champs if isinstance(champs, list) else []

            champs_pairs: list[tuple[str, str]] = []
            colonnes_pairs: list[tuple[str, str]] = []

            for ch in champs_list:
                if not isinstance(ch, dict):
                    continue
                cid = str(ch.get("id") or "").strip()
                clab = str(ch.get("label") or "").strip()
                if cid or clab:
                    champs_pairs.append((cid, clab))

                cols = ch.get("colonnes")
                cols_list = cols if isinstance(cols, list) else []
                for col in cols_list:
                    if not isinstance(col, dict):
                        continue
                    col_id = str(col.get("id") or "").strip()
                    col_lab = str(col.get("label") or "").strip()
                    if col_id or col_lab:
                        colonnes_pairs.append((col_id, col_lab))

            if not str(code).strip():
                continue
            records.append(
                {
                    "code": str(code).strip(),
                    "group": _code_group_from_new(str(code)),
                    "titre": titre,
                    "description": desc,
                    "champs": champs_pairs,
                    "colonnes": colonnes_pairs,
                    "source_json": str(p.as_posix()),
                }
            )
    return records


def _build_docs(
    new_code_records: Sequence[Mapping[str, Any]],
) -> dict[str, list[NewCodeDoc]]:
    # Group by B1/B2/.../C1/etc. to improve precision.
    by_group: dict[str, list[Mapping[str, Any]]] = {}
    for r in new_code_records:
        group = str(r.get("group") or "")
        by_group.setdefault(group, []).append(r)

    out: dict[str, list[NewCodeDoc]] = {}
    for group, recs in by_group.items():
        # Build doc TF and DF over n-grams for this group.
        doc_tfs: list[Counter[str]] = []
        for r in recs:
            # Matching target per user request:
            #   Métrique (CSV) ~ titre + labels in champs + labels in colonnes
            titre = str(r.get("titre") or "")
            champs_pairs = r.get("champs")
            colonnes_pairs = r.get("colonnes")
            champs_pairs_list = champs_pairs if isinstance(champs_pairs, list) else []
            colonnes_pairs_list = (
                colonnes_pairs if isinstance(colonnes_pairs, list) else []
            )

            champs_labels_list = [
                str(lab) for _id, lab in champs_pairs_list if str(lab).strip()
            ]
            colonnes_labels_list = [
                str(lab) for _id, lab in colonnes_pairs_list if str(lab).strip()
            ]

            labels = " ".join([*champs_labels_list, *colonnes_labels_list])
            text = f"{titre} {labels}".strip()
            grams = _word_ngrams(_tokenize_soft(text), ngram_range=(1, 3))
            doc_tfs.append(Counter(grams))

        df: Counter[str] = Counter()
        for tf in doc_tfs:
            df.update(tf.keys())
        n_docs = len(doc_tfs)

        def idf(term: str) -> float:
            return math.log((n_docs + 1.0) / (float(df.get(term, 0)) + 1.0)) + 1.0

        docs: list[NewCodeDoc] = []
        for r, tf in zip(recs, doc_tfs):
            norm2 = 0.0
            for term, c in tf.items():
                if c <= 0:
                    continue
                w = (1.0 + math.log(float(c))) * idf(term)
                norm2 += w * w
            norm = math.sqrt(norm2) if norm2 > 0 else 0.0

            champs_pairs = r.get("champs")
            colonnes_pairs = r.get("colonnes")
            champs_pairs_list = champs_pairs if isinstance(champs_pairs, list) else []
            colonnes_pairs_list = (
                colonnes_pairs if isinstance(colonnes_pairs, list) else []
            )

            docs.append(
                NewCodeDoc(
                    code=str(r.get("code") or ""),
                    group=str(r.get("group") or ""),
                    titre=str(r.get("titre") or ""),
                    description=str(r.get("description") or ""),
                    champs=tuple(
                        (str(i).strip(), str(label).strip())
                        for i, label in champs_pairs_list
                        if str(i).strip() or str(label).strip()
                    ),
                    colonnes=tuple(
                        (str(i).strip(), str(label).strip())
                        for i, label in colonnes_pairs_list
                        if str(i).strip() or str(label).strip()
                    ),
                    source_json=str(r.get("source_json") or ""),
                    tf=tf,
                    norm=norm,
                )
            )
        out[group] = docs

    return out


def _score_query_against_docs(
    query: str, docs: list[NewCodeDoc]
) -> list[tuple[float, NewCodeDoc]]:
    if not query or not docs:
        return []

    # DF/IDF computed from docs TFs (group-local).
    df: Counter[str] = Counter()
    for d in docs:
        df.update(d.tf.keys())
    n_docs = len(docs)

    def idf(term: str) -> float:
        return math.log((n_docs + 1.0) / (float(df.get(term, 0)) + 1.0)) + 1.0

    q_grams = _word_ngrams(_tokenize_soft(query), ngram_range=(1, 3))
    q_tf = Counter(q_grams)
    if not q_tf:
        return []

    q_w: dict[str, float] = {}
    q_norm2 = 0.0
    for term, tf in q_tf.items():
        if tf <= 0:
            continue
        w = (1.0 + math.log(float(tf))) * idf(term)
        q_w[term] = w
        q_norm2 += w * w
    q_norm = math.sqrt(q_norm2) if q_norm2 > 0 else 0.0
    if q_norm <= 0:
        return []

    scored: list[tuple[float, NewCodeDoc]] = []
    for d in docs:
        if d.norm <= 0:
            continue
        dot = 0.0
        for term, qwt in q_w.items():
            c = d.tf.get(term, 0)
            if c <= 0:
                continue
            dwt = (1.0 + math.log(float(c))) * idf(term)
            dot += qwt * dwt
        if dot <= 0:
            continue
        scored.append((dot / (q_norm * d.norm), d))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="match-vsme-codes",
        description="Match `code_vsme` (CSV) with new VSME codes (JSON) via text similarity.",
    )
    p.add_argument(
        "--json-dir",
        default="data/new_vsme_codes",
        help="Directory containing JSON files with new codes (default: data/new_vsme_codes)",
    )
    p.add_argument(
        "--csv",
        default="vsme_extractor/data/indicateurs_vsme.csv",
        help="Input CSV path (default: vsme_extractor/data/indicateurs_vsme.csv)",
    )
    p.add_argument(
        "--output",
        default="correspondance_vsme_codes.csv",
        help="Output CSV path (default: correspondance_vsme_codes.csv)",
    )
    p.add_argument(
        "--output-simple",
        default="correspondance_vsme_codes_simplifiee.csv",
        help=(
            "Optional simplified output CSV with only: code_vsme, new_code, "
            "matched_champs_id, matched_colonne_id (default: correspondance_vsme_codes_simplifiee.csv)"
        ),
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Keep top-k candidate matches per code (default: 3)",
    )
    ns = p.parse_args(argv)

    json_dir = Path(ns.json_dir)
    csv_path = Path(ns.csv)
    out_path = Path(ns.output)
    out_simple_path = Path(ns.output_simple) if ns.output_simple else None

    if not json_dir.exists() or not json_dir.is_dir():
        raise SystemExit(f"JSON directory not found: {json_dir}")
    if not csv_path.exists() or not csv_path.is_file():
        raise SystemExit(f"CSV file not found: {csv_path}")

    new_records = load_new_codes(json_dir)
    if not new_records:
        raise SystemExit(f"No codes found in JSON dir: {json_dir}")

    docs_by_group = _build_docs(new_records)

    enc = _detect_encoding(csv_path) or "utf-8"
    df = pd.read_csv(csv_path, sep=";", encoding=enc, on_bad_lines="skip")

    if "code_vsme" not in df.columns or "Métrique" not in df.columns:
        raise SystemExit(
            "Input CSV must contain columns 'code_vsme' and 'Métrique'. "
            f"Found: {list(df.columns)}"
        )

    out_rows: list[dict[str, object]] = []

    def _cosine_tokens(a: str, b: str) -> float:
        ta = Counter(_tokenize_soft(a))
        tb = Counter(_tokenize_soft(b))
        if not ta or not tb:
            return 0.0
        dot = sum(float(ta[t]) * float(tb.get(t, 0)) for t in ta.keys())
        if dot <= 0:
            return 0.0
        na = math.sqrt(sum(float(v) * float(v) for v in ta.values()))
        nb = math.sqrt(sum(float(v) * float(v) for v in tb.values()))
        if na <= 0 or nb <= 0:
            return 0.0
        return float(dot / (na * nb))

    def _best_pair(
        metric: str, pairs: tuple[tuple[str, str], ...]
    ) -> tuple[str, str, float]:
        best_id = ""
        best_label = ""
        best_score = 0.0
        for pid, plabel in pairs:
            if not plabel:
                continue
            s = _cosine_tokens(metric, plabel)
            if s > best_score:
                best_score = s
                best_id = pid
                best_label = plabel
        return best_id, best_label, float(best_score)

    for _, row in df.iterrows():
        code_vsme = str(row.get("code_vsme") or "").strip()
        metric = str(row.get("Métrique") or "").strip()
        if not code_vsme:
            continue

        group = _code_group_from_vsme(code_vsme)
        candidates = docs_by_group.get(group) or []
        # Rule: the new code MUST share the same prefix as the legacy `code_vsme`
        # (e.g. B1_* -> B1-*, C9_* -> C9-*). If we have no JSON codes for that
        # group, we do not fall back to other groups.
        if not candidates:
            out_rows.append(
                {
                    "code_vsme": code_vsme,
                    "metrique": metric,
                    "group": group,
                    "new_code": "",
                    "new_titre": "",
                    "new_description": "",
                    "description_nouveau_code": "",
                    "new_champs_ids": "",
                    "new_colonnes_ids": "",
                    "matched_champ_id": "",
                    "matched_champ_label": "",
                    "matched_champ_score": 0.0,
                    "matched_colonne_id": "",
                    "matched_colonne_label": "",
                    "matched_colonne_score": 0.0,
                    "score": 0.0,
                    "source_json": "",
                    "top_matches": "",
                }
            )
            continue

        scored = _score_query_against_docs(metric, candidates)
        top_k = max(1, int(ns.top_k))
        best = scored[:top_k]

        if not best:
            out_rows.append(
                {
                    "code_vsme": code_vsme,
                    "metrique": metric,
                    "group": group,
                    "new_code": "",
                    "new_titre": "",
                    "new_description": "",
                    "description_nouveau_code": "",
                    "new_champs_ids": "",
                    "new_colonnes_ids": "",
                    "matched_champ_id": "",
                    "matched_champ_label": "",
                    "matched_champ_score": 0.0,
                    "matched_colonne_id": "",
                    "matched_colonne_label": "",
                    "matched_colonne_score": 0.0,
                    "score": 0.0,
                    "source_json": "",
                    "top_matches": "",
                }
            )
            continue

        best_score, best_doc = best[0]
        champ_id, champ_label, champ_score = _best_pair(metric, best_doc.champs)
        col_id, col_label, col_score = _best_pair(metric, best_doc.colonnes)
        top_matches = " | ".join([f"{d.code} ({s:.3f})" for s, d in best])
        out_rows.append(
            {
                "code_vsme": code_vsme,
                "metrique": metric,
                "group": group,
                "new_code": best_doc.code,
                "new_titre": best_doc.titre,
                "new_description": best_doc.description,
                # Alias explicite (plus facile à consommer côté utilisateur)
                "description_nouveau_code": best_doc.description,
                "new_champs_ids": ";".join([i for i, _ in best_doc.champs if i]),
                "new_colonnes_ids": ";".join([i for i, _ in best_doc.colonnes if i]),
                "matched_champ_id": champ_id,
                "matched_champ_label": champ_label,
                "matched_champ_score": float(champ_score),
                "matched_colonne_id": col_id,
                "matched_colonne_label": col_label,
                "matched_colonne_score": float(col_score),
                "score": float(best_score),
                "source_json": best_doc.source_json,
                "top_matches": top_matches,
            }
        )

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(out_path, index=False, encoding="utf-8")

    if out_simple_path is not None:
        # User-facing simplified export (stable column names)
        simple_cols = [
            "code_vsme",
            "metrique",
            "new_code",
            "matched_champ_id",
            "matched_colonne_id",
        ]
        simple = out_df[[c for c in simple_cols if c in out_df.columns]].copy()
        if "matched_champ_id" in simple.columns:
            simple["matched_champs_id"] = simple["matched_champ_id"]
            simple = simple.drop(columns=["matched_champ_id"])

        # Enforce requested column order
        ordered = [
            "code_vsme",
            "metrique",
            "new_code",
            "matched_champs_id",
            "matched_colonne_id",
        ]
        simple = simple[[c for c in ordered if c in simple.columns]]
        # User wants the header with accent (same as input CSV)
        if "metrique" in simple.columns:
            simple["Métrique"] = simple["metrique"]
            simple = simple.drop(columns=["metrique"])
            simple = simple[
                [
                    "code_vsme",
                    "Métrique",
                    "new_code",
                    "matched_champs_id",
                    "matched_colonne_id",
                ]
            ]
        simple.to_csv(out_simple_path, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
