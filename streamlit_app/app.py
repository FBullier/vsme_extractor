from __future__ import annotations

import os
import re
import tempfile
from io import BytesIO

import pandas as pd
import streamlit as st
from dotenv import find_dotenv, load_dotenv

from vsme_extractor import VSMExtractor
from vsme_extractor.indicators import get_indicators
from vsme_extractor.logging_utils import configure_logging_from_env


@st.cache_resource
def _init_logging() -> bool:
    """Initialise le logging *une seule fois* côté Streamlit.

    Streamlit ré-exécute le script à chaque interaction utilisateur.
    Si on configure les handlers à chaque run, on cumule des handlers et on obtient
    des lignes de logs dupliquées (même message écrit N fois).
    """
    return configure_logging_from_env(reset_handlers=False)


def _code_sort_key(code: str) -> tuple:
    """Clé de tri “naturel” pour des codes comme B1, B10, C8_1."""
    s = (code or "").strip()
    m = re.match(r"^([A-Za-z]+)(\d+)(?:_(\d+))?$", s)
    if not m:
        return ("~", 10**9, 10**9, s)
    letters = m.group(1).upper()
    num = int(m.group(2))
    idx = int(m.group(3)) if m.group(3) is not None else 0
    return (letters, num, idx, s)


@st.cache_data
def load_all_indicators() -> pd.DataFrame:
    """Charge la liste complète des indicateurs (sans filtre `.env`)."""
    rows = get_indicators(apply_env_filter=False)
    df = pd.DataFrame(rows)
    if "code_vsme" in df.columns:
        df = df.sort_values(
            by="code_vsme",
            key=lambda s: s.astype(str).map(_code_sort_key),
            ignore_index=True,
        )
    return df


@st.cache_resource
def get_extractor() -> VSMExtractor:
    """Construit l'extracteur (cache Streamlit pour éviter de ré-instancier à chaque action)."""
    return VSMExtractor()


def main() -> None:
    """Point d'entrée Streamlit (UI upload -> sélection -> extraction -> affichage)."""
    # Charge le .env (utile pour SCW_API_KEY, etc.)
    load_dotenv(find_dotenv(usecwd=True), override=True)

    # Active le logging applicatif si les variables d'env sont définies (opt-in).
    # Important dans Streamlit : on évite de configurer plusieurs fois (sinon doublons).
    _init_logging()

    st.set_page_config(page_title="VSME Extractor", layout="wide")
    st.title("VSME Extractor")

    st.markdown(
        """
Cette application Streamlit permet :
- d'uploader un PDF,
- de choisir les indicateurs (par `code_vsme`),
- de lancer l'extraction,
- de visualiser et télécharger le résultat.
"""
    )

    # --- Upload ---
    uploaded = st.file_uploader("Uploader un PDF", type=["pdf"])  # noqa: RUF100

    # --- Choix indicateurs ---
    df_ind = load_all_indicators()
    if df_ind.empty:
        st.error("Aucun indicateur n'a été chargé.")
        return

    if "code_vsme" not in df_ind.columns:
        st.error("Le CSV d'indicateurs ne contient pas la colonne 'code_vsme'.")
        return

    df_ind["label"] = (
        df_ind["code_vsme"].astype(str) + " — " + df_ind["Métrique"].astype(str)
    )

    default_codes = (
        df_ind[df_ind["defaut"].astype(str).str.strip().eq("1")]["code_vsme"]
        .astype(str)
        .tolist()
        if "defaut" in df_ind.columns
        else []
    )

    col_left, col_right = st.columns([2, 1])

    with col_left:
        selected_labels = st.multiselect(
            "Indicateurs à extraire (laisser vide = indicateurs par défaut `defaut=1`)",
            options=df_ind["label"].tolist(),
            default=[
                lbl
                for lbl in df_ind["label"].tolist()
                if lbl.split(" — ", 1)[0] in set(default_codes)
            ],
        )

    with col_right:
        st.caption("Paramètres")
        retrieval_method = st.selectbox(
            "Retrieval",
            options=["count", "count_score", "bm25", "bm25_souple"],
            index=0,
            help=(
                "count = matching substring (robuste OCR). "
                "count_score = count + filtrage (score relatif + coverage + densité). "
                "bm25 = lexical strict. "
                "bm25_souple = candidates via count puis BM25 avec tokenisation plus tolérante + seuils."
            ),
        )
        top_k = st.number_input(
            "Top-k extraits", min_value=1, max_value=12, value=6, step=1
        )

    selected_codes = [lbl.split(" — ", 1)[0].strip() for lbl in selected_labels]

    # --- Action ---
    disabled = uploaded is None
    run = st.button("Lancer l'extraction", type="primary", disabled=disabled)

    if run:
        if uploaded is None:
            st.warning("Merci d'uploader un PDF.")
            return

        if not os.getenv("SCW_API_KEY"):
            st.error(
                "SCW_API_KEY manquant. Définis-le dans ton environnement ou ton .env."
            )
            return

        # Applique la sélection via env var (compatible avec le pipeline existant).
        if selected_codes:
            os.environ["VSME_CODE_VSME_LIST"] = ",".join(selected_codes)
        else:
            # vide => fallback sur `defaut == 1`
            os.environ["VSME_CODE_VSME_LIST"] = ""

        # Sauvegarde du PDF en fichier temporaire.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name

        st.info(f"PDF chargé : {uploaded.name}")
        st.info(
            f"Indicateurs sélectionnés : {len(selected_codes) if selected_codes else 'défaut (defaut=1)'}"
        )

        with st.spinner("Extraction en cours…"):
            extractor = get_extractor()
            # Ajuste dynamiquement quelques paramètres de l'instance.
            extractor.retrieval_method = retrieval_method
            extractor.top_k_snippets = int(top_k)

            df, stats = extractor.extract_from_pdf(tmp_path)

        st.success("Extraction terminée")

        c1, c2, c3 = st.columns(3)
        c1.metric("Tokens input", int(stats.total_input_tokens))
        c2.metric("Tokens output", int(stats.total_output_tokens))
        c3.metric("Coût total (€)", f"{stats.total_cost_eur:.4f}")

        st.subheader("Résultats")
        # Streamlit deprecates `use_container_width` in favor of `width`.
        # Keep backward compatibility depending on installed Streamlit version.
        try:
            st.dataframe(df, width="stretch")
        except TypeError:
            st.dataframe(df, use_container_width=True)

        # Download Excel
        buf = BytesIO()
        df.to_excel(buf, index=False)
        st.download_button(
            "Télécharger (.xlsx)",
            data=buf.getvalue(),
            file_name=f"{os.path.splitext(uploaded.name)[0]}.vsme.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


if __name__ == "__main__":
    main()
