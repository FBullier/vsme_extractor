# Architecture du projet VSME Extractor

Ce document décrit l’arborescence **côté code** et le rôle des principaux fichiers/modules.

---

## 1) Arborescence (code)

```text
.
├── .env
├── main.py
├── README.md
├── ARCHITECTURE.md
├── pyproject.toml
├── requirements.txt
├── exemples/
│   ├── example_extract_pdf.py
│   └── example_usage.ipynb
└── vsme_extractor/
    ├── __init__.py
    ├── cli.py
    ├── config.py
    ├── extraction.py
    ├── indicators.py
    ├── llm_client.py
    ├── logging_utils.py
    ├── pdf_loader.py
    ├── pipeline.py
    ├── prompts.py
    ├── retrieval.py
    └── stats.py
```

---

## 2) Points d’entrée

### 2.1 CLI
- Point d’entrée : [`main()`](vsme_extractor/cli.py:76) dans [`vsme_extractor/cli.py`](vsme_extractor/cli.py:1)
- Installée via le script console défini dans [`pyproject.toml`](pyproject.toml:1) (commande `vsme-extract`)
- Deux usages :
  - **Extraction** : `vsme-extract <fichier.pdf|dossier/>`
  - **Stats** : `vsme-extract --count <dossier_resultats>`

### 2.2 API Python (package)
- API publique exposée dans [`vsme_extractor/__init__.py`](vsme_extractor/__init__.py:1)
- Principales classes exportées :
  - [`VSMExtractor`](vsme_extractor/pipeline.py:52)
  - [`ExtractionStats`](vsme_extractor/pipeline.py:21)

---

## 3) Flux principal d’extraction (vue fonctionnelle)

1. La CLI appelle [`VSMExtractor.extract_from_pdf()`](vsme_extractor/pipeline.py:72)
2. Le PDF est chargé via [`load_pdf()`](vsme_extractor/pdf_loader.py:8) → `page_texts` + `full_text`
3. La liste d’indicateurs est chargée via [`get_indicators()`](vsme_extractor/indicators.py:17)
4. La langue du document est détectée via [`detect_document_language()`](vsme_extractor/pipeline.py:35)
   - `langdetect` est initialisé avec un seed global pour être déterministe.
5. Pour chaque indicateur :
   - (optionnel) traduction des mots-clés via LLM avec **cache** interne (évite des appels répétés)
   - sélection d’extraits pertinents via [`find_relevant_snippets()`](vsme_extractor/retrieval.py:10) avec `method="count"` (défaut) ou `method="bm25"`
   - extraction LLM via [`extract_value_for_metric()`](vsme_extractor/extraction.py:9), avec parsing robuste et tentative de “repair” JSON (optionnelle)
6. Les résultats sont agrégés dans un `DataFrame`, exportés en `.vsme.xlsx`, et les coûts/tokens sont renvoyés via [`ExtractionStats`](vsme_extractor/pipeline.py:28)

---

## 4) Rôle des fichiers (responsabilités)

### Fichiers racine
- [`.env`](.env:1)
  Configuration runtime (variables d’environnement). Chargé via `python-dotenv` dans la CLI et les exemples.

- [`main.py`](main.py:1)
  Exemple minimal d’utilisation de la librairie `vsme_extractor` (ce n’est plus la CLI).

- [`pyproject.toml`](pyproject.toml:1)  
  Métadonnées, dépendances, et point d’entrée script via la section `project.scripts` (voir [`pyproject.toml`](pyproject.toml:22)).

- [`requirements.txt`](requirements.txt:1)  
  Alternative pour installation via pip.

- [`README.md`](README.md:1)  
  Documentation utilisateur (installation/configuration/usage).

- [`ARCHITECTURE.md`](ARCHITECTURE.md:1)  
  Documentation d’architecture (ce fichier).

### Package [`vsme_extractor/`](vsme_extractor/__init__.py:1)
- [`vsme_extractor/cli.py`](vsme_extractor/cli.py:1)
  CLI installable (commande `vsme-extract`) : parsing des args, configuration logging, appels à l’API du package.

- [`vsme_extractor/config.py`](vsme_extractor/config.py:1)
  Configuration LLM + paramètres de coût (variables d’environnement) via [`load_llm_config()`](vsme_extractor/config.py:18).

- [`vsme_extractor/llm_client.py`](vsme_extractor/llm_client.py:1)  
  Client LLM (OpenAI-compatible) :
  - appel non-stream via [`LLM.invoke()`](vsme_extractor/llm_client.py:78)
  - streaming via [`LLM.invoke_stream()`](vsme_extractor/llm_client.py:150)
  - normalisation/estimation des tokens et coût estimé

- [`vsme_extractor/prompts.py`](vsme_extractor/prompts.py:1)  
  Prompts :
  - consignes globales via [`EXTRACTION_SYSTEM_PROMPT`](vsme_extractor/prompts.py:3)
  - génération du prompt utilisateur via [`build_user_prompt()`](vsme_extractor/prompts.py:14)

- [`vsme_extractor/extraction.py`](vsme_extractor/extraction.py:1)  
  Extraction d’une métrique :
  - construit le prompt
  - appelle le LLM
  - parse une réponse JSON (avec fallback minimal) via [`extract_value_for_metric()`](vsme_extractor/extraction.py:9)

- [`vsme_extractor/pdf_loader.py`](vsme_extractor/pdf_loader.py:1)  
  Chargement PDF via [`load_pdf()`](vsme_extractor/pdf_loader.py:8).

- [`vsme_extractor/retrieval.py`](vsme_extractor/retrieval.py:1)  
  Retrieval “lexical” simple (comptage d’occurrences de tokens) via [`find_relevant_snippets()`](vsme_extractor/retrieval.py:5).

- [`vsme_extractor/indicators.py`](vsme_extractor/indicators.py:1)  
  Chargement de la liste des indicateurs via [`get_indicators()`](vsme_extractor/indicators.py:17) :
  - possibilité de surcharger le CSV via variable d’environnement
  - fallback sur une ressource packagée du projet
  - détection d’encodage (chardet) avant lecture

- [`vsme_extractor/pipeline.py`](vsme_extractor/pipeline.py:1)  
  Orchestrateur principal :
  - init config + client LLM dans [`VSMExtractor.__init__()`](vsme_extractor/pipeline.py:53)
  - pipeline complet dans [`VSMExtractor.extract_from_pdf()`](vsme_extractor/pipeline.py:65)
  - calcul des coûts/tokens, et construction des résultats finaux

- [`vsme_extractor/stats.py`](vsme_extractor/stats.py:1)  
  Post-traitement “complétude” :
  - lit les `.vsme.xlsx` d’un répertoire
  - calcule les occurrences renseignées via [`count_filled_indicators()`](vsme_extractor/stats.py:21)

- [`vsme_extractor/logging_utils.py`](vsme_extractor/logging_utils.py:1)
  Configuration logging centralisée via [`configure_logging()`](vsme_extractor/logging_utils.py:13) et activation “opt-in” via [`configure_logging_from_env()`](vsme_extractor/logging_utils.py:88).
  Points importants :
  - `reset_handlers` (par défaut `True`) permet de contrôler si on supprime les handlers existants du root logger.
  - Variables d’environnement supportées :
    - `SME_LOG_LEVEL`
    - `VSME_LOG_FILE`
    - `VSME_LOG_STDOUT`

---

## 5) Observabilité (logs)

- La CLI configure les handlers logging au démarrage via [`configure_logging()`](vsme_extractor/logging_utils.py:13) (comportement CLI : `reset_handlers=True`).
- Dans un notebook/script, l’activation est “opt-in” via [`configure_logging_from_env()`](vsme_extractor/logging_utils.py:88) et vous pouvez choisir `reset_handlers=False` si votre environnement a déjà un logging configuré.
- Chaque module utilise un logger par module (`logging.getLogger(__name__)`).
- Le streaming LLM écrit directement sur stdout dans [`LLM.invoke_stream()`](vsme_extractor/llm_client.py:150) (token par token), indépendamment des logs structurés.