# Architecture du projet VSME Extractor

Ce document décrit l’arborescence **côté code** et le rôle des principaux fichiers/modules.

---

## 1) Arborescence (code)

```text
.
├── .env
├── .env.example
├── MANIFEST.in
├── main.py
├── README.md
├── ARCHITECTURE.md
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── requirements-streamlit.txt
├── pytest.ini
├── tests/
├── streamlit_app/
├── exemples/
│   ├── example_extract_pdf.py
│   ├── example_cli_extract_pdf_json.py
│   ├── example_cli_extract_pdf_xlsx.py
│   └── example_usage.ipynb
└── vsme_extractor/
    ├── __init__.py
    ├── cli.py
    ├── config.py
    ├── error_reporting.py
    ├── extraction.py
    ├── indicators.py
    ├── llm_client.py
    ├── logging_utils.py
    ├── pdf_loader.py
    ├── pipeline.py
    ├── prompts.py
    ├── retrieval.py
    └── stats.py
    └── data/
        └── table_codes_portail_rse.csv
```

---

## 2) Points d’entrée

### 2.1 CLI
- Point d’entrée : [`main()`](vsme_extractor/cli.py:288) dans [`vsme_extractor/cli.py`](vsme_extractor/cli.py:1)
- Installée via le script console défini dans [`pyproject.toml`](pyproject.toml:1) (commande `vsme-extract`)
- Usages principaux :
  - **Extraction** : `vsme-extract <fichier.pdf|dossier/>`
  - **Stats** : `vsme-extract --count <dossier_resultats>`
  - **Listing** : `vsme-extract --list-current-indicators` / `vsme-extract --list-all-indicators`
  - **Formats de sortie** : `--output-format json|xlsx` (défaut : `json`) (voir [`build_parser()`](vsme_extractor/cli.py:108)).

### 2.2 API Python (package)
- API publique exposée dans [`vsme_extractor/__init__.py`](vsme_extractor/__init__.py:1)
- Principales classes exportées :
  - [`VSMExtractor`](vsme_extractor/pipeline.py:52)
  - [`ExtractionStats`](vsme_extractor/pipeline.py:21)

---

## 3) Flux principal d’extraction (vue fonctionnelle)

1. La CLI appelle [`VSMExtractor.extract_from_pdf()`](vsme_extractor/pipeline.py:72)
2. Le PDF est chargé via [`load_pdf()`](vsme_extractor/pdf_loader.py:8) → `page_texts` + `full_text`
3. La liste d’indicateurs est chargée via [`get_indicators()`](vsme_extractor/indicators.py:20)
4. La langue du document est détectée via [`detect_document_language()`](vsme_extractor/pipeline.py:35)
   - `langdetect` est initialisé avec un seed global pour être déterministe.
5. Pour chaque indicateur :
   - (optionnel) traduction des mots-clés via LLM avec **cache** interne (évite des appels répétés)
   - sélection d’extraits pertinents via [`find_relevant_snippets_with_details()`](vsme_extractor/pipeline.py:287) avec `method="count"` (défaut) ou `method="count_refine"`
   - si aucun extrait n’est jugé pertinent, l’extracteur renvoie **NA** et évite un appel LLM (pas de fallback sur le début du document) (voir [`VSMExtractor.extract_from_pdf()`](vsme_extractor/pipeline.py:331))
   - extraction LLM via [`extract_value_for_metric()`](vsme_extractor/extraction.py:9), avec parsing robuste et tentative de “repair” JSON (optionnelle)
6. Les résultats sont agrégés dans un `DataFrame`, puis exportés selon le format CLI :
   - **JSON** : `<pdf>.vsme.json` (défaut)
   - **Excel** : `<pdf>.vsme.xlsx` (si `--output-format xlsx`)
   Les coûts/tokens sont renvoyés via [`ExtractionStats`](vsme_extractor/pipeline.py:31).

### 3.1 Sortie JSON (CLI)
Quand `--output-format json` (défaut), la CLI écrit un JSON avec :
- `pdf`: chemin du PDF
- `results`: liste des indicateurs (une ligne par indicateur)
- `stats`: tokens + coût
- `status` (optionnel) : bloc de statut (activé par défaut, désactivable via `--json-no-status`) (voir [`build_parser()`](vsme_extractor/cli.py:108)).

Par défaut la CLI **retire** les champs de debug retrieval (pour limiter la taille du JSON) via [`_strip_retrieval_details()`](vsme_extractor/cli.py:97). Ils peuvent être inclus via `--json-include-retrieval-details`.

### 3.2 Enrichissement RSE (CLI)
Si le fichier packagé [`vsme_extractor/data/table_codes_portail_rse.csv`](vsme_extractor/data/table_codes_portail_rse.csv:1) est présent, la CLI enrichit chaque ligne d’indicateur JSON avec :
- `matched_rse_code`
- `matched_rse_champs_id`
- `matched_rse_colonne_id`

Chargement + jointure : [`_load_rse_mapping()`](vsme_extractor/cli.py:27) et [`_enrich_results_with_rse()`](vsme_extractor/cli.py:71).

### 3.3 Rapports d’erreur (CLI)
En cas d’échec d’initialisation ou d’extraction, la CLI écrit un rapport JSON d’erreur (à côté du PDF, ou dans le dossier target) via [`build_error_report()`](vsme_extractor/cli.py:20) / [`write_error_report()`](vsme_extractor/cli.py:20), implémentés dans [`vsme_extractor/error_reporting.py`](vsme_extractor/error_reporting.py:1).

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

- [`vsme_extractor/error_reporting.py`](vsme_extractor/error_reporting.py:1)
  Génération/écriture d’un rapport d’erreur JSON côté CLI (évite d’inonder la console avec un traceback par défaut).

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
  Retrieval “lexical” :
  - `count` (comptage d’occurrences de tokens)
  - `count_refine` (candidats via `count` puis ranking TF‑IDF n‑grams)
   via [`find_relevant_snippets()`](vsme_extractor/retrieval.py:240).

- [`vsme_extractor/indicators.py`](vsme_extractor/indicators.py:1)  
  Chargement de la liste des indicateurs via [`get_indicators()`](vsme_extractor/indicators.py:20) :
  - possibilité de surcharger le CSV via variable d’environnement
  - fallback sur une ressource packagée du projet
  - détection d’encodage (chardet) avant lecture
  - filtrage optionnel par `code_vsme` via `VSME_CODE_VSME_LIST` (sinon fallback sur `defaut == 1`)

- [`vsme_extractor/pipeline.py`](vsme_extractor/pipeline.py:1)  
  Orchestrateur principal :
  - init config + client LLM dans [`VSMExtractor.__init__()`](vsme_extractor/pipeline.py:53)
  - pipeline complet dans [`VSMExtractor.extract_from_pdf()`](vsme_extractor/pipeline.py:65)
  - calcul des coûts/tokens, et construction des résultats finaux

- [`vsme_extractor/stats.py`](vsme_extractor/stats.py:1)  
  Post-traitement “complétude” :
  - lit les `.vsme.xlsx` d’un répertoire
  - calcule les occurrences renseignées via [`count_filled_indicators()`](vsme_extractor/stats.py:25)

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
