# Projet VSME Extractor

## 1) Objectif

Ce projet extrait automatiquement des indicateurs VSME (ESG) à partir de rapports PDF d’entreprise et produit un fichier Excel `.vsme.xlsx` contenant les valeurs extraites (ou `NA` si non trouvées).

Le cœur de l’extraction se trouve dans le package `vsme_extractor/`.
La CLI installable est définie dans [`vsme_extractor/cli.py`](vsme_extractor/cli.py:1) (commande `vsme-extract`).
Les exemples d’utilisation (CLI, scripts + notebooks) sont disponibles dans [`examples/`](examples/example_cli_extract_pdf_json.py:1).

---

## 2) Fonctionnalités principales

- Extraction d’un PDF vers un fichier Excel `.vsme.xlsx`.
- Option d’export JSON `.vsme.json` (utile pour intégration / API / pipelines).
- Extraction en batch sur un dossier contenant des PDFs.
- Calcul de complétude des indicateurs sur un dossier de résultats (accepte `.vsme.xlsx` **et** `.vsme.json`, génère `stats_completude.xlsx`).
- Estimation des tokens (et du coût estimé en euros) par exécution.
- Logging configurable (stdout et/ou fichier), pour faciliter le debug et l’exploitation.
- Gestion optionnelle du rate limit (HTTP 429) avec attente et retry configurable.
- Plusieurs méthodes de sélection d’extraits (retrieval) avant appel LLM : `count`, `count_refine`.

---

## 3) Prérequis

- Python `>= 3.11` (voir `pyproject.toml`)
- Dépendances Python (voir `pyproject.toml` et/ou `requirements.txt`)
- Accès à une API LLM (Scaleway) via clé API.

---

## 4) Installation

### Option A — installation du package (recommandé)
Installation “packagée” (permet d’utiliser la commande `vsme-extract`) :

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

### Installation avec options (extras)

Le projet expose des dépendances optionnelles via [`pyproject.toml`](pyproject.toml:19) :

- Avec l'app Streamlit :
```bash
pip install ".[streamlit]"
```

- Dépendances de dev (tests/lint) :
```bash
pip install ".[dev]"
```

- Outils qualité (pre-commit, pyright, pip-audit) :
```bash
pip install ".[quality]"
```

> Note : l’extra `quality` installe des outils de contrôle (lint/format/type/audit). Il ne modifie pas l’exécutable `vsme-extract`.

> Remarque : ce repo conserve aussi [`requirements.txt`](requirements.txt:1). Vous pouvez continuer à l’utiliser si vous le souhaitez (duplication acceptée).

### Option B — via requirements.txt (si vous préférez)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Option C — via uv (si vous utilisez uv)
```bash
uv venv
source .venv/bin/activate
uv pip install .
```

---

## 5) Configuration (variables d’environnement)

Le projet lit la configuration via des variables d’environnement et charge automatiquement un fichier [`.env`](.env:1) au lancement de la CLI et des exemples (via `python-dotenv`).

Un exemple complet (avec des valeurs “recommandées”) est fourni dans [`.env.example`](.env.example:1).

Variables principales :

- `SCW_API_KEY` : clé API Scaleway (**obligatoire**, pas de défaut) (voir [`load_llm_config()`](vsme_extractor/config.py:28)).
- `SCW_BASE_URL` : URL base (défaut : `https://api.scaleway.ai/06f1a171-1eef-4d8b-aed5-b78189d17335/v1`) (voir [`load_llm_config()`](vsme_extractor/config.py:28)).
- `SCW_MODEL_NAME` : nom du modèle (défaut : `gpt-oss-120b`) (voir [`load_llm_config()`](vsme_extractor/config.py:28)).

- `VSME_API_PROTOCOL` : protocole OpenAI-compatible (défaut : `chat.completions`). Valeurs supportées : `chat.completions` | `responses` (voir [`load_llm_config()`](vsme_extractor/config.py:28)).
- `VSME_INVOKE_MODE` : mode d’invocation LLM (défaut : `invoke`). Valeurs supportées : `invoke` | `invoke_stream` (voir [`load_llm_config()`](vsme_extractor/config.py:28)).

- `VSM_INPUT_COST_EUR` : coût €/million de tokens en entrée (défaut : `0.15`) (voir [`EURO_COST_PER_MILLION_INPUT`](vsme_extractor/config.py:7)).
- `VSM_OUTPUT_COST_EUR` : coût €/million de tokens en sortie (défaut : `0.60`) (voir [`EURO_COST_PER_MILLION_OUTPUT`](vsme_extractor/config.py:8)).

- `VSM_INDICATORS_PATH` : chemin optionnel vers un CSV d’indicateurs.
  - défaut : CSV packagé [`vsme_extractor/data/indicateurs_vsme.csv`](vsme_extractor/data/indicateurs_vsme.csv:1) (voir [`DEFAULT_INDICATORS_PATH`](vsme_extractor/indicators.py:17)).
- `VSME_CODE_VSME_LIST` : liste optionnelle de `code_vsme` à extraire (séparateurs acceptés : virgule, point-virgule, espaces).
  - défaut (si vide/absent) : l’extracteur conserve les lignes dont `defaut == 1` (voir [`get_indicators()`](vsme_extractor/indicators.py:20)).

Variables de sortie CLI (optionnelles) :
- `VSME_OUTPUT_FORMAT` : `json` (défaut) ou `xlsx`.
  - Peut aussi être forcé via l’option CLI `--output-format` (voir [`build_parser()`](vsme_extractor/cli.py:96)).
- `VSME_OUTPUT_JSON_INCLUDE_STATUS` : active le bloc `status` dans le JSON.
  - défaut : activé (peut être désactivé via `--json-no-status` ou `VSME_OUTPUT_JSON_INCLUDE_STATUS=0`) (voir [`build_parser()`](vsme_extractor/cli.py:96)).

Variables JSON (optionnelles) :
- `VSME_OUTPUT_JSON_INCLUDE_RETRIEVAL_DETAILS` : si `1`, conserve dans chaque indicateur (`results`) les champs de debug retrieval :
  - `Pages candidates`
  - `Pages conservées`
  - `Retrieval par page`
  Par défaut (`0`), ces champs ne sont pas inclus dans le JSON (voir [`build_parser()`](vsme_extractor/cli.py:96) et le filtrage dans [`main()`](vsme_extractor/cli.py:205)).

Enrichissement RSE (JSON) :
- Quand `VSME_OUTPUT_FORMAT=json`, la CLI enrichit chaque indicateur avec des champs issus de la table de correspondance [`vsme_extractor/data/table_codes_portail_rse.csv`](vsme_extractor/data/table_codes_portail_rse.csv:1) (jointure sur `code_vsme` / `Code indicateur`) :
  - `matched_rse_code`
  - `matched_rse_champs_id`
  - `matched_rse_colonne_id`

Variables de robustesse (optionnelles) :
- Rate limit (HTTP 429) :
  - `VSME_RATE_LIMIT_MAX_RETRIES` : nombre de réessais après un 429 (défaut : `0` = aucun retry) (voir [`load_llm_config()`](vsme_extractor/config.py:28)).
  - `VSME_RATE_LIMIT_RETRY_SLEEP_S` : attente (secondes) avant retry (défaut : `60`) (voir [`load_llm_config()`](vsme_extractor/config.py:28)).
  - `VSME_RATE_LIMIT_USE_RETRY_AFTER` : si l’en-tête HTTP `Retry-After` est présent, l’utiliser (défaut : `1`) (voir [`load_llm_config()`](vsme_extractor/config.py:28)).
- Healthcheck LLM au démarrage (utile pour diagnostiquer l’environnement) :
  - `VSME_LLM_HEALTHCHECK` : active/désactive le check (défaut : `1`) (voir [`VSMExtractor.__init__()`](vsme_extractor/pipeline.py:72)).
  - `VSME_LLM_HEALTHCHECK_STRICT` : si `0`, un échec du check n’empêche pas le démarrage (défaut : `1`) (voir [`VSMExtractor.__init__()`](vsme_extractor/pipeline.py:72)).

Variables de retrieval (optionnelles) :
- `retrieval_method` est configurable côté code/app (voir [`VSMExtractor`](vsme_extractor/pipeline.py:70) et l’app Streamlit).
- `VSME_RETRIEVAL_METHOD` : méthode utilisée par la CLI (défaut : `count`) (voir [`build_parser()`](vsme_extractor/cli.py:26)).
- La méthode `count_refine` peut filtrer des pages jugées non pertinentes ; si rien ne passe, l’indicateur est renvoyé à `NA` sans appel LLM.

### Méthodes de retrieval (sélection des extraits/pages)

Ces méthodes sont implémentées dans [`find_relevant_snippets()`](vsme_extractor/retrieval.py:240) et utilisées par [`VSMExtractor`](vsme_extractor/pipeline.py:70).

- `count` (défaut)
  - Principe : comptage d’occurrences (substring) des mots-clés dans chaque page.
  - Portée : le scoring est fait sur **toutes les pages** du PDF, puis on conserve les `k` meilleures pages/extraits (par défaut `k=6`, voir [`VSMExtractor`](vsme_extractor/pipeline.py:65)).
    Ensuite, au moment de l’appel LLM, le contexte est limité aux **6 premiers extraits** maximum (voir [`extract_value_for_metric()`](vsme_extractor/extraction.py:14)).
  - Sélection :
    - on tokenise la requête (mots-clés) ;
    - chaque page reçoit un score = somme des occurrences des tokens dans la page ;
    - seules les pages avec score > 0 sont retenues ;
    - tri décroissant par score, puis on garde les `k` premières.
  - Avantages : robuste sur texte bruité/OCR, simple et efficace.
  - Limites : peut produire des faux positifs (occurrences fortuites), pas de vraie notion de « similarité ».

- `count_refine`
  - Principe : sélectionne d’abord des pages candidates via `count`, puis applique un ranking TF‑IDF *word n‑grams* (1–3) avec tokenisation souple, et conserve uniquement les pages dont le score est ≥ `rel_thr` (par défaut `0.40`) du meilleur score.
  - Particularité : si aucune page ne passe le seuil relatif, l’indicateur est renvoyé `NA` sans appel LLM.

Variables de logging (optionnelles, “opt-in”, utilisées par la CLI et les exemples) :

Logs applicatifs (audit “normal”) :
- `SME_LOG_LEVEL` : `DEBUG|INFO|WARNING|ERROR` (défaut CLI : `INFO`) (voir [`build_parser()`](vsme_extractor/cli.py:18)).
- `VSME_LOG_FILE` : chemin vers un fichier log (optionnel, défaut : non défini) (voir [`build_parser()`](vsme_extractor/cli.py:18)).
- `VSME_LOG_STDOUT` : `1|0` (true/false, yes/no, on/off) (défaut CLI : activé) (voir [`build_parser()`](vsme_extractor/cli.py:18)).

Logs des *prompts* envoyés au LLM (audit “prompt”, **potentiellement sensible**) :
- `VSME_PROMPT_FILE` : chemin vers un fichier dédié aux prompts (optionnel, défaut : non défini → prompts non loggués) (voir [`_configure_prompts_logger()`](vsme_extractor/llm_client.py:86)).
- `VSME_PROMPT_STDOUT` : `1|0` (true/false, yes/no, on/off) (défaut : désactivé) (voir [`_env_bool()`](vsme_extractor/llm_client.py:43)).

Notes :
- Rotation des fichiers de logs : si `VSME_LOG_FILE` et/ou `VSME_PROMPT_FILE` sont configurés, les fichiers sont soumis à :
  - rotation quotidienne (à minuit),
  - rotation dès qu’un fichier dépasse ~10 Mo,
  - rétention 7 jours (suppression des fichiers plus anciens).
- Dans la CLI, le logging réinitialise les handlers existants (comportement par défaut) via `reset_handlers=True` dans [`configure_logging()`](vsme_extractor/logging_utils.py:50).
- Dans un notebook/app qui configure déjà le logging, vous pouvez appeler [`configure_logging_from_env()`](vsme_extractor/logging_utils.py:130) avec `reset_handlers=False`.
- Les prompts **ne sont jamais écrits** dans `VSME_LOG_FILE` : ils sont envoyés uniquement vers les sorties dédiées (`VSME_PROMPT_FILE` et/ou `VSME_PROMPT_STDOUT`).

Exemple de `.env` (ne pas committer de secrets) :
```dotenv
SCW_API_KEY=xxxxxxxxxxxxxxxx
# SCW_BASE_URL=https://api.scaleway.ai/<...>/v1
# SCW_MODEL_NAME=gpt-oss-120b

# Coûts (optionnels)
VSM_INPUT_COST_EUR=0.15
VSM_OUTPUT_COST_EUR=0.60

# CSV indicateurs (optionnel)
# VSM_INDICATORS_PATH=/chemin/vers/indicateurs.csv

# Filtrage indicateurs (optionnel)
# Exemple: VSME_CODE_VSME_LIST=B3_1,B3_2,C1_1
# Si vide/absent: fallback sur `defaut == 1`
# VSME_CODE_VSME_LIST=

# Logs applicatifs (audit “normal”)
SME_LOG_LEVEL=INFO
VSME_LOG_FILE=./tmp/vsme.log
VSME_LOG_STDOUT=1

# Audit prompts LLM (WARNING: peut contenir des données sensibles)
# Prompts loggués séparément de VSME_LOG_FILE (sorties dédiées ci-dessous).
VSME_PROMPT_FILE=./tmp/vsme_prompts.log
VSME_PROMPT_STDOUT=0
```

---

## 6) Usage (copier-coller)

### 6.1 Exemples (scripts + notebooks)
Les tutoriels sont fournis dans [`examples/`](examples/example_cli_extract_pdf_json.py:1) :

- Script (extraction via la CLI, sortie JSON) : [`examples/example_cli_extract_pdf_json.py`](examples/example_cli_extract_pdf_json.py:1)
  ```bash
  .venv/bin/python examples/example_cli_extract_pdf_json.py
  ```

- Script (extraction via la CLI, sortie XLSX) : [`examples/example_cli_extract_pdf_xlsx.py`](examples/example_cli_extract_pdf_xlsx.py:1)
  ```bash
  .venv/bin/python examples/example_cli_extract_pdf_xlsx.py
  ```

- Notebook (extraction PDF, batch dossier, stats complétude) : [`examples/example_cli_usage.ipynb`](examples/example_cli_usage.ipynb:1)

- Notebook (CLI JSON) : [`examples/example_cli_extract_pdf_json.ipynb`](examples/example_cli_extract_pdf_json.ipynb:1)
  - Version notebook du script [`examples/example_cli_extract_pdf_json.py`](examples/example_cli_extract_pdf_json.py:1)
  - Appelle la CLI programmatique via [`vsme_extractor.cli.main()`](vsme_extractor/cli.py:288)
  - Produit un fichier `*.vsme.json` à côté du PDF

- Notebook (CLI XLSX) : [`examples/example_cli_extract_pdf_xlsx.ipynb`](examples/example_cli_extract_pdf_xlsx.ipynb:1)
  - Version notebook du script [`examples/example_cli_extract_pdf_xlsx.py`](examples/example_cli_extract_pdf_xlsx.py:1)
  - Appelle la CLI programmatique via [`vsme_extractor.cli.main()`](vsme_extractor/cli.py:288)
  - Produit un fichier `*.vsme.xlsx` à côté du PDF

> Les exemples chargent [`.env`](.env:1) (si présent) et peuvent activer le logging via `SME_LOG_LEVEL` / `VSME_LOG_FILE` / `VSME_LOG_STDOUT`.

### 6.2 Utiliser la CLI installée (`vsme-extract`)
Après installation via `pip install .` :

- Aide :
  ```bash
  vsme-extract --help
  ```

- Lister les indicateurs actuellement utilisés (après application de `VSM_INDICATORS_PATH` / `VSME_CODE_VSME_LIST`) :
  ```bash
  vsme-extract --list-current-indicators
  ```
  Sortie : une table `code_vsme / Code indicateur / Métrique` triée par `code_vsme`.

- Lister la liste complète des indicateurs (sans filtre `.env`) :
  ```bash
  vsme-extract --list-all-indicators
  ```

- Extraction d’un PDF :
  ```bash
  vsme-extract ./chemin/rapport.pdf
  ```
  Sortie par défaut : `./chemin/rapport.vsme.json`

  Forcer XLSX :
  ```bash
  vsme-extract ./chemin/rapport.pdf --output-format xlsx
  ```
  Sortie : `./chemin/rapport.vsme.xlsx`

  Options JSON :
  - `--output-format json` (défaut)
  - `--json-include-status` (défaut) / `--json-no-status`
  - (alternativement via `.env`) `VSME_OUTPUT_FORMAT` et `VSME_OUTPUT_JSON_INCLUDE_STATUS`

  Si le bloc `status` est activé, la sortie JSON contient :
    - `status.completed` (bool)
    - `status.error` (type/message si exception)
    - `status.filters` (trace des filtres effectifs)
    - `status.missing_codes` (codes demandés absents du CSV indicateurs)

  Détails retrieval (optionnel) :
  - par défaut, les champs `Pages candidates`, `Pages conservées`, `Retrieval par page` **ne sont pas inclus** dans chaque indicateur du JSON
  - pour les inclure :
    - `vsme-extract ./chemin/rapport.pdf --json-include-retrieval-details`
    - ou `VSME_OUTPUT_JSON_INCLUDE_RETRIEVAL_DETAILS=1`

- Extraction d’un dossier de PDFs :
  ```bash
  vsme-extract ./chemin/dossier_pdfs/
  ```

- Calcul de complétude sur un dossier de résultats :
  ```bash
  vsme-extract --count ./chemin/dossier_resultats/
  ```
  Sortie : `./chemin/dossier_resultats/stats_completude.xlsx`

#### Forcer la liste de codes en CLI (surcharge `.env`)
La sélection d’indicateurs peut être pilotée par `.env` via `VSME_CODE_VSME_LIST`. Pour un run ponctuel, le CLI accepte aussi :

```bash
vsme-extract ./chemin/rapport.pdf --codes B3_1,B3_2,C1_1
```

Cette option surcharge `VSME_CODE_VSME_LIST` pour l’exécution courante.

#### Logging pour la CLI (audit)
Le format de log par défaut inclut maintenant l’emplacement du code (`filename:lineno:funcName`) via [`DEFAULT_LOG_FORMAT`](vsme_extractor/logging_utils.py:10), ce qui facilite l’audit.

Vous avez 2 façons de configurer les logs :

1) via options CLI :
```bash
vsme-extract ./chemin/rapport.pdf --log-level DEBUG --log-file ./logs/vsme.log
```

2) via variables d’environnement (pratiques en prod/notebooks) :
```bash
export SME_LOG_LEVEL=DEBUG
export VSME_LOG_FILE=./logs/vsme.log
export VSME_LOG_STDOUT=1
vsme-extract ./chemin/rapport.pdf
```

Notes :
- `INFO` donne les étapes haut niveau (début/fin, pages, nb d’indicateurs, export) + **la taille du contexte par indicateur** (nb d’extraits/pages utilisés, taille en caractères, estimation grossière des tokens).
- `DEBUG` ajoute le détail (progression, traduction keywords cache hit/miss, retrieval, parsing/repair JSON).

⚠️ Fenêtre de contexte / risque de dépassement :
- Par indicateur, le contexte envoyé au LLM est limité à **6 extraits/pages** (voir la limite `[:6]` dans [`extract_value_for_metric()`](vsme_extractor/extraction.py:23)).
- Si le retrieval ne trouve rien (`ctx_selected` vide), l’indicateur est renvoyé à `NA` **sans fallback** (pas d’utilisation du début du document), afin de réduire les faux positifs (voir [`extract_from_pdf()`](vsme_extractor/pipeline.py:321)).
- Malgré ces limites, il existe un risque théorique de dépasser la fenêtre de contexte si les pages sont très longues (tokens ≠ caractères). Les logs “Indicator context …” permettent de surveiller ce point.

#### Audit des prompts envoyés au LLM (optionnel)
Par défaut, les prompts **ne sont pas loggués**.

Pour activer un traçage des prompts (utile pour audit / reproductibilité), utilisez les variables d’environnement :
- `VSME_PROMPT_FILE` (recommandé) : écrit les prompts dans un fichier dédié
- `VSME_PROMPT_STDOUT=1` : écrit aussi les prompts sur stdout

Exemple :
```bash
export VSME_PROMPT_FILE=./tmp/vsme_prompts.log
export VSME_PROMPT_STDOUT=0
vsme-extract ./chemin/rapport.pdf
```

Important :
- Ces prompts peuvent contenir du texte extrait du PDF (potentiellement sensible).
- Ils sont écrits **séparément** des logs applicatifs : ils n’apparaissent pas dans `VSME_LOG_FILE`.

---

## 7) Formats d’entrées/sorties

### Entrée
- PDF (texte extractible via `langchain-community` / `pypdf`).

### Sortie extraction
Un fichier Excel `.vsme.xlsx` (par défaut) contenant typiquement les colonnes :
- `Code indicateur`
- `Thématique`
- `Métrique`
- `Valeur`
- `Unité extraite`
- `Paragraphe source`

Optionnellement, un fichier JSON `.vsme.json` peut être produit (voir `VSME_OUTPUT_FORMAT`).

Schéma JSON (quand activé) :
- `pdf` : chemin du PDF
- `results` : liste d’objets avec les colonnes de sortie
- Chaque indicateur (`results[i]`) inclut aussi (si disponible) les champs RSE :
  - `matched_rse_code`, `matched_rse_champs_id`, `matched_rse_colonne_id` (voir [`vsme_extractor/data/table_codes_portail_rse.csv`](vsme_extractor/data/table_codes_portail_rse.csv:1))
- Les champs retrieval suivants sont optionnels (désactivés par défaut) :
  - `Pages candidates`, `Pages conservées`, `Retrieval par page`
- `stats` : compteurs tokens + coût estimé
- `status` (si `VSME_OUTPUT_JSON_INCLUDE_STATUS=1`) :
  - `completed` : bool
  - `error` : `null` ou `{type, message}`
  - `filters` : trace des variables utilisées (`VSME_CODE_VSME_LIST`, `VSM_INDICATORS_PATH`)
  - `missing_codes` : codes demandés absents du référentiel indicateurs

### Sortie statistiques
Un fichier Excel `stats_completude.xlsx` contenant des métriques de complétude par indicateur (nombre de fichiers où renseigné, etc.).

---

## 8) Développement (structure et points d’entrée)

Arborescence logique :
- [`vsme_extractor/cli.py`](vsme_extractor/cli.py:1) : CLI installable (commande `vsme-extract`).
- [`main.py`](main.py:1) : exemple minimal d’utilisation (librairie), pas la CLI.
- [`examples/example_cli_extract_pdf_json.py`](examples/example_cli_extract_pdf_json.py:1) : script tutoriel (CLI JSON).
- [`examples/example_cli_extract_pdf_xlsx.py`](examples/example_cli_extract_pdf_xlsx.py:1) : script tutoriel (CLI XLSX).
- [`examples/example_cli_usage.ipynb`](examples/example_cli_usage.ipynb:1) : notebook tutoriel (CLI programmatique).
- [`vsme_extractor/pipeline.py`](vsme_extractor/pipeline.py:1) : orchestration extraction (charge PDF, sélectionne snippets, appelle LLM).
- [`vsme_extractor/extraction.py`](vsme_extractor/extraction.py:1) : extraction d’une métrique (prompt + parsing JSON).
- [`vsme_extractor/llm_client.py`](vsme_extractor/llm_client.py:1) : client LLM (OpenAI compatible) + estimation tokens/coûts.
- [`vsme_extractor/indicators.py`](vsme_extractor/indicators.py:1) : chargement des indicateurs (CSV packagé ou surchargé via env).
- [`vsme_extractor/stats.py`](vsme_extractor/stats.py:1) : calcul de complétude sur des fichiers `.vsme.xlsx`.

Note : le calcul de complétude accepte aussi `.vsme.json` générés par la CLI.
- [`vsme_extractor/logging_utils.py`](vsme_extractor/logging_utils.py:1) : configuration centralisée du logging (dont [`configure_logging_from_env()`](vsme_extractor/logging_utils.py:83)).

### App Streamlit (optionnelle)

Une mini-app Streamlit est disponible dans [`streamlit_app/`](streamlit_app/README.md:1).
Elle permet d'uploader un PDF, de choisir des indicateurs (`code_vsme`) et de lancer l'extraction.

### Qualité / checks (optionnel)

- Fichiers de configuration :
  - Pre-commit : [`.pre-commit-config.yaml`](.pre-commit-config.yaml:1)
  - Pyright : [`pyrightconfig.json`](pyrightconfig.json:1)

- Lint / format (ruff) :
  ```bash
  ruff check .
  ruff format .
  ```

- Typage (pyright) :
  ```bash
  pyright
  ```

- Audit dépendances (pip-audit) :
  ```bash
  pip-audit
  ```

  Note : `pip-audit` peut afficher un “Skip Reason” pour `vsme-extractor` lui-même si le projet n’est pas publié sur PyPI (normal).

- Hooks git (pre-commit) :
  ```bash
  pre-commit install
  pre-commit run -a
  ```

  Les hooks incluent `ruff` + `ruff-format` + `pyright` + `pytest`.

---

## 9) Limites connues / Troubleshooting

- PDFs scannés (image) : l’extraction texte peut être très mauvaise sans OCR.
- Qualité des réponses LLM : le modèle peut parfois renvoyer un JSON invalide ou une valeur incorrecte si le contexte est ambigu. Le parsing inclut un fallback + une tentative de “repair” JSON dans [`extract_value_for_metric()`](vsme_extractor/extraction.py:9).
- Coûts : l’extraction dépend du nombre d’indicateurs, de la longueur du document, et des appels additionnels éventuels (traduction/correction JSON).
- Retrieval : par défaut, retrieval “count” ; l’alternative “filtrante” est `count_refine` (voir [`find_relevant_snippets()`](vsme_extractor/retrieval.py:240) et [`VSMExtractor`](vsme_extractor/pipeline.py:70)).
- Logs : utilisez `--log-level DEBUG` et/ou `--log-file` pour diagnostiquer un cas.
- Rate limit (HTTP 429) : en cas de limite « N requêtes/minute », activer un retry avec attente via `VSME_RATE_LIMIT_*`.
- Streamlit upload (403) : selon l’environnement, il peut être nécessaire d’ajuster la config locale Streamlit (voir [`.streamlit/config.toml`](.streamlit/config.toml:1)).

---

## 10) Citer vsme_extractor

Si vous utilisez `vsme_extractor` dans vos recherches, projets ou publications, merci de le citer ainsi :

```bibtex
@misc{vsme_extractor2026,
  title={vsme_extractor: Extraction d'indicateurs VSME depuis des rapports PDF},
  author={François Bullier},
  year={2026},
  url={https://github.com/FBullier/vsme_extractor},
  note={Version 0.1}
}
```

En citant `vsme_extractor`, vous aidez d’autres personnes à découvrir et réutiliser ce travail.

## 11) Licence / Auteur / Contact

- Licence : MIT (voir [`LICENCE.md`](LICENCE.md:1)).
- Auteur : François Bullier
