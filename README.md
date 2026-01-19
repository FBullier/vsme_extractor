# Projet VSME Extractor

## 1) Objectif

Ce projet extrait automatiquement des indicateurs VSME (ESG) à partir de rapports PDF d’entreprise et produit un fichier Excel `.vsme.xlsx` contenant les valeurs extraites (ou `NA` si non trouvées).

Le cœur de l’extraction se trouve dans le package `vsme_extractor/`.
La CLI installable est définie dans [`vsme_extractor/cli.py`](vsme_extractor/cli.py:1) (commande `vsme-extract`).
Les exemples d’utilisation “librairie” (script + notebook) sont disponibles dans [`exemples/`](exemples/example_extract_pdf.py:1).

---

## 2) Fonctionnalités principales

- Extraction d’un PDF vers un fichier Excel `.vsme.xlsx`.
- Extraction en batch sur un dossier contenant des PDFs.
- Calcul de complétude des indicateurs sur un dossier de `.vsme.xlsx` (génère `stats_completude.xlsx`).
- Estimation des tokens (et du coût estimé en euros) par exécution.
- Logging configurable (stdout et/ou fichier), pour faciliter le debug et l’exploitation.

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

Variables principales :
- `SCW_API_KEY` : clé API Scaleway (obligatoire).
- `SCW_BASE_URL` : URL base (défaut : endpoint Scaleway dans le code).
- `SCW_MODEL_NAME` : nom du modèle (défaut : `gpt-oss-120b`).
- `VSM_INPUT_COST_EUR` : coût €/million de tokens en entrée (défaut : `0.15`).
- `VSM_OUTPUT_COST_EUR` : coût €/million de tokens en sortie (défaut : `0.60`).
- `VSM_INDICATORS_PATH` : chemin optionnel vers un CSV d’indicateurs (sinon fallback vers le CSV packagé, par défaut [`vsme_extractor/data/indicateurs_vsme.csv`](vsme_extractor/data/indicateurs_vsme.csv:1)).
- `VSME_CODE_VSME_LIST` : liste optionnelle de `code_vsme` à extraire (séparateurs acceptés : virgule, point-virgule, espaces).
  - si `VSME_CODE_VSME_LIST` est vide/absent : l’extracteur prend par défaut les lignes dont `defaut == 1`.

Variables de logging (optionnelles, “opt-in”, utilisées par la CLI et les exemples) :

Logs applicatifs (audit “normal”) :
- `SME_LOG_LEVEL` : `DEBUG|INFO|WARNING|ERROR`
- `VSME_LOG_FILE` : chemin vers un fichier log (optionnel)
- `VSME_LOG_STDOUT` : `1|0` (true/false, yes/no, on/off)

Logs des *prompts* envoyés au LLM (audit “prompt”, **potentiellement sensible**) :
- `VSME_PROMPT_FILE` : chemin vers un fichier dédié aux prompts (optionnel)
- `VSME_PROMPT_STDOUT` : `1|0` (true/false, yes/no, on/off)

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

### 6.1 Utiliser la librairie (script + notebook)
Deux tutoriels sont fournis dans [`exemples/`](exemples/example_extract_pdf.py:1) :

- Script (extraction d’un PDF) : [`exemples/example_extract_pdf.py`](exemples/example_extract_pdf.py:1)
  ```bash
  python exemples/example_extract_pdf.py
  ```

- Notebook (extraction PDF, batch dossier, stats complétude) : [`exemples/example_usage.ipynb`](exemples/example_usage.ipynb:1)

> Les exemples chargent [`.env`](.env:1) (si présent) et peuvent activer le logging via `SME_LOG_LEVEL` / `VSME_LOG_FILE` / `VSME_LOG_STDOUT`.

### 6.2 Utiliser la CLI installée (`vsme-extract`)
Après installation via `pip install .` :

- Aide :
  ```bash
  vsme-extract --help
  ```

- Lister les indicateurs chargés (après application de `VSM_INDICATORS_PATH` / `VSME_CODE_VSME_LIST`) :
  ```bash
  vsme-extract --list-indicators
  ```
  Sortie : une table `code_vsme / Code indicateur / Métrique` triée par `code_vsme`.

- Extraction d’un PDF :
  ```bash
  vsme-extract ./chemin/rapport.pdf
  ```
  Sortie : `./chemin/rapport.vsme.xlsx`

- Extraction d’un dossier de PDFs :
  ```bash
  vsme-extract ./chemin/dossier_pdfs/
  ```

- Calcul de complétude sur un dossier de résultats :
  ```bash
  vsme-extract --count ./chemin/dossier_resultats/
  ```
  Sortie : `./chemin/dossier_resultats/stats_completude.xlsx`

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
- Si le retrieval ne trouve rien, fallback : on utilise un extrait du début du document (`full_text[:10000]`) dans [`extract_from_pdf()`](vsme_extractor/pipeline.py:190).
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
Un fichier Excel `.vsme.xlsx` contenant typiquement les colonnes :
- `Code indicateur`
- `Thématique`
- `Métrique`
- `Valeur`
- `Unité extraite`
- `Paragraphe source`

### Sortie statistiques
Un fichier Excel `stats_completude.xlsx` contenant des métriques de complétude par indicateur (nombre de fichiers où renseigné, etc.).

---

## 8) Développement (structure et points d’entrée)

Arborescence logique :
- [`vsme_extractor/cli.py`](vsme_extractor/cli.py:1) : CLI installable (commande `vsme-extract`).
- [`main.py`](main.py:1) : exemple minimal d’utilisation (librairie), pas la CLI.
- [`exemples/example_extract_pdf.py`](exemples/example_extract_pdf.py:1) : script tutoriel.
- [`exemples/example_usage.ipynb`](exemples/example_usage.ipynb:1) : notebook tutoriel.
- [`vsme_extractor/pipeline.py`](vsme_extractor/pipeline.py:1) : orchestration extraction (charge PDF, sélectionne snippets, appelle LLM).
- [`vsme_extractor/extraction.py`](vsme_extractor/extraction.py:1) : extraction d’une métrique (prompt + parsing JSON).
- [`vsme_extractor/llm_client.py`](vsme_extractor/llm_client.py:1) : client LLM (OpenAI compatible) + estimation tokens/coûts.
- [`vsme_extractor/indicators.py`](vsme_extractor/indicators.py:1) : chargement des indicateurs (CSV packagé ou surchargé via env).
- [`vsme_extractor/stats.py`](vsme_extractor/stats.py:1) : calcul de complétude sur des fichiers `.vsme.xlsx`.
- [`vsme_extractor/logging_utils.py`](vsme_extractor/logging_utils.py:1) : configuration centralisée du logging (dont [`configure_logging_from_env()`](vsme_extractor/logging_utils.py:83)).

---

## 9) Limites connues / Troubleshooting

- PDFs scannés (image) : l’extraction texte peut être très mauvaise sans OCR.
- Qualité des réponses LLM : le modèle peut parfois renvoyer un JSON invalide ou une valeur incorrecte si le contexte est ambigu. Le parsing inclut un fallback + une tentative de “repair” JSON dans [`extract_value_for_metric()`](vsme_extractor/extraction.py:9).
- Coûts : l’extraction dépend du nombre d’indicateurs, de la longueur du document, et des appels additionnels éventuels (traduction/correction JSON).
- Retrieval : par défaut, retrieval “count” ; une alternative BM25 existe côté code via `method="bm25"` dans [`find_relevant_snippets()`](vsme_extractor/retrieval.py:10) et `retrieval_method="bm25"` dans [`VSMExtractor`](vsme_extractor/pipeline.py:56).
- Logs : utilisez `--log-level DEBUG` et/ou `--log-file` pour diagnostiquer un cas.

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
