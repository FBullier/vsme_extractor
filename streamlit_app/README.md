# App Streamlit — VSME Extractor

Cette mini-app permet :
- d'uploader un PDF,
- de sélectionner des indicateurs (par `code_vsme`),
- de lancer l'extraction,
- de visualiser et télécharger le résultat.

## Lancer l'app

1) Installer les dépendances :

```bash
pip install .
pip install ".[streamlit]"
```

2) Définir la config (clé Scaleway, etc.) dans [`.env`](../.env:1).

3) Démarrer :

```bash
streamlit run streamlit_app/app.py
```
