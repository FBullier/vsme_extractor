from __future__ import annotations

EXTRACTION_SYSTEM_PROMPT = """Tu es un assistant d'extraction de données ESG.
Ta tâche est d'extraire **uniquement la valeur demandée** depuis un contexte PDF d'entreprise.
Règles:
- Réponds STRICTEMENT au format JSON valide: {"valeur": "..."}
- Le champ "valeur" doit **uniquement** contenir la donnée demandée (ex: un nombre + unité si applicable), sans commentaire.
- Si la valeur est absente/non publiée, réponds: {"valeur": "NA"}
- Privilégie l'année la plus récente disponible (par défaut 2024 si le document est 2024/2025). 
- Si plusieurs chiffres existent (ex: brut vs net), choisis le total consolidé le plus pertinent et le plus récent.
- Respecte l'unité cible si possible; sinon, renvoie l'unité telle que publiée.
"""

def build_user_prompt(metric: str, unite: str, context: str) -> str:
    """Construit le prompt utilisateur envoyé au LLM pour une métrique donnée."""
    return f"""Contexte PDF (extraits) :
{context}

Extrait la métrique suivante :
Métrique: {metric}
Unité cible: {unite}

Réponds AU FORMAT JSON : {{"valeur": "<valeur_seule>", "unité": "<unité_seule>", "paragraphe": "<texte_ou_se_trouve_la_valeur>"}}
Si la valeur est manquante dans le Contexte PDF (extraits) renvoit : {{"valeur": "NA", "unité":"NA", "paragraphe": "NA"}}.
"""
