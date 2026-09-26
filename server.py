import sys
import os
import logging
import contextlib

# Suppression des logs parasites sur STDOUT
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("GLiNER2.5-Decide")

print("Chargement du modèle GLiNER 2.5...", file=sys.stderr)

with contextlib.redirect_stdout(sys.stderr):
    from gliner2 import AutoExtractor
    model = AutoExtractor.from_pretrained("fastino/GLiNER2.5-Decide")

print("Modèle prêt !", file=sys.stderr)

# Dictionnaire de traduction FR -> EN pour maximiser la précision Zero-Shot
TRANSLATIONS = {
    "positif": "positive",
    "négatif": "negative",
    "neutre": "neutral"
}
REVERSE_TRANSLATIONS = {v: k for k, v in TRANSLATIONS.items()}

@mcp.tool()
def classify_text(text: str, labels: list[str]) -> dict:
    """Classifie un texte selon une liste de labels fournis."""
    try:
        # 1. Traduction optionnelle des labels FR -> EN pour l'inférence
        en_labels = [TRANSLATIONS.get(l.lower(), l) for l in labels]

        # 2. Construction d'un JSON Schema strictly valide à la racine
        schema = {
            "type": "object",
            "properties": {
                "sentiment": {
                    "type": "string",
                    "enum": en_labels,
                    "description": "The sentiment or class of the text"
                }
            },
            "required": ["sentiment"]
        }

        # Debug log dans stderr
        print(f"[DEBUG] Text: {text}", file=sys.stderr)
        print(f"[DEBUG] Schema: {schema}", file=sys.stderr)

        # 3. Essai via extract_json (génération structurée)
        res = model.extract_json(text, schema)
        print(f"[DEBUG] Raw extract_json output: {res}", file=sys.stderr)

        prediction = None

        if isinstance(res, dict) and "sentiment" in res and res["sentiment"]:
            prediction = res["sentiment"]
        
        # 4. Fallback : Si extract_json renvoie du vide, on tente la méthode native de GLiNER
        if not prediction and hasattr(model, "predict_entities"):
            print("[DEBUG] Fallback vers predict_entities...", file=sys.stderr)
            entities = model.predict_entities(text, en_labels)
            if entities:
                # On prend l'entité retenue avec le plus haut score
                prediction = sorted(entities, key=lambda x: x.get("score", 0), reverse=True)[0].get("label")

        # 5. Traduction inverse du résultat EN -> FR si nécessaire
        final_label = REVERSE_TRANSLATIONS.get(prediction, prediction) if prediction else "neutre"

        return {"status": "success", "prediction": final_label}

    except Exception as e:
        print(f"[ERROR] Exception in classify_text: {e}", file=sys.stderr)
        return {"status": "error", "message": str(e)}

@mcp.tool()
def extract_entities(text: str, entity_types: list[str]) -> dict:
    """Extrait des entités nommées à partir d'un texte."""
    try:
        if hasattr(model, "predict_entities"):
            res = model.predict_entities(text, entity_types)
        else:
            res = model.extract_entities(text, entity_types)
        return {"status": "success", "entities": res}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    mcp.run()