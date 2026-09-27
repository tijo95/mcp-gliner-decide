import sys
import os
import logging
import contextlib

# Silencer les warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("GLiNER2.5-Multi")

print("Chargement du modèle GLiNER 2.5 Multilingue...", file=sys.stderr)

with contextlib.redirect_stdout(sys.stderr):
    from gliner2 import AutoExtractor
    from gliner2.classification import Classifier, ClassificationSchema, SchemaError

    model = AutoExtractor.from_pretrained("fastino/gliner2.5-multi-v1")
    classifier = Classifier(model)

print("Modèle prêt !", file=sys.stderr)

# Le modèle (mDeBERTa v3 multilingue) comprend les labels en français :
# aucune traduction intermédiaire n'est nécessaire.
TASK_NAME = "classification"


def _normalise(labels) -> list[str]:
    """Nettoie et déduplique les labels (ordre et casse d'origine conservés)."""
    result, seen = [], set()
    for label in labels:
        label = str(label).strip()
        key = label.lower()
        if label and key not in seen:
            seen.add(key)
            result.append(label)
    return result


@mcp.tool()
def classify_text(text: str, labels: list[str]) -> dict:
    """Classifie le texte en une classe parmi les labels fournis, via la tête de
    classification dédiée de GLiNER 2.5. Les labels peuvent être en français :
    le modèle est multilingue, aucune traduction n'est nécessaire."""
    try:
        clean_labels = _normalise(labels)
        if not clean_labels:
            return {"status": "error", "message": "Aucun label valide fourni."}

        schema = ClassificationSchema().single(TASK_NAME, clean_labels)
        result = classifier.classify(text, schema)
        task = result[TASK_NAME]

        return {
            "status": "success",
            "prediction": task.label,
            "confidence": task.confidence,
            "probabilities": {k: float(v) for k, v in task.probabilities.items()},
        }

    except SchemaError as e:
        # Un label contient un jeton réservé (parenthèses, marqueurs [P], [L], ...)
        return {"status": "error", "message": f"Labels invalides pour la classification : {e}"}
    except Exception as e:
        print(f"[ERROR] Exception in classify_text: {e}", file=sys.stderr)
        return {"status": "error", "message": str(e)}


@mcp.tool()
def extract_entities(text: str, entity_types: list[str]) -> dict:
    """Extrait les entités du texte pour les types demandés. Les types peuvent
    être en français ou en anglais : le modèle est multilingue, aucune
    traduction n'est nécessaire."""
    try:
        clean_types = _normalise(entity_types)
        if not clean_types:
            return {"status": "error", "message": "Aucun type d'entité valide fourni."}

        # Le résultat de GLiNER 2.x est un dict : {'entities': {type: [entités, ...]}}
        raw_res = model.extract_entities(text, clean_types, include_confidence=True)
        raw_entities = raw_res.get("entities", {}) if isinstance(raw_res, dict) else {}

        formatted_entities = {}
        confidences = {}
        for etype in clean_types:
            items = raw_entities.get(etype, []) or []
            texts, confs = [], []
            for item in items:
                if isinstance(item, dict):
                    texts.append(item.get("text"))
                    confs.append(item.get("confidence"))
                else:
                    texts.append(item)
                    confs.append(None)
            formatted_entities[etype] = texts
            confidences[etype] = confs

        return {"status": "success", "entities": formatted_entities, "confidences": confidences}

    except Exception as e:
        print(f"[ERROR] Exception in extract_entities: {e}", file=sys.stderr)
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    mcp.run()