import sys
from mcp.server.fastmcp import FastMCP
from gliner2 import GLiNER2

# Initialisation du serveur MCP
mcp = FastMCP("GLiNER2.5-Decide")

# Chargement du modèle sur le CPU (RAM)
print("Chargement du modèle GLiNER 2.5...", file=sys.stderr)
model = GLiNER2.from_pretrained("fastino/GLiNER2.5-Decide", device="cpu")
print("Modèle prêt !", file=sys.stderr)

@mcp.tool()
def classify_text(text: str, labels: list[str]) -> dict:
    """Classifie ou prend une décision sur un texte en fonction d'une liste de labels ou d'options."""
    results = model.predict_labels(text, labels)
    return {"predictions": results}

@mcp.tool()
def extract_entities(text: str, entity_types: list[str]) -> dict:
    """Extrait des entités nommées (nom, date, lieu, produit, etc.) à partir d'un texte."""
    entities = model.extract_entities(text, entity_types)
    return {"entities": entities}

if __name__ == "__main__":
    mcp.run()