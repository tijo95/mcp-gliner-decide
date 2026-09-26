# MCP GLiNER 2.5 — Serveur MCP pour Unsloth Studio

## Description

Serveur MCP (Model Context Protocol) basé sur **GLiNER 2.5** (`fastino/GLiNER2.5-Decide`) pour :

- **L'extraction d'entités nommées** (personnes, lieux, produits, dates, etc.) à partir de n'importe quel texte.
- **La classification Zero-Shot ultra-rapide** : décision structurée d'un texte selon une liste de labels fournies par l'utilisateur, sans réentraînement (inférence CPU, latence très faible).

Le serveur expose deux outils MCP :

| Outil | Description |
|-------|-------------|
| `classify_text` | Classe un texte selon une liste de labels fournis (avec traduction automatique FR → EN pour maximiser la précision Zero-Shot). |
| `extract_entities` | Extrait des entités nommées à partir d'un texte selon les types d'entités demandés. |

Le modèle est téléchargé automatiquement depuis Hugging Face au premier lancement (puis mis en cache).

## Installation

### Prérequis

- [Git](https://git-scm.com/)
- **Python 3.13**

### Étapes

```bash
# 1. Cloner le dépôt
git clone https://github.com/tijo95/mcp-gliner-decide.git
cd mcp-gliner-decide

# 2. Créer un environnement virtuel (venv) avec Python 3.13
python -m venv venv

# 3. Activer le venv (Windows)
venv\Scripts\activate

# 4. Installer les dépendances
pip install -r requirements.txt
```

> 💡 Le modèle `fastino/GLiNER2.5-Decide` sera téléchargé au premier lancement du serveur. Assurez-vous d'une connexion Internet la première fois.

## Utilisation

Lancer le serveur MCP :

```bash
venv\Scripts\python.exe server.py
```

Le serveur communique via `stdin`/`stdout` (protocole MCP stdio) et affiche ses logs de chargement sur `stderr`.

## Configuration Unsloth Studio

Ajouter le snippet suivant dans la configuration MCP d'Unsloth Studio (remplacez `VOTRE_CHemin` par le chemin réel de votre dépôt) :

```json
{
  "mcpServers": {
    "GLiNER2.5-Decide": {
      "command": "C:\Users\VOTRE_CHemin\mcp-gliner-decide\venv\Scripts\python.exe",
      "args": ["C:\Users\VOTRE_CHemin\mcp-gliner-decide\server.py"]
    }
  }
}
```

Une fois chargé, les outils `classify_text` et `extract_entities` seront disponibles dans vos conversations.

### Exemples d'appels

**Classification :**

```json
{"text": "Ce produit est excellent, je le recommande !", "labels": ["positif", "négatif", "neutre"]}
```

**Extraction d'entités :**

```json
{"text": "Marie Dupont a visité Paris le 15 mars 2025.", "entity_types": ["person", "location", "date"]}
```

## Structure du projet

```
mcp-gliner/
├── server.py          # Serveur MCP principal (outils classify_text / extract_entities)
├── mcp-gliner.py      # Variante FastMCP alternative
├── requirements.txt   # Dépendances Python verrouillées (pip freeze)
├── .gitignore
└── README.md
```

## Dépannage

- **Le modèle tarde à charger au premier lancement** : normal, il est téléchargé depuis Hugging Face (mis en cache dans `~/.cache/huggingface`).
- **Erreur d'encodage** : le serveur force `PYTHONIOENCODING=utf-8` ; sur Windows, vérifiez que votre terminal est en UTF-8.
- **Port/stdio** : le serveur utilise uniquement le stdio ; aucun port réseau n'est ouvert.
