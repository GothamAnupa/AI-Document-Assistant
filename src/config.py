from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
DB_ROOT = APP_DIR / "db"
ACTIVE_DB_FILE = DB_ROOT / "active_db.txt"
DEFAULT_DB_DIR = DB_ROOT / "current"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"
TOP_K = 4
