import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_DIR = Path(__file__).parent.parent
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    return config

CONFIG = load_config()

def get_gemini_api_key():
    return os.getenv("GEMINI_API_KEY")

def get_gemini_model():
    return os.getenv("GEMINI_MODEL", CONFIG.get("DEFAULT_GEMINI_MODEL", "gemini-2.5-flash"))
