from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PACKAGE_DIR / "config.ini"
DEFAULT_DATA_PATH = PACKAGE_DIR.parent / "data"
