from pathlib import Path
import tempfile

PACKAGE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PACKAGE_DIR / "config.ini"
DEFAULT_DATA_PATH = PACKAGE_DIR.parent / "data"
REMOTE_DATA_ORG = 'pyrfume'
REMOTE_DATA_REPO = 'pyrfume-data'
REMOTE_DATA_PATH = 'https://raw.githubusercontent.com/%s/%s' % (REMOTE_DATA_ORG, REMOTE_DATA_REPO)
TEMP_LOCAL = tempfile.TemporaryDirectory()
MANIFEST_NAME = 'manifest.toml'


class LocalDataError(Exception):
    pass


class RemoteDataError(Exception):
    pass