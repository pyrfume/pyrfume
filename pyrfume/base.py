import configparser
import json
import logging
import pickle
import tempfile
from pathlib import Path
from pprint import pprint
from typing import Any, Iterable, List, Union

import pandas as pd
import requests
import toml

PACKAGE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PACKAGE_DIR / "config.ini"
DEFAULT_DATA_PATH = PACKAGE_DIR.parent.parent / "pyrfume-data"
REMOTE_DATA_ORG = "pyrfume"
REMOTE_DATA_REPO = "pyrfume-data"
REMOTE_DATA_PATH = "https://raw.githubusercontent.com/%s/%s" % (REMOTE_DATA_ORG, REMOTE_DATA_REPO)
TEMP_LOCAL = tempfile.TemporaryDirectory()
MANIFEST_NAME = "manifest.toml"


class LocalDataError(Exception):
    pass


class RemoteDataError(Exception):
    pass


logger = logging.getLogger("pyrfume")


def init_config(overwrite: bool = False):
    if overwrite or not CONFIG_PATH.exists():
        config = configparser.ConfigParser()
        config["PATHS"] = {"pyrfume-data": str(DEFAULT_DATA_PATH)}
        config["DATABASE"] = {"schema_name": "UNDEFINED"}
        with open(CONFIG_PATH, "w") as f:
            config.write(f)


def reset_config():
    init_config(overwrite=True)


def read_config(header: str, key: str):
    config = configparser.ConfigParser()
    init_config()
    config.read(CONFIG_PATH)
    return config[header][key]


def write_config(header: str, key: str, value: Any):
    config = configparser.ConfigParser()
    init_config()
    config.read(CONFIG_PATH)
    config[header][key] = value
    with open(CONFIG_PATH, "w") as f:
        config.write(f)


def set_data_path(path: str, create: bool = True):
    path = Path(path).resolve()
    if create:
        path.mkdir(exist_ok=True, parents=True)
    if not path.exists():
        raise Exception("Could not find path %s" % path)
    write_config("PATHS", "pyrfume-data", str(path))


def get_data_path(create: bool = True) -> str:
    path = read_config("PATHS", "pyrfume-data")
    path = Path(path).resolve()
    if create:
        path.mkdir(exist_ok=True, parents=True)
    if not path.exists():
        raise Exception("Could not find data path %s" % path)
    return path


def get_remote_data_path(branch="main"):
    path = REMOTE_DATA_PATH + "/" + branch
    return path


def localize_remote_data(rel_path: str, branch: str = "main", quiet: bool = False) -> str:
    url = get_remote_data_path(branch=branch) + "/" + rel_path
    target_path = Path(TEMP_LOCAL.name) / rel_path
    response = requests.get(url)
    if response.status_code != 200:
        if not quiet:
            raise RemoteDataError("Could not get file at %s" % url)
        else:
            return None
    target_path.parent.mkdir(exist_ok=True, parents=True)
    with open(target_path, "wb") as f:
        f.write(response.content)
    return target_path


def get_remote_archives_info(branch: str = "main") -> dict:
    url = "https://api.github.com/repos/%s/%s/git/trees/%s" % (
        REMOTE_DATA_ORG,
        REMOTE_DATA_REPO,
        branch,
    )
    response = requests.get(url)
    if response.status_code != 200:
        raise RemoteDataError("Could not get archive list at %s" % url)
    info = json.loads(response.content)
    return info


def list_archives(branch: str = "main", remote: bool = None) -> List[str]:
    archives = []
    if remote:
        info = get_remote_archives_info(branch=branch)
        for item in info["tree"]:
            if item["type"] == "tree":
                archives.append(item["path"])
    else:
        path = get_data_path()
        for directory in path.iterdir():
            if (directory / "manifest.toml").is_file():
                archives.append(directory.name)
        if not len(archives):
            f = logger.info if remote is None else logger.warning
            f("No local archives found; searching remotely...")
            return list_archives(branch=branch, remote=True)
    archives = sorted(archives)
    return archives


def load_manifest(archive_name: str, remote: bool = None) -> dict:
    rel_path = archive_name + "/" + MANIFEST_NAME
    return load_data(rel_path, remote=remote)


def show_files(archive_name: str, remote: bool = None, raw: bool = False):
    manifest = load_manifest(archive_name, remote=remote)
    items = manifest["processed"]
    if raw:
        items.update(manifest["raw"])
    pprint(items)


def resolve_lfs(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve the full dataframe if the current one is a pointer to the real data."""
    if df.index.name and "redirect" in df.index.name.lower():
        sha = df.index[0]
        lookup = load_data("lfs-cache.csv")
        try:
            url = lookup.loc[sha, "url"]
        except KeyError:
            raise KeyError(f"Could not find {sha} in the large file cache.")
        df = pd.read_csv(url, index_col=0)
    return df


def load_data(  # noqa: C901 (too complex -- TODO)
    rel_path: str, remote: str = None, cids: Iterable = None, quiet: bool = False, **kwargs
) -> Union[pd.DataFrame, dict]:
    if remote:
        full_path = localize_remote_data(rel_path, quiet=quiet)
    else:
        full_path = get_data_path() / rel_path
        if not full_path.exists():
            if remote is None:
                logger.info("Did not find file %s locally; fetching remotely." % full_path)
                return load_data(rel_path, remote=True, quiet=quiet, **kwargs)
            elif not quiet:
                raise LocalDataError("Could not get file at %s" % full_path)
    if not full_path:
        return None

    is_csv = full_path.suffix in [".csv", ".txt"]
    is_pickle = full_path.suffix in [".pkl", ".pickle", ".p"]
    is_excel = full_path.suffix in [".xls", ".xlsx"]
    is_manifest = full_path.suffix == Path(MANIFEST_NAME).suffix

    if is_pickle:
        with open(full_path, "rb") as f:
            data = pickle.load(f)
    elif is_excel:
        data = pd.read_excel(full_path, **kwargs)
    elif is_csv:
        if "index_col" not in kwargs:
            kwargs["index_col"] = 0
        if cids:
            # First check for a CIDs file specific to this dataset
            cids_path = str(Path(rel_path).parent / "molecules.csv")
            cids_data = load_data(cids_path, remote=remote, quiet=True)
            if cids_data is not None:
                all_cids_ = cids_data.index.tolist()
            else:
                # If not found, fall back on the generic Pyrfume list of CIDs
                from pyrfume.odorants import all_cids

                all_cids_ = all_cids()  # Thousands of CIDs to choose from.
            skip_f = (
                lambda line_num: False if not line_num else all_cids_[line_num - 1] not in cids
            )  # noqa: E73
            kwargs["skiprows"] = skip_f
        data = pd.read_csv(full_path, **kwargs)
        if cids and sorted(data.index.tolist()) != sorted(cids):
            logger.warning(
                "CIDs returned did not match CIDs requested; does this dataset have a custom CID list?"
            )
    elif is_manifest:
        with open(full_path, "r") as f:
            data = toml.load(f)
    else:
        raise LocalDataError("%s has an unknown data type" % rel_path)
    if isinstance(data, pd.DataFrame):
        # Check to see if this is an LFS pointer that needs to be resolved
        data_resolved = resolve_lfs(data)
        if remote and not data_resolved.equals(data):
            # If it was, save the resolved data locally to avoid downloading a large file again
            save_data(data_resolved, rel_path, create_archive=True)
        data = data_resolved
    return data


def save_data(data: pd.DataFrame, rel_path: str, create_archive: bool = True, **kwargs):
    full_path = get_data_path() / rel_path
    is_pickle = any(str(full_path).endswith(x) for x in (".pkl", ".pickle", ".p"))
    is_csv = any(str(full_path).endswith(x) for x in (".csv"))
    archive_path = full_path.parent
    if not archive_path.is_dir():
        if create_archive:
            archive_path.mkdir(exist_ok=True)
        else:
            raise LocalDataError(
                "Archive %s does not exist; use `create_archive=True` to create it"
                % archive_path.name
            )
    if is_pickle:
        with open(full_path, "wb") as f:
            pickle.dump(data, f)
    elif is_csv:
        data.to_csv(full_path, **kwargs)
    else:
        raise Exception("Unsupported extension in file name %s" % full_path.name)


def lfs_hash(rel_path: str, block_size=256 * 128):
    """SHA256 hash of large files for lfs redirect."""
    from hashlib import sha256

    sha256 = sha256()
    with open(rel_path, 'rb') as f:
        for chunk in iter(lambda: f.read(block_size), b''):
            sha256.update(chunk)
    return sha256.hexdigest()
