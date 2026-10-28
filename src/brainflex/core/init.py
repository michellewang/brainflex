"""Project initialization."""

import os
import re
import shutil
from pathlib import Path

from brainflex.utils.constants import (
    BRAINFLEX_DIRNAME,
    ENV_EXAMPLE_FILENAME,
    ENV_FILENAME,
    ENV_VAR_GID,
    ENV_VAR_UID,
    VPN_CONFIG_DIRNAME,
    VPN_CONFIG_FILENAME,
)


def _set_variable(content: str, name: str, value: str) -> str:
    """Set a variable in .env file content, uncommenting it if needed.

    The variable is appended if it is not in the content.
    """
    line = f"{name}={value}"
    pattern = re.compile(rf"^#?[ \t]*{re.escape(name)}=.*$", flags=re.MULTILINE)
    content, n_subs = pattern.subn(line, content, count=1)
    if n_subs == 0:
        if content and not content.endswith("\n"):
            content += "\n"
        content += f"{line}\n"
    return content


def init(root: Path, fpath_vpn_config: Path) -> None:
    """Create the Docker Compose configuration of the project.

    Raises FileNotFoundError if an expected path doesn't exist, and FileExistsError if
    the .env file or the VPN configuration file already exists in the project.
    """
    dpath_brainflex = root / BRAINFLEX_DIRNAME
    if not dpath_brainflex.is_dir():
        raise FileNotFoundError(f"Not a brainflex project: {dpath_brainflex} not found")

    fpath_example = root / ENV_EXAMPLE_FILENAME
    if not fpath_example.is_file():
        raise FileNotFoundError(f"{fpath_example} not found")

    if not fpath_vpn_config.is_file():
        raise FileNotFoundError(f"VPN configuration file {fpath_vpn_config} not found")

    fpath_env = root / ENV_FILENAME
    if fpath_env.exists():
        raise FileExistsError(f"Project already initialized: {fpath_env} exists")

    fpath_vpn_config_dst = root / VPN_CONFIG_DIRNAME / VPN_CONFIG_FILENAME
    # the file may already have been put in the right place
    copy_vpn_config = not (
        fpath_vpn_config_dst.exists()
        and fpath_vpn_config.resolve() == fpath_vpn_config_dst.resolve()
    )
    if copy_vpn_config and fpath_vpn_config_dst.exists():
        raise FileExistsError(
            f"Project already initialized: {fpath_vpn_config_dst} exists"
        )

    content = fpath_example.read_text()
    content = _set_variable(content, ENV_VAR_UID, str(os.getuid()))
    content = _set_variable(content, ENV_VAR_GID, str(os.getgid()))
    fpath_env.write_text(content)

    if copy_vpn_config:
        fpath_vpn_config_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(fpath_vpn_config, fpath_vpn_config_dst)

    # contains the VPN credentials
    fpath_vpn_config_dst.chmod(0o600)
