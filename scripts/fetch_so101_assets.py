"""Install the pinned SO-101 follower URDF, its meshes, and upstream license."""

import os
from pathlib import Path
from urllib.request import urlopen

import so101_utils.assets as assets

COMMIT = assets.COMMIT
URL = assets.URL
DEST = Path(__file__).resolve().parents[1] / "assets" / "SO101"
URDF = assets.URDF
PREFIX = assets.PREFIX
SOURCE = assets.SOURCE
ET = assets.ET
OLD_DOWNLOADS = assets.OLD_DOWNLOADS
_mesh_paths = assets._mesh_paths
_tool_frame_x_out_fingers = assets._tool_frame_x_out_fingers
_validate_stl = assets._validate_stl
_archive_files = assets._archive_files
_install = assets._install
_read_unique = assets._read_unique


def main() -> None:
    assets.DEST = DEST
    assets.urlopen = urlopen
    assets.main()


def ensure_assets(dest: Path | None = None) -> Path:
    assets.urlopen = urlopen
    return assets.ensure_assets(DEST if dest is None else dest)


if __name__ == "__main__":
    main()
