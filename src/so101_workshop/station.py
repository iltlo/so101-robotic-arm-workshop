"""Find or create the directory that holds this station's yaml files."""

import os
from pathlib import Path

TEMPLATES = Path(__file__).resolve().parent / "templates"


def resolve_station(explicit: str | None = None) -> Path:
    """Repo config when launched from a checkout, otherwise ~/.so101-workshop."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("SO101_WORKSHOP_HOME")
    if env:
        return Path(env).expanduser().resolve()
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        marker = candidate / "pyproject.toml"
        if marker.is_file() and (candidate / "config" / "robot.yaml").is_file():
            text = marker.read_text(errors="ignore")
            if "so101-workshop" in text or 'name = "so101-control"' in text:
                return candidate
        if candidate == candidate.parent:
            break
    return (Path.home() / ".so101-workshop").resolve()


def seed_station(root: Path) -> None:
    """Copy packaged defaults into a station. Existing files stay.

    Every template is copied, so new camera presets (reCamera, Logitech,
    USB webcam) appear on an existing station without overwriting edits.
    """
    config = root / "config"
    config.mkdir(parents=True, exist_ok=True)
    for template in sorted(TEMPLATES.glob("*.yaml")):
        dest = config / template.name
        if dest.exists():
            continue
        dest.write_text(template.read_text())


def prepare_station(explicit: str | None = None) -> Path:
    root = resolve_station(explicit)
    seed_station(root)
    return root
