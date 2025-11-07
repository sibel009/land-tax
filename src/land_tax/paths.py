"""Project directory helpers."""
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent.parent
DATA_DIR = REPO_ROOT / "data"
PARAMS_DIR = DATA_DIR / "params"
SOLUTIONS_DIR = DATA_DIR / "solutions"
DYNAMICS_DIR = DATA_DIR / "dynamics"

__all__ = [
    "PACKAGE_ROOT",
    "REPO_ROOT",
    "DATA_DIR",
    "PARAMS_DIR",
    "SOLUTIONS_DIR",
    "DYNAMICS_DIR",
]
