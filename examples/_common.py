"""Shared setup for the examples: a cached spectral database and output paths."""

import warnings
from pathlib import Path

from hyperatmos import SpectralDatabase, standard_atmosphere

warnings.filterwarnings("ignore", message="SpectralDatabase uses synthetic")

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT.parent / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = ROOT / ".cache" / "demo_db_380_2500.npz"

DEMO_NOTE = "synthetic demo spectroscopy (illustrative only)"


def demo_database() -> SpectralDatabase:
    """Build once, then load from cache. Pass HITRAN line lists to build() for real work."""
    if DB_PATH.exists():
        return SpectralDatabase.load(DB_PATH)
    DB_PATH.parent.mkdir(exist_ok=True)
    db = SpectralDatabase.build(standard_atmosphere(), (380.0, 2500.0))
    db.save(DB_PATH)
    return db
