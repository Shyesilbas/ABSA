"""Proje kökünden etkileşimli tahmin: `python predict.py` (venv açıkken)."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
runpy.run_path(str(SRC / "app" / "predict.py"), run_name="__main__")
