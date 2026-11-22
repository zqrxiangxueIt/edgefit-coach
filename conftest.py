# root conftest: puts the repo root on sys.path so tests can
# `from server... import` / `from edge... import` on a FRESH clone
# (without this, plain `pytest` fails with ModuleNotFoundError)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
