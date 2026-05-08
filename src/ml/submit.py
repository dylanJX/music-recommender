"""
submit.py — Standalone submission generator.

train.py already handles submission as part of its main flow.
This module is provided for re-running submission separately if needed.

Run: py -3 -m src.ml.submit
"""
from __future__ import annotations

from src.ml.train import main

if __name__ == "__main__":
    main()
