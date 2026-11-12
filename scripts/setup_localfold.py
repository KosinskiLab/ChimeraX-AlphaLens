#!/usr/bin/env python3
"""Repository entry point; release downloads use the self-contained src/predict_setup.py."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[1] / 'src' / 'predict_setup.py'), run_name='__main__')
