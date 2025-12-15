"""
Wrapper to run src.preprocess_and_filter.main()
"""

import sys
from pathlib import Path

# ensure project root on PYTHONPATH (so src.* imports work)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocess_and_filter import main as preprocess_main

if __name__ == "__main__":
    preprocess_main()
