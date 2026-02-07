import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROOT_PATH = str(ROOT)
SRC_PATH = str(ROOT / "src")

if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
