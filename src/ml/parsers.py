"""
parsers.py — Re-exports the 4 parser functions from data/hw4.py.
"""
import importlib.util
from pathlib import Path

_HW4_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "hw4.py"
_spec = importlib.util.spec_from_file_location("hw4", _HW4_PATH)
_hw4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hw4)

read_train_history = _hw4.read_train_history
read_test_items = _hw4.read_test_items
read_track_metadata = _hw4.read_track_metadata
reorder_to_sample = _hw4.reorder_to_sample

DATA_DIR = _HW4_PATH.parent

if __name__ == "__main__":
    print("Smoke-testing parsers against real data files...")

    history = read_train_history(str(DATA_DIR / "trainItem2.txt"))
    print(f"  read_train_history: {len(history)} users")

    test_users = read_test_items(str(DATA_DIR / "testItem2.txt"))
    print(f"  read_test_items:    {len(test_users)} users, "
          f"{sum(len(v) for v in test_users.values())} pairs")

    meta = read_track_metadata(str(DATA_DIR / "trackData2.txt"))
    print(f"  read_track_metadata: {len(meta)} tracks")

    print("All parsers OK.")
