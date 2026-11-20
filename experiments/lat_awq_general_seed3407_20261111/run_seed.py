"""Use the original experiment runner with a separate output directory."""
import importlib.util
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE.parents[1] / 'outputs' / BASE.name
ORIGINAL = BASE.parent / 'lat_awq_general_20261107' / 'source' / 'general_run.py'


if __name__ == '__main__':
    spec = importlib.util.spec_from_file_location('original_general_run', ORIGINAL)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.OUT = OUT
    runner.main()
