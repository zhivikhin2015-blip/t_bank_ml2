"""Запуск исходных научных проверок в изолированной структуре эксперимента."""
import atexit
import importlib
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'scripts'))
from reproduce import prepare_workspace

_workspace = None


def load_tests(loader, standard_tests, pattern):
    global _workspace
    cache = ROOT / '.cache' / 'tests'
    cache.mkdir(parents=True, exist_ok=True)
    _workspace = tempfile.TemporaryDirectory(dir=cache)
    atexit.register(_workspace.cleanup)
    workspace = prepare_workspace(Path(_workspace.name), attach_raw=False)
    sys.path.insert(0, str(workspace))
    checks = importlib.import_module('research_v2.tests')
    return loader.loadTestsFromTestCase(checks.ScientificChecks)
