"""Make the package and the vendored library importable from a fresh clone without `pip install -e .`."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "third_party"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def load_by_path(module_name: str, relative_path: str):
    """Load a single module without executing the package __init__.

    Used by the light tests so that pure-logic components (the autoreset probe, the capacity arithmetic) can be
    checked without torch, pennylane or a quantum backend installed.
    """
    import importlib.util

    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module          # dataclasses needs this
    spec.loader.exec_module(module)
    return module
