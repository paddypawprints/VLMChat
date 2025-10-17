"""Top-level utils package proxy.

Tests and some modules import `utils.*` as a top-level package. The actual
implementation lives under `src.utils`. To avoid changing many imports, this
module attempts to import and re-export the `src.utils` package when available.
"""
try:
    # If running from repo root, ensure src is on sys.path and import
    import sys
    import os
    # ensure `src` is on sys.path so `from src import ...` works
    if os.path.abspath('src') not in [os.path.abspath(p) for p in sys.path]:
        sys.path.insert(0, 'src')
    # Add the real src/utils directory to this package's __path__ so imports
    # like `import utils.image_utils` resolve to src/utils/image_utils.py
    src_utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'utils'))
    if os.path.isdir(src_utils_path) and src_utils_path not in __path__:
        __path__.insert(0, src_utils_path)
    # also attempt to import src.utils to populate attributes if present
    try:
        from src import utils as _src_utils
        for _name in dir(_src_utils):
            if not _name.startswith('_'):
                try:
                    globals()[_name] = getattr(_src_utils, _name)
                except Exception:
                    pass
    except Exception:
        pass
except Exception:
    # best-effort only; real imports will still work when running the app normally
    pass
