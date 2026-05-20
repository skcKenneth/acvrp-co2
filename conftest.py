"""
Project-root conftest.py.

Pytest discovers this file when run from the project root and uses its
location as the "rootdir". As a side-effect, the directory containing
this file is prepended to sys.path, which makes `import src.<module>`
work for the test modules under `tests/`.

You should not need to edit this file. It only exists so that running
`pytest` from the project root just works.
"""
import sys
from pathlib import Path

# Insert the project root (the directory holding this file) at the
# front of sys.path. This guarantees that `import src...` resolves to
# the in-repo package even if a similarly-named package is installed
# system-wide.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
