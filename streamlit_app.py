"""Repo-root entrypoint for Streamlit Community Cloud (zero-config default).

The real app is src/app.py; it puts src/ on sys.path itself, so this just runs it.
"""

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).parent / "src" / "app.py"), run_name="__main__")
