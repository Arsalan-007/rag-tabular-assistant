"""
Vector-store packaging.

The prebuilt Chroma store is shipped as an immutable tarball (`data/chroma.tar.gz`,
tracked with Git LFS) rather than as a live directory, because Chroma writes
internal bookkeeping to its SQLite file on every query -- a committed live
directory would show as "modified" after any read.

At runtime:
  * `ensure_store()` extracts the tarball into `data/chroma/` (gitignored) the
    first time it's needed. A fresh clone, CI, and Streamlit Cloud all get a
    working store in ~1 s with no rebuild.
  * `pack_store()` re-creates the tarball from `data/chroma/` after `make ingest`.
"""

from __future__ import annotations

import tarfile

from config import CHROMA_DIR, DATA_DIR

TARBALL = DATA_DIR / "chroma.tar.gz"


def store_is_materialised() -> bool:
    return (CHROMA_DIR / "chroma.sqlite3").exists()


def ensure_store() -> None:
    """Extract data/chroma.tar.gz into data/chroma/ if the store isn't there yet."""
    if store_is_materialised():
        return
    if not TARBALL.exists():
        raise FileNotFoundError(
            f"No vector store at {CHROMA_DIR} and no tarball at {TARBALL}. "
            "Run `make ingest` to build it (needs network for the arXiv PDFs)."
        )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(TARBALL, "r:gz") as tf:
        tf.extractall(DATA_DIR)  # noqa: S202 -- our own artifact
    if not store_is_materialised():
        raise RuntimeError(f"{TARBALL} did not contain a 'chroma/' directory")


def pack_store() -> None:
    """(Re)create the shipped tarball from the current data/chroma/ directory."""
    if not store_is_materialised():
        raise FileNotFoundError(f"nothing to pack: {CHROMA_DIR} is empty")
    with tarfile.open(TARBALL, "w:gz") as tf:
        tf.add(CHROMA_DIR, arcname="chroma")
    print(f"[store] packed {CHROMA_DIR} -> {TARBALL} "
          f"({TARBALL.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "pack":
        pack_store()
    else:
        ensure_store()
        print(f"[store] materialised at {CHROMA_DIR}")
