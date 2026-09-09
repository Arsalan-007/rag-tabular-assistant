"""Shared fixtures.

Several tests flip `config.settings` (retrieval_mode, rerank, ...) to exercise a
pipeline variant. Snapshot and restore the whole settings object around every
test so that ordering never matters.
"""

from __future__ import annotations

import pytest

import config


@pytest.fixture(autouse=True)
def _restore_settings():
    before = config.settings.model_dump()
    yield
    for key, val in before.items():
        setattr(config.settings, key, val)
