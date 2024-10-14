from __future__ import annotations

import importlib.metadata

import pyroccor as m


def test_version():
    assert importlib.metadata.version("pyroccor") == m.__version__
