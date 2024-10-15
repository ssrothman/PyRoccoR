"""
Copyright (c) 2024 Simon Rothman. All rights reserved.

PyRoccoR: Numpy ufuncs for Rochester corrections
"""

from __future__ import annotations

from ._version import version as __version__

from ._roccor import RoccoR
from ._ufunc import kScaleDT, kSmearMC, kSpreadMC

__all__ = ["__version__", "RoccoR"]

