from __future__ import annotations

import numpy as _np


_ALIASES = {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "str": str,
    "unicode": str,
}

for _name, _value in _ALIASES.items():
    try:
        getattr(_np, _name)
    except AttributeError:
        setattr(_np, _name, _value)
