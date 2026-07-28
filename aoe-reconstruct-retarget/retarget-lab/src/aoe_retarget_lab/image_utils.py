from __future__ import annotations

import io

import numpy as np
from PIL import Image


def decode_rgb(blob: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"))
