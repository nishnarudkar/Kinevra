"""Shared in-process types for the vision layer (not serialised, so not in schemas.py)."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.uint8]  # BGR (OpenCV order) unless a function says otherwise


@dataclass(frozen=True, slots=True)
class Frame:
    idx: int  # index of the frame in the source (decoded order), stable for seeking
    t: float  # seconds since the source started; strictly increasing
    image: Image
