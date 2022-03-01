from typing import List, Optional, Tuple
import numpy as np


class Tensor:
    sparse: bool = False
    sparsity: Optional[float] = None
    centered: bool = False
    normalized: bool = False
    range: Optional[Tuple[float, float]] = None

    shape: List[Optional[int]]
    dtype: np.dtype = np.float32


class dtype:
    pass


class Nominal(dtype):
    range: Tuple[Optional[int], Optional[int]]


class Ordinal(dtype):
    d: int | str


class Interval(dtype):
    pass


class Rational(dtype):
    pass
