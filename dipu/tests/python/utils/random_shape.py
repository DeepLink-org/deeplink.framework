from typing import List, Tuple
import numpy as np


__all__ = ["ShapeGenerator"]


class ShapeGenerator:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def random_shape(
        self,
        numel_range: Tuple[int, int],
        rank_range: Tuple[int, int],
        retry: int = 10,
    ) -> List[int]:
        """
        Generate a random shape. Ranges are inclusive.
        """
        assert 0 < numel_range[0] <= numel_range[1]
        assert 0 < rank_range[0] <= rank_range[1]
        assert retry > 0
        rank = self.rng.integers(rank_range[0], rank_range[1], endpoint=True)
        while True:
            retry -= 1
            shape = self._try_random_shape(numel_range, rank)
            if retry <= 0 or shape.prod() in range(numel_range[0], numel_range[1] + 1):
                return shape.tolist()

    def _try_random_shape(self, numel_range: Tuple[int, int], rank: int) -> np.ndarray:
        assert 0 < numel_range[0] <= numel_range[1]
        lognumel_range = np.log(numel_range)
        lognumel = self.rng.uniform(*lognumel_range)
        logshape = self._random_partition(lognumel, rank)
        shape = np.exp(logshape).round().astype(int)
        return shape

    def _random_partition(self, total: float, part: int) -> np.ndarray:
        """
        Randomly partition a total into part parts.
        """
        assert total > 0
        assert part > 0
        parts = self.rng.random(part)
        parts /= parts.sum()
        return total * parts
