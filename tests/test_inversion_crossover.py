import numpy as np
from ex2.Ex2 import MagicSquareProblem


def is_valid_permutation(square: np.ndarray) -> bool:
    flat = square.flatten()
    return np.array_equal(np.sort(flat), np.arange(1, flat.size + 1))


def test_crossover_produces_permutation():
    size = 5
    parent1 = MagicSquareProblem(size)
    parent2 = MagicSquareProblem(size)

    child = parent1.crossover(parent2)
    assert is_valid_permutation(child.square)


def test_multiple_crossovers():
    size = 4
    p1 = MagicSquareProblem(size)
    p2 = MagicSquareProblem(size)
    for _ in range(10):
        child = p1.crossover(p2, crossover_points=3)
        assert is_valid_permutation(child.square)
