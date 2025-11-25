import itertools
from typing import Dict, Generator, Iterable, List, Tuple

from river.datasets import synth


def generate_sea_stream_with_drifts(
    n_samples: int = 5000,
    drift_points: List[int] = None,
    seed: int = 42,
    noise: float = 0.0,
) -> Generator[Tuple[Dict[str, float], int, int], None, None]:
    """
    Yields a stream of (x, y, concept_idx) tuples from SEA with abrupt concept drifts.

    - Four concepts are used by default corresponding to SEA variants [0, 1, 2, 3].
    - Drifts occur at the provided drift_points indices (0-based, strictly increasing).
    - After the last drift, the final concept continues until n_samples.
    """
    if drift_points is None:
        drift_points = [1250, 2500, 3750]

    if any(dp <= 0 for dp in drift_points):
        raise ValueError("drift_points must be positive indices (0 < dp < n_samples)")
    if sorted(drift_points) != drift_points:
        raise ValueError("drift_points must be strictly increasing")
    if drift_points[-1] >= n_samples:
        raise ValueError("last drift point must be < n_samples")

    # Define up to 4 SEA concepts (variants 0..3). If more segments are requested,
    # the last available concept is repeated.
    variants = [0, 1, 2, 3]
    max_concepts = len(variants)

    # Build concept schedule boundaries (segments)
    boundaries = [0] + drift_points + [n_samples]
    segments = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

    # Create iterators (generators) for each SEA variant with different seeds
    sea_iters: List[Iterable[Tuple[Dict[str, float], int]]] = []
    for i, v in enumerate(variants):
        # Derive distinct seeds for reproducibility across variants
        it = iter(synth.SEA(variant=v, seed=seed + 100 * i, noise=noise))
        sea_iters.append(it)

    # Produce samples according to the schedule
    for seg_idx, (start, end) in enumerate(segments):
        concept_idx = min(seg_idx, max_concepts - 1)
        current_iter = sea_iters[concept_idx]
        for _ in range(start, end):
            x, y = next(current_iter)
            yield x, y, concept_idx


def make_sea_arrays_with_drifts(
    n_samples: int = 5000,
    drift_points: List[int] = None,
    seed: int = 42,
    noise: float = 0.0,
) -> Tuple[List[List[float]], List[int], List[int], List[str]]:
    """
    Materializes the SEA stream with drifts into X, y arrays suitable for scikit-learn,
    preserving feature order.
    Returns: X (list of feature lists), y (list of labels), concepts (per-sample concept index), feature_names (ordered)
    """
    X: List[List[float]] = []
    y: List[int] = []
    concepts: List[int] = []
    feature_names: List[str] = []

    stream = generate_sea_stream_with_drifts(
        n_samples=n_samples, drift_points=drift_points, seed=seed, noise=noise
    )
    first_x, first_y, first_concept = next(stream)
    feature_names = sorted(first_x.keys())
    X.append([first_x[name] for name in feature_names])
    y.append(first_y)
    concepts.append(first_concept)

    for x_i, y_i, c_i in stream:
        X.append([x_i[name] for name in feature_names])
        y.append(y_i)
        concepts.append(c_i)

    return X, y, concepts, feature_names

