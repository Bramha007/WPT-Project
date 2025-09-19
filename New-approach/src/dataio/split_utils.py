import random
from typing import List, Tuple

def subsample_pairs(
    pairs: List[Tuple[str, str]],
    fraction: float,
    seed: int = 42,
    max_items: int | None = None,
) -> List[Tuple[str, str]]:
    """Deterministically pick a fraction (and optional cap) of (img, xml) pairs."""
    if fraction >= 1.0 and not max_items:
        return pairs
    rng = random.Random(seed)
    idxs = list(range(len(pairs)))
    rng.shuffle(idxs)
    take = int(len(pairs) * max(0.0, min(1.0, fraction)))
    if max_items is not None:
        take = min(take if take > 0 else len(pairs), max_items)
    chosen = idxs[:take] if take > 0 else idxs
    return [pairs[i] for i in chosen]
