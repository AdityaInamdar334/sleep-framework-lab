"""Concept-only helpers for paper discussion; not Google's LTI or RL trainer."""

def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance in Unicode codepoints, not model tokens."""
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1,
                               previous[j] + 1,
                               previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def imitation_reward(candidate: str, reference: str,
                     semantic_match: bool, gamma: float = 0.5,
                     max_normalized_distance: float = 0.5) -> float:
    """Illustrative mixture: human-supplied semantic bit + capped edit similarity.
    Paper uses a learned semantic reward and sequence-level Levenshtein threshold.
    This code demonstrates the design logic, not the exact training algorithm.
    """
    if not 0 <= gamma <= 1:
        raise ValueError('gamma out of range')
    length = max(len(candidate), len(reference), 1)
    normalized = edit_distance(candidate, reference) / length
    absolute = (1 - normalized) if normalized <= max_normalized_distance else 0.0
    return gamma * float(semantic_match) + (1 - gamma) * absolute
