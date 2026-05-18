import json
import numpy as np


def parse_histogram(path):
    charges, counts = [], []
    with open(path) as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    charges.append(float(parts[0]))
                    counts.append(float(parts[1]))
                except ValueError:
                    continue
    return np.array(charges), np.array(counts)


def trim_histogram(charges, counts, pad=20):
    nz = np.where(counts > 0)[0]
    if len(nz) == 0:
        return charges, counts
    lo = max(0, nz[0] - pad)
    hi = min(len(counts) - 1, nz[-1] + pad)
    return charges[lo : hi + 1], counts[lo : hi + 1]


def load_initial_values(path):
    """Load optional initial values from JSON.

    Accepted forms:
      * {"theta": [...]} — full raw optimiser vector.
      * A previous fit result JSON with nested "ped", "spe", "lam", "dark_count" fields.
      * A flat mapping with raw parameter names or physical names.
    """
    if path is None:
        return None
    with open(path) as fh:
        return json.load(fh)
