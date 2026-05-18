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


def roi_lower_sigma_for_voltage(voltage):
    """Voltage-dependent lower fit boundary in pedestal sigma units."""
    v = float(voltage)
    if 53.0 <= v < 54.0:
        return 2.5
    if 54.0 <= v < 55.0:
        return 3.0
    return 3.5


def select_fit_roi(charges, counts, ped_mean, ped_sigma, lower_sigma=3.5):
    """Select the likelihood ROI [Q0 - lower_sigma*sigma0, max_Q]."""
    lower = float(ped_mean) - float(lower_sigma) * float(ped_sigma)
    mask = np.asarray(charges) >= lower
    if not np.any(mask):
        return charges, counts, lower
    return charges[mask], counts[mask], lower


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
