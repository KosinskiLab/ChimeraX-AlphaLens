"""Catch gross coordinate failures independently of the model's confidence scores."""
import numpy as np


def check_ca_geometry(coordinates, chain_ids):
    xyz = np.asarray(coordinates, dtype=float)
    ids = np.asarray(chain_ids)
    n = len(xyz)
    close_pairs = 0
    # Bound temporary memory for large complexes; exclude the diagonal and count once.
    for start in range(0, n, 256):
        delta = xyz[start:start+256, None, :] - xyz[None, :, :]
        distance2 = np.einsum('ijk,ijk->ij', delta, delta)
        upper = np.arange(n)[None, :] > np.arange(start, min(start+256, n))[:, None]
        close_pairs += int(np.count_nonzero((distance2 < 4) & upper))
    adjacent = np.linalg.norm(np.diff(xyz, axis=0), axis=1)[ids[:-1] == ids[1:]]
    broken = int(np.count_nonzero((adjacent < 2.5) | (adjacent > 4.5)))
    if close_pairs > max(5, n // 10) or broken > max(3, len(adjacent) // 10):
        raise ValueError(f'Prediction failed geometry checks: {close_pairs} Cα pairs closer than 2 Å; '
                         f'{broken}/{len(adjacent)} adjacent Cα distances outside 2.5–4.5 Å. '
                         'The coordinates may be collapsed or broken. Output files are retained.')
    return dict(ca_pairs_under_2A=close_pairs, adjacent_ca_outliers=broken,
                adjacent_ca_median=float(np.median(adjacent)) if len(adjacent) else None)
