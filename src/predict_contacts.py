"""Compare and export contact probabilities for exactly the selected residue pairs."""
import csv
from pathlib import Path
import numpy as np


def region_pairs(chains, region=None):
    """Unique inter-chain output-row pairs inside the original plot rectangle."""
    rows = [r for chain in chains for r in chain['residue_map']]
    core = {i for chain in chains for i in chain['core_rows']}
    if region is None:
        left = right = [i for i, r in enumerate(rows) if r['source_row'] in core]
    else:
        y0, y1, x0, x1 = region
        left = [i for i, r in enumerate(rows) if y0 <= r['source_row'] <= y1]
        right = [i for i, r in enumerate(rows) if x0 <= r['source_row'] <= x1]
    if not left or not right:
        return np.empty((0, 2), dtype=int)
    i, j = np.meshgrid(left, right, indexing='ij')
    ids = np.array([r['chain_id'] for r in rows])
    keep = ids[i] != ids[j]
    pairs = np.stack([np.minimum(i[keep], j[keep]), np.maximum(i[keep], j[keep])], axis=1)
    return np.unique(pairs, axis=0)


def measure(pae, pairs):
    from .contacts import distogram_atom
    residues = pae.row_residues_or_atoms()
    coordinates = []
    for residue in residues:
        atom = distogram_atom(residue)
        coordinates.append(atom.coord if atom is not None else [np.nan]*3)
    xyz = np.asarray(coordinates)
    i, j = pairs.T
    distance = np.linalg.norm(xyz[i]-xyz[j], axis=1)
    probability = np.maximum(pae.metrics['contact'][i, j], pae.metrics['contact'][j, i]) if pae.has_contact else None
    nullable = lambda values: [float(v) if np.isfinite(v) else None for v in values]
    return dict(probability=nullable(probability) if probability is not None else None,
                cutoff_A=float(pae.metrics['cutoff']) if pae.has_contact else None,
                distance_A=nullable(distance))


def stats(values, threshold):
    probabilities = values.get('probability')
    valid = sorted((p for p in probabilities if p is not None), reverse=True) if probabilities is not None else []
    return dict(count=sum(p >= threshold for p in valid) if valid else None,
                top10_mean=float(np.mean(valid[:10])) if valid else None,
                cutoff_A=values.get('cutoff_A'),
                structural_count=(sum(d is not None and d < 8 for d in values['distance_A'])
                                  if any(d is not None for d in values['distance_A']) else None))


def export(directory, chains, pairs, predicted, source=None, threshold=.5):
    """CSV contains the union of geometric and probabilistic contacts, before/after."""
    rows = [r for c in chains for r in c['residue_map']]
    n = len(pairs)
    source = source or dict(probability=None, cutoff_A=None, distance_A=[None]*n)
    if len(predicted['distance_A']) != n or len(source['distance_A']) != n:
        raise ValueError('Contact snapshot does not match the selected region.')
    old_p = source.get('probability') or [None]*n
    new_p = predicted.get('probability') or [None]*n
    if len(old_p) != n or len(new_p) != n:
        raise ValueError('Contact probabilities do not match the selected region.')
    fields = ['chain_1', 'residue_1', 'chain_2', 'residue_2', 'source_probability',
              'source_cutoff_A', 'prediction_probability', 'prediction_cutoff_A',
              'source_distance_A', 'prediction_distance_A']
    path = Path(directory) / 'region_contacts.csv'
    strongest = []
    count = 0
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for k, (i, j) in enumerate(pairs):
            a, b = rows[i], rows[j]
            an = str(a['number']) + a.get('insertion_code', '')
            bn = str(b['number']) + b.get('insertion_code', '')
            if new_p[k] is not None:
                strongest.append((new_p[k], f'{a["chain_id"]}:{an}–{b["chain_id"]}:{bn}'))
            old_d, new_d = source['distance_A'][k], predicted['distance_A'][k]
            if not (any(p is not None and p >= threshold for p in (old_p[k], new_p[k]))
                    or any(d is not None and d < 8 for d in (old_d, new_d))):
                continue
            writer.writerow(dict(zip(fields, [a['chain_id'], an, b['chain_id'], bn, old_p[k],
                source.get('cutoff_A'), new_p[k], predicted.get('cutoff_A'), old_d, new_d])))
            count += 1
    strongest.sort(key=lambda item: -item[0])
    return dict(source=stats(source, threshold), prediction=stats(predicted, threshold),
                threshold=threshold, pairs_evaluated=n, csv_rows=count, csv=str(path),
                strongest=[dict(pair=label, probability=float(p)) for p, label in strongest[:10]])


def summary(report):
    def describe(values):
        if values['count'] is None:
            return 'unavailable'
        return f'{values["count"]} (d < {values["cutoff_A"]:g} Å)'
    before, after = report['source'], report['prediction']
    result = f'Contacts P ≥ {report["threshold"]:g}: {describe(before)} → {describe(after)}\n'
    fmt = lambda x: f'{x:.3f}' if x is not None else 'unavailable'
    result += f'Top-10 mean contact P: {fmt(before["top10_mean"])} → {fmt(after["top10_mean"])}\n'
    count = lambda v: str(v) if v is not None else 'unavailable'
    result += f'Model contacts (CB/CA < 8 Å): {count(before["structural_count"])} → {count(after["structural_count"])}\n'
    if report['strongest']:
        result += 'Strongest: ' + ', '.join(f'{v["pair"]} {v["probability"]:.3f}' for v in report['strongest'][:5]) + '\n'
    result += f'All region contacts: {report["csv"]}'
    return result
