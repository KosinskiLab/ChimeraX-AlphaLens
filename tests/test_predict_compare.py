import math

import numpy as np
import pytest
from alphalens import predict_compare as pc


def two_chains(gap=4.5, n=6):
    '''Two parallel strands, chain B gap Å from chain A: residues {atom: xyz} and chain ids.'''
    residues, chains = [], []
    for c, y in (('A', 0.0), ('B', gap)):
        for k in range(n):
            x = 3.8 * k
            residues.append({'N': [x - 1.2, y, 0.3], 'CA': [x, y, 0.0], 'C': [x + 1.2, y, -0.3],
                             'O': [x + 1.4, y + (0.9 if c == 'A' else -0.9), -0.4],
                             'CB': [x, y + (0.8 if c == 'A' else -0.8), 1.0]})
            chains.append(c)
    return residues, chains


def moved(residues, chains, which, shift=(0, 0, 0), rotation=None):
    out = []
    r = np.eye(3) if rotation is None else rotation
    for res, c in zip(residues, chains):
        if which is None or c == which:
            out.append({k: list(r @ np.asarray(v) + np.asarray(shift)) for k, v in res.items()})
        else:
            out.append(dict(res))
    return out


def test_identical_interfaces_score_one():
    native, chains = two_chains()
    (p,), empty = pc.dockq(native, native, chains)
    assert empty == []
    assert p['dockq'] == 1.0 and p['fnat'] == 1.0 and p['irmsd'] == 0 and p['lrmsd'] == 0
    assert p['kept'] == p['native_contacts'] > 0 and p['new_contacts'] == 0
    assert (p['class'], p['words']) == ('high', 'same interface')


def test_a_rigid_motion_of_the_whole_model_changes_nothing():
    native, chains = two_chains()
    a = math.radians(40)
    rot = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    model = moved(native, chains, None, shift=(10, -3, 7), rotation=rot)
    (p,), _ = pc.dockq(native, model, chains)
    assert p['dockq'] == pytest.approx(1.0, abs=1e-3) and p['lrmsd'] == pytest.approx(0, abs=1e-2)


def test_a_ligand_pushed_away_loses_contacts_and_follows_the_formula():
    native, chains = two_chains()
    model = moved(native, chains, 'B', shift=(0, 3.0, 0))     # B 3 Å farther from A
    (p,), _ = pc.dockq(native, model, chains)
    assert p['receptor'] == 'A' and p['ligand'] == 'B'        # Equal sizes: the first is the receptor
    assert p['lrmsd'] == pytest.approx(3.0, abs=0.01)
    assert p['fnat'] < 1 and p['kept'] < p['native_contacts']
    expected = (p['fnat'] + 1 / (1 + (p['irmsd'] / 1.5) ** 2) + 1 / (1 + (p['lrmsd'] / 8.5) ** 2)) / 3
    assert p['dockq'] == pytest.approx(expected, abs=1e-3)
    assert 0 < p['irmsd'] < 3.0                               # The interface superposition halves it


def test_pairs_without_source_contacts_are_listed_not_scored():
    native, chains = two_chains(gap=30.0)
    pairs, empty = pc.dockq(native, native, chains)
    assert pairs == [] and empty == ['A–B']
    assert 'nothing to compare' in pc.summary_lines(dict(pairs=[], without_source_contacts=empty))[0]


def test_only_atoms_in_both_models_count():
    native, chains = two_chains()
    model = [dict(r) for r in native]
    for r in model:
        r.pop('CB')                                           # The re-prediction lacks CB: ignored
        r['CG'] = [99.0, 99.0, 99.0]                          # An atom the source lacks: ignored
    (p,), _ = pc.dockq(native, model, chains)
    assert p['dockq'] == 1.0


@pytest.mark.parametrize('score, name', [(0.95, 'high'), (0.80, 'high'), (0.6, 'medium'), (0.3, 'acceptable'),
                                         (0.1, 'incorrect')])
def test_capri_classes(score, name):
    assert pc.dockq_class(score)[0] == name


def test_summary_lines_name_contacts_and_distances():
    native, chains = two_chains()
    model = moved(native, chains, 'B', shift=(0, 3.0, 0))
    pairs, empty = pc.dockq(native, model, chains)
    lines = pc.summary_lines(dict(pairs=pairs, without_source_contacts=empty, dockq=pairs[0]['dockq']))
    assert lines[0].startswith('Interface vs source: DockQ')
    assert 'source contacts kept' in lines[1] and 'B 3.0 Å from its source place with A superposed' in lines[1]
