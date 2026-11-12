import csv
import json
import pickle

import numpy as np
import pytest

from alphalens import agreement, contacts, data

N = 6
CHAINS = ['A', 'A', 'A', 'B', 'B', 'B']


def contact_matrix(p04, p15=0.0):
    '''Symmetric P with inter-chain contacts (0, 4) and (1, 5).'''
    P = np.zeros((N, N), np.float32)
    P[0, 4] = P[4, 0] = p04
    P[1, 5] = P[5, 1] = p15
    return P


def write_json(path, P):
    path.write_text(json.dumps({'pae': np.full((N, N), 5.0).tolist(),
                                'contact_probs': P.tolist()}))


def make_af3(tmp_path, seeds, distogram_seeds=()):
    '''AF3 output dir; seeds: {seed: (score, P)}. Samples of a seed share contact_probs.'''
    d = tmp_path / 'af3'
    d.mkdir()
    lines = ['seed,sample,ranking_score']
    for seed, (score, P) in seeds.items():
        for k in range(2):
            sd = d / f'seed-{seed}_sample-{k}'
            sd.mkdir()
            (sd / 'model.cif').write_text('')
            write_json(sd / 'confidences.json', P)
            lines.append(f'{seed},{k},{score - 0.01 * k}')
    (d / 'ranking_scores.csv').write_text('\n'.join(lines) + '\n')
    best = max(seeds, key=lambda s: seeds[s][0])
    (d / 'job_model.cif').write_text('')
    write_json(d / 'job_confidences.json', seeds[best][1])
    for seed in distogram_seeds:
        dd = d / f'seed-{seed}_distogram'
        dd.mkdir()
        np.savez_compressed(dd / f'job_seed-{seed}_distogram.npz',
                            distogram=logits_for(seeds[seed][1]).astype(np.float16))
    return d


def logits_for(P):
    '''64-bin logits whose P(d < 8 Å) equals P: mass split between bin 0 and the last bin.'''
    probs = np.zeros((N, N, 64), np.float32)
    probs[..., 0] = P
    probs[..., -1] = 1 - P
    return np.log(np.maximum(probs, 1e-9))


def test_distogram_contact_matches_metrics():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(9, 9, 64)).astype(np.float32)
    x = x + x.transpose(1, 0, 2)
    for cutoff in (8.0, 12.0):
        a = data.distogram_contact(x, data.AF_BIN_EDGES, True, cutoff, chunk=4)
        b = data.distogram_metrics(x, data.AF_BIN_EDGES, True, cutoff)['contact']
        np.testing.assert_allclose(a, b, rtol=1e-6)


def test_af2_run_labels():
    names = [f'model_{i}_multimer_v3_pred_0' for i in (3, 1, 2)]
    assert data._af2_run_labels(names) == ['model 3', 'model 1', 'model 2']
    names = ['model_1_multimer_v3_pred_0', 'model_1_multimer_v3_pred_1']
    assert data._af2_run_labels(names) == ['model 1 pred 0', 'model 1 pred 1']
    assert data._af2_run_labels(['model_2_ptm', 'model_1_ptm']) == ['model 2', 'model 1']
    assert data._af2_run_labels(['ranked_x', 'model_1']) == ['ranked_x', 'model_1']


def test_af3_runs_one_per_seed_best_first(tmp_path):
    d = make_af3(tmp_path, {7: (0.5, contact_matrix(0.3)), 9: (0.8, contact_matrix(0.9)),
                            4: (0.2, contact_matrix(0.1))})
    runs = data.agreement_runs(str(d / 'job_model.cif'))
    assert [r['label'] for r in runs] == ['seed 9', 'seed 7', 'seed 4']
    assert {r['kind'] for r in runs} == {'seed'}
    assert {r['source'] for r in runs} == {'contact_probs'}
    # The shown model's seed uses the confidences file the plot already read
    assert runs[0]['path'] == str(d / 'job_confidences.json')
    runs = data.agreement_runs(str(d / 'seed-7_sample-1' / 'model.cif'))
    assert runs[1]['path'] == str(d / 'seed-7_sample-1' / 'confidences.json')
    assert runs[0]['path'] == str(d / 'seed-9_sample-0' / 'confidences.json')


def test_shortfall_names_missing_seed_folders(tmp_path):
    import shutil
    d = make_af3(tmp_path, {1: (0.9, contact_matrix(0.9)), 2: (0.5, contact_matrix(0.3)),
                            3: (0.1, contact_matrix(0.1))})
    for sd in d.glob('seed-*_sample-*'):        # Only the top model copied
        shutil.rmtree(sd)
    runs = data.agreement_runs(str(d / 'job_model.cif'))
    assert len(runs) == 1
    msg = data.agreement_shortfall(str(d / 'job_model.cif'), runs)
    assert msg.startswith('1 of the 3 seeds have files here') and 'seed-*_sample-*' in msg


def test_agreement_counts_and_chain_scores(tmp_path):
    d = make_af3(tmp_path, {1: (0.9, contact_matrix(0.9, 0.6)), 2: (0.5, contact_matrix(0.3)),
                            3: (0.1, contact_matrix(0.1))})
    runs = data.agreement_runs(str(d / 'job_model.cif'))
    agr, msgs = agreement.load(runs, 8.0, N)
    assert msgs == [] and agr.n == 3 and agr.kind == 'seed'
    assert agr.labels == ['seed 1', 'seed 2', 'seed 3']
    assert agr.counts(0.2)[0, 4] == 2 and agr.counts(0.2)[4, 0] == 2
    assert agr.counts(0.5)[0, 4] == 1 and agr.counts(0.5)[1, 5] == 1
    assert agr.counts(0.05)[0, 1] == 0
    np.testing.assert_allclose(agr.pair_values(0, 4), [0.9, 0.3, 0.1], atol=1e-3)
    scores = agr.chain_pair_scores(CHAINS, 0.5)
    means, hits = scores[('A', 'B')]
    # 3 x 3 inter-chain block, so the top-10 mean is the mean of all 9 values
    np.testing.assert_allclose(means, [1.5 / 9, 0.3 / 9, 0.1 / 9], atol=1e-3)
    assert hits == 1
    assert agreement.load(runs[:1], 8.0, N) == (None, [])


def test_rows_size_mismatch_and_known(tmp_path):
    d = make_af3(tmp_path, {1: (0.9, contact_matrix(0.9)), 2: (0.5, contact_matrix(0.3))})
    runs = data.agreement_runs(str(d / 'job_model.cif'))
    rows = [0, 1, 2, 3, 4]                              # Residue 5 not in the structure
    agr, _ = agreement.load(runs, 8.0, N, rows=rows)
    assert agr.stack.shape == (2, 5, 5) and agr.counts(0.2)[0, 4] == 2
    known = {runs[1]['path']: np.ones((5, 5), np.float32)}
    agr, _ = agreement.load(runs, 8.0, N, rows=rows, known=known)
    assert agr.pair_values(0, 1)[1] == 1.0              # Taken from known, not the file
    agr, msgs = agreement.load(runs, 8.0, N + 1)
    assert agr is None and len(msgs) == 2 and 'tokens' in msgs[0]


def test_cutoff_needs_distogram(tmp_path):
    d = make_af3(tmp_path, {1: (0.9, contact_matrix(0.9)), 2: (0.5, contact_matrix(0.3)),
                            3: (0.1, contact_matrix(0.6))}, distogram_seeds=(1, 3))
    runs = data.agreement_runs(str(d / 'job_model.cif'))
    assert [r['source'] for r in runs] == ['distogram', 'contact_probs', 'distogram']
    agr, msgs = agreement.load(runs, 8.0, N)
    np.testing.assert_allclose(agr.pair_values(0, 4), [0.9, 0.3, 0.6], atol=2e-3)
    agr, msgs = agreement.load(runs, 10.0, N)          # contact_probs are fixed at 8 Å
    assert agr.labels == ['seed 1', 'seed 3'] and 'fixed at 8' in msgs[0]


def test_af2_models_with_and_without_distogram(tmp_path):
    d = tmp_path / 'af2'
    d.mkdir()
    order = ['model_2_multimer_v3_pred_0', 'model_1_multimer_v3_pred_0',
             'model_3_multimer_v3_pred_0']
    (d / 'ranking_debug.json').write_text(json.dumps({'order': order}))
    (d / 'ranked_0.pdb').write_text('')
    for name, p in zip(order, (0.9, 0.4, None)):
        result = {'predicted_aligned_error': np.zeros((N, N), np.float32)}
        if p is not None:
            result = {'distogram': {'logits': logits_for(contact_matrix(p)),
                                    'bin_edges': data.AF_BIN_EDGES}, **result}
        with open(d / f'result_{name}.pkl', 'wb') as f:
            pickle.dump(result, f)
    runs = data.agreement_runs(str(d / 'ranked_0.pdb'))
    assert [r['label'] for r in runs] == ['model 2', 'model 1', 'model 3']
    assert {r['kind'] for r in runs} == {'model'}
    agr, msgs = agreement.load(runs, 8.0, N)
    assert agr.labels == ['model 2', 'model 1'] and 'no distogram' in msgs[0]
    np.testing.assert_allclose(agr.pair_values(4, 0), [0.9, 0.4], atol=2e-3)


def test_memory_limit(tmp_path, monkeypatch):
    d = make_af3(tmp_path, {1: (0.9, contact_matrix(0.9)), 2: (0.5, contact_matrix(0.3))})
    runs = data.agreement_runs(str(d / 'job_model.cif'))
    monkeypatch.setattr(agreement, 'MAX_BYTES', 100)
    with pytest.raises(ValueError, match='GB'):
        agreement.load(runs, 8.0, N)


class FakePAE:
    '''What contacts.py reads from an AlphaLensPAE.'''

    def __init__(self, P, agr=None):
        from chimerax.atomic import Residue
        self.metrics = {'contact': P, 'cutoff': 8.0}
        self.pae_matrix = np.arange(N * N, dtype=np.float32).reshape(N, N)
        self._rows = [Residue(c, 10 + i, 'ALA', np.array([3.0 * i, 0, 0]))
                      for i, c in enumerate(CHAINS)]
        self.agreement = agr

    has_contact = True

    def row_residues_or_atoms(self):
        return self._rows

    def row_residue_or_atom_name(self, i):
        r = self._rows[i]
        return f'/{r.chain_id}:{r.number}'


def test_export_pairs_and_csv(tmp_path):
    d = make_af3(tmp_path, {1: (0.9, contact_matrix(0.9, 0.1)), 2: (0.5, contact_matrix(0.3, 0.6))})
    agr, _ = agreement.load(data.agreement_runs(str(d / 'job_model.cif')), 8.0, N)
    P = contact_matrix(0.9, 0.1)
    P[0, 1] = P[1, 0] = 0.99                           # Intra-chain: never exported
    pae = FakePAE(P)
    assert contacts.export_pairs(pae, 0.2).tolist() == [[0, 4]]
    pae.agreement = agr
    # (1, 5) is a contact only in seed 2; with agreement it is exported too
    assert contacts.export_pairs(pae, 0.2, agreement=agr).tolist() == [[0, 4], [1, 5]]
    assert contacts.export_pairs(pae, 0.2, region=(4, 5, 0, 0), agreement=agr).tolist() == [[0, 4]]
    path = tmp_path / 'c.csv'
    n = contacts.write_contacts_csv(pae, str(path), 0.2, agreement=agr)
    rows = list(csv.DictReader(open(path)))
    assert n == 2 and len(rows) == 2
    r = rows[0]
    assert (r['chain_1'], r['residue_1'], r['chain_2'], r['residue_2']) == ('A', '10', 'B', '14')
    assert r['p_contact'] == '0.900' and r['pae_1_2'] == '4.00' and r['pae_2_1'] == '24.00'
    assert r['model_distance'] == '12.00' and r['expected_distance'] == ''
    assert r['seeds_with_p_ge_0.20'] == '2' and r['seeds'] == '2'
    assert r['p_seed_1'] == '0.900' and r['p_seed_2'] == '0.300'
    assert rows[1]['seeds_with_p_ge_0.20'] == '1' and rows[1]['p_contact'] == '0.100'
