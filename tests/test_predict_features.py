"""Precomputed AlphaPulldown / AlphaFold features as prediction alignments (no ChimeraX needed)."""

import json
import lzma
import pickle
import subprocess
import sys
import time
import types

import numpy as np
import pytest

from alphalens import predict_features as pf
from alphalens.predict import PredictionJob

HH = {c: i for i, c in enumerate(pf.HHBLITS)}


def codes(rows):
    return np.array([[HH[c] for c in r] for r in rows], np.int32)


def feature_dict(seq, rows, deletions=None, all_rows=None, species=None):
    fd = dict(sequence=np.array([seq.encode()], object), msa=codes([seq] + rows),
              deletion_matrix_int=np.array(deletions if deletions is not None
                                           else np.zeros((len(rows) + 1, len(seq))), np.int32))
    if all_rows is not None:
        fd['msa_all_seq'] = codes([seq] + all_rows)
        fd['deletion_matrix_int_all_seq'] = np.zeros((len(all_rows) + 1, len(seq)), np.int32)
        fd['msa_species_identifiers_all_seq'] = np.array([b''] + [s.encode() for s in species], object)
    return fd


def monomeric_object(seq, fd, description='X'):
    '''A pickle as AlphaPulldown writes it: alphapulldown.objects.MonomericObject with a runner.'''
    mod = types.ModuleType('alphapulldown.objects')
    jack = types.ModuleType('alphafold.data.tools.jackhmmer')

    class MonomericObject:
        pass

    class Jackhmmer:
        pass
    MonomericObject.__module__, Jackhmmer.__module__ = mod.__name__, jack.__name__
    MonomericObject.__qualname__, Jackhmmer.__qualname__ = 'MonomericObject', 'Jackhmmer'
    mod.MonomericObject, jack.Jackhmmer = MonomericObject, Jackhmmer
    saved = {k: sys.modules.get(k) for k in (mod.__name__, jack.__name__)}
    sys.modules[mod.__name__], sys.modules[jack.__name__] = mod, jack
    try:
        obj = MonomericObject()
        obj.description, obj.sequence, obj.feature_dict = description, seq, fd
        obj._uniprot_runner = Jackhmmer()
        obj._uniprot_runner.binary_path = '/opt/conda/bin/jackhmmer'
        return pickle.dumps(obj, protocol=4)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_reads_alphapulldown_object_and_plain_dict(tmp_path):
    fd = feature_dict('MKVL', ['MKIL', '-KVL'], all_rows=['MRVL'], species=['ECOLI'])
    (tmp_path / 'P12345.pkl.xz').write_bytes(lzma.compress(monomeric_object('MKVL', fd, 'P12345')))
    (tmp_path / 'plain.pkl').write_bytes(pickle.dumps(fd))
    for name in ('P12345.pkl.xz', 'plain.pkl'):
        f = pf.load_af2(tmp_path / name)
        assert f['sequence'] == 'MKVL' and pf.decode(f['msa']) == ['MKVL', 'MKIL', '-KVL']
        assert pf.decode(f['all_msa']) == ['MKVL', 'MRVL'] and f['all_species'] == ['', 'ECOLI']
        assert len(f['sha256']) == 64


def test_refuses_other_code_whole_complexes_and_damage(tmp_path):
    import os
    (tmp_path / 'evil.pkl').write_bytes(pickle.dumps(os.getcwd))
    with pytest.raises(ValueError, match='posix.getcwd|does not read'):
        pf.load_af2(tmp_path / 'evil.pkl')
    fd = feature_dict('MKVL', ['MKIL'])
    fd['asym_id'] = np.array([1, 1, 2, 2])
    (tmp_path / 'complex.pkl').write_bytes(pickle.dumps(fd))
    with pytest.raises(ValueError, match='whole complex'):
        pf.load_af2(tmp_path / 'complex.pkl')
    data = bytearray(lzma.compress(pickle.dumps(feature_dict('MKVL', []))))
    data[len(data) // 2] ^= 0xFF
    (tmp_path / 'broken.pkl.xz').write_bytes(bytes(data))
    with pytest.raises(ValueError, match='damaged'):
        pf.load_af2(tmp_path / 'broken.pkl.xz')


def test_processed_restype_order_is_recoded(tmp_path):
    seq = 'MKVL'
    fd = dict(sequence=np.array([seq.encode()], object),
              msa=np.array([[pf.RESTYPES.index(c) for c in seq]], np.int32),
              deletion_matrix=np.zeros((1, 4)))
    (tmp_path / 'f.pkl').write_bytes(pickle.dumps(fd))
    assert pf.decode(pf.load_af2(tmp_path / 'f.pkl')['msa']) == ['MKVL']


def test_locate():
    assert pf.locate('MKVLAAKVL', 'KVL', hint=6) == 6
    assert pf.locate('MKVLAAW', 'AAW', hint=0) == 4
    with pytest.raises(ValueError, match='2 times'):
        pf.locate('MKVLAAKVL', 'KVL')
    with pytest.raises(ValueError, match='not part'):
        pf.locate('MKVL', 'WWW', name='P1')


def test_a3m_deletions_crop_and_gap_rows():
    f = dict(msa=codes(['MKVLA', 'MRV-A', '--V--', 'M---A']),
             deletions=np.array([[0] * 5, [0, 2, 0, 0, 1], [0] * 5, [0] * 5]))
    text, rows = pf.unpaired_a3m(f, 1, 3, 'A')
    # Columns 2-4: a row with no residue left (the last) is dropped; the insertion before the
    # first kept column is dropped, the one inside is kept
    assert text == '>A\nKVL\n>A_1\nRV-\n>A_2\n-V-\n' and rows == 3
    assert pf.a3m_rows(codes(['MV']), np.array([[0, 3]]), ['h']) == '>h\nMxxxV\n'


def test_species_pairing_follows_alphafold_multimer():
    q1, q2 = 'AAAA', 'CCCC'
    chain1 = (codes([q1, 'AAAG', 'AAGG', 'AGGG', 'AAAA']), ['', 'HUMAN', 'HUMAN', 'YEAST', 'ECOLI'])
    chain2 = (codes([q2, 'CCCD', 'CCDD', 'CDDD']), ['', 'HUMAN', 'MOUSE', 'YEAST'])
    rows = pf.pair_rows([chain1, chain2])
    # Query first; HUMAN (min 1 hit: most similar of chain 1 is row 1), YEAST; ECOLI and MOUSE
    # are in one chain only; ordered by the product of row indices
    assert rows.tolist() == [[0, 0], [1, 1], [3, 3]]
    many = (codes([q1] + ['AAAG'] * 601), [''] + ['RAT'] * 601)
    other = (codes([q2, 'CCCD']), ['', 'RAT'])
    assert pf.pair_rows([many, other]).tolist() == [[0, 0]]          # > 600 hits: skipped
    three = pf.pair_rows([chain1, chain2, (codes(['WWWW', 'WWWY']), ['', 'ECOLI'])])
    assert three.tolist() == [[0, 0, 0], [1, 1, -1], [4, -1, 1], [3, 3, -1]]     # By index product


def test_paired_a3ms_are_row_aligned():
    a = dict(msa=codes(['MKV']), deletions=np.zeros((1, 3), int), all_msa=codes(['MKV', 'MRV']),
             all_deletions=np.zeros((2, 3), int), all_species=['', 'HUMAN'])
    b = dict(msa=codes(['WY']), deletions=np.zeros((1, 2), int), all_msa=codes(['WY', 'WF']),
             all_deletions=np.zeros((2, 2), int), all_species=['', 'HUMAN'])
    c = dict(msa=codes(['GG']), deletions=np.zeros((1, 2), int), all_msa=None, all_deletions=None,
             all_species=None)
    texts, rows = pf.paired_a3ms([a, b, c], [0, 0, 0], [3, 2, 2], ['A', 'B', 'C'])
    assert rows == 1
    assert texts == ['>A\nMKV\n>A_paired_1\nMRV\n', '>B\nWY\n>B_paired_1\nWF\n', '>C\nGG\n>C_paired_1\n--\n']


def test_names_from_paths_and_text():
    assert pf.accessions_in('Q15014+Q9NV56') == ['Q15014', 'Q9NV56']
    assert pf.accessions_in('q15014_and_q9nv56_model') == ['Q15014', 'Q9NV56']
    assert pf.accessions_in('Q15014:385-460+P05067-2') == ['Q15014', 'P05067-2']
    assert pf.accessions_in('seed-2683111181_sample-0 ranked_0') == []
    assert pf.guess_names('/x/Q15014+Q9NV56/ranked_0.cif', ['A', 'B']) == {'A': 'Q15014', 'B': 'Q9NV56'}
    assert pf.guess_names('/x/q15014_and_q9nv56/seed-1_sample-0/model.cif', ['A', 'B']) == \
        {'A': 'Q15014', 'B': 'Q9NV56'}
    assert pf.guess_names('/x/P61626_homo/ranked_0.pdb', ['A', 'B']) == {'A': 'P61626', 'B': 'P61626'}
    assert pf.guess_names('/x/fold/model.cif', ['A', 'B']) == {}
    assert pf.parse_chain_accessions('A: P0A836, B: P0AGE9', ['A', 'B']) == {'A': 'P0A836', 'B': 'P0AGE9'}
    assert pf.parse_chain_accessions('P0A836 P0AGE9', ['A', 'B']) == {'A': 'P0A836', 'B': 'P0AGE9'}
    assert pf.parse_chain_accessions('P61626', ['A', 'B']) == {'A': 'P61626', 'B': 'P61626'}
    with pytest.raises(ValueError, match='no chain C'):
        pf.parse_chain_accessions('C: P1', ['A', 'B'])
    with pytest.raises(ValueError, match='3 feature IDs for 2'):
        pf.parse_chain_accessions('P1 P2 P3', ['A', 'B'])


def lab_database(tmp_path):
    '''The lab layout: alphafold2|alphafold3/<organism>/features.'''
    root = tmp_path / 'input_features'
    af2 = root / 'alphafold2' / 'Escherichia_coli' / 'features'
    af3 = root / 'alphafold3' / 'Bacillus_subtilis' / 'features'
    af2.mkdir(parents=True); af3.mkdir(parents=True)
    a = feature_dict('MKVLAW', ['MKVLAF', 'MRVLAW'], all_rows=['MKVLAY', 'MKVIAW'], species=['HUMAN', 'YEAST'])
    b = feature_dict('GGHHS', ['GGHHT'], all_rows=['GAHHS', 'GGHHA'], species=['YEAST', 'MOUSE'])
    (af2 / 'P11111.pkl.xz').write_bytes(lzma.compress(monomeric_object('MKVLAW', a)))
    (af2 / 'P22222.pkl.xz').write_bytes(lzma.compress(pickle.dumps(b)))
    job = dict(dialect='alphafold3', version=3, name='Q33333', sequences=[dict(protein=dict(
        id='A', sequence='MKTRPK', unpairedMsa='>query\nMKTRPK\n>hit\nMKTxxRPA\n>gap\n--TR--\n',
        pairedMsa='>query\nMKTRPK\n>tr|X|X_BACSU/1-6\nMRTRPK\n', templates=[]))])
    (af3 / 'Q33333_af3_input.json.xz').write_bytes(lzma.compress(json.dumps(job).encode()))
    return root


def test_find_and_choose_features(tmp_path):
    root = lab_database(tmp_path)
    found = pf.find_features(root, 'p11111')
    assert found['af2'].name == 'P11111.pkl.xz' and found['af3'] is None
    assert pf.find_features(root, 'Q33333')['af3'].name == 'Q33333_af3_input.json.xz'
    chains = [dict(chain_id='A', name='P11111', sequence='KVL'), dict(chain_id='B', name='P22222', sequence='GG')]
    chosen = pf.choose_features(root, chains, 'af3')       # Not all have AF3 features: AF2 for all
    assert [c['kind'] for c in chosen] == ['af2', 'af2']
    with pytest.raises(ValueError, match='AlphaFold 3 features only'):
        pf.choose_features(root, [dict(chain_id='A', name='Q33333', sequence='MK')], 'af2')
    assert pf.choose_features(root, [dict(chain_id='A', name='Q33333', sequence='MK')], 'af3')[0]['kind'] == 'af3'
    with pytest.raises(ValueError, match='No features for P99999'):
        pf.choose_features(root, [dict(chain_id='A', name='P99999', sequence='MK')], 'af3')
    with pytest.raises(ValueError, match='not found'):
        pf.feature_directories(tmp_path / 'missing')
    # An AlphaPulldown output folder: both formats side by side
    flat = tmp_path / 'ap_features'
    flat.mkdir()
    (flat / 'P11111.pkl').write_bytes(b'')
    (flat / 'P11111_af3_input.json').write_text('{}')
    assert {k: v.name for k, v in pf.find_features(flat, 'P11111').items()} == \
        {'af2': 'P11111.pkl', 'af3': 'P11111_af3_input.json'}


def test_prepare_af2_heteromer_homomer_and_af3_job(tmp_path):
    root = lab_database(tmp_path)
    spec = dict(directory=str(tmp_path / 'out'), root=str(root), backend='af2', job=None, chains=[
        dict(chain_id='A', name='P11111', sequence='KVLA', start_hint=1),
        dict(chain_id='B', name='P22222', sequence='GGHHS', start_hint=0)])
    args = pf.prepare(spec)
    out = tmp_path / 'out'
    assert args == [f'--a3m={out}/chain_A.a3m,{out}/chain_B.a3m',
                    f'--paired-a3m={out}/paired_A.a3m,{out}/paired_B.a3m']
    assert (out / 'chain_A.a3m').read_text() == '>A\nKVLA\n>A_1\nKVLA\n>A_2\nRVLA\n'
    # YEAST is in both chains: one paired row
    assert (out / 'paired_A.a3m').read_text() == '>A\nKVLA\n>A_paired_1\nKVIA\n'
    assert (out / 'paired_B.a3m').read_text() == '>B\nGGHHS\n>B_paired_1\nGAHHS\n'
    prov = json.loads((out / 'features.json').read_text())
    assert prov[0]['residues'] == [2, 5] and prov[0]['paired_rows'] == 1 and prov[1]['kind'] == 'af2'
    homo = dict(spec, directory=str(tmp_path / 'homo'), chains=[dict(c, chain_id=cid) for cid, c in
                                                               zip('AB', [spec['chains'][0]] * 2)])
    assert not any(a.startswith('--paired-a3m') for a in pf.prepare(homo))
    job = tmp_path / 'input.json'
    job.write_text(json.dumps(dict(dialect='alphafold3', version=1, sequences=[
        dict(protein=dict(id='A', sequence='TRPK'))])))
    args = pf.prepare(dict(directory=str(tmp_path / 'af3'), root=str(root), backend='af3', job=str(job),
                           chains=[dict(chain_id='A', name='Q33333', sequence='TRPK', start_hint=2)]))
    assert args == []
    p = json.loads(job.read_text())['sequences'][0]['protein']
    assert p['unpairedMsa'] == '>query\nTRPK\n>hit\nTxxRPA\n>gap\nTR--\n'     # Inner insertions kept
    assert p['pairedMsa'] == '>query\nTRPK\n>tr|X|X_BACSU/1-6\nTRPK\n' and p['templates'] == []


def test_script_execs_command_with_alignment_arguments(tmp_path):
    root = lab_database(tmp_path)
    spec = tmp_path / 'spec.json'
    spec.write_text(json.dumps(dict(directory=str(tmp_path / 'out'), root=str(root), backend='af2', job=None,
                                    chains=[dict(chain_id='A', name='P11111', sequence='MKVLAW', start_hint=0)])))
    script = pf.__file__
    run = subprocess.run([sys.executable, '-B', script, str(spec), '--', sys.executable, '-c',
                          'import sys; print("ENGINE", sys.argv[1:])'], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert f"ENGINE ['--a3m={tmp_path}/out/chain_A.a3m']" in run.stdout
    assert 'features: reading P11111' in run.stdout
    spec.write_text(json.dumps(dict(json.loads(spec.read_text()), chains=[
        dict(chain_id='A', name='P11111', sequence='WWWW', start_hint=0)])))
    run = subprocess.run([sys.executable, '-B', script, str(spec), '--', 'true'], capture_output=True, text=True)
    assert run.returncode == 1 and pf.ERROR_PREFIX + 'The chain sequence' in run.stdout


def test_prediction_job_runs_features_step_first(tmp_path):
    root = lab_database(tmp_path)
    engine = tmp_path / 'engine'
    engine.write_text(f'#!{sys.executable}\n' + '''
import json, pathlib, sys
args = dict(a[2:].split('=', 1) if '=' in a else (a[2:], True) for a in sys.argv[1:])
d = pathlib.Path(args['out']).parent
(d/'engine_args.json').write_text(json.dumps(sys.argv[1:]))
conf = dict(token_chain_ids=['A']*3+['B']*2, token_res_ids=[1,2,3,1,2], pae=[[3]*5 for _ in range(5)])
(d/'fold_confidences.json').write_text(json.dumps(conf))
(d/'fold_summary_confidences.json').write_text(json.dumps(dict(ptm=.8, iptm=.7, mean_plddt=90)))
three = dict(K='LYS', V='VAL', L='LEU', G='GLY')
lines = [f'ATOM  {i+1:5d}  CA  {three[aa]} {c}{r:4d}    {i*3.8:8.3f}{0.:8.3f}{0.:8.3f}{1.:6.2f}{90.:6.2f}          C  '
         for i, (c, r, aa) in enumerate([('A',1,'K'),('A',2,'V'),('A',3,'L'),('B',1,'G'),('B',2,'G')])]
(d/'fold.pdb').write_text('\\n'.join(lines)+'\\nEND\\n')
''')
    engine.chmod(0o755)
    chains = [dict(sequence='KVL', residue_map=[dict(number=n) for n in (2, 3, 4)]),
              dict(sequence='GG', residue_map=[dict(number=1), dict(number=2)])]
    with pytest.raises(ValueError, match='one alignment source'):
        PredictionJob(tmp_path / 'x', chains, executable=engine, search=True,
                      features=dict(root=str(root), names=['P11111', 'P22222']))
    job = PredictionJob(tmp_path / 'job', chains, executable=engine, cache=tmp_path / 'cache',
                        features=dict(root=str(root), names=['P11111', 'P22222'])).start()
    end = time.monotonic() + 20
    while job.poll() == 'running' and time.monotonic() < end:
        time.sleep(0.05)
    assert job.state == 'complete', job.error
    engine_args = json.loads((job.directory / 'engine_args.json').read_text())
    assert any(a.startswith('--a3m=') for a in engine_args) and any(a.startswith('--paired-a3m=') for a in engine_args)
    assert '--search' not in engine_args
    assert job.manifest['msa_source'].startswith('precomputed features from')
    assert [f['name'] for f in job.manifest['features']] == ['P11111', 'P22222']
    assert not job.msa_cache.exists()               # Feature runs leave the MSA cache alone
    bad = PredictionJob(tmp_path / 'bad', chains, executable=engine, cache=tmp_path / 'cache',
                        features=dict(root=str(root), names=['P11111', 'P99999'])).start()
    end = time.monotonic() + 20
    while bad.poll() == 'running' and time.monotonic() < end:
        time.sleep(0.05)
    assert bad.state == 'failed' and bad.error.startswith('precomputed features: No AlphaFold 2 features for P99999')
