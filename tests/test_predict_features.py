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
    # Pickling imports each class's module, parents first: stand in for the whole packages, so
    # this works without AlphaPulldown or AlphaFold installed, as the reader must.
    names = ('alphapulldown', 'alphapulldown.objects', 'alphafold', 'alphafold.data',
             'alphafold.data.tools', 'alphafold.data.tools.jackhmmer')
    stand_ins = {name: types.ModuleType(name) for name in names}
    for name in ('alphapulldown', 'alphafold', 'alphafold.data', 'alphafold.data.tools'):
        stand_ins[name].__path__ = []
    mod, jack = stand_ins['alphapulldown.objects'], stand_ins['alphafold.data.tools.jackhmmer']

    class MonomericObject:
        pass

    class Jackhmmer:
        pass
    MonomericObject.__module__, Jackhmmer.__module__ = mod.__name__, jack.__name__
    MonomericObject.__qualname__, Jackhmmer.__qualname__ = 'MonomericObject', 'Jackhmmer'
    mod.MonomericObject, jack.Jackhmmer = MonomericObject, Jackhmmer
    missing = object()
    saved = {k: sys.modules.get(k, missing) for k in names}
    sys.modules.update(stand_ins)
    try:
        obj = MonomericObject()
        obj.description, obj.sequence, obj.feature_dict = description, seq, fd
        obj._uniprot_runner = Jackhmmer()
        obj._uniprot_runner.binary_path = '/opt/conda/bin/jackhmmer'
        return pickle.dumps(obj, protocol=4)
    finally:
        for k, v in saved.items():
            if v is missing:
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
    # AlphaPulldown's AF3 names: every ID is in the file name, not only the one before its first dot
    af3 = '/x/Q15650+Q8N9N2+Q9H1I8/Q15650_af3_input.json_and_Q8N9N2_af3_input.json_and_Q9H1I8_af3_input.json_model.cif'
    assert pf.guess_names(af3, ['A', 'B', 'C']) == {'A': 'Q15650', 'B': 'Q8N9N2', 'C': 'Q9H1I8'}
    assert pf.guess_names('/x/agreement_test/Q15650+A0MZ66/Q15650_af3_input.json_and_A0MZ66_af3_input.json_model.cif',
                          ['A', 'B'], ['MKV', 'GGH']) == {'A': 'Q15650', 'B': 'A0MZ66'}
    # One ID for all chains only when they are one protein
    server = '/x/fold_a0abd7fqg_and_p18004/fold_a0abd7fqg_and_p18004_model_0.cif'
    assert pf.guess_names(server, ['A', 'B'], ['MKV', 'GGH']) == {}
    assert pf.guess_names(server, ['A', 'B'], ['MKV', 'MKV']) == {'A': 'P18004', 'B': 'P18004'}
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


def test_gapped_chain_takes_the_columns_of_each_segment(tmp_path):
    # Columns 1, 2, 5, 6 (0-based 0, 1, 4, 5): an insertion stays inside a run, not across the cut
    text = pf.crop_a3m_text('>q\nABCDEF\n>h\nAxBC-EF\n>g\n--CD--\n', [0, 1, 4, 5], None, 'ABCDEF')
    assert text == '>q\nABEF\n>h\nAxBEF\n'
    root = lab_database(tmp_path)
    spec = dict(directory=str(tmp_path / 'out'), root=str(root), backend='af2', job=None, chains=[
        dict(chain_id='A', name='P11111', sequence='MKAW', start_hint=0, segments=[['MK', 0], ['AW', 4]]),
        dict(chain_id='B', name='P22222', sequence='GGHHS', start_hint=0)])
    pf.prepare(spec)
    out = tmp_path / 'out'
    assert (out / 'chain_A.a3m').read_text().split('\n')[:2] == ['>A', 'MKAW']
    assert (out / 'paired_A.a3m').read_text().split('\n')[:2] == ['>A', 'MKAW']
    prov = json.loads((out / 'features.json').read_text())[0]
    assert prov['segments'] == [[1, 2], [5, 6]] and prov['residues'] == [1, 6]
    with pytest.raises(ValueError, match='not in order'):
        pf.prepare(dict(spec, directory=str(tmp_path / 'bad'), chains=[
            dict(chain_id='A', name='P11111', sequence='AWMK', segments=[['AW', 4], ['MK', 0]])]))


# ---- A3M files in AlphaFold 3's form (AlphaFold Server downloads)

def server_download(tmp_path, job='fold_x'):
    '''A download's model and msas folder: heteromer chains a (KVLAW) and b (GGHHS).'''
    d = tmp_path / job
    (d / 'msas').mkdir(parents=True)
    (d / f'{job}_model_0.cif').write_text('')
    msas = {
        'unpaired_msa_chains_a': '>query\nKVLAW\n>UniRef90_A\nKVxxLAW\n>UniRef90_B\n--LA-\n',
        'paired_msa_chains_a': ('>query\nKVLAW\n>tr|A0A001|A0A001_YEAST/1-5 x OS=Yeast\nKILAW\n'
                                '>sp|P11111|NAME_HUMAN/1-5\nRVLAW\n>tr|A0A002|A0A002_ECOLI/1-5\nKVLAF\n'),
        'unpaired_msa_chains_b': '>query\nGGHHS\n>UniRef90_C\nGAHHS\n',
        'paired_msa_chains_b': ('>query\nGGHHS\n>tr|A0A003|A0A003_HUMAN/1-5\nGAHHT\n'
                                '>tr|A0A004|A0A004_YEAST/1-5\nGAHHS\n'),
    }
    for name, text in msas.items():
        (d / 'msas' / f'{job}_{name}.a3m').write_text(text)
    (d / 'msas' / 'other_job_unpaired_msa_chains_a.a3m').write_text('>query\nWWW\n')
    return d


def test_server_msas_found_beside_the_model(tmp_path):
    d = server_download(tmp_path)
    found = pf.server_msas(d / 'fold_x_model_0.cif')
    assert sorted(found) == ['A', 'B']
    assert found['A']['sequence'] == 'KVLAW' and found['A']['paired'].endswith('fold_x_paired_msa_chains_a.a3m')
    assert pf.server_msas(tmp_path / 'nothing' / 'model.cif') == {}
    # Identical chains share one pair of files (assumed naming chains_a_b)
    (d / 'msas' / 'fold_x_unpaired_msa_chains_c_d.a3m').write_text('>query\nMMM\n')
    found = pf.server_msas(d / 'fold_x_model_0.cif')
    assert found['C'] is found['D'] and found['C']['paired'] is None


def test_a3m_arrays_and_species():
    msa, dels, headers = pf.a3m_arrays('>q\nKVLA\n>h1\nKvvV-Xa\n>h2\nUBZX\n')
    assert pf.decode(msa) == ['KVLA', 'KV-X', 'CDEX'] and headers == ['q', 'h1', 'h2']
    assert dels.tolist() == [[0, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0]]       # Trailing insertion dropped
    assert pf.species_of('tr|A0A080I4K2|A0A080I4K2_ECOLX/1-103 [subseq from] Protein') == 'ECOLX'
    assert pf.species_of('>sp|Q15650|TRIP4_HUMAN/1-581 x') == 'HUMAN'
    assert pf.species_of('UniRef90_Q6SJ25/1-103 n=392') == '' and pf.species_of('query') == ''


def test_prepare_server_msas_for_af2_and_af3(tmp_path):
    d = server_download(tmp_path)
    found = pf.server_msas(d / 'fold_x_model_0.cif')
    chains = [dict(chain_id='A', name='chain A', sequence='VLA', start_hint=1, **{k: found['A'][k] for k in ('unpaired', 'paired')}),
              dict(chain_id='B', name='chain B', sequence='GGHHS', start_hint=0, **{k: found['B'][k] for k in ('unpaired', 'paired')})]
    out = tmp_path / 'af2'
    args = pf.prepare(dict(directory=str(out), root='', backend='af2', job=None, chains=chains))
    assert args == [f'--a3m={out}/chain_A.a3m,{out}/chain_B.a3m', f'--paired-a3m={out}/paired_A.a3m,{out}/paired_B.a3m']
    # Columns 2-4; the insertion before a kept column inside the crop is kept, the row with no
    # residue left is dropped
    assert (out / 'chain_A.a3m').read_text() == '>A\nVLA\n>A_1\nVxxLA\n>A_2\n-LA\n'
    # Paired by species as AlphaFold-Multimer does: HUMAN and YEAST in both chains, ECOLI in one
    assert (out / 'paired_A.a3m').read_text() == '>A\nVLA\n>A_paired_1\nVLA\n>A_paired_2\nILA\n'
    assert (out / 'paired_B.a3m').read_text() == '>B\nGGHHS\n>B_paired_1\nGAHHT\n>B_paired_2\nGAHHS\n'
    prov = json.loads((out / 'features.json').read_text())
    assert prov[0]['kind'] == 'a3m' and prov[0]['residues'] == [2, 4] and prov[0]['paired_rows'] == 2
    assert prov[0]['paired'] == found['A']['paired'] and len(prov[0]['paired_sha256']) == 64
    job = tmp_path / 'input.json'
    job.write_text(json.dumps(dict(dialect='alphafold3', version=1, sequences=[
        dict(protein=dict(id='A', sequence='VLA')), dict(protein=dict(id='B', sequence='GGHHS'))])))
    assert pf.prepare(dict(directory=str(tmp_path / 'af3'), root='', backend='af3', job=str(job), chains=chains)) == []
    a, b = (s['protein'] for s in json.loads(job.read_text())['sequences'])
    assert a['unpairedMsa'] == '>query\nVLA\n>UniRef90_A\nVxxLA\n>UniRef90_B\n-LA\n'
    assert a['pairedMsa'].startswith('>query\nVLA\n>tr|A0A001|A0A001_YEAST/1-5 x OS=Yeast\nILA\n')
    assert b['pairedMsa'] == (d / 'msas' / 'fold_x_paired_msa_chains_b.a3m').read_text() and a['templates'] == []
    bad = dict(chains[1], sequence='WWW')
    with pytest.raises(ValueError, match='not part'):
        pf.prepare(dict(directory=str(tmp_path / 'bad'), root='', backend='af2', job=None, chains=[chains[0], bad]))


def test_prediction_job_takes_server_msas(tmp_path):
    d = server_download(tmp_path)
    found = pf.server_msas(d / 'fold_x_model_0.cif')
    engine = tmp_path / 'engine'
    engine.write_text(f'#!{sys.executable}\nimport sys; print("ENGINE", sys.argv[1:]); sys.exit(3)\n')
    engine.chmod(0o755)
    chains = [dict(sequence='VLA', source_chain='A', residue_map=[dict(number=n) for n in (2, 3, 4)]),
              dict(sequence='GGHHS', source_chain='B', residue_map=[dict(number=n) for n in range(1, 6)])]
    msas = [dict(unpaired=found[c]['unpaired'], paired=found[c]['paired']) for c in 'AB']
    with pytest.raises(ValueError, match='unpaired A3M'):
        PredictionJob(tmp_path / 'x', chains, executable=engine, features=dict(msas=msas[:1], origin='o'))
    job = PredictionJob(tmp_path / 'job', chains, executable=engine, cache=tmp_path / 'cache',
                        features=dict(msas=msas, origin=f'AlphaFold Server MSAs in {d / "msas"}')).start()
    end = time.monotonic() + 20
    while job.poll() == 'running' and time.monotonic() < end:
        time.sleep(0.05)
    log = (job.directory / 'prediction.log').read_text()
    assert f'--paired-a3m={job.directory}/features/paired_A.a3m' in log, log
    assert job.manifest['msa_source'] == f'AlphaFold Server MSAs in {d / "msas"}'
    spec = json.loads((job.directory / 'features_spec.json').read_text())
    assert [c['name'] for c in spec['chains']] == ['chain A', 'chain B'] and spec['chains'][0]['paired'] == msas[0]['paired']
    (d / 'msas' / 'fold_x_unpaired_msa_chains_b.a3m').write_text('>query\nGGHHS\n')
    (d / 'msas' / 'fold_x_paired_msa_chains_b.a3m').write_text('>query\nGGHHW\n')
    bad = PredictionJob(tmp_path / 'bad', chains, executable=engine, cache=tmp_path / 'cache',
                        features=dict(msas=msas, origin='o')).start()
    end = time.monotonic() + 20
    while bad.poll() == 'running' and time.monotonic() < end:
        time.sleep(0.05)
    assert bad.state == 'failed' and bad.error.startswith('source alignments: The first sequences of'), bad.error


def af3_output(tmp_path, compress=False):
    '''An AlphaFold 3 output folder: <name>_data.json with each chain's MSAs, a top model and a sample.'''
    d = tmp_path / 'Q1_and_Q2'
    (d / 'seed-7_sample-0').mkdir(parents=True)
    job = dict(dialect='alphafold3', version=4, name='Q1_and_Q2', sequences=[
        dict(ligand=dict(id='L', ccdCodes=['ATP'])),
        dict(protein=dict(id=['A', 'C'], sequence='KVLAW', unpairedMsa='>query\nKVLAW\n>u1\nKVxLAW\n',
                          pairedMsa='>query\nKVLAW\n>tr|A0A001|A0A001_YEAST/1-5\nKILAW\n', templates=[])),
        dict(protein=dict(id='B', sequence='GGHHS', unpairedMsa='>query\nGGHHS\n', pairedMsa='',
                          templates=[]))])
    text = json.dumps(job, indent=2)
    path = d / ('Q1_and_Q2_data.json.xz' if compress else 'Q1_and_Q2_data.json')
    path.write_bytes(lzma.compress(text.encode()) if compress else text.encode())
    for model in (d / 'Q1_and_Q2_model.cif', d / 'seed-7_sample-0' / 'model.cif'):
        model.write_text('')
    return d, path


def test_af3_data_json_beside_the_model_or_above_a_sample(tmp_path):
    for compress in (False, True):
        d, path = af3_output(tmp_path / str(compress), compress)
        assert pf.af3_data_json(d / 'Q1_and_Q2_model.cif') == path
        assert pf.af3_data_json(d / 'seed-7_sample-0' / 'model.cif') == path
        chains = pf.load_af3_chains(path)
        assert sorted(chains) == ['A', 'B', 'C'] and chains['C']['unpaired'] == chains['A']['unpaired']
        assert chains['B']['sequence'] == 'GGHHS' and chains['B']['paired'] == ''
    # An input JSON without MSAs (a job not run yet) is not taken
    d = tmp_path / 'input'
    d.mkdir()
    (d / 'x_data.json').write_text(json.dumps(dict(sequences=[dict(protein=dict(id='A', sequence='KV'))])))
    (d / 'x_model.cif').write_text('')
    assert pf.af3_data_json(d / 'x_model.cif') is None
    assert pf.af3_data_json(tmp_path / 'nothing' / 'model.cif') is None


def test_prepare_af3_data_json_chains_for_af2_and_af3(tmp_path):
    d, path = af3_output(tmp_path)
    chains = [dict(chain_id='A', name='chain C', sequence='VLA', start_hint=1, af3_json=str(path), af3_chain='C'),
              dict(chain_id='B', name='chain B', sequence='GGHHS', start_hint=0, af3_json=str(path), af3_chain='B')]
    out = tmp_path / 'af2'
    args = pf.prepare(dict(directory=str(out), root='', backend='af2', job=None, chains=chains))
    assert args[0] == f'--a3m={out}/chain_A.a3m,{out}/chain_B.a3m'
    assert (out / 'chain_A.a3m').read_text() == '>A\nVLA\n>A_1\nVxLA\n'
    assert (out / 'paired_B.a3m').read_text() == '>B\nGGHHS\n'           # B has no paired MSA: query only
    prov = json.loads((out / 'features.json').read_text())
    assert prov[0]['af3_chain'] == 'C' and prov[0]['path'] == str(path) and 'paired_sha256' not in prov[0]
    job = tmp_path / 'input.json'
    job.write_text(json.dumps(dict(dialect='alphafold3', version=1, sequences=[
        dict(protein=dict(id='A', sequence='VLA')), dict(protein=dict(id='B', sequence='GGHHS'))])))
    assert pf.prepare(dict(directory=str(tmp_path / 'af3'), root='', backend='af3', job=str(job), chains=chains)) == []
    a, b = (s['protein'] for s in json.loads(job.read_text())['sequences'])
    assert a['unpairedMsa'] == '>query\nVLA\n>u1\nVxLA\n' and a['pairedMsa'] == '>query\nVLA\n>tr|A0A001|A0A001_YEAST/1-5\nILA\n'
    assert b['pairedMsa'] == ''
    with pytest.raises(ValueError, match='no protein chain L'):
        pf.prepare(dict(directory=str(tmp_path / 'bad'), root='', backend='af2', job=None,
                        chains=[chains[0], dict(chains[1], af3_chain='L')]))


# ---- A native AlphaFold 2 run (features.pkl of the whole complex) and AlphaPulldownSnakemake layout

RT = {c: i for i, c in enumerate(pf.RESTYPES)}


def native_af2_output(tmp_path, chain_map=None):
    '''A native AlphaFold 2 multimer output: features.pkl (processed: residue_constants order, a
    deletion before the second column of the paired row, padded to 6 rows) and msas/chain_id_map.json.'''
    d = tmp_path / 'native'
    (d / 'msas').mkdir(parents=True)
    rows = ['KVLAGGH', 'RVLAGAH',        # query; a paired row
            'KILA---', '----GAN']        # unpaired rows of A and of B
    msa = np.array([[RT[c] for c in r] for r in rows] + [[0] * 7] * 2, np.int32)
    dels = np.zeros(msa.shape, np.float32)
    dels[1, 1] = 2
    mask = np.ones(msa.shape, np.float32)
    mask[4:] = 0
    fd = dict(msa=msa, deletion_matrix=dels, msa_mask=mask, num_alignments=np.array(4, np.int32),
              asym_id=np.array([1.] * 4 + [2.] * 3))
    (d / 'features.pkl').write_bytes(pickle.dumps(fd))
    (d / 'msas' / 'chain_id_map.json').write_text(json.dumps(chain_map or {
        'B': dict(description='b', sequence='GGH'), 'A': dict(description='a', sequence='KVLA')}))
    (d / 'ranked_0.pdb').write_text('')
    return d


def test_native_af2_features_found_by_chain_map(tmp_path):
    d = native_af2_output(tmp_path)
    path, sequences = pf.af2_complex_features(d / 'ranked_0.pdb')
    assert path == d / 'features.pkl' and sequences == ['KVLA', 'GGH']      # In chain order, not file order
    (d / 'msas' / 'chain_id_map.json').unlink()
    assert pf.af2_complex_features(d / 'ranked_0.pdb') is None
    assert pf.af2_complex_features(tmp_path / 'nothing' / 'ranked_0.pdb') is None


def test_prepare_merged_crops_the_whole_complex_msa(tmp_path):
    d = native_af2_output(tmp_path)
    chains = [dict(chain_id='A', sequence='VLA'), dict(chain_id='B', sequence='GH')]
    spec = dict(directory=str(tmp_path / 'out'), merged=str(d / 'features.pkl'), source_query='KVLAGGH',
                columns=[1, 2, 3, 5, 6], chains=chains)
    assert pf.prepare_merged(spec) == [f'--a3m={tmp_path}/out/merged.a3m']
    # The insertion before column 1 is at the crop's edge and dropped; padded rows left out; the
    # unpaired rows keep their gaps in the other chain's columns
    assert (tmp_path / 'out' / 'merged.a3m').read_text() == \
        '>query\nVLAGH\n>row_1\nVLAAH\n>row_2\nILA--\n>row_3\n---AN\n'
    prov = json.loads((tmp_path / 'out' / 'features.json').read_text())
    assert prov[0]['kind'] == 'af2_complex' and prov[0]['rows'] == 4 and prov[0]['source_rows'] == 4
    inside = dict(spec, directory=str(tmp_path / 'inside'), columns=[0, 1, 2, 3, 4, 5, 6],
                  chains=[dict(sequence='KVLA'), dict(sequence='GGH')])
    pf.prepare_merged(inside)
    assert '>row_1\nRxxVLAGAH\n' in (tmp_path / 'inside' / 'merged.a3m').read_text()   # Kept inside the crop
    with pytest.raises(ValueError, match='not those columns'):
        pf.prepare_merged(dict(spec, chains=[dict(sequence='KVL'), dict(sequence='GH')]))
    with pytest.raises(ValueError, match='not the sequence'):
        pf.prepare_merged(dict(spec, source_query='KVLAGGW', chains=[dict(sequence='VLA'), dict(sequence='GW')]))


def test_prepare_dispatches_merged_and_job_records_it(tmp_path):
    d = native_af2_output(tmp_path)
    engine = tmp_path / 'engine'
    engine.write_text(f'#!{sys.executable}\nimport sys; print("ENGINE", sys.argv[1:]); sys.exit(3)\n')
    engine.chmod(0o755)
    chains = [dict(sequence='VLA', source_chain='A', residue_map=[dict(number=n, source_row=n - 1) for n in (2, 3, 4)]),
              dict(sequence='GH', source_chain='B', residue_map=[dict(number=2, source_row=5), dict(number=3, source_row=6)])]
    crop = dict(source_query='KVLAGGH', columns=[1, 2, 3, 5, 6])
    with pytest.raises(ValueError, match='source sequence and the columns'):
        PredictionJob(tmp_path / 'x', chains, executable=engine, features=dict(merged=str(d / 'features.pkl')))
    job = PredictionJob(tmp_path / 'job', chains, executable=engine, cache=tmp_path / 'cache',
                        features=dict(merged=str(d / 'features.pkl'), msa_crop=crop,
                                      origin=f'AlphaFold 2 features in {d / "features.pkl"}')).start()
    end = time.monotonic() + 20
    while job.poll() == 'running' and time.monotonic() < end:
        time.sleep(0.05)
    log = (job.directory / 'prediction.log').read_text()
    assert f"'--a3m={job.directory}/features/merged.a3m'" in log, log
    assert job.manifest['msa_source'].startswith('AlphaFold 2 features in') and job.manifest['msa_crop'] == crop


def test_snakemake_layout_features(tmp_path):
    out = tmp_path / 'out'
    (out / 'features').mkdir(parents=True)
    fold = out / 'predictions' / 'P00648+P11540:5-80'
    fold.mkdir(parents=True)
    (fold / 'ranked_0.pdb').write_text('')
    (out / 'features' / 'P00648.pkl').write_bytes(b'')
    assert pf.snakemake_features(fold / 'ranked_0.pdb', ['A', 'B']) is None       # P11540's file missing
    (out / 'features' / 'P11540.pkl.xz').write_bytes(b'')
    assert pf.snakemake_features(fold / 'ranked_0.pdb', ['A', 'B']) == \
        (out / 'features', {'A': 'P00648', 'B': 'P11540'})                       # The range left out
    assert pf.snakemake_features(fold / 'ranked_0.pdb', ['A', 'B', 'C']) is None
    other = tmp_path / 'elsewhere' / 'P00648+P11540'
    other.mkdir(parents=True)
    assert pf.snakemake_features(other / 'ranked_0.pdb', ['A', 'B']) is None
    # AlphaPulldown's own names: "_and_", homo-oligomers, chopped regions
    for name, ids, expect in (('P00648_and_P11540', 'AB', ['P00648', 'P11540']),
                              ('P00648_homo_3er', 'ABC', ['P00648'] * 3),
                              ('P00648_and_P11540_1-10_20-30', 'AB', ['P00648', 'P11540'])):
        f = out / 'predictions' / name
        f.mkdir()
        assert pf.snakemake_features(f / 'ranked_0.pdb', list(ids)) == (out / 'features', dict(zip(ids, expect)))
