import json
from pathlib import Path
import sys
import time

import pytest
from alphalens import predict
from alphalens.predict import PredictionJob, PredictionProgress, crop_chains, size_warning, validate_a3m, validate_chains, validate_result


def rows():
    return [dict(chain_id=c, chain_index=i, aa='A', number=i + 10, insertion_code='')
            for c in 'AB' for i in range(8)]


def test_progress_reports_engine_milestones_and_preserves_partial_lines():
    progress = PredictionProgress()
    assert progress.total is None
    progress.feed('  multimer: weights.bin (1/8)\n')
    assert progress.stage == 'Downloading model weights' and progress.total is None
    progress.feed('search: chain search 1 of 2: queued\n')
    assert 'queued' in progress.stage and progress.total is None
    progress.feed('search: paired search: downloading\n')
    assert 'downloading' in progress.stage and progress.total is None
    progress.feed('799 residues, 512 MSA rows, 1024 extra, 4 passes -> /tmp/in\n')
    assert progress.stage == 'Loading model and preparing GPU'
    progress.feed('AF2 multimer: 799 residues, 512 MSA rows, 1024 extra, 4 passes\n  pass 1/')
    assert (progress.completed, progress.total) == (0, 4)
    progress.feed('4: mean pLDDT 59.02\n' + 'metal: long diagnostics ' * 500 + '\n')
    assert (progress.completed, progress.total) == (1, 4)
    progress.feed('  pass 2/4: mean pLDDT 56.99\n  pass 3/4: mean pLDDT 58.04\n')
    assert (progress.completed, progress.total) == (3, 4)
    progress.feed('  pass 4/4: mean pLDDT 58.02\n')
    assert progress.stage == 'Writing prediction files' and progress.total is None


def test_progress_ignores_unknown_and_invalid_messages():
    progress = PredictionProgress()
    progress.feed('pass 2/0: invalid\npass 5/4: invalid\nAF2 monomer: invalid, 0 passes\n')
    assert progress.total is None
    progress.feed('AF2 monomer: 108 residues, 512 MSA rows, 1024 extra, 1 passes\n')
    assert (progress.completed, progress.total) == (0, 1)
    progress.feed('unknown output without newline' * 1000)
    assert len(progress._pending) <= 4096
    assert (progress.completed, progress.total) == (0, 1)


def test_crop_extends_contiguously_and_preserves_numbering():
    chains = crop_chains(rows(), (2, 4, 9, 12), 2)
    assert [len(c['sequence']) for c in chains] == [7, 7]
    assert chains[0]['residue_map'][0]['number'] == 10
    assert chains[1]['residue_map'][0]['source_row'] == 8
    assert chains[0]['core_rows'] == [2, 3, 4]


@pytest.mark.parametrize('region', [None, (0, 3, 4, 6), (0, 2, 14, 19)])
def test_crop_rejects_ambiguous_regions(region):
    with pytest.raises(ValueError):
        crop_chains(rows(), region)


def test_crop_rejects_missing_sequence_positions():
    r = rows(); r[3]['chain_index'] = 99
    with pytest.raises(ValueError, match='missing residues'):
        crop_chains(r, (2, 4, 9, 12), 0)


def test_crop_spans_multiple_chains_and_unions_overlapping_axes():
    r = [dict(chain_id=c, chain_index=i, aa='A', number=i+1, insertion_code='')
         for c in 'ABC' for i in range(4)]
    chains = crop_chains(r, (2, 6, 5, 10), 1)
    assert [c['source_chain'] for c in chains] == list('ABC')
    assert [c['core_rows'] for c in chains] == [[2, 3], [4, 5, 6, 7], [8, 9, 10]]
    assert [len(c['sequence']) for c in chains] == [3, 4, 4]


def test_chain_labels_do_not_collide_after_26_chains():
    from alphalens.predict import chain_ids
    assert chain_ids(28) == 'ABCDEFGHIJKLMNOPQRSTUVWXYZab'
    assert len(set(chain_ids(62))) == 62
    with pytest.raises(ValueError, match='PDB'):
        chain_ids(63)


def test_default_seed_is_random_per_job_and_explicit_seed_is_retained(runner, monkeypatch):
    from alphalens import predict
    seeds = iter([1234, 2345])
    monkeypatch.setattr(predict, 'random_seed', lambda: next(seeds))
    assert runner(search=True).options['seed'] == 1234
    assert runner(search=True).options['seed'] == 2345
    assert runner(search=True, seed=0).options['seed'] == 0


def test_sequence_validation_has_no_residue_count_cap():
    validate_chains([dict(sequence='A'*410, residue_map=[{}]*410),
                     dict(sequence='A'*389, residue_map=[{}]*389)])
    validate_chains([dict(sequence='A'*512, residue_map=[{}]*512)]*2)
    validate_chains([dict(sequence='A'*1024, residue_map=[{}]*1024)])
    validate_chains([dict(sequence='A'*10000, residue_map=[{}]*10000)]*2)
    with pytest.raises(ValueError, match='standard'):
        validate_chains([dict(sequence='AX', residue_map=[{}, {}])]*2)


def test_large_crop_is_allowed_with_nonblocking_warning():
    r = [dict(chain_id=c, chain_index=i, aa='A', number=i+1, insertion_code='')
         for c in 'AB' for i in range(600)]
    small = crop_chains(r, (50, 549, 650, 1149), 6)
    large = crop_chains(r, (50, 549, 650, 1149), 7)
    assert sum(len(c['sequence']) for c in small) == 1024
    assert sum(len(c['sequence']) for c in large) == 1028
    assert not size_warning(small)
    assert '1,028 residues including flanks' in size_warning(large)
    assert 'You can proceed' in size_warning(large)


def test_a3m_query_validation(tmp_path):
    p = tmp_path/'test.a3m'
    p.write_text('#comment\n>query\nAAaA\nA\n>hit\nVVVV\n')
    validate_a3m(p, 'AAAA')
    with pytest.raises(ValueError, match='query'):
        validate_a3m(p, 'AAAC')


@pytest.fixture
def runner(tmp_path):
    binary = tmp_path/'engine'
    # Deliberately controlled engine output: test process/cache protocol, not AF2 accuracy.
    binary.write_text(f'#!{sys.executable}\n' + '''
import json, pathlib, sys, time
args = dict(a[2:].split('=', 1) if '=' in a else (a[2:], True) for a in sys.argv[1:])
if args['seed'] == '99': time.sleep(60)
if args['seed'] == '98': sys.exit(7)
d = pathlib.Path(args['out']).parent
conf = dict(token_chain_ids=['A','A','B','B'], token_res_ids=[1,2,1,2], pae=[[3]*4 for _ in range(4)])
(d/'fold_confidences.json').write_text(json.dumps(conf))
(d/'fold_summary_confidences.json').write_text(json.dumps(dict(ptm=.8,iptm=.7,mean_plddt=90)))
lines=[]
for i in range(4):
 c='A' if i<2 else 'B'; r=i%2+1
 lines.append(f'ATOM  {i+1:5d}  CA  ALA {c}{r:4d}    {i*3.:8.3f}{0.:8.3f}{0.:8.3f}{1.:6.2f}{90.:6.2f}          C  ')
(d/'fold.pdb').write_text('\\n'.join(lines)+'\\nEND\\n')
if 'search' in args: (d/'fold.a3m').write_text('>query\\nAAAA\\n')
''')
    binary.chmod(0o755)
    chains = [dict(sequence='AA', residue_map=[{}, {}])]*2
    def make(name='job', **kw):
        return PredictionJob(tmp_path/name, chains, executable=binary, cache=tmp_path/'cache', **kw)
    return make


def finish(job):
    end = time.monotonic()+5
    while job.poll() == 'running' and time.monotonic()<end:
        time.sleep(.02)
    assert job.state != 'running'
    return job


def test_real_process_and_msa_cache(runner):
    first = finish(runner(search=True).start())
    assert first.state == 'complete', first.error
    assert first.result['mean_cross_chain_pae'] == 3
    second = finish(runner('second', search=True, seed=1).start())
    assert second.state == 'complete'
    assert second.manifest['msa_source'] == 'cache'
    assert '--search' not in second.manifest['args']
    manifest = json.loads((second.directory/'job.json').read_text())
    assert manifest['revision'] == 'unverified'
    assert len(manifest['msa_sha256']) == 64
    conf = second.directory/'fold_confidences.json'
    data = json.loads(conf.read_text()); data['token_res_ids'].reverse(); conf.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='token order'):
        validate_result(second.directory, second.chains)


def test_no_implicit_sequence_upload(runner):
    with pytest.raises(ValueError, match='Choose remote'):
        runner()


def test_af3_terms_gate_options_and_cross_model_msa_cache(runner, monkeypatch):
    monkeypatch.setattr(predict.sys, 'platform', 'darwin')
    with pytest.raises(ValueError, match='terms'):
        runner(search=True, backend='af3')
    first = finish(runner(search=True).start())
    af3 = runner('af3', search=True, backend='af3', accept_af3_terms=True)
    assert af3.model == 'af3'
    assert af3.msa_cache == first.msa_cache
    finish(af3.start())
    assert af3.state == 'complete', af3.error
    assert af3.manifest['msa_source'] == 'cache'
    assert '--steps=200' in af3.manifest['args']
    assert '--recycles=10' in af3.manifest['args'] and '--samples=5' in af3.manifest['args']
    assert af3.manifest['runtime_environment'] == dict(LOCALFOLD_GEMM_TENSOR='0', LOCALFOLD_ATTN_TENSOR='0',
                                                      NSUnbufferedIO='YES')
    assert '--update=euler' in af3.manifest['args'] and '--schedule=karras' in af3.manifest['args']
    assert not any(a.startswith('--max-extra') for a in af3.manifest['args'])
    assert af3.manifest['af3_terms_accepted'] is True
    assert af3.manifest['engine'] == 'LocalFold native Metal'


def test_cuda_engine_label_and_no_metal_settings(runner, monkeypatch):
    monkeypatch.setattr(predict.sys, 'platform', 'linux')
    af3 = finish(runner('af3-cuda', search=True, backend='af3', accept_af3_terms=True).start())
    assert af3.state == 'complete', af3.error
    assert af3.manifest['engine'] == 'LocalFold native CUDA'
    assert 'runtime_environment' not in af3.manifest
    af2 = finish(runner('af2-cuda', search=True, seed=2).start())
    assert af2.state == 'complete', af2.error
    assert af2.manifest['engine'] == 'LocalFold native CUDA'


def test_invalid_contact_matrix_is_rejected(runner):
    job = finish(runner(search=True).start())
    path = job.directory/'fold_confidences.json'
    data = json.loads(path.read_text())
    data['contact_probs'] = [[.5]*4 for _ in range(4)]
    path.write_text(json.dumps(data))
    assert validate_result(job.directory, job.chains)['has_contact_probabilities']
    data['contact_probs'][0][1] = 1.5
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='contact-probability'):
        validate_result(job.directory, job.chains)


def test_cancel_reaps_process_and_releases_lock(runner):
    job = runner(search=True, seed=99).start()
    with pytest.raises(ValueError, match='Another local'):
        runner('blocked', search=True).start()
    job.cancel(); finish(job)
    assert job.state == 'cancelled'
    assert job.process.poll() is not None
    assert finish(runner('next', search=True).start()).state == 'complete'


def test_timeout_and_engine_errors(runner):
    timed = finish(runner(search=True, seed=99, timeout=.05).start())
    assert timed.state == 'failed' and 'limit' in timed.error
    failed = finish(runner('failed', search=True, seed=98).start())
    assert failed.state == 'failed' and 'status 7' in failed.error


def test_default_has_no_timeout_but_manual_cancel_still_works(runner):
    job = runner(search=True, seed=99).start()
    job.started -= 3600
    assert job.poll() == 'running'
    assert job.cancelled_at is None
    assert job.manifest['timeout_seconds'] is None
    job.cancel(); finish(job)
    assert job.state == 'cancelled'


@pytest.mark.parametrize('timeout', [0, -1, float('inf'), float('nan')])
def test_invalid_time_limit(runner, timeout):
    with pytest.raises(ValueError, match='time limit'):
        runner(search=True, timeout=timeout)


def test_existing_output_preserved(runner):
    job = runner(search=True); job.directory.mkdir()
    sentinel = job.directory/'precious'; sentinel.write_text('keep')
    with pytest.raises(FileExistsError): job.start()
    assert sentinel.read_text() == 'keep'
    assert job.lock is None


def test_crop_paired_msa_preserves_rows_order_and_insertions(tmp_path):
    from alphalens.predict import crop_a3m
    source = tmp_path/'source.a3m'; target = tmp_path/'cropped.a3m'
    source.write_text('>query\nACDEFGHI\n>paired\nACdDEFGhHI\n>unpaired\nACDE----\n>outside\nA-------\n')
    # Reverse the chain order, keeping FGH then CD. Discard insertions at crop starts.
    crop_a3m(source, target, 'ACDEFGHI', [4, 5, 6, 1, 2])
    assert target.read_text() == '>query\nFGHCD\n>paired\nFGhHCdD\n>unpaired\n---CD\n'
    validate_a3m(target, 'FGHCD')


def test_crop_msa_rejects_ragged_alignment(tmp_path):
    from alphalens.predict import crop_a3m
    source = tmp_path/'source.a3m'; source.write_text('>query\nAAAA\n>bad\nAAA\n')
    with pytest.raises(ValueError, match='aligned length'):
        crop_a3m(source, tmp_path/'out', 'AAAA', [0, 2])


def test_manifest_failure_after_spawn_reaps_child(runner, monkeypatch):
    job = runner(search=True, seed=99)
    def fail():
        raise OSError('disk full')
    monkeypatch.setattr(job, '_save', fail)
    with pytest.raises(OSError, match='disk full'):
        job.start()
    assert job.state == 'failed'
    assert job.process.poll() is not None
    assert job.lock is None
    assert finish(runner('after_disk_error', search=True).start()).state == 'complete'


def test_a3m_headers_in_any_encoding_pass_through(tmp_path, monkeypatch):
    # ChimeraX switches to the C locale (ASCII text files) once a PDB file has been opened
    import locale
    from alphalens.predict import crop_a3m
    monkeypatch.setattr(locale, 'getpreferredencoding', lambda *a: 'ascii')
    source, cropped = tmp_path/'source.a3m', tmp_path/'cropped.a3m'
    source.write_bytes(b'>query caf\xc3\xa9\nACDE\n>hit \xe9t\xe9\nAaC-E\n')
    validate_a3m(source, 'ACDE')
    crop_a3m(source, cropped, 'ACDE', [0, 1, 3])
    assert cropped.read_bytes() == b'>query caf\xc3\xa9\nACE\n>hit \xe9t\xe9\nAaCE\n'


def test_engine_platform_follows_the_os_not_python_architecture(monkeypatch):
    # An x86_64 Python under Rosetta reports machine x86_64 but still launches the Metal engine.
    import platform
    monkeypatch.setattr(platform, 'machine', lambda: 'x86_64')
    for system, native in (('darwin', 'metal'), ('linux', 'cuda'), ('win32', None)):
        monkeypatch.setattr(predict.sys, 'platform', system)
        assert predict.engine_platform() == native


def test_cuda_engine_built_for_another_gpu_generation_is_refused(tmp_path, monkeypatch):
    # Seen on the EMBL cluster: A100-built engines on an H100 gave NaN, or a wrong AF2 monomer.
    binary = tmp_path / 'bin' / 'localfold-af2'
    binary.parent.mkdir()
    binary.write_text('#!/bin/sh\n')
    binary.chmod(0o755)
    receipt = dict(revision=predict.ENGINE_REVISION, sha256=predict.digest(binary), native='cuda',
                   cuda_compute_capability='8.0')
    binary.with_suffix('.json').write_text(json.dumps(receipt))
    monkeypatch.setattr(predict.sys, 'platform', 'linux')
    chains = [dict(sequence='AA', residue_map=[{}, {}])] * 2
    def make():
        return PredictionJob(tmp_path / 'job', chains, executable=binary, cache=tmp_path / 'cache', search=True)
    monkeypatch.setattr(predict, 'cuda_compute_capability', lambda: '9.0')
    with pytest.raises(ValueError, match='compute capability 8.0, but this GPU has 9.0'):
        make()
    assert not (tmp_path / 'job').exists()
    monkeypatch.setattr(predict, 'cuda_compute_capability', lambda: '8.0')
    make()                                                   # the same generation
    monkeypatch.setattr(predict, 'cuda_compute_capability', lambda: None)
    make()                                                   # cannot tell: the engine reports its own errors
    binary.write_text('#!/bin/sh\n# replaced by the user\n')
    monkeypatch.setattr(predict, 'cuda_compute_capability', lambda: '9.0')
    make()                                                   # no longer the receipt's engine: unverified
    monkeypatch.setattr(predict.sys, 'platform', 'darwin')
    receipt['sha256'] = predict.digest(binary)
    binary.with_suffix('.json').write_text(json.dumps(receipt))
    make()                                                   # Metal engines carry no GPU generation
