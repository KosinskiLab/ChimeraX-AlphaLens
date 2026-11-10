import json
from pathlib import Path
import sys
import time

import pytest
from alphalens.predict import PredictionJob, crop_chains, validate_a3m, validate_chains, validate_result


def rows():
    return [dict(chain_id=c, chain_index=i, aa='A', number=i + 10, insertion_code='')
            for c in 'AB' for i in range(8)]


def test_crop_extends_contiguously_and_preserves_numbering():
    chains = crop_chains(rows(), (2, 4, 9, 12), 2)
    assert [len(c['sequence']) for c in chains] == [7, 7]
    assert chains[0]['residue_map'][0]['number'] == 10
    assert chains[1]['residue_map'][0]['source_row'] == 8
    assert chains[0]['core_rows'] == [2, 3, 4]


@pytest.mark.parametrize('region', [None, (0, 3, 4, 6), (0, 8, 9, 11), (0, 2, 14, 19)])
def test_crop_rejects_ambiguous_regions(region):
    with pytest.raises(ValueError):
        crop_chains(rows(), region)


def test_crop_rejects_missing_sequence_positions():
    r = rows(); r[3]['chain_index'] = 99
    with pytest.raises(ValueError, match='missing residues'):
        crop_chains(r, (2, 4, 9, 12), 0)


def test_sequence_limits():
    with pytest.raises(ValueError, match='400'):
        validate_chains([dict(sequence='A'*201, residue_map=[{}]*201)]*2)
    with pytest.raises(ValueError, match='standard'):
        validate_chains([dict(sequence='AX', residue_map=[{}, {}])]*2)


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
