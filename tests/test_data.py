import gzip
import lzma

import numpy as np
import pytest

from alphalens import data


def test_missing_seed_distogram_never_uses_another_seed(tmp_path):
    sample = tmp_path / 'seed-2_sample-0'
    sample.mkdir()
    structure = sample / 'model.cif'
    structure.touch()
    (sample / 'confidences.json').write_text('{"pae": [[1]], "contact_probs": [[0.1]]}')
    other = tmp_path / 'seed-1_distogram'
    other.mkdir()
    np.savez(other / 'seed-1.npz', distogram=np.ones((1, 1, 64)) / 64)

    files = data.find_prediction_files(str(structure))
    assert files.model_name == 'seed-2_sample-0'
    assert files.distogram_path is None
    assert data.read_pae(files.pae_path)[1][0, 0] == pytest.approx(0.1)


def test_unknown_seed_requires_unambiguous_distogram(tmp_path):
    for seed in (1, 2):
        folder = tmp_path / f'seed-{seed}_distogram'
        folder.mkdir()
        (folder / 'prediction.npz').touch()
    assert data._af3_distogram(str(tmp_path)) is None
    assert data._af3_distogram(str(tmp_path), '2') == str(
        tmp_path / 'seed-2_distogram' / 'prediction.npz')


@pytest.mark.parametrize('suffix,opener', [('.gz', gzip.open), ('.xz', lzma.open)])
def test_compressed_npz_discovery_loading_and_agreement(tmp_path, suffix, opener):
    sample = tmp_path / 'seed-2_sample-0'
    sample.mkdir()
    structure = sample / 'model.cif'
    structure.touch()
    (sample / 'confidences.json').write_text('{"pae": [[1]], "contact_probs": [[0.1]]}')
    folder = tmp_path / 'seed-2_distogram'
    folder.mkdir()
    plain = tmp_path / 'source.npz'
    values = np.zeros((1, 1, 64), np.float32)
    values[..., 0] = 0.75
    values[..., -1] = 0.25
    np.savez(plain, distogram=values)
    path = folder / ('prediction.npz' + suffix)
    with opener(path, 'wb') as f:
        f.write(plain.read_bytes())

    assert data.distogram_file_ok(str(path))
    loaded, edges, logits = data.read_distogram(str(path))
    np.testing.assert_array_equal(loaded, values)
    assert not logits
    assert data.find_prediction_files(str(structure)).distogram_path == str(path)
    runs = data.agreement_runs(str(structure))
    assert runs[0]['source'] == 'distogram'
    assert runs[0]['path'] == str(path)
    assert data.run_contact_probs(runs[0])[0, 0] == pytest.approx(0.75)


def _distogram(n=3):
    values = np.zeros((n, n, 64), np.float32)
    values[..., 0] = 0.75
    values[..., -1] = 0.25
    return values


@pytest.mark.parametrize('name,opener,compress', [
    ('d.npz', open, False), ('d_c.npz', open, True),
    ('d.npz.gz', gzip.open, False), ('d.npz.xz', lzma.open, False)])
def test_progress_counts_the_file_and_data_are_unchanged(tmp_path, name, opener, compress):
    values = _distogram()
    plain = tmp_path / 'source.npz'
    (np.savez_compressed if compress else np.savez)(plain, distogram=values)
    path = tmp_path / name
    with opener(path, 'wb') as f:
        f.write(plain.read_bytes())

    progress = data.LoadProgress()
    assert progress.fraction() is None and 'Loading' in progress.message()
    metrics = data.load_distogram_metrics(str(path), progress=progress)

    np.testing.assert_allclose(metrics['contact'], 0.75)
    assert progress.total == path.stat().st_size
    assert progress.bytes_read > 0
    assert progress.computed is not None
    assert 0.9 <= progress.fraction() <= 1
    assert 'computing' in progress.message()


def test_progress_on_a_plain_pickle_and_large_reads(tmp_path, monkeypatch):
    import pickle
    path = tmp_path / 'result.pkl'
    logits = np.log(_distogram(40) + 1e-6)
    path.write_bytes(pickle.dumps({'distogram': {'logits': logits, 'bin_edges': data.AF_BIN_EDGES}},
                                  protocol=4))
    monkeypatch.setattr(data._ProgressFile, 'CHUNK', 1024)      # Every large read is split
    progress = data.LoadProgress()
    values, _, are_logits = data.read_distogram(str(path), progress)
    np.testing.assert_array_equal(values, logits)
    assert are_logits
    assert progress.bytes_read == path.stat().st_size


def test_progress_file_read_all_and_read_part(tmp_path):
    path = tmp_path / 'blob'
    path.write_bytes(bytes(range(256)) * 100)
    progress = data.LoadProgress()
    with data._open_maybe_gz(str(path), progress) as f:
        assert f.read(10) == bytes(range(10))
        assert f.tell() == 10
        rest = f.read()
    assert len(rest) == 25590
    assert progress.bytes_read == 25600


def test_progress_over_several_files(monkeypatch):
    progress = data.LoadProgress(steps=4)
    assert progress.fraction() == 0
    progress.next_file(1, 'seed 7')
    assert progress.fraction() == pytest.approx(0.25)
    assert progress.message() == 'Loading seed 7 (2 of 4) ...'
    progress.total = 2_000_000_000
    progress.bytes_read = 1_000_000_000
    assert progress.fraction() == pytest.approx(1.45 / 4)
    monkeypatch.setattr(data.time, 'monotonic', lambda: progress.started + 100)
    assert progress.message() == 'Loading seed 7 (2 of 4): 1.00 GB of 2.00 GB, about 3 min left'
    progress.computed = 0.5
    assert progress.message() == 'Loading seed 7 (2 of 4): computing contact probabilities'
    progress.next_file(2, 'seed 8')
    assert progress.computed is None and progress.bytes_read == 0


@pytest.mark.parametrize('known_first,steps', [(False, 2), (True, 1)])
def test_agreement_reports_only_the_files_it_reads(tmp_path, known_first, steps):
    from alphalens import agreement
    values = np.zeros((2, 2, 64), np.float32)
    values[..., 0] = 0.75
    values[..., -1] = 0.25
    runs = []
    for seed in (1, 2):
        path = tmp_path / f'seed-{seed}.npz'
        np.savez(path, distogram=values)
        runs.append({'label': f'seed {seed}', 'kind': 'seed', 'source': 'distogram', 'path': str(path)})
    known = {runs[0]['path']: np.full((2, 2), 0.75, np.float32)} if known_first else None
    progress = data.LoadProgress()
    agr, msgs = agreement.load(runs, 8.0, 2, None, known, progress)
    assert msgs == []
    assert agr.n == 2
    assert progress.steps == steps
    assert progress.step == steps - 1
    assert 'seed 2' in progress.label
    assert progress.bytes_read > 0


def test_progress_message_and_time_left(monkeypatch):
    progress = data.LoadProgress()
    progress.total = 2_000_000_000
    progress.bytes_read = 500_000_000
    assert progress.fraction() == pytest.approx(0.225)
    assert progress.message() == 'Loading distogram: 500 MB of 2.00 GB'      # Too early to tell
    monkeypatch.setattr(data.time, 'monotonic', lambda: progress.started + 100)
    assert progress.seconds_left() == pytest.approx(300)
    assert progress.message() == 'Loading distogram: 500 MB of 2.00 GB, about 5 min left'
    progress.bytes_read = 1_990_000_000
    assert 'less than a minute left' in progress.message()
    progress.bytes_read = 3_000_000_000                              # Rewinds count twice
    assert progress.fraction() == pytest.approx(0.9)
    progress.computed = 0.5
    assert progress.fraction() == pytest.approx(0.95)
    assert progress.seconds_left() is None


@pytest.mark.parametrize('ranking', [None, 'seed,sample,ranking_score\n',
                                    'seed,sample,ranking_score\n1,0,invalid\n'])
@pytest.mark.parametrize('distogram_seeds', [(1,), (1, 2)])
def test_top_af3_model_does_not_infer_seed_from_directory_order(tmp_path, ranking, distogram_seeds):
    model = tmp_path / 'test_model.cif'
    model.touch()
    (tmp_path / 'test_confidences.json').write_text('{"pae": [[1]], "contact_probs": [[0.1]]}')
    for seed in (1, 2):
        (tmp_path / f'seed-{seed}_sample-0').mkdir()
    for seed in distogram_seeds:
        folder = tmp_path / f'seed-{seed}_distogram'
        folder.mkdir()
        (folder / 'd.npz').touch()
    if ranking is not None:
        (tmp_path / 'ranking_scores.csv').write_text(ranking)
    files = data.find_prediction_files(str(model))
    assert files.model_name == 'top'
    assert files.distogram_path is None
    assert data.read_pae(files.pae_path)[1][0, 0] == pytest.approx(0.1)
    assert len(data._af3_ranked_samples(str(tmp_path))) == 2


def test_top_af3_model_uses_ranked_seed(tmp_path):
    model = tmp_path / 'test_model.cif'
    model.touch()
    (tmp_path / 'ranking_scores.csv').write_text('seed,sample,ranking_score\n1,0,0.1\n2,0,0.9\n')
    for seed in (1, 2):
        folder = tmp_path / f'seed-{seed}_distogram'
        folder.mkdir()
        (folder / 'd.npz').touch()
    files = data.find_prediction_files(str(model))
    assert files.model_name == 'seed-2_sample-0'
    assert files.distogram_path == str(tmp_path / 'seed-2_distogram' / 'd.npz')
