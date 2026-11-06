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
