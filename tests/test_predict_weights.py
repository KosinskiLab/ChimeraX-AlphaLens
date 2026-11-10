import hashlib
import json
import pytest

from alphalens import predict_weights as weights


def test_correct_head_is_added_to_separate_cache_without_altering_upstream(tmp_path, monkeypatch):
    original = tmp_path/'model-multimer'; original.mkdir()
    shard = original/'weights.bin'; shard.write_bytes(b'example trunk weights')
    data = json.dumps(dict(tensors=dict(trunk=dict(file='weights.bin')))).encode()
    (original/'manifest.json').write_bytes(data)
    monkeypatch.setattr(weights, 'MANIFEST_SHA256', hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(weights, 'download', lambda *args: pytest.fail('Unexpected download'))
    root = weights.prepare(tmp_path)
    enhanced = root/'model-multimer'
    result = json.loads((enhanced/'manifest.json').read_text())
    assert (original/'manifest.json').read_bytes() == data
    assert shard.read_bytes() == b'example trunk weights'
    assert (enhanced/'weights.bin').read_bytes() == shard.read_bytes()
    assert result['tensors']['af2DistogramHalfLogitsWeights']['shape'] == [128, 64]
    assert result['tensors']['af2DistogramHalfLogitsBias']['byteOffset'] == 32768
    head = (enhanced/'alphalens-distogram.f32.bin').read_bytes()
    assert len(head) == 33024 and hashlib.sha256(head).hexdigest() == weights.HEAD_SHA256
    assert weights.prepare(tmp_path) == root


def test_head_corruption_and_unpinned_manifest_are_rejected(tmp_path, monkeypatch):
    head = tmp_path/'bad-head'; head.write_bytes(b'bad')
    monkeypatch.setattr(weights, 'HEAD_FILE', head)
    with pytest.raises(ValueError, match='damaged'):
        weights.prepare(tmp_path)
    monkeypatch.undo()
    monkeypatch.setattr(weights, 'download', lambda url, target: target.write_bytes(b'{}'))
    with pytest.raises(ValueError, match='pinned'):
        weights.prepare(tmp_path)
