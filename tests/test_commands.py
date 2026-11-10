import csv
import json
import sys
import threading
import types

import numpy as np
import pytest

from alphalens import agreement, cmd


class UserError(Exception):
    pass


class PAE:
    def __init__(self, path, structure, matrix):
        from chimerax.atomic import Residue
        self.structure = structure
        self.pae_matrix = matrix
        self.matrix_size = self._full_size = len(matrix)
        self._rows = None
        self.metrics = None
        self.agreement = None
        self._agreement_pending = None
        self.residues = [Residue(c, 1, 'ALA', np.array([i * 3., 0, 0]))
                         for i, c in enumerate(('A', 'B'))]

    def reduce_matrix_to_residues_in_structure(self):
        return True

    def set_distogram_metrics(self, metrics, full_size):
        if full_size != self._full_size:
            raise ValueError('Distogram size mismatch')
        self.metrics = metrics

    @property
    def has_contact(self):
        return self.metrics is not None and 'contact' in self.metrics

    @property
    def has_distogram(self):
        return self.metrics is not None and 'std' in self.metrics

    def row_residues_or_atoms(self):
        return self.residues

    def row_residue_or_atom_name(self, i):
        return f'/{self.residues[i].chain_id}:1'


class UI:
    def __init__(self, gui=False):
        self.is_gui = gui
        self.pending = []

    def thread_safe(self, callback, *args):
        self.pending.append((callback, args))

    def drain(self):
        while self.pending:
            callback, args = self.pending.pop(0)
            callback(*args)


class Plot:
    threshold = 0.2

    def __init__(self, pae):
        self.pae = pae
        self.loaded = 0

    def closed(self):
        return False

    def display(self, shown):
        pass

    def agreement_loading(self, *args):
        pass

    def agreement_loaded(self, **kw):
        self.loaded += 1


@pytest.fixture
def session(monkeypatch):
    errors = types.ModuleType('chimerax.core.errors')
    errors.UserError = UserError
    monkeypatch.setitem(sys.modules, 'chimerax.core.errors', errors)
    pae_module = types.ModuleType('alphalens.pae')
    pae_module.AlphaLensPAE = PAE
    monkeypatch.setitem(sys.modules, 'alphalens.pae', pae_module)
    messages = []
    logger = types.SimpleNamespace(info=lambda *a, **k: messages.append(a),
                                   warning=lambda *a, **k: messages.append(a))
    return types.SimpleNamespace(ui=UI(), logger=logger)


def prediction(tmp_path):
    sample = tmp_path / 'seed-1_sample-0'
    sample.mkdir()
    structure = sample / 'model.cif'
    structure.touch()
    (sample / 'confidences.json').write_text('{"pae": [[1, 2], [3, 4]]}')
    folder = tmp_path / 'seed-1_distogram'
    folder.mkdir()
    path = folder / 'prediction.npz'
    path.write_bytes(b'not a numpy archive')
    return types.SimpleNamespace(filename=str(structure), deleted=False), path


@pytest.mark.parametrize('wait', [False, True])
def test_failed_distogram_export_raises_and_preserves_existing_file(session, tmp_path, wait):
    structure, _ = prediction(tmp_path)
    out = tmp_path / 'contacts.csv'
    out.write_text('previous export\n')
    with pytest.raises(UserError, match='distogram'):
        cmd.alphalens(session, structure, save_contacts=str(out), wait=wait)
    assert out.read_text() == 'previous export\n'


def test_failed_agreement_does_not_export_only_shown_model(session, tmp_path, monkeypatch):
    structure, _ = prediction(tmp_path)
    pae = PAE('', structure, np.zeros((2, 2)))
    pae.metrics = {'contact': np.ones((2, 2)), 'cutoff': 8.0}
    structure.alphafold_pae = pae
    monkeypatch.setattr(cmd, '_check_agreement_runs', lambda s: [{}, {}])
    monkeypatch.setattr(agreement, 'load', lambda *a: (None, ['unreadable seed']))
    out = tmp_path / 'contacts.csv'
    with pytest.raises(UserError, match='agreement'):
        cmd.alphalens(session, structure, mode='agreement', save_contacts=str(out), wait=True)
    assert not out.exists()


def test_wait_finishes_pending_agreement_and_exports_once(session, tmp_path, monkeypatch):
    structure, _ = prediction(tmp_path)
    pae = PAE('', structure, np.zeros((2, 2)))
    pae.metrics = {'contact': np.ones((2, 2)), 'cutoff': 8.0}
    structure.alphafold_pae = pae
    structure._alphalens_plot = plot = Plot(pae)
    runs = [{'kind': 'seed'}, {'kind': 'seed'}]
    monkeypatch.setattr(cmd, '_check_agreement_runs', lambda s: runs)
    started, release, queued = threading.Event(), threading.Event(), threading.Event()
    agr = agreement.Agreement(['seed 1', 'seed 2'], 'seed', np.ones((2, 2, 2)), 8.)

    def load(*args):
        started.set()
        assert release.wait(5), 'test did not release the worker'
        return agr, []

    monkeypatch.setattr(agreement, 'load', load)
    session.ui.is_gui = True
    queue = session.ui.thread_safe
    session.ui.thread_safe = lambda *args: (queue(*args), queued.set())
    callbacks = []
    cmd.start_agreement(session, pae, plot, 8., on_loaded=lambda: callbacks.append('first'))
    assert started.wait(5)
    release.set()
    assert queued.wait(5)
    # The worker finished, but ChimeraX has not processed its GUI callback yet.
    out = tmp_path / 'contacts.csv'
    exports = []
    save = cmd.save_contacts_csv

    def export(*args, **kw):
        exports.append(args[2])
        return save(*args, **kw)

    monkeypatch.setattr(cmd, 'save_contacts_csv', export)
    cmd.alphalens(session, structure, mode='agreement', save_contacts=str(out), wait=True)
    assert pae.agreement is agr
    assert callbacks == ['first']
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]['seeds'] == '2'
    before = out.read_bytes()
    session.ui.drain()
    assert callbacks == ['first']
    assert len(exports) == 1
    assert out.read_bytes() == before


def test_wait_finishes_pending_distogram_before_export(session, tmp_path, monkeypatch):
    structure, path = prediction(tmp_path)
    pae = PAE('', structure, np.zeros((2, 2)))
    structure.alphafold_pae = pae
    structure._alphalens_plot = Plot(pae)
    np.savez(path, distogram=np.ones((2, 2, 64), np.float32) / 64)
    session.ui.is_gui = True
    queued = threading.Event()
    queue = session.ui.thread_safe
    session.ui.thread_safe = lambda *args: (queue(*args), queued.set())
    callbacks = []
    cmd.start_distogram(session, structure, pae, None, str(path), 8.,
                        on_loaded=lambda: callbacks.append('loaded'))
    assert queued.wait(5)
    assert not pae.has_contact
    out = tmp_path / 'contacts.csv'
    cmd.alphalens(session, structure, save_contacts=str(out), wait=True)
    assert pae.has_distogram
    assert out.exists()
    assert callbacks == ['loaded']
    session.ui.drain()
    assert callbacks == ['loaded']


def test_contact_probs_are_usable_while_the_distogram_loads(session, tmp_path):
    structure, path = prediction(tmp_path)
    conf = tmp_path / 'seed-1_sample-0' / 'confidences.json'
    conf.write_text(json.dumps({'pae': [[1, 2], [3, 4]],
                               'contact_probs': [[0., .9], [.9, 0.]]}))
    pae, dpath, contact_probs, _ = cmd.prepare_prediction(session, structure)
    assert dpath == str(path)
    assert contact_probs is not None
    assert pae.has_contact and not pae.has_distogram
    assert pae.metrics['contact_only']


def test_contact_probs_wait_for_the_distogram_with_another_cutoff(session, tmp_path):
    structure, path = prediction(tmp_path)
    conf = tmp_path / 'seed-1_sample-0' / 'confidences.json'
    conf.write_text(json.dumps({'pae': [[1, 2], [3, 4]],
                               'contact_probs': [[0., .9], [.9, 0.]]}))
    pae, dpath, _, _ = cmd.prepare_prediction(session, structure, cutoff=6.0)
    assert dpath == str(path)
    assert not pae.has_contact


def test_export_is_refused_while_the_distogram_is_still_loading(session, tmp_path):
    structure, _ = prediction(tmp_path)
    pae = PAE('', structure, np.zeros((2, 2)))
    pae.metrics = {'contact': np.ones((2, 2)), 'cutoff': 8.0, 'contact_only': True}
    pae._distogram_load = types.SimpleNamespace(pending=True, wait=lambda: None)
    structure.alphafold_pae = pae
    structure._alphalens_plot = Plot(pae)
    session.ui.is_gui = True
    out = tmp_path / 'contacts.csv'
    with pytest.raises(UserError, match='still loading'):
        cmd.alphalens(session, structure, save_contacts=str(out))
    assert not out.exists()


def test_background_load_reports_pending_until_applied(session):
    applied = []
    load = cmd._BackgroundLoad(lambda: (1,), applied.append)
    assert load.pending
    load.start(session, True, 'test')
    assert applied == [1]
    assert not load.pending


def test_bad_distogram_keeps_passive_pae_view_available(session, tmp_path):
    structure, _ = prediction(tmp_path)
    pae = cmd.alphalens(session, structure, wait=True)
    assert not pae.has_contact
    np.testing.assert_array_equal(pae.pae_matrix, [[1, 2], [3, 4]])


def test_failed_export_does_not_poison_a_later_export(session, tmp_path):
    structure, path = prediction(tmp_path)
    np.savez(path, distogram=np.ones((2, 2, 64), np.float32) / 64)
    with pytest.raises(IsADirectoryError):
        cmd.alphalens(session, structure, save_contacts=str(tmp_path), wait=True)
    out = tmp_path / 'contacts.csv'
    cmd.alphalens(session, structure, save_contacts=str(out), wait=True)
    assert out.exists()


@pytest.mark.parametrize('mode', [None, 'std'])
def test_bad_distogram_af3_fallback_is_only_used_for_supported_operations(session, tmp_path, mode):
    structure, _ = prediction(tmp_path)
    conf = tmp_path / 'seed-1_sample-0' / 'confidences.json'
    conf.write_text(json.dumps({'pae': [[1, 2], [3, 4]],
                               'contact_probs': [[0., .9], [.9, 0.]]}))
    out = tmp_path / 'contacts.csv'
    if mode == 'std':
        with pytest.raises(UserError, match='distogram'):
            cmd.alphalens(session, structure, mode=mode, save_contacts=str(out), wait=True)
        assert not out.exists()
    else:
        cmd.alphalens(session, structure, save_contacts=str(out), wait=True)
        with out.open() as f:
            rows = list(csv.DictReader(f))
        assert rows[0]['p_contact'] == '0.900'


def test_requested_distogram_provenance_is_available_before_loading(session, tmp_path):
    structure, path = prediction(tmp_path)
    external = tmp_path / 'external.npz'
    path.rename(external)
    pae, dpath, _, _ = cmd.prepare_prediction(session, structure, distogram=str(external))
    assert not pae.has_contact
    assert pae._distogram_file == dpath == str(external)
