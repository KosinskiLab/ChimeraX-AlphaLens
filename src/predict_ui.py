"""ChimeraX dialog for the experimental local AF2-Multimer runner."""
from datetime import datetime
from pathlib import Path
import signal
import os
import time
import uuid

from Qt.QtCore import QTimer
from Qt.QtWidgets import (QApplication, QCheckBox, QDialog, QFormLayout, QHBoxLayout,
                          QLabel, QLineEdit, QPushButton, QSpinBox, QVBoxLayout)
from . import predict


def snapshot(plot, flanks):
    """Read the live source once, before starting a subprocess."""
    from chimerax.atomic import Residue
    import numpy as np
    pae = plot.pae
    if pae.structure.deleted or plot.closed():
        raise ValueError('The source plot or structure has been closed.')
    indices = {r: i for chain in pae.structure.chains for i, r in enumerate(chain.residues)
               if r is not None}
    rows = []
    for r in pae.row_residues_or_atoms():
        if not isinstance(r, Residue) or r.polymer_type != Residue.PT_AMINO or r not in indices:
            rows.append(None)
        else:
            rows.append(dict(chain_id=r.chain_id, number=int(r.number), insertion_code=r.insertion_code,
                             aa=r.one_letter_code, chain_index=indices[r]))
    chains = predict.crop_chains(rows, plot._prediction_region, flanks)
    rra = pae.row_residues_or_atoms()
    for c in chains:
        for entry in c['residue_map']:
            ca = rra[entry['source_row']].find_atom('CA')
            entry['source_ca_scene'] = [float(v) for v in ca.scene_coord] if ca is not None else None
    a, b = [c['core_rows'] for c in chains]
    matrix = pae.matrix('pae')
    plddt = pae.row_plddt()
    source = dict(model=pae.structure.name, model_id=pae.structure.id_string,
                  file=getattr(pae.structure, 'filename', None),
                  core_mean_pae=float((matrix[np.ix_(a, b)].mean() + matrix[np.ix_(b, a)].mean()) / 2),
                  core_mean_plddt=float(np.nanmean(plddt[a + b])))
    if not np.isfinite(source['core_mean_plddt']):
        source['core_mean_plddt'] = None
    return chains, source


def source_alignment(plot):
    """Only recognize a source MSA when its query exactly matches the plotted tokens."""
    filename = getattr(plot.pae.structure, 'filename', None)
    if not filename:
        return None
    from chimerax.atomic import Residue
    residues = plot.pae.row_residues_or_atoms()
    if any(not isinstance(r, Residue) or r.polymer_type != Residue.PT_AMINO for r in residues):
        return None
    query = ''.join(r.one_letter_code for r in residues)
    p = Path(filename)
    for candidate in (p.with_suffix('.a3m'), p.parent / 'input.a3m'):
        if candidate.is_file():
            try:
                predict.validate_a3m(candidate, query)
                return candidate, query
            except (ValueError, OSError):
                pass
    return None


def import_result(session, job):
    from . import cmd
    import numpy as np
    structure = cmd._open_structure(session, str(job.directory / 'fold.pdb'))
    structure.name = f'{job.directory.name} LocalFold AF2-Multimer'
    pae = cmd.alphalens(session, structure, pae_file=str(job.directory / 'fold_confidences.json'), wait=True)
    pae.color_plddt(log_command=False)
    # Superpose on the first selected chain only, preserving the predicted interface.
    # Snapshot coordinates remain valid even if the source model has been closed.
    from chimerax.geometry import align_points
    pairs = [(r.find_atom('CA'), entry['source_ca_scene'])
             for r, entry in zip(pae.row_residues_or_atoms(), job.chains[0]['residue_map'])
             if entry['source_row'] in job.chains[0]['core_rows'] and entry.get('source_ca_scene') is not None]
    pairs = [(atom, xyz) for atom, xyz in pairs if atom is not None]
    alignment_rmsd = None
    if len(pairs) >= 3:
        transform, alignment_rmsd = align_points(np.array([atom.coord for atom, _ in pairs]),
                                                 np.array([xyz for _, xyz in pairs]))
        structure.scene_position = transform
    # Compare the same residue pairs, excluding added flanks from the score comparison.
    offsets = [0, len(job.chains[0]['sequence'])]
    core = [[offset + i for i, r in enumerate(c['residue_map']) if r['source_row'] in c['core_rows']]
            for offset, c in zip(offsets, job.chains)]
    a, b = core
    matrix = pae.matrix('pae')
    comparison = dict(source=job.source,
                      aligned_source_chain=job.chains[0]['source_chain'],
                      aligned_chain_ca_rmsd=alignment_rmsd,
                      predicted_core_mean_pae=float((matrix[np.ix_(a, b)].mean() + matrix[np.ix_(b, a)].mean()) / 2),
                      predicted_core_mean_plddt=float(np.mean(pae.row_plddt()[a + b])))
    predict.write_json(job.directory / 'comparison.json', comparison)
    session.logger.info(f'Local prediction: {job.directory}\nipTM {job.result["iptm"]:.3f}; '
                        f'mean pLDDT {job.result["mean_plddt"]:.1f}. '
                        f'Same selected pairs: mean PAE {job.source["core_mean_pae"]:.2f} → '
                        f'{comparison["predicted_core_mean_pae"]:.2f} Å. '
                        'Cropping can change confidence; this does not establish better accuracy.')
    return structure, comparison


class PredictionDialog(QDialog):
    def __init__(self, plot):
        super().__init__(plot.session.ui.main_window)
        self.plot, self.session = plot, plot.session
        self.source_msa = source_alignment(plot)
        self.job = self.output_model = self.comparison = None
        self.setWindowTitle('Re-predict interface · experimental')
        self.resize(660, 400)
        layout = QVBoxLayout(self)
        intro = QLabel('Local AF2-Multimer on Apple Silicon • two protein chains • ≤400 residues\n'
                       'The first run downloads public AF2 weights (~77 MB).')
        intro.setWordWrap(True); layout.addWidget(intro)
        form = QFormLayout(); form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        layout.addLayout(form)
        self.flanks = QSpinBox(); self.flanks.setRange(0, 100); self.flanks.setValue(10)
        form.addRow('Flanking residues on each side', self.flanks)
        self.preview = QLabel(); form.addRow('Crop', self.preview)
        self.executable = QLineEdit(predict.default_executable()); form.addRow('LocalFold executable', self.executable)
        self.output = QLineEdit(str(predict.data_directory() / 'predictions')); form.addRow('Results directory', self.output)
        self.msa = QLineEdit(); self.msa.setPlaceholderText('Existing merged A3M for exactly these two sequences')
        form.addRow('MSA file (optional)', self.msa)
        self.reuse = QCheckBox('Reuse and crop the source MSA (no upload)')
        self.reuse.setEnabled(self.source_msa is not None)
        self.reuse.setChecked(self.source_msa is not None)
        form.addRow(self.reuse)
        api = os.environ.get('LOCALFOLD_MMSEQS_API', 'https://api.colabfold.com').rstrip('/')
        self.search = QCheckBox(f'Search remotely if uncached: send sequences to {api}')
        form.addRow(self.search)
        self.seed = QSpinBox(); self.seed.setRange(0, 2147483647); form.addRow('Random seed', self.seed)
        self.recycles = QSpinBox(); self.recycles.setRange(0, 10); self.recycles.setValue(3)
        form.addRow('Recycles', self.recycles)
        note = QLabel('PAE and pLDDT are compared on the same selected residues. Higher confidence after\n'
                      'cropping is not proof of better accuracy. This weight bundle has no contact-probability head.')
        note.setWordWrap(True); layout.addWidget(note)
        self.status = QLabel('Ready'); self.status.setWordWrap(True); layout.addWidget(self.status)
        buttons = QHBoxLayout(); layout.addLayout(buttons)
        self.start_button = QPushButton('Predict'); self.cancel_button = QPushButton('Cancel job')
        self.cancel_button.setEnabled(False)
        buttons.addWidget(self.start_button); buttons.addWidget(self.cancel_button)
        self.start_button.clicked.connect(self.start)
        self.cancel_button.clicked.connect(lambda: self.job.cancel() if self.job else None)
        self.flanks.valueChanged.connect(self.refresh)
        self.reuse.toggled.connect(lambda checked: self.search.setChecked(False) if checked else None)
        self.search.toggled.connect(lambda checked: self.reuse.setChecked(False) if checked else None)
        self.reuse.toggled.connect(self.msa_controls)
        self.search.toggled.connect(self.msa_controls)
        self.timer = QTimer(self); self.timer.setInterval(100); self.timer.timeout.connect(self.poll)
        QApplication.instance().aboutToQuit.connect(self.shutdown)
        self.msa_controls()
        self.refresh()

    def msa_controls(self, *args):
        self.msa.setEnabled(not self.reuse.isChecked() and not self.search.isChecked())

    def refresh(self):
        try:
            chains, _ = snapshot(self.plot, self.flanks.value())
            self.preview.setText(' + '.join(f'{c["source_chain"]}: {len(c["sequence"])} residues' for c in chains))
            self.preview.setToolTip('\n'.join(c['sequence'] for c in chains))
            self.start_button.setEnabled(True)
        except Exception as e:
            self.preview.setText(str(e)); self.start_button.setEnabled(False)

    def start(self):
        try:
            chains, source = snapshot(self.plot, self.flanks.value())
            name = datetime.now().strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:6]
            msa = (self.msa.text().strip() or None) if self.msa.isEnabled() else None
            crop = None
            if self.reuse.isChecked():
                msa, query = self.source_msa
                crop = dict(source_query=query, columns=[r['source_row'] for c in chains for r in c['residue_map']])
            self.job = predict.PredictionJob(Path(self.output.text()).expanduser() / name, chains,
                          executable=self.executable.text(), search=self.search.isChecked(),
                          msa=msa, msa_crop=crop, seed=self.seed.value(),
                          recycles=self.recycles.value(), source=source).start()
        except Exception as e:
            self.status.setText(str(e)); return
        for widget in (self.flanks, self.executable, self.output, self.msa, self.reuse, self.search, self.seed, self.recycles,
                       self.start_button):
            widget.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.timer.start()
        self.status.setText('Running…')

    def poll(self):
        try:
            state = self.job.poll()
            if state == 'running':
                elapsed = time.monotonic() - self.job.started
                with (self.job.directory / 'prediction.log').open('rb') as log:
                    log.seek(0, 2); size = log.tell(); log.seek(max(0, size - 1500))
                    lines = log.read().decode(errors='replace').splitlines()
                latest = lines[-1][-200:] if lines else 'Starting LocalFold…'
                self.status.setText(f'Running · {elapsed:.1f} s · MSA: {self.job.manifest["msa_source"]}\n{latest}')
                return
            self.timer.stop(); self.cancel_button.setEnabled(False)
            if state == 'complete':
                self.output_model, self.comparison = import_result(self.session, self.job)
                old = self.job.source['core_mean_pae']; new = self.comparison['predicted_core_mean_pae']
                before = self.job.source['core_mean_plddt']
                plddt = (f'Same residues: pLDDT {before:.1f} → '
                         f'{self.comparison["predicted_core_mean_plddt"]:.1f}\n') if before is not None else ''
                self.status.setText(f'Complete in {time.monotonic()-self.job.started:.1f} s · '
                                    f'ipTM {self.job.result["iptm"]:.3f}\n'
                                    f'Same selected pairs: mean PAE {old:.2f} → {new:.2f} Å\n'
                                    + plddt +
                                    f'Results: {self.job.directory}')
            else:
                self.status.setText(self.job.error or 'Cancelled. Partial files retained in the results directory.')
        except Exception as e:
            self.timer.stop(); self.cancel_button.setEnabled(False)
            # A log/read failure must not leave an unpolled GPU job running.
            try:
                self.shutdown()
            except Exception:
                self.job._release()
            self.status.setText(f'Could not load prediction: {e}. Files: {self.job.directory}')
        finally:
            if self.job.state != 'running':
                for widget in (self.flanks, self.executable, self.output, self.msa, self.search, self.seed,
                               self.recycles, self.start_button):
                    widget.setEnabled(True)
                self.reuse.setEnabled(self.source_msa is not None)
                self.msa_controls()

    def shutdown(self):
        if self.job and self.job.state == 'running':
            self.job.cancel()
            try:
                self.job.process.wait(timeout=1)
            except Exception:
                try:
                    os.killpg(self.job.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.job.process.wait()
            self.job.poll()

    def closeEvent(self, event):
        if self.job and self.job.state == 'running':
            self.job.cancel()
            self.status.setText('Cancelling… Close again when the process stops.')
            event.ignore()
        else:
            QApplication.instance().aboutToQuit.disconnect(self.shutdown)
            dialogs = getattr(self.session, '_alphalens_prediction_dialogs', [])
            if self in dialogs:
                dialogs.remove(self)
            self.plot = None
            event.accept()
            self.deleteLater()


def show_prediction_dialog(plot):
    dialog = PredictionDialog(plot)
    dialogs = getattr(plot.session, '_alphalens_prediction_dialogs', None)
    if dialogs is None:
        dialogs = plot.session._alphalens_prediction_dialogs = []
    dialogs.append(dialog)  # Keep the job alive even if the source plot is closed.
    dialog.show()
    return dialog
