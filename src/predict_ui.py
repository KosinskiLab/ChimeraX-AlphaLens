"""ChimeraX dialog for experimental local AF2 and AF3 prediction."""
from datetime import datetime
from pathlib import Path
import signal
import os
import time
import uuid

from Qt.QtCore import QTimer
from Qt.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QFormLayout, QHBoxLayout,
                          QLabel, QLineEdit, QPlainTextEdit, QProgressBar, QPushButton,
                          QScrollArea, QSpinBox, QVBoxLayout, QWidget)
from . import predict
from . import predict_contacts


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
    output_pairs = predict_contacts.region_pairs(chains, plot._prediction_region)
    if not len(output_pairs):
        raise ValueError('Select a region containing inter-chain residue pairs.')
    source_rows = np.array([r['source_row'] for c in chains for r in c['residue_map']])
    source_pairs = source_rows[output_pairs]
    a, b = source_pairs.T
    matrix = pae.matrix('pae')
    plddt = pae.row_plddt()
    source = dict(model=pae.structure.name, model_id=pae.structure.id_string,
                  file=getattr(pae.structure, 'filename', None),
                  region=plot._prediction_region,
                  contact_snapshot=predict_contacts.measure(pae, source_pairs),
                  core_mean_pae=float((matrix[a, b].mean() + matrix[b, a].mean()) / 2),
                  core_mean_plddt=float(np.nanmean(plddt[np.unique(source_pairs)])))
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
    model_name = 'AF3' if job.backend == 'af3' else 'AF2-Multimer' if len(job.chains) > 1 else 'AF2-monomer'
    structure.name = f'{job.directory.name} LocalFold {model_name}'
    pae = cmd.alphalens(session, structure, pae_file=str(job.directory / 'fold_confidences.json'), wait=True)
    pae.color_plddt(log_command=False)
    contact_pairs = predict_contacts.region_pairs(job.chains, job.source.get('region'))
    if len(contact_pairs):
        job.contact_report = predict_contacts.export(job.directory, job.chains, contact_pairs,
            predict_contacts.measure(pae, contact_pairs), job.source.get('contact_snapshot'))
        predict.write_json(job.directory / 'contact_report.json', job.contact_report)
        session.logger.info(predict_contacts.summary(job.contact_report))
    if job.source.get('input_mode') == 'sequence':
        confidence = (f'ipTM {job.result["iptm"]:.3f}' if len(job.chains) > 1
                      else f'pTM {job.result["ptm"]:.3f}')
        session.logger.info(f'Local prediction: {job.directory}\n{confidence}; '
                            f'mean pLDDT {job.result["mean_plddt"]:.1f}. '
                            'Predicted from sequence; no source-model comparison is available.')
        return structure, None
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
    a, b = contact_pairs.T
    matrix = pae.matrix('pae')
    comparison = dict(source=job.source,
                      aligned_source_chain=job.chains[0]['source_chain'],
                      aligned_chain_ca_rmsd=alignment_rmsd,
                      predicted_core_mean_pae=float((matrix[a, b].mean() + matrix[b, a].mean()) / 2),
                      predicted_core_mean_plddt=float(np.mean(pae.row_plddt()[np.unique(contact_pairs)])),
                      contacts=job.contact_report)
    predict.write_json(job.directory / 'comparison.json', comparison)
    session.logger.info(f'Local prediction: {job.directory}\nipTM {job.result["iptm"]:.3f}; '
                        f'mean pLDDT {job.result["mean_plddt"]:.1f}. '
                        f'Same selected pairs: mean PAE {job.source["core_mean_pae"]:.2f} → '
                        f'{comparison["predicted_core_mean_pae"]:.2f} Å.')
    return structure, comparison


class PredictionDialog(QDialog):
    def __init__(self, plot=None, session=None):
        session = plot.session if plot is not None else session
        super().__init__(session.ui.main_window)
        self.plot, self.session = plot, session
        self.source_msa = source_alignment(plot) if plot is not None else None
        self.entries = self.lookup_future = None
        self.job = self.output_model = self.comparison = None
        self.setWindowTitle('Local protein prediction · experimental')
        self.resize(660, 400)
        layout = QVBoxLayout(self)
        intro = QLabel('Local protein prediction on Apple Silicon\n'
                       'Model weights download on the first run and are cached locally.')
        intro.setWordWrap(True); layout.addWidget(intro)
        self.input_widgets = []
        if plot is None:
            label = QLabel('Paste a raw sequence, multiple FASTA entries, or UniProt IDs.\n'
                           'UniProt lookup sends only the IDs to UniProt; review the sequences before predicting.')
            label.setWordWrap(True); layout.addWidget(label)
            self.input_text = QPlainTextEdit(); self.input_text.setMaximumHeight(100)
            self.input_text.setPlaceholderText('P69905\nP68871\n\nor >chain_A\nSEQUENCE\n>chain_B\nSEQUENCE')
            layout.addWidget(self.input_text)
            self.load_button = QPushButton('Load / resolve sequences'); layout.addWidget(self.load_button)
            self.resolved = QPlainTextEdit(); self.resolved.setReadOnly(True); self.resolved.setMaximumHeight(100)
            self.resolved.setPlaceholderText('Resolved sequences will appear here for review.')
            layout.addWidget(self.resolved)
            self.input_widgets = [self.input_text, self.load_button]
        form = QFormLayout(); form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        layout.addLayout(form)
        self.backend = QComboBox(); self.backend.addItem('AlphaFold 2 (monomer / multimer)', 'af2')
        self.backend.addItem('AlphaFold 3', 'af3')
        form.addRow('Model', self.backend)
        self.af3_terms = QCheckBox('I accept DeepMind’s AF3 terms for eligible non-commercial use')
        self.af3_terms.setChecked(predict.af3_terms_accepted())
        form.addRow(self.af3_terms)
        self.af3_terms_links = QLabel('<a href="https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md">AF3 parameter terms</a> · '
                                     '<a href="https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_PROHIBITED_USE_POLICY.md">Prohibited-use policy</a>')
        self.af3_terms_links.setOpenExternalLinks(True); form.addRow(self.af3_terms_links)
        self.af3_steps = QSpinBox(); self.af3_steps.setRange(1, 1000); self.af3_steps.setValue(200)
        self.af3_steps_label = QLabel('AF3 diffusion steps')
        form.addRow(self.af3_steps_label, self.af3_steps)
        for widget in (self.af3_terms, self.af3_terms_links, self.af3_steps, self.af3_steps_label):
            widget.hide()
        self.flanks = QSpinBox(self); self.flanks.setRange(0, 100); self.flanks.setValue(10)
        if plot is not None:
            form.addRow('Flanking residues on each side', self.flanks)
        else:
            self.flanks.hide()
            self.ranges = []
            self.range_area = QScrollArea(); self.range_area.setWidgetResizable(True)
            self.range_area.setFixedHeight(100)
            ranges_widget = QWidget(); self.range_form = QFormLayout(ranges_widget)
            self.range_area.setWidget(ranges_widget)
            form.addRow('Chain ranges (optional)', self.range_area)
        self.preview = QLabel(); form.addRow('Selected sequences', self.preview)
        self.size_warning = QLabel(); self.size_warning.setWordWrap(True)
        self.size_warning.setMinimumHeight(2*self.size_warning.fontMetrics().lineSpacing()+4)
        self.size_warning.setStyleSheet('color: #a65c00;')
        form.addRow(self.size_warning)
        self.executable = QLineEdit(predict.default_executable()); form.addRow('LocalFold executable', self.executable)
        self.output = QLineEdit(str(predict.data_directory() / 'predictions')); form.addRow('Results directory', self.output)
        self.msa = QLineEdit(); self.msa.setPlaceholderText('Existing A3M for exactly the selected sequence(s)')
        form.addRow('Existing MSA (A3M)', self.msa)
        self.reuse = QCheckBox('Reuse and crop the source MSA (no upload)')
        self.reuse.setEnabled(self.source_msa is not None)
        self.reuse.setChecked(self.source_msa is not None)
        if self.source_msa is None:
            self.reuse.setToolTip('No matching .a3m or input.a3m beside this structure. '
                                 'MSAs embedded in AF3 JSON are not imported automatically.')
        if plot is not None:
            form.addRow(self.reuse)
        api = os.environ.get('LOCALFOLD_MMSEQS_API', 'https://api.colabfold.com').rstrip('/')
        self.search = QCheckBox(f'Search remotely if uncached: send sequences to {api}')
        form.addRow(self.search)
        self.msa_hint = QLabel('Choose an MSA to enable Predict: enter an A3M file above '
                              'or check “Search remotely if uncached”.')
        self.msa_hint.setWordWrap(True)
        self.msa_hint.setStyleSheet('color: #a65c00; font-weight: bold;')
        self.msa_hint.setMinimumHeight(2*self.msa_hint.fontMetrics().lineSpacing()+4)
        form.addRow(self.msa_hint)
        self.seed = QSpinBox(); self.seed.setRange(0, 2147483647); self.seed.setValue(predict.random_seed())
        form.addRow('Seed', self.seed)
        self.randomize_seed = QCheckBox('Generate a new random seed for each prediction')
        self.randomize_seed.setChecked(True); self.seed.setEnabled(False)
        self.randomize_seed.toggled.connect(lambda checked: self.seed.setEnabled(not checked))
        form.addRow(self.randomize_seed)
        self.recycles = QSpinBox(); self.recycles.setRange(0, 10); self.recycles.setValue(3)
        form.addRow('Recycles', self.recycles)
        self.time_limit = QSpinBox(); self.time_limit.setRange(0, 10080)
        self.time_limit.setSpecialValueText('No limit'); self.time_limit.setValue(0)
        form.addRow('Time limit (minutes)', self.time_limit)
        self.status = QPlainTextEdit('Ready'); self.status.setReadOnly(True)
        self.status.setFixedHeight(5*self.status.fontMetrics().lineSpacing()+16)
        layout.addWidget(self.status)
        self.progress_label = QLabel('Not started'); self.progress_label.setWordWrap(True)
        layout.addWidget(self.progress_label)
        self.progress_bar = QProgressBar(); self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0); self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)
        buttons = QHBoxLayout(); layout.addLayout(buttons)
        self.start_button = QPushButton('Predict'); self.cancel_button = QPushButton('Cancel job')
        self.cancel_button.setEnabled(False)
        buttons.addWidget(self.start_button); buttons.addWidget(self.cancel_button)
        self.start_button.clicked.connect(self.start)
        self.cancel_button.clicked.connect(self.cancel)
        self.flanks.valueChanged.connect(self.refresh)
        self.lookup_timer = QTimer(self); self.lookup_timer.setInterval(100)
        self.lookup_timer.timeout.connect(self.poll_lookup)
        if plot is None:
            self.input_text.textChanged.connect(self.invalidate_input)
            self.load_button.clicked.connect(self.load_sequences)
        self.reuse.toggled.connect(lambda checked: self.search.setChecked(False) if checked else None)
        self.search.toggled.connect(lambda checked: self.reuse.setChecked(False) if checked else None)
        self.reuse.toggled.connect(self.msa_controls)
        self.search.toggled.connect(self.msa_controls)
        self.msa.textChanged.connect(self.refresh)
        self.backend.currentIndexChanged.connect(self.backend_changed)
        self.af3_terms.toggled.connect(self.refresh)
        self.timer = QTimer(self); self.timer.setInterval(100); self.timer.timeout.connect(self.poll)
        QApplication.instance().aboutToQuit.connect(self.shutdown)
        self.msa_controls()
        self.refresh()

    def backend_changed(self):
        backend = self.backend.currentData()
        self.executable.setText(predict.default_executable(backend))
        for widget in (self.af3_terms, self.af3_terms_links, self.af3_steps, self.af3_steps_label):
            widget.setVisible(backend == 'af3')
        self.refresh()

    def display(self, shown):
        """ChimeraX's Tools menu calls display(True) on the returned tool."""
        self.setVisible(shown)

    def reject(self):
        # Escape must follow the same cancellation path as the window close button.
        self.close()

    def msa_controls(self, *args):
        self.msa.setEnabled(not self.reuse.isChecked() and not self.search.isChecked())
        self.refresh()

    def set_progress(self, stage, completed=None, total=None):
        self.progress_label.setText(stage)
        if total is None:
            self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(completed)

    def set_status(self, message):
        # A fixed viewport keeps timer updates from resizing the configuration
        # dialog. Long errors and result paths remain selectable and scrollable.
        self.status.setPlainText(message)

    def show_error(self, message):
        self.status.setStyleSheet('color: #c43b36; font-weight: bold;')
        self.set_status(message)
        self.session.logger.warning(message)

    def cancel(self):
        if self.job and self.job.state == 'running':
            self.job.cancel()
            self.set_progress('Cancelling…')

    def invalidate_input(self):
        self.entries = None
        self.resolved.clear()
        self.refresh()

    def load_sequences(self):
        from concurrent.futures import Future
        import threading
        from .predict_inputs import resolve_entries
        text = self.input_text.toPlainText()
        future = self.lookup_future = Future()
        # The worker owns only text and a Future, never Qt objects or the dialog.
        def resolve():
            try:
                future.set_result(resolve_entries(text))
            except Exception as e:
                future.set_exception(e)
        self.entries = None
        self.input_text.setEnabled(False); self.load_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.set_status('Resolving sequences…')
        threading.Thread(target=resolve, daemon=True, name='AlphaLens UniProt lookup').start()
        self.lookup_timer.start()

    def poll_lookup(self):
        if not self.lookup_future.done():
            return
        self.lookup_timer.stop()
        self.input_text.setEnabled(True); self.load_button.setEnabled(True)
        try:
            self.entries = self.lookup_future.result()
            for field in self.ranges:
                self.input_widgets.remove(field)
            while self.range_form.rowCount():
                self.range_form.removeRow(0)
            self.ranges = []
            for cid in predict.chain_ids(len(self.entries)):
                field = QLineEdit(); field.setPlaceholderText('Whole chain, or 25-120')
                self.range_form.addRow(f'Chain {cid}', field)
                field.textChanged.connect(self.refresh)
                self.ranges.append(field)
            self.input_widgets.extend(self.ranges)
            self.resolved.setPlainText('\n'.join(f'>{e["label"]} ({len(e["sequence"])} residues)\n{e["sequence"]}'
                                                 for e in self.entries))
            self.set_status('Review the resolved sequences, choose ranges/MSA, then click Predict.')
        except Exception as e:
            self.set_status(str(e))
        self.lookup_future = None
        self.refresh()

    def prediction_input(self):
        if self.plot is not None:
            return snapshot(self.plot, self.flanks.value())
        if self.entries is None:
            raise ValueError('Load sequences before predicting.')
        from .predict_inputs import sequence_chains
        return sequence_chains(self.entries, [field.text() for field in self.ranges])

    def refresh(self, *args):
        msa_ready = self.reuse.isChecked() or self.search.isChecked() or bool(self.msa.text().strip())
        self.msa_hint.setVisible(not msa_ready)
        try:
            chains, _ = self.prediction_input()
            preview = ' + '.join(f'{c["source_chain"]}: {len(c["sequence"])} residues' for c in chains[:4])
            if len(chains) > 4:
                preview += f' + … ({len(chains)} chains total)'
            self.preview.setText(preview)
            self.preview.setToolTip('\n'.join(c['sequence'] for c in chains))
            warning = predict.size_warning(chains)
            self.size_warning.setText(warning); self.size_warning.setVisible(bool(warning))
            running = self.job is not None and self.job.state == 'running'
            terms_ready = self.backend.currentData() != 'af3' or self.af3_terms.isChecked()
            self.start_button.setEnabled(msa_ready and terms_ready and not running)
        except Exception as e:
            self.size_warning.clear(); self.size_warning.hide()
            self.preview.setText(str(e)); self.start_button.setEnabled(False)

    def start(self):
        self.status.setStyleSheet('')
        self.set_progress('Preparing prediction…')
        try:
            if self.randomize_seed.isChecked():
                self.seed.setValue(predict.random_seed())
            chains, source = self.prediction_input()
            name = datetime.now().strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:6]
            msa = (self.msa.text().strip() or None) if self.msa.isEnabled() else None
            crop = None
            if self.reuse.isChecked():
                msa, query = self.source_msa
                crop = dict(source_query=query, columns=[r['source_row'] for c in chains for r in c['residue_map']])
            self.job = predict.PredictionJob(Path(self.output.text()).expanduser() / name, chains,
                          executable=self.executable.text(), search=self.search.isChecked(),
                          msa=msa, msa_crop=crop, seed=self.seed.value(),
                          recycles=self.recycles.value(), source=source,
                          timeout=self.time_limit.value()*60 or None, backend=self.backend.currentData(),
                          accept_af3_terms=self.af3_terms.isChecked(), diffusion_steps=self.af3_steps.value()).start()
            if self.backend.currentData() == 'af3':
                try:
                    predict.record_af3_terms_acceptance()
                except OSError as error:
                    self.session.logger.warning(f'Could not remember AF3 terms acceptance: {error}')
        except Exception as e:
            self.set_progress('Could not start', 0, 100)
            self.show_error(str(e)); return
        for widget in (self.flanks, self.executable, self.output, self.msa, self.reuse, self.search, self.seed, self.recycles,
                       self.time_limit, self.randomize_seed, self.backend, self.af3_terms, self.af3_steps, self.start_button):
            widget.setEnabled(False)
        for widget in self.input_widgets:
            widget.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.timer.start()
        self.set_status('Running…')

    def poll(self):
        try:
            state = self.job.poll()
            if state == 'running':
                elapsed = time.monotonic() - self.job.started
                progress = self.job.read_progress()
                stage = progress.stage
                if progress.total:
                    stage += f' · {progress.completed}/{progress.total} passes completed'
                if self.job.cancelled_at is not None:
                    self.set_progress('Cancelling…')
                else:
                    self.set_progress(stage, progress.completed, progress.total)
                self.set_status(f'Running · {elapsed:.1f} s · MSA: {self.job.manifest["msa_source"]}')
                return
            self.timer.stop(); self.cancel_button.setEnabled(False)
            if state == 'complete':
                self.set_progress('Opening prediction…')
                self.output_model, self.comparison = import_result(self.session, self.job)
                self.set_progress('Complete', 100, 100)
                if self.comparison is None:
                    score = (f'ipTM {self.job.result["iptm"]:.3f}' if len(self.job.chains) > 1
                             else f'pTM {self.job.result["ptm"]:.3f}')
                    self.set_status(f'Complete in {time.monotonic()-self.job.started:.1f} s · {score}\n'
                                        f'Mean pLDDT {self.job.result["mean_plddt"]:.1f}\n'
                                        f'Seed: {self.job.options["seed"]}\n'
                                        + (predict_contacts.summary(self.job.contact_report)+'\n' if self.job.contact_report else '')
                                        + f'Results: {self.job.directory}')
                    return
                old = self.job.source['core_mean_pae']; new = self.comparison['predicted_core_mean_pae']
                before = self.job.source['core_mean_plddt']
                plddt = (f'Same residues: pLDDT {before:.1f} → '
                         f'{self.comparison["predicted_core_mean_plddt"]:.1f}\n') if before is not None else ''
                self.set_status(f'Complete in {time.monotonic()-self.job.started:.1f} s · '
                                    f'ipTM {self.job.result["iptm"]:.3f}\n'
                                    f'Same selected pairs: mean PAE {old:.2f} → {new:.2f} Å\n'
                                    + plddt + f'Seed: {self.job.options["seed"]}\n'
                                    + predict_contacts.summary(self.job.contact_report) + '\n' +
                                    f'Results: {self.job.directory}')
            else:
                self.set_progress('Failed' if self.job.error else 'Cancelled', 0, 100)
                if self.job.error:
                    self.show_error(self.job.error)
                else:
                    self.set_status('Cancelled. Partial files retained in the results directory.')
        except Exception as e:
            self.timer.stop(); self.cancel_button.setEnabled(False)
            # A log/read failure must not leave an unpolled GPU job running.
            try:
                self.shutdown()
            except Exception:
                self.job._release()
            self.set_progress('Failed', 0, 100)
            self.show_error(f'Could not load prediction: {e}. Files: {self.job.directory}')
        finally:
            if self.job.state != 'running':
                for widget in (self.flanks, self.executable, self.output, self.msa, self.search, self.seed,
                               self.recycles, self.time_limit, self.randomize_seed, self.backend,
                               self.af3_terms, self.af3_steps, self.start_button):
                    widget.setEnabled(True)
                self.reuse.setEnabled(self.source_msa is not None)
                self.seed.setEnabled(not self.randomize_seed.isChecked())
                self.msa_controls()
                for widget in self.input_widgets:
                    widget.setEnabled(True)

    def shutdown(self):
        if self.job and self.job.state == 'running':
            self.cancel()
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
            self.cancel()
            self.set_status('Cancelling… Close again when the process stops.')
            event.ignore()
        else:
            self.lookup_timer.stop()
            QApplication.instance().aboutToQuit.disconnect(self.shutdown)
            dialogs = getattr(self.session, '_alphalens_prediction_dialogs', [])
            if self in dialogs:
                dialogs.remove(self)
            self.plot = None
            event.accept()
            self.deleteLater()


def show_prediction_dialog(plot=None, session=None):
    session = plot.session if plot is not None else session
    dialog = PredictionDialog(plot, session)
    dialogs = getattr(session, '_alphalens_prediction_dialogs', None)
    if dialogs is None:
        dialogs = session._alphalens_prediction_dialogs = []
    dialogs.append(dialog)  # Keep the job alive even if the source plot is closed.
    dialog.show()
    return dialog
