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
                          QSpinBox, QVBoxLayout, QWidget)
from . import predict
from . import predict_contacts
from . import predict_features
from .predict_setup import native_platform

PLATFORM_TEXT = {'metal': 'Local protein prediction on Apple Silicon (Metal)',
                 'cuda': 'Local protein prediction on an NVIDIA GPU (CUDA)',
                 None: 'Local protein prediction needs Apple Silicon macOS or Linux x86_64 with an NVIDIA GPU'}
SETUP_REQUIREMENTS = {'metal': 'Requires Apple Silicon and Xcode Command Line Tools',
                      'cuda': 'Requires an NVIDIA GPU, the CUDA toolkit (nvcc, cuBLAS) and g++',
                      None: 'Requires Apple Silicon macOS, or Linux x86_64 with an NVIDIA GPU'}


def peptide_bonded(a, b):
    """b follows a in one continuous segment: bonded C-N and no jump in the residue numbers.

    AlphaPulldown regions (P1:1-100:150-200) are one chain whose numbering jumps across the gap,
    with no unresolved positions in between, so position in the chain alone cannot see the gap.
    """
    step = b.number - a.number
    if not (step == 1 or (step == 0 and b.insertion_code != a.insertion_code)):
        return False
    c, n = a.find_atom('C'), b.find_atom('N')
    if c is None or n is None:
        return False
    import numpy as np
    # 1.33 A bonded; 2.5 A tolerates the stretched bonds of low-confidence predicted regions.
    return float(np.linalg.norm(c.coord - n.coord)) <= 2.5


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
            i = indices[r]
            previous = r.chain.residues[i - 1] if i else None
            rows.append(dict(chain_id=r.chain_id, number=int(r.number), insertion_code=r.insertion_code,
                             aa=r.one_letter_code, chain_index=i,
                             break_before=previous is not None and not peptide_bonded(previous, r)))
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


def structure_chain_ids(plot):
    from chimerax.atomic import Residue
    ids = []
    for r in plot.pae.row_residues_or_atoms():
        if isinstance(r, Residue) and r.polymer_type == Residue.PT_AMINO and r.chain_id not in ids:
            ids.append(r.chain_id)
    return ids


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
    if job.source.get('input_mode') == 'sequence':
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
    return structure, comparison


def show_prediction_report(session, directory):
    from Qt.QtCore import Qt
    from .predict_report import report_text
    text = report_text(directory)
    dialog = QDialog(session.ui.main_window)
    dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    dialog.setWindowTitle('AlphaLens prediction report')
    dialog.resize(760, 480)
    layout = QVBoxLayout(dialog)
    content = QPlainTextEdit(text); content.setReadOnly(True)
    layout.addWidget(content)
    buttons = QHBoxLayout(); layout.addLayout(buttons)
    copy = QPushButton('Copy report'); copy.clicked.connect(lambda: QApplication.clipboard().setText(text))
    close = QPushButton('Close'); close.clicked.connect(dialog.close)
    buttons.addWidget(copy); buttons.addStretch(1); buttons.addWidget(close)
    reports = getattr(session, '_alphalens_prediction_reports', None)
    if reports is None:
        reports = session._alphalens_prediction_reports = []
    reports.append(dialog)
    dialog.finished.connect(lambda: reports.remove(dialog) if dialog in reports else None)
    dialog.show(); dialog.raise_()
    return dialog


class Section(QWidget):
    """A titled block that opens and closes; a one-line summary stands in for it while closed."""
    def __init__(self, title, expanded=False):
        super().__init__()
        from Qt.QtCore import Qt
        from Qt.QtWidgets import QToolButton
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(2)
        header = QHBoxLayout(); outer.addLayout(header)
        self.button = QToolButton(); self.button.setText(title); self.button.setCheckable(True)
        self.button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.button.setStyleSheet('QToolButton { border: none; font-weight: bold; }')
        self.summary = QLabel(); self.summary.setStyleSheet('color: gray;')
        # The summary takes the free width; alone (open), the button keeps to the left
        header.addWidget(self.button, 0, Qt.AlignmentFlag.AlignLeft); header.addWidget(self.summary, 1)
        self.body = QWidget(); outer.addWidget(self.body)
        self._arrows = (Qt.ArrowType.RightArrow, Qt.ArrowType.DownArrow)
        self.button.toggled.connect(self.set_expanded)
        self.button.setChecked(expanded); self.set_expanded(expanded)

    def set_expanded(self, expanded):
        self.button.setArrowType(self._arrows[bool(expanded)])
        self.body.setVisible(expanded); self.summary.setVisible(not expanded)
        window = self.window()
        if window is not self and window.isVisible():
            # Give the space back when closing; never shrink below what is shown
            QTimer.singleShot(0, lambda: window.resize(window.width(), window.sizeHint().height()))


def _option(text, tooltip=''):
    """
    An alignment choice, a radio button. A click always chooses it: clicking the chosen one again
    must not clear the choice (it did, and Predict went grey without a visible reason). Code can
    still clear it with setChecked(False): no choice yet is a valid state of the dialog.
    """
    from Qt.QtWidgets import QRadioButton

    class Choice(QRadioButton):
        def nextCheckState(self):
            if not self.isChecked():
                self.setChecked(True)

    button = Choice(text); button.setAutoExclusive(False)
    if tooltip:
        button.setToolTip(tooltip)
    return button


def _indented(*widgets, form=False):
    """Details of an option, indented beneath it and shown only while it is chosen."""
    box = QWidget(); layout = QFormLayout(box) if form else QHBoxLayout(box)
    layout.setContentsMargins(24, 0, 0, 4)
    if form:
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        for label, widget in widgets:
            layout.addRow(label, widget)
    else:
        for widget in widgets:
            layout.addWidget(widget, 1 if isinstance(widget, QLineEdit) else 0)
    return box


class PredictionDialog(QDialog):
    def __init__(self, plot=None, session=None):
        from Qt.QtWidgets import QGroupBox, QHeaderView, QTableWidget, QAbstractItemView
        session = plot.session if plot is not None else session
        super().__init__(session.ui.main_window)
        self.plot, self.session = plot, session
        self.source_msa = source_alignment(plot) if plot is not None else None
        self.entries = self.lookup_future = None
        self.job = self.output_model = self.comparison = None
        self.setup_job = self.pending_prediction = None
        self.setWindowTitle('AlphaLens prediction · experimental')
        self.resize(640, 400)
        layout = QVBoxLayout(self)
        title = QLabel('Re-predict the selected interface' if plot is not None else 'Predict a structure')
        title.setStyleSheet('font-size: 15px; font-weight: bold;')
        layout.addWidget(title)
        intro = QLabel(PLATFORM_TEXT.get(native_platform(), PLATFORM_TEXT[None])
                       + '. Model weights download on the first run and are cached locally.')
        intro.setWordWrap(True); intro.setStyleSheet('color: gray;'); layout.addWidget(intro)

        # ---- What to predict
        self.input_widgets = []
        self.flanks = QSpinBox(self); self.flanks.setRange(0, 100); self.flanks.setValue(10)
        self.flanks.setSuffix(' residues'); self.flanks.setToolTip(
            'Added on each side of the selected residues, up to chain ends and chain breaks; '
            'excluded from the score comparison')
        self.preview = QLabel(); self.preview.setWordWrap(True)
        if plot is None:
            self.input_text = QPlainTextEdit(); self.input_text.setFixedHeight(
                4*self.input_text.fontMetrics().lineSpacing() + 12)
            self.input_text.setPlaceholderText('UniProt IDs: P69905 P68871 · AlphaPulldown style: '
                                               'P0A836:1-100+P0AGE9, P69905:2\n'
                                               'a raw sequence, or FASTA:\n>chain_A\nSEQUENCE')
            self.input_text.setToolTip('Paste a raw sequence, FASTA entries, or UniProt IDs. UniProt lookup '
                                       'sends only the IDs to UniProt. AlphaPulldown notation sets copies '
                                       '(ID:2) and a 1-based range (ID:25-120); "+" joins the chains of one '
                                       'complex.')
            layout.addWidget(self.input_text)
            row = QHBoxLayout(); layout.addLayout(row)
            self.load_button = QPushButton('Load sequences')
            self.load_button.setToolTip('Resolve the input (UniProt lookup in the background) for review')
            row.addWidget(self.preview, 1); row.addWidget(self.load_button)
            self.preview.setText('Paste the input, then Load sequences to review the chains.')
            self.chain_table = QTableWidget(0, 4)
            self.chain_table.setHorizontalHeaderLabels(['Chain', 'Entry', 'Length', 'Range (optional)'])
            self.chain_table.verticalHeader().hide()
            self.chain_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            self.chain_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
            header = self.chain_table.horizontalHeader()
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            for column in (0, 2, 3):
                header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
            self.chain_table.hide()
            layout.addWidget(self.chain_table)
            self.ranges = []
            self.input_widgets = [self.input_text, self.load_button]
            self.flanks.hide()
        else:
            row = QHBoxLayout(); layout.addLayout(row)
            row.addWidget(self.preview, 1)
            row.addWidget(QLabel('Flanks')); row.addWidget(self.flanks)
        self.size_warning = QLabel(); self.size_warning.setWordWrap(True)
        self.size_warning.setStyleSheet('color: #a65c00;'); self.size_warning.hide()
        layout.addWidget(self.size_warning)

        # ---- Model
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

        # ---- Alignment: one choice; the details of the chosen one appear beneath it
        msa_box = QGroupBox('Alignment (MSA)'); layout.addWidget(msa_box)
        msa_layout = QVBoxLayout(msa_box)
        self.reuse = _option('Reuse and crop the MSA of this prediction (no upload)')
        self.reuse.setEnabled(self.source_msa is not None)
        self.reuse.setChecked(self.source_msa is not None)
        if self.source_msa is None:
            self.reuse.setToolTip('No matching .a3m or input.a3m beside this structure. '
                                  'MSAs embedded in AF3 JSON are not imported automatically.')
        if plot is not None:
            msa_layout.addWidget(self.reuse)
        self.features = _option('Precomputed AlphaPulldown features (no search)',
                                'Alignments from AlphaPulldown feature files, one a protein: AF2 '
                                '<ID>.pkl[.xz] (AF-Multimer pairing by species) or, for AlphaFold 3, '
                                '<ID>_af3_input.json[.xz]. Read and converted when the prediction starts.')
        msa_layout.addWidget(self.features)
        self.features_root = QLineEdit(predict_features.default_root())
        self.features_root.setToolTip(f'The lab database ({predict_features.LAB_ROOT}, through the /g mount), '
                                      f'the public copy ({predict_features.PUBLIC_ROOT}; only the IDs are '
                                      'sent and the files are cached in ~/.cache/alphalens/features), or '
                                      'a folder of AlphaPulldown features')
        self.features_names = QLineEdit()
        self.features_names.setPlaceholderText('UniProt ID of each chain, e.g. A: P0A836, B: P0AGE9')
        self.features_names.setToolTip('The name of each chain\'s feature file (UniProt accession), as '
                                       '"A: P0A836, B: P0AGE9" or in chain order. Filled in from the '
                                       'prediction\'s folder name or the UniProt lookup when possible.')
        self.features_details = _indented(('Feature IDs', self.features_names),
                                          ('From', self.features_root), form=True)
        msa_layout.addWidget(self.features_details)
        self.use_a3m = _option('A3M file')
        msa_layout.addWidget(self.use_a3m)
        self.msa = QLineEdit(); self.msa.setPlaceholderText('Existing A3M for exactly the selected sequence(s)')
        browse = QPushButton('Browse…'); browse.clicked.connect(self.browse_a3m)
        self.msa_details = _indented(self.msa, browse)
        msa_layout.addWidget(self.msa_details)
        api = os.environ.get('LOCALFOLD_MMSEQS_API', 'https://api.colabfold.com').rstrip('/')
        self.search = _option(f'Search remotely if uncached: send sequences to {api}',
                              'MMseqs2 search on the ColabFold server, unless the same query is in the '
                              'local MSA cache. Queues can take minutes.')
        msa_layout.addWidget(self.search)
        # Why Predict is off, if it is: shown above the buttons (and as the button's tooltip)
        self.msa_hint = QLabel('Choose an alignment to enable Predict.')
        self.msa_hint.setWordWrap(True)
        self.msa_hint.setStyleSheet('color: #c06000; font-weight: bold;')
        self.msa_hint.setMinimumHeight(self.msa_hint.fontMetrics().lineSpacing()+4)
        self._guessed_names = ''
        if plot is not None:
            filename = getattr(plot.pae.structure, 'filename', None)
            guess = predict_features.guess_names(filename, structure_chain_ids(plot)) if filename else {}
            self._guessed_names = predict_features.format_names(guess)
            self.features_names.setText(self._guessed_names)
            if self._guessed_names:
                self.features.setText(f'Precomputed AlphaPulldown features (no search) · {self._guessed_names}')

        # ---- Settings most predictions never change
        self.advanced = Section('Settings')
        layout.addWidget(self.advanced)
        settings = QFormLayout(self.advanced.body)
        settings.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        settings.setContentsMargins(24, 0, 0, 0)
        self.recycles = QSpinBox(); self.recycles.setRange(0, 10); self.recycles.setValue(3)
        settings.addRow('Recycles', self.recycles)
        self.seed = QSpinBox(); self.seed.setRange(0, 2147483647); self.seed.setValue(predict.random_seed())
        self.randomize_seed = QCheckBox('new random seed for each prediction')
        self.randomize_seed.setChecked(True); self.seed.setEnabled(False)
        self.randomize_seed.toggled.connect(lambda checked: self.seed.setEnabled(not checked))
        seed_row = QHBoxLayout(); seed_row.addWidget(self.seed); seed_row.addWidget(self.randomize_seed, 1)
        settings.addRow('Seed', seed_row)
        self.af3_steps = QSpinBox(); self.af3_steps.setRange(1, 1000); self.af3_steps.setValue(200)
        self.af3_steps_label = QLabel('AF3 diffusion steps')
        settings.addRow(self.af3_steps_label, self.af3_steps)
        self.af3_samples = QSpinBox(); self.af3_samples.setRange(1, 10); self.af3_samples.setValue(5)
        self.af3_samples_label = QLabel('AF3 diffusion samples')
        settings.addRow(self.af3_samples_label, self.af3_samples)
        self.time_limit = QSpinBox(); self.time_limit.setRange(0, 10080)
        self.time_limit.setSpecialValueText('No limit'); self.time_limit.setValue(0)
        self.time_limit.setSuffix(' min')
        settings.addRow('Time limit', self.time_limit)
        self.output = QLineEdit(str(predict.data_directory() / 'predictions'))
        settings.addRow('Results folder', self.output)
        self.executable = QLineEdit(predict.default_executable())
        settings.addRow('LocalFold engine', self.executable)
        for widget in (self.af3_terms, self.af3_terms_links, self.af3_steps, self.af3_steps_label,
                       self.af3_samples, self.af3_samples_label):
            widget.hide()
        self.setup_hint = QLabel(); self.setup_hint.setWordWrap(True)
        self.setup_hint.setStyleSheet('color: gray;')
        layout.addWidget(self.setup_hint)

        # ---- Running
        self.progress_row = QWidget(); progress = QHBoxLayout(self.progress_row)
        progress.setContentsMargins(0, 0, 0, 0)
        self.progress_label = QLabel('Not started'); self.progress_label.setWordWrap(True)
        self.progress_bar = QProgressBar(); self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0); self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedWidth(160)
        progress.addWidget(self.progress_label, 1); progress.addWidget(self.progress_bar)
        layout.addWidget(self.progress_row)
        self.progress_row.hide()            # Shown from the first Predict on
        self.status = QPlainTextEdit('Ready'); self.status.setReadOnly(True)
        self.status.setFixedHeight(4*self.status.fontMetrics().lineSpacing()+14)
        layout.addWidget(self.status)
        layout.addWidget(self.msa_hint)
        buttons = QHBoxLayout(); layout.addLayout(buttons)
        self.start_button = QPushButton('Predict'); self.cancel_button = QPushButton('Cancel job')
        self.start_button.setDefault(True)
        self.cancel_button.setEnabled(False)
        self.cancel_button.setToolTip('Stops the running prediction or engine setup')
        self.report_button = QPushButton('Prediction report…'); self.report_button.setEnabled(False)
        self.report_button.setToolTip('The scores and comparison of the last prediction, once it has finished')
        self.report_directory = None
        self.report_button.clicked.connect(lambda: show_prediction_report(self.session, self.report_directory))
        buttons.addWidget(self.report_button); buttons.addStretch(1)
        buttons.addWidget(self.cancel_button); buttons.addWidget(self.start_button)
        self.start_button.clicked.connect(self.start)
        self.cancel_button.clicked.connect(self.cancel)
        self.flanks.valueChanged.connect(self.refresh)
        self.lookup_timer = QTimer(self); self.lookup_timer.setInterval(100)
        self.lookup_timer.timeout.connect(self.poll_lookup)
        if plot is None:
            self.input_text.textChanged.connect(self.invalidate_input)
            self.load_button.clicked.connect(self.load_sequences)
        sources = (self.reuse, self.features, self.use_a3m, self.search)
        for box in sources:
            box.toggled.connect(lambda checked, box=box: [b.setChecked(False) for b in sources
                                                          if b is not box] if checked else None)
            box.toggled.connect(self.msa_controls)
        self.msa.textChanged.connect(lambda text: self.use_a3m.setChecked(True) if text.strip() else None)
        self.features_names.textChanged.connect(self.refresh)
        self.features_root.textChanged.connect(self.refresh)
        self.msa.textChanged.connect(self.refresh)
        self.executable.textChanged.connect(self.refresh)
        self.backend.currentIndexChanged.connect(self.backend_changed)
        self.af3_terms.toggled.connect(self.refresh)
        for widget in (self.recycles, self.time_limit, self.af3_steps, self.af3_samples):
            widget.valueChanged.connect(self.refresh)
        self.randomize_seed.toggled.connect(self.refresh)
        self.timer = QTimer(self); self.timer.setInterval(100); self.timer.timeout.connect(self.poll)
        QApplication.instance().aboutToQuit.connect(self.shutdown)
        self.msa_controls()
        self.refresh()
        self.fit_height()

    def fit_height(self):
        '''Grow to show everything (rows were added); never shrink what the user enlarged.'''
        QTimer.singleShot(0, lambda: self.resize(self.width(), max(self.height(), self.sizeHint().height())))

    def browse_a3m(self):
        from Qt.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(self, 'Alignment for the selected sequences', '',
                                              'A3M alignments (*.a3m *.a3m.gz);;All files (*)')
        if path:
            self.msa.setText(path)

    def backend_changed(self):
        backend = self.backend.currentData()
        self.executable.setText(predict.default_executable(backend))
        self.recycles.setValue(10 if backend == 'af3' else 3)
        for widget in (self.af3_terms, self.af3_terms_links, self.af3_steps, self.af3_steps_label,
                       self.af3_samples, self.af3_samples_label):
            widget.setVisible(backend == 'af3')
        self.refresh()
        self.fit_height()

    def display(self, shown):
        """ChimeraX's Tools menu calls display(True) on the returned tool."""
        self.setVisible(shown)

    def reject(self):
        # Escape must follow the same cancellation path as the window close button.
        self.close()

    def msa_controls(self, *args):
        self.features_details.setVisible(self.features.isChecked())
        self.msa_details.setVisible(self.use_a3m.isChecked())
        self.refresh()
        self.fit_height()

    def feature_request(self, chains):
        '''dict(root, names) for the chains of this prediction, from the Feature IDs field.'''
        root = self.features_root.text().strip()
        if not root:
            raise ValueError('Enter where the features are: a folder or the database URL.')
        # Whether the folder exists is checked by the prediction's process: a stale network
        # mount must not hang the window here.
        ids = (structure_chain_ids(self.plot) if self.plot is not None
               else list(predict.chain_ids(len(self.entries or []))))
        mapping = predict_features.parse_chain_accessions(self.features_names.text(), ids)
        missing = [c['source_chain'] for c in chains if not mapping.get(c['source_chain'])]
        if missing:
            raise ValueError(f'Enter the feature ID (UniProt accession) of chain {", ".join(missing)}.')
        return dict(root=str(Path(root).expanduser()) if not predict_features.is_url(root) else root,
                    names=[mapping[c['source_chain']] for c in chains])

    def set_progress(self, stage, completed=None, total=None):
        self.progress_row.show()
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
        self.log_message(message, warning=True)

    def log_message(self, message, *, warning=False, show_start=False):
        """Keep the same prediction messages in this window and ChimeraX's Log."""
        start = 0 if self.status.document().isEmpty() else self.status.document().characterCount()
        self.status.appendPlainText(message)
        if show_start:
            # Long paths at the end must not hide the scores at the start.
            cursor = self.status.textCursor()
            cursor.setPosition(start)
            self.status.setTextCursor(cursor)
            self.status.ensureCursorVisible()
            self.status.verticalScrollBar().setValue(cursor.block().firstLineNumber())
        logger = self.session.logger.warning if warning else self.session.logger.info
        logger(message)

    def cancel(self):
        if self.setup_job and self.setup_job.state == 'running':
            self.setup_job.cancel()
            self.set_progress('Cancelling setup…')
        elif self.job and self.job.state == 'running':
            self.job.cancel()
            self.set_progress('Cancelling…')

    def invalidate_input(self):
        self._lookup_error = None
        if self.entries is not None:
            self.entries = None
            self.set_chain_rows([])
        self.refresh()

    def set_chain_rows(self, entries):
        '''One table row per chain: what was resolved, and its optional range.'''
        from Qt.QtWidgets import QTableWidgetItem
        for field in self.ranges:
            if field in self.input_widgets:
                self.input_widgets.remove(field)
        self.chain_table.setRowCount(0)
        self.ranges = []
        for row, (cid, entry) in enumerate(zip(predict.chain_ids(len(entries)) if entries else [], entries)):
            self.chain_table.insertRow(row)
            cells = (cid, entry['label'], str(len(entry['sequence'])))
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setToolTip(f'>{entry["label"]}\n{entry["sequence"]}')
                self.chain_table.setItem(row, column, item)
            field = QLineEdit(entry.get('range', '')); field.setPlaceholderText('whole chain, or 25-120')
            field.setFrame(False)
            field.textChanged.connect(self.refresh)
            self.chain_table.setCellWidget(row, 3, field)
            self.ranges.append(field)
        self.input_widgets.extend(self.ranges)
        rows = min(len(entries), 6)
        self.chain_table.setFixedHeight(self.chain_table.horizontalHeader().height() + 4
                                        + sum(self.chain_table.rowHeight(r) for r in range(rows)))
        self.chain_table.setVisible(bool(entries))
        self.fit_height()

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
        self.status.setStyleSheet('')   # An earlier error must not color this lookup's messages
        self.log_message('Resolving sequences…')
        threading.Thread(target=resolve, daemon=True, name='AlphaLens UniProt lookup').start()
        self.lookup_timer.start()

    def poll_lookup(self):
        if not self.lookup_future.done():
            return
        self.lookup_timer.stop()
        self.input_text.setEnabled(True); self.load_button.setEnabled(True)
        try:
            self.entries = self.lookup_future.result()
            self.set_chain_rows(self.entries)
            names = {cid: e.get('accession') or next(iter(predict_features.accessions_in(e['label'])), None)
                     for cid, e in zip(predict.chain_ids(len(self.entries)), self.entries)}
            if all(names.values()) and self.features_names.text() in ('', self._guessed_names):
                self._guessed_names = predict_features.format_names(names)
                self.features_names.setText(self._guessed_names)
            self.log_message('Review the chains (hover for the sequence), set ranges if needed, choose '
                             'an alignment, then click Predict.')
        except Exception as e:
            self._lookup_error = str(e)
            self.show_error(str(e))
        self.lookup_future = None
        self.refresh()

    def prediction_input(self):
        if self.plot is not None:
            return snapshot(self.plot, self.flanks.value())
        if self.entries is None:
            if getattr(self, '_lookup_error', None):
                raise ValueError(f'The input could not be read: {self._lookup_error}')
            if self.input_text.toPlainText().strip():
                raise ValueError('Click Load sequences to read the input.')
            raise ValueError('Paste a sequence, FASTA entries or UniProt IDs, then click Load sequences.')
        from .predict_inputs import sequence_chains
        return sequence_chains(self.entries, [field.text() for field in self.ranges])

    def source_changed(self):
        '''The plot's box, model or window changed: show what Predict would use now.'''
        if not self.is_busy():
            self.refresh()      # A running job keeps its own input; refreshed when it ends

    def is_busy(self):
        return any(job is not None and job.state == 'running' for job in (self.job, self.setup_job))

    def needs_setup(self):
        '''The managed engine is missing, or was built from an earlier LocalFold revision.'''
        path = Path(self.executable.text()).expanduser().resolve()
        managed = predict.data_directory() / 'bin' / f'localfold-{self.backend.currentData()}'
        return path == managed.resolve() and (not path.exists() or predict.engine_outdated(path))

    def refresh(self, *args):
        needs_setup = self.needs_setup()
        update = needs_setup and Path(self.executable.text()).expanduser().exists()
        self.setup_hint.setText(
            ('This AlphaLens uses a newer LocalFold engine; Update & Predict rebuilds it in the background '
             '(about a minute, once). ' if update else
             'First use: Install & Predict downloads and builds LocalFold in the background. ')
            + SETUP_REQUIREMENTS.get(native_platform(), SETUP_REQUIREMENTS[None]) + '; setup can be cancelled.')
        self.setup_hint.setVisible(needs_setup)
        # '&&': a single & is Qt's mnemonic marker and would not be shown
        self.start_button.setText(('Update && Predict' if update else 'Install && Predict') if needs_setup else 'Predict')
        af3 = self.backend.currentData() == 'af3'
        summary = [f'{self.recycles.value()} recycles',
                   'random seed' if self.randomize_seed.isChecked() else f'seed {self.seed.value()}']
        if af3:
            summary.append(f'{self.af3_steps.value()} steps × {self.af3_samples.value()} samples')
        summary.append(f'{self.time_limit.value()} min limit' if self.time_limit.value() else 'no time limit')
        self.advanced.summary.setText(' · '.join(summary))
        chains, problem = None, ''
        if self.lookup_future is not None:
            problem = 'Reading the input (UniProt lookup)…'
        else:
            try:
                chains, _ = self.prediction_input()
            except Exception as e:
                problem = str(e)
        if chains:
            preview = ' + '.join(f'{c["source_chain"]}: {len(c["sequence"])} residues'
                                 + (f' in {predict.segment_count(c)} segments' if c.get('residue_ids') else '')
                                 for c in chains[:4])
            if len(chains) > 4:
                preview += f' + … ({len(chains)} chains total)'
            if len(chains) > 1:
                preview += f' · {sum(len(c["sequence"]) for c in chains)} in total'
            self.preview.setStyleSheet('')
            self.preview.setText(preview)
            self.preview.setToolTip('\n'.join(c['sequence'] for c in chains))
            warning = predict.size_warning(chains)
            self.size_warning.setText(warning); self.size_warning.setVisible(bool(warning))
            problem = self.blocking_reason(chains)
        else:
            self.size_warning.clear(); self.size_warning.hide()
            self.preview.setStyleSheet('color: #c06000;')
            self.preview.setText(problem)
            self.preview.setToolTip(problem)
        self.start_button.setEnabled(not problem)
        self.start_button.setToolTip(problem or ('Install LocalFold, then predict' if needs_setup and not update
                                                 else 'Rebuild LocalFold, then predict' if needs_setup
                                                 else 'Start the prediction'))
        self.msa_hint.setText(problem)
        self.msa_hint.setVisible(bool(problem))

    def blocking_reason(self, chains):
        '''
        Why Predict cannot start with these chains, or '' when it can. Only real problems: a missing
        choice or file the prediction needs, the AF3 terms, a missing engine, a running job.
        '''
        if self.is_busy():
            return 'A prediction or engine setup is running: wait for it, or Cancel it.'
        if self.backend.currentData() == 'af3' and not self.af3_terms.isChecked():
            return 'Accept DeepMind’s AF3 terms (under Model) to predict with AlphaFold 3, or choose AlphaFold 2.'
        if not any(b.isChecked() for b in (self.reuse, self.features, self.use_a3m, self.search)):
            return 'Choose an alignment (MSA) to enable Predict.'
        if self.use_a3m.isChecked() and not self.msa.text().strip():
            return 'Enter the A3M file of the chosen alignment, or Browse… for it.'
        if self.features.isChecked():
            try:
                self.feature_request(chains)
            except ValueError as e:
                return str(e)
        path = self.executable.text().strip()
        if not path:
            return 'Enter the LocalFold engine under Settings.'
        if self.needs_setup():
            if native_platform() is None:
                return 'LocalFold runs on Apple Silicon macOS or Linux x86_64 with an NVIDIA GPU; it cannot be installed here.'
        elif not (Path(path).expanduser().is_file() and os.access(Path(path).expanduser(), os.X_OK)):
            return (f'No LocalFold engine at {path}. Fix the path under Settings, or restore '
                    f'{predict.data_directory() / "bin" / ("localfold-" + self.backend.currentData())}, '
                    'which Install & Predict sets up.')
        return ''

    def set_busy_controls(self, busy):
        for widget in (self.flanks, self.executable, self.output, self.msa_details, self.search, self.seed,
                       self.recycles, self.time_limit, self.randomize_seed, self.backend, self.af3_terms,
                       self.af3_steps, self.af3_samples, self.start_button, self.features, self.use_a3m,
                       self.features_details, *self.input_widgets):
            widget.setEnabled(not busy)
        self.reuse.setEnabled(not busy and self.source_msa is not None)
        self.cancel_button.setEnabled(busy)
        if busy:
            self.start_button.setToolTip('A prediction or engine setup is running: wait for it, or Cancel it.')
        if not busy:
            self.seed.setEnabled(not self.randomize_seed.isChecked())
            self.msa_controls()

    def start(self):
        if self.is_busy():
            return
        self.status.setStyleSheet('')
        self.set_progress('Preparing prediction…')
        try:
            if self.randomize_seed.isChecked():
                self.seed.setValue(predict.random_seed())
            chains, source = self.prediction_input()
            name = datetime.now().strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:6]
            msa = (self.msa.text().strip() or None) if self.use_a3m.isChecked() else None
            crop = None
            if self.reuse.isChecked():
                msa, query = self.source_msa
                crop = dict(source_query=query, columns=[r['source_row'] for c in chains for r in c['residue_map']])
            features = self.feature_request(chains) if self.features.isChecked() else None
            request = dict(directory=Path(self.output.text()).expanduser() / name, chains=chains,
                           features=features,
                           executable=self.executable.text(), search=self.search.isChecked(),
                           msa=msa, msa_crop=crop, seed=self.seed.value(), recycles=self.recycles.value(), source=source,
                           timeout=self.time_limit.value()*60 or None, backend=self.backend.currentData(),
                           accept_af3_terms=self.af3_terms.isChecked(), diffusion_steps=self.af3_steps.value(),
                           diffusion_samples=self.af3_samples.value())
            if request['backend'] == 'af3' and not request['accept_af3_terms']:
                raise ValueError('Confirm acceptance of DeepMind’s AF3 parameter terms before predicting with AF3.')
            if self.needs_setup():
                from .predict_setup import SetupJob
                # Freeze the selected regions, sequences and settings at this click.
                self.pending_prediction = request
                self.setup_job = SetupJob(predict.data_directory(), request['backend']).start()
                self.set_busy_controls(True)
                self.cancel_button.setText('Cancel setup')
                self.log_message(f'Installing LocalFold {request["backend"].upper()} for first use. '
                                 f'Prediction will start when ready. Setup log: {self.setup_job.log_path}')
                self.timer.start()
            else:
                self.launch_prediction(request)
        except Exception as error:
            self.timer.stop()
            self.pending_prediction = None
            self.set_progress('Could not start', 0, 100)
            self.show_error(str(error))
            self.set_busy_controls(False)

    def launch_prediction(self, request):
        self.setup_job = self.pending_prediction = None
        self.job = predict.PredictionJob(**request).start()
        if request['backend'] == 'af3':
            try:
                predict.record_af3_terms_acceptance()
            except OSError as error:
                self.log_message(f'Could not remember AF3 terms acceptance: {error}', warning=True)
        self.set_busy_controls(True)
        self.cancel_button.setText('Cancel job')
        self._logged_stage = None
        self.log_message(f'Starting {self.job.backend.upper()} prediction · Seed: {self.job.options["seed"]}\n'
                         f'Alignment source: {self.job.manifest["msa_source"]}\n'
                         f'Results: {self.job.directory}')
        self.timer.start()

    def poll_setup(self):
        setup = self.setup_job
        try:
            state = setup.poll()
            for message in setup.read_messages():
                self.log_message(message)
            if state == 'running':
                stage = 'Cancelling setup…' if setup.cancelled_at else setup.stage
                self.set_progress(f'{stage} · {time.monotonic()-setup.started:.1f} s')
                return
            self.timer.stop()
            request = self.pending_prediction
            self.pending_prediction = self.setup_job = None
            if state == 'complete':
                self.log_message(f'LocalFold setup complete in {time.monotonic()-setup.started:.1f} s.')
                self.launch_prediction(request)
            elif state == 'cancelled':
                self.set_progress('Setup cancelled', 0, 100)
                self.log_message(f'LocalFold setup cancelled. Retry Install & Predict when ready. Log: {setup.log_path}')
            else:
                self.set_progress('Setup failed', 0, 100)
                self.show_error(setup.error)
        except Exception as error:
            setup.shutdown()
            self.timer.stop()
            self.setup_job = self.pending_prediction = None
            self.set_progress('Setup failed', 0, 100)
            self.show_error(str(error))
        finally:
            if not self.is_busy():
                self.cancel_button.setText('Cancel job')
                self.set_busy_controls(False)

    def poll(self):
        if self.setup_job is not None:
            self.poll_setup()
            return
        try:
            state = self.job.poll()
            if state == 'running':
                elapsed = time.monotonic() - self.job.started
                progress = self.job.read_progress()
                stage = progress.stage
                if progress.total:
                    stage += f' · {progress.completed}/{progress.total} passes completed'
                if self.job.cancelled_at is not None:
                    stage = 'Cancelling…'
                self.set_progress(f'{stage} · {elapsed:.1f} s', progress.completed, progress.total)
                if stage != self._logged_stage:
                    self.log_message(f'{elapsed:.1f} s · {stage}')
                    self._logged_stage = stage
                return
            self.timer.stop(); self.cancel_button.setEnabled(False)
            if state == 'complete':
                self.set_progress('Opening prediction…')
                self.output_model, self.comparison = import_result(self.session, self.job)
                self.set_progress('Complete', 100, 100)
                from .predict_report import save_report
                report = save_report(self.job.directory)
                self.report_directory = self.job.directory
                self.report_button.setEnabled(True)
                self.log_message(report, show_start=True)
                # Opening the result plot can cover this dialog; keep its comparison visible.
                self.raise_()
            else:
                self.set_progress('Failed' if self.job.error else 'Cancelled', 0, 100)
                if self.job.error:
                    self.show_error(self.job.error)
                else:
                    self.log_message(f'Cancelled. Partial files retained in {self.job.directory}')
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
                self.set_busy_controls(False)

    def shutdown(self):
        if self.setup_job is not None:
            self.setup_job.shutdown()
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
        if self.is_busy():
            self.cancel()
            self.log_message('Cancelling… Close again when the process stops.')
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
