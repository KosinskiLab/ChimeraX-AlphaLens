"""
Commands:

  alphalens open <dir-or-structure-file> [rank N] [mode pae|contact|model|std|entropy|agreement]
                [cutoff 8] [contacts true|false] [minProb 0.5] [saveContacts file.csv] ...
  alphalens [#model] [paeFile path] [distogram path] [mode ...] [cutoff 8] [palette ..] [range ..]
                [contacts true|false] [minProb 0.5] [saveContacts file.csv]
  alphalens <dir-or-structure-file> [rank N] ...      same as "alphalens open"

"mode model" (the Model contacts view up to 0.5) is the same as "mode distance".
"""

import os
import threading
import time
from concurrent.futures import Future

import numpy as np

from .contacts import DEFAULT_MIN_P

MODE_NAMES = ('pae', 'contact', 'distance', 'model', 'std', 'entropy', 'agreement')


class _BackgroundLoad:
    '''A data read whose result can be applied by either the GUI or a waiting command.

    The worker only reads data. Both finish() and wait() run on the session thread,
    so waiting never needs to process Qt events or wait for a queued GUI callback.
    '''

    def __init__(self, work, done):
        self._work, self._done = work, done
        self._future = Future()
        self._finished = False

    def start(self, session, wait, name):
        def read():
            try:
                self._future.set_result(self._work())
            except Exception as e:
                self._future.set_exception(e)

        if wait or not session.ui.is_gui:
            read()
            self.wait()
        else:
            def run():
                read()
                session.ui.thread_safe(self.finish)
            threading.Thread(target=run, name=name, daemon=True).start()

    def finish(self):
        if self._finished:
            return                       # A waiting command already applied this result
        try:
            result = self._future.result()
            self._finished = True
            self._done(*result)
        finally:
            self._finished = True
            # Do not retain full-size matrices or callbacks after applying the result.
            self._future = self._work = self._done = None

    def wait(self):
        self.finish()

    @property
    def pending(self):
        return not self._finished


def alphalens(session, structure=None, pae_file=None, distogram=None, mode=None, cutoff=8.0,
             palette=None, range=None, divider_lines=None, contacts=None, min_prob=None,
             save_contacts=None, wait=False, rank=None):
    '''Show PAE / contact probability / distogram spread / agreement plot for a predicted structure.'''
    from chimerax.core.errors import UserError
    from .pae import AlphaLensPAE

    if mode == 'distance':
        mode = 'model'                  # Model contacts view ("distance": its name in 0.6 to 1.3)
    if isinstance(structure, str):      # "alphalens <path>": the same as "alphalens open <path>"
        if pae_file is not None:
            raise UserError('"paeFile" needs an open structure, e.g. "alphalens #1 paeFile ..."')
        return alphalens_open(session, structure, rank=rank or 0, distogram=distogram, mode=mode,
                             cutoff=cutoff, palette=palette, range=range,
                             divider_lines=divider_lines, contacts=contacts, min_prob=min_prob,
                             save_contacts=save_contacts, wait=wait)
    if rank is not None:
        raise UserError('"rank" applies to a prediction folder: "alphalens <folder> rank N"')

    structure = _resolve_structure(session, structure)

    pae_colormap = None
    if palette is not None or range is not None:
        from chimerax.core.colors import colormap_with_range
        pae_colormap = colormap_with_range(palette, range, default_colormap_name='pae',
                                           full_range=(0, 30))

    if mode == 'agreement':
        _check_agreement_runs(structure)

    def loaded(pae, plot):
        '''Steps that need the contact probabilities of the shown model.'''
        if mode in ('std', 'entropy') and not pae.has_distogram:
            raise UserError(f'Mode "{mode}" needs a distogram; it could not be loaded for '
                            f'{structure}')
        if (contacts or save_contacts is not None or mode in ('contact', 'agreement')) \
                and not pae.has_contact:
            raise UserError(f'{structure}: no contact probabilities: the distogram could '
                            f'not be loaded and no confidence JSON contact_probs are available')
        _apply_contacts(session, pae, plot, contacts, min_prob)

        def finish():
            if save_contacts is not None:
                if min_prob is not None:
                    t = min_prob
                else:
                    t = plot.threshold if plot is not None and not plot.closed() else DEFAULT_MIN_P
                region = (plot._struct_region if plot is not None and not plot.closed()
                          and plot.pae is pae else None)
                save_contacts_csv(session, pae, save_contacts, t, region=region)
        if mode == 'agreement':
            start_agreement(session, pae, plot, cutoff, wait=wait, show=True, on_loaded=finish)
        else:
            finish()

    # Re-show an existing plot unless new data or a new cutoff was requested.
    plot = getattr(structure, '_alphalens_plot', None)
    pae = getattr(structure, 'alphafold_pae', None)
    reuse = (isinstance(pae, AlphaLensPAE) and pae_file is None and distogram is None
             and (pae.metrics is None or pae.metrics.get('cutoff') == cutoff)
             and (not session.ui.is_gui or (plot is not None and not plot.closed())))
    if reuse:
        if wait:
            for name in ('_distogram_load', '_agreement_load'):
                load = getattr(pae, name, None)
                if load is not None:
                    load.wait()
        load = getattr(pae, '_distogram_load', None)
        loading = load is not None and load.pending
        if (save_contacts is not None or mode == 'agreement') and (loading or not pae.has_contact):
            raise UserError(f'{structure}: no contact probabilities (no distogram or confidence '
                            f'JSON contact_probs, or the distogram is still loading)')
        if plot is not None and not plot.closed():
            plot.display(True)
            if pae_colormap is not None:
                plot.set_pae_colormap(pae_colormap)
            if divider_lines is not None:
                plot.show_chain_dividers(divider_lines)
            if mode not in (None, 'agreement'):
                plot.set_mode(mode)
        else:
            plot = None
        loaded(pae, plot)
        return pae

    pae, dpath, contact_probs, pae_path = prepare_prediction(session, structure, pae_file,
                                                             distogram, cutoff)

    if mode in ('std', 'entropy') and dpath is None:
        raise UserError(f'Mode "{mode}" needs a distogram and none was found for {structure}')
    if mode in ('contact', 'agreement') and dpath is None and not pae.has_contact:
        raise UserError(f'Mode "{mode}" needs a distogram or confidence JSON contact_probs; '
                        f'none found for {structure}')
    if (contacts or save_contacts is not None) and dpath is None and not pae.has_contact:
        raise UserError(f'"contacts" and "saveContacts" need a distogram or confidence JSON contact_probs; '
                        f'none found for {structure}')

    plot = None
    if session.ui.is_gui:
        from .tool import AlphaLensPlot
        old = getattr(structure, '_alphalens_plot', None)
        if old is not None and not old.closed():
            old.delete()
        dividers = True if divider_lines is None else divider_lines
        plot = AlphaLensPlot(session, pae, cutoff=cutoff, pae_colormap=pae_colormap,
                            divider_lines=dividers)
        structure._alphalens_plot = plot
        if pae.has_contact:
            plot.set_data_status('contact probabilities from confidence JSON; no distogram')
            if mode == 'contact':
                plot.set_mode('contact')
        if mode == 'model':
            plot.set_mode('model')          # Model contacts; compared with P once loaded
        if dpath and not wait and pae.has_contact:
            _apply_contacts(session, pae, plot, contacts, min_prob)     # Redrawn once loaded

    if dpath:
        start_distogram(session, structure, pae, plot, dpath, cutoff,
                        None if mode == 'agreement' else mode, wait, contact_probs, pae_path,
                        on_loaded=lambda: loaded(pae, plot))
    else:
        loaded(pae, plot)
    return pae


def restore_plot(session, st):
    '''
    Recreate a plot saved in a session. The session holds its settings and the paths of the
    prediction files; the matrices are read again from those files. Returns the plot or None.
    '''
    from chimerax.core.errors import UserError
    from .tool import AlphaLensPlot
    structure = st.get('structure')
    if structure is None or structure.deleted:
        return None
    if st.get('model_file') and not getattr(structure, 'filename', None):
        structure.filename = st['model_file']   # Finds the other files, seeds and models

    def usable(key, what):
        path = st.get(key)
        if path and not os.path.exists(path):
            session.logger.warning(f'AlphaLens {structure}: the saved {what} {path} was not found')
            return None
        return path

    cutoff = st.get('cutoff', 8.0)
    pae_file, distogram = usable('pae_file', 'PAE file'), usable('distogram_file', 'distogram')
    try:
        pae, dpath, contact_probs, pae_path = prepare_prediction(session, structure, pae_file,
                                                                 distogram, cutoff)
    except UserError as e:
        session.logger.warning(f'AlphaLens {structure}: the plot was not restored: {e}. The '
                               f'structure and its contact lines are unchanged.')
        return None
    pae_colormap = None
    if st.get('pae_colormap') is not None:
        from chimerax.core.colors import Colormap
        values, colors = st['pae_colormap']
        pae_colormap = Colormap(np.array(values), np.array(colors))
    plot = AlphaLensPlot(session, pae, cutoff=cutoff, pae_colormap=pae_colormap,
                         divider_lines=st.get('divider_lines', True))
    structure._alphalens_plot = plot
    if pae.has_contact:
        plot.set_data_status('contact probabilities from confidence JSON; no distogram')
    plot.restore_settings(st)
    finish = lambda: None if plot.closed() else plot.restore_loaded(st)
    if dpath:
        start_distogram(session, structure, pae, plot, dpath, cutoff, None, False, contact_probs,
                        pae_path, on_loaded=finish)
    else:
        finish()
    return plot


def _apply_contacts(session, pae, plot, contacts, min_prob):
    if contacts is None:
        return
    min_p = 0.5 if min_prob is None else min_prob
    if plot is not None and not plot.closed():
        plot.show_structure_contacts(contacts, min_p)
        return
    from . import contacts as c
    if contacts:
        c.show_contacts(pae, min_p=min_p)
    else:
        c.hide_contacts(pae.structure)


def save_contacts_csv(session, pae, path, min_prob, region=None):
    '''Write the contacts with P >= min_prob (in the shown model or any seed / model) to CSV.'''
    from chimerax.core.errors import UserError
    from .contacts import write_contacts_csv, region_text
    if not pae.has_contact and pae.agreement is None:
        raise UserError(f'{pae.structure} has no contact probabilities to save')
    path = os.path.expanduser(path)
    n = write_contacts_csv(pae, path, min_prob, region=region, agreement=pae.agreement)
    agr = pae.agreement
    where = f' in the shown model or any of the {agr.n} {agr.kind}s' if agr is not None else ''
    extra = f', with P in each {agr.kind}' if agr is not None else ''
    session.logger.info(f'AlphaLens {pae.structure}: saved {n} contacts with P ≥ {min_prob:g}'
                        f'{where}{region_text(pae, region)}{extra} to {path}')
    return n


# -----------------------------------------------------------------------------
# Agreement between seeds / models
#
def _check_agreement_runs(structure):
    from chimerax.core.errors import UserError
    from . import data
    spath = getattr(structure, 'filename', None)
    runs = data.agreement_runs(spath) if spath else []
    if len(runs) < 2:
        raise UserError(f'Mode "agreement" needs at least two seeds (AlphaFold3) or models '
                        f'(AlphaFold2) of this prediction; {structure}: '
                        f'{data.agreement_shortfall(spath, runs)}')
    return runs


def start_agreement(session, pae, plot, cutoff, wait=False, show=False, on_loaded=None):
    '''
    Read the contact probabilities of every seed (AF3) or model (AF2) of the prediction and
    attach them as pae.agreement, in a thread in the GUI. show: switch the plot to the
    agreement view when done. on_loaded: called only after agreement is available.
    '''
    if pae.agreement is not None:
        if show and plot is not None and not plot.closed():
            plot.agreement_loaded(show=True)
        if on_loaded is not None:
            on_loaded()
        return
    if pae._agreement_pending is not None:          # Already loading
        pae._agreement_pending.append((show, on_loaded))
        if wait or not session.ui.is_gui:
            pae._agreement_load.wait()
        return
    runs = _check_agreement_runs(pae.structure)
    known = {}
    mt = pae.metrics
    if mt is not None and 'contact' in mt and mt.get('path') \
            and abs(mt.get('cutoff', 8.0) - cutoff) < 1e-6:
        known[mt['path']] = mt['contact']
    rows, full_size = pae._rows, pae._full_size
    pae._agreement_pending = [(show, on_loaded)]
    from .data import LoadProgress
    progress = LoadProgress()
    if plot is not None and not plot.closed():
        plot.agreement_loading(len(runs), runs[0]['kind'], progress)

    def work():
        from . import agreement
        t0 = time.time()
        try:
            agr, msgs = agreement.load(runs, cutoff, full_size, rows, known, progress)
            err = None
        except Exception as e:
            agr, msgs, err = None, [], f'{type(e).__name__}: {e}'
        return agr, msgs, err, time.time() - t0

    def done(agr, msgs, err, seconds):
        _agreement_loaded(session, pae, plot, len(runs), agr, msgs, err, seconds)

    pae._agreement_load = load = _BackgroundLoad(work, done)
    load.start(session, wait, 'alphalens-agreement')


def _agreement_loaded(session, pae, plot, n_runs, agr, msgs, err, seconds):
    pending = pae._agreement_pending or []
    pae._agreement_pending = None
    s = pae.structure
    plot_open = plot is not None and not plot.closed()
    if s.deleted:
        # The window may now show another model of this prediction. Its load must settle,
        # but callbacks (for example exports) still belong to the deleted source model.
        if not plot_open:
            return
        current = plot.pae
        if not plot.shows_prediction_of(pae):
            return                          # The window shows another prediction now
        if current is pae or current.structure.deleted or current.matrix_size != pae.matrix_size:
            plot.agreement_failed('The model used to load agreement was closed')
            return
        pae, s = current, current.structure
        pending = [(show, None) for show, _ in pending]
    for m in msgs:
        session.logger.warning(f'AlphaLens {s}: agreement: skipped {m}')
    plot_open = plot is not None and not plot.closed() and plot.shows_prediction_of(pae)
    if agr is None:
        reason = err or f'fewer than two of the {n_runs} seeds or models have contact probabilities'
        session.logger.warning(f'AlphaLens {s}: no agreement view: {reason}')
        if plot_open:
            plot.agreement_failed(reason)
        from chimerax.core.errors import UserError
        raise UserError(f'AlphaLens {s}: could not load agreement: {reason}')
    else:
        pae.agreement = agr
        if plot_open:
            cur = plot.pae
            if cur is not pae and cur.agreement is None and cur.matrix_size == pae.matrix_size:
                cur.agreement = agr         # Model switched while loading (same prediction)
        log_agreement(session, pae, plot.threshold if plot_open else DEFAULT_MIN_P, seconds)
        if plot_open:
            plot.agreement_loaded(show=any(show for show, _ in pending))
    for _, cb in pending:
        if cb is not None:
            cb()


def log_agreement(session, pae, threshold, seconds=None):
    '''Log the interface score of every seed / model per pair of chains as a table.'''
    from html import escape
    from .agreement import TOP_N
    from .contacts import row_chain_ids
    agr = pae.agreement
    msg = f'AlphaLens {escape(str(pae.structure))}: agreement between {agr.n} {agr.kind}s'
    if seconds is not None:
        msg += f' (read in {seconds:.1f} s)'
    scores = agr.chain_pair_scores(row_chain_ids(pae), threshold)
    if not scores:
        session.logger.info(msg + '.', is_html=True)
        return
    msg += (f'. Per pair of chains: mean of the {TOP_N} highest P(d &lt; {agr.cutoff:g} Å) '
            f'in each {agr.kind}, and the number of {agr.kind}s with a residue pair at '
            f'P ≥ {threshold:.2f}.')
    th = '<th style="padding:1px 6px">'
    td = '<td style="padding:1px 6px; text-align:right">'
    head = ''.join(f'{th}{escape(label)}</th>' for label in agr.labels)
    lines = [f'<table style="border-collapse:collapse"><tr>{th}Chains</th>{head}'
             f'{th}P ≥ {threshold:.2f}</th></tr>']
    for (a, b), (means, hits) in scores.items():
        cells = ''.join(f'{td}{v:.2f}</td>' for v in means)
        lines.append(f'<tr><td style="padding:1px 6px">{escape(a)}–{escape(b)}</td>{cells}'
                     f'{td}{hits} of {agr.n}</td></tr>')
    lines.append('</table>')
    session.logger.info(msg + ''.join(lines), is_html=True)


# -----------------------------------------------------------------------------
#
def prepare_prediction(session, structure, pae_file=None, distogram=None, cutoff=8.0, preread=None):
    '''
    Read PAE (and confidence JSON contact_probs) for a structure, attach an AlphaLensPAE object as
    structure.alphafold_pae. Returns (pae, distogram_path or None, contact_probs, pae_path).
    preread: (files, (pae matrix, contact_probs) or None) from read_prediction_files(), read in a
    thread so that the window stays usable meanwhile.
    '''
    from chimerax.core.errors import UserError
    from . import data
    from .pae import AlphaLensPAE

    spath = getattr(structure, 'filename', None)
    read = None
    if preread is not None:
        files, read = preread
    else:
        files = data.find_prediction_files(spath) if spath else data.PredictionFiles(None)

    chimerax_pae = getattr(structure, 'alphafold_pae', None)     # e.g. from "alphafold fetch"
    pae_path = pae_file or files.pae_path or getattr(chimerax_pae, '_pae_path', None)
    if pae_path is None:
        raise UserError(f'Could not find a PAE file for {structure} '
                        f'(looked next to {spath}); give one with "paeFile <path>"')
    try:
        if read is not None and pae_path == files.pae_path:
            matrix, contact_probs = read
        else:
            matrix, contact_probs = data.read_pae(pae_path)
    except Exception as e:
        raise UserError(f'Could not read PAE from {pae_path}: {e}')

    pae = AlphaLensPAE(pae_path, structure, matrix)
    pae.model_path = os.path.realpath(spath) if spath else None    # Tells its prediction's models apart
    if not pae.reduce_matrix_to_residues_in_structure():
        raise UserError(f'Structure {structure} has {pae.num_residue_rows} residues and '
                        f'{pae.num_atom_rows} non-polymer atoms, which does not match the '
                        f'{pae.matrix_size} x {pae.matrix_size} PAE matrix in {pae_path}')
    structure.alphafold_pae = pae       # Lets "alphafold pae/contacts" use this data too

    if distogram is not None:
        if not data.distogram_file_ok(distogram):
            raise UserError(f'{distogram} does not contain a distogram')
        dpath = distogram
    else:
        dpath = files.distogram_path

    pae._distogram_file = dpath

    kind = {'af2': 'AlphaFold2', 'af3': 'AlphaFold3', 'afserver': 'AlphaFold Server', 'colabfold': 'ColabFold',
            'localfold': 'LocalFold'}.get(files.kind, '')
    model = f' {files.model_name}' if files.model_name else ''
    msg = f'AlphaLens {structure}: {kind}{model}, PAE from {pae_path}'
    if dpath:
        msg += f', distogram {dpath}'
    elif contact_probs is not None:
        msg += ', no distogram: contact view from confidence JSON contact_probs, spread unavailable'
    else:
        msg += ', no distogram available (PAE only)'
    session.logger.info(msg)

    # Usable at once; a distogram, if there is one, replaces them when it has been read.
    # They are fixed at 8 Å, so another cutoff has to wait for the distogram.
    if contact_probs is not None and (dpath is None or cutoff == 8.0):
        try:
            pae.set_distogram_metrics(data.contact_metrics(contact_probs, pae_path, cutoff=8.0),
                                      contact_probs.shape[0])
            if dpath is None:
                session.logger.info(f'AlphaLens {structure}: ' + _interface_summary(pae))
        except ValueError as e:
            session.logger.warning(f'AlphaLens {structure}: ignoring contact_probs: {e}')
        if cutoff != 8.0:
            session.logger.warning('confidence JSON contact_probs are fixed at 8 Å; cutoff ignored')
    return pae, dpath, contact_probs, pae_path


def start_distogram(session, structure, pae, plot, dpath, cutoff, mode=None, wait=False,
                    contact_probs=None, pae_path=None, on_loaded=None):
    '''Load a distogram (in a thread in the GUI) and attach the derived matrices.'''
    def done(metrics, error, seconds, shared_with=None):
        _distogram_loaded(session, structure, pae, plot, metrics, error, seconds, mode,
                          contact_probs, pae_path, on_loaded, shared_with)
    shared, other = _read_distogram(session, pae, dpath, cutoff)
    if shared is not None:
        done(shared, None, None, other)
        return
    from .data import LoadProgress
    progress = LoadProgress()
    pae._distogram_progress = progress      # Shown again if the window comes back to this model
    if plot is not None:
        plot.distogram_loading(dpath, progress)
    pae._distogram_load = load = _BackgroundLoad(lambda: _load_metrics(dpath, cutoff, progress), done)
    load.start(session, wait, 'alphalens-distogram')


def _read_distogram(session, pae, dpath, cutoff):
    '''
    The matrices another open model has computed from the same distogram file, and that model:
    the samples of an AlphaFold3 seed share one, so choosing another sample needs no second read.
    Only if the file is unchanged since, and neither model left residues out, so that the
    matrices match row for row. (None, None) otherwise.
    '''
    from .data import file_signature
    models = getattr(session, 'models', None)
    if models is None or getattr(pae, '_rows', None) is not None:
        return None, None
    try:
        real, signature = os.path.realpath(dpath), file_signature(dpath)
    except OSError:
        return None, None
    for s in models.list():
        other = getattr(s, 'alphafold_pae', None)
        if other is None or other is pae or not getattr(other, 'has_distogram', False) \
                or getattr(other, '_rows', 0) is not None:
            continue
        mt = other.metrics
        if (getattr(other, '_full_size', None) == pae._full_size and mt.get('path')
                and mt.get('file_signature') == signature and abs(mt.get('cutoff', 8.0) - cutoff) < 1e-6
                and os.path.realpath(mt['path']) == real):
            return mt, s
    return None, None


def _load_metrics(path, cutoff, progress=None):
    from .data import load_distogram_metrics
    t0 = time.time()
    try:
        m, err = load_distogram_metrics(path, cutoff, progress), None
    except Exception as e:
        m, err = None, f'{type(e).__name__}: {e}'
    return m, err, time.time() - t0


def _distogram_loaded(session, structure, pae, plot, metrics, error, seconds, mode,
                      contact_probs=None, pae_path=None, on_loaded=None, shared_with=None):
    if structure.deleted or getattr(structure, 'alphafold_pae', None) is not pae:
        return              # Closed or superseded while loading
    plot_open = plot is not None and not plot.closed() and plot.pae is pae
    if error is None:
        try:
            pae.set_distogram_metrics(metrics, metrics['contact'].shape[0])
        except ValueError as e:
            error = str(e)
    if error is not None:
        session.logger.warning(f'AlphaLens {structure}: {error}')
        if plot_open:
            plot.distogram_failed(error)
        if contact_probs is not None:
            # Fall back to the AF3 contact probabilities stored with the PAE
            from .data import contact_metrics
            try:
                pae.set_distogram_metrics(contact_metrics(contact_probs, pae_path),
                                          contact_probs.shape[0])
            except ValueError as e:
                session.logger.warning(f'AlphaLens {structure}: ignoring contact_probs: {e}')
            else:
                session.logger.info(f'AlphaLens {structure}: using confidence JSON contact_probs instead')
                if plot_open:
                    plot.set_data_status(f'contact_probs from confidence JSON; distogram failed: {error}')
                    plot.data_changed()
        if on_loaded:
            on_loaded()          # Validate requested operations; never silently skip an export
        return
    how = f'loaded in {seconds:.1f} s' if shared_with is None else f'as read for {shared_with}'
    session.logger.info(f'AlphaLens {structure}: distogram {how}. ' + _interface_summary(pae))
    if plot_open:
        plot.distogram_loaded()
        if mode is not None:
            plot.set_mode(mode)
    if on_loaded:
        on_loaded()


def _interface_summary(pae):
    '''Inter-chain contact probabilities, overall and per chain pair.'''
    from .contacts import row_chain_ids
    chains = row_chain_ids(pae)
    ids = [c for c in dict.fromkeys(chains) if c]
    if len(ids) < 2:
        return ''
    cp_all, pm = pae.metrics['contact'], pae.pae_matrix
    c = pae.metrics['cutoff']
    inter = chains[:, None] != chains[None, :]
    cp = np.where(inter, cp_all, -1)
    i, j = np.unravel_index(np.argmax(cp), cp.shape)
    lines = [f'Inter-chain: max P(d<{c:g} Å) = {cp[i, j]:.2f} '
             f'({pae.row_residue_or_atom_name(i)} – {pae.row_residue_or_atom_name(j)}), '
             f'{int((cp > 0.5).sum() // 2)} residue pairs with P > 0.5, '
             f'min inter-chain PAE {pm[inter].min():.1f} Å']
    if len(ids) > 2:
        lines.append(f'  chains   max P   pairs P>0.5   min PAE (Å)')
        rows = {cid: np.flatnonzero(chains == cid) for cid in ids}
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                ra, rb = rows[ids[a]], rows[ids[b]]
                both = np.maximum(cp_all[np.ix_(ra, rb)], cp_all[np.ix_(rb, ra)].T)
                minpae = min(pm[np.ix_(ra, rb)].min(), pm[np.ix_(rb, ra)].min())
                lines.append(f'  {ids[a]}-{ids[b]}      {both.max():.2f}   '
                             f'{int((both > 0.5).sum()):>8d}      {minpae:.1f}')
    return '\n'.join(lines)


def _resolve_structure(session, structure):
    if structure is not None:
        return structure
    from chimerax.atomic import AtomicStructure
    from chimerax.core.errors import UserError
    ss = session.models.list(type=AtomicStructure)
    if len(ss) == 1:
        return ss[0]
    if not ss:
        raise UserError('No structure open; use "alphalens open <prediction directory>"')
    raise UserError(f'{len(ss)} structures open; specify one, e.g. "alphalens #{ss[0].id_string}"')


# -----------------------------------------------------------------------------
#
def _open_structure(session, spath, log=True, log_info=True, files=None):
    '''
    Open a predicted model file and give it a short recognisable name. log_info False leaves
    out ChimeraX's report on the model (chain table, ModelCIF scores as residue attributes):
    for another model of a prediction already reported. ChimeraX 1.12 makes that report in
    time quadratic in the chain length: 1.5 s of the 1.6 s it takes to open a 2,800-residue
    AF3 model. files: data.find_prediction_files(spath), if already known.
    '''
    from chimerax.core.errors import UserError
    from chimerax.core.commands import run, quote_path_if_necessary
    from chimerax.atomic import AtomicStructure
    from . import data
    models = run(session, f'open {quote_path_if_necessary(spath)}' + ('' if log_info else ' logInfo false'),
                 log=log)
    structs = [m for m in (models or []) if isinstance(m, AtomicStructure)]
    if not structs:
        raise UserError(f'No atomic structure opened from {spath}')
    s = structs[0]
    if files is None:
        files = data.find_prediction_files(spath)
    pred_dir = os.path.dirname(spath)
    stem = os.path.splitext(os.path.basename(spath))[0]
    if files.kind == 'af3':
        if stem == 'model':
            pred_dir = os.path.dirname(pred_dir)
            stem = files.model_name
        else:
            stem = 'top'
    elif files.kind == 'afserver':
        stem = files.model_name.replace(' ', '_')
    kind = {'af2': 'AF2', 'af3': 'AF3', 'afserver': 'AF Server', 'localfold': 'LocalFold'}.get(files.kind, '')
    s.name = ' '.join(x for x in (os.path.basename(pred_dir), kind, stem) if x)
    return s


def alphalens_open(session, path, rank=0, **kw):
    '''Open a prediction directory (or structure file) and show the AlphaLens plot.'''
    from chimerax.core.errors import UserError
    from . import data
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isdir(path):
        try:
            spath = data.structure_path_for_directory(path, rank)
        except ValueError as e:
            raise UserError(str(e))
    elif os.path.exists(path):
        spath = path
    else:
        raise UserError(f'{path} does not exist')
    s = _open_structure(session, spath)
    return alphalens(session, s, **kw)


def read_prediction_files(spath, structure=True):
    '''
    What showing a model needs from disk, read in a thread so that the window stays usable while
    large files, or files on a network mount, come in: the prediction's files, the PAE (with AF3
    contact probabilities), and with structure the model file itself, so that opening it next
    finds it cached. Returns (files, (pae matrix, contact_probs) or None).
    '''
    from . import data
    files = data.find_prediction_files(spath)
    if structure:
        try:
            with open(spath, 'rb') as f:
                while f.read(16 << 20):
                    pass
        except OSError:
            pass
    return files, (data.read_pae(files.pae_path) if files.pae_path else None)


def _pae_ready(s, cutoff):
    '''s has its AlphaLens data attached already, for this contact cutoff.'''
    from .pae import AlphaLensPAE
    pae = getattr(s, 'alphafold_pae', None)
    return isinstance(pae, AlphaLensPAE) and (pae.metrics is None or pae.metrics.get('cutoff') == cutoff)


def _open_model(session, real_path):
    from chimerax.atomic import AtomicStructure
    return next((m for m in session.models.list(type=AtomicStructure) if not m.deleted
                 and getattr(m, 'filename', None) and os.path.realpath(m.filename) == real_path), None)


def _read_then_show(session, plot, label, path, work, show, wait):
    '''
    Run work() (file reads) in a thread, then show(result) here: unless wait, the window stays
    usable and shows the read as progress. A later choice in the window supersedes this one.
    '''
    token = plot.switch_started(label, path)

    def read():
        try:
            return work(), None
        except Exception as e:
            return None, f'{type(e).__name__}: {e}'

    def done(result, error):
        if plot.closed() or not plot.switch_finished(token):
            return                          # Closed, or another model was chosen meanwhile
        try:
            if error is not None:
                raise RuntimeError(error)
            show(result)
        except Exception as e:
            if wait:
                raise
            session.logger.error(f'AlphaLens: could not show {label}: {e}')
        finally:
            if not plot.closed():
                plot.refill_model_menu()
    _BackgroundLoad(read, done).start(session, wait, 'alphalens-model')


def switch_model(session, plot, spath, wait=True):
    '''
    Show another model of the same prediction in an existing plot window. The files of a model
    not shown before are read first, in a thread unless wait is true.
    '''
    real = os.path.realpath(spath)
    s = _open_model(session, real)
    if s is not None and _pae_ready(s, plot.cutoff):
        _show_in_plot(session, plot, s, same_prediction=True)
        return s

    def show(preread):
        from chimerax.core.commands import run
        old = plot.pae.structure
        s = _open_model(session, real)
        if s is None:
            s = _open_structure(session, spath, log_info=False, files=preread[0])   # Already reported
            if not old.deleted:
                try:        # Same sequence, so CA atoms pair up one to one
                    run(session, f'align #{s.id_string}@CA toAtoms #{old.id_string}@CA '
                                 f'move structures', log=False)
                except Exception:
                    pass
        _show_in_plot(session, plot, s, same_prediction=True, preread=preread)
    _read_then_show(session, plot, os.path.basename(os.path.dirname(spath)) if os.path.basename(spath) == 'model.cif'
                    else os.path.basename(spath), spath, lambda: read_prediction_files(spath), show, wait)
    return _open_model(session, real) if wait else None


def show_structure(session, plot, s, wait=True):
    '''
    Show another open prediction in an existing plot window (its Model menu). If that structure
    has a window of its own, that window is brought forward instead. Its files are read first,
    in a thread unless wait is true.
    '''
    other = getattr(s, '_alphalens_plot', None)
    if other is not None and other is not plot and not other.closed():
        other.display(True)
        return other
    if s is plot.pae.structure:
        return plot
    spath = getattr(s, 'filename', None)
    if _pae_ready(s, plot.cutoff) or not spath:
        _show_in_plot(session, plot, s, same_prediction=False)
        return plot

    def show(preread):
        if not s.deleted:
            _show_in_plot(session, plot, s, same_prediction=False, preread=preread)
    _read_then_show(session, plot, str(s), spath, lambda: read_prediction_files(spath, structure=False),
                    show, wait)
    return plot


def _show_in_plot(session, plot, s, same_prediction, preread=None):
    '''
    Show structure s in plot. Another model of the same prediction replaces the shown one in 3D
    too (hidden, as the window is about one model); another prediction leaves it as it is.
    preread: what read_prediction_files() read for s.
    '''
    from . import contacts, predict_compare as pc
    pae = getattr(s, 'alphafold_pae', None)
    dpath = contact_probs = pae_path = None
    if not _pae_ready(s, plot.cutoff):
        pae, dpath, contact_probs, pae_path = prepare_prediction(session, s, cutoff=plot.cutoff,
                                                                 preread=preread)
        shared, other = _read_distogram(session, pae, dpath, plot.cutoff) if dpath else (None, None)
        if shared is not None:              # Before the window draws it: drawn once, complete
            pae.set_distogram_metrics(shared, shared['contact'].shape[0])
            session.logger.info(f'AlphaLens {s}: distogram as read for {other}. ' + _interface_summary(pae))
            dpath = None
    old = plot.pae.structure
    # A comparison with the source follows the shown sample: end it here, start it again below
    following = None
    if not old.deleted and pc.comparing(old):
        following = getattr(old, '_alphalens_superposed', 'chain') if same_prediction else None
        pc.compare_view(session, old, False)
    other = getattr(s, '_alphalens_plot', None)
    if other is not None and other is not plot and not other.closed():
        other.delete()
    if not old.deleted:
        contacts.hide_contacts(old)         # Lines this window drew on it (set_pae: the clicked one)
        if same_prediction:
            old.display = False
        old._alphalens_plot = None
    s.display = True
    s._alphalens_plot = plot
    plot.set_pae(pae, same_prediction=same_prediction)
    if pae.has_contact and dpath is None and pae.metrics.get('contact_only'):
        plot.set_data_status('contact probabilities from confidence JSON; no distogram')
    if dpath:
        start_distogram(session, s, pae, plot, dpath, plot.cutoff, None, False,
                        contact_probs, pae_path, on_loaded=lambda: plot.model_data_loaded(pae))
    else:
        plot.refresh_structure_contacts()
    if following is not None:
        if getattr(s, '_alphalens_superposed', None) is None:
            s._alphalens_superposed = following
        try:
            pc.compare_view(session, s, True)
        except ValueError as e:
            session.logger.warning(f'AlphaLens: {e}')


def has_prediction_data(s):
    '''
    True for an open structure AlphaLens can show: PAE data attached to it, or a PAE file found
    next to its model file (remembered per file, as the menus ask whenever models change).
    '''
    if s.deleted:
        return False
    if getattr(s, 'alphafold_pae', None) is not None:
        return True
    path = getattr(s, 'filename', None)
    if not path:
        return False
    known = getattr(s, '_alphalens_pae_found', None)
    if known is not None and known[0] == path:
        return known[1]
    from . import data
    try:
        found = data.find_prediction_files(path).pae_path is not None
    except Exception:
        found = False
    s._alphalens_pae_found = (path, found)
    return found


def predicted_structures(session):
    '''The open structures AlphaLens can show, in model number order.'''
    from chimerax.atomic import AtomicStructure
    return [s for s in session.models.list(type=AtomicStructure) if has_prediction_data(s)]


def choose_prediction_folder(session, plot=None):
    '''
    Ask for a prediction folder and open its top model: in plot's window if one is given, else in
    a window of its own. Returns the plot window, or None if no folder was chosen.
    '''
    from Qt.QtWidgets import QFileDialog
    parent = plot.tool_window.ui_area if plot is not None and not plot.closed() else session.ui.main_window
    path = QFileDialog.getExistingDirectory(parent, 'Open an AlphaFold prediction folder')
    if not path:
        return None
    if plot is None or plot.closed():
        pae = alphalens_open(session, path)
        return getattr(pae.structure, '_alphalens_plot', None)
    from chimerax.core.errors import UserError
    from . import data
    try:
        spath = data.structure_path_for_directory(path, 0)
    except ValueError as e:
        raise UserError(str(e))
    return show_structure(session, plot, _open_structure(session, spath), wait=False)


# -----------------------------------------------------------------------------
#
def start_tool_from_menu(session):
    '''
    Tools → Structure Prediction → AlphaLens: a new window for the first open prediction that has
    none (shown predictions first); if they all have one, bring those forward; with no
    prediction open, ask for a prediction folder.
    '''
    from chimerax.core.commands import run
    from chimerax.core.errors import UserError
    from .tool import AlphaLensPlot
    plots = [t for t in session.tools.list() if isinstance(t, AlphaLensPlot) and not t.closed()]
    shown_paths = set().union(*(p.model_paths() for p in plots)) if plots else set()
    candidates = sorted(predicted_structures(session), key=lambda s: not s.display)    # Stable
    for s in candidates:
        p = getattr(s, '_alphalens_plot', None)
        if p is not None and not p.closed():
            continue
        f = getattr(s, 'filename', None)
        if f and os.path.realpath(f) in shown_paths:
            continue                        # Another model of a prediction a window shows
        try:
            run(session, f'alphalens #{s.id_string}')
        except UserError as e:
            session.logger.warning(f'AlphaLens: {e}')
            continue
        return getattr(s, '_alphalens_plot', None)
    if plots:
        for p in plots:
            p.display(True)
        session.logger.info('AlphaLens: each open prediction has a window. To look at another one, '
                            'choose "Open a prediction folder…" in a window\'s Model menu.')
        return plots[0]
    if session.models.list():
        session.logger.info('AlphaLens: no open structure has a PAE file next to it; choose a '
                            'prediction folder')
    return choose_prediction_folder(session)


# -----------------------------------------------------------------------------
#
def alphalens_predict(session):
    from chimerax.core.errors import UserError
    if not session.ui.is_gui:
        raise UserError('The prediction dialog requires the ChimeraX GUI.')
    from .predict_ui import show_prediction_dialog
    return show_prediction_dialog(session=session)


def register_commands(logger):
    from chimerax.core.commands import (CmdDesc, register, OpenFileNameArg, FileNameArg,
                                        SaveFileNameArg, FloatArg, BoolArg, IntArg, EnumOf,
                                        ColormapArg, ColormapRangeArg, Or)
    from chimerax.atomic import AtomicStructureArg
    from . import helpdocs
    common = [('distogram', OpenFileNameArg),
              ('mode', EnumOf(MODE_NAMES)),
              ('cutoff', FloatArg),
              ('palette', ColormapArg),
              ('range', ColormapRangeArg),
              ('divider_lines', BoolArg),
              ('contacts', BoolArg),
              ('min_prob', FloatArg),
              ('save_contacts', SaveFileNameArg),
              ('wait', BoolArg)]

    def desc():
        # An open structure, or a prediction folder / model file to open as "alphalens open" does
        return CmdDesc(optional=[('structure', Or(AtomicStructureArg, OpenFileNameArg))],
                       keyword=[('pae_file', OpenFileNameArg), ('rank', IntArg)] + common,
                       synopsis='Show the AlphaLens plot (PAE, contact probability, model '
                                'contacts, spread, agreement) for a predicted structure',
                       url=helpdocs.help_url(helpdocs.REFERENCE, 'commands'))

    def odesc():
        return CmdDesc(required=[('path', FileNameArg)],
                       keyword=[('rank', IntArg)] + common,
                       synopsis='Open an AlphaFold2/AlphaFold3 prediction and show the AlphaLens plot',
                       url=helpdocs.help_url(helpdocs.REFERENCE, 'commands'))

    register('alphalens', desc(), alphalens, logger=logger)
    register('alphalens open', odesc(), alphalens_open, logger=logger)
    register('alphalens predict', CmdDesc(synopsis='Predict a protein or interface from sequences or UniProt IDs',
                                          url=helpdocs.help_url(helpdocs.PREDICTION)),
             alphalens_predict, logger=logger)
