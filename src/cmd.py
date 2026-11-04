"""
Commands:

  distopae open <dir-or-structure-file> [rank N] [mode pae|contact|std|entropy] [cutoff 8]
                [contacts true|false] [minProb 0.5] ...
  distopae [#model] [paeFile path] [distogram path] [mode ...] [cutoff 8] [palette ..] [range ..]
                [contacts true|false] [minProb 0.5]
"""

import os
import threading
import time

import numpy as np

MODE_NAMES = ('pae', 'contact', 'std', 'entropy')


def distopae(session, structure=None, pae_file=None, distogram=None, mode=None, cutoff=8.0,
             palette=None, range=None, divider_lines=None, contacts=None, min_prob=0.5,
             wait=False):
    '''Show PAE / contact probability / distogram spread plot for a predicted structure.'''
    from chimerax.core.errors import UserError
    from .pae import DistoPAE

    structure = _resolve_structure(session, structure)

    pae_colormap = None
    if palette is not None or range is not None:
        from chimerax.core.colors import colormap_with_range
        pae_colormap = colormap_with_range(palette, range, default_colormap_name='pae',
                                           full_range=(0, 30))

    # Re-show an existing plot unless new data or a new cutoff was requested.
    plot = getattr(structure, '_distopae_plot', None)
    pae = getattr(structure, 'alphafold_pae', None)
    reuse = (isinstance(pae, DistoPAE) and pae_file is None and distogram is None
             and (pae.metrics is None or pae.metrics.get('cutoff') == cutoff)
             and (not session.ui.is_gui or (plot is not None and not plot.closed())))
    if reuse:
        if plot is not None and not plot.closed():
            plot.display(True)
            if pae_colormap is not None:
                plot.set_pae_colormap(pae_colormap)
            if divider_lines is not None:
                plot.show_chain_dividers(divider_lines)
            if mode is not None:
                plot.set_mode(mode)
        _apply_contacts(session, pae, plot, contacts, min_prob)
        return pae

    pae, dpath, contact_probs, pae_path = prepare_prediction(session, structure, pae_file,
                                                             distogram, cutoff)

    if mode in ('std', 'entropy') and dpath is None:
        raise UserError(f'Mode "{mode}" needs a distogram and none was found for {structure}')
    if mode == 'contact' and dpath is None and not pae.has_contact:
        raise UserError(f'Mode "contact" needs a distogram or AF3 contact_probs; '
                        f'none found for {structure}')
    if contacts and dpath is None and not pae.has_contact:
        raise UserError(f'"contacts true" needs a distogram or AF3 contact_probs; '
                        f'none found for {structure}')

    plot = None
    if session.ui.is_gui:
        from .tool import DistoPAEPlot
        old = getattr(structure, '_distopae_plot', None)
        if old is not None and not old.closed():
            old.delete()
        dividers = True if divider_lines is None else divider_lines
        plot = DistoPAEPlot(session, pae, cutoff=cutoff, pae_colormap=pae_colormap,
                            divider_lines=dividers)
        structure._distopae_plot = plot
        if pae.has_contact:
            plot.set_data_status('contact probabilities from AF3 confidences; no distogram')
            if mode == 'contact':
                plot.set_mode('contact')

    def loaded():
        _apply_contacts(session, pae, plot, contacts, min_prob)

    if dpath:
        start_distogram(session, structure, pae, plot, dpath, cutoff, mode, wait,
                        contact_probs, pae_path, on_loaded=loaded)
    else:
        loaded()
    return pae


def _apply_contacts(session, pae, plot, contacts, min_prob):
    if contacts is None:
        return
    if plot is not None and not plot.closed():
        plot.show_structure_contacts(contacts, min_prob)
        return
    from . import contacts as c
    if contacts:
        c.show_contacts(pae, min_p=min_prob)
    else:
        c.hide_contacts(pae.structure)


# -----------------------------------------------------------------------------
#
def prepare_prediction(session, structure, pae_file=None, distogram=None, cutoff=8.0):
    '''
    Read PAE (and AF3 contact_probs) for a structure, attach a DistoPAE object as
    structure.alphafold_pae. Returns (pae, distogram_path or None, contact_probs, pae_path).
    '''
    from chimerax.core.errors import UserError
    from . import data
    from .pae import DistoPAE

    spath = getattr(structure, 'filename', None)
    files = data.find_prediction_files(spath) if spath else data.PredictionFiles(None)

    pae_path = pae_file or files.pae_path
    if pae_path is None:
        raise UserError(f'Could not find a PAE file for {structure} '
                        f'(looked next to {spath}); give one with "paeFile <path>"')
    try:
        matrix, contact_probs = data.read_pae(pae_path)
    except Exception as e:
        raise UserError(f'Could not read PAE from {pae_path}: {e}')

    pae = DistoPAE(pae_path, structure, matrix)
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

    kind = {'af2': 'AlphaFold2', 'af3': 'AlphaFold3', 'colabfold': 'ColabFold'}.get(files.kind, '')
    model = f' {files.model_name}' if files.model_name else ''
    msg = f'DistoPAE {structure}: {kind}{model}, PAE from {pae_path}'
    if dpath:
        msg += f', distogram {dpath}'
    elif contact_probs is not None:
        msg += ', no distogram: contact view from AF3 contact_probs, spread unavailable'
    else:
        msg += ', no distogram available (PAE only)'
    session.logger.info(msg)

    if dpath is None and contact_probs is not None:
        try:
            pae.set_distogram_metrics(data.contact_metrics(contact_probs, pae_path, cutoff=8.0),
                                      contact_probs.shape[0])
            session.logger.info(f'DistoPAE {structure}: ' + _interface_summary(pae))
        except ValueError as e:
            session.logger.warning(f'DistoPAE {structure}: ignoring contact_probs: {e}')
        if cutoff != 8.0:
            session.logger.warning('AF3 contact_probs are fixed at 8 Å; cutoff ignored')
    return pae, dpath, contact_probs, pae_path


def start_distogram(session, structure, pae, plot, dpath, cutoff, mode=None, wait=False,
                    contact_probs=None, pae_path=None, on_loaded=None):
    '''Load a distogram (in a thread in the GUI) and attach the derived matrices.'''
    def done(metrics, error, seconds):
        _distogram_loaded(session, structure, pae, plot, metrics, error, seconds, mode,
                          contact_probs, pae_path, on_loaded)
    if plot is not None:
        plot.distogram_loading(dpath)
    if wait or not session.ui.is_gui:
        done(*_load_metrics(dpath, cutoff))
    else:
        def work():
            result = _load_metrics(dpath, cutoff)
            session.ui.thread_safe(done, *result)
        threading.Thread(target=work, name='distopae-distogram', daemon=True).start()


def _load_metrics(path, cutoff):
    from .data import load_distogram_metrics
    t0 = time.time()
    try:
        m, err = load_distogram_metrics(path, cutoff), None
    except Exception as e:
        m, err = None, f'{type(e).__name__}: {e}'
    return m, err, time.time() - t0


def _distogram_loaded(session, structure, pae, plot, metrics, error, seconds, mode,
                      contact_probs=None, pae_path=None, on_loaded=None):
    if structure.deleted or getattr(structure, 'alphafold_pae', None) is not pae:
        return              # Closed or superseded while loading
    plot_open = plot is not None and not plot.closed() and plot.pae is pae
    if error is None:
        try:
            pae.set_distogram_metrics(metrics, metrics['contact'].shape[0])
        except ValueError as e:
            error = str(e)
    if error is not None:
        session.logger.warning(f'DistoPAE {structure}: {error}')
        if plot_open:
            plot.distogram_failed(error)
        if contact_probs is not None:
            # Fall back to the AF3 contact probabilities stored with the PAE
            from .data import contact_metrics
            try:
                pae.set_distogram_metrics(contact_metrics(contact_probs, pae_path),
                                          contact_probs.shape[0])
            except ValueError:
                return
            session.logger.info(f'DistoPAE {structure}: using AF3 contact_probs instead')
            if plot_open:
                plot.set_data_status(f'contact_probs from AF3 confidences; distogram failed: {error}')
                plot.data_changed()
            if on_loaded:
                on_loaded()
        return
    session.logger.info(f'DistoPAE {structure}: distogram loaded in {seconds:.1f} s. '
                        + _interface_summary(pae))
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
        raise UserError('No structure open; use "distopae open <prediction directory>"')
    raise UserError(f'{len(ss)} structures open; specify one, e.g. "distopae #{ss[0].id_string}"')


# -----------------------------------------------------------------------------
#
def _open_structure(session, spath, log=True):
    '''Open a predicted model file and give it a short recognisable name.'''
    from chimerax.core.errors import UserError
    from chimerax.core.commands import run, quote_path_if_necessary
    from chimerax.atomic import AtomicStructure
    from . import data
    models = run(session, f'open {quote_path_if_necessary(spath)}', log=log)
    structs = [m for m in (models or []) if isinstance(m, AtomicStructure)]
    if not structs:
        raise UserError(f'No atomic structure opened from {spath}')
    s = structs[0]
    files = data.find_prediction_files(spath)
    pred_dir = os.path.dirname(spath)
    stem = os.path.splitext(os.path.basename(spath))[0]
    if files.kind == 'af3':
        if stem == 'model':
            pred_dir = os.path.dirname(pred_dir)
            stem = files.model_name
        else:
            stem = 'top'
    kind = {'af2': 'AF2', 'af3': 'AF3'}.get(files.kind, '')
    s.name = ' '.join(x for x in (os.path.basename(pred_dir), kind, stem) if x)
    return s


def distopae_open(session, path, rank=0, **kw):
    '''Open a prediction directory (or structure file) and show the DistoPAE plot.'''
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
    return distopae(session, s, **kw)


def switch_model(session, plot, spath):
    '''Show another model of the same prediction in an existing plot window.'''
    from chimerax.atomic import AtomicStructure
    from chimerax.core.commands import run
    from .pae import DistoPAE
    old = plot.pae.structure
    real = os.path.realpath(spath)
    s = next((m for m in session.models.list(type=AtomicStructure)
              if getattr(m, 'filename', None) and os.path.realpath(m.filename) == real), None)
    if s is None:
        s = _open_structure(session, spath)
        if not old.deleted:
            try:        # Same sequence, so CA atoms pair up one to one
                run(session, f'align #{s.id_string}@CA toAtoms #{old.id_string}@CA '
                             f'move structures', log=False)
            except Exception:
                pass
    pae = getattr(s, 'alphafold_pae', None)
    dpath = contact_probs = pae_path = None
    if not isinstance(pae, DistoPAE) or (pae.metrics is not None
                                         and pae.metrics.get('cutoff') != plot.cutoff):
        pae, dpath, contact_probs, pae_path = prepare_prediction(session, s, cutoff=plot.cutoff)
    other = getattr(s, '_distopae_plot', None)
    if other is not None and other is not plot and not other.closed():
        other.delete()
    if not old.deleted:
        old.display = False
        old._distopae_plot = None
    s.display = True
    s._distopae_plot = plot
    plot.set_pae(pae)
    if pae.has_contact and dpath is None and pae.metrics.get('contact_only'):
        plot.set_data_status('contact probabilities from AF3 confidences; no distogram')
    if dpath:
        start_distogram(session, s, pae, plot, dpath, plot.cutoff, None, False,
                        contact_probs, pae_path, on_loaded=plot.refresh_structure_contacts)
    else:
        plot.refresh_structure_contacts()
    return s


# -----------------------------------------------------------------------------
#
def start_tool_from_menu(session):
    from chimerax.atomic import AtomicStructure
    ss = session.models.list(type=AtomicStructure)
    if len(ss) != 1:
        session.logger.warning('DistoPAE: open a prediction with "distopae open <directory>" '
                               'or run "distopae #N" for an open predicted structure')
        return None
    try:
        distopae(session, ss[0])
    except Exception as e:
        session.logger.warning(f'DistoPAE: {e}')
        return None
    return getattr(ss[0], '_distopae_plot', None)


# -----------------------------------------------------------------------------
#
def register_commands(logger):
    from chimerax.core.commands import (CmdDesc, register, OpenFileNameArg, FileNameArg,
                                        FloatArg, BoolArg, IntArg, EnumOf, ColormapArg,
                                        ColormapRangeArg)
    from chimerax.atomic import AtomicStructureArg
    common = [('distogram', OpenFileNameArg),
              ('mode', EnumOf(MODE_NAMES)),
              ('cutoff', FloatArg),
              ('palette', ColormapArg),
              ('range', ColormapRangeArg),
              ('divider_lines', BoolArg),
              ('contacts', BoolArg),
              ('min_prob', FloatArg),
              ('wait', BoolArg)]
    desc = CmdDesc(optional=[('structure', AtomicStructureArg)],
                   keyword=[('pae_file', OpenFileNameArg)] + common,
                   synopsis='Show PAE / contact probability / distogram spread plot')
    register('distopae', desc, distopae, logger=logger)
    odesc = CmdDesc(required=[('path', FileNameArg)],
                    keyword=[('rank', IntArg)] + common,
                    synopsis='Open AlphaFold2/AlphaFold3 prediction and show DistoPAE plot')
    register('distopae open', odesc, distopae_open, logger=logger)
