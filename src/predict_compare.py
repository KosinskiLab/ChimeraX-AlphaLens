"""
A re-predicted interface against its source: DockQ, and a quick look at both in 3D.

DockQ (Basu & Wallner 2016, PLoS ONE 11:e0161879) with the source model as the reference, over
the residues that were re-predicted (core and flanks), for each pair of chains:

  Fnat   the fraction of the source's residue contacts (any heavy atoms within 5 Å) that the
         re-prediction keeps
  iRMSD  backbone (N, CA, C, O) RMSD of the source's interface residues (heavy atoms within
         10 Å of the other chain), after superposing them
  LRMSD  backbone RMSD of the smaller chain after superposing the larger one
  DockQ  (Fnat + 1/(1 + (iRMSD/1.5)²) + 1/(1 + (LRMSD/8.5)²)) / 3

The source is itself a prediction, so DockQ here measures how alike the two interfaces are, not
how right either is. CAPRI classes: >= 0.80 high, >= 0.49 medium, >= 0.23 acceptable.

The scoring needs only numpy; the 3D view and superposition functions need ChimeraX.
"""
import json
from pathlib import Path

import numpy as np

BACKBONE = ('N', 'CA', 'C', 'O')
CONTACT = 5.0
INTERFACE = 10.0
CLASSES = ((0.80, 'high', 'same interface'), (0.49, 'medium', 'similar interface'),
           (0.23, 'acceptable', 'partly the same interface'), (0.0, 'incorrect', 'a different interface'))
SOURCE_RGBA = (200, 200, 200, 255)          # The source's residues in the comparison view
CHAIN_RGBA = ((31, 119, 180, 255), (255, 127, 14, 255), (44, 160, 44, 255), (214, 39, 40, 255),
              (148, 103, 189, 255), (140, 86, 75, 255), (227, 119, 194, 255), (23, 190, 207, 255))


def dockq_class(score):
    '''(CAPRI class, plain words) of a DockQ score.'''
    return next((name, words) for edge, name, words in CLASSES if score >= edge)


def superposition(mobile, fixed):
    '''Rotation R and translation t that best place mobile (n, 3) onto fixed: x -> R x + t.'''
    mc, fc = mobile.mean(0), fixed.mean(0)
    u, _, vt = np.linalg.svd((mobile - mc).T @ (fixed - fc))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r, fc - r @ mc


def rmsd(a, b):
    return float(np.sqrt(((a - b) ** 2).sum(1).mean()))


def _close_pairs(a_xyz, a_res, b_xyz, b_res, cutoff, block=512):
    '''Residue pairs (a residue, b residue) with any atoms closer than cutoff.'''
    pairs = set()
    c2 = np.float32(cutoff * cutoff)
    for s in range(0, len(a_xyz), block):
        d2 = ((a_xyz[s:s + block, None, :] - b_xyz[None, :, :]) ** 2).sum(-1)
        i, j = np.nonzero(d2 < c2)
        pairs.update(zip(a_res[s:s + block][i].tolist(), b_res[j].tolist()))
    return pairs


def dockq(native, model, chain_of):
    '''
    DockQ of each pair of chains with source contacts.

    native, model: one dict per residue, in the same order, {atom name: (x, y, z)} of its heavy
    atoms in the source and in the re-prediction; chain_of: the chain of each residue. Only the
    atoms a residue has in both are used. Returns [dict(chains, receptor, ligand, dockq, fnat,
    kept, native_contacts, new_contacts, irmsd, lrmsd, class, words)], and [chain pairs without
    source contacts].
    '''
    names, n_xyz, m_xyz, res = [], [], [], []
    for k, (a, b) in enumerate(zip(native, model)):
        for name in sorted(set(a) & set(b)):
            names.append(name); n_xyz.append(a[name]); m_xyz.append(b[name]); res.append(k)
    n_xyz = np.asarray(n_xyz, np.float32).reshape(-1, 3)
    m_xyz = np.asarray(m_xyz, np.float32).reshape(-1, 3)
    res = np.asarray(res, int)
    names = np.asarray(names)
    chain_of = np.asarray(chain_of)
    atom_chain = chain_of[res] if len(res) else np.zeros(0, chain_of.dtype)
    backbone = np.isin(names, BACKBONE)
    chains = list(dict.fromkeys(chain_of.tolist()))
    results, empty = [], []
    for x, c1 in enumerate(chains):
        for c2 in chains[x + 1:]:
            a, b = atom_chain == c1, atom_chain == c2
            native_contacts = _close_pairs(n_xyz[a], res[a], n_xyz[b], res[b], CONTACT)
            if not native_contacts:
                empty.append(f'{c1}–{c2}')
                continue
            model_contacts = _close_pairs(m_xyz[a], res[a], m_xyz[b], res[b], CONTACT)
            kept = len(native_contacts & model_contacts)
            fnat = kept / len(native_contacts)
            near = _close_pairs(n_xyz[a], res[a], n_xyz[b], res[b], INTERFACE)
            interface = {i for i, _ in near} | {j for _, j in near}
            sel = backbone & np.isin(res, sorted(interface))
            r, t = superposition(m_xyz[sel].astype(float), n_xyz[sel].astype(float))
            irmsd = rmsd(m_xyz[sel] @ r.T + t, n_xyz[sel])
            size = {c: int((chain_of == c).sum()) for c in (c1, c2)}
            receptor, ligand = (c1, c2) if size[c1] >= size[c2] else (c2, c1)
            rec, lig = backbone & (atom_chain == receptor), backbone & (atom_chain == ligand)
            r, t = superposition(m_xyz[rec].astype(float), n_xyz[rec].astype(float))
            lrmsd = rmsd(m_xyz[lig] @ r.T + t, n_xyz[lig])
            score = (fnat + 1 / (1 + (irmsd / 1.5) ** 2) + 1 / (1 + (lrmsd / 8.5) ** 2)) / 3
            name, words = dockq_class(score)
            results.append(dict(chains=f'{c1}–{c2}', receptor=receptor, ligand=ligand, dockq=round(score, 3),
                                fnat=round(fnat, 3), kept=kept, native_contacts=len(native_contacts),
                                new_contacts=len(model_contacts - native_contacts),
                                irmsd=round(irmsd, 2), lrmsd=round(lrmsd, 2), **{'class': name}, words=words))
    return results, empty


def summary_lines(interface):
    '''Report lines for comparison.json's "interface".'''
    if not interface:
        return []
    pairs, empty = interface.get('pairs', []), interface.get('without_source_contacts', [])
    if not pairs:
        return [f'Interface vs source: no source contacts between {", ".join(empty) or "the chains"} '
                'over these residues, so nothing to compare']
    lines = []
    if len(pairs) > 1:
        lines.append(f'Interface vs source: DockQ {interface["dockq"]:.2f} on average over {len(pairs)} '
                     'chain pairs (re-predicted residues; the source as reference)')
    for p in pairs:
        head = (f'Interface vs source: DockQ {p["dockq"]:.2f} · {p["class"]}: {p["words"]}' if len(pairs) == 1
                else f'  {p["chains"]}: DockQ {p["dockq"]:.2f} · {p["class"]}: {p["words"]}')
        lines.append(head)
        lines.append(f'  {p["chains"]}: {p["kept"]} of {p["native_contacts"]} source contacts kept '
                     f'(Fnat {p["fnat"]:.2f}), {p["new_contacts"]} new; interface RMSD {p["irmsd"]:.1f} Å; '
                     f'{p["ligand"]} {p["lrmsd"]:.1f} Å from its source place with {p["receptor"]} superposed')
    return lines


# -----------------------------------------------------------------------------
# With ChimeraX: the atoms of both models, the comparison view, superposition
# -----------------------------------------------------------------------------

def heavy_atoms(residue, scene=True):
    '''{name: (x, y, z)} of a residue's heavy atoms, scene coordinates (else the model's own).'''
    return {a.name: [float(v) for v in (a.scene_coord if scene else a.coord)]
            for a in residue.atoms if a.element.name != 'H'}


def interface_scores(job_chains, result_rows):
    '''DockQ against the source atoms recorded at submission (residue_map "source_atoms").'''
    native, model, chain_of = [], [], []
    rows = iter(result_rows)
    for chain in job_chains:
        for entry in chain['residue_map']:
            r = next(rows)
            if entry.get('source_atoms') is None or r is None:
                continue
            native.append(entry['source_atoms'])
            model.append(heavy_atoms(r))
            chain_of.append(chain['source_chain'])
    if len(set(chain_of)) < 2:
        return None
    pairs, empty = dockq(native, model, chain_of)
    out = dict(pairs=pairs, without_source_contacts=empty, residues='re-predicted (with flanks)')
    if pairs:
        out['dockq'] = round(float(np.mean([p['dockq'] for p in pairs])), 3)
        out['class'], out['words'] = dockq_class(out['dockq'])
    return out


def job_files(structure):
    '''(job directory, residue_map, comparison) of a local prediction result, else None.'''
    path = getattr(structure, 'filename', None)
    if not path:
        return None
    d = Path(path).parent
    try:
        chains = json.loads((d / 'residue_map.json').read_text(encoding='utf-8'))
        comparison = json.loads((d / 'comparison.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    return d, chains, comparison


def find_source(session, comparison):
    '''The open source model of a comparison: the same file, else the same model id and name.'''
    from chimerax.atomic import AtomicStructure
    import os
    source = comparison.get('source') or {}
    path = source.get('file')
    models = [m for m in session.models.list(type=AtomicStructure) if not m.deleted]
    if path:
        real = os.path.realpath(path)
        hit = next((m for m in models if getattr(m, 'filename', None)
                    and os.path.realpath(m.filename) == real), None)
        if hit is not None:
            return hit
    return next((m for m in models if m.id_string == source.get('model_id')
                 and m.name == source.get('model')), None)


def matched_residues(source, chains, result):
    '''[(source residue or None, result residue)] in result order, by the recorded source labels.'''
    lookup = {(r.chain_id, int(r.number), r.insertion_code): r for r in source.residues} if source else {}
    from chimerax.atomic import Residue
    out = []
    rows = [ra for ra in result.alphafold_pae.row_residues_or_atoms()] if getattr(result, 'alphafold_pae', None) \
        else list(result.residues)
    rows = iter(rows)
    for chain in chains:
        for entry in chain['residue_map']:
            r = next(rows, None)
            key = (entry.get('chain_id'), int(entry.get('number', 0)), entry.get('insertion_code') or '')
            out.append((lookup.get(key), r if isinstance(r, Residue) else None, chain, entry))
    return out


def superpose(session, result, how='chain'):
    '''
    Move the re-prediction onto its source: on the first selected chain's core CA atoms ('chain',
    as it opens) or on the backbone of the source's interface residues ('interface'). Uses the
    source as it is now if it is open, else as it was at submission. Returns the RMSD (Å).
    '''
    from chimerax.geometry import align_points
    found = job_files(result)
    if found is None:
        raise ValueError(f'{result} is not a re-predicted interface with a source to compare.')
    _, chains, comparison = found
    source = find_source(session, comparison)
    matched = matched_residues(source, chains, result)
    mobile, fixed = [], []
    if how == 'chain':
        first = chains[0]
        for src, res, chain, entry in matched:
            if chain is not first or entry['source_row'] not in first['core_rows'] or res is None:
                continue
            ca = res.find_atom('CA')
            target = (src.find_atom('CA').scene_coord if src is not None and src.find_atom('CA') is not None
                      else entry.get('source_ca_scene'))
            if ca is not None and target is not None:
                mobile.append(ca.coord); fixed.append(target)
    else:
        keys = interface_residue_keys(matched)
        for k, (src, res, chain, entry) in enumerate(matched):
            if k not in keys or res is None:
                continue
            target_atoms = heavy_atoms(src) if src is not None else (entry.get('source_atoms') or {})
            for name in BACKBONE:
                a = res.find_atom(name)
                if a is not None and name in target_atoms:
                    mobile.append(a.coord); fixed.append(target_atoms[name])
    if len(mobile) < 3:
        raise ValueError('Too few matching atoms to superpose on.')
    transform, value = align_points(np.array(mobile, float), np.array(fixed, float))
    result.scene_position = transform
    result._alphalens_superposed = how        # Compare superposes the same way again
    return float(value)


def interface_residue_keys(matched):
    '''Indexes (in result order) of the source's interface residues: heavy atoms within 10 Å of
    another chain, from the source atoms recorded at submission.'''
    xyz, res, chain = [], [], []
    for k, (src, r, ch, entry) in enumerate(matched):
        for v in (entry.get('source_atoms') or {}).values():
            xyz.append(v); res.append(k); chain.append(ch['source_chain'])
    xyz, res, chain = np.asarray(xyz, np.float32).reshape(-1, 3), np.asarray(res), np.asarray(chain)
    keys = set()
    for c in dict.fromkeys(chain.tolist()):
        a, b = chain == c, chain != c
        if a.any() and b.any():
            keys |= {i for i, _ in _close_pairs(xyz[a], res[a], xyz[b], res[b], INTERFACE)}
    return keys


def comparing(result):
    return getattr(result, '_alphalens_compare', None) is not None


def compare_view(session, result, on=True):
    '''
    Show the re-prediction (colored by chain) with the same residues of its source (grey) only,
    and look at them; on=False puts back the colors and display of both. Returns a message.
    '''
    from chimerax.core.commands import run
    from chimerax.alphafold.pae import _residue_and_atom_spec
    saved = getattr(result, '_alphalens_compare', None)
    if not on:
        if saved is None:
            return 'Nothing to restore.'
        for m, state in saved['state'].items():
            if not m.deleted and len(m.residues) == len(state[0]) and len(m.atoms) == len(state[2]):
                m.residues.ribbon_displays, m.residues.ribbon_colors = state[0], state[1]
                m.atoms.displays, m.atoms.colors = state[2], state[3]
                m.display = state[4]
        result._alphalens_compare = None
        return 'Restored the colors and display of the re-prediction and its source.'
    found = job_files(result)
    if found is None:
        raise ValueError(f'{result} is not a re-predicted interface with a source to compare.')
    _, chains, comparison = found
    source = find_source(session, comparison)
    if source is None:
        raise ValueError('The source model is not open: open '
                         f'{(comparison.get("source") or {}).get("file") or "it"} to compare.')
    matched = matched_residues(source, chains, result)
    # On the source, as last chosen (the first chain unless the interface was chosen): a job
    # folder opened later is where it was predicted, and the source may have moved since
    try:
        superpose(session, result, getattr(result, '_alphalens_superposed', 'chain'))
    except ValueError:
        pass
    state = {}
    for m in (source, result):
        state[m] = (m.residues.ribbon_displays.copy(), m.residues.ribbon_colors.copy(),
                    m.atoms.displays.copy(), m.atoms.colors.copy(), m.display)
    if saved is None:
        result._alphalens_compare = dict(state=state)
    source_res = [s for s, _, _, _ in matched if s is not None]
    # Source: only the re-predicted residues, grey; re-prediction: by chain
    source.residues.ribbon_displays = False
    source.atoms.displays = False
    from chimerax.atomic import Residues
    sr = Residues(source_res)
    sr.ribbon_displays = True
    sr.ribbon_colors = SOURCE_RGBA
    result.residues.ribbon_displays = True
    result.atoms.displays = False
    for k, chain in enumerate(chains):
        rs = Residues([r for _, r, ch, _ in matched if ch is chain and r is not None])
        rs.ribbon_colors = CHAIN_RGBA[k % len(CHAIN_RGBA)]
    source.display = result.display = True
    spec = _residue_and_atom_spec(source_res) + ' ' + f'#{result.id_string}'
    # Cartoons just switched on are built when a frame is drawn; "view" sees only what is drawn
    try:
        session.update_loop.update_graphics_now()
        run(session, f'view {spec}', log=False)
    except Exception:
        pass                # The comparison stands without the camera move
    return (f'Comparing {result} (by chain) with the same residues of {source} (grey). Right-click the '
            'plot > Compare with source again to restore their colors and display.')
