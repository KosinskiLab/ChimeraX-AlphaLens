"""
Predicted contacts: residue pairs with high contact probability P(d < cutoff).

Used for the contact dots on the plot and for drawing contacts on the structure as
pseudobonds. "Interesting" pairs are inter-chain pairs; for a single-chain prediction,
long-range pairs (|i - j| >= LONG_RANGE).
"""

import numpy as np

LONG_RANGE = 24
GROUP_NAME = 'DistoPAE contacts'

# Probability bands shared by the plot dots, the banded color scale and the pseudobonds.
BAND_EDGES = (0.2, 0.5, 0.8)
BAND_LABELS = ('0.2–0.5', '0.5–0.8', '≥ 0.8')
BAND_RGB = ((252, 146, 114), (239, 59, 44), (165, 15, 21))     # light / mid / dark red


def band_index(p):
    '''0, 1, 2 for the three bands; -1 below the lowest edge.'''
    return np.searchsorted(np.asarray(BAND_EDGES), p, side='right') - 1


def row_chain_ids(pae):
    from chimerax.atomic import Residue
    return np.array([(ra.chain_id if isinstance(ra, Residue) else ra.residue.chain_id)
                     if ra is not None else '' for ra in pae.row_residues_or_atoms()])


def contact_pairs(pae, min_p, region=None, max_pairs=None):
    '''
    Residue-pair contacts with P >= min_p, as (pairs (k, 2) with i < j, probs (k,)),
    strongest first. P is symmetrized as max(P[i,j], P[j,i]).

    region: (r1, r2, c1, c2) inclusive row / column ranges of the plot (a dragged box);
    a pair is kept if one residue is in the rows and the other in the columns.
    '''
    if not pae.has_contact:
        return np.zeros((0, 2), int), np.zeros(0, np.float32)
    P = pae.metrics['contact']
    n = P.shape[0]
    chains = row_chain_ids(pae)
    cand = np.argwhere(np.triu(np.maximum(P, P.T) >= min_p, 1))
    if len(cand) == 0:
        return np.zeros((0, 2), int), np.zeros(0, np.float32)
    i, j = cand[:, 0], cand[:, 1]
    if len(set(chains) - {''}) > 1:
        keep = chains[i] != chains[j]
    else:
        keep = (j - i) >= LONG_RANGE
    if region is not None:
        r1, r2, c1, c2 = region
        in_r = lambda k: (k >= r1) & (k <= r2)
        in_c = lambda k: (k >= c1) & (k <= c2)
        keep &= (in_r(i) & in_c(j)) | (in_c(i) & in_r(j))
    i, j = i[keep], j[keep]
    p = np.maximum(P[i, j], P[j, i])
    order = np.argsort(-p, kind='stable')
    if max_pairs is not None:
        order = order[:max_pairs]
    return np.stack([i[order], j[order]], axis=1), p[order]


def filter_pairs(pairs, probs, min_p, region=None, max_pairs=None):
    '''Subset of (pairs, probs) sorted strongest first with P >= min_p, inside region.'''
    k = int(np.searchsorted(-probs, -min_p, side='right'))
    pairs, probs = pairs[:k], probs[:k]
    if region is not None and len(pairs):
        r1, r2, c1, c2 = region
        i, j = pairs[:, 0], pairs[:, 1]
        in_r = lambda x: (x >= r1) & (x <= r2)
        in_c = lambda x: (x >= c1) & (x <= c2)
        keep = (in_r(i) & in_c(j)) | (in_c(i) & in_r(j))
        pairs, probs = pairs[keep], probs[keep]
    if max_pairs is not None:
        pairs, probs = pairs[:max_pairs], probs[:max_pairs]
    return pairs, probs


def representative_atom(ra):
    from chimerax.atomic import Residue
    if ra is None or ra.deleted:
        return None
    if not isinstance(ra, Residue):
        return ra
    pa = ra.principal_atom          # CA / C4', as used by "alphafold contacts"
    if pa is not None:
        return pa
    for name in ('CA', "C4'", 'P'):
        a = ra.find_atom(name)
        if a is not None:
            return a
    return ra.atoms[0] if ra.num_atoms else None



def ensure_lines_visible(structure):
    '''
    Contact lines are hidden inside a space-filling (sphere) display, which ChimeraX uses
    by default for large multi-chain models. If no cartoon is shown and most atoms are
    displayed, switch to cartoon so the lines can be seen (logged as a command).
    '''
    s = structure
    if s.residues.ribbon_displays.any():
        return
    atoms = s.atoms
    if len(atoms) == 0 or atoms.displays.sum() < 0.5 * len(atoms):
        return
    from chimerax.core.commands import run
    run(s.session, f'cartoon #{s.id_string}; hide #{s.id_string} atoms')
    s.session.logger.info('DistoPAE: switched to cartoon so contact lines are visible')


def show_contacts(pae, min_p=0.5, region=None, max_pairs=2000, log=True, candidates=None):
    '''
    Draw contacts as pseudobonds (CA-CA) colored by probability band. Returns count.
    candidates: precomputed (pairs, probs) from contact_pairs() with a threshold <= min_p.
    '''
    s = pae.structure
    if candidates is not None:
        pairs, probs = filter_pairs(*candidates, min_p, region=region, max_pairs=max_pairs)
    else:
        pairs, probs = contact_pairs(pae, min_p, region=region, max_pairs=max_pairs)
    rra = pae.row_residues_or_atoms()
    if len(pairs):
        ensure_lines_visible(s)
    pbg = s.pseudobond_group(GROUP_NAME, create_type='normal')
    pbg.pseudobonds.delete()
    pbg.dashes = 0
    pbg.display = True
    count = 0
    for (i, j), p in zip(pairs, probs):
        a1, a2 = representative_atom(rra[i]), representative_atom(rra[j])
        if a1 is None or a2 is None:
            continue
        b = pbg.new_pseudobond(a1, a2)
        b.color = BAND_RGB[max(0, band_index(p))] + (255,)
        b.radius = 0.12 + 0.25 * float(p)
        count += 1
    if log:
        where = ''
        if region is not None:
            r1, r2, c1, c2 = region
            name = pae.row_residue_or_atom_name
            where = f' between {name(r1)}–{name(r2)} and {name(c1)}–{name(c2)}'
        msg = f'DistoPAE {s}: showing {count} predicted contacts with P ≥ {min_p:g}{where}'
        if len(pairs) >= max_pairs:
            msg += f' (strongest {max_pairs})'
        if count:
            name = pae.row_residue_or_atom_name
            top = ', '.join(f'{name(i)}–{name(j)} {p:.2f}' for (i, j), p in zip(pairs[:8], probs[:8]))
            msg += f'. Strongest: {top}'
        s.session.logger.info(msg)
    return count


def hide_contacts(structure):
    if structure.deleted:
        return
    pbg = structure.pseudobond_group(GROUP_NAME, create_type=None)
    if pbg is not None:
        pbg.pseudobonds.delete()


# -----------------------------------------------------------------------------
# One picked contact (click on a dot in the plot)
# -----------------------------------------------------------------------------

PICK_GROUP_NAME = 'DistoPAE picked contact'
PICK_RGBA = (255, 0, 0, 255)            # Bright red, drawn thicker than the contact links


def _residue_spec(ra):
    from chimerax.atomic import Residue
    r = ra if isinstance(ra, Residue) else ra.residue
    return f'#{r.structure.id_string}/{r.chain_id}:{r.number}{r.insertion_code}'


def show_picked_contact(pae, i, j, log=True, zoom=False):
    '''Highlight one residue pair: thick red line, residue labels, side chains shown.'''
    from chimerax.atomic import Residue
    from chimerax.core.commands import run
    s = pae.structure
    clear_picked_contact(s)
    ensure_lines_visible(s)
    rra = pae.row_residues_or_atoms()
    ra1, ra2 = rra[i], rra[j]
    a1, a2 = representative_atom(ra1), representative_atom(ra2)
    if a1 is None or a2 is None:
        return None
    g = s.pseudobond_group(PICK_GROUP_NAME, create_type='normal')
    g.dashes = 0
    g.display = True
    b = g.new_pseudobond(a1, a2)
    b.color = PICK_RGBA
    b.radius = 0.6
    # Show side chains (remember which residues we turned on, to undo on the next pick)
    shown = []
    for ra in (ra1, ra2):
        r = ra if isinstance(ra, Residue) else ra.residue
        if not r.atoms.displays.any():
            r.atoms.displays = True
            shown.append(r)
    specs = ' '.join(_residue_spec(ra) for ra in (ra1, ra2))
    run(s.session, f'label {specs} residues', log=False)
    s._distopae_picked = (shown, specs)
    if zoom:
        run(s.session, f'view {specs} pad 0.6', log=False)
    d = float(np.linalg.norm(np.asarray(a1.scene_coord) - np.asarray(a2.scene_coord)))
    if log:
        name = pae.row_residue_or_atom_name
        mt = pae.metrics or {}
        parts = [f'P(d<{mt.get("cutoff", 8):g}Å) {max(mt["contact"][i, j], mt["contact"][j, i]):.2f}'
                 if 'contact' in mt else None,
                 f'PAE {pae.pae_matrix[i, j]:.1f} / {pae.pae_matrix[j, i]:.1f} Å',
                 f'E[d] {mt["mean"][i, j]:.1f} Å' if 'mean' in mt else None,
                 f'{a1.name}–{a2.name} distance in model {d:.1f} Å']
        s.session.logger.info(f'DistoPAE {s}: contact {name(i)} – {name(j)}: '
                              + ', '.join(x for x in parts if x))
    return d


def clear_picked_contact(structure):
    from chimerax.core.commands import run
    if structure.deleted:
        return
    g = structure.pseudobond_group(PICK_GROUP_NAME, create_type=None)
    if g is not None:
        g.pseudobonds.delete()
    picked = getattr(structure, '_distopae_picked', None)
    if picked:
        shown, specs = picked
        for r in shown:
            if not r.deleted:
                r.atoms.displays = False
        try:
            run(structure.session, f'~label {specs} residues', log=False)
        except Exception:
            pass
        structure._distopae_picked = None
