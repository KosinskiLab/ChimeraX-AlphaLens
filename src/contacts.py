"""
Predicted contacts: residue pairs with high contact probability P(d < cutoff).

Used for the contact dots on the plot, for drawing contacts on the structure as
pseudobonds and for the CSV export. "Interesting" pairs are inter-chain pairs; for a
single-chain prediction, long-range pairs (|i - j| >= LONG_RANGE).

Also the distances and contacts in the model itself (model_distances, model_contacts),
measured between the atoms the distogram describes.
"""

import numpy as np

LONG_RANGE = 24
GROUP_NAME = 'AlphaLens contacts'
DEFAULT_MIN_P = 0.2         # Initial slider threshold; also used by "saveContacts" without a plot

# Band edges of the banded contact color scale of the matrix.
BAND_EDGES = (0.2, 0.5, 0.8)

# AlphaFold pLDDT colors, lowest band first: very low, low, confident, very high.
PLDDT_COLORS = ((255, 125, 69), (255, 219, 19), (101, 203, 243), (0, 83, 214))

# Predicted contacts (plot dots and lines on the structure) use the pLDDT bands with P x 100
# as the score: orange P < 0.5, yellow 0.5-0.7, light blue 0.7-0.9, dark blue >= 0.9. The
# color depends on P alone, never on the slider threshold, so it reads as a confidence.
CONTACT_COLOR_EDGES = (0.5, 0.7, 0.9)
CONTACT_COLORS = PLDDT_COLORS


def contact_band(p):
    '''Index into CONTACT_COLORS of contact probabilities p (0 = lowest band).'''
    p = np.asarray(p, np.float32)       # float32 edges too: P stored as 0.9 is in the 0.9 band
    return np.searchsorted(np.array(CONTACT_COLOR_EDGES, np.float32), p, side='right')


def contact_rgb(p):
    '''(k, 3) uint8 colors of contact probabilities p in the pLDDT-style bands.'''
    return np.array(CONTACT_COLORS, np.uint8)[contact_band(p)]


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
    keep = _keep(chains, i, j, region)
    i, j = i[keep], j[keep]
    p = np.maximum(P[i, j], P[j, i])
    order = np.argsort(-p, kind='stable')
    if max_pairs is not None:
        order = order[:max_pairs]
    return np.stack([i[order], j[order]], axis=1), p[order]


def _keep(chains, i, j, region=None):
    '''Mask of pairs (i < j) to show: inter-chain (long-range for one chain), inside region.'''
    if len(set(chains) - {''}) > 1:
        keep = chains[i] != chains[j]
    else:
        keep = (j - i) >= LONG_RANGE
    if region is not None:
        r1, r2, c1, c2 = region
        in_r = lambda k: (k >= r1) & (k <= r2)
        in_c = lambda k: (k >= c1) & (k <= c2)
        keep &= (in_r(i) & in_c(j)) | (in_c(i) & in_r(j))
    return keep


def region_text(pae, region):
    if region is None:
        return ''
    r1, r2, c1, c2 = region
    name = pae.row_residue_or_atom_name
    return f' between {name(r1)}–{name(r2)} and {name(c1)}–{name(c2)}'


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
    s.session.logger.info('AlphaLens: switched to cartoon so contact lines are visible')


MAX_LINES = 10000            # Most contact lines drawn on the structure at once


def show_contacts(pae, min_p=0.5, region=None, max_pairs=MAX_LINES, log=True, candidates=None):
    '''
    Draw contacts as pseudobonds (CA-CA), colored by P (contact_rgb) and thicker for higher
    P. Returns count.
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
    ends = [(representative_atom(rra[i]), representative_atom(rra[j])) for i, j in pairs]
    ok = np.array([a1 is not None and a2 is not None for a1, a2 in ends], bool)
    count = int(ok.sum())
    if count:
        from chimerax.atomic import Atoms
        a1 = Atoms([e[0] for e, k in zip(ends, ok) if k])
        a2 = Atoms([e[1] for e, k in zip(ends, ok) if k])
        b = pbg.new_pseudobonds(a1, a2)
        p = probs[ok]
        rgba = np.empty((count, 4), np.uint8)
        rgba[:, :3] = contact_rgb(p)
        rgba[:, 3] = np.round(255 * (0.3 + 0.7 * p))     # Weak contacts fade into the background
        b.colors = rgba
        b.radii = (0.12 + 0.25 * p).astype(np.float32)
    if log:
        msg = (f'AlphaLens {s}: showing {count} predicted contacts with P ≥ {min_p:g}'
               f'{region_text(pae, region)}')
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
# Export to CSV
# -----------------------------------------------------------------------------

def export_pairs(pae, min_p, region=None, agreement=None):
    '''
    Pairs (k, 2), i < j, with P >= min_p in the shown model or, with agreement, in any seed
    or model; inter-chain (long-range for one chain), inside region. Sorted by the number
    of seeds / models with P >= min_p, then by P in the shown model.
    '''
    P = pae.metrics['contact'] if pae.has_contact else None
    n = pae.pae_matrix.shape[0]
    mask = np.zeros((n, n), bool)
    if P is not None:
        mask |= np.maximum(P, P.T) >= min_p
    counts = agreement.counts(min_p) if agreement is not None else None
    if counts is not None:
        mask |= counts > 0
    cand = np.argwhere(np.triu(mask, 1))
    if len(cand) == 0:
        return np.zeros((0, 2), int)
    i, j = cand[:, 0], cand[:, 1]
    keep = _keep(row_chain_ids(pae), i, j, region)
    i, j = i[keep], j[keep]
    p = np.maximum(P[i, j], P[j, i]) if P is not None else np.zeros(len(i))
    c = counts[i, j] if counts is not None else np.zeros(len(i))
    order = np.lexsort((-p, -c))
    return np.stack([i[order], j[order]], axis=1)


def _row_fields(ra):
    '''chain, residue number, residue name, atom name (atoms of ligands only).'''
    from chimerax.atomic import Residue
    if ra is None:
        return ['', '', '', '']
    r = ra if isinstance(ra, Residue) else ra.residue
    return [r.chain_id, f'{r.number}{r.insertion_code}', r.name,
            '' if isinstance(ra, Residue) else ra.name]


def contact_rows(pae, pairs, min_p, agreement=None):
    '''Header and rows of the contact table for export_pairs() output.'''
    mt = pae.metrics or {}
    P, mean = mt.get('contact'), mt.get('mean')
    pm = pae.pae_matrix
    rra = pae.row_residues_or_atoms()
    header = ['chain_1', 'residue_1', 'name_1', 'atom_1', 'chain_2', 'residue_2', 'name_2',
              'atom_2', 'p_contact', 'pae_1_2', 'pae_2_1', 'expected_distance', 'model_distance']
    if agreement is not None:
        counts = agreement.counts(min_p)
        kind = agreement.kind
        header += [f'{kind}s_with_p_ge_{min_p:.2f}', f'{kind}s']
        header += ['p_' + label.replace(' ', '_') for label in agreement.labels]
    f2 = lambda x: f'{float(x):.2f}'
    f3 = lambda x: f'{float(x):.3f}'
    rows = []
    for i, j in pairs:
        row = _row_fields(rra[i]) + _row_fields(rra[j])
        row.append(f3(max(P[i, j], P[j, i])) if P is not None else '')
        row += [f2(pm[i, j]), f2(pm[j, i])]
        row.append(f2(mean[i, j]) if mean is not None else '')
        a1, a2 = representative_atom(rra[i]), representative_atom(rra[j])
        row.append(f2(np.linalg.norm(np.asarray(a1.coord) - np.asarray(a2.coord)))
                   if a1 is not None and a2 is not None else '')
        if agreement is not None:
            row += [int(counts[i, j]), agreement.n] + [f3(v) for v in agreement.pair_values(i, j)]
        rows.append(row)
    return header, rows


def write_contacts_csv(pae, path, min_p, region=None, agreement=None):
    '''Write the contacts with P >= min_p to a CSV file. Returns the number of contacts.'''
    import csv
    pairs = export_pairs(pae, min_p, region=region, agreement=agreement)
    header, rows = contact_rows(pae, pairs, min_p, agreement=agreement)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return len(rows)


# -----------------------------------------------------------------------------
# One picked contact (click on a dot in the plot)
# -----------------------------------------------------------------------------

PICK_GROUP_NAME = 'AlphaLens picked contact'
PICK_RGB = (230, 0, 230)                # Magenta, as the ring around the clicked dot in the plot;
PICK_RGBA = PICK_RGB + (255,)           # not a contact color. Thicker than the contact lines.


def _residue_spec(ra):
    from chimerax.atomic import Residue
    r = ra if isinstance(ra, Residue) else ra.residue
    return f'#{r.structure.id_string}/{r.chain_id}:{r.number}{r.insertion_code}'


def show_picked_contact(pae, i, j, log=True, zoom=False):
    '''Highlight one residue pair: thick magenta line, residue labels, side chains shown.'''
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
    s._alphalens_picked = (shown, specs)
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
        s.session.logger.info(f'AlphaLens {s}: contact {name(i)} – {name(j)}: '
                              + ', '.join(x for x in parts if x))
    return d


def clear_picked_contact(structure):
    from chimerax.core.commands import run
    if structure.deleted:
        return
    g = structure.pseudobond_group(PICK_GROUP_NAME, create_type=None)
    if g is not None:
        g.pseudobonds.delete()
    picked = getattr(structure, '_alphalens_picked', None)
    if picked:
        shown, specs = picked
        for r in shown:
            if not r.deleted:
                r.atoms.displays = False
        try:
            run(structure.session, f'~label {specs} residues', log=False)
        except Exception:
            pass
        structure._alphalens_picked = None


# -----------------------------------------------------------------------------
# Contacts in the model
# -----------------------------------------------------------------------------

PURINES = {'A', 'G', 'DA', 'DG'}
PYRIMIDINES = {'C', 'U', 'DC', 'DT'}


def distogram_atom(ra):
    '''
    The atom whose distances the distogram predicts (AlphaFold2 and AlphaFold3): CB (CA for
    glycine) for amino acids, C4 for purines, C2 for pyrimidines, the atom itself for ligand
    atoms. Falls back to the principal atom (CA / C4') if that atom is missing.
    '''
    from chimerax.atomic import Residue
    if ra is None or ra.deleted:
        return None
    if not isinstance(ra, Residue):
        return ra
    if ra.polymer_type == Residue.PT_PROTEIN:
        a = ra.find_atom('CA' if ra.name == 'GLY' else 'CB')
    elif ra.name in PURINES:
        a = ra.find_atom('C4')
    elif ra.name in PYRIMIDINES:
        a = ra.find_atom('C2')
    else:
        a = None
    return a if a is not None else representative_atom(ra)


def contact_mask(coords, cutoff, chunk=256):
    '''N x N boolean matrix of d(i, j) < cutoff; rows with NaN coordinates have no contacts.'''
    xyz = np.asarray(coords, np.float32)
    n = len(xyz)
    mask = np.zeros((n, n), bool)
    c2 = float(cutoff) ** 2
    for i0 in range(0, n, chunk):
        d = xyz[i0:i0 + chunk, None, :] - xyz[None, :, :]
        with np.errstate(invalid='ignore'):
            mask[i0:i0 + chunk] = (d * d).sum(axis=-1) < c2
    return mask


def distance_matrix(coords, chunk=256):
    '''N x N float32 distances between coords; NaN in rows and columns without coordinates.'''
    xyz = np.asarray(coords, np.float32)
    n = len(xyz)
    d = np.empty((n, n), np.float32)
    for i0 in range(0, n, chunk):
        diff = xyz[i0:i0 + chunk, None, :] - xyz[None, :, :]
        d[i0:i0 + chunk] = np.sqrt((diff * diff).sum(axis=-1))
    return d


def model_distances(pae):
    '''
    Distances in the model between the distogram_atom()s of the plot rows (N x N, Å, NaN where
    a row has no atom) and the (N, 3) coordinates used. Cached on the PAE object, which belongs
    to one model.
    '''
    cached = getattr(pae, '_model_distances', None)
    atoms = [distogram_atom(ra) for ra in pae.row_residues_or_atoms()]
    xyz = np.array([a.coord if a is not None else (np.nan,) * 3 for a in atoms], np.float32)
    if cached is not None and np.array_equal(cached[1], xyz, equal_nan=True):
        return cached
    pae._model_contacts = None
    pae._model_distances = (distance_matrix(xyz), xyz)
    return pae._model_distances


def model_contacts(pae, cutoff):
    '''
    Contacts in the model, d < cutoff between distogram_atom()s, as an N x N boolean matrix
    over the plot rows, and the (N, 3) coordinates used (NaN where a row has no atom).
    '''
    dist, xyz = model_distances(pae)
    cached = getattr(pae, '_model_contacts', None)
    if cached is not None and cached[0] == cutoff:
        return cached[1], cached[2]
    with np.errstate(invalid='ignore'):
        mask = dist < cutoff                # NaN (no atom) compares False
    pae._model_contacts = (cutoff, mask, xyz)
    return mask, xyz


DISTANCE_FAR = 40.0         # Model distances view: grey from the cutoff (dark) to this (white)


def distance_view_values(dist, cutoff, contact_values=None, floor=0.0, far=DISTANCE_FAR):
    '''
    Values of the Model distances view, for one colormap that is grey below 0 and the
    contact-probability colormap from 0 to 1:

      pairs closer than cutoff   their contact value (predicted P on the shown contact scale,
                                 at least floor so none disappears; 1 without contact values)
      other pairs                -1 - t, t from 0 at the cutoff (dark grey) to 1 at far (white)
      no coordinates             -2 (white)

    Max-pooling a zoomed-out overview keeps contacts over distances, the highest P among
    contacts and the closest pair among the rest.
    '''
    dist = np.asarray(dist, np.float32)
    with np.errstate(invalid='ignore'):
        close = dist < cutoff
        t = np.clip((dist - cutoff) / (far - cutoff), 0, 1)
    vals = np.where(np.isnan(dist), np.float32(-2), -1 - t)
    if contact_values is None:
        c = np.float32(1)
    else:
        c = np.maximum(np.asarray(contact_values, np.float32), floor)
    return np.where(close, c, vals).astype(np.float32)


def model_distance(pae, i, j):
    '''(distance, "CB–CB") between the distogram atoms of rows i and j in the model, or None.'''
    rra = pae.row_residues_or_atoms()
    a1, a2 = distogram_atom(rra[i]), distogram_atom(rra[j])
    if a1 is None or a2 is None:
        return None
    d = float(np.linalg.norm(np.asarray(a1.coord, np.float64) - np.asarray(a2.coord, np.float64)))
    return d, f'{a1.name}–{a2.name}'
