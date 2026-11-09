"""
Hide parts of a predicted model, per matrix row, and restore exactly what was hidden.

Used by the pLDDT slider and by "Hide selected" in the plot.
"""

import numpy as np


def true_runs(mask):
    '''[(start, end)], end exclusive, of the runs of True in a 1-D bool array.'''
    edges = np.flatnonzero(np.diff(np.concatenate([[0], np.asarray(mask, np.int8), [0]])))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


class RowHider:
    '''
    Hides the ribbons and atoms of matrix rows, for any number of named reasons. A row is
    hidden while any reason covers it. Only what this object turned off is turned on again,
    so side chains that were not shown stay off.
    '''

    def __init__(self, n_rows, residue_rows, residues, atom_rows, atoms):
        self._n = n_rows
        self._residue_rows, self._residues = residue_rows, residues
        self._atom_rows, self._atoms = atom_rows, atoms
        self._ribbon_off = np.zeros(len(residue_rows), bool)     # Ribbons turned off here
        self._atom_off = np.zeros(len(atom_rows), bool)          # Atoms turned off here
        self._reasons = {}
        self.hidden_rows = np.zeros(n_rows, bool)

    @classmethod
    def for_pae(cls, pae):
        from chimerax.atomic import Atoms, Residue, Residues, concatenate
        rows = pae.row_residues_or_atoms()
        residue_rows = np.array([i for i, ra in enumerate(rows) if isinstance(ra, Residue)], int)
        single = [i for i, ra in enumerate(rows) if ra is not None and not isinstance(ra, Residue)]
        residues = Residues([rows[i] for i in residue_rows])
        atoms = residues.atoms
        atom_rows = residue_rows[residues.indices(atoms.residues)]
        if single:              # AF3 tokens that are single atoms of a ligand
            atoms = concatenate([atoms, Atoms([rows[i] for i in single])])
            atom_rows = np.concatenate([atom_rows, np.array(single, int)])
        return cls(len(rows), residue_rows, residues, atom_rows, atoms)

    def apply(self, **reasons):
        '''Set the rows of the given reasons (bool arrays) and update the model.'''
        self._reasons.update(reasons)
        want = np.zeros(self._n, bool)
        for mask in self._reasons.values():
            want |= mask
        self._update(self._residues, 'ribbon_displays', self._residue_rows, self._ribbon_off, want)
        self._update(self._atoms, 'displays', self._atom_rows, self._atom_off, want)
        self.hidden_rows = want

    def restore(self):
        '''Show everything this object hid.'''
        self._reasons.clear()
        self.apply()

    @staticmethod
    def _update(items, attribute, rows, turned_off, want):
        if len(rows) == 0:
            return
        hidden = want[rows]
        shown = getattr(items, attribute)
        turn_on = turned_off & ~hidden
        turn_off = hidden & shown                # Also what was shown again since, e.g. by "show"
        if not (turn_on.any() or turn_off.any()):
            return
        shown[turn_on] = True
        shown[turn_off] = False
        turned_off[turn_on] = False
        turned_off[turn_off] = True
        setattr(items, attribute, shown)

    def selected_rows(self):
        '''Bool array: rows with an atom selected in ChimeraX.'''
        mask = np.zeros(self._n, bool)
        if len(self._atom_rows):
            mask[self._atom_rows[self._atoms.selected]] = True
        return mask
