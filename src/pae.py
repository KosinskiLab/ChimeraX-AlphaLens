"""
PAE container that also carries distogram-derived matrices.

Subclasses ChimeraX's AlphaFoldPAE so that everything built on ``structure.alphafold_pae``
(``alphafold pae #1 colorDomains true``, ``alphafold contacts``, PAE domain clustering,
pLDDT coloring) keeps working with data opened by this bundle.
"""

import numpy as np
from chimerax.alphafold.pae import AlphaFoldPAE

# The banded AlphaFold pLDDT colors (very low < 50, low < 70, confident < 90, very high), as
# in the AlphaFold DB and the plot's pLDDT track. ChimeraX's "alphafold" palette blends them.
PLDDT_BANDS = ((50, '#FF7D45'), (70, '#FFDB13'), (90, '#65CBF3'), (100, '#0053D6'))


def plddt_palette(scale=1.0):
    '''"color bfactor" palette with sharp steps at 50, 70 and 90 (times scale).'''
    stops, lo = [], 0
    for hi, color in PLDDT_BANDS:
        start = lo if lo == 0 else lo + 1e-5
        stops += [f'{start * scale:.7g},{color}', f'{hi * scale:.7g},{color}']
        lo = hi
    return ':'.join(stops)


class AlphaLensPAE(AlphaFoldPAE):

    def __init__(self, pae_path, structure, pae_matrix):
        # AlphaFoldPAE.__init__ reads the matrix itself via the module-level
        # read_pae_matrix(); hand it the matrix we already loaded (it may come from a
        # .pkl.gz file the stock reader does not understand).
        import chimerax.alphafold.pae as afpae
        orig = afpae.read_pae_matrix
        afpae.read_pae_matrix = lambda path: pae_matrix
        try:
            AlphaFoldPAE.__init__(self, pae_path, structure)
        finally:
            afpae.read_pae_matrix = orig
        self._full_size = pae_matrix.shape[0]
        self._rows = None               # Matrix rows kept after trimming deleted residues
        self.distogram_path = None
        self._distogram_file = None     # Requested source, retained while loading or on failure
        self.metrics = None             # dict from data.distogram_metrics, trimmed like PAE
        self.agreement = None           # agreement.Agreement over the seeds / models
        self._agreement_pending = None  # Callbacks while the agreement data load
        self._agreement_load = None     # Waitable background reads (results applied on UI thread)
        self._distogram_load = None
        self._model_contacts = None     # Cache of contacts.model_contacts()
        self._model_distances = None    # Cache of contacts.model_distances()

    # ---------------------------------------------------------------------------
    #
    def reduce_matrix_to_residues_in_structure(self):
        rra = self.row_residues_or_atoms()
        if len(rra) == self.matrix_size:
            return True
        from chimerax.alphafold.pae import _include_deleted_residues
        full_rra = _include_deleted_residues(rra)
        if len(full_rra) != self.matrix_size:
            return False
        self._rows = rows = [i for i, ra in enumerate(full_rra) if ra is not None]
        self._pae_matrix = self._pae_matrix[rows, :][:, rows]
        return True

    # ---------------------------------------------------------------------------
    #
    def set_distogram_metrics(self, metrics, full_size):
        """Attach matrices computed from the distogram (full prediction size)."""
        if full_size != self._full_size:
            raise ValueError(f'Distogram covers {full_size} tokens but the PAE matrix has '
                             f'{self._full_size}; is it from the same prediction?')
        rows = self._rows
        out = {}
        for k, v in metrics.items():
            if isinstance(v, np.ndarray) and v.ndim == 2 and rows is not None:
                v = v[rows, :][:, rows]
            out[k] = v
        self.metrics = out
        self.distogram_path = metrics.get('path')

    @property
    def has_distogram(self):
        '''Full distogram loaded (contact, spread and expected distance available).'''
        return self.metrics is not None and 'std' in self.metrics

    @property
    def has_contact(self):
        return self.metrics is not None and 'contact' in self.metrics

    # ---------------------------------------------------------------------------
    #
    def matrix(self, kind):
        if kind == 'pae':
            return self._pae_matrix
        if self.metrics is None:
            return None
        return self.metrics.get(kind)

    # ---------------------------------------------------------------------------
    #
    def row_plddt(self):
        """pLDDT (0-100) per matrix row taken from structure B-factors."""
        from chimerax.atomic import Residue
        vals = []
        for ra in self.row_residues_or_atoms():
            if ra is None or ra.deleted:
                vals.append(np.nan)
            elif isinstance(ra, Residue):
                ca = ra.find_atom('CA') or ra.find_atom("C4'") or ra.find_atom('P')
                vals.append(ca.bfactor if ca is not None else float(np.mean(ra.atoms.bfactors)))
            else:
                vals.append(ra.bfactor)
        p = np.array(vals, dtype=np.float32)
        finite = p[np.isfinite(p)]
        if finite.size and finite.max() <= 1.0:
            p = p * 100            # Some tools store pLDDT as 0-1
        return p

    # ---------------------------------------------------------------------------
    #
    def color_plddt(self, log_command=False):
        '''Color the structure by pLDDT in the banded AlphaFold colors.'''
        m = self.structure
        if m is None or m.deleted:
            return
        bf = m.atoms.bfactors
        scale = 0.01 if len(bf) and bf.max() <= 1.0 else 1.0      # pLDDT stored as 0-1
        cmd = f'color bfactor #{m.id_string} palette "{plddt_palette(scale)}"'
        if not log_command:
            cmd += ' log false'
        from chimerax.core.commands import run
        run(m.session, cmd, log=log_command)
