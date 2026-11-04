"""
Run the parts of the bundle that do not need ChimeraX: import src/ as the package
"distopae" without executing its __init__ (which needs ChimeraX), and give contacts.py
a minimal chimerax.atomic when ChimeraX is not available.
"""

import os
import sys
import types

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'src'))

pkg = types.ModuleType('distopae')
pkg.__path__ = [SRC]
sys.modules.setdefault('distopae', pkg)

try:
    import chimerax.atomic      # noqa: F401
except ImportError:
    chimerax = types.ModuleType('chimerax')
    chimerax.__path__ = []
    atomic = types.ModuleType('chimerax.atomic')

    class Residue:
        def __init__(self, chain_id, number, name, coord, insertion_code=''):
            self.chain_id, self.number, self.name = chain_id, number, name
            self.insertion_code = insertion_code
            self.deleted = False
            self.principal_atom = Atom('CA', coord, self)

    class Atom:
        def __init__(self, name, coord, residue):
            self.name, self.coord, self.residue = name, coord, residue
            self.deleted = False

    atomic.Residue, atomic.Atom = Residue, Atom
    chimerax.atomic = atomic
    sys.modules['chimerax'] = chimerax
    sys.modules['chimerax.atomic'] = atomic
