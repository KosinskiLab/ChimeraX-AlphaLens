"""Contact colors, contacts in the model and the pLDDT palette (no ChimeraX needed)."""

import importlib
import sys
import types

import numpy as np

from alphalens import contacts


def _pae_module():
    '''src/pae.py, with a stand-in for chimerax.alphafold.pae outside ChimeraX.'''
    try:
        import chimerax.alphafold.pae       # noqa: F401
    except ImportError:
        af = types.ModuleType('chimerax.alphafold')
        af.__path__ = []
        afpae = types.ModuleType('chimerax.alphafold.pae')
        afpae.AlphaFoldPAE = object
        sys.modules.setdefault('chimerax.alphafold', af)
        sys.modules.setdefault('chimerax.alphafold.pae', afpae)
    return importlib.import_module('alphalens.pae')


def test_contact_rgb_is_blue_and_darker_for_higher_p():
    rgb = contacts.contact_rgb(np.linspace(0, 1, 11)).astype(int)
    assert rgb.shape == (11, 3)
    assert tuple(rgb[0]) == contacts.CONTACT_RAMP[0][1]
    assert tuple(rgb[-1]) == contacts.CONTACT_RAMP[-1][1]
    assert (rgb[:, 2] >= rgb[:, 0]).all()                   # Blue, never red
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722])
    assert (np.diff(lum) < 0).all()                         # Darker as P grows
    assert tuple(contacts.contact_rgb(1.5)) == tuple(rgb[-1])   # Clipped


def test_picked_contact_is_not_red():
    r, g, b = contacts.PICK_RGB
    assert g > 100                                          # Gold, not red


def test_contact_mask():
    xyz = np.array([[0, 0, 0], [5, 0, 0], [20, 0, 0], [np.nan] * 3, [7.9, 0, 0]], np.float32)
    m = contacts.contact_mask(xyz, 8.0, chunk=2)
    assert m.dtype == bool and (m == m.T).all()
    assert m[0, 1] and m[0, 4] and m[1, 4]
    assert not m[0, 2] and not m[1, 2]
    assert not m[3].any() and not m[:, 3].any()             # No coordinates, no contacts
    assert m[0, 0] and not m[3, 3]


def test_plddt_palette_bands():
    pae = _pae_module()
    pal = pae.plddt_palette()
    assert pal == ('0,#FF7D45:50,#FF7D45:50.00001,#FFDB13:70,#FFDB13:70.00001,'
                   '#65CBF3:90,#65CBF3:90.00001,#0053D6:100,#0053D6')
    values = [float(stop.split(',')[0]) for stop in pal.split(':')]
    assert all(b > a for a, b in zip(values, values[1:]))   # Strictly increasing
    small = [float(stop.split(',')[0]) for stop in pae.plddt_palette(0.01).split(':')]
    assert all(b > a for a, b in zip(small, small[1:])) and small[-1] == 1.0
