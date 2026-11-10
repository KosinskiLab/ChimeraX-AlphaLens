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


def test_distance_matrix_matches_contact_mask():
    xyz = np.array([[0, 0, 0], [5, 0, 0], [20, 0, 0], [np.nan] * 3, [7.9, 0, 0]], np.float32)
    d = contacts.distance_matrix(xyz, chunk=2)
    assert d.dtype == np.float32 and d.shape == (5, 5)
    assert np.isclose(d[0, 2], 20) and np.isclose(d[1, 4], 2.9) and d[0, 0] == 0
    assert np.isnan(d[3]).all() and np.isnan(d[:, 3]).all()
    with np.errstate(invalid='ignore'):
        assert ((d < 8.0) == contacts.contact_mask(xyz, 8.0)).all()


def test_distance_view_values():
    far = contacts.DISTANCE_FAR
    d = np.array([[0.0, 5.0, 8.0], [24.0, far, 100.0], [np.nan, 7.99, 3.0]], np.float32)
    p = np.array([[1.0, 0.0, 0.9], [0.5, 0.5, 0.5], [0.5, 0.4, 0.7]], np.float32)
    v = contacts.distance_view_values(d, 8.0, p, floor=0.1)
    assert v.dtype == np.float32
    # Contacts (d < 8): their P, at least the floor, so none disappears
    assert v[0, 0] == 1.0 and np.isclose(v[0, 1], 0.1) and np.isclose(v[2, 1], 0.4)
    # Farther pairs: -1 at the cutoff (dark grey) to -2 at far and beyond (white)
    assert v[0, 2] == -1.0 and np.isclose(v[1, 0], -1.5) and v[1, 1] == -2.0 and v[1, 2] == -2.0
    assert v[2, 0] == -2.0                                  # No coordinates: white
    # Max-pooling keeps a contact over any distance, and the closest of the farther pairs
    assert (v[d < 8] >= 0).all() and (v[~(d < 8)] <= -1).all()
    assert contacts.distance_view_values(d, 8.0)[0, 1] == 1.0   # Without P: one color


def test_plddt_palette_bands():
    pae = _pae_module()
    pal = pae.plddt_palette()
    assert pal == ('0,#FF7D45:50,#FF7D45:50.00001,#FFDB13:70,#FFDB13:70.00001,'
                   '#65CBF3:90,#65CBF3:90.00001,#0053D6:100,#0053D6')
    values = [float(stop.split(',')[0]) for stop in pal.split(':')]
    assert all(b > a for a, b in zip(values, values[1:]))   # Strictly increasing
    small = [float(stop.split(',')[0]) for stop in pae.plddt_palette(0.01).split(':')]
    assert all(b > a for a, b in zip(small, small[1:])) and small[-1] == 1.0


def test_distance_and_contact_caches_follow_coordinates_without_gui_triggers():
    atoms = [types.SimpleNamespace(coord=np.array([0., 0., 0.]), deleted=False),
             types.SimpleNamespace(coord=np.array([5., 0., 0.]), deleted=False)]
    pae = types.SimpleNamespace(row_residues_or_atoms=lambda: atoms)
    original = contacts.model_distances(pae)
    assert contacts.model_distances(pae) is original
    assert contacts.model_contacts(pae, 8.)[0][0, 1]
    atoms[1].coord[0] = 20.
    assert not contacts.model_contacts(pae, 8.)[0][0, 1]
    assert contacts.model_distances(pae)[0][0, 1] == 20.
    atoms[1].deleted = True
    assert np.isnan(contacts.model_distances(pae)[0][0, 1])
    assert not contacts.model_contacts(pae, 8.)[0][0, 1]
