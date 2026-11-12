import numpy as np

from alphalens.hide import RowHider, true_runs


class Items:
    '''Stands in for ChimeraX collections: reading an attribute gives a copy of the array.'''

    def __init__(self, ribbons=None, displays=None, selected=None):
        self._ribbons, self._displays, self._selected = ribbons, displays, selected

    ribbon_displays = property(lambda self: self._ribbons.copy(),
                               lambda self, v: setattr(self, '_ribbons', v.copy()))
    displays = property(lambda self: self._displays.copy(),
                        lambda self, v: setattr(self, '_displays', v.copy()))
    selected = property(lambda self: self._selected.copy())


def hider():
    '''Four residues with two atoms each. Side-chain atoms are off, residue 3 has no ribbon.'''
    residues = Items(ribbons=np.array([True, True, True, False]))
    atoms = Items(displays=np.array([True, False] * 4), selected=np.zeros(8, bool))
    return RowHider(4, np.arange(4), residues, np.repeat(np.arange(4), 2), atoms), residues, atoms


def test_state_round_trip_shows_what_an_earlier_hider_hid():
    first, residues, atoms = hider()
    first.apply(plddt=np.array([True, True, False, False]))
    assert first.state() == {'ribbon_off': [0, 1], 'atom_off': [0, 2]}
    restored = RowHider(4, np.arange(4), residues, np.repeat(np.arange(4), 2), atoms)
    restored.set_state(first.state())
    restored.restore()
    assert residues.ribbon_displays.tolist() == [True, True, True, False]
    assert atoms.displays.tolist() == [True, False] * 4


def test_set_state_ignores_indices_outside_the_model():
    h, _, _ = hider()
    h.set_state({'ribbon_off': [1, 99, -1], 'atom_off': [8]})
    assert h.state() == {'ribbon_off': [1], 'atom_off': []}


def mask(*rows):
    m = np.zeros(4, bool)
    m[list(rows)] = True
    return m


def test_true_runs():
    assert true_runs(np.array([1, 1, 0, 1, 0, 0, 1, 1, 1], bool)) == [(0, 2), (3, 4), (6, 9)]
    assert true_runs(np.zeros(3, bool)) == []


def test_hide_and_restore_exactly_what_was_shown():
    h, residues, atoms = hider()
    before_r, before_a = residues.ribbon_displays, atoms.displays
    h.apply(plddt=mask(0, 3))
    np.testing.assert_array_equal(residues.ribbon_displays, [False, True, True, False])
    np.testing.assert_array_equal(atoms.displays[:2], [False, False])
    h.apply(plddt=mask())
    np.testing.assert_array_equal(residues.ribbon_displays, before_r)
    np.testing.assert_array_equal(atoms.displays, before_a)


def test_row_stays_hidden_while_any_reason_covers_it():
    h, residues, _ = hider()
    h.apply(plddt=mask(0), selected=mask(0, 1))
    h.apply(plddt=mask())
    assert not residues.ribbon_displays[0]
    assert h.hidden_rows.tolist() == [True, True, False, False]
    h.apply(selected=mask())
    assert residues.ribbon_displays[:2].all()


def test_apply_hides_again_what_was_shown_since():
    h, residues, atoms = hider()
    h.apply(plddt=mask(0))
    residues._ribbons[0] = True
    atoms._displays[0] = True
    h.apply()
    assert not residues.ribbon_displays[0]
    assert not atoms.displays[0]
    h.restore()
    assert residues.ribbon_displays[0]
    assert atoms.displays[0]


def test_selected_rows_are_those_with_a_selected_atom():
    h, _, atoms = hider()
    atoms._selected[[1, 4]] = True
    assert h.selected_rows().tolist() == [True, False, True, False]
