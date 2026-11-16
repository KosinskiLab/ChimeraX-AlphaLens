from alphalens.docking import share_dock_column, DOCK_KEEP_PX

# Measured in ChimeraX 1.12 (2316 x 1382 window): Log, Models, then the docked plots.
LOG, MODELS = (589, 23, 523), (97, 97, 223)          # (height, minimum, hint)


def share(rows, is_plot):
    h, m, s = zip(*rows)
    return share_dock_column(list(h), list(m), list(s), is_plot)


def test_two_plots_get_equal_room():
    # Qt left the first plot at its minimum (356) under the second (454): 54 px of matrix.
    rows = [(295, 23, 523), MODELS, (356, 487, 907), (454, 487, 907)]
    sizes = share(rows, [False, False, True, True])
    assert sum(sizes) == sum(r[0] for r in rows)
    assert sizes[2] == sizes[3] >= 487
    assert sizes[1] >= 97


def test_single_plot_gets_its_hint_and_the_rest_goes_back():
    rows = [LOG, MODELS, (517, 487, 907)]
    sizes = share(rows, [False, False, True])
    assert sizes[2] == 907
    assert sum(sizes) == sum(r[0] for r in rows)
    assert sizes[0] >= DOCK_KEEP_PX and sizes[1] >= 97


def test_other_tools_keep_their_minimum_and_small_heights():
    rows = [(60, 23, 523), (97, 97, 223), (800, 487, 907)]
    sizes = share(rows, [False, False, True])
    assert sizes == [60, 97, 800]                 # Smaller than DOCK_KEEP_PX: not shrunk further


def test_plots_that_do_not_fit_take_room_down_to_the_minimums():
    # Measured: two plots at their minimum (487) leave the Log 131 px, as Qt itself does.
    rows = [(295, 23, 523), MODELS, (356, 487, 907), (454, 487, 907)]
    assert share(rows, [False, False, True, True]) == [131, 97, 487, 487]


def test_plots_alone_share_everything():
    sizes = share([(300, 487, 907), (1500, 487, 907)], [True, True])
    assert sizes == [900, 900]


def test_no_plot_no_change():
    rows = [LOG, MODELS]
    assert share(rows, [False, False]) == [589, 97]
