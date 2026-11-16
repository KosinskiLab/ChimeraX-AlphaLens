'''
Room for a plot docked into the ChimeraX main window.

Qt's dock layout gives a newly docked window whatever the tools above it leave over and
shrinks a crowded column to each window's minimum. The plot's controls have fixed heights,
so docking a second plot under the first used to squeeze the first matrix to a few pixels.
When a plot is docked, the plots in its dock column share the column equally (up to the
height they ask for), and the other tools keep at least DOCK_KEEP_PX or their own minimum.
'''

DOCK_KEEP_PX = 160      # Other tools in the column (Log, Models, ...) keep at least this much


def share_dock_column(heights, minimums, hints, is_plot, keep=DOCK_KEEP_PX):
    '''
    New heights for the windows of one dock column, top to bottom, with the same total.
    heights: current heights; minimums / hints: each window's minimum and preferred height;
    is_plot: which windows are AlphaLens plots.
    '''
    n = len(heights)
    plots = [i for i in range(n) if is_plot[i]]
    others = [i for i in range(n) if not is_plot[i]]
    if not plots:
        return list(heights)
    total = sum(heights)
    reserve = {i: max(minimums[i], min(heights[i], keep)) for i in others}
    room = total - sum(reserve.values())
    share = max(room // len(plots), 0)
    sizes = list(heights)
    for i in plots:
        sizes[i] = max(minimums[i], share if not others else min(hints[i], share))
    spare = total - sum(sizes[i] for i in plots) - sum(reserve.values())
    # Plots at their minimum that still do not fit: the other tools give up room down to their own minimum.
    for i in sorted(others, key=lambda i: minimums[i] - reserve[i]):
        if spare >= 0:
            break
        take = min(-spare, reserve[i] - minimums[i])
        reserve[i] -= take
        spare += take
    # Room the plots do not need goes back to the other tools, in proportion to their heights.
    spare = max(spare, 0)
    weight = sum(heights[i] for i in others)
    given = 0
    for k, i in enumerate(others):
        if k == len(others) - 1:
            extra = spare - given
        else:
            extra = spare * heights[i] // weight if weight else spare // len(others)
        given += extra
        sizes[i] = reserve[i] + extra
    return sizes


def redock_if_lost(main_window, dock_widget):
    '''
    Qt can lose a docked window's place in the layout: with two plots docked, floated and
    docked again, the second came back "docked" but loose over the graphics, outside every dock
    column (ChimeraX 1.12 on macOS changes the window flags on floating). Dock it again into
    its area. Returns True when it did.
    '''
    central = main_window.centralWidget()
    if (dock_widget.isFloating() or not dock_widget.isVisible() or central is None
            or not dock_widget.geometry().intersects(central.geometry())):
        return False
    area = main_window.dockWidgetArea(dock_widget)
    from Qt.QtCore import Qt
    if area == Qt.DockWidgetArea.NoDockWidgetArea:
        area = Qt.DockWidgetArea.RightDockWidgetArea
    main_window.removeDockWidget(dock_widget)
    main_window.addDockWidget(area, dock_widget)
    dock_widget.show()
    return True


def fit_docked_plot(main_window, dock_widget, plot_docks):
    '''
    Share the dock column of dock_widget (just docked) among the plots in it, see
    share_dock_column(). plot_docks: the QDockWidgets of all AlphaLens plots.
    Returns the new sizes, or None when there was nothing to do.
    '''
    from Qt.QtCore import Qt
    from Qt.QtWidgets import QDockWidget
    if dock_widget.isFloating() or not dock_widget.isVisible():
        return None
    area = main_window.dockWidgetArea(dock_widget)
    if area not in (Qt.DockWidgetArea.LeftDockWidgetArea, Qt.DockWidgetArea.RightDockWidgetArea):
        return None
    x = dock_widget.geometry().left()
    column = sorted((d for d in main_window.findChildren(QDockWidget)
                     if d.isVisible() and not d.isFloating()
                     and main_window.dockWidgetArea(d) == area and d.geometry().left() == x),
                    key=lambda d: d.geometry().top())
    is_plot = [d in plot_docks for d in column]
    sizes = share_dock_column([d.height() for d in column],
                              [d.minimumSizeHint().height() for d in column],
                              [d.sizeHint().height() for d in column], is_plot)
    if sizes == [d.height() for d in column]:
        return None
    main_window.resizeDocks(column, sizes, Qt.Orientation.Vertical)
    return sizes
