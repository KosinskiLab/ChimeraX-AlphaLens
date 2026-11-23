'''
AlphaLens help pages: docs/user/tools/*.html in the installed bundle, generated from the
repository's Markdown by scripts/build_help.py.
'''
import os

PLOT = 'alphalens.html'
REFERENCE = 'alphalens-reference.html'
PREDICTION = 'alphalens-prediction.html'

ONLINE = 'https://github.com/KosinskiLab/ChimeraX-AlphaLens/blob/main/'
SOURCES = {PLOT: 'README.md', REFERENCE: 'docs/reference.md', PREDICTION: 'docs/local-prediction.md'}


def help_url(page, fragment=''):
    '''
    "help:user/tools/<page>" when ChimeraX finds the page among the installed bundles' docs,
    else the page's Markdown on GitHub (a checkout run without installing, for example). Both
    forms work with the "help" command and in a tool's context menu.
    '''
    frag = f'#{fragment}' if fragment else ''
    rel = f'user/tools/{page}'
    try:
        from chimerax.core import toolshed
        if any(os.path.exists(os.path.join(d, rel)) for d in toolshed.get_help_directories()):
            return f'help:{rel}{frag}'
    except Exception:
        pass
    return ONLINE + SOURCES[page] + frag


def show_help(session, page, fragment=''):
    '''Show a help page in ChimeraX's help viewer.'''
    from chimerax.core.commands import run
    run(session, f'help {help_url(page, fragment)}', log=False)
