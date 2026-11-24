# Developing AlphaLens

## Install from source

    git clone https://github.com/KosinskiLab/ChimeraX-AlphaLens.git

Then, in the ChimeraX command line, change to the checkout first:

    cd /path/to/AlphaLens
    devel install .        # install from the checkout
    devel build .          # or only build the wheel into dist/
    devel clean .          # remove build files

ChimeraX builds a `pyproject.toml` bundle from its current directory, whatever path is given to
`devel`: without the `cd`, `devel install` fails with "package directory 'src' does not exist"
(seen in ChimeraX 1.10).

## Tests

The file readers, agreement, CSV export and command loading behavior are tested without ChimeraX:

    pip install numpy pytest
    pytest tests

For a headless integration check with real ChimeraX structures and commands, run from the checkout:

    ChimeraX --nogui --exit --cmd "open tests/chimerax_smoke.py"

This creates temporary synthetic AF2 and AF3 predictions and checks agreement, compressed
distograms, CSV export, seed matching and loading errors without installing the bundle.

The plot window itself is checked in a real ChimeraX window (Qt's offscreen mode crashes it), on
predictions with distograms:

    ALPHALENS_WINDOW_DIRS="/path/to/af3_prediction;/path/to/af2_prediction" \
        ChimeraX --exit --cmd "open tests/chimerax_window_check.py"

It opens each prediction as a user would and checks every view, the Model contacts matrix, hover,
contacts on the structure, the clicked contact, pLDDT colors, CSV export, agreement and model
switching. It tests this checkout without installing it (`ALPHALENS_SOURCE=installed` tests the
installed bundle) and exits with status 1 if any check fails; the script header lists the options.
`tests/chimerax_gui_check.py` checks the menus, Help, zoom and the pLDDT slider's label, and
`tests/chimerax_model_switch_check.py` checks that choosing another model in the Model menu
updates everything that belongs to a model, `tests/chimerax_source_msa_check.py` re-predicts
a box from the source's own MSA in each layout AlphaLens reads it from, and
`tests/chimerax_greyed_check.py` checks, for any predictions given, that every control that can
be greyed out is off exactly when it should be and says why; their headers say what they need. The release
procedure for maintainers is `.claude/skills/alphalens-release/SKILL.md`.

The images in the README and the reference are rendered from the installed bundle by
`docs/render_readme_images.py`; its header says which predictions it needs and how to run it.

## Local prediction

How the LocalFold engine is pinned, built and checked, the dialog in detail, the job folder, the
measurements so far and the prediction checks: [Local prediction: notes for developers](local-prediction-notes.md).
