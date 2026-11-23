---
name: alphalens-release
description: AlphaLens releases and test builds. Use when bumping the version, building or installing a wheel, running the test suites (pytest, ChimeraX smoke test, window check), or putting a new wheel on the lab share, GitHub or the ChimeraX Toolshed.
---

# Release AlphaLens

A release is a wheel that leaves this checkout: installed for a user, copied to the lab share,
attached to a GitHub release or uploaded to the ChimeraX Toolshed. Collaborators tell builds
apart only by version, so every such wheel gets a version of its own; two different wheels under
one version are the failure to avoid.

Run ChimeraX with `PYTHONDONTWRITEBYTECODE=1` throughout: `.pyc` files written into the app bundle
break its macOS signature. `CX=/Applications/ChimeraX-1.12.app/Contents/MacOS/ChimeraX` below.

## 1. Version

- `pyproject.toml` `version`: minor for new views or commands, patch for fixes.
- `CHANGELOG.md`: a `## X.Y.Z` section on top, written for users.
- `README.md`: the install line `chimerax_alphalens-X.Y.Z-py3-none-any.whl`.
- `pyproject.toml` `description` and `[project.readme]` text: the Toolshed page shows them. Bring
  them in line when a view, a command, or what local prediction downloads or sends changes.
- Help pages: after any change to `README.md`, `docs/reference.md` or `docs/local-prediction.md`,
  run `scripts/build_help.py` with a Python that has `markdown` (ChimeraX's:
  `/Applications/ChimeraX-1.12.app/Contents/bin/python3.11 -I scripts/build_help.py`) and commit
  `src/docs/user/tools/*.html` with them. `tests/test_help_pages.py` fails on a stale page.

## 2. Test the checkout

All three must pass before building:

1. `pytest tests` (a Python with numpy and pytest; no ChimeraX).
2. `$CX --nogui --exit --cmd "open tests/chimerax_smoke.py"`: the log ends with
   `AlphaLens ChimeraX smoke checks PASSED`.
3. The window check, in a real window (Qt's offscreen mode crashes ChimeraX); exit status 0:

   ```bash
   ALPHALENS_WINDOW_DIRS="<af3 prediction>;<af2 prediction>" \
       $CX --exit --cmd "open tests/chimerax_window_check.py"
   ```

   Use predictions with distograms, so every view runs, plus one AF3 prediction with two or more
   seeds and no distogram, so that Agreement from `contact_probs` and a greyed-out Spread are
   checked too (a local copy of such a prediction); the script header lists the options.
   When menus, Help or zoom changed, also run `tests/chimerax_gui_check.py` with two predictions
   (`ALPHALENS_GUI_DIRS="<prediction>;<another>"`; the AF2 and AF3 folders of one complex are
   the hard case); with `ALPHALENS_SOURCE=installed` it also opens the installed help pages.
   When model switching or what the window draws on the structure changed (contact lines,
   clicked contact, dragged box, hiding, coloring), also run `tests/chimerax_model_switch_check.py`
   with an AF2 prediction whose pickles kept the distogram and an AF3 one with samples
   (`ALPHALENS_SWITCH_DIRS="<af2>;<af3>"`; about 3 min on local copies of the lab pair below).
   Lab pair with distograms in both backends (about 5 min over the SMB mount, mostly the AF2
   agreement): `<lab benchmark>/predictions/full_unfiltered_v3/human/{af3,af2}/pos_pairs/predictions/Q15014+Q9NV56`.
   A large case (5,167 residues, 3 GB distogram, about 10 min to load over SMB):
   `<a five-chain AF3 fold from a campaign>`.

When the prediction dialog, `predict*.py` or the plot's box selection changed, also run the
mapping check (about 15 s, no GPU or network; exit status 0):

```bash
ALPHALENS_MAPPING_SOURCE="<prediction with contact probabilities>" ALPHALENS_TEST_OUTPUT="<new folder>" \
    $CX --cmd "open tests/chimerax_prediction_mapping_check.py"
```

and the gapped-chain check (about 30 s, real AF2 and AF3 runs; needs engines at the pinned
LocalFold revision or newer, set `ALPHALENS_LOCALFOLD`/`ALPHALENS_LOCALFOLD_AF3` to test engines
without rebuilding the managed ones; exit status 0):

```bash
ALPHALENS_PILOT="<barnase-barstar LocalFold folder>" ALPHALENS_TEST_OUTPUT="<new folder>" \
    $CX --cmd "open tests/chimerax_gapped_prediction_check.py"
```

`tests/test_alphapulldown_parity.py` mirrors AlphaPulldown's region and numbering tests; when
AlphaPulldown changes how it handles regions, update it there first.

## 3. Build and check the wheel

1. `cd` into the checkout first (the build needs it), then `$CX --nogui --exit --cmd "devel build . exit true"`.
2. `dist/chimerax_alphalens-X.Y.Z-py3-none-any.whl`: unzip it and `cmp` every `src/*.py` with
   `chimerax/alphalens/*.py`; `METADATA` says `Version: X.Y.Z`; it holds
   `chimerax/alphalens/docs/user/tools/*.html` and `alphalens_images/*.png`.
3. Install it: `$CX --nogui --exit --cmd "toolshed install $PWD/dist/<wheel> reinstall true; toolshed list"`.
   The list shows `AlphaLens (X.Y.Z)`.
4. Rerun the window check against the installed wheel: `ALPHALENS_SOURCE=installed`.

## 4. Publish

When including `setup_localfold.py` in a release, copy `src/predict_setup.py` under
that asset name. It is self-contained; `scripts/setup_localfold.py` is a repository
entry point and requires the checkout.

Each step runs on the user's request; ask when the request does not cover it.

- **Repository**: commit on a branch, push, open a pull request to `KosinskiLab/ChimeraX-AlphaLens`.
- **GitHub release**: `gh release create vX.Y.Z dist/<wheel> -R KosinskiLab/ChimeraX-AlphaLens --title "ChimeraX-AlphaLens X.Y.Z"`
  with the changelog section as notes (tags are `vX.Y.Z`).
- **Lab share**: copy the wheel to `<lab share>/` and `README.md` and `CHANGELOG.md`
  to `<lab share>/chimerax-alphalens/` (and `docs/*.md` plus `docs/images/` to its
  `docs/`, which the README links to), `cmp` each copy, then check on the cluster
  (`ssh embl ls -la ...`) that they are readable as the earlier wheels are (`-rwxr-xr-x`).
  Collaborators install from there.
- **ChimeraX Toolshed**: the user uploads the wheel at https://cxtoolshed.rbvi.ucsf.edu/submit_app/
  (Sign in with Google, their account); give them that URL and the wheel's path. Once AlphaLens is
  listed there, ChimeraX offers each uploaded version to its users as an update, so every release
  goes there too.
