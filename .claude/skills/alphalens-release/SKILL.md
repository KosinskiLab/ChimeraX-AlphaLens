---
name: alphalens-release
description: AlphaLens releases and test builds. Use when bumping the version, building or installing a wheel, running the test suites (pytest, ChimeraX smoke test, window check), or putting a new wheel on the lab share or GitHub.
---

# Release AlphaLens

A release is a wheel that leaves this checkout: installed for a user, copied to the lab share,
or attached to a GitHub release. Collaborators tell builds apart only by version, so every such
wheel gets a version of its own; two different wheels under one version are the failure to avoid.

Run ChimeraX with `PYTHONDONTWRITEBYTECODE=1` throughout: `.pyc` files written into the app bundle
break its macOS signature. `CX=/Applications/ChimeraX-1.12.app/Contents/MacOS/ChimeraX` below.

## 1. Version

- `pyproject.toml` `version`: minor for new views or commands, patch for fixes.
- `CHANGELOG.md`: a `## X.Y.Z` section on top, written for users.
- `README.md`: the install line `chimerax_alphalens-X.Y.Z-py3-none-any.whl`.

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

   Use predictions with distograms, so every view runs; the script header lists the options.
   Lab pair with distograms in both backends (about 5 min over the SMB mount, mostly the AF2
   agreement): `<lab benchmark>/predictions/full_unfiltered_v3/human/{af3,af2}/pos_pairs/predictions/Q15014+Q9NV56`.
   A large case (5,167 residues, 3 GB distogram, about 10 min to load over SMB):
   `<a five-chain AF3 fold from a campaign>`.

## 3. Build and check the wheel

1. `cd` into the checkout first (the build needs it), then `$CX --nogui --exit --cmd "devel build . exit true"`.
2. `dist/chimerax_alphalens-X.Y.Z-py3-none-any.whl`: unzip it and `cmp` every `src/*.py` with
   `chimerax/alphalens/*.py`; `METADATA` says `Version: X.Y.Z`.
3. Install it: `$CX --nogui --exit --cmd "toolshed install $PWD/dist/<wheel> reinstall true; toolshed list"`.
   The list shows `AlphaLens (X.Y.Z)` and no `DistoPAE`: the bundle's old name registers the
   `distopae` command too, so remove it with `toolshed uninstall DistoPAE`.
4. Rerun the window check against the installed wheel: `ALPHALENS_SOURCE=installed`.

## 4. Publish

Each step runs on the user's request; ask when the request does not cover it.

- **Repository**: commit on a branch, push, open a pull request to `KosinskiLab/AlphaLens`.
- **GitHub release**: `gh release create vX.Y.Z dist/<wheel> -R KosinskiLab/AlphaLens --title "ChimeraX-AlphaLens X.Y.Z"`
  with the changelog section as notes (tags are `vX.Y.Z`).
- **Lab share**: copy the wheel to `<lab share>/` and `README.md` and `CHANGELOG.md`
  to `<lab share>/chimerax-alphalens/`, `cmp` each copy, then check on the cluster
  (`ssh embl ls -la ...`) that they are readable as the earlier wheels are (`-rwxr-xr-x`).
  Collaborators install from there.
