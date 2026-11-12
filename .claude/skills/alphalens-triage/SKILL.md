---
name: alphalens-triage
description: AlphaLens problem reports from users. Use when someone says a view is greyed out or empty (spread, agreement, contacts), a command fails, loading is slow, or a prediction folder does not open.
---

# Triage an AlphaLens report

Work through the checks in order; the first that explains the report usually is the answer.
Reply with the cause, what the user can do now, and whether anything needs fixing in AlphaLens.

## 1. Which bundle is running

Ask for, or run, `toolshed list` in their ChimeraX:

- `AlphaLens (X.Y.Z)`: compare with the newest wheel on the lab share
  (`/g/kosinski/dima/software/chimerax_alphalens-*.whl`) and with `CHANGELOG.md`.
- `DistoPAE` (the name up to 0.4.0), alone or next to AlphaLens: both register `distopae`, so the
  command may reach the old code. `toolshed uninstall DistoPAE`, then install the current wheel.
- `alphalens <folder>` without `open` works from 0.5.0 on; older bundles need `alphalens open` /
  `distopae open`.

## 2. Does the prediction contain the data the view needs

A greyed-out tab's tooltip names what is missing. The data behind each view:

| View | Needs |
|---|---|
| PAE, Model distances | the model and its PAE file |
| Contact probability | a distogram, or AF3 `contact_probs` (in every AF3 confidences file) |
| Spread | a distogram of the shown model: AF3 `seed-<S>_distogram/*.npz[.gz/.xz]` for that model's own seed (another seed's never substitutes), or an AF2 result pickle with a `distogram` key |
| Agreement | contact probabilities of at least two AF3 seeds, or two AF2 models with distograms, present locally |

Check the folder directly: `ls <fold>/seed-*_distogram`; for a pickle,
`xz -dc result_model_1*.pkl.xz | grep -a -c distogram` (0 means none). AF3 writes distograms only
with `--save_distogram`; AlphaPulldown drops them from AF2 pickles unless told to keep them
(README, "Getting a distogram").

Lab predictions, as of 2026-11-07:
- benchmark_26: only `predictions/full_unfiltered_v3/<organism>/{af2,af3}/` kept distograms; the
  per-organism copies such as `predictions/human/af3/pos_pairs/` did not. AF3 runs there have one
  seed, so agreement is AF2-only.
- Aruna (TRIP4-ASCC) report candidates: re-predicted AF3 distograms are linked into
  `XFoldRuns/aruna/output_alphafold3/predictions/<fold>/` for 167 of 180 seeds. A sample from an
  unlinked seed shows Spread greyed; another sample in the Model menu may have it.

## 3. Slow loading and odd folder names

- A distogram is N x N x 64 values: about 3 GB at 5,000 residues, which takes about 10 min
  through the `/g` SMB mount from a home connection and peaks near 5 GB of RAM. The plot shows PAE
  at once and fills in the other views when the read finishes; agreement reads every seed or
  model. For repeated work, copy the fold with `scripts/fetch_pred.sh`.
- Folds of residue ranges are named with `:` (`...+Q5T481:385-460`); through SMB on a Mac they show
  as an 8.3 name such as `Q15650~1`. Open that name, or a colon-free symlink made on the cluster.

## 4. Reproduce

Gather the AlphaLens and ChimeraX versions, the exact command, the Log text and a listing of
the prediction folder. Reproduce with the same command on the same folder, then with
`tests/chimerax_window_check.py` on it (`ALPHALENS_WINDOW_DIRS=<folder>`). A failure there that
the data explain is a data problem; any other failure is a bug for the next release
(`alphalens-release`).
