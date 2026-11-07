# Changelog

## 0.6.0

* **Model distances** view replaces Model contacts: the CB–CB distances in the shown model in grey
  (8 Å dark to 40 Å and beyond white), with the pairs closer than 8 Å colored by the predicted
  contact probability in the same matrix. `mode distance`; `mode model` still works.

## 0.5.1

* Never use another AF3 seed's distogram when the requested seed's file is missing; an
  unknown seed is resolved only when the available distograms belong to one seed.
* Read and discover `.npz.gz` and `.npz.xz` distograms, including for agreement, and copy
  them with `scripts/fetch_pred.sh`.
* `wait true` finishes background distogram and agreement reads before returning, without
  applying their results or running their callbacks twice.
* Explicit data-dependent commands report loading failures. A failed agreement request no
  longer exports a CSV with only the shown model's contacts. Optional distogram failures
  still allow PAE viewing and the AF3 contact-probability fallback.
* Update repository links and document contact display limits and the Color PAE Domains button.

## 0.5.0

* **Renamed from DistoPAE to AlphaLens**, as the plot now does more than show the distogram and
  PAE: bundle ChimeraX-AlphaLens, command `alphalens`, tool Tools > Structure Prediction >
  AlphaLens, repository KosinskiLab/AlphaLens. `distopae` and `distopae open` still
  work. Remove the old bundle with `toolshed uninstall DistoPAE`.
* Predicted contacts are blue instead of red (blue reads as confident; red is kept for clashes):
  the dots on the plot and the lines on the structure, darker and thicker for higher P, with the
  scale next to the color bar. Weak contact lines are semi-transparent. The clicked contact is a
  gold line, matching the gold ring around its dot.
* **Model contacts** view: the residue pairs closer than 8 Å in the shown model (CB–CB, CA for
  glycine, as in the distogram), colored by the predicted contact probability; `mode model`.
  Hovering gives the CB–CB distance in the model in every view.
* The contact view is labeled "Contact probability", the slider "Predicted contacts", and the tabs
  explain their content in tooltips.
* Show Contacts always draws all predicted contacts above the slider threshold, also after a box
  was dragged, up to 10,000 lines (was 2,000).
* Color pLDDT uses the banded AlphaFold pLDDT colors (< 50, 50–70, 70–90, > 90) instead of
  ChimeraX's smooth "alphafold" palette.
* `alphalens <folder or model file>` works without `open`.

## 0.4.0

First public release.

* Agreement view: for every residue pair, the number of AlphaFold3 seeds or AlphaFold2 models in
  which P(d < 8 Å) reaches the slider threshold, with P per seed or model on hover and a
  per-chain-pair table (top-10 mean P per seed or model) in the log.
* Export CSV button and `saveContacts` option: the contacts above the threshold with residues, P,
  PAE in both directions, expected and model distances, and P in every seed or model.
* `mode agreement` for scripts, also without a window.
* Messages say why the agreement view is unavailable: seed folders not copied, or result
  pickles without a distogram.
* Spread colorbar ticks at round values.
* README images of the plot; install from a checkout with `cd` first (ChimeraX 1.10);
  `scripts/fetch_pred.sh` works with macOS rsync.
* Tests for the file readers, the agreement and the CSV export that run without ChimeraX.
* Package metadata points to this repository and carries no personal contact details.
* README rewritten: installation for users, how to get a distogram from each predictor, commands,
  recognised layouts, remote data, limitations.
* `scripts/fetch_pred.sh` copies the files the plot reads from any machine reachable over ssh.

## 0.1.0 – 0.3.1

Internal versions.
