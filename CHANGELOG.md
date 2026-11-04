# Changelog

## 0.4.0

First public release.

* Agreement view: for every residue pair, the number of AlphaFold3 seeds or AlphaFold2 models in
  which P(d < 8 Å) reaches the slider threshold, with P per seed or model on hover and a
  per-chain-pair table (top-10 mean P per seed or model) in the log.
* Export CSV button and `saveContacts` option: the contacts above the threshold with residues, P,
  PAE in both directions, expected and model distances, and P in every seed or model.
* `mode agreement` for scripts, also without a window.
* Tests for the file readers, the agreement and the CSV export that run without ChimeraX.
* Package metadata points to this repository and carries no personal contact details.
* README rewritten: installation for users, how to get a distogram from each predictor, commands,
  recognised layouts, remote data, limitations.
* `scripts/fetch_pred.sh` copies the files the plot reads from any machine reachable over ssh.

## 0.1.0 – 0.3.1

Internal versions.
