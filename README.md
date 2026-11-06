# ChimeraX-AlphaLens

*Formerly DistoPAE (up to version 0.4). The `distopae` command still works.*

A [ChimeraX](https://www.cgl.ucsf.edu/chimerax/) plot for AlphaFold2 and AlphaFold3 predictions. Like
ChimeraX's built-in AlphaFold error plot, it shows the PAE matrix with a pLDDT track. It adds views
computed from the predicted distogram, compares the predicted contacts with the contacts in the
model, shows which contacts are reproduced across seeds or models, draws the predicted contacts on
the structure and exports them to CSV.

<p align="center"><img src="docs/images/contacts.png" width="640"
  alt="AlphaLens plot in the contact-probability view, with dots on the inter-chain contacts"></p>

*The contact-probability view for an AlphaFold2 prediction of E. coli succinyl-CoA synthetase,
SucC–SucD (UniProt P0A836, P0AGE9). Dots mark the inter-chain pairs with P ≥ 0.5 (red in this
image from version 0.4; blue from 0.5 on). The ringed one was clicked; its values are in the line
under the slider.*

| View | Matrix | Colors (dark = confident) |
|---|---|---|
| PAE | predicted aligned error (Å) | 0–30 Å |
| Contact probability | predicted P(d < 8 Å) from the distogram, or AF3 `contact_probs` if there is no distogram | √P scale, so weak contacts show |
| Model contacts | residue pairs closer than 8 Å in the shown model, colored by the predicted P; white where the model has no contact | as the contact-probability view |
| Spread σ / H | standard deviation (Å) or entropy (bits) of the predicted distance distribution | 0 to the 99th percentile of the model |
| Agreement | number of AF3 seeds or AF2 models in which the pair has P ≥ the slider threshold | 0 to the number of seeds or models |

<p align="center"><img src="docs/images/views.png" width="760"
  alt="The PAE, contact, spread and agreement views of one prediction"></p>

*The four views of the same prediction. In the spread view, grey marks pairs whose distance is
mostly beyond the last distogram bin. The agreement view counts the five AlphaFold2 models in which
each pair has P ≥ 0.5.*

## Install

Requires ChimeraX 1.6 or newer; tested with 1.10 and 1.12. The bundle is pure Python, so one wheel works on
macOS, Linux and Windows.

1. Download `chimerax_alphalens-<version>-py3-none-any.whl` from the
   [Releases](https://github.com/KosinskiLab/chimerax-alphalens/releases) page. Keep the file name:
   pip needs it.
2. In the ChimeraX command line:

       toolshed install ~/Downloads/chimerax_alphalens-0.5.0-py3-none-any.whl

3. Restart ChimeraX.

`toolshed list` shows the installed version (look for AlphaLens). To update, install the newer wheel
in the same way. To remove the bundle, run `toolshed uninstall AlphaLens`.

**Upgrading from DistoPAE:** install AlphaLens as above, then remove the old bundle with
`toolshed uninstall DistoPAE` and restart ChimeraX. Scripts that call `distopae` keep working.

### From source

    git clone https://github.com/KosinskiLab/chimerax-alphalens.git

Then, in the ChimeraX command line, change to the checkout first:

    cd /path/to/chimerax-alphalens
    devel install .        # install from the checkout
    devel build .          # or only build the wheel into dist/
    devel clean .          # remove build files

ChimeraX builds a `pyproject.toml` bundle from its current directory, whatever path is given to
`devel`: without the `cd`, `devel install` fails with "package directory 'src' does not exist"
(seen in ChimeraX 1.10).

## Quick start

    alphalens open /path/to/prediction_dir            # top-ranked model
    alphalens open /path/to/prediction_dir rank 2     # AF2 ranked_2 / AF3 third-best sample
    alphalens open /path/to/seed-1_sample-0/model.cif # one AF3 sample

`open` can be left out: `alphalens /path/to/prediction_dir` does the same.

For a structure that is already open, use Tools > Structure Prediction > AlphaLens (when exactly
one structure is open) or `alphalens #1`.

The plot reads the PAE file next to the structure, plus the distogram if the prediction kept it.
If the distogram is missing, you get the PAE view and, for AF3, the contact view from
`contact_probs`. The tabs that need a distogram are greyed out with a tooltip saying why. Only a
missing PAE file is an error.

## Getting a distogram

The contact view needs either a distogram or AF3 `contact_probs`; the spread view needs a
distogram. AF3 always writes `contact_probs` to its confidences file, but most pipelines do not
keep the distogram by default:

| Predictor | Setting | Where the distogram ends up |
|---|---|---|
| AlphaFold2 (`run_alphafold.py`) | kept by default | `result_<model>.pkl`, key `distogram` |
| AlphaPulldown, AlphaFold2 backend | `--noremove_keys_from_pickles`, and `--storage_mode vanilla` (the default) if your version has the option | `result_<model>.pkl[.gz\|.xz]` |
| AlphaFold3 v3.0.2 or newer | `--save_distogram` | `seed-<S>_distogram/<name>_seed-<S>_distogram.npz` |
| AlphaPulldown 2.1.4 or newer, AlphaFold3 backend | `--save_distogram` | as for AlphaFold3 |
| ColabFold | `--save-all` (untested) | `*_all_rank_NNN_*.pickle` |

A distogram holds N × N × 64 values, so it is large: a 2,000-residue complex gives about 0.5 GB
(AF3, float16) or 1 GB (AF2, float32) before compression. The plot opens at once with PAE and loads
the distogram in the background.

## Using the plot

* **Views:** arrows, tabs or the ← / → keys. Hovering shows the values for a residue pair.
* **Zoom:** scroll wheel, trackpad pinch, or + / −. **Pan:** Shift-drag or middle-drag. **Fit**
  (or 0 / Esc) resets the view. Matrices larger than 900 residues are pooled for display
  (contacts max-pooled, so no contact disappears; PAE and spread mean-pooled). When you zoom in,
  the visible region is redrawn at full resolution, and hovering always reports the exact
  per-residue values.
* **Model:** the dropdown lists the AF2 ranked models or AF3 samples of the same prediction that
  are present locally. The chosen model opens in the same window, superposed on the previous one,
  which is hidden; the view and zoom are kept.
* **Contact dots:** in the contact-probability view, inter-chain pairs above the slider threshold
  (long-range pairs for a single chain) are drawn as blue dots that stay visible at any zoom,
  darker blue for higher P (the scale is next to the color bar). Right-click to turn them off or
  show them in every view.
* **Predicted contacts P ≥ slider:** one threshold (0.05–1, default 0.2) for the dots, the 3D
  lines, the agreement view and the CSV export. "hide in matrix too" also blanks matrix pixels
  below it.
* **Show Contacts:** draws all predicted contacts above the slider threshold on the structure as
  CA–CA lines, in the same blue as the dots, thicker for higher P; weak contacts are drawn
  semi-transparent, so the confident interfaces stand out. Move the slider down to 0.05 to see
  every contact. Up to 10,000 lines are drawn, strongest first.
* **Click a dot:** draws that contact on the structure as a thick gold line (the plot rings the
  dot in the same gold), labels both residues and shows their side chains. The log gives P, both
  PAE values, the expected distance (with a distogram) and the CA–CA distance in the model.
  Clicking another dot replaces it. Right-click > Clear clicked contact removes it; with
  right-click > Zoom 3D view to clicked contact switched on, the camera also moves there.
* **Drag a box:** only the contacts inside it, above the slider threshold, are drawn on the
  structure, for example the block between two chains; a click on empty plot space or Show
  Contacts restores all. The log lists the strongest pairs. By default, dragging a box also colors
  those residues (an off-diagonal box uses two colors); this can be switched off in the
  right-click menu.
* **Model contacts:** the residue pairs whose CB atoms (CA for glycine, the atoms the distogram
  describes) are closer than 8 Å in the shown model, colored by the predicted contact probability
  on the same scale as the contact-probability view. Model contacts with P near 0 get the lightest
  green, so none disappears. Flip between the two views with ← / → at the same zoom: a contact
  that is predicted but white here is not formed in this model, and a pale contact here is in the
  model without support from the prediction. Hovering gives the CB–CB distance in the model in
  every view. Without contact probabilities (AF2 without a distogram) the model contacts are grey.
* **Color pLDDT** colors the structure in the four AlphaFold pLDDT bands (< 50, 50–70, 70–90,
  > 90), as the pLDDT track of the plot, instead of ChimeraX's smooth "alphafold" palette.
* Large models that ChimeraX shows as spheres are switched to cartoon the first time contact lines
  are drawn, because the lines would otherwise be hidden inside the atoms.
* **Spread:** the last distogram bin collects every distance above 21.7 Å, so the spread is
  undefined where that bin dominates. Pairs with P(d > 21.7 Å) > 0.5 are drawn grey; the threshold
  is in the right-click menu.
* **Right-click menu:** contact color scale (√P, linear, or bands at 0.05 / 0.2 / 0.5 / 0.8),
  spread as standard deviation or entropy, color the plot from the structure, chain divider
  lines, pLDDT track, Log agreement table, and Save image.

With more than one chain, the log gets an interface table per chain pair: maximum P, number of
residue pairs with P > 0.5, and minimum PAE.

The data are also attached as `structure.alphafold_pae`, so ChimeraX's own commands, such as
`alphafold pae #1 colorDomains true` and `alphafold contacts`, work on the same model.

## Agreement between seeds and models

A contact predicted by one seed but not by the others is weak evidence. The **Agreement** view
counts, for every residue pair, the independent runs of the prediction in which P(d < 8 Å)
reaches the slider threshold:

* **AlphaFold3:** one run per seed. The samples of a seed share one distogram and the same
  `contact_probs`, so they are not independent. The seed's distogram is used if it was saved,
  otherwise the `contact_probs` of one of its samples.
* **AlphaFold2:** one run per model (`model_1` … `model_5`, and every prediction per model). The
  result pickles must have kept the distogram (see [Getting a distogram](#getting-a-distogram)).

The seeds' or models' files must be present, not only the top model's: for AF3, the
`seed-*_sample-*` folders (or `seed-*_distogram`). `scripts/fetch_pred.sh` copies them. If they are
missing, the tab stays grey and its tooltip says how many seeds have files.

<p align="center"><img src="docs/images/agreement.png" width="640"
  alt="Agreement view in which the inter-chain contacts appear in only one of five models"></p>

*E. coli AccB–AccC (UniProt P0ABD8, P24182): no inter-chain pair reaches P ≥ 0.5 in all five
AlphaFold2 models. The hovered pair, in the line under the slider, has P 0.94 in one model and at
most 0.15 in the other four.*

The runs are read the first time the view is shown, in the background; five AF3 seeds of a
1,200-residue complex take a few seconds. Hovering shows the count and P in each run, for example
`2 of 5 seeds (P 0.91 0.12 0.88 0.05 0.07)`. Moving the slider recounts.

With more than one chain, the log gets a table per chain pair: the mean of the 10 highest
inter-chain P in each seed or model, and the number of seeds or models with a residue pair at
P ≥ the threshold. Right-click > Log agreement table logs it again for the current threshold.
The agreement data are kept when you switch to another model of the same prediction.

## Export to CSV

**Export CSV** saves the contacts with P ≥ the slider threshold: inter-chain pairs (long-range
pairs for a single chain), only those inside the dragged box if there is one. With the agreement
loaded, a pair is included if it reaches the threshold in any seed or model. One row per residue
pair, most reproduced and most probable first:

| Column | Content |
|---|---|
| `chain_1`, `residue_1`, `name_1`, `atom_1` (and `_2`) | the two residues; `atom` only for ligand atoms in AF3 |
| `p_contact` | P(d < 8 Å) in the shown model |
| `pae_1_2`, `pae_2_1` | PAE in both directions (Å) |
| `expected_distance` | mean of the predicted distance distribution (Å; needs a distogram; capped by the last bin at 21.7 Å) |
| `model_distance` | distance in the shown model between CA atoms (C4′ for nucleotides, the atom itself for ligands) (Å) |
| `seeds_with_p_ge_0.20`, `seeds` | with the agreement loaded: in how many of how many seeds (or `models`) the pair reaches the threshold |
| `p_seed_<S>` / `p_model_<N>` | P in each seed or model |

## Commands

    alphalens [#model] [paeFile file] [distogram file]
             [mode pae|contact|model|std|entropy|agreement]
             [cutoff 8] [contacts true|false] [minProb 0.5] [saveContacts file.csv]
             [palette name] [range low,high] [dividerLines true|false] [wait true|false]
    alphalens open path [rank N] [same options except paeFile]
    alphalens path [rank N] [...]          # same as "alphalens open path"

| Option | Meaning |
|---|---|
| `paeFile`, `distogram` | files with non-standard names; otherwise they are found next to the structure |
| `mode` | view to show: `contact` the predicted contact probability, `model` the contacts in the model, `std` / `entropy` the spread; `agreement` reads every seed or model first and logs the agreement table |
| `cutoff` | contact distance in Å (default 8), also for the model contacts; needs a distogram, because AF3 `contact_probs` are fixed at 8 Å |
| `contacts`, `minProb` | draw or remove the contacts with P ≥ minProb on the structure (default 0.5; also moves the slider) |
| `saveContacts` | write the contacts to a CSV file as Export CSV does, with P ≥ minProb if given, otherwise the slider threshold (0.2 without a plot) |
| `palette`, `range` | PAE color map, as in `alphafold pae` |
| `dividerLines` | chain divider lines on the plot |
| `wait` | load the distogram (and the agreement data) before the command returns, for scripts |
| `rank` | 0 = top model; AF2 `ranked_N`, AF3 N-th best sample |

For example, without a window (`ChimeraX --nogui`), this logs the agreement table and writes the
contacts reached in any seed, with P per seed:

    alphalens open /path/to/prediction_dir mode agreement saveContacts ~/contacts.csv

## Recognised layouts

* **AlphaFold2 / AlphaPulldown:** `ranked_N.pdb|cif` → order in `ranking_debug.json` →
  `pae_<model>.json` (or the PAE in the result pickle). The distogram is
  `result_<model>.pkl['distogram']`.
* **AlphaFold3:** `<name>_model.cif` + `<name>_confidences.json`, or `seed-S_sample-K/model.cif` +
  `confidences.json`. The distogram is `seed-S_distogram/*.npz` (key `distogram`; logits or
  probabilities are detected automatically).
* **ColabFold** (untested): `*_unrelaxed_rank_NNN_*.pdb` + `*_scores_rank_NNN_*.json`, and
  `*_all_rank_NNN_*.pickle` with `--save-all`.
* JSON, pickle and npz files can be `.gz` or `.xz` compressed. AlphaFold DB PAE files are read
  too. For other layouts, give the files with `paeFile` and `distogram`.

## Predictions on a remote machine

Opening predictions over a network mount is slow, because every file lookup and the large result
pickles go over the network. It is faster to copy only the files the plot reads:

    scripts/fetch_pred.sh user@cluster:/path/to/prediction_dir ~/predictions

and then, in ChimeraX, `alphalens open ~/predictions/prediction_dir`. The script uses rsync, so it
needs ssh access to the machine. `--no-pickles` skips the AF2 / ColabFold result pickles, which are
large; without them the AF2 plot shows PAE only.

## Limitations

* The plot is not saved in ChimeraX sessions; run `alphalens` again after restoring a session.
* AF3 `contact_probs` are fixed at 8 Å; another `cutoff` needs the distogram.
* The agreement view keeps N × N values per seed or model in memory (2 bytes each) and refuses
  more than 4 GB: five seeds work up to about 20,000 residues, twenty up to about 10,000.
* ColabFold layouts are untested, and have no model switching or agreement view.

## Development

The file readers, the agreement and the CSV export are tested without ChimeraX:

    pip install numpy pytest
    pytest tests

The plot window itself needs ChimeraX: build and install with `devel install`, then check it on an
AlphaFold2 and an AlphaFold3 prediction.

## License

MIT; see [LICENSE](LICENSE). Bug reports and questions go to the
[issue tracker](https://github.com/KosinskiLab/chimerax-alphalens/issues).
