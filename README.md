# ChimeraX-AlphaLens

*Formerly DistoPAE (up to version 0.4). The `distopae` command still works.*

AlphaLens is a [ChimeraX](https://www.cgl.ucsf.edu/chimerax/) tool for inspecting
AlphaFold2/AlphaFold3 predictions and running small local protein predictions. Like
ChimeraX's built-in AlphaFold error plot, it shows the PAE matrix with a pLDDT track. It adds views
computed from the predicted distogram, compares the predicted contacts with the contacts in the
model, shows which contacts are reproduced across seeds or models, draws the predicted contacts on
the structure and exports them to CSV. On Apple Silicon Macs and on Linux with an NVIDIA GPU,
it can also predict from raw sequences, FASTA or UniProt IDs, or re-predict an interface
selected in the plot.

<p align="center"><img src="docs/images/contacts.png" width="640"
  alt="AlphaLens plot in the contact-probability view, with dots on the inter-chain contacts"></p>

*The contact-probability view for an AlphaFold2 prediction of E. coli succinyl-CoA synthetase,
SucC–SucD (UniProt P0A836, P0AGE9). Dots mark the inter-chain pairs with P ≥ 0.5, colored by P in
the pLDDT bands. The ringed one was clicked; its values are in the line under the slider.*

| View | Matrix | Colors (dark = confident) |
|---|---|---|
| PAE | predicted aligned error (Å) | 0–30 Å |
| Contact probability | predicted P(d < 8 Å) from the distogram, or `contact_probs` in the confidences JSON (AlphaFold3 and local predictions) if there is no distogram | √P scale, so weak contacts show |
| Model contacts | the contacts in the shown model (CB–CB < 8 Å) against the predicted ones (P ≥ the slider threshold) | green: in the model and predicted; vermillion: predicted, not in the model; purple: in the model, not predicted; faint grey: farther pairs, 8 to 40 Å |
| Spread σ / H | standard deviation (Å) or entropy (bits) of the predicted distance distribution | 0 to the 99th percentile of the model |
| Agreement | number of AF3 seeds or AF2 models in which the pair has P ≥ the slider threshold | 0 to the number of seeds or models |

<p align="center"><img src="docs/images/views.png" width="760"
  alt="The PAE, contact-probability, model-contact, spread and agreement views of one prediction, and its contacts drawn on the structure"></p>

*The five views of the same prediction, each with the predicted contacts as dots, and its
inter-chain contacts with P ≥ 0.5 drawn on the structure by Show Contacts. The Model contacts view
marks the predicted pairs by its own colors instead of dots. In the spread view, light grey marks
pairs whose distance is mostly beyond the last distogram bin. The agreement view counts the five AlphaFold2 models in which each pair has
P ≥ 0.5.*

## Install

Version 1.5.1 keeps plots docked in the main window readable. Version 1.5.0 compares a re-predicted interface with its source (DockQ, and a Compare view in 3D). Version 1.4.2 opens the prediction dialog at once on large complexes. Version 1.4.1 says why **Predict** is off, above the buttons and on hover, and turns it off only
for real problems. Version 1.4.0 predicts discontinuous regions as AlphaPulldown does (one chain
with the gap kept), writes predicted structures in the source numbering, reads AlphaPulldown
notation (`P0A836:1-100+P0AGE9`), adds the Model contacts view, has a simpler prediction dialog,
and builds LocalFold from KosinskiLab's fork. Version 1.3.0 added local prediction on Linux with an NVIDIA GPU, through LocalFold's CUDA
engines, beside the Metal engines on Apple Silicon. It can take the alignments of a local
prediction from precomputed AlphaPulldown features (the lab database, its public copy or a
feature folder) instead of an MSA search, colors predicted contacts by a fixed P scale in the
pLDDT colors, and adds a "dots on plot" switch next to the slider. Since 1.2.1, AlphaLens sets
up LocalFold when you first predict, reopens saved local prediction folders, and keeps the
selected-region preview in sync with the plot.

Requires ChimeraX 1.6 or newer; tested with 1.10 and 1.12. The bundle is pure Python, so one wheel works on
macOS, Linux and Windows. The optional local prediction engine requires Apple Silicon
macOS 13 or newer (Metal), or Linux x86_64 with an NVIDIA GPU (CUDA). It is not available
on Windows or on Intel Macs. It is tested with ChimeraX 1.12.

1. Download `chimerax_alphalens-<version>-py3-none-any.whl` from the
   [Releases](https://github.com/KosinskiLab/AlphaLens/releases) page. Keep the file name:
   pip needs it.
2. In the ChimeraX command line:

       toolshed install ~/Downloads/chimerax_alphalens-1.5.1-py3-none-any.whl

3. Restart ChimeraX.

`toolshed list` shows the installed version (look for AlphaLens). To update, install the newer wheel
in the same way. To remove the bundle, run `toolshed uninstall AlphaLens`.

**Upgrading from DistoPAE:** install AlphaLens as above, then remove the old bundle with
`toolshed uninstall DistoPAE` and restart ChimeraX. Scripts that call `distopae` keep working.

### From source

    git clone https://github.com/KosinskiLab/AlphaLens.git

Then, in the ChimeraX command line, change to the checkout first:

    cd /path/to/AlphaLens
    devel install .        # install from the checkout
    devel build .          # or only build the wheel into dist/
    devel clean .          # remove build files

ChimeraX builds a `pyproject.toml` bundle from its current directory, whatever path is given to
`devel`: without the `cd`, `devel install` fails with "package directory 'src' does not exist"
(seen in ChimeraX 1.10).

### Optional local prediction engine (Apple Silicon or Linux with an NVIDIA GPU)

Open **AlphaLens Prediction**, choose the model and input, then click **Install & Predict**
if that engine is missing. AlphaLens downloads and builds the selected LocalFold engine
in the background, then starts the prediction with the submitted input and settings.
**Cancel setup** stops installation; retry the same button when ready. On a Mac, setup requires
Xcode Command Line Tools (`xcode-select --install` if missing). A fresh AF3 engine build
took about 30 seconds on the tested M5 Max, excluding model weights and MSA search. On Linux,
setup compiles LocalFold's CUDA engines for the machine's GPU (NVIDIA T4 or newer) and needs
the CUDA toolkit (`nvcc`, cuBLAS 12) and `g++`; see the
[prediction guide](docs/local-prediction.md#linux-with-an-nvidia-gpu-cuda). The plugin installation
and opening its prediction dialog do not install engines. Existing engines are reused.

For optional manual installation of both engines, download
[setup_localfold.py](https://github.com/KosinskiLab/AlphaLens/releases/download/v1.5.1/setup_localfold.py)
and run `python3 ~/Downloads/setup_localfold.py`. First-use setup uses ChimeraX's bundled
Python. The first AF2 prediction downloads about 77 MB of model weights; AF3 downloads
about 277 MB after you accept DeepMind's terms in the dialog.
Existing MSAs stay local;
remote MSA search sends sequences to ColabFold only when explicitly selected.
See the [prediction guide](docs/local-prediction.md) for details and measured limits.

## Quick start

    alphalens predict                               # sequence, FASTA or UniProt input
    alphalens open /path/to/prediction_dir            # top-ranked model
    alphalens open /path/to/prediction_dir rank 2     # AF2 ranked_2 / AF3 third-best sample
    alphalens open /path/to/seed-1_sample-0/model.cif # one AF3 sample

`open` can be left out: `alphalens /path/to/prediction_dir` does the same.

For a structure that is already open, use Tools > Structure Prediction > AlphaLens (when exactly
one structure is open) or `alphalens #1`.

The plot reads the PAE file next to the structure, plus the distogram if the prediction kept it.
If the distogram is missing, you get the PAE view and, for AF3, the contact view from
`contact_probs` from the same seed. A missing seed distogram is never replaced by another seed's
distogram. A top-level AF3 model needs a readable `ranking_scores.csv` to identify its seed;
without it, open the specific `seed-S_sample-K/model.cif` or supply a known matching
`distogram` explicitly. The tabs that need unavailable data are greyed out with a tooltip saying why.
A readable PAE file is required; an unreadable optional distogram still leaves the PAE view usable.
Explicit requests for views, contacts or CSV export fail if their required data cannot be loaded.

## Getting a distogram

The contact view needs either a distogram or confidence JSON `contact_probs` (including local AF2 and AF3); the spread view needs a
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
(AF3, float16) or 1 GB (AF2, float32) before compression. The plot opens at once with PAE, and for
AF3 with the contacts from `contact_probs`, and loads the distogram in the background. A progress
bar above the plot shows how far it has come; the Spread tab turns on when it finishes.

## Using the plot

* **Views:** the tabs, or the ← / → keys in the plot. Hovering shows the values for a residue pair;
  otherwise the line under the slider gives a hint for the shown view. Switching views keeps the
  window's size.
* **Zoom:** scroll wheel, trackpad pinch, or + / −. **Pan:** Shift-drag or middle-drag. **Fit**
  (or 0 / Esc) resets the view. Matrices larger than 900 residues are pooled for display
  (contacts max-pooled, so no contact disappears; PAE and spread mean-pooled). When you zoom in,
  the visible region is redrawn at full resolution, and hovering always reports the exact
  per-residue values.
* **Model:** the dropdown lists the AF2 ranked models or AF3 samples of the same prediction that
  are present locally. The chosen model opens in the same window, superposed on the previous one,
  which is hidden; the view and zoom are kept.
* **Contact dots:** in every view but Model contacts, inter-chain pairs above the slider threshold
  (pairs at least 24 matrix positions apart for a single chain) are drawn as dots that stay
  visible at any zoom. Zoomed in, where a dot would cover a whole matrix cell, it becomes a ring,
  so that the cell's own color shows through.
  Their color is a fixed scale of P with the pLDDT colors and bands (P × 100 as the score):
  orange below 0.5, yellow 0.5–0.7, light blue 0.7–0.9, dark blue 0.9 and above. The slider
  only decides which pairs are drawn, not their color; the legend next to the color bar fades
  the range below the threshold. Untick **dots on plot** next to the slider to see the matrix
  colors underneath; right-click to keep the dots to the contact-probability view. At most the 40,000 strongest pairs are
  shown as dots; CSV export is not capped by this display limit.
* **Hide by pLDDT:** the vertical slider right of the pLDDT track ("hide") hides the parts of the
  model with pLDDT below its value; a line marks the value and hidden residues fade on the track.
* **Hide selected:** drag along the pLDDT track to select residues, then right-click the plot and
  choose Hide selected. It also hides residues selected in the 3D view. Show hidden restores them.
* **Contact probability ≥ slider:** one threshold on the predicted P (0.05–1, default 0.2) for
  the dots, the 3D lines, the agreement view and the CSV export. "hide in matrix too" also blanks matrix pixels
  below it.
* **Show Contacts:** draws all predicted contacts above the slider threshold on the structure as
  CA–CA lines, in the same colors as the dots, thicker for higher P; weak contacts are drawn
  semi-transparent, so the confident interfaces stand out. Move the slider down to 0.05 to see
  every contact. Up to 10,000 lines are drawn, strongest first.
* **Click a dot:** draws that contact on the structure as a thick magenta line (the plot rings
  the dot in the same magenta), labels both residues and shows their side chains. The log gives P, both
  PAE values, the expected distance (with a distogram) and the CA–CA distance in the model.
  Clicking another dot replaces it. Right-click > Clear clicked contact removes it; with
  right-click > Zoom 3D view to clicked contact switched on, the camera also moves there.
* **Drag a box:** only the contacts inside it, above the slider threshold, are drawn on the
  structure, for example the block between two chains; a click on empty plot space or Show
  Contacts restores all. The log lists the strongest pairs. By default, dragging a box also colors
  those residues (an off-diagonal box uses two colors); this can be switched off in the
  right-click menu.
* **Model contacts:** compares the contacts in the shown model, CB–CB closer than 8 Å (CA for
  glycine: the atoms the distogram describes), with the predicted contacts, P ≥ the slider
  threshold. Each pair is green when it is in the model and predicted, vermillion when it is
  predicted but not formed in the model, and purple when the model has it but the prediction
  does not support it; the other pairs are faint grey, darker for closer pairs (8 to 40 Å). The
  line under the slider counts them for the inter-chain pairs, for example "87 of 88 predicted
  inter-chain contacts (P ≥ 0.50) are formed, 1 not; 11 inter-chain contacts of the model are not
  predicted": switch between AF3 samples or AF2 models in the Model menu to see which one forms
  the predicted interface. Click a colored pair to show it on the structure, as a dot. Hovering
  gives the category and the CB–CB distance in the model (in every view). In a zoomed-out
  overview, the predicted contacts the model misses are kept over the others. Without contact
  probabilities (AF2 without a distogram), the model's contacts are drawn in one color.
* **Color pLDDT** colors the structure in the four AlphaFold pLDDT bands (< 50, 50–70, 70–90,
  > 90), as the pLDDT track of the plot, instead of ChimeraX's smooth "alphafold" palette.
* **Color PAE Domains** colors the structure using ChimeraX's PAE domain clustering.
* Large models that ChimeraX shows as spheres are switched to cartoon the first time contact lines
  are drawn, because the lines would otherwise be hidden inside the atoms.
* **Spread:** the last distogram bin collects every distance above 21.7 Å, so the spread is
  undefined where that bin dominates. Pairs with P(d > 21.7 Å) > 0.5 are drawn light grey; the
  threshold is in the right-click menu.
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

## Sessions

`save mysession.cxs` keeps the plot with the session. The session stores the plot's settings and
the paths of the PAE file and the distogram, never the matrices, so it stays small. When it opens,
the plot is rebuilt from those files with the same view, contact threshold, dragged box, clicked
contact, pLDDT slider and hidden residues. If the distogram has moved, the plot opens without
Spread; if the PAE file is gone too, the structure and its contact lines still open, without the
plot. Opening such a session needs AlphaLens 0.9 or newer. The plot's zoom and window position are
not saved.

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
| `mode` | view to show: `contact` the predicted contact probability, `model` the contacts in the model against the predicted ones (`distance`, its former name, also works), `std` / `entropy` the spread; `agreement` reads every seed or model first and logs the agreement table |
| `cutoff` | contact distance in Å (default 8), also for the contacts in the Model contacts view; needs a distogram, because `contact_probs` are fixed at 8 Å |
| `contacts`, `minProb` | draw or remove the contacts with P ≥ minProb on the structure (default 0.5; also moves the slider) |
| `saveContacts` | write the contacts to a CSV file as Export CSV does, with P ≥ minProb if given, otherwise the slider threshold (0.2 without a plot) |
| `palette`, `range` | PAE color map, as in `alphafold pae` |
| `dividerLines` | chain divider lines on the plot |
| `wait` | finish the distogram and requested agreement loading before the command returns, including reads already running in the background; use `wait true` before consuming a GUI command's CSV output |
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
* PAE JSON and result pickle files, and NPZ distograms, can be `.gz` or `.xz` compressed.
  AlphaFold DB PAE files are read too. For other layouts, give the files with `paeFile` and `distogram`.

## Local protein and interface prediction (experimental)

On Apple Silicon, or on Linux with an NVIDIA GPU, AlphaLens predicts protein complexes from raw sequences,
FASTA or UniProt IDs using `alphalens predict` (also under Tools → Structure Prediction
→ AlphaLens Prediction). It can also re-predict regions spanning multiple protein chains.
Drag an interface box, then right-click **Re-predict selected interface…**.
Choose **AlphaFold 2** (monomer or multimer automatically) or **AlphaFold 3** in the model
selector. AF3 uses 10 recycles, 200 diffusion steps and five samples by default. Seeds are randomized
for each prediction; uncheck the random-seed option to enter a reproducible seed.
Reuse/crop an existing MSA or explicitly choose a
remote ColabFold search. Results open in a new plot with PAE/pLDDT comparisons over the
same residues, contact counts, top-ten mean contact probability and strongest residue
pairs. `region_contacts.csv` records source/new probabilities and model distances with
original residue labels. The report gives **DockQ against the source** (is it the same
interface?), also in the result plot's heading; **Compare with source** shows the re-prediction
by chain with the same residues of the source in grey. The full score comparison is saved as
`prediction_report.txt` and can be reopened through **Prediction report…** in the dialog or
result plot’s context menu.
A job folder opens again later with `alphalens open <job folder>`.
The original structure is retained. There is no fixed residue-count cap;
inputs above 1,024 total residues (including flanks) show a warning and can still run.
The engine checks its estimated GPU-memory requirement. The optional time limit defaults
to **No limit**; **Cancel job** remains available. This integration uses LocalFold's native
engines: Metal on macOS and CUDA on Linux. LocalFold’s browser WebGPU backend is not connected
to AlphaLens.
Tested 175–195-residue predictions
took about 2–3 seconds with cached MSAs on an M5 Max. A full 799-residue pair took
25.9 seconds with a supplied MSA and a 3.26 GB engine-reported buffer peak. Fresh
remote searches can take minutes. The progress bar shows activity during preparation
and MSA search, then completed inference passes for AF2 or an activity indicator for AF3. **Predict**
is off only while something it needs is missing (a box across two chains, an alignment, the AF3
terms, the engine); the reason is shown above the buttons and when you hover over Predict.

### Alignments from precomputed AlphaPulldown features

Instead of an MSA search, choose **Precomputed AlphaPulldown features** and give each chain's
UniProt ID (filled in from the prediction's folder name, e.g. `Q15014+Q9NV56`, or from the
UniProt lookup). **Features from** is one of:

* `/g/alphafold/input_features`, the lab database through the `/g` mount (the default when it is
  mounted): AF2 features of 16 proteomes, AF3 features of *B. subtilis* and *L. monocytogenes*;
* `https://s3.embl.de/alphapulldown`, the same database, public (the default otherwise). Only
  the IDs are sent; the files are cached in `~/.cache/alphalens/features`;
* any folder of AlphaPulldown features (`<ID>.pkl[.xz]`, `<ID>_af3_input.json[.xz]`).

AF2 feature pickles (AlphaPulldown's `MonomericObject`, or AlphaFold's own per-chain feature
dict) give each chain its unpaired alignment, and their `msa_all_seq` is paired by species as
AlphaFold-Multimer does. AlphaFold 3 uses `<ID>_af3_input.json` when every chain has one (its
alignments go into the job, AF3 pairs them), else the AF2 features. Re-predicted regions are
cropped from the full-length features. Templates are not used, as with the search. Feature
files of a whole complex (an AF-Multimer `features.pkl`) are refused. The features are found,
read and converted in the prediction's own process before LocalFold starts, so a slow network
mount does not block ChimeraX; the job folder keeps the alignments (`features/`) and which files
they came from with their checksums (`features/features.json`). Set `ALPHALENS_FEATURES_DIR`
to change the default location.

See [setup, workflow, measurements and limitations](docs/local-prediction.md).

## Predictions on a remote machine

Opening predictions over a network mount is slow, because every file lookup and the large result
pickles go over the network. It is faster to copy only the files the plot reads:

    scripts/fetch_pred.sh user@cluster:/path/to/prediction_dir ~/predictions

and then, in ChimeraX, `alphalens open ~/predictions/prediction_dir`. The script uses rsync, so it
needs ssh access to the machine. `--no-pickles` skips the AF2 / ColabFold result pickles, which are
large; without them the AF2 plot shows PAE only.

## Limitations

* `contact_probs` (AlphaFold3 and local predictions) are fixed at 8 Å; another `cutoff` needs the distogram.
* The agreement view keeps N × N values per seed or model in memory (2 bytes each) and refuses
  more than 4 GB: five seeds work up to about 20,000 residues, twenty up to about 10,000.
* ColabFold layouts are untested, and have no model switching or agreement view.

## Development

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
The release procedure for maintainers is `.claude/skills/alphalens-release/SKILL.md`.

The images in this README are rendered from the installed bundle by
`docs/render_readme_images.py`; its header says which predictions it needs and how to run it.

## License

MIT; see [LICENSE](LICENSE). Bug reports and questions go to the
[issue tracker](https://github.com/KosinskiLab/AlphaLens/issues).
