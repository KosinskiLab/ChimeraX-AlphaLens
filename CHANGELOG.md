# Changelog

## 1.6.1

* **Re-predict selected interface reuses the MSA that comes with more kinds of prediction**, after
  a report from Tom Goddard: the `msas` folder of an AlphaFold Server download, the
  `<name>_data.json` of an AlphaFold 3 run (also compressed, and for a sample's
  `seed-*_sample-*/model.cif`), the `features.pkl` of an AlphaFold 2 run, an AlphaPulldown run's
  features in the `features` folder beside `predictions` (as AlphaPulldownSnakemake writes them),
  and an A3M of a ColabFold run. Before, **Reuse and crop the MSA of this prediction** was greyed
  out unless a LocalFold A3M lay beside the structure. AlphaFold 2 and AlphaFold 3 both take these
  MSAs; AlphaFold 2 pairs the chains' UniProt hits by species, as AlphaFold-Multimer does. A
  prediction with ligands or nucleic acids no longer prevents it.
* Local prediction is no longer called experimental (the dialog's title, the README, the help).
* **AlphaFold Server downloads open as prediction folders** (`alphalens open <folder>`, Open a
  prediction folder…), and the Model menu lists their five models with ranking scores. Before,
  only a `*_model_N.cif` opened, alone, and the folder was refused ("Could not find a rank 0 model").
* **The prediction dialog follows its window to another prediction.** With the dialog open, showing
  another prediction in the window (Model menu) kept the first one's MSA to reuse and its feature
  IDs: Predict then failed, or reused the other prediction's MSA.
* **Feature IDs are guessed for every chain of a heteromer.** AlphaPulldown's AlphaFold 3 names
  (`Q15650_af3_input.json_and_A0MZ66_af3_input.json_model.cif`) gave the first protein's ID to all
  chains, as did a name with only one recognisable ID; a single ID now applies to all chains only
  when they are one protein.

## 1.6.0

Easier to find your way around, after feedback from Tom Goddard (ChimeraX):

* **Help.** The plot window and the prediction dialog have a Help button (lower right) and Help in
  their right-click menu (Ctrl-click on a Mac); the prediction dialog had no context menu
  before. The manual is now installed with AlphaLens and opens in ChimeraX's help viewer:
  this README, the reference and the local prediction guide as linked pages. `help alphalens`
  and `usage alphalens` link to the command reference.
* **The Model menu** lists, below the models or samples of the shown prediction, the other open
  predictions (any open structure with a PAE file next to it) and **Open a prediction
  folder…**; choosing one shows it in the same window. It was greyed out whenever the
  prediction had only one model.
* **Tools → Structure Prediction → AlphaLens** opens a window for the first open prediction that
  has none, so choosing it again gives each prediction its own window; with every prediction
  shown it brings their windows forward, and with none open it asks for a prediction folder.
  Before, with two structures open, it only logged how to use the command.
* **Zoom keeps the point under the pointer in place** (scroll wheel, pinch), and + / − zoom about
  the middle of the view. Zooming after a pan used to jump to the matrix's top left corner. The
  top right corner of the plot now says how to zoom, and once zoomed in how to pan (Shift-drag);
  saved images leave it out. Contact dots are a little easier to click.
* **Choosing another model in the Model menu** carries over what the window drew on the
  structure: the contact lines, the clicked contact (its values for the new model go to the log),
  the hidden residues, the dragged box (still there for Re-predict selected interface) and its
  colors; Color pLDDT and Color PAE Domains color the new model from its own pLDDT and PAE. Before,
  the box was dropped, and the model left behind kept its contact lines: hidden with it, they came
  back when it was chosen again, even with Show Contacts off. Choosing models in quick succession
  could turn Show Contacts off; going back to a model still being read shows its progress again.
* The pLDDT slider's value ("hide <70") is shown beside the slider, or below the pLDDT track when
  there is no room, instead of over the bottom right corner of the matrix. While zooming, the slider
  and its value move with the matrix in the same frame; they used to flash where they had been,
  often on the diagonal.
* **Zoomed in, the pLDDT track stays in view** at the bottom of the plot, under the visible
  columns, with residue numbers and the hide slider; before, it scrolled away and you had to pan
  down to it.
* **Choosing another model no longer freezes the window**: its files are read in the background
  ("Reading ..." above the plot), and a choice made meanwhile replaces it. On a network mount the
  window froze for seconds. The samples of an AF3 seed share its distogram instead of reading
  and computing it again for each sample, and an AF2 result pickle is read only as far as its
  distogram (less than half of the file). Whether a pickle holds a distogram is checked once.
* The greyed-out Spread tab says how to get a distogram, and the Agreement tab says when it uses
  the contact probabilities that AlphaFold3 writes for every sample (no distogram needed).
* Credits: the prediction dialog, the README and the Toolshed description name LocalFold by
  Sergey Ovchinnikov, which local prediction runs.
* The local prediction guide is shorter and written for users; engine details, measurements and
  the prediction checks are in notes for developers.
* The repository is now `KosinskiLab/ChimeraX-AlphaLens`; links are updated.

## 1.5.2

* No change to the plot or to prediction. The bundle's description, which the ChimeraX Toolshed
  shows, is up to date: it covers Model contacts and local prediction, including what local
  prediction downloads and what it sends to UniProt and ColabFold. The development status is
  Beta, and the authors are Dima Molodenskiy and the Kosinski Group, EMBL Hamburg.
* Needs ChimeraX 1.10 or newer (it said 1.6). ChimeraX 1.6 and 1.7 lack functions AlphaLens
  uses to show a clicked contact, select along the pLDDT track and compare with the source;
  1.8 and 1.9 have them but are untested.

## 1.5.1

* Plots docked into the main window stay readable. Docking a second plot under the first used
  to squeeze the first one's matrix to about 50 px, because its controls kept their height.
  The matrix now never gets smaller than 200 px. When a plot is docked, the plots in that
  column share its height equally (up to the size they ask for), and the other tools there
  (Log, Models) keep at least 160 px or their own minimum.
* A plot docked again after floating (two plots floated and docked again) could come back
  loose over the graphics instead of in the dock column; it is now docked properly.
* Add `tests/chimerax_dock_check.py` (two plots docked one after the other, then floated and
  docked again at once) and unit tests of how a dock column is shared.

## 1.5.0

* Is the re-predicted interface the same as the source's? The report, the dialog and the result
  plot's heading give **DockQ against the source** for each pair of chains, over the
  re-predicted residues: the source's contacts kept (Fnat) and new, the interface RMSD and how
  far the smaller chain moved, with the CAPRI class ("DockQ 0.92 · high: same interface"). It
  matches the official DockQ 2.1.3 exactly on the checked pairs. For AF3 results the heading
  follows the sample shown. Saved in `comparison.json`, so reopened job folders show it.
* **Compare with source** (dialog button, and the result plot's right-click menu): the
  re-prediction superposed on the source (as last chosen) and colored by chain, with the same
  residues of the source in grey, the rest of the source hidden, both in view; again to restore
  their colors and display. A source is compared with one re-prediction at a time: comparing
  another (a new job, or another AF3 sample in the Model menu, which the comparison follows)
  first restores the one before. Residues and atoms added or deleted meanwhile (`addh`,
  `swapaa`) do not stop the rest being restored.
* The result plot's right-click menu superposes the re-prediction on the source's first selected
  chain (as it opens) or on the backbone of the source interface, and logs the RMSD. The source's
  heavy atoms are recorded at Predict, so this works after the source is closed; for jobs from
  before 1.5.0 the interface comes from the open source.
* The report says how well the first chain superposed (CA RMSD).
* The job files a result's right-click menu needs are read once per version of the files, not on
  every right-click.
* Add `tests/chimerax_compare_check.py` (AF2 and AF3 re-predictions, the comparison view and its
  restore, after a sample switch, with two results of one source and after `addh`, both
  superpositions, a reopened job and one from before 1.5.0) and unit tests of the DockQ scoring.

## 1.4.2

* The re-predict dialog opens at once: 0.06 s instead of 0.44 s for a 677-residue prediction,
  0.08 s instead of 2.9 s for 2,783 residues. Typing in the dialog or changing the flanks no
  longer stalls (it took 1.5 s a keystroke at 2,783 residues). The crop looked up each residue's
  neighbour through `chain.residues`, which ChimeraX rebuilds on every access, and ran twice on
  opening and again on every change; it now builds each chain's residues once, and the dialog's
  preview reuses the crop until the box, the flanks or the model change (Predict always crops
  afresh).
* Switching models in the plot's Model menu is about twice as fast for large AF3 models (1.5 s
  instead of 2.9 s at 2,783 residues): the other model of the prediction opens without ChimeraX's
  report on it, which ChimeraX 1.12 makes in time quadratic in the chain length.
* Measured (M5 Max, real window): the right-click menu opens in under 20 ms, views switch in
  10-200 ms, the sequence dialog and the prediction report in under 0.1 s.

## 1.4.1

* **Predict** is off only when something it needs is missing, and says what: the reason is shown
  above the buttons and when the pointer rests on the button. Until now the reason was a plain
  line under the dialog's title (for example after a box dragged inside one chain), or nothing at
  all: clicking the chosen alignment option again cleared it, and Predict went grey.
* The reasons name the fix: the box covers chain A only (drag it across two chains, in a block such
  as A × B); choose an alignment; enter the A3M file; enter the feature ID of chain B; accept the
  AF3 terms; no LocalFold engine at the given path (also a file that is not executable, which
  used to fail only after the click); a prediction is running; click Load sequences (the input
  changed); the input could not be read (with the UniProt or parsing error).
* Fewer false stops: X (an unknown residue) is folded as AlphaFold's UNK by both engines, so it is
  no longer refused; ligand and nucleic-acid rows in a dragged box are left out instead of
  refusing the box. U, O, B, Z and J are refused with the residue and what to use instead.
* Add `tests/chimerax_predict_button_check.py`: every case where Predict is off or must stay on,
  with the hover tooltip of the disabled button, in a real window.

## 1.4.0

* **Model contacts** replaces the Model distances view. It answers the view's question directly:
  each pair is green when the shown model forms it (CB–CB < 8 Å) and the prediction supports it
  (P ≥ the slider threshold), vermillion when it is predicted but not formed in the model, purple
  when the model has it but the prediction does not; farther pairs are faint grey. The line under
  the slider counts them for the inter-chain pairs ("87 of 88 predicted inter-chain contacts are
  formed, 1 not; 11 contacts of the model are not predicted"), so AF3 samples and AF2 models can
  be compared from the Model menu. Its dots take the three colors, so the inter-chain pairs stay
  visible in a zoomed-out overview; a dot or colored pair is clicked to show it on the structure.
  Until now the model's contacts were drawn in the green of P, under the contact dots that cover
  the same pairs, so at an interface they could hardly be seen; most of the matrix was grey
  distances. `mode model` (or `distance`) still selects it.
* The Model menu of a local AF3 prediction lists its five samples, best first, so they can be
  compared; it listed only the opened one.
* Zoomed in, contact dots become rings, so the matrix cell under a dot shows its color.
* Switching views no longer widens the window: the legend under the plot is drawn to the
  window's width (the Model distances legend made the window about 400 px wider, and it stayed so).
* Readable in dark mode: chain labels, the pLDDT label and the legend text follow the window's
  colors; they were dark grey on the dark ChimeraX background.
* A quieter window: the views are a segmented control (the ← / → arrow buttons are gone; the keys
  still switch views); the heading is the view's name and one grey line about the model, and the
  interaction hints moved to the line under the slider, shown while the pointer is not on the plot;
  buttons are grouped (structure colors, contacts, files); the pLDDT slider is labeled "hide";
  the "pLDDT" label is no longer clipped; undefined spread is light grey instead of mid grey.
* Asking the plot for an unknown view now names the views; `distance` works as `model`.
* Discontinuous regions, as AlphaPulldown predicts them. A box dragged across a gap in the
  model (an AlphaPulldown region such as `P1:1-100:150-200`, or unresolved residues) is
  re-predicted as one chain with the gap kept; sequence input takes several ranges a chain
  (`1-100,150-200`). The model sees the gap through the residue index, as AlphaPulldown's AF2 and
  AF3 backends do: the segments stay one chain, far apart in sequence, rather than being joined
  into a continuous chain or split into separate chains. Barnase with residues 36–45 cut out:
  the gap's ends stay 14–15 Å apart (16 Å in the full-length model) and the rest is within
  0.3–0.4 Å of the full-length prediction; joined into a continuous chain, the ends are bonded
  and the rest is 1.6 Å away. Precomputed AlphaPulldown features are cropped segment by segment.
* Predicted structures are written in the source numbering: the chain IDs and residue numbers of
  the re-predicted region (insertion codes included), or the UniProt positions of a sequence
  range, so `#1/A:120` and `#2/A:120` are the same residue. The engine's own files (chains A, B,
  ... numbered from 1) are kept in `engine_output/`; `job.json` records the numbering.
* Until now a box across a gap in an AlphaPulldown region was predicted as one continuous chain,
  fusing the two segments: the gap was invisible to the crop, which only looked for unresolved
  residues. Flanks now stop at gaps as at chain ends.
* AlphaPulldown notation in the sequence box: `P0A836:1-100+P0AGE9`, `P69905:2` (two copies),
  `Q8I2G6:2:1-100:150-200`. Each identifier is looked up once.
* A redesigned prediction dialog: the chains in a table (entry, length, range; hover for the
  sequence), one alignment choice with only its own fields shown (reuse the source MSA,
  precomputed features, an A3M file with Browse…, or the ColabFold search), and recycles, seed,
  AF3 sampling, time limit, results folder and engine path under **Settings**, closed by default
  with a one-line summary. The sequence dialog is about 650 px tall instead of 1,150. The plot's
  hint line mentions that a dragged box can be re-predicted from the right-click menu.
* LocalFold comes from KosinskiLab's fork (`github.com/KosinskiLab/localfold`, protected branch
  `alphalens`), pinned at `837a15c`: upstream's `68c5919` plus the residueIds patch. Upstream is
  moving the native engines into private repositories, which would have broken first-time setup.
  Engines built by an earlier AlphaLens are rebuilt once by **Update & Predict**; without
  residueIds the new engines give the same structures as before (identical coordinates, AF2 and
  AF3). Cached MSAs stay valid.

## 1.3.0

* Local prediction on Linux x86_64 with an NVIDIA GPU. **Install & Predict** builds LocalFold's
  CUDA engines, from the same pinned revision as the Metal ones, when `nvcc`, cuBLAS and `g++`
  are available. A missing tool, a machine without a GPU and another user's LocalFold build
  files are reported with what to do. The engines find cuBLAS without the CUDA module loaded.
  Each engine is built for the GPU generation it was set up on, and AlphaLens refuses to run it
  on another: A100-built engines gave NaN or wrong structures on an H100.
  Checked on an A100 against the 1BRS crystal and the Metal results (see the prediction guide).
* `job.json` names the engine: "LocalFold native Metal" or "LocalFold native CUDA". The AF3
  settings that work around the Metal tensor path are set on macOS only.
* Add `tests/localfold_engine_check.py`: real AF2-Multimer, AF2-monomer and AF3 predictions
  through `PredictionJob` without ChimeraX, scored against 1BRS and, optionally, an earlier run.
* The AlphaPulldown-feature unit tests no longer need AlphaPulldown or AlphaFold installed; five
  of them failed on machines without those packages.
* Local predictions can take their alignments from precomputed AlphaPulldown features instead of
  an MSA search: tick **Use precomputed AlphaPulldown features** in the prediction dialog. The
  features come from the lab database (`/g/alphafold/input_features`), its public copy
  (`https://s3.embl.de/alphapulldown`, downloaded once into `~/.cache/alphalens/features`) or any
  AlphaPulldown feature folder. Chain IDs are filled in from the prediction's folder name or the
  UniProt lookup. AF2 pickles (AlphaPulldown `MonomericObject` or AlphaFold feature dicts) are
  read without AlphaPulldown or AlphaFold installed; their `msa_all_seq` is paired by species as
  AlphaFold-Multimer does. AlphaFold 3 uses `<ID>_af3_input.json` features when all chains have
  them. Re-predicted regions are cropped from the full-length features. Reading and converting
  run in the prediction's process, so ChimeraX stays responsive on a slow mount.
* Predicted contacts (dots on the plot, lines on the structure) are colored by a fixed scale of
  P with the pLDDT colors and bands: orange below 0.5, yellow 0.5–0.7, light blue 0.7–0.9, dark
  blue 0.9 and above. Until now they ran from light to dark blue, so at the usual thresholds
  every contact looked confident, and the legend redrew its ramp from the slider value to 1,
  which looked as if the colors were rescaled with the slider. The legend now shows the whole
  0–1 scale with the range below the threshold faded and the threshold marked.
* A **dots on plot** box next to the slider switches the contact dots off, to see the PAE or
  other matrix colors underneath (until now only in the right-click menu).
* The slider is labeled "Contact probability ≥" and the legend "contact probability", not
  "Predicted contacts" / "contacts", which could be read as the contacts in the model.
* A clicked contact is magenta (it was gold, too close to the new yellow band).

## 1.2.1

* Install the optional LocalFold engine on first prediction with **Install & Predict**.
  Setup runs in the background, can be cancelled and retried, and continues with the
  submitted sequences/region and settings when ready. Installing the AlphaLens plugin
  or opening the dialog does not download/build an engine. Existing/custom engines
  are preserved. Missing Xcode Command Line Tools produce actionable instructions.

* Local predictions can be opened again: `alphalens open <job folder>` and
  `alphalens open <job folder>/fold.pdb` find `fold_confidences.json`, and `alphalens #N`
  shows the plot of a result again after its window was closed. Until now these stopped
  with "Could not find a PAE file" or "Could not find a rank 0 model". `rank N` opens the
  other AF3 diffusion samples in the order of `fold_ranking_scores.csv`.
* The re-prediction dialog follows the plot. A box dragged while the dialog is open updates
  the chains and lengths it shows; before, it kept showing the earlier box and predicted
  the new one. Closing the source plot or model turns Predict off with the reason.
* Dragging along the pLDDT track only selects residues. It no longer colors the model lime
  and grey when "Dragging box colors structure" is on.
* After a failed sequence or UniProt lookup, the messages of the next lookup are no longer
  shown in the red error style.
* Alignments (A3M) with non-ASCII bytes in their headers are read and cropped unchanged.
  ChimeraX uses ASCII as its default text encoding once a PDB file has been opened, which
  is the case after the first local prediction.
* Messages and tooltips say "confidence JSON contact_probs" throughout; local AF2
  predictions carry them too, so "AF3 contact_probs" was wrong for those plots.
* Add `tests/chimerax_prediction_mapping_check.py`: a real-window check that needs no GPU
  or network. A stand-in engine returns the cropped source structure, so PAE, pLDDT and
  contact comparisons, CSV rows and the superposition have exact expected values.

## 1.1.3

* Show actual prediction stages and elapsed time instead of repeatedly showing the
  MSA source during inference. Flush AF3 progress output while prediction runs.
* Append stage updates, errors and the identical full score/contact report to both
  the prediction window and ChimeraX Log. Keep earlier runs visible when predicting
  again; avoid repeated messages on every timer tick.

## 1.1.2

* Fix collapsed local AF3 predictions on M5 hardware by disabling the pinned
  LocalFold engine's tensor fast path for AF3. Additional recycles or diffusion
  steps did not fix this backend failure. Conventional Metal kernels recover
  normal geometry; the barnase–barstar control is 0.59 Å Cα RMSD from its crystal.
* Use 10 recycles, 200 diffusion steps and five samples by default for AF3;
  expose the sample count. AF2 defaults are unchanged.
* Reject grossly overlapping or broken Cα geometry before importing a prediction,
  independently of pLDDT. Preserve the output files for inspection.
* Keep score comparisons accessible after closing the prediction dialog. Save
  `prediction_report.txt`, log the full report, and add Prediction report buttons
  to the dialog and result plot's context menu. Reports also reconstruct older jobs.
* AF3 results made with 1.1.0/1.1.1 on the affected M5 path need to be re-predicted.
  Those releases verified execution, not structural accuracy of this backend.

## 1.1.1

* Fix the plot window growing in several steps as the pointer moves across PAE
  cells. Hover statistics now wrap in a fixed-height area and do not enlarge the
  window or refit the matrix. Full text remains selectable and available as a tooltip.
* Verify repeated hover updates keep window size, viewport size and zoom unchanged,
  including after zooming and manually resizing the window.

## 1.1.0

* Choose AlphaFold 2 or AlphaFold 3 for local prediction on Apple Silicon. The setup
  script builds both engines; AF3 weights download after accepting DeepMind's terms.
  AF3 defaults to three recycles, 200 diffusion steps, Euler/Karras sampling and one sample.
* Predict multiple chains from FASTA/UniProt or a selected region spanning multiple chains.
  Remove the two-chain restriction; PDB output supports up to 62 distinct chain IDs.
* Generate a fresh random seed for each run, with a manual seed option for reproducibility.
* Report contacts in the selected region: source/new counts at P ≥ 0.5, top-ten mean
  contact probability, strongest residue pairs, and a CSV with original residue labels,
  probabilities and model distances. Include genuine AF2-Multimer contact probabilities
  using its matching distogram head. On the tested 291-residue crop this added 0.14 s
  (2.9%) to median inference runtime, without changing coordinates, PAE or scores.
* Share cached MSAs across AF2 and AF3; retain earlier cached alignments.
* Include the preview-build fixes: no hard residue-count cap, optional time limit,
  prediction progress, clearer MSA selection and a stable, scrollable result area.
* Remove the region-comparison explanatory paragraph from the prediction dialog.

## 1.0.1a4

* Fix the prediction dialog growing taller with every status update. Status and
  result text now use a fixed-height, scrollable area; long messages remain readable.
* Check that 100 progress updates and long errors leave the window size unchanged.

## 1.0.1a3

* Add prediction progress: an activity bar during model/alignment preparation and
  remote MSA search, completed inference passes during folding, and a completion
  state after the result opens. Elapsed time remains visible; queue time has no ETA.
* Explain the required MSA choice beside its controls and enable Predict only after
  an alignment source is chosen. Startup failures are highlighted and logged.

## 1.0.1a2

* Remove the fixed local prediction length cap. Inputs above 1,024 total residues,
  including flanks, show a non-blocking warning about runtime and GPU memory.
* Replace the fixed five-minute job timeout with an optional time limit in minutes,
  defaulting to No limit. Manual cancellation and the native GPU-memory check remain.
* Cover large input, flanks, unlimited runtime, optional timeouts and cancellation with
  regression checks. A real 799-residue pair completed in 25–26 seconds on the tested
  M5 Max with a supplied MSA (3.26 GB engine buffer peak).
* Clarify that AlphaLens uses the native Metal engine; the browser WebGPU backend is
  not integrated. This is a test build; memory and speed still depend on the hardware.

## 1.0.0

* **Local prediction on Apple Silicon:** predict one or two proteins from raw sequence,
  FASTA, or UniProt accession/entry/isoform IDs with `alphalens predict`, also available
  as Tools > Structure Prediction > AlphaLens Prediction. Preview resolved sequences
  and select residue ranges before running. One chain uses AF2-monomer; two use AF2-Multimer.
* **Re-predict a selected interface:** drag an off-diagonal box and choose Re-predict
  selected interface. Contiguous crops include adjustable flanks. Reuse/crop the source
  MSA, supply an alignment, or explicitly choose remote ColabFold search. The result opens
  in its own plot, superposed on the first chain, with PAE/pLDDT comparisons over the same
  selected residues. Original residue mapping and job provenance are saved.
* The separate LocalFold engine downloads public AF2 weights automatically, caches MSAs,
  keeps ChimeraX responsive, and supports cancellation. The experimental feature is
  limited to 400 total residues and does not run AF3. Tested 108–195-residue examples ran
  in about 1–3 seconds with cached MSAs on an M5 Max; fresh remote MSA searches can take
  minutes. Higher confidence after cropping does not establish better accuracy.
* Fix AF3 distogram seed provenance and retain explicitly supplied distogram paths across
  model switching. Contacts inside a selected region are no longer lost to the global
  display cap; CSV export respects the selected region.
* Model distances refresh after coordinate changes. AF2 contacts restore with saved
  sessions, and background agreement loading finishes safely if its source model closes.
* Add real-window checks, input/provenance tests, a prediction guide, measured examples,
  and a standalone engine setup script distributed with the release.

## 0.10.0

* Opening a session with a plot no longer logs `restore_snapshot for "AlphaLensPlot" returned None`.
  Sessions saved with 0.9.0 open normally. The session test now fails on any warning or error logged
  during a normal save and restore.

## 0.9.0

* **The plot is saved in ChimeraX sessions** (`.cxs`). A session stores the plot's settings and the
  paths of the PAE file and the distogram, never the matrices, so it stays small. When the session
  opens, the plot is rebuilt from those files, with the same view, spread measure, contact threshold,
  dragged box, clicked contact, pLDDT slider and hidden residues. Contact lines and what was shown
  or hidden in the structure come back as saved.
* If a distogram has moved, the plot opens without Spread and says so in the log. If the PAE file is
  gone too, the session still opens with the structure and its contact lines, without the plot.
* Opening a session that has a plot needs AlphaLens 0.9 or newer. The plot's zoom and window
  position are not saved.

## 0.8.0

* **pLDDT slider** right of the pLDDT track: hides the parts of the 3D model below its value.
* Dragging along the pLDDT track selects those residues; right-click, **Hide selected** hides them
  and **Show hidden** restores them. Residues selected in the 3D view can be hidden the same way.
  Only what was hidden this way is shown again, so side chains that were off stay off.
* Right-clicking the plot no longer starts a drag or clears the box drawn by the last drag.

## 0.7.0

* **Progress bar while distograms load**, above the plot: gigabytes read and the time left, then
  the computing step, and for the Agreement view which seed or model is being read. Large
  distograms over a network share take minutes.
* **Contact dots on every matrix** (PAE, Contact probability, Model distances, Spread, Agreement),
  not only on the contact-probability view. Right-click, "Show contact dots in all views", keeps
  them to the contact-probability view.
* AF3 contacts appear at once from `contact_probs`, and the distogram values replace them when it
  has loaded. Before, the Contact probability tab and the contact lines waited for the distogram.
  Export CSV and the Agreement tab stay off until the distogram has loaded, as they use its data.

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
