# Changelog

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
