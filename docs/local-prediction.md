# Experimental local protein and interface prediction

AlphaLens 1.1.0 can predict protein complexes from sequences or UniProt IDs, or re-predict
selected regions spanning multiple protein chains, with AlphaFold 2 or AlphaFold 3 on
Apple Silicon, using [LocalFold's native Metal implementation](https://github.com/sokrypton/localfold/tree/68c59196fa7d32c6edfc061ed369889c1cdb680b/metal).
The prediction runs in a separate process; ChimeraX remains interactive. The original
structure stays open, and the result gets its own AlphaLens plot. The integration uses
LocalFold’s native Metal executable, not its browser WebGPU backend; other GPU backends
are not yet connected to AlphaLens.

## Setup

Install the AlphaLens 1.1.0 wheel. The local prediction feature remains experimental.
The engine requires Apple Silicon, macOS 13 or newer, Git, Python 3 and Xcode Command
Line Tools. It does not require a Python ML environment inside ChimeraX.

Download `setup_localfold.py` from the GitHub release and run `python3 ~/Downloads/setup_localfold.py`.
Alternatively, from an AlphaLens checkout, run in Terminal:

```sh
python3 scripts/setup_localfold.py
```

The script builds LocalFold at commit `68c59196fa7d32c6edfc061ed369889c1cdb680b`,
installs `~/.local/share/alphalens/bin/localfold-af2` and `localfold-af3`, and records each
revision and binary SHA-256 beside it. Run setup again when upgrading from 1.0.
Use `--models af2` or `--models af3` to build just one engine. `--prefix /path` changes the installation directory;
`--source /path/to/clean/localfold` uses an existing checkout at that exact revision.
The dialog also accepts an explicit executable path (`ALPHALENS_LOCALFOLD` sets its default).
`ALPHALENS_LOCALFOLD_AF3` sets the AF3 executable default. An executable without a
matching provenance sidecar is recorded as unverified.

Choose **AlphaFold 2** or **AlphaFold 3** in the model selector. Weights are cached in
`~/.cache/alphalens/localfold`:

- AF2 downloads about 77 MB for each int5 bundle. It uses `model_1_ptm` for one chain
  and `model_1_multimer_v3` for complexes, with three recycles, at most 512 MSA rows
  and 1024 extra rows. AlphaLens includes the matching multimer-v3 float32 distogram
  head (33 KB, CC BY 4.0) in a separate enhanced cache; the original cache is preserved.
- AF3 downloads about 277 MB of LocalFold's int8 parameter representation after you
  check acceptance of [DeepMind's terms](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md)
  and [prohibited-use policy](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_PROHIBITED_USE_POLICY.md).
  Acceptance is remembered locally. Defaults are three recycles, 512 MSA rows,
  200 diffusion steps, one sample and the Euler/Karras sampler. The diffusion step
  count is editable. These settings are not DeepMind's full default sampling protocol.

These are quantized AlphaFold parameters run by LocalFold, not models trained by
AlphaLens. AF3 weights are not bundled or redistributed in the AlphaLens wheel.

## Predict from sequence, FASTA, or UniProt

Open **Tools → Structure Prediction → AlphaLens Prediction**, or run:

```text
alphalens predict
```

No open structure is needed. Paste one raw amino-acid sequence (line wrapping is
allowed), multiple FASTA entries, or whitespace-separated UniProt identifiers.
Use FASTA headers to separate sequences; plain wrapped sequence lines form one chain.
For example, paste `P69905 P68871` to retrieve the two human hemoglobin subunits.
Accessions, isoforms such as `P05067-2`, and entry IDs such as `RNBR_BACAM` are supported;
ambiguous gene symbols are not looked up automatically.

Click **Load / resolve sequences**. UniProt retrieval uses its
[official FASTA endpoint](https://github.com/ebi-uniprot/uniprot-manual/blob/main/help/api_retrieve_entries.md)
in a background thread, sending only the identifiers. Review the resolved headers and
sequences. Optional per-chain ranges are **1-based and inclusive** (e.g. `50-157` for
the tested barnase segment from `P00648`). Optional ranges select domains from longer UniProt proteins; there is no fixed
residue-count cap. Residue maps
retain positions in the original input sequence. Nonstandard amino acids in the
selected ranges are rejected.

Supply an A3M whose query exactly matches the selected sequence(s), or explicitly
choose remote MSA search. The local MSA cache is checked first. Click **Predict**.
With AF2, single chains use the monomer model and complexes use multimer. AF3 handles
both through its own model. Single chains report pTM/pLDDT; complexes also report ipTM.
Sequence-only jobs open a new PAE plot without inventing a comparison to a source
structure. Both models include contact probabilities. UniProt provenance records the requested/resolved identifiers, FASTA
header, release, retrieval time, original sequence hash and selected ranges.

## Re-predict a selected interface

1. Open a prediction in AlphaLens and drag a rectangle across an off-diagonal
   interface block. Either axis can span multiple protein chains; only inter-chain
   pairs inside the rectangle are included in the comparison.
2. Right-click the plot and choose **Re-predict selected interface…**.
3. Choose flanks (default 10 residues on each side). The crop preserves contiguous
   sequence intervals, clips at chain ends, and rejects gaps in the source sequence,
   and nonstandard amino acids. Large inputs show a warning without disabling Predict.
4. Choose an MSA source. **Reuse and crop the source MSA** is selected automatically
   when a matching `<structure-stem>.a3m` or `input.a3m` is found beside the source.
   Embedded MSAs in AF3 `*_data.json` files are not imported automatically.
   Its first query must exactly match the plotted protein tokens. Cropping preserves
   existing paired/unpaired rows and their insertion encoding. Alternatively, supply
   a merged A3M for exactly the cropped sequences, in the displayed order.
5. If no suitable MSA exists, explicitly enable **Search remotely if uncached**.
   This sends the selected sequences to `https://api.colabfold.com`, unless the exact
   query already exists in the local MSA cache. `LOCALFOLD_MMSEQS_API` can select a
   compatible private endpoint; the dialog displays the configured endpoint.
6. Click **Predict** (enabled once an MSA source is chosen). The progress bar shows
   activity during preparation and remote search, then the number of completed
   inference passes for AF2. AF3 uses an activity indicator because its native engine
   does not emit incremental diffusion progress. The displayed AF2 fraction covers inference passes,
   not total runtime; remote queues have no known percentage or ETA. The bar reaches
   Complete only after the validated result opens. Use **Cancel job** to stop it. A process-wide file lock serializes
   jobs across AlphaLens windows and instances. The optional **Time limit (minutes)**
   defaults to **No limit**. A positive value covers the whole job, including MSA search. Closing a running dialog cancels its job;
   closing the source model does not interrupt an already-started prediction.

**Generate a new random seed for each prediction** is enabled by default. Uncheck it
to enter a fixed seed (including zero); the actual seed is shown in results and saved
in the job manifest.

A result opens automatically, superposed on the first selected chain's Cα atoms at
job submission. The whole predicted complex moves together, preserving the predicted
interface. Source coordinates are not supplied as a template or restraints.

The dialog reports ipTM and compares mean PAE in both directions over exactly the
selected residue pairs, excluding added flanks. It also compares mean pLDDT over the
same selected residues. The contact report gives source/new counts with P ≥ 0.5,
the mean of the ten highest contact probabilities (or all pairs when fewer than ten
exist), the strongest residue pairs, and CB-distance contacts below 8 Å (CA for glycine).
These are residue-pair probabilities, not a probability that the proteins interact.
Source and result distance cutoffs are reported explicitly; missing source probabilities
remain unavailable. Flanks are excluded. All outputs are retained in timestamped job directories under
`~/.local/share/alphalens/predictions` (editable in the dialog):

- `fold.pdb`, `fold_confidences.json`, `fold_summary_confidences.json`;
- `input.json`, `residue_map.json`, and the supplied/cropped/searched A3M;
- `job.json`: model/settings, engine and MSA hashes, source, timing and status;
- `comparison.json` for region jobs, and `prediction.log`;
- `contact_report.json` and `region_contacts.csv` for complexes. The CSV contains the
  union of pairs with source/new P ≥ 0.5 or source/new model distance below 8 Å, with
  original chain/residue labels, both probabilities and cutoffs, and both distances.

Residue maps preserve original chain IDs, residue numbers, insertion codes, sequence
positions and matrix rows. Prediction chains are renamed A, B, … and renumbered from 1. The current PDB output
format supports 62 distinct chain IDs; the earlier two-chain restriction is removed.
MSA caches include the ordered sequences, configured endpoint and engine revision.
AF2 and AF3 share matching cached MSAs, including legacy AF2 caches.
Cache hits reuse existing alignments; they do not claim a new database search.

## Large inputs and time limits

There is **no fixed residue-count cap**. Above 1,024 total residues across all chains (including flanks), the dialog shows a non-blocking warning about runtime
and GPU memory. Predict remains enabled. The pinned native engine checks its estimated
allocations against Metal's recommended working-set budget and refuses an input that
does not fit; removing the count cap does not remove physical memory requirements.

The optional **Time limit (minutes)** defaults to **No limit**. Set a positive number
to cancel after that duration, including MSA search. **Cancel job** and closing the
running dialog still cancel the process. Larger inputs can take substantially longer.
The selected time limit is recorded in `job.json` (`null` means no limit).

A real 799-residue LAMP2–SH2D2A prediction (410 + 389 residues), three recycles,
512 MSA + 1024 extra rows, completed through the region dialog in **25.9 seconds**
on the M5 Max/64 GB with its saved unpaired MSA. The native engine reported a
**3.26 GB peak buffer allocation**, which is not total system memory usage. The
largest observed GUI heartbeat gap was 0.18 s. The prediction passed sequence,
coordinate and confidence-file validation. This is a capacity check, not an accuracy
benchmark: its ipTM was 0.212. Larger input is not evidence of better predictions.
Validation tests accept inputs beyond the former cap and cover the warning threshold
with flanks. This machine's real prediction measurement covers 799 residues. The installed-build sequence-input check
also completed in 24.8 seconds; see the [recorded capacity checks](local-prediction-large-benchmark.json).

## What is verified, and what is not

Measurements on an Apple M5 Max (40 GPU cores, 64 GB unified memory), macOS 27.2,
ChimeraX 1.12, on 10 November 2026:

| Case | Residues | Wall time | Result |
| --- | ---: | ---: | --- |
| Barnase–barstar, first successful native GPU run | 195 | 21.0 s | ipTM 0.923; mean pLDDT 97.4 |
| Same complex, cached MSA, three seeds | 195 | 2.26–2.81 s | ipTM 0.922–0.923 |
| ChimeraX: start through imported plot, cached MSA | 195 | 2.36 s | ipTM 0.922 |
| ChimeraX: crop source MSA and predict | 175 | 3.03 s | ipTM 0.898 |
| ChimeraX: cached crop, different seed | 175 | 2.46 s | ipTM 0.835 |
| ChimeraX: two pasted FASTA sequences | 195 | 2.35 s | ipTM 0.923 |
| ChimeraX: raw sequence, first monomer run including weight download | 108 | 7.07 s | pTM 0.884; pLDDT 97.5 |
| ChimeraX: UniProt `RNBR_BACAM`, range 50–157, cached MSA | 108 | 1.52 s | pTM 0.884; pLDDT 97.5 |

The sequence/UniProt timings start at Predict, after sequence resolution/review;
UniProt lookup time is excluded. The first successful multimer GPU measurement already had downloaded weights; it includes
10.0 s of MSA lookup and first-use shader compilation. It is not a from-scratch
installation timing. A later fresh crop search was still queued after 154 s and was
cancelled. **Fresh remote MSAs have unpredictable latency; source-MSA reuse is the
interactive path.** Subsequent warm GUI repeats were about 2.1–2.4 s.
The [recorded measurements](local-prediction-benchmark.json) include timings,
confidence values, crystal comparisons and GUI heartbeat counts.

Sequences came from LocalFold's public 1BRS crystal fixture, chains A/D (108 + 87
residues). The crystal was used only for validation, not as prediction input. Complex
Cα RMSD was 0.603–0.620 Å across three full-complex seeds, with an approximately
1.59 GB process peak memory footprint in the first successful run. This is a small
smoke benchmark on a known complex, not evidence of general accuracy, equivalent
accuracy to full-precision AF2, or performance on other GPUs or untested sequence lengths.

The 175-residue crop did **not** improve confidence: mean PAE over the same selected
pairs increased from 2.86 to 3.52 Å (another seed: 4.06 Å). Cropping changes the problem
and can inflate or reduce ipTM. Higher confidence alone does not establish better
structural accuracy. Retain domain context and compare multiple seeds and crop sizes.

AF2-monomer, AF2-Multimer and AF3 predictions now include contact probabilities. The
full distogram needed for the spread view is not exported by this integration.
Ligands, nucleic acids, mutations, disconnected sequence grafts and distributed MSA
querying are outside the current protein-only dialog.

On the same Mac, three interleaved runs of a 291-residue crop gave median AF2-Multimer
wall times of **4.666 s without** and **4.803 s with** the matching distogram head:
**+0.136 s (2.9%)**, with identical PDB coordinates, PAE and summary scores. Engine
peak buffers were 1.39 GB in both cases. The original 134-second job spent about
130 seconds obtaining its remote MSA. An AF3 run of this crop with the cached MSA,
200 Euler/Karras diffusion steps and three recycles took **22.77 s including the
first weight download**, with **5.82 s of inference**. It produced validated coordinates,
PAE and contact probabilities; its ipTM was 0.15. This verifies execution and timing,
not biological correctness or equivalence to DeepMind's implementation.
See [the recorded model/head benchmark](local-prediction-models-benchmark.json).

## Regression checks

`tests/chimerax_prediction_models_check.py` runs real AF2 and AF3 predictions in a
disposable ChimeraX window. It covers three-chain sequence input, a region spanning
three chains, random seeds, AF3 terms gating, probability imports, contact CSV mapping,
GUI responsiveness and window-height stability. Its public fixture splits barstar
into two fragments solely to test chain handling.

`python3 -m pytest -q` exercises crop mapping, gaps, malformed/aligned MSA handling,
process cancellation/timeouts, locking, output validation, cache reuse and output
preservation, in addition to the existing AlphaLens tests.

`tests/chimerax_local_prediction_check.py` is an opt-in real-GPU test for a disposable
ChimeraX session. Set `ALPHALENS_PILOT` to a real LocalFold output directory containing
`fold.pdb`, `fold_confidences.json`, and `fold.a3m`; set `ALPHALENS_TEST_OUTPUT` to a new
directory. It runs three actual predictions, checks Qt responsiveness, imports/plots
results, compares mapped scores, superposes structures, and closes the source during
the final prediction. It exits the test application. `ALPHALENS_REMOTE_MSA=1` also
requests a remote search for the cropped public pilot sequences if uncached.

`tests/chimerax_sequence_prediction_check.py` uses the same environment variables and
public pilot. It checks the Tools-menu entry and `alphalens predict` command, resolves
`RNBR_BACAM` against live UniProt, and runs actual FASTA-pair, raw-monomer and
UniProt-monomer predictions. It reuses matching pilot alignments, verifies absent
monomer ipTM and source comparisons, and checks original UniProt residue numbering and Escape-key cancellation.

## Attribution

LocalFold code is by Sergey Ovchinnikov and is distributed under its
[Beer-Ware license](https://github.com/sokrypton/localfold/blob/68c59196fa7d32c6edfc061ed369889c1cdb680b/LICENSE),
retained beside the installed executable. AlphaFold 2 parameters are by Google DeepMind,
under [CC BY 4.0](https://github.com/google-deepmind/alphafold#model-parameters).
LocalFold supplies the quantized parameter representation and native inference engine;
AlphaLens supplies selection, job management, mapping and visualization.

The bundled AF2-Multimer distogram head comes from `params_model_1_multimer_v3.npz`,
unmodified apart from storing its float32 weights and bias consecutively. Attribution
and the full CC BY 4.0 license are included in `src/weights/`. AF3 parameters and
outputs remain subject to DeepMind's separate AF3 terms linked above.
