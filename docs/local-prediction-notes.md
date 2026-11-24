# Local prediction: notes for developers

The [user guide](local-prediction.md) says how to predict. These notes keep what is behind it:
how the LocalFold engine is pinned, built and checked, what the dialog does in detail, where files
go, the measurements made so far, the regression checks, and the history of fixes. Measurements
are from the AlphaLens version and machine named with them.

## Engine

Local prediction (from AlphaLens 1.5.0) runs [LocalFold's native Metal implementation](https://github.com/KosinskiLab/localfold/tree/837a15ca7c72c35c75739740bc7ed71ee9d5e09b/metal)
on Apple Silicon and its [native CUDA implementation](https://github.com/KosinskiLab/localfold/tree/837a15ca7c72c35c75739740bc7ed71ee9d5e09b/cuda)
on Linux with an NVIDIA GPU, at the same revision. Both take the same options and write the same
output files. The prediction runs in a separate process. The integration uses LocalFold's native
executables, not its browser WebGPU backend; Windows and Intel Macs have no native LocalFold
engine.

### Setup and pinning

Install missing Xcode Command Line Tools with `xcode-select --install`, then retry.
Setup logs are retained in `~/.local/share/alphalens/setup-logs/`. A lock prevents two
installers from building concurrently; another dialog can retry when setup finishes.

Manual setup is optional: download `setup_localfold.py` from the release and run
`python3 ~/Downloads/setup_localfold.py`, or use `python3 scripts/setup_localfold.py`
from a checkout. This builds both engines by default. `--models af2` or `--models af3`
selects one. `--prefix /path` changes the installation directory; `--source /path/to/clean/localfold`
uses a clean checkout at the pinned revision. The release script is the self-contained
`src/predict_setup.py`; `scripts/setup_localfold.py` is a repository entry point.

LocalFold is pinned at `837a15ca7c72c35c75739740bc7ed71ee9d5e09b` of KosinskiLab's fork
([KosinskiLab/localfold](https://github.com/KosinskiLab/localfold), protected branch `alphalens`):
upstream `68c59196fa7d32c6edfc061ed369889c1cdb680b` (`sokrypton/localfold`, branch `opus-55-opt`) plus
one patch, `residueIds` (below). The fork keeps the pinned revision reachable while upstream moves
the native engines into private repositories. Engines built by an earlier AlphaLens are rebuilt
once: the button reads **Update & Predict**. Without `residueIds` the patched engines give the
same structures as upstream's (identical coordinates for AF2 and AF3 in the barnase–barstar check).
Executables go in `~/.local/share/alphalens/bin`, with revision and SHA-256 receipts.

A fresh AF3 engine download and build took **30.4 seconds** on the tested M5 Max,
with Xcode Command Line Tools already installed. The following 195-residue prediction
took 10.6 seconds with cached weights and MSA; the GUI's longest heartbeat gap was
0.27 seconds. An uncached first prediction also downloads model weights and may wait
for a remote MSA search. Those costs are separate from engine setup.
The dialog also accepts an explicit executable path (`ALPHALENS_LOCALFOLD` sets its default).
`ALPHALENS_LOCALFOLD_AF3` sets the AF3 executable default. An executable without a
matching provenance sidecar is recorded as unverified.

### Model parameters

Choose **AlphaFold 2** or **AlphaFold 3** in the model selector. Weights are cached in
`~/.cache/alphalens/localfold`:

- AF2 downloads about 77 MB for each int5 bundle. It uses `model_1_ptm` for one chain
  and `model_1_multimer_v3` for complexes, with three recycles, at most 512 MSA rows
  and 1024 extra rows. AlphaLens includes the matching multimer-v3 float32 distogram
  head (33 KB, CC BY 4.0) in a separate enhanced cache; the original cache is preserved.
- AF3 downloads about 277 MB of LocalFold's int8 parameter representation after you
  check acceptance of [DeepMind's terms](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md)
  and [prohibited-use policy](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_PROHIBITED_USE_POLICY.md).
  Acceptance is remembered locally. Defaults are 10 recycles, 512 MSA rows, 200 diffusion steps, five samples and
  the Euler/Karras sampler. The diffusion step and sample counts are editable.
  The best-ranked sample is opened; the remaining samples stay in the result directory.

These are quantized AlphaFold parameters run by LocalFold, not models trained by
AlphaLens. AF3 weights are not bundled or redistributed in the AlphaLens wheel.

### CUDA engines

Each engine is compiled for the GPU generation `nvidia-smi` reports (its compute capability),
and setup writes the toolkit's library directory into it: predictions run without the CUDA
module loaded. The receipt beside each engine records the compute capability, and AlphaLens
refuses to predict on a GPU of another generation. Built on an A100 (8.0) and run on an H100
(9.0), the pinned engines gave NaN for AF2-Multimer and AF3, and an AF2 monomer 3.4 Å from the
crystal that passed every output check. To switch generations with the same home directory (for
example A100 and H100 nodes), delete `~/.local/share/alphalens/bin/localfold-*` and click
**Install & Predict** on the new GPU.
On a cluster, run setup on a GPU node; on a login node without a GPU, setup stops with
"No NVIDIA GPU found".

At the pinned revision, LocalFold's CUDA build script also builds a featuriser for its design
model (RFD4-Proteina), which AlphaLens does not use. On glibc older than 2.34 (Rocky/RHEL 8,
Ubuntu 20.04) it fails to link, and the build rewrites two committed probe binaries. Setup
restores those files and continues only when that featuriser is the sole failure and every
requested engine was built; any other failure stops setup with the build log. The build logs to
LocalFold's fixed `/tmp/localfold-cuda-build.log`; if another user's copy is in the way,
setup names the file to remove.

Checked on EMBL's cluster on 2026-11-11 (A100-PCIE-40GB, driver 590.48, CUDA 12.8.0, g++ 12.3,
Rocky Linux 8.10, 8 cores). Fresh setup took 125 s for AF2 and 177 s for AF3, excluding weights.
Barnase–barstar (PDB 1BRS chains A/D, 195 residues, cached MSA, seed 1), against the crystal
and against Metal on the M5 Max with the same seed and MSA:

| Case | Cα RMSD to 1BRS | Cα RMSD to Metal | ipTM (Metal) | pLDDT (Metal) | Interface pairs at P ≥ 0.5 (Metal) |
|---|---|---|---|---|---|
| AF2-Multimer with the contact head | 0.640 Å | 0.054 Å | 0.921 (0.922) | 97.39 (97.42) | 46 (46) |
| AF2 monomer (barnase) | 0.379 Å | 0.011 Å | – | 97.44 (97.42) | – |
| AF3, five samples × 200 steps | 0.603 Å | 0.262 Å | 0.93 (0.94) | 95.79 (95.90) | 52 (52) |

With weights cached, the predictions took 3.6, 3.2 and 13.8 s; repeats at the same seed gave
byte-identical structures. Built on an H100 80GB (driver 590.44, the same toolchain), setup took
119 and 170 s, and the same cases gave Cα RMSDs to 1BRS of 0.637, 0.377 and 0.605 Å (0.047,
0.021 and 0.263 Å to Metal), in 2.0, 1.8 and 2.5 s with weights cached. This shows the CUDA engines reproduce the Metal results on one
small known complex; it is not a general accuracy benchmark. The
[recorded check](local-prediction-cuda-check.json) has the per-case values.

## The dialog in detail

Click **Load sequences**. UniProt retrieval uses its
[official FASTA endpoint](https://github.com/ebi-uniprot/uniprot-manual/blob/main/help/api_retrieve_entries.md)
in a background thread, sending only the identifiers. Review the chains in the table (hover
for a sequence). Optional per-chain ranges are **1-based and inclusive** (e.g. `50-157` for
the tested barnase segment from `P00648`); several ranges, `1-100,150-200`, make one chain
with a gap (see [Discontinuous regions](#discontinuous-regions)). Optional ranges select domains from longer UniProt proteins; there is no fixed
residue-count cap. Residue maps
retain positions in the original input sequence. X (an unknown residue) is folded as
AlphaFold's UNK; U, O, B, Z and J are refused, naming the residue and what to use instead
(C for selenocysteine, K for pyrrolysine, X for the ambiguous codes).

Supply an A3M whose query exactly matches the selected sequence(s), use precomputed
AlphaPulldown features (below), or explicitly choose remote MSA search. The local MSA cache
is checked first. Click **Predict**.
With AF2, single chains use the monomer model and complexes use multimer. AF3 handles
both through its own model. Single chains report pTM/pLDDT; complexes also report ipTM.
Sequence-only jobs open a new PAE plot without inventing a comparison to a source
structure. Both models include contact probabilities. UniProt provenance records the requested/resolved identifiers, FASTA
header, release, retrieval time, original sequence hash and selected ranges.

**Predict** is off only while something it needs is missing: a box across two protein chains,
an alignment (and its A3M file or feature IDs), acceptance of the AF3 terms, an engine at the
given path, or the end of a running job. The reason is shown above the buttons, and again when
the pointer rests on Predict.

For a re-predicted region, **Reuse and crop the source MSA** is selected automatically when the
source's own alignment is found beside its structure (`predict_ui.source_alignment`), in this
order:

1. A merged A3M of the whole complex whose first query is exactly the plotted tokens (all of
   them protein): `<structure-stem>.a3m`, `input.a3m`, then any other `*.a3m` in the folder
   (ColabFold's `<job>.a3m`, whose `#lengths` line is skipped). Its columns are cropped;
   existing paired/unpaired rows and their insertion encoding are kept.
2. A native AlphaFold 2 run's `features.pkl` (also `.xz`, `.gz`) beside the model, when the
   chains of `msas/chain_id_map.json` (written by AlphaFold-Multimer; read in the window,
   ordered by AlphaFold's chain IDs) are exactly the plotted sequence. Its `msa` is the merged
   MSA AlphaFold-Multimer used (paired rows first, then each chain's rows with gaps elsewhere;
   residue_constants order; rows padded to 512 are left out by `msa_mask`). The job's process
   crops its columns into `features/merged.a3m` (`--a3m`), as for a merged A3M.
3. An AlphaFold Server download's `msas/` folder: `<job>_unpaired_msa_chains_<id>.a3m` and
   `<job>_paired_msa_chains_<id>.a3m`, one pair a chain, each with the chain's sequence as its
   query (`<job>` is the model's name without `_model_<n>`; identical chains are assumed to
   share one pair, named `chains_a_b`; no such download has been seen yet). A chain's files are
   used when their query is exactly its plotted sequence.
4. An AlphaFold 3 data-pipeline JSON, `<name>_data.json` (or `.json.xz`, `.json.gz`) beside
   `<name>_model.cif`, or in the folder above a sample's `seed-<S>_sample-<K>/model.cif`
   (AlphaFold 3 and AlphaPulldown's AF3 runs). Only its first 64 KB are read in the window, to
   see that it holds MSAs; each chain's `unpairedMsa`/`pairedMsa` (or `...Path`) is read, by
   chain ID, and its sequence checked in the job's process.
5. AlphaPulldown features laid out as AlphaPulldownSnakemake writes them: the model in
   `<out>/predictions/<fold>/`, each protein's feature file (`<name>.pkl[.xz]`,
   `<name>_af3_input.json[.xz]`) in `<out>/features/`. The names come from the fold's name
   (`A+B`, `A_and_B`, `A_homo_2er`; ranges `A:1-50` or `A_1-50` are left out, the features
   being of the whole protein), one a protein chain. This is the precomputed-features route with
   that folder (`msa_source` `AlphaPulldown features in <folder>`). AlphaPulldown's own test data
   (`test/test_data/predictions` beside `features`) has the same layout.

Non-protein chains stop only 1 and 2. In 3 and 4 the paired MSA is the chain's UniProt
hits, not row-aligned with the other chains: AF3 pairs them by species itself, so for AF3 both
are cropped to the predicted columns and sent inline (templates off). For AF2 they are cropped
first, then converted as precomputed features are: the unpaired MSA becomes the chain's
`--a3m`, and the paired one is paired by species (from the UniProt entry names, as AlphaFold's
`msa_identifiers`) into row-aligned `--paired-a3m` files. This runs in the job's process
(`predict_features.py`), so the window does not wait on large MSAs (cropping a 118 MB A3M takes
about 3 s). The manifest's `msa_source` is `AlphaFold Server MSAs in <folder>` or `AlphaFold 3
MSAs in <file>`, and `features` records each chain's files and their SHA-256.

Tested on real predictions of each layout (`tests/chimerax_source_msa_check.py`): barnase–barstar
(1BRS chains, the LocalFold pilot's sequences) folded on the EMBL cluster by AlphaFold 2.3.2
(native, `run_alphafold.py`) and by AlphaPulldown's AF2 backend from the same MSAs
(`/scratch/dima/alphalens-af2-layouts-20261124`), AlphaPulldown's test predictions, lab AF3 runs,
two AlphaFold Server downloads, and the LocalFold pilot. AlphaPulldown runs kept elsewhere than
beside their features fall back to **Precomputed AlphaPulldown features**.

Alternatively, a merged A3M for exactly the cropped sequences, in the displayed order, or
**Precomputed AlphaPulldown features**: the chain IDs are guessed from the model's path
(`Q15014+Q9NV56`, `q15014_and_q9nv56`, AlphaPulldown's `Q15014_af3_input.json_and_..._model.cif`;
one ID for all chains only when they are one protein), and the region is cropped from the
full-length features. When the window shows another model or prediction, the dialog looks for
its MSA again and re-guesses the IDs (unless typed in).

**Search remotely if uncached** sends the selected sequences to `https://api.colabfold.com`, unless
the exact query already exists in the local MSA cache. `LOCALFOLD_MMSEQS_API` can select a
compatible private endpoint; the dialog displays the configured endpoint.

The progress bar shows activity during preparation and remote search, then the number of
completed inference passes for AF2. AF3 uses an activity indicator because its native engine does
not emit incremental diffusion progress. The displayed AF2 fraction covers inference passes, not
total runtime; remote queues have no known percentage or ETA. The bar reaches Complete only after
the validated result opens. A process-wide file lock serializes jobs across AlphaLens windows and
instances. A positive **Time limit** covers the whole job, including MSA search. Closing a
running dialog cancels its job; closing the source model does not interrupt an already-started
prediction.

**Settings** (closed by default; its summary line shows the values) holds recycles, the seed,
AF3 diffusion steps and samples, the time limit, the results folder and the engine path.
**New random seed for each prediction** is enabled by default. Uncheck it
to enter a fixed seed (including zero); the actual seed is shown in results and saved
in the job manifest.

A result opens automatically, superposed on the first selected chain's Cα atoms at
job submission. The whole predicted complex moves together, preserving the predicted
interface. Source coordinates are not supplied as a template or restraints.

### Comparison with the source

CAPRI classes: 0.80 and above high (the same interface), 0.49 medium (similar), 0.23
acceptable (partly the same), below that a different interface. The source is a prediction
too, so this says how alike the two interfaces are, not which is right. The scores match the
official DockQ 2.1.3 on the pairs we checked (two AF2 models of SucC–SucD, AF2 against AF3,
two gapped re-predictions: the same DockQ, Fnat, iRMSD and LRMSD). The ligand RMSD includes
the whole smaller chain, so long disordered tails can make it large while the interface
agrees (Fnat and the interface RMSD show that). With AF3, the heading follows the sample
shown in the Model menu.

**Compare with source** (a button in the dialog, and in the result plot's right-click menu)
superposes the re-prediction on the source as last chosen, shows it colored by chain together
with the same residues of the source in grey, hides the rest of the source and looks at them;
clicking it again restores the colors and display of both. A source is compared with one
re-prediction at a time: comparing another (a new job, or another AF3 sample, which the
comparison follows) first restores the one before. The right-click menu also superposes the re-prediction on the source's
first selected chain (as it opens: the second chain's offset is the change in the interface)
or on the backbone of the source interface (the local differences). Both use the source as it
is if it is open, else as it was when Predict was clicked; the source's heavy atoms are kept
in `residue_map.json` for that (jobs from before 1.5.0 lack them and need the source open).

### Report and job folder

The dialog reports ipTM and compares mean PAE in both directions over exactly the
selected residue pairs, excluding added flanks. It also compares mean pLDDT over the
same selected residues. The contact report gives source/new counts with P ≥ 0.5,
the mean of the ten highest contact probabilities (or all pairs when fewer than ten
exist), the strongest residue pairs, and CB-distance contacts below 8 Å (CA for glycine).
These are residue-pair probabilities, not a probability that the proteins interact.
Source and result distance cutoffs are reported explicitly; missing source probabilities
remain unavailable. Flanks are excluded. All outputs are retained in timestamped job directories under
`~/.local/share/alphalens/predictions` (editable in the dialog):

- `fold.pdb` (and `fold_sample<k>.pdb` for AF3) in the source numbering, `fold_confidences.json`,
  `fold_summary_confidences.json`; `engine_output/` keeps the engine's own structure files;
- `input.json`, `residue_map.json`, and the supplied/cropped/searched A3M;
- `job.json`: model/settings, engine and MSA hashes, source, timing and status;
- `comparison.json` for region jobs (with the DockQ comparison under `interface`), and `prediction.log`;
- `contact_report.json` and `region_contacts.csv` for complexes. The CSV contains the
  union of pairs with source/new P ≥ 0.5 or source/new model distance below 8 Å, with
  original chain/residue labels, both probabilities and cutoffs, and both distances.

Residue maps preserve original chain IDs, residue numbers, insertion codes, sequence
positions and matrix rows. The structures are written in that numbering: the source's chain IDs
and residue numbers for a re-predicted region (so `#1/A:120` and `#2/A:120` are the same residue),
UniProt or input positions for sequence input. The engine numbers chains A, B, … from 1; its files
are kept in `engine_output/`, and `job.json` records the numbering (`numbering`). Source chain IDs
longer than one character, which PDB files cannot hold, are replaced by A, B, … and the log says
so. The current PDB output format supports 62 distinct chain IDs.
MSA caches include the ordered sequences, configured endpoint and the revision of the search
client (unchanged by the `residueIds` patch, so earlier caches stay valid).
AF2 and AF3 share matching cached MSAs, including legacy AF2 caches.
Cache hits reuse existing alignments; they do not claim a new database search.

## Discontinuous regions

AlphaPulldown predicts several regions of one protein (`P1:1-100:150-200`) as **one chain with a
gap**: its AF2 backend keeps the original residue index of each residue, and its AF3 fork takes
the residue numbers (`residueIds`). The model then sees two segments of one chain, far apart in
sequence, rather than a continuous chain (which bonds the ends) or two chains (another molecule).
AlphaLens does the same, through the `residueIds` patch of its LocalFold:

* **Sequence input:** several ranges a chain, `1-100,150-200` (or `1-100:150-200`); ranges must
  follow one another without overlap; adjacent ranges simply join.
* **Re-predicting a box across a gap:** a gap is where the residue numbers jump, a residue is
  unresolved, or a peptide bond is missing (C–N over 2.5 Å). The source numbers become the
  residue index when they jump at every gap (AlphaPulldown and UniProt numbering keep the true
  gap size); otherwise residues are numbered 1, 2, … with the gaps the numbering shows, or 33
  residues when the size is unknown (beyond the relative-position range of 32).
* **Precomputed features** are cropped segment by segment; a reused source MSA by its columns.
* The dialog shows the segments ("A: 98 residues in 2 segments"); `residue_map.json` holds each
  chain's `residue_ids`, which are sent to LocalFold as `residueIds`.

Check (M5 Max, 10 recycles AF3 / 3 AF2, seed 7): barnase–barstar with barnase residues 36–45
cut out. With the gap, the ends (C35–N46) stay 13.8 Å (AF2) and 15.5 Å (AF3) apart, 16 Å
(CA–CA) in the full-length prediction, and the rest of barnase is within 0.33 / 0.41 Å RMSD of
the full-length prediction. Joined into a continuous chain, the ends are bonded (1.45 / 1.34 Å)
and the rest is 1.64 / 1.60 Å away.

## Precomputed AlphaPulldown features

**Precomputed AlphaPulldown features** takes each chain's alignments from a feature file
named by its UniProt ID, so no MSA search runs. **Features from** is the lab database
`/g/alphafold/input_features` (default when `/g` is mounted; AF2 features of 16 proteomes under
`alphafold2/<organism>/features/<ID>.pkl.xz`, AF3 features of two under
`alphafold3/<organism>/features/<ID>_af3_input.json.xz`), its public copy
`https://s3.embl.de/alphapulldown` (default otherwise; only the IDs are sent, files are cached
in `~/.cache/alphalens/features`), or any AlphaPulldown feature folder. `ALPHALENS_FEATURES_DIR`
sets the default.

* AF2 pickles: AlphaPulldown's `MonomericObject`/`ChoppedObject` (its `feature_dict`) or
  AlphaFold's per-chain feature dict, plain or `.xz`/`.gz`/`.bz2`. They are read without
  AlphaPulldown or AlphaFold installed: the unpickler builds numpy arrays and stands in for
  their classes, and refuses anything else. `msa` becomes each chain's unpaired A3M (deletion
  counts as insertions); for a heteromer, `msa_all_seq` is paired by species exactly as
  AlphaFold-Multimer's `msa_pairing.pair_sequences` and `reorder_paired_rows` do (checked
  row for row on lab pairs). LocalFold receives `--a3m=<one a chain>` and
  `--paired-a3m=<one a chain>` and merges, crops and deduplicates them as for a search.
* AF3 JSON (`<ID>_af3_input.json`): used by the AlphaFold 3 model when every chain has one;
  `unpairedMsa` and `pairedMsa` go into the job, and AF3's own species pairing applies.
* A feature file of a whole complex (AF-Multimer `features.pkl` with several `asym_id`s) is
  refused; templates are not used.

The prediction's process finds, reads (or downloads) and converts the features before it
becomes LocalFold, so ChimeraX stays responsive while a file comes over the `/g` mount (a few
seconds each). The job folder keeps `features/chain_*.a3m`, `features/paired_*.a3m` and
`features/features.json` (file, sha256, residue range, row counts). A failure there is
reported as "precomputed features: …" (missing IDs, a sequence that is not in the features,
a damaged read). Feature runs do not add to the MSA cache.

Measured on this Mac (M5 Max), Q15014+Q9NV56 (288 + 204 residues) from the lab database:
features ready in 2 s, the AF2-Multimer prediction in 11 s; ipTM 0.595 and pTM 0.570
reproduce the AlphaPulldown model of the same features (iptm+ptm 0.592), with the same 86
predicted contacts at P ≥ 0.5.

## Large inputs and time limits

There is **no fixed residue-count cap**. Above 1,024 total residues across all chains (including flanks), the dialog shows a non-blocking warning about runtime
and GPU memory. Predict remains enabled. On a Mac, the pinned native engine checks its estimated
allocations against Metal's recommended working-set budget and refuses an input that
does not fit. The CUDA engines size their work to the card's free memory. Removing the count
cap does not remove physical memory requirements.

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
first weight download**, with **5.82 s of inference**. Its files passed the earlier
sequence and finite-value checks, but subsequent testing found collapsed coordinates
from the M5 tensor fast path. This earlier run is an execution timing only and must
not be used as a valid AF3 prediction. Re-predict with 1.1.2 or newer; see the quality
correction below.
See [the recorded model/head benchmark](local-prediction-models-benchmark.json).

## Regression checks

`tests/chimerax_prediction_mapping_check.py` needs no GPU, engine or network. Set
`ALPHALENS_MAPPING_SOURCE` to a prediction folder with contact probabilities and two chains
in contact, and `ALPHALENS_TEST_OUTPUT` to a new folder. A stand-in engine returns the
cropped source structure, rigidly moved, with the source PAE and contact probabilities, so
the reported comparisons must be identical before and after, the prediction must land on
the source, and every CSV row must hold the values of the residues it names. It also covers
both box orientations, a box dragged while the dialog is open, reopening a job folder,
sequence input and a saved session (`ALPHALENS_SOURCE=installed`).

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

`tests/localfold_engine_check.py` needs no ChimeraX and runs on macOS (Metal) and Linux (CUDA).
It runs real AF2-Multimer (with the contact head), AF2-monomer and AF3 predictions of
barnase–barstar through `PredictionJob` with an installed engine, then scores each against the
1BRS crystal (LocalFold's `tools/fixtures/1brs-crystal.pdb`). `--reference` compares each case
with an earlier run, for example on the other platform. `python3 tests/localfold_engine_check.py --help`
lists the inputs; it needs numpy and exits 0 only when every case passes.

## Attribution

LocalFold code is by Sergey Ovchinnikov. Setup installs its LICENSE file beside the executables as
`LocalFold-LICENSE.txt`. AlphaFold 2 parameters are by Google DeepMind, under
[CC BY 4.0](https://github.com/google-deepmind/alphafold#model-parameters).
LocalFold supplies the quantized parameter representation and native inference engine;
AlphaLens supplies selection, job management, mapping and visualization.

The bundled AF2-Multimer distogram head comes from `params_model_1_multimer_v3.npz`,
unmodified apart from storing its float32 weights and bias consecutively. Attribution
and the full CC BY 4.0 license are included in `src/weights/`. AF3 parameters and
outputs remain subject to DeepMind's separate AF3 terms linked above.

## AF3 quality correction in 1.1.2

The M5 tensor fast path in the pinned native engine produced collapsed AF3 coordinates.
Increasing 3 to 10 recycles, 200 to 400 diffusion steps, or 512 to 1024 MSA rows did
not resolve it. AlphaLens now sets `LOCALFOLD_GEMM_TENSOR=0` and
`LOCALFOLD_ATTN_TENSOR=0` for AF3 and records these settings in `job.json`.
AF3 jobs from 1.1.0/1.1.1 on this path should be re-predicted. Their earlier timing
checks demonstrated execution, not structural accuracy.

For a 366-residue user crop at the same seed and MSA, corrected AF3 gives pLDDT 72.0,
ipTM 0.46 and no Cα pairs closer than 2 Å, in 24.7 s with 10 recycles and five
200-step samples. The failed path gave pLDDT 48.1 and hundreds of overlaps. A public
barnase–barstar control gives pLDDT 95.8, ipTM 0.93 and complex Cα RMSD 0.59 Å from
PDB 1BRS without a template. This supports the corrected implementation on these
cases; a high score still does not establish a biologically valid interaction.
The [recorded AF3 quality checks](local-prediction-af3-quality.json) include the
settings grid, backend comparison, geometry counts and cached-MSA timings.

A job folder opens again like any other prediction: `alphalens open <job folder>`
(`rank 1`, `rank 2`, … for the other AF3 samples) or `alphalens open <job folder>/fold.pdb`.

Results now include `prediction_report.txt`. **Prediction report…** in the dialog or
the result plot's context menu reopens the full comparison, including after restarting
ChimeraX. Older reports can be reconstructed from the existing JSON files.

From 1.1.3, the prediction window retains a scrollable history of stage changes and
complete reports across repeated runs. The same prediction messages appear in the
ChimeraX Log. The live stage label shows elapsed time; the alignment source is
recorded once at submission. AF3 uses `NSUnbufferedIO=YES` so its stage output is
available during inference, rather than buffered until the process exits.
