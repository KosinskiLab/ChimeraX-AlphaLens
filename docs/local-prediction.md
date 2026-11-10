# Experimental local protein and interface prediction

AlphaLens 0.11.0a2 can predict one or two protein sequences, or re-predict two
selected protein intervals, with AlphaFold2 on
Apple Silicon, using [LocalFold's native Metal implementation](https://github.com/sokrypton/localfold/tree/68c59196fa7d32c6edfc061ed369889c1cdb680b/metal).
The prediction runs in a separate process; ChimeraX remains interactive. The original
structure stays open, and the result gets its own AlphaLens plot.

## Setup

Install this AlphaLens branch in a disposable ChimeraX profile when evaluating it.
The engine requires Apple Silicon, macOS 13 or newer, Git, Python 3 and Xcode Command
Line Tools. It does not require a Python ML environment inside ChimeraX.

From the AlphaLens checkout, run in Terminal:

```sh
python3 scripts/setup_localfold.py
```

The script builds LocalFold at commit `68c59196fa7d32c6edfc061ed369889c1cdb680b`,
installs `~/.local/share/alphalens/bin/localfold-af2`, and records its revision and
binary SHA-256 beside it. `--prefix /path` changes the installation directory;
`--source /path/to/clean/localfold` uses an existing checkout at that exact revision.
The dialog also accepts an explicit executable path (`ALPHALENS_LOCALFOLD` sets its default).
An executable without a matching provenance sidecar is recorded as unverified.

The first prediction for each model automatically downloads LocalFold's public int5
weight bundle, about 77 MB each, into `~/.cache/alphalens/localfold`. These are quantized
AlphaFold parameters, not a new model trained by AlphaLens. The engine uses
`model_1_ptm` for one chain and `model_1_multimer_v3` for two chains, with three recycles
and at most 512 MSA + 1024 extra rows by default.
No AF3 weights are downloaded, and this prototype does not run AF3.

## Predict from sequence, FASTA, or UniProt

Open **Tools → Structure Prediction → AlphaLens Prediction**, or run:

```text
alphalens predict
```

No open structure is needed. Paste one raw amino-acid sequence (line wrapping is
allowed), one/two FASTA entries, or one/two whitespace-separated UniProt identifiers.
Use FASTA headers to separate two sequences; plain wrapped sequence lines form one chain.
For example, paste `P69905 P68871` to retrieve the two human hemoglobin subunits.
Accessions, isoforms such as `P05067-2`, and entry IDs such as `RNBR_BACAM` are supported;
ambiguous gene symbols are not looked up automatically.

Click **Load / resolve sequences**. UniProt retrieval uses its
[official FASTA endpoint](https://github.com/ebi-uniprot/uniprot-manual/blob/main/help/api_retrieve_entries.md)
in a background thread, sending only the identifiers. Review the resolved headers and
sequences. Optional chain A/B ranges are **1-based and inclusive** (e.g. `50-157` for
the tested barnase segment from `P00648`). Ranges are applied before the 400-residue
limit, so longer UniProt proteins can supply a domain. Residue maps
retain positions in the original input sequence. Nonstandard amino acids in the
selected ranges are rejected.

Supply an A3M whose query exactly matches the selected sequence(s), or explicitly
choose remote MSA search. The local MSA cache is checked first. Click **Predict**.
Single chains use the monomer model and report pTM/pLDDT; two chains use multimer and
also report ipTM. Sequence-only jobs open a new PAE plot without inventing a comparison
to a source structure. Monomer outputs also include contact probabilities from this
weight bundle. UniProt provenance records the requested/resolved identifiers, FASTA
header, release, retrieval time, original sequence hash and selected ranges.

## Re-predict a selected interface

1. Open a prediction in AlphaLens and drag a rectangle across an off-diagonal
   interface block. Each axis must cover one protein chain.
2. Right-click the plot and choose **Re-predict selected interface…**.
3. Choose flanks (default 10 residues on each side). The crop preserves contiguous
   sequence intervals, clips at chain ends, and rejects gaps in the source sequence,
   nonstandard amino acids, and more than 400 total residues.
4. Choose an MSA source. **Reuse and crop the source MSA** is selected automatically
   when a matching `<structure-stem>.a3m` or `input.a3m` is found beside the source.
   Its first query must exactly match the plotted protein tokens. Cropping preserves
   existing paired/unpaired rows and their insertion encoding. Alternatively, supply
   a merged A3M for exactly the two cropped sequences, in the displayed order.
5. If no suitable MSA exists, explicitly enable **Search remotely if uncached**.
   This sends the selected sequences to `https://api.colabfold.com`, unless the exact
   query already exists in the local MSA cache. `LOCALFOLD_MMSEQS_API` can select a
   compatible private endpoint; the dialog displays the configured endpoint.
6. Click **Predict**. Use **Cancel job** to stop it. A process-wide file lock serializes
   jobs across AlphaLens windows and instances. The prototype cancels jobs exceeding
   five minutes, including MSA search. Closing a running dialog cancels its job;
   closing the source model does not interrupt an already-started prediction.

A result opens automatically, superposed on the first selected chain's Cα atoms at
job submission. The whole predicted complex moves together, preserving the predicted
interface. Source coordinates are not supplied as a template or restraints.

The dialog reports ipTM and compares mean PAE in both directions over exactly the
selected residue pairs, excluding added flanks. It also compares mean pLDDT over the
same selected residues. All outputs are retained in timestamped job directories under
`~/.local/share/alphalens/predictions` (editable in the dialog):

- `fold.pdb`, `fold_confidences.json`, `fold_summary_confidences.json`;
- `input.json`, `residue_map.json`, and the supplied/cropped/searched A3M;
- `job.json`: model/settings, engine and MSA hashes, source, timing and status;
- `comparison.json` for region jobs, and `prediction.log`.

Residue maps preserve original chain IDs, residue numbers, insertion codes, sequence
positions and matrix rows. Prediction chains are renamed A/B and renumbered from 1.
MSA caches include the ordered sequences, configured endpoint and engine/model version.
Cache hits reuse existing alignments; they do not claim a new database search.

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
accuracy to full-precision AF2, or performance on other GPUs or 400-residue inputs.

The 175-residue crop did **not** improve confidence: mean PAE over the same selected
pairs increased from 2.86 to 3.52 Å (another seed: 4.06 Å). Cropping changes the problem
and can inflate or reduce ipTM. Higher confidence alone does not establish better
structural accuracy. Retain domain context and compare multiple seeds and crop sizes.

The current multimer bundle lacks a distogram/contact-probability head. Multimer results have
PAE, pLDDT and model-distance views; contact probability, spread and contact-based
agreement views are unavailable for that model. Monomer results include contact
probabilities, but still do not export the full distogram required for spread. Ligands, nucleic acids, AF3, mutations, disconnected
sequence grafts and distributed MSA querying are outside this prototype.

## Regression checks

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
retained beside the installed executable. AlphaFold parameters are by Google DeepMind,
under [CC BY 4.0](https://github.com/google-deepmind/alphafold#model-parameters).
LocalFold supplies the quantized parameter representation and native inference engine;
AlphaLens supplies selection, job management, mapping and visualization.
