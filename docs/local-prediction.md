# Local prediction

AlphaLens can predict a protein complex from sequences or UniProt IDs, or re-predict an
interface you select on the plot, with AlphaFold 2 or AlphaFold 3 on your own computer: an Apple
Silicon Mac, or Linux with an NVIDIA GPU. The prediction runs in the background, so ChimeraX stays
responsive, and the result opens in an AlphaLens window of its own. Windows and Intel Macs are not
supported yet.

Predictions run [LocalFold](https://github.com/sokrypton/localfold) by Sergey Ovchinnikov, with
quantized copies of the AlphaFold parameters, so its models can differ slightly from those of
AlphaFold itself. A confident score is not proof of an interaction.

## Setup

You need:

* **Mac:** Apple Silicon, macOS 13 or newer, and the Xcode Command Line Tools (install them with
  `xcode-select --install`).
* **Linux:** x86_64 with an NVIDIA GPU from the T4 on (Turing, Ampere, Ada, Hopper, Blackwell) and
  its driver; while setup runs, the CUDA 12 toolkit (`nvcc`, cuBLAS; 12.8 or newer for Blackwell)
  and `g++` on the `PATH` (on a cluster, load the CUDA module before starting ChimeraX); `git`,
  `curl`, `gzip`, `sha256sum`, and glibc 2.28 or newer.

No Python environment is needed. The first time, the button in the prediction dialog reads
**Install & Predict**: it downloads and builds the prediction engine in the background, then
predicts. **Cancel setup** stops it; after an error, click the button again. **Update & Predict**
means that a new AlphaLens needs a newer engine; it is rebuilt once.

On a cluster, run the first prediction on a GPU node. The engine is built for the GPU it was set
up on: on a GPU of another generation (A100 and H100 nodes sharing a home folder, for example)
AlphaLens refuses to predict and says how to build it again.

The first prediction with each AlphaFold version also downloads its parameters, about 80 MB for
AlphaFold 2 and 280 MB for AlphaFold 3, and keeps them for later. AlphaFold 3 parameters are
downloaded only after you accept Google DeepMind's
[terms of use](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md)
and [prohibited use policy](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_PROHIBITED_USE_POLICY.md)
in the dialog.

## Predict from sequences, FASTA or UniProt IDs

1. Choose **Tools → Structure Prediction → AlphaLens Prediction**, or run `alphalens predict`. No
   structure needs to be open.
2. Paste one of:
    * a sequence (line breaks are fine);
    * FASTA entries, one chain each;
    * UniProt IDs separated by spaces: accessions (`P69905`), isoforms (`P05067-2`) or entry
      names (`RNBR_BACAM`). `P69905 P68871` gives the two subunits of human hemoglobin;
    * AlphaPulldown notation: `P0A836:1-100+P0AGE9` (a region of one protein and a whole second
      one), `P69905:2` (two copies), `Q8I2G6:2:1-100:150-200` (two copies of two regions).
3. Click **Load sequences** and check the chains in the table (hover for the sequence). A range
   such as `50-157` (first and last residue, counted from 1) predicts part of a chain;
   `1-100,150-200` predicts two regions as one chain with a gap (see
   [Regions with a gap](#regions-with-a-gap)).
4. Choose **AlphaFold 2** or **AlphaFold 3**.
5. Choose an alignment: an A3M file whose first sequence is exactly the chosen sequences,
   [precomputed AlphaPulldown features](#precomputed-alphapulldown-features), or
   **Search remotely if uncached**, which sends the sequences to the ColabFold MSA server.
   Alignments are kept, so the same sequences are not searched twice.
6. Click **Predict**.

UniProt IDs are looked up at UniProt; only the IDs are sent. A residue written X is predicted as
an unknown residue. U, O, B, Z and J are refused, with what to write instead.

## Re-predict a selected interface

1. In an AlphaLens window, drag a box across the interface between two chains (a block off the
   diagonal). Either side of the box can span several chains.
2. Right-click the plot and choose **Re-predict selected interface…**.
3. Choose the flanks, the residues added on each side of the box (10 by default; they stop at the
   ends of chains and at gaps).
4. Choose the alignment. **Reuse and crop the source MSA** is chosen for you when the MSA the
   prediction was made with is at hand:
   * an A3M of all its chains beside the model (LocalFold, ColabFold);
   * the `features.pkl` of an AlphaFold 2 run;
   * the `msas` folder of an AlphaFold Server download;
   * the `<name>_data.json` of an AlphaFold 3 run (also compressed, `.json.xz`);
   * for AlphaPulldown, the proteins' features in the `features` folder beside `predictions`, as
     AlphaPulldownSnakemake writes them.

   You can also give an A3M for exactly the cropped sequences, use precomputed AlphaPulldown
   features from elsewhere (the protein IDs come from the folder name, such as
   `Q15014+Q9NV56`), or search remotely.
5. Click **Predict**. When Predict is greyed out, the line above the buttons says what is missing.

The result opens superposed on the source's first selected chain; the source is not used as a
template. Only proteins are predicted: ligands and nucleic acids in the box are left out, and a box
with a modified residue is refused.

A crop changes the problem: confidence can go up or down without the structure getting better.
Keep whole domains in the box, and compare a few seeds and crop sizes.

### Is it the same interface?

The report, the dialog and the heading of the result's window compare the new interface with the
source by DockQ (Basu & Wallner 2016), for each pair of chains:

```text
Interface vs source: DockQ 0.92 · high: same interface
  A–B: 44 of 46 source contacts kept (Fnat 0.96), 3 new; interface RMSD 0.7 Å; B 1.3 Å from its source place with A superposed
```

DockQ of 0.80 and above is high (the same interface), 0.49 medium (similar), 0.23 acceptable
(partly the same); below that the interface is a different one. The source is a prediction too, so
DockQ says how alike the two interfaces are, not which one is right. Long disordered tails can
make the last distance large while the interface agrees; the contacts kept (Fnat) and the
interface RMSD show that.

The report also gives ipTM, the mean PAE and pLDDT over the selected residues (without the flanks)
in the source and in the new prediction, and their contacts with P ≥ 0.5, strongest first. These
are probabilities that two residues are in contact, not that the proteins interact.

**Compare with source**, in the dialog and in the result's right-click menu, shows the
re-prediction colored by chain with the same residues of the source in grey, and hides the rest of
the source; click it again to restore both. The right-click menu can also superpose the
re-prediction on the source's first selected chain, or on the backbone of the source interface.

## Regions with a gap

Two regions of one protein, `1-100,150-200`, are predicted as one chain with a gap, as AlphaPulldown
does with `P1:1-100:150-200`: AlphaFold knows that the two segments are far apart in the sequence,
instead of joining them or treating them as two molecules. A box across a gap in a model (an
AlphaPulldown region, or missing residues) keeps the gap. The dialog lists the segments, for
example "A: 98 residues in 2 segments".

## Precomputed AlphaPulldown features

**Precomputed AlphaPulldown features** takes each chain's alignments from an AlphaPulldown feature
file named by its UniProt ID, so no alignment search runs. **Features from** is the public
AlphaPulldown feature database at EMBL, with AlphaFold 2 features for 16 proteomes and AlphaFold 3
features for two (only the IDs are sent, and the files are kept), or a folder of your own
AlphaPulldown features. AlphaFold 3 uses the `<ID>_af3_input.json` files when every chain has one.
Templates are not used, and the features of a whole complex are refused.

## Settings, large inputs and time limits

**Settings** (its summary line shows the values) holds the recycles, the seed, the AlphaFold 3
diffusion steps and samples, a time limit, the results folder and the engine. Each prediction gets
a new random seed unless you untick **New random seed for each prediction** and give one.

There is no size limit. Above 1,024 residues in all, the dialog warns that the prediction may be
slow or may not fit in the GPU's memory; you can still predict. The time limit (none by default)
covers the whole job, including the alignment search. **Cancel job**, or closing the dialog, stops a
prediction. A remote alignment search can wait in a queue for minutes; reusing an alignment is
fast.

## Results

Each prediction is saved in a folder of its own under the results folder (by default
`~/.local/share/alphalens/predictions`): the structure `fold.pdb` (and `fold_sample<k>.pdb` for the
other AlphaFold 3 samples), its confidences, the alignment, `region_contacts.csv` with the contacts
of a complex, and `prediction_report.txt`. Residues keep the numbering of the input: the source's
chains and residue numbers for a re-predicted interface, so `#1/A:120` and `#2/A:120` are the same
residue, and UniProt numbering for UniProt IDs.

The result's window has every view except Spread, which needs the full distogram. Open a result
again like any prediction, `alphalens open <folder>` (`rank 1`, `rank 2`, … for the other AF3
samples). **Prediction report…**, in the dialog and in the result's right-click menu, shows the
report again. The dialog keeps a history of its messages and reports, which also go to the log.

## Credits

Local prediction runs [LocalFold](https://github.com/sokrypton/localfold) by Sergey Ovchinnikov.
The AlphaFold 2 parameters are by Google DeepMind, under
[CC BY 4.0](https://github.com/google-deepmind/alphafold#model-parameters); AlphaLens includes the
AlphaFold 2 multimer distogram head, so that predictions have contact probabilities. The
AlphaFold 3 parameters are under Google DeepMind's terms of use, linked above; AlphaLens does not
include them. Remote alignment searches use the [ColabFold](https://github.com/sokrypton/ColabFold)
MMseqs2 server.
