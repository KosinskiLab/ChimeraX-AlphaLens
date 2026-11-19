# ChimeraX-AlphaLens

AlphaLens is a [ChimeraX](https://www.cgl.ucsf.edu/chimerax/) tool for inspecting AlphaFold2 and
AlphaFold3 predictions. Like ChimeraX's own AlphaFold error plot, it shows the PAE matrix with a
pLDDT track. It also shows:

* predicted contact probabilities, on the plot and drawn on the structure;
* the contacts in the model compared with the predicted ones;
* how uncertain each predicted distance is (needs the distogram);
* how many AF3 seeds or AF2 models agree on each contact.

It exports contacts to CSV. On Apple Silicon Macs and on Linux with an NVIDIA GPU, it can also
run small predictions locally (experimental).

<p align="center"><img src="docs/images/contacts.png" width="640"
  alt="AlphaLens plot in the contact-probability view, with dots on the inter-chain contacts"></p>

*Contact probabilities for an AlphaFold2 prediction of E. coli SucC–SucD. Dots mark the
inter-chain pairs with P ≥ 0.5.*

## Install

Requires ChimeraX 1.6 or newer (tested with 1.10 and 1.12) on macOS, Linux or Windows.

1. Download `chimerax_alphalens-<version>-py3-none-any.whl` from the
   [Releases](https://github.com/KosinskiLab/AlphaLens/releases) page. Keep the file name.
2. In the ChimeraX command line:

       toolshed install ~/Downloads/chimerax_alphalens-1.5.1-py3-none-any.whl

3. Restart ChimeraX.

`toolshed list` shows the installed version. To update, install the newer wheel in the same way.
To remove AlphaLens, run `toolshed uninstall AlphaLens`. AlphaLens was called DistoPAE up to
version 0.4; if you still have it, run `toolshed uninstall DistoPAE`. The `distopae` command
still works.

Local prediction needs no separate install: the first time you predict, the button reads
**Install & Predict** and sets up the engine. It needs Apple Silicon with macOS 13 or newer and
the Xcode Command Line Tools, or Linux x86_64 with an NVIDIA GPU and the CUDA toolkit. See the
[prediction guide](docs/local-prediction.md#setup).

## Quick start

    alphalens open /path/to/prediction_dir          # top-ranked model
    alphalens open /path/to/prediction_dir rank 2   # AF2 ranked_2, or the AF3 third-best sample
    alphalens #1                                    # a structure that is already open
    alphalens predict                               # local prediction

You can also use **Tools → Structure Prediction → AlphaLens** or **AlphaLens Prediction**.

AlphaLens looks for the PAE file and the distogram next to the structure. It reads output folders
from AlphaFold2, AlphaPulldown, AlphaFold3 and ColabFold. Most pipelines do not keep the
distogram by default. Without it, you get the PAE view and, for AF3, the contact view. Tabs that
need missing data are greyed out, and their tooltip says why. To keep distograms, see
[Getting a distogram](docs/reference.md#getting-a-distogram).

## Views

| View | Shows |
|---|---|
| PAE | predicted aligned error |
| Contact probability | probability that two residues are closer than 8 Å |
| Model contacts | the model's contacts against the predicted ones: green in both, vermillion predicted only, purple in the model only |
| Spread | uncertainty of the predicted distance (needs a distogram) |
| Agreement | in how many AF3 seeds or AF2 models a pair is predicted to be in contact |

<p align="center"><img src="docs/images/views.png" width="760"
  alt="The PAE, contact-probability, model-contact, spread and agreement views of one prediction, and its contacts drawn on the structure"></p>

## Using the plot

* **Hover** over the matrix to see a residue pair's values. **← / →** switch views.
* **Zoom** with the scroll wheel or a pinch. **Pan** with Shift-drag. **Fit** or Esc resets the view.
* The **contact probability slider** sets the threshold for the dots, the 3D contacts, the
  agreement view and the CSV export.
* **Show Contacts** draws the predicted contacts on the structure.
* **Click a dot** to show that contact in 3D. Its values appear in the log.
* **Drag a box** to keep only the contacts inside it, for example between two chains.
* **Model** switches between the AF2 models or AF3 samples of the prediction.
* The **hide** slider next to the pLDDT track hides low-pLDDT parts. To hide a region, drag along
  the pLDDT track, then right-click and choose **Hide selected**.
* **Color pLDDT** and **Color PAE Domains** color the structure.
* **Export CSV** saves the contacts above the threshold.
* The **right-click menu** has more options, such as color scales and Save image.
* `save mysession.cxs` saves the plot with the session.

## Local prediction (experimental)

1. Run `alphalens predict`.
2. Paste sequences, FASTA, UniProt IDs or AlphaPulldown notation, such as
   `P0A836:1-100+P0AGE9`.
3. Choose AlphaFold 2 or AlphaFold 3.
4. Choose an alignment: an existing MSA, precomputed AlphaPulldown features, or a remote
   ColabFold search. The remote search sends your sequences to ColabFold.
5. Click **Predict**. The result opens in a new plot.

To re-predict part of a complex, drag a box across an interface in the plot, then right-click
and choose **Re-predict selected interface…**. The report compares the new prediction with the
source, including DockQ.

## More

* [Reference](docs/reference.md): every view and control, CSV columns, sessions, command options,
  supported layouts, limitations
* [Local prediction guide](docs/local-prediction.md)
* [Changelog](CHANGELOG.md)
* [Development](docs/development.md): installing from source and running the tests

## License

MIT; see [LICENSE](LICENSE). Bug reports and questions go to the
[issue tracker](https://github.com/KosinskiLab/AlphaLens/issues).
