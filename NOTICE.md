# Licensing and attribution

Original code, analysis results, figures and text are dedicated to the public domain under [CC0 1.0](LICENSE), except for the dataset-derived material listed below.

## ManyPeptidesMD

The data source is [ManyPeptidesMD](https://huggingface.co/datasets/transferable-samplers/many-peptides-md), released with Charlie B. Tan et al., [*Amortized Sampling with Transferable Normalizing Flows*](https://proceedings.neurips.cc/paper_files/paper/2025/hash/8780dbea239de8fd1401f9ec6721608d-Abstract-Conference.html), NeurIPS 2025, DOI [10.52202/085713-3154](https://doi.org/10.52202/085713-3154). Its [dataset card](https://huggingface.co/datasets/transferable-samplers/many-peptides-md/raw/main/README.md) specifies the MIT licence but supplies no copyright holder or copyright year.

The MIT notice below is retained for these dataset-derived files:

| Files | Material |
| --- | --- |
| `data/train_tetrapeptides.txt`, `data/test_tetrapeptides.txt` | Sequence lists identifying the selected dataset splits. |
| `model/bundle/bundle.pkl` | Training-derived site definitions, vocabulary, torsion emission bank and baseline energies and rates. |
| `model/bundle/net.pt` | Weights fitted from the dataset. |
| `results/states/*.npz` | All 16 reference-derived state models. |
| `results/dtrajs/*.reference.npz` | All 16 discretised MD reference trajectories. |
| `results/dtrajs/*.model_seed0.npz`, `*.model_seed1.npz`, `*.model_seed2.npz`, `*.model_seed3.npz` in that directory | All 52 discretised model samples using the training-derived emission bank and reference states. |
| `results/dtrajs/*.trade.npz` | All 13 sets of model state paths used for the length comparison. |
| `results/references/*.json` | All 30 reference summaries. |
| `results/landscapes/*.npz` | Reference and model TICA projections, state labels and torsions derived from the dataset and trained emission bank. |
| `figures/gcde-motion.mp4`, `figures/gcde-motion.png`, `figures/conformation-atlas.png` | Visualisations containing dataset-derived conformations. The original rendering work is CC0; the underlying material retains this MIT notice. |

Raw MD trajectories and PDB archives are downloaded separately from the dataset and retain its terms. Newly regenerated copies of the listed material retain the same notice.

### MIT permission notice

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Installed dependencies

Dependencies are installed separately; their source and binaries are not bundled here. Their own notices and any separately licensed components remain applicable.

| Dependency | Licence |
| --- | --- |
| NumPy | [BSD-3-Clause](https://raw.githubusercontent.com/numpy/numpy/main/LICENSE.txt) |
| SciPy | [BSD-3-Clause](https://raw.githubusercontent.com/scipy/scipy/main/LICENSE.txt) |
| PyTorch | [BSD-style licence and component notices](https://raw.githubusercontent.com/pytorch/pytorch/main/LICENSE) |
| deeptime | [LGPL-3.0](https://raw.githubusercontent.com/deeptime-ml/deeptime/main/LICENSE.txt) |
| MDTraj | [LGPL-2.1 or later](https://raw.githubusercontent.com/mdtraj/mdtraj/main/LICENSE) |
| PeptideBuilder | [MIT](https://raw.githubusercontent.com/clauswilke/PeptideBuilder/master/LICENSE.txt) |
| Biopython | [Biopython Licence Agreement](https://raw.githubusercontent.com/biopython/biopython/master/LICENSE.rst); some files additionally permit BSD-3-Clause use. |
| OpenMM, optional for MD timing | [MIT for the public and low-level API, reference and CPU platforms, and application code; LGPL for GPU platforms](https://docs.openmm.org/latest/userguide/library/01_introduction.html#license). Dependencies carry separate terms. |
