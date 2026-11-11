# Licences

Everything written for this repository (code, text, figures, the trained weights and the results) is dedicated to
the public domain under [CC0 1.0](LICENSE).

Four kinds of file are derived from **ManyPeptidesMD**, the public MD corpus the model was trained and scored on
(Tan et al., *Amortized Sampling with Transferable Normalizing Flows*, [arXiv:2508.18175](https://arxiv.org/abs/2508.18175);
[transferable-samplers/many-peptides-md](https://huggingface.co/datasets/transferable-samplers/many-peptides-md),
revision `1af9336`), which is under the MIT licence below:

- the torsion bank inside `models/implicit-310K.npz`: torsion angles of frames of its training trajectories;
- `pepdyn/residues.json`: bond lengths and angles measured on its starting structures;
- `data/reference/`, `data/reference_seed1/`, `data/octapeptides/`: state definitions, state labels and per-site
  statistics of its test and validation trajectories;
- `data/traces/`: torsion angles of 200 ns of one of its test trajectories.

```
MIT License

Copyright (c) 2025 transferable-samplers contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
