# Online Adaptive Galerkin-Free ROMs via Grassmann Interpolation and Geodesic Subspace Updates

This repository contains the reproduction scripts and environment definitions for the results presented in:

**Paper: Online adaptive non-intrusive model reduction via manifold interpolation and subspace updates: application to FSI convergence acceleration.**

The repository is organized by benchmark. To reproduce a benchmark, enter its folder, install the dependencies listed in that folder's `requirements.txt`, provide the required data, and run the documented reproduction script.

## Repository layout

```text
flow-induced-vibrations/
    reproduce_flow_induced_vibrations.py
    requirements.txt

lid-driven/
    reproduce_fsi_lid_driven_cavity.py
    requirements.txt

turek-fsi-2/
    reproduce_turek_case.py
    requirements.txt
```

Each benchmark folder contains a README with its command-line arguments and example commands.

## General workflow

From the repository root, choose a benchmark:

```bash
cd flow-induced-vibrations
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce_flow_induced_vibrations.py
```

The same workflow applies to `lid-driven/` and `turek-fsi-2/`. The exact command and data-path options are documented in each folder's README.

The scripts create an output directory named `results_figures_arrays` by default. It contains the figures and numerical arrays generated during reproduction. Use `--output-dir` to select another location.

## Data

Some data-containing folders are intentionally empty (with relative symbolic links) in this public repository because the full simulation data is distributed separately.

The data can be obtained from the following Zenodo repository:

**Zenodo repository:** Online-adaptive-GF-ROMs-Data, https://doi.org/10.5281/zenodo.21728041 .

After downloading the data, place it in the expected folder or pass its location using the corresponding script argument. Do not commit the full data collection to this repository.

## Environments

The benchmark environments were tested with the Python versions and package versions recorded in each folder's `requirements.txt`. The requirements files install the corresponding `rom_am` release directly from GitHub.

For reproducibility, use the requirements file belonging to the benchmark being reproduced. The three benchmarks may use different versions of NumPy, SciPy, scikit-learn, and `rom_am`.

## Citation

- Incoming

## License

[![CC BY-NC-ND 4.0][cc-by-nc-nd-shield]][cc-by-nc-nd]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License][cc-by-nc-nd].

[![CC BY-NC-ND 4.0][cc-by-nc-nd-image]][cc-by-nc-nd]

[cc-by-nc-nd]: http://creativecommons.org/licenses/by-nc-nd/4.0/
[cc-by-nc-nd-image]: https://licensebuttons.net/l/by-nc-nd/4.0/88x31.png
[cc-by-nc-nd-shield]: https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg
