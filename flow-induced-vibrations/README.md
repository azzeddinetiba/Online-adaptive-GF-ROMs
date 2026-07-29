# Flow-Induced Vibrations

This folder reproduces the flow-induced-vibrations benchmark for the oscillating-cylinder case. The script trains and evaluates global, local, adaptive, and recursive DMDc reduced-order models, then writes the benchmark figures and numerical result arrays to an output directory.

## Setup

Run these commands from this folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The MATLAB dataset is expected at `oscillating_cylinder_benchmark_dataset_v2.mat` by default. If the dataset is stored elsewhere, pass its path with `--dataset-file`.

### Note on numerical reproducibility (macOS arm64)

Results (e.g. GROUSE orthonormality residuals, error curves) were generated with a
**native arm64** Python/numpy/scipy build on Apple Silicon. Running under an x86_64 
build via Rosetta 2 uses a different OpenBLAS kernel (Sandybridge fallback vs. neoversen1) 
and a different SIMD path (SSE-only vs. NEON), which introduces small, architecture-consistent floating-point differences in results on linear-algebra-heavy quantities.

If using conda on Apple Silicon, make sure the environment is pinned to `osx-arm64`:

```bash
CONDA_SUBDIR=osx-arm64 conda create -n venv python=3.13.5
conda activate venv
conda config --env --set subdir osx-arm64
python -m pip install -r requirements.txt
```

## Run

```bash
python reproduce_flow_induced_vibrations.py
```

The script saves figures and arrays to `results_figures_arrays/` by default.

## Arguments

`--dataset-file PATH`

: Path to the MATLAB dataset file. Default: `data/oscillating_cylinder_benchmark_dataset_v2.mat`.

`--output-dir PATH`

: Directory where generated figures and arrays are written. Default: `results_figures_arrays`.

`--skip-projection-errors`

: Skip the expensive orthogonal projection error computation. By default, projection errors are computed.

`--skip-online-angle-comp`

: Intended to skip the computation of the successive subspace rotation angles between each two successive subspaces during the online update.

`--skip-total-angle-comp`

: Intended to skip the computation of the overall subspace rotation angles between each the initial and the current subspace during the online update.

Example:

```bash
python reproduce_flow_induced_vibrations.py \
    --dataset-file /path/to/oscillating_cylinder_benchmark_dataset_v2.mat \
    --output-dir results_figures_arrays
```

## Outputs

The output directory contains the latent-space, projection-error, angle-shift, model-comparison, and basis-mode figures described in the script module docstring, together with the saved numerical arrays.