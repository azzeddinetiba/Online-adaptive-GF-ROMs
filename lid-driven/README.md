# FSI Lid-Driven Cavity

This folder reproduces the FSI lid-driven-cavity benchmark for the available co-simulation time-step cases. The script trains and evaluates the global, local, adaptive, nonlinear, parametric, and rDMDc models used in the paper.

## Setup

Run these commands from this folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The default data root is `./fomData`. The data-loading module is imported from `./dataLoadingScripts/`.

## Run

Run both time-step cases:

```bash
python reproduce_fsi_lid_driven_cavity.py
```

Run one case:

```bash
python reproduce_fsi_lid_driven_cavity.py --case dt03
python reproduce_fsi_lid_driven_cavity.py --case dt01
```

## Arguments

`--data-root PATH`

: Root directory containing the FOM training and test co-simulation data. Default: `./fomData`.

`--output-dir PATH`

: Directory where generated figures and arrays are written. Default: `results_figures_arrays`.

`--case {dt03,dt01,both}`

: Select the co-simulation time-step case. `dt03` runs the `dt = 0.3 s` comparison, `dt01` runs the `dt = 0.1 s` latent-space and mode-shape workflow, and `both` runs both cases. Default: `both`.

`--skip-projection-errors`

: Intended to skip the expensive orthogonal projection error computation for the `dt03` case.

`--skip-regularization-sweep`

: Intended to skip the nonlinear-regression regularization sweep for the `dt03` case and use only the `auto` smoothing setting.

`--skip-online-angle-comp`

: Intended to skip the computation of the successive subspace rotation angles between each two successive subspaces during the online update.

`--skip-total-angle-comp`

: Intended to skip the computation of the overall subspace rotation angles between each the initial and the current subspace during the online update.

The two skip options currently use `store_false` in the script, so their parser values are inverted relative to the option names: the default invocation has both computations disabled, while supplying a skip option enables the corresponding computation. This should be corrected in the script if conventional `--skip-*` behavior is desired.

Example with a custom data and output location:

```bash
python reproduce_fsi_lid_driven_cavity.py \
    --data-root /path/to/fomData \
    --output-dir results_figures_arrays \
    --case both
```

## Outputs

Generated figures and arrays are written to `results_figures_arrays/` by default. The `dt03` case produces the prediction, projection-error, angle-shift, and accuracy-comparison figures. The `dt01` case produces the latent-space, prediction, and force-mode figures.