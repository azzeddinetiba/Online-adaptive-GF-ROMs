# Turek FSI

This folder reproduces the Turek FSI benchmark. The reproduction script trains and evaluates global static, local static, adaptive GROUSE, nonlinear, parametric, and recursive DMDc models for online interface-force prediction.

## Setup

Run these commands from this folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The script loads the data through an `importData` module. Its current default loader path is `../trainScriptForTrack/`, while this repository contains a `trainScripts/` folder. Pass `--scripts-root` if the loader is located in the checked-in folder or elsewhere. The default data root is `../trainDataForTrack`; adjust it with `--data-root` if the downloaded data is stored elsewhere.

## Run

```bash
python reproduce_turek_case.py
```

## Arguments

`--data-root PATH`

: Root directory containing the FOM training and test co-simulation data. Default: `../trainDataForTrack`.

`--scripts-root PATH`

: Directory containing the `importData` data-loading module. Default: `../trainScriptForTrack/`.

`--output-dir PATH`

: Directory where all generated figures and arrays are written. Default: `results_figures_arrays`.

`--skip-projection-errors`

: Skip the expensive orthogonal projection-error and subspace-angle tracking calculations. By default, these calculations are performed.

`--skip-online-angle-comp`

: Intended to skip the computation of the successive subspace rotation angles between each two successive subspaces during the online update.

`--skip-total-angle-comp`

: Intended to skip the computation of the overall subspace rotation angles between each the initial and the current subspace during the online update.

Example with downloaded data in custom locations:

```bash
python reproduce_turek_case.py \
    --data-root /path/to/trainDataForTrack \
    --scripts-root /path/to/trainScriptForTrack \
    --output-dir results_figures_arrays
```

To reduce runtime when only the model-accuracy comparison is needed:

```bash
python reproduce_turek_case.py --skip-projection-errors
```

## Outputs

All figures and saved arrays are placed in the one output directory. The script produces latent-space trajectory figures, the latent prediction-step figure, optional projection-error and angle-shift figures, the final model-accuracy comparison, and the corresponding numerical arrays.