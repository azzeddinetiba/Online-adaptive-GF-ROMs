"""
Reproduce results and figures for the FSI lid-driven-cavity benchmark.

This script trains and evaluates several surrogate models for online interface-force
prediction, for two co-simulation time-step settings (dt = 0.3 s and dt = 0.1 s):

  - Global static basis with linear interpolation (FluidSurrog)
  - Global static basis with nonlinear (cubic) regression, swept over regularizations
  - Global static basis with parametric regression
  - Local static basis with linear interpolation (TrackedFluidSurrog, no basis update)
  - Proposed: Local basis with GROUSE update + linear interpolation
  - Proposed: Local basis with GROUSE update + nonlinear regression, swept over regularizations
  - Recursive DMDc (rDMDc)

Case dt = 0.3 s (--case dt03) reproduces:
  - figReducedPredictionStep_alt_Lid.pdf : Combining local regressions in the latent space
  - figCompareProjErrorsLid.pdf          : Orthogonal projection error comparison
  - combinedAngleShiftLid.pdf            : Evolution of the subspace rotation
  - figCompareFieldsCompleteLid.pdf      : Online relative error comparison (all models)

Case dt = 0.1 s (--case dt01) reproduces:
  - figReducedSnaps3DLid.pdf     : 3-D phase-space view in the latent space (per parameter)
  - figReducedPredictionStepLid.pdf : Combining local regressions in the latent space
  - modes_forces_lid.png         : Interface-force field shapes of selected basis modes

All figures/arrays are saved in the output directory (default: results_figures_arrays/).
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from scipy.ndimage import uniform_filter1d

sys.path.append("./dataLoadingScripts/")

from importData import importData
from rom_am.fluid_surrogate import FluidSurrog
from rom_am.tracked_fluid_surrogate import TrackedFluidSurrog
from rom_am.rdmdc import RDMDC
from rom_am.utils import angles


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reproduce the FSI lid-driven-cavity benchmark results."
    )
    parser.add_argument(
        "--data-root",
        default="./fomData",
        help="Root directory containing the FOM training/test co-simulation data.",
    )
    parser.add_argument(
        "--output-dir",
        default="results_figures_arrays",
        help="Directory where figures and arrays will be written.",
    )
    parser.add_argument(
        "--case",
        choices=["dt03", "dt01", "both"],
        default="both",
        help="Which time-step case to reproduce.",
    )
    parser.add_argument(
        "--skip-projection-errors",
        action="store_true",
        help="Skip the expensive orthogonal projection error computation (dt03 case).",
    )
    parser.add_argument(
        "--skip-regularization-sweep",
        action="store_true",
        help="Skip the regularization sweep for the nonlinear regressors (dt03 case); "
             "uses only 'auto' smoothing instead.",
    )
    parser.add_argument(
        "--skip-online-angle-comp",
        action="store_true",
        help="Skip the computation of the online subspace rotationa angle.",
    )
    parser.add_argument(
        "--skip-total-angle-comp",
        action="store_true",
        help="Skip the computation of the total subspace rotationa angle.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
else:
    args = argparse.Namespace(
        data_root="./fomData",
        output_dir="results_figures_arrays",
        case="both",
        skip_projection_errors=False,
        skip_regularization_sweep=False,
        skip_online_angle_comp=False,
        skip_total_angle_comp=False,
    )

DATA_ROOT = args.data_root
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams["text.latex.preamble"] = " \\usepackage{amsmath}"
plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

LINE_CYCLER = cycler(
    color=["#009E73", "#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#E69F00"]
)
ERROR_LINE_CYCLER = cycler(
    color=["#88CCEE", "#44AA99", "#CC79A7", "#DDCC77", "#332288", "#ffa600", "#C7DD77",
           "#882255", "#AA4499"]
)

PROJ_LINE_CYCLER   = cycler(color=["#E69F00", "#56B4E9", "#F0E442"])


# ---------------------------------------------------------------------------
# Training parameter grid: (f, mu) pairs, encoded as folder name suffixes
# ---------------------------------------------------------------------------
F_MU_CODES = [
    "05mu08", "05mu12", "05mu17",
    "08mu08", "08mu12", "08mu17",
    "12mu08", "12mu12", "12mu17",
    "16mu08", "16mu12", "16mu17",
    "20mu08", "20mu12", "20mu17",
]
PARAM_INS = np.array([
    [0.5, 0.5, 0.5, 0.875, 0.875, 0.875, 1.25, 1.25, 1.25, 1.625, 1.625, 1.625, 2.0, 2.0, 2.0],
    [0.8, 1.25, 1.7, 0.8, 1.25, 1.7, 0.8, 1.25, 1.7, 0.8, 1.25, 1.7, 0.8, 1.25, 1.7],
])
RANK = 8  # max of 99.99% energy criterion-based rank
# This rank is calculated on an a priori basis. Computing it inside
# TrackedFluidSurrog.train() is not yet implemented. It will be soon.
# That would not change the behaviour or computing time.


def build_names(train_subdir):
    """Build the list of training data directories for a given dt sub-folder."""
    return [f"{DATA_ROOT}/{train_subdir}/{code}/coSimData/" for code in F_MU_CODES]


def relative_error(pred, true, true_norms):
    """Column-wise relative L2 error, normalized by precomputed true-field norms."""
    return np.linalg.norm(pred - true, axis=0) / true_norms


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


def smooth(data, window):
    return uniform_filter1d(data, size=window, mode="nearest")


def train_solid_rom(loadData_forSolid, dispConvData_forSolid, dispData_forSolid):
    """Train the auxiliary solid-side reducer used to encode interface displacements."""
    solidROM = FluidSurrog(maxLen=11000, reTrainThres=780)
    solidROM.train(
        loadData_forSolid, dispConvData_forSolid, dispData_forSolid,
        rank_pres=0.999999, rank_disp=20, smoothing=1e-6,
        kernel="polyC", degree=1, norm=[True, True],
        center=[True, True], normalization=["max", "max"],
        norm_regr="max", solidReduc=None,
    )
    return solidROM


# ===========================================================================
# CASE dt = 0.3 s : model comparison (errors, projection errors, subspace angle)
# ===========================================================================
def run_case_dt03():
    print("=" * 70)
    print("Case dt = 0.3 s")
    print("=" * 70)

    names = build_names("train_dt03")
    remove_dts = [94] * 15
    cutoff_incr = [299]
    test_names = [f"{DATA_ROOT}/dt03/10mu10/coSimData/"]
    test_remove_dts = [94]

    reTrainThres = 50
    maxLen = 2100

    # -- Solid-side ROM (shared by all fluid models) --------------------------
    print("Loading training data and training the solid ROM ...")
    (loadData_forFluid, dispData_forFluid, loadConvData_forFluid, size_forFluid,
     loadData_forSolid, dispData_forSolid, dispConvData_forSolid, size_forSolid) = importData(
        names, remove_dts, remove_corners=True, remove_noise=True, cutoff_incr=cutoff_incr,
    )
    solidROM = train_solid_rom(loadData_forSolid, dispConvData_forSolid, dispData_forSolid)

    def load_train_data(return_list):
        return importData(
            names, remove_dts, remove_noise=True, return_list=return_list,
            remove_corners=True, cutoff_incr=cutoff_incr,
        )

    def load_test_data():
        return importData(
            test_names, test_remove_dts, remove_noise=True, remove_corners=True,
            cutoff_incr=cutoff_incr,
        )

    # -- Model: Proposed - Local basis with GROUSE update, linear interpolation
    print("Training Model: Proposed (GROUSE, linear) ...")
    (loadData_forFluid, dispData_forFluid, loadConvData_forFluid, size_forFluid,
     _, _, _, _) = load_train_data(return_list=True)

    updateThres = 120
    t0 = time.time()
    fluidSurr = TrackedFluidSurrog(
        reTrainThres=reTrainThres, maxLen=maxLen,
        updateBasis=True, updateThres=updateThres, updateOmega=True,
    )
    fluidSurr.train(
        dispData_forFluid, loadConvData_forFluid, loadData_forFluid,
        rank_pres=RANK, smoothing="auto", kernel="polyC", degree=1,
        norm=[False, True], norm_regr="max", normalization=["max", "norm"],
        params=PARAM_INS, weights=True, solidReduc=solidROM.reducLoad,
        multiple_param_regressor=True, cleanup=False,
    )
    t1 = time.time()
    print(f"  Training time [Linear regression]: {t1 - t0:.3f} s")

    (loadTestData_Fl, dispTestData_Fl, loadConvTestData_Fl, flTestSize,
     _, _, _, _) = load_test_data()

    # -- Model: Global static basis, linear interpolation ----------------------
    print("Training Model: Global static basis (linear) ...")
    (loadData_forFluid, dispData_forFluid, loadConvData_forFluid, size_forFluid,
     _, _, _, _) = load_train_data(return_list=False)

    fluidSurr2 = FluidSurrog(reTrainThres=reTrainThres, maxLen=maxLen)
    fluidSurr2.train(
        dispData_forFluid, loadConvData_forFluid, loadData_forFluid,
        rank_pres=RANK, smoothing="auto", kernel="polyC", degree=1,
        norm=[False, True], norm_regr="max", normalization=["max", "norm"],
        weights=True, solidReduc=solidROM.reducLoad, multiple_param_regressor=False,
    )

    results2 = np.empty_like(loadTestData_Fl)
    for i in range(dispTestData_Fl.shape[1]):
        results2[:, i] = fluidSurr2.predict(
            dispTestData_Fl[:, [i]], loadConvTestData_Fl[:, [i]], solidReduc=solidROM.reducLoad
        ).ravel()
        fluidSurr2.augmentData(
            dispTestData_Fl[:, [i]], loadConvTestData_Fl[:, [i]], loadTestData_Fl[:, [i]],
            solidReduc=solidROM.reducLoad,
        )

    # -- Model: Local static basis, linear interpolation (no basis update) -----
    print("Training Model: Local static basis (no update) ...")
    (loadData_forFluid, dispData_forFluid, loadConvData_forFluid, size_forFluid,
     _, _, _, _) = load_train_data(return_list=True)

    fluidSurr3 = TrackedFluidSurrog(
        reTrainThres=reTrainThres, maxLen=maxLen, updateBasis=True,
        updateThres=2_200_000, updateOmega=True,
    )
    fluidSurr3.train(
        dispData_forFluid, loadConvData_forFluid, loadData_forFluid,
        rank_pres=RANK, smoothing="auto", hidden_layers=np.array([400, 300, 400, 300, 20]),
        kernel="polyC", degree=1, norm=[False, True], norm_regr="max",
        normalization=["max", "norm"], params=PARAM_INS, weights=True,
        solidReduc=solidROM.reducLoad, multiple_param_regressor=True,
    )
    fluidSurr3.initialize_predictions(np.array([[1.0], [1.0]]))

    (loadTestData_Fl, dispTestData_Fl, loadConvTestData_Fl, flTestSize,
     _, _, _, _) = load_test_data()

    results3 = np.empty_like(loadTestData_Fl)
    for i in range(dispTestData_Fl.shape[1]):
        results3[:, i] = fluidSurr3.predict(
            dispTestData_Fl[:, [i]], loadConvTestData_Fl[:, [i]],
            solidReduc=solidROM.reducLoad, params=np.array([[1.0], [1.0]]),
        ).ravel()
        fluidSurr3.augmentData(
            dispTestData_Fl[:, [i]], loadConvTestData_Fl[:, [i]], loadTestData_Fl[:, [i]],
            params=np.array([[1.0], [1.0]]), solidReduc=solidROM.reducLoad,
        )

    # -- FIGURE : Combining local regressions in the latent space --------------
    print("Generating figReducedPredictionStep_alt_Lid.pdf ...")
    (loadTestData_Fl, dispTestData_Fl, loadConvTestData_Fl, flTestSize,
     _, _, _, _) = load_test_data()

    fluidSurr.initialize_predictions(np.array([[1.0], [1.0]]))

    fig, ax = plt.subplots(1, 2, figsize=(8.3, 3.9), sharey=False)
    ax[0].set_prop_cycle(LINE_CYCLER)
    ax[1].set_prop_cycle(LINE_CYCLER)

    k = 0
    j = 1
    strtIncrems = 15 * [100]
    strtIncremPred = 0

    for i in range(fluidSurr._p):
        visInput = np.vstack((fluidSurr.reducedDispData[i], fluidSurr.reducedPrevLoadData[i]))
        visOutput = fluidSurr.reducedLoadData[i]
        ax[0].plot(visInput[k, strtIncrems[i]:], visOutput[j, strtIncrems[i]:], "-",
                   linewidth=0.8, alpha=0.5, label="${\\boldsymbol{x}}_{" + str(i + 1) + "}^{n, r}$")

    new_input = np.vstack((
        solidROM.reducLoad.encode(dispTestData_Fl, high_dim=False),
        fluidSurr.reducLoad.encode(loadConvTestData_Fl),
    ))
    true_output = fluidSurr.reducLoad.encode(loadTestData_Fl)
    ax[1].plot(new_input[k, strtIncremPred:], true_output[j, strtIncremPred:], "k",
               linewidth=1.0, label="$\\textnormal{True }  \\boldsymbol{x}_{*}^{n, r}$")

    new_preds = np.empty((fluidSurr._p, fluidSurr.reducLoad.latent_dim, dispTestData_Fl.shape[1]))
    for i in range(fluidSurr._p):
        xTest = np.vstack((
            solidROM.reducLoad.encode(dispTestData_Fl, high_dim=False),
            fluidSurr.reducLoadLocals[i].encode(loadConvTestData_Fl, invertModesAccumulated=True),
        ))
        xTestCalibed = xTest.copy()
        xTestCalibed[-fluidSurr.reducLoadLocals[i].pod.kept_rank:, :] = \
            fluidSurr.calibrationQs[i] @ xTestCalibed[-fluidSurr.reducLoadLocals[i].pod.kept_rank:, :]

        new_preds[i, :, :] = fluidSurr.regressor[i].predict(xTest)
        new_preds[i, :] = fluidSurr.calibrationQs[i] @ new_preds[i, :]
        ax[1].plot(xTestCalibed[k, strtIncrems[i]:], new_preds[i, j, strtIncrems[i]:], "-",
                   linewidth=0.7, alpha=0.6, label="$\\widehat{\\boldsymbol{x}}_{" + str(i + 1) + "}^{n, r}$")

    predicted_output = np.dot(new_preds.T, fluidSurr.reducLoad.weights).T
    ax[1].plot(new_input[k, strtIncremPred:], predicted_output[j, strtIncremPred:], "--",
               linewidth=1.0, alpha=1, color="palegoldenrod",
               label="$\\textnormal{Predicted} \\boldsymbol{x}_{*}^{n, r}$")

    ax[0].set_xlabel(r"$" + str(k + 1) + "^{st}$ \\textnormal{Comp. of} $\\boldsymbol{z}^{n, r}_{k}$", fontsize=8)
    ax[1].set_xlabel(r"$" + str(k + 1) + "^{st}$ \\textnormal{Comp. of} $\\boldsymbol{z}^{n, r}_{*}$", fontsize=8)
    ax[0].set_ylabel(r"$" + str(j + 1) + "^{nd}$ \\textnormal{Comp. of} $\\boldsymbol{x}^{n, r}_{k}$", fontsize=8)
    ax[1].legend(loc="best", ncol=4, fontsize=6.5)
    ax[0].legend(loc="best", ncol=4, fontsize=6.5)
    for a in ax:
        a.tick_params(axis="both", which="major", labelsize=8)
        a.grid(alpha=0.3)
    fig.tight_layout(w_pad=0.2)
    fig.savefig(f"{OUTPUT_DIR}/figReducedPredictionStep_alt_Lid.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)

    # -- Online prediction + optional projection-error tracking ---------------
    (loadTestData_Fl, dispTestData_Fl, loadConvTestData_Fl, flTestSize,
     _, _, _, _) = load_test_data()

    computeOnlineAngle = not args.skip_online_angle_comp
    computeTotalAngle = not args.skip_total_angle_comp
    computeProjectionErrs = not args.skip_projection_errors

    firstPredictedBasis = fluidSurr.cloneBasis.copy()
    totalAngle = []

    print("Running online prediction (Proposed - GROUSE, linear) ...")
    results = np.empty_like(loadTestData_Fl)
    relGlobalOrthNorms, relOrthNorms, relOrthNormsNoUpdate = [], [], []
    relRefOrthNorms = np.empty((fluidSurr._p, dispTestData_Fl.shape[1]))

    t0 = time.time()
    for i in range(dispTestData_Fl.shape[1]):
        results[:, i] = fluidSurr.predict(
            dispTestData_Fl[:, [i]], loadConvTestData_Fl[:, [i]],
            solidReduc=solidROM.reducLoad, params=np.array([[1.0], [1.0]]),
        ).ravel()
        trueField = loadTestData_Fl[:, [i]]

        if computeProjectionErrs:
            trueFieldNorm = np.linalg.norm(trueField)
            relGlobalOrthNorms.append(
                np.linalg.norm(trueField - fluidSurr2.reducLoad.decode(
                    fluidSurr2.reducLoad.encode(trueField))) / trueFieldNorm)
            relOrthNormsNoUpdate.append(
                np.linalg.norm(trueField - fluidSurr3.reducLoad.decode(
                    fluidSurr3.reducLoad.encode(trueField))) / trueFieldNorm)
            relOrthNorms.append(
                np.linalg.norm(trueField - fluidSurr.reducLoad.rom.decenter(
                    fluidSurr.reducLoad.rom.denormalize(
                        fluidSurr.cloneBasis @ fluidSurr.cloneBasis.T
                        @ fluidSurr.reducLoad.rom.normalize(
                            fluidSurr.reducLoad.rom.center(trueField)
                        )))) / trueFieldNorm)
            for j_ in range(PARAM_INS.shape[1]):
                relRefOrthNorms[j_, i] = np.linalg.norm(
                    trueField - fluidSurr.reducLoadLocals[j_].decode(
                        fluidSurr.reducLoadLocals[j_].encode(trueField))) / trueFieldNorm

        fluidSurr.augmentData(
            dispTestData_Fl[:, [i]], loadConvTestData_Fl[:, [i]], loadTestData_Fl[:, [i]],
            solidReduc=solidROM.reducLoad, params=np.array([[1.0], [1.0]]),
            stepsize=None, computeAngle=computeOnlineAngle,
        )
        if computeTotalAngle:
            totalAngle.append(angles(firstPredictedBasis, fluidSurr.cloneBasis))

    if args.skip_online_angle_comp and args.skip_total_angle_comp and args.skip_projection_errors:
        # Those three operations are expensive
        print(
            f"Total prediction and update time for the proposed method"
            f" [Linear regression]: "
            f"{time.time() - t0:.3f} s"
        )

    if computeProjectionErrs:
        np.save(f"{OUTPUT_DIR}/relGlobalOrthNorms_lid.npy", np.array(relGlobalOrthNorms))
        np.save(f"{OUTPUT_DIR}/relOrthNormsNoUpdate_lid.npy", np.array(relOrthNormsNoUpdate))
        np.save(f"{OUTPUT_DIR}/relOrthNorms_lid.npy", np.array(relOrthNorms))

        print("Generating figCompareProjErrorsLid.pdf ...")
        fig, ax = plt.subplots(figsize=(6.3, 3.4))
        ax.set_prop_cycle(PROJ_LINE_CYCLER)
        w = 10
        ax.semilogy(smooth(relGlobalOrthNorms, w), label="\\textnormal{Global basis}")
        ax.semilogy(smooth(relOrthNormsNoUpdate, w), label="\\textnormal{Local - w/o update}")
        raw = relOrthNorms
        smoothed = smooth(raw, w)
        ax.semilogy(raw, "k", lw=0.3, alpha=0.35)
        ax.semilogy(smoothed, "k", lw=1.2, label="\\textnormal{Local - Grouse}")
        ax.set_ylabel("\\textnormal{Relative orthogonal projection error}", fontsize=11)
        ax.set_xlabel("\\textnormal{Online steps}", fontsize=11)
        ax.grid()
        ax.legend(fontsize=11, loc="best")
        fig.tight_layout()
        fig.savefig(f"{OUTPUT_DIR}/figCompareProjErrorsLid.pdf", bbox_inches="tight")
        plt.close(fig)
    else:
        print("Skipping projection error computation and figure.")

    if computeOnlineAngle and computeTotalAngle:
        print("Generating combinedAngleShiftLid.pdf ...")
        fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(6.3, 2.4))
        ax1.plot(moving_average(
            np.array([np.linalg.norm(i) for i in fluidSurr.recursive_angles]), 1),
            color="darkslategray", linewidth=0.75)
        ax1.set_xlabel("\\textnormal{Online steps}", fontsize=9)
        ax1.set_ylabel("\\textnormal{Online recursive distance}", fontsize=9, color="darkslategray")
        ax1.tick_params(axis="both", which="major", labelsize=9)
        ax1.tick_params(axis="y", labelcolor="darkslategray")
        ax1.grid(alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(np.array([np.linalg.norm(i[0]) for i in totalAngle]), color="darkcyan", linewidth=1)
        ax2.set_ylabel("\\textnormal{Total distance}", fontsize=9, color="darkcyan")
        ax2.tick_params(axis="y", labelcolor="darkcyan", labelsize=9)

        ax3.plot(np.rad2deg(np.array([i[0].max() for i in totalAngle])),
                 color="darkslategray", linewidth=1)
        ax3.set_xlabel("\\textnormal{Online steps}", fontsize=9)
        ax3.set_ylabel("\\textnormal{Maximum total angle (\N{DEGREE SIGN})}", fontsize=9)
        ax3.tick_params(axis="both", which="major", labelsize=9)
        ax3.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(f"{OUTPUT_DIR}/combinedAngleShiftLid.pdf", bbox_inches="tight")
        plt.close(fig)

    # -- Nonlinear regressors, swept over regularization -----------------------
    regul_alphas = ["auto"] if args.skip_regularization_sweep else [1e-7, 1e-8, 1e-9, "auto"]

    print("Training Model: Proposed (GROUSE, nonlinear), regularization sweep ...")
    errsNlns = []
    training_times = []
    pre_prediction_times = []
    online_times = []
    for k_reg in regul_alphas:
        (loadData_forFluid, dispData_forFluid, loadConvData_forFluid, size_forFluid,
         _, _, _, _) = load_train_data(return_list=True)

        t0= time.time()
        fluidSurrNln = TrackedFluidSurrog(
            reTrainThres=reTrainThres, maxLen=maxLen, updateBasis=True,
            updateThres=updateThres, updateOmega=True,
        )
        fluidSurrNln.train(
            dispData_forFluid, loadConvData_forFluid, loadData_forFluid,
            rank_pres=RANK, smoothing=k_reg, kernel="polyC", degree=2,
            norm=[False, True], norm_regr="max", normalization=["max", "norm"],
            params=PARAM_INS, weights=True, solidReduc=solidROM.reducLoad,
            multiple_param_regressor=True, cleanup=True,
        )
        t1 = time.time()
        training_times.append(t1-t0)


        t0 = time.time()
        fluidSurrNln.initialize_predictions(np.array([[1.0], [1.0]]))
        t1 = time.time()
        pre_prediction_times.append(t1-t0)

        (loadTestData_Fl, dispTestData_Fl, loadConvTestData_Fl, flTestSize,
         _, _, _, _) = load_test_data()

        t0 = time.time()
        resultsNln = np.empty_like(loadTestData_Fl)
        for i in range(dispTestData_Fl.shape[1]):
            resultsNln[:, i] = fluidSurrNln.predict(
                dispTestData_Fl[:, [i]], loadConvTestData_Fl[:, [i]],
                solidReduc=solidROM.reducLoad, params=np.array([[1.0], [1.0]]),
            ).ravel()
            fluidSurrNln.augmentData(
                dispTestData_Fl[:, [i]], loadConvTestData_Fl[:, [i]], loadTestData_Fl[:, [i]],
                solidReduc=solidROM.reducLoad, params=np.array([[1.0], [1.0]]),
                stepsize=None, computeAngle=False,
            )
        t1 = time.time()
        online_times.append(t1-t0)

        trueNorms = np.linalg.norm(loadTestData_Fl, axis=0)
        errsNln = relative_error(resultsNln, loadTestData_Fl, trueNorms)[:-4]
        errsNlns.append(errsNln.copy())

    print(f"  Training time [Nln regression]: {np.mean(training_times):.3f} s")
    print(f"  Pre-prediction time: {np.mean(pre_prediction_times):.3f} s")
    print(
        f"Total prediction and update time for the proposed method"
        f" [Nonlinear regression]: "
        f"{np.mean(online_times):.3f} s"
    )

    errsNlns = np.array(errsNlns)
    mean_TrNln = moving_average(np.mean(errsNlns, axis=0), 3)

    print("Training Model: Global static basis (nonlinear), regularization sweep ...")
    errs2Nlns = []
    for k_reg in regul_alphas:
        (loadData_forFluid, dispData_forFluid, loadConvData_forFluid, size_forFluid,
         _, _, _, _) = load_train_data(return_list=False)

        fluidSurr2Nln = FluidSurrog(reTrainThres=reTrainThres, maxLen=maxLen)
        fluidSurr2Nln.train(
            dispData_forFluid, loadConvData_forFluid, loadData_forFluid,
            rank_pres=RANK, smoothing=k_reg, kernel="polyC", degree=2,
            norm=[False, True], norm_regr="max", normalization=["max", "norm"],
            weights=None, solidReduc=solidROM.reducLoad, multiple_param_regressor=False,
        )

        (loadTestData_Fl, dispTestData_Fl, loadConvTestData_Fl, flTestSize,
         _, _, _, _) = load_test_data()

        results2Nln = np.empty_like(loadTestData_Fl)
        for i in range(dispTestData_Fl.shape[1]):
            results2Nln[:, i] = fluidSurr2Nln.predict(
                dispTestData_Fl[:, [i]], loadConvTestData_Fl[:, [i]], solidReduc=solidROM.reducLoad
            ).ravel()
            fluidSurr2Nln.augmentData(
                dispTestData_Fl[:, [i]], loadConvTestData_Fl[:, [i]], loadTestData_Fl[:, [i]],
                solidReduc=solidROM.reducLoad,
            )

        trueNorms = np.linalg.norm(loadTestData_Fl, axis=0)
        errs2Nln = relative_error(results2Nln, loadTestData_Fl, trueNorms)[:-4]
        errs2Nlns.append(errs2Nln.copy())

    errs2Nlns = np.array(errs2Nlns)
    mean_2Nln = moving_average(np.mean(errs2Nlns, axis=0), 10)

    # -- Model: Global static basis, parametric regression ---------------------
    print("Training Model: Global static basis (parametric) ...")
    (loadData_forFluid, dispData_forFluid, loadConvData_forFluid, size_forFluid,
     _, _, _, _) = load_train_data(return_list=False)
    (loadData_forFluid_list, _, _, _, _, _, _, _) = load_train_data(return_list=True)

    fluidSurr4 = FluidSurrog(reTrainThres=reTrainThres, maxLen=maxLen)
    fluidSurr4.train(
        dispData_forFluid, loadConvData_forFluid, loadData_forFluid,
        rank_pres=RANK, smoothing="auto", kernel="polyC", degree=1,
        norm=[False, True], norm_regr="max", normalization=["max", "norm"],
        weights=True, solidReduc=solidROM.reducLoad, multiple_param_regressor=False,
        params=np.repeat(PARAM_INS, [a.shape[1] for a in loadData_forFluid_list], axis=1),
    )

    (loadTestData_Fl, dispTestData_Fl, loadConvTestData_Fl, flTestSize,
     _, _, _, _) = load_test_data()

    results4 = np.empty_like(loadTestData_Fl)
    for i in range(dispTestData_Fl.shape[1]):
        results4[:, i] = fluidSurr4.predict(
            dispTestData_Fl[:, [i]], loadConvTestData_Fl[:, [i]],
            solidReduc=solidROM.reducLoad, params=np.array([[1.0], [1.0]]),
        ).ravel()
        fluidSurr4.augmentData(
            dispTestData_Fl[:, [i]], loadConvTestData_Fl[:, [i]], loadTestData_Fl[:, [i]],
            params=np.array([[1.0], [1.0]]), solidReduc=solidROM.reducLoad,
        )

    # -- Model: rDMDc baseline --------------------------------------------------
    print("Training rDMDc ...")
    (loadData_forFluid, dispData_forFluid, loadConvData_forFluid, size_forFluid,
     _, _, _, _) = load_train_data(return_list=False)

    fluidSurrRdmd = RDMDC(epsilon=0.01, lambdaForgetBasis=0.95, lambdaForget=0.95)
    fluidSurrRdmd.decompose(
        loadConvData_forFluid[:, :], rank=RANK, Y=loadData_forFluid[:, :],
        u_input=solidROM.reducLoad.encode(dispData_forFluid[:, :]),
        precomp_std=fluidSurr.reducLoad.rom.snap_norms,
        precomp_mean=fluidSurr.reducLoad.rom.mean_flow,
        precomputed_modes=fluidSurr3.reducLoad.pod.modes.copy(), # Pre initialized with 
                                                                 # the interpolated modes
    )

    (loadTestData_Fl, dispTestData_Fl, loadConvTestData_Fl, flTestSize,
     _, _, _, _) = load_test_data()

    t0 = time.time()
    resultsRdmd = np.empty_like(loadTestData_Fl)
    for i in range(dispTestData_Fl.shape[1]):
        resultsRdmd[:, i] = fluidSurrRdmd.predict(
            loadConvTestData_Fl[:, [i]], solidROM.reducLoad.encode(dispTestData_Fl[:, [i]])
        ).ravel()
        fluidSurrRdmd.update(
            loadTestData_Fl[:, [i]], loadConvTestData_Fl[:, [i]],
            solidROM.reducLoad.encode(dispTestData_Fl[:, [i]]),
        )
    print(f"Total prediction and update time for rDMDC: {time.time() - t0:.3f} s")

    # -- FIGURE : Online relative error, all models -----------------------------
    print("Generating figCompareFieldsCompleteLid.pdf ...")
    trueField = loadTestData_Fl.copy()
    trueNorms = np.linalg.norm(trueField, axis=0)

    errs = moving_average(relative_error(results, trueField, trueNorms)[:-4], 3)
    errs2 = moving_average(relative_error(results2, trueField, trueNorms)[:-4], 3)
    errs3 = moving_average(relative_error(results3, trueField, trueNorms)[:-4], 3)
    errs4 = moving_average(relative_error(results4, trueField, trueNorms)[:-4], 3)
    errsRdmd = moving_average(relative_error(resultsRdmd, trueField, trueNorms)[:-4], 3)

    np.save(f"{OUTPUT_DIR}/errs_lid.npy", errs)
    np.save(f"{OUTPUT_DIR}/errs2_lid.npy", errs2)
    np.save(f"{OUTPUT_DIR}/errs3_lid.npy", errs3)
    np.save(f"{OUTPUT_DIR}/errs4_lid.npy", errs4)
    np.save(f"{OUTPUT_DIR}/errsRdmd_lid.npy", errsRdmd)
    np.save(f"{OUTPUT_DIR}/mean_TrNln_lid.npy", mean_TrNln)
    np.save(f"{OUTPUT_DIR}/mean_2Nln_lid.npy", mean_2Nln)

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.set_prop_cycle(ERROR_LINE_CYCLER)
    ax.semilogy(errs2, label="\\textnormal{Global static $\\mathcal{V}$ - Linear $\\mathcal{I}$}", linewidth=1)
    ax.semilogy(mean_2Nln, label="\\textnormal{Global static $\\mathcal{V}$ - Nln $\\mathcal{I}$}", linewidth=1.6)
    ax.semilogy(errs4, label="\\textnormal{Global static $\\mathcal{V}$ - Parametric $\\mathcal{I}$}", linewidth=1)
    ax.semilogy(errsRdmd, label="\\textnormal{rDMDc}", linewidth=1)
    ax.semilogy(errs, label="\\textnormal{Current - Linear $\\mathcal{I}$}", linewidth=1)
    ax.semilogy(errs3, label="\\textnormal{Local static $\\mathcal{V}$ - Linear $\\mathcal{I}$}", linewidth=1)
    ax.semilogy(mean_TrNln, "k", label="\\textnormal{Current - Nln $\\mathcal{I}$}", linewidth=1.2)
    ax.set_xlabel("\\textnormal{Online steps}", fontsize=9)
    ax.set_ylabel("\\textnormal{Relative error} $e^n$", fontsize=9)
    ax.tick_params(axis="both", which="major", labelsize=9)
    ax.tick_params(axis="both", which="minor", labelsize=9)
    ax.legend(shadow=False, fontsize=7.5, ncol=2, loc="best")
    ax.grid()
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/figCompareFieldsCompleteLid.pdf", bbox_inches="tight")
    plt.close(fig)

    print("Case dt = 0.3 s done.\n")


# ===========================================================================
# CASE dt = 0.1 s : latent-space trajectories and force-field mode shapes
# ===========================================================================
def run_case_dt01():
    print("=" * 70)
    print("Case dt = 0.1 s")
    print("=" * 70)

    names = build_names("train_data_010")
    remove_dts = [282] * 15
    cutoff_incr = [897]
    test_names = [f"{DATA_ROOT}/dt01/10mu10/coSimData/"]
    test_remove_dts = [282]

    reTrainThres = 50
    maxLen = 2100
    updateThres = 120

    print("Loading training data and training the solid ROM ...")
    (loadData_forFluid, dispData_forFluid, loadConvData_forFluid, size_forFluid,
     loadData_forSolid, dispData_forSolid, dispConvData_forSolid, size_forSolid) = importData(
        names, remove_dts, remove_corners=True, remove_noise=True, cutoff_incr=cutoff_incr,
    )
    solidROM = train_solid_rom(loadData_forSolid, dispConvData_forSolid, dispData_forSolid)

    print("Training Model: Proposed (GROUSE, linear) ...")
    (loadData_forFluid, dispData_forFluid, loadConvData_forFluid, size_forFluid,
     _, _, _, _) = importData(
        names, remove_dts, remove_noise=True, return_list=True, remove_corners=True,
        cutoff_incr=cutoff_incr,
    )

    fluidSurr = TrackedFluidSurrog(
        reTrainThres=reTrainThres, maxLen=maxLen, updateBasis=True,
        updateThres=updateThres, updateOmega=True,
    )
    fluidSurr.train(
        dispData_forFluid, loadConvData_forFluid, loadData_forFluid,
        rank_pres=RANK, smoothing=1e-4, kernel="polyC", degree=1,
        norm=[False, True], norm_regr="max", normalization=["max", "norm"],
        params=PARAM_INS, weights=True, solidReduc=solidROM.reducLoad,
        multiple_param_regressor=True, cleanup=False,
    )

    # Pre-Prediction phase for the new parameter
    fluidSurr.initialize_predictions(np.array([[1.0], [1.0]]))

    # -- FIGURE : 3-D phase-space view in the latent space, per parameter ------
    print("Generating figReducedSnaps3DLid.pdf ...")
    classical_phase_space = True
    k, p, j = 0, 3, 4
    fig, ax = plt.subplots(3, 3, figsize=(6.3, 6.3), subplot_kw={"projection": "3d"})
    base_elev, base_azim = 23, 120
    line_cycler_local = cycler(color=["#ffa600", "#58508d", "#bc5090", "#ff6361", "#ffa600"])
    strtIncrems = 9 * [2]
    poss = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]

    for i in range(9):
        a = ax[poss[i]]
        a.set_prop_cycle(line_cycler_local)
        a.view_init(elev=base_elev, azim=base_azim)

        if classical_phase_space:
            visInput = fluidSurr.reducedLoadData[i].copy()
        else:
            temp = fluidSurr.reducedPrevLoadData[i].copy()
            visInput = np.vstack((fluidSurr.reducedDispData[i], fluidSurr.calibrationQs[i] @ temp))
        visOutput = fluidSurr.reducedLoadData[i].copy()
        visOutputCalibed = fluidSurr.calibrationQs[i] @ visOutput

        a.plot(visInput[k, strtIncrems[i]:], visInput[p, strtIncrems[i]:], visOutput[j, strtIncrems[i]:],
               "-", linewidth=0.7, alpha=0.8, label=rf"$x_{i + 1}$")
        a.plot(visOutputCalibed[k, strtIncrems[i]:], visOutputCalibed[p, strtIncrems[i]:],
               visOutputCalibed[j, strtIncrems[i]:], "-", linewidth=0.7, alpha=0.8,
               label=rf"$\hat{{x}}_{i + 1}$")

        a.set_xlabel(rf"${k + 1}^{{st}}$ Comp.", fontsize=6, labelpad=-6)
        a.set_ylabel(rf"${p + 1}^{{rd}}$ Comp.", fontsize=6, labelpad=-6)
        a.set_zlabel(rf"${j + 1}^{{rd}}$ Comp.", fontsize=6, labelpad=-6)
        a.tick_params(axis="both", which="major", labelsize=5, pad=-3)
        a.legend(loc="upper left", fontsize=5, frameon=False, bbox_to_anchor=(0.05, 0.95), borderaxespad=0)
        a.grid(False)

    for r in range(3):
        for c in range(3):
            if c != 0:
                ax[r, c].tick_params(left=False)
            if r != 2:
                ax[r, c].tick_params(bottom=False)

    fig.tight_layout(pad=0.5, w_pad=2.0, h_pad=0.3)
    fig.savefig(f"{OUTPUT_DIR}/figReducedSnaps3DLid.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)

    # -- Test data --------------------------------------------------------------
    (loadTestData_Fl, dispTestData_Fl, loadConvTestData_Fl, flTestSize,
     _, _, _, _) = importData(
        test_names, test_remove_dts, remove_noise=True, remove_corners=True, cutoff_incr=cutoff_incr,
    )

    # -- FIGURE : Combining local regressions in the latent space --------------
    print("Generating figReducedPredictionStepLid.pdf ...")

    fig, ax = plt.subplots(1, 2, figsize=(8.3, 3.2), sharey=False)
    ax[0].set_prop_cycle(LINE_CYCLER)
    ax[1].set_prop_cycle(LINE_CYCLER)

    k = 1
    j = 1
    strtIncrems = 15 * [0]
    strtIncremPred = 0

    for i in range(fluidSurr._p):
        visInput = np.vstack((fluidSurr.reducedDispData[i], fluidSurr.reducedPrevLoadData[i]))
        visOutput = fluidSurr.reducedLoadData[i]
        ax[0].plot(visInput[k, strtIncrems[i]:], visOutput[j, strtIncrems[i]:], "-",
                   linewidth=0.8, alpha=0.5, label="${\\boldsymbol{x}}_{" + str(i + 1) + "}^{n, r}$")

    new_input = np.vstack((
        solidROM.reducLoad.encode(dispTestData_Fl, high_dim=False),
        fluidSurr.reducLoad.encode(loadConvTestData_Fl),
    ))
    true_output = fluidSurr.reducLoad.encode(loadTestData_Fl)
    ax[1].plot(new_input[k, strtIncremPred:], true_output[j, strtIncremPred:], "k",
               linewidth=1.0, label="$\\textnormal{True }  \\boldsymbol{x}_{*}^{n, r}$")

    new_preds = np.empty((fluidSurr._p, fluidSurr.reducLoad.latent_dim, dispTestData_Fl.shape[1]))
    for i in range(fluidSurr._p):
        xTest = np.vstack((
            solidROM.reducLoad.encode(dispTestData_Fl, high_dim=False),
            fluidSurr.reducLoadLocals[i].encode(loadConvTestData_Fl, invertModesAccumulated=True),
        ))
        xTestCalibed = xTest.copy()
        xTestCalibed[-fluidSurr.reducLoadLocals[i].pod.kept_rank:, :] = \
            fluidSurr.calibrationQs[i] @ xTestCalibed[-fluidSurr.reducLoadLocals[i].pod.kept_rank:, :]

        new_preds[i, :, :] = fluidSurr.regressor[i].predict(xTest)
        new_preds[i, :] = fluidSurr.calibrationQs[i] @ new_preds[i, :]
        ax[1].plot(xTestCalibed[k, strtIncrems[i]:], new_preds[i, j, strtIncrems[i]:], "-",
                   linewidth=0.7, alpha=0.6, label="$\\widehat{\\boldsymbol{x}}_{" + str(i + 1) + "}^{n, r}$")

    predicted_output = np.dot(new_preds.T, fluidSurr.reducLoad.weights).T
    ax[1].plot(new_input[k, strtIncremPred:], predicted_output[j, strtIncremPred:], "--",
               linewidth=1.0, alpha=1, color="palegoldenrod",
               label="$\\textnormal{Predicted} \\boldsymbol{x}_{*}^{n, r}$")

    ax[0].set_xlabel(r"$" + str(k + 1) + "^{st}$ \\textnormal{Comp. of} $\\boldsymbol{z}^{n, r}_{k}$", fontsize=8)
    ax[1].set_xlabel(r"$" + str(k + 1) + "^{st}$ \\textnormal{Comp. of} $\\boldsymbol{z}^{n, r}_{*}$", fontsize=8)
    ax[0].set_ylabel(r"$" + str(j + 1) + "^{th}$ \\textnormal{Comp. of} $\\boldsymbol{x}^{n, r}_{k}$", fontsize=8)
    ax[1].legend(loc="best", ncol=4, fontsize=6.5)
    ax[0].legend(loc="best", ncol=4, fontsize=6.5)
    ax[0].tick_params(axis="both", which="major", labelsize=8)
    ax[1].tick_params(axis="both", which="major", labelsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].grid(alpha=0.3)
    fig.tight_layout(w_pad=0.2)
    fig.savefig(f"{OUTPUT_DIR}/figReducedPredictionStepLid.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)

    # -- Online run (no projection-error tracking needed for this case) --------
    (loadTestData_Fl, dispTestData_Fl, loadConvTestData_Fl, flTestSize,
     _, _, _, _) = importData(
        test_names, test_remove_dts, remove_noise=True, remove_corners=True, cutoff_incr=cutoff_incr,
    )

    print("Running online prediction ...")
    results = np.empty_like(loadTestData_Fl)
    for i in range(dispTestData_Fl.shape[1]):
        results[:, i] = fluidSurr.predict(
            dispTestData_Fl[:, [i]], loadConvTestData_Fl[:, [i]],
            solidReduc=solidROM.reducLoad, params=np.array([[1.0], [1.0]]),
        ).ravel()
        fluidSurr.augmentData(
            dispTestData_Fl[:, [i]], loadConvTestData_Fl[:, [i]], loadTestData_Fl[:, [i]],
            solidReduc=solidROM.reducLoad, params=np.array([[1.0], [1.0]]),
            stepsize=None, computeAngle=False,
        )

    # -- FIGURE : Interface-force field shapes of selected basis modes ---------
    try:
        from plot_forces import plot_four_forces

        print("Generating modes_forces_lid.png ...")
        coords_path = f"{DATA_ROOT}/train_dt03/05mu08/coSimData/coords_interf.npy"
        n_mode = 2
        which_basis = [1, 9, 15, -1]
        loads_of_modes = [
            np.vstack((
                np.zeros((2, fluidSurr.reducLoad.latent_dim)),
                fluidSurr.reducLoadLocals[idx].pod.modes,
                np.zeros((2, fluidSurr.reducLoad.latent_dim)),
            ))
            for idx in which_basis
        ]
        titles = ["${\\Phi}_{2, 3}$", "${\\Phi}_{8, 3}$", "${\\Phi}_{*, 3}^0$", "$\\widebar{\\Phi}_{*, 3}^{N_t}$"]

        plot_four_forces(
            loads_paths=None,
            coords_path=coords_path,
            vtk_path=Path(f"{DATA_ROOT}/Structure_0_1.vtk"),
            snapshot=n_mode,
            output_path=Path(f"{OUTPUT_DIR}/modes_forces_lid.png"),
            window_size=(2000, 500),
            loads=loads_of_modes,
            titles=titles,
            global_target_factor=0.8,
            cmap="viridis",
        )
    except Exception as exc:  # pragma: no cover - requires pyvista + VTK data
        print(f"  Skipping modes_forces_lid.png (requires pyvista/VTK data): {exc}")

    print("Case dt = 0.1 s done.\n")


if __name__ == "__main__":
    if args.case in ("dt03", "both"):
        run_case_dt03()
    if args.case in ("dt01", "both"):
        run_case_dt01()

    print("All figures/arrays saved to:", OUTPUT_DIR)
    print("Done.")
