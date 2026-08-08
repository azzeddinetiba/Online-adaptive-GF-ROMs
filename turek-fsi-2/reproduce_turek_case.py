"""
Reproduce results and figures for the Turek FSI benchmark (fsi_turek_2 test case).

This script trains and evaluates several surrogate models for online interface-force
prediction on the Turek FSI benchmark:

  - Global static basis with linear interpolation (FluidSurrog)
  - Global static basis with nonlinear (quadratic) regression
  - Global static basis with parametric regression
  - Local static basis with linear interpolation (TrackedFluidSurrog, no basis update)
  - Proposed: Local basis with GROUSE update + linear interpolation
  - Proposed: Local basis with GROUSE update + nonlinear regression
  - Recursive DMDc (rDMDc)

Figures generated:
  - turekFigReducedSnaps.pdf         : Phase-space view in the latent space (before/after calibration)
  - turekFigReducedSnaps3D.pdf       : 3-D phase-space view in the latent space
  - figReducedPredictionStepTurek.pdf : Combining local regressions in the latent space
  - figCompareProjErrorsTurek.pdf    : Orthogonal projection error comparison
  - combinedAngleShiftTurek.pdf      : Evolution of the subspace rotation
  - figTurekCompareAccuracies.pdf    : Online relative error comparison (all models)

All figures/arrays are saved in the output directory (default: results_figures_arrays/).
"""

import argparse
import copy
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from scipy.ndimage import uniform_filter1d
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reproduce the Turek FSI benchmark results."
    )
    parser.add_argument(
        "--data-root",
        default="./trainData",
        help="Root directory containing the FOM training/test co-simulation data.",
    )
    parser.add_argument(
        "--scripts-root",
        default="./trainScripts/",
        help="Directory containing the importData data-loading module.",
    )
    parser.add_argument(
        "--output-dir",
        default="results_figures_arrays",
        help="Directory where figures and arrays will be written.",
    )
    parser.add_argument(
        "--skip-projection-errors",
        action="store_true",
        help="Skip the expensive orthogonal projection error / subspace angle tracking.",
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
        data_root="./trainData",
        scripts_root="./trainScripts/",
        output_dir="results_figures_arrays",
        skip_projection_errors=False,
        skip_online_angle_comp=False,
        skip_total_angle_comp=False,
    )

sys.path.append(args.scripts_root)

from importData import importData
from rom_am.fluid_surrogate import FluidSurrog
from rom_am.tracked_fluid_surrogate import TrackedFluidSurrog
from rom_am.rdmdc import RDMDC
from rom_am.utils import angles

DATA_ROOT = args.data_root
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})
plt.rcParams["text.latex.preamble"] = " \\usepackage{amsmath}"

LINE_CYCLER = cycler(
    color=["#009E73", "#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#E69F00"]
)
ERROR_LINE_CYCLER = cycler(
    color=["#88CCEE", "#44AA99", "#CC79A7", "#DDCC77", "#332288", "#ffa600", "#CC6677",
           "#882255", "#AA4499"]
)
PROJ_LINE_CYCLER = cycler(color=["#E69F00", "#56B4E9", "#F0E442"])
LOCAL_LINE_CYCLER = cycler(color=["#ffa600", "#58508d", "#bc5090", "#ff6361", "#ffa600"])

# ---------------------------------------------------------------------------
# Training parameter grid: (velocity, density-ratio) pairs
# ---------------------------------------------------------------------------
DT_NAME = "dt01"
DT = 0.01
REMOVE_DTS_SINGLE = int(8 / 0.008)
CUTOFF_INCR = int(20 / DT) - 1
RANK = 18  # max of 99.99% energy criterion-based rank
# This rank is calculated on an a priori basis. Computing it inside
# TrackedFluidSurrog.train() is not yet implemented. It will be soon.
# That would not change the behaviour or computing time.

TRAIN_CODES = ["vel0-9rho0-9", "vel0-9rho1-1", "vel1-1rho1-1", "vel1-1rho0-9"]
PARAM_INS = np.array([
    [0.9, 0.9, 1.1, 1.1],
    [0.9, 1.1, 1.1, 0.9],
])
TEST_CODE = "vel1-0rho1-0"


def build_names(codes):
    return [f"{DATA_ROOT}/{DT_NAME}/{code}/coSimData/" for code in codes]


def relative_error(pred, true, true_norms):
    """Column-wise relative L2 error, normalized by precomputed true-field norms."""
    return np.linalg.norm(pred - true, axis=0) / true_norms


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


def smooth(data, window):
    return uniform_filter1d(data, size=window, mode="nearest")


def load_train_data(names, remove_dts, return_list):
    return importData(
        names, remove_dts, remove_noise=True, return_list=return_list,
        remove_corners=True, cutoff_incr=[CUTOFF_INCR],
    )


def load_test_data():
    return importData(
        [f"{DATA_ROOT}/{DT_NAME}/{TEST_CODE}/coSimData/"], [REMOVE_DTS_SINGLE],
        remove_noise=True, remove_corners=True, cutoff_incr=[CUTOFF_INCR],
    )


print("dt =", DT)
names = build_names(TRAIN_CODES)
remove_dts = [REMOVE_DTS_SINGLE] * len(TRAIN_CODES)

# ===========================================================================
# 1.  Solid-side ROM (shared by all fluid models)
# ===========================================================================
print("Training the solid ROM ...")
(_, _, _, _, loadData_forSolid, dispData_forSolid,
 dispConvData_forSolid, size_forSolid) = importData(
    names, remove_dts, remove_corners=True, remove_noise=True, cutoff_incr=[CUTOFF_INCR],
)
solidROM = FluidSurrog(maxLen=11000, reTrainThres=780)
solidROM.train(
    loadData_forSolid, dispConvData_forSolid, dispData_forSolid,
    rank_pres=RANK, rank_disp=60, smoothing=1e-6,
    kernel="polyC", degree=1, norm=[True, True],
    center=[True, True], normalization=["max", "max"],
    norm_regr="max", solidReduc=None,
)
print(f"  reduced from {solidROM.reducLoad.high_dim} to {solidROM.reducLoad.latent_dim}")

solidReducSaved = copy.deepcopy(solidROM.reducLoad)
del loadData_forSolid, dispData_forSolid, dispConvData_forSolid

# ===========================================================================
# 2.  Model: Proposed - Local basis with GROUSE update, linear interpolation
# ===========================================================================
print("Training Model: Proposed (GROUSE, linear) ...")
loadData_forFluid, dispData_forFluid, loadConvData_forFluid, size_forFluid, _, _, _, _ = \
    load_train_data(names, remove_dts, return_list=True)

reTrainThres = 60
maxLen = 11000
updateThres = 200

fluidSurr = TrackedFluidSurrog(
    reTrainThres=reTrainThres, maxLen=maxLen,
    updateBasis=True, updateThres=updateThres, updateOmega=True,
)
t0 = time.time()
fluidSurr.train(
    dispData_forFluid, loadConvData_forFluid, loadData_forFluid,
    rank_pres=RANK, smoothing=1e-5, kernel="polyC", degree=1,
    norm=[False, True], norm_regr=None, normalization=["max", "norm"],
    params=PARAM_INS, weights=True, solidReduc=solidReducSaved,
    multiple_param_regressor=True, cleanup=False,
)
print(f"  Training time [Linear regression]: {time.time() - t0:.3f} s")

del loadData_forFluid, dispData_forFluid, loadConvData_forFluid

t0 = time.time()
fluidSurr.initialize_predictions(np.array([[1.0], [1.0]]))
print(f"  Pre-prediction time: {time.time() - t0:.3f} s")


# ===========================================================================
# 3.  Refresh the solid ROM against the trained fluid reducer
# ===========================================================================
_, _, _, _, loadData_forSolid, dispData_forSolid, dispConvData_forSolid, size_forSolid = \
    importData(names, remove_dts, remove_corners=True, remove_noise=True, cutoff_incr=[CUTOFF_INCR])

solidROM = FluidSurrog(maxLen=11000, reTrainThres=780)
solidROM.train(
    loadData_forSolid, dispConvData_forSolid, dispData_forSolid,
    rank_pres=20, rank_disp=60, smoothing=1e-6,
    kernel="polyC", degree=1, norm=[True, True],
    center=[True, True], normalization=["max", "max"],
    norm_regr="max", solidReduc=fluidSurr.reducLoad, precomputedReducLoad=solidReducSaved,
)
print(f"  reduced from {solidROM.reducLoad.high_dim} to {solidROM.reducLoad.latent_dim}")
del loadData_forSolid, dispData_forSolid, dispConvData_forSolid

# ===========================================================================
# 4.  FIGURE : Phase-space view in the latent space (2-D)
# ===========================================================================
print("Generating turekFigReducedSnaps.pdf ...")
classical_phase_space = True

fig, ax = plt.subplots(2, 2, figsize=(9, 7.5), sharex="col", sharey="row")
k, j = 0, 1
strtIncrems = [10, 10, 10, 10]
poss = [(0, 0), (0, 1), (1, 0), (1, 1)]

for i in range(fluidSurr._p):
    ax[poss[i]].set_prop_cycle(LOCAL_LINE_CYCLER)

    if classical_phase_space:
        visInput = fluidSurr.reducedLoadData[i].copy()
    else:
        temp = fluidSurr.reducedPrevLoadData[i].copy()
        visInput = np.vstack((fluidSurr.reducedDispData[i], fluidSurr.calibrationQs[i] @ temp))

    visOutput = fluidSurr.reducedLoadData[i].copy()
    visOutputCalibed = fluidSurr.calibrationQs[i] @ visOutput

    ax[poss[i]].plot(visInput[k, strtIncrems[i]:], visOutput[j, strtIncrems[i]:], "-",
                     linewidth=0.8, alpha=0.8, label=r"$\boldsymbol{f}_" + str(i) + "^r$")
    ax[poss[i]].plot(visInput[k, strtIncrems[i]:], visOutputCalibed[j, strtIncrems[i]:], "-",
                     linewidth=0.8, alpha=0.8, label=r"$\widehat{\boldsymbol{f}}_" + str(i) + "^r$")
    ax[poss[i]].legend(loc="best")

    if i in (0, 1):
        ax[poss[i]].tick_params(bottom=False)
    else:
        ax[poss[i]].set_xlabel(r" $" + str(k + 1) + "^{st}$ \\textnormal{Component}", fontsize=12)
    if i in (1, 3):
        ax[poss[i]].tick_params(left=False)
    else:
        ax[poss[i]].set_ylabel(r" $" + str(j + 1) + "^{st}$ \\textnormal{Component}", fontsize=12)

fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/turekFigReducedSnaps.pdf", bbox_inches="tight")
plt.close(fig)

# ===========================================================================
# 5.  FIGURE : Phase-space view in the latent space (3-D)
# ===========================================================================
print("Generating turekFigReducedSnaps3D.pdf ...")

fig, ax = plt.subplots(2, 2, figsize=(9, 7.5), subplot_kw={"projection": "3d"})
base_elev, base_azim = 23, 120
strtIncrems = [20, 0, 0, 0]
k, v, j = 0, 1, 2

for i in range(fluidSurr._p):
    ax[poss[i]].set_prop_cycle(LOCAL_LINE_CYCLER)
    ax[poss[i]].view_init(elev=base_elev, azim=base_azim)

    visInput = fluidSurr.reducedLoadData[i].copy()
    visOutput = fluidSurr.reducedLoadData[i].copy()
    visOutputCalibed = fluidSurr.calibrationQs[i] @ visOutput

    t0_s = strtIncrems[i]
    ax[poss[i]].plot(visInput[k, t0_s:], visInput[v, t0_s:], visOutput[j, t0_s:],
                     linewidth=0.8, alpha=0.8, label=rf"$\boldsymbol{{f}}_{i + 1}^r$")
    ax[poss[i]].plot(visOutputCalibed[k, t0_s:], visOutputCalibed[v, t0_s:], visOutputCalibed[j, t0_s:],
                     linewidth=0.8, alpha=0.8, label=rf"$\widehat{{\boldsymbol{{f}}}}_{i + 1}^r$")

    ax[poss[i]].legend(loc="best")
    ax[poss[i]].grid(False)
    ax[poss[i]].set_xlabel(rf"${k + 1}^{{st}}$ Input Component", fontsize=11)
    ax[poss[i]].set_ylabel(rf"${v + 1}^{{nd}}$ Input Component", fontsize=11)
    ax[poss[i]].set_zlabel(rf"${j + 1}^{{rd}}$ Output Component", fontsize=11)
    ax[poss[i]].sharez(ax[0, 1])

fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/turekFigReducedSnaps3D.pdf", bbox_inches="tight")
plt.close(fig)

# ===========================================================================
# 6.  Test data
# ===========================================================================
(loadTestData_Fl, dispTestData_Fl, loadConvTestData_Fl, flTestSize,
 loadTestData_Sol, dispTestData_Sol, dispConvTestData_Sol, solTestSize) = load_test_data()

# ===========================================================================
# 7.  FIGURE : Combining local regressions in the latent space
# ===========================================================================
print("Generating figReducedPredictionStepTurek.pdf ...")

fig, ax = plt.subplots(1, 2, figsize=(8.3, 3.9), sharey=False)
ax[0].set_prop_cycle(LINE_CYCLER)
ax[1].set_prop_cycle(LINE_CYCLER)

k, j = 0, 0
strtIncrems_lat = [20, 0, 0, 0]
strtIncremPred = 0

for i in range(fluidSurr._p):
    visInput = np.vstack((fluidSurr.reducedDispData[i], fluidSurr.reducedPrevLoadData[i]))
    visOutput = fluidSurr.reducedLoadData[i]
    ax[0].plot(visInput[k, strtIncrems_lat[i]:], visOutput[j, strtIncrems_lat[i]:], "-",
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
    ax[1].plot(xTestCalibed[k, strtIncrems_lat[i]:], new_preds[i, j, strtIncrems_lat[i]:], "-",
               linewidth=0.7, alpha=0.6, label="$\\widehat{\\boldsymbol{x}}_{" + str(i + 1) + "}^{n, r}$")

predicted_output = np.dot(new_preds.T, fluidSurr.reducLoad.weights).T
ax[1].plot(new_input[k, strtIncremPred:], predicted_output[j, strtIncremPred:], "--",
           linewidth=1.0, alpha=1, color="palegoldenrod",
           label="$\\textnormal{Predicted } \\boldsymbol{x}_{*}^{n, r}$")

ax[0].set_xlabel(r"$" + str(k + 1) + "^{st}$ \\textnormal{Comp. of} $\\boldsymbol{z}^{n, r}_{k}$", fontsize=8)
ax[1].set_xlabel(r"$" + str(k + 1) + "^{st}$ \\textnormal{Comp. of} $\\boldsymbol{z}^{n, r}_{*}$", fontsize=8)
ax[0].set_ylabel(r"$" + str(j + 1) + "^{st}$ \\textnormal{Comp. of} $\\boldsymbol{x}^{n, r}_{k}$", fontsize=8)
ax[1].legend(loc="best", ncol=4, fontsize=6.5)
ax[0].legend(loc="best", ncol=4, fontsize=6.5)
for a in ax:
    a.tick_params(axis="both", which="major", labelsize=8)
    a.grid(alpha=0.3)
fig.tight_layout(w_pad=0.2)
fig.savefig(f"{OUTPUT_DIR}/figReducedPredictionStepTurek.pdf", bbox_inches="tight")
plt.close(fig)

# ===========================================================================
# 8.  Model: Global static basis, linear interpolation
# ===========================================================================
print("Training Model: Global static basis (linear) ...")
loadData_forFluid, dispData_forFluid, loadConvData_forFluid, size_forFluid, _, _, _, _ = \
    load_train_data(names, remove_dts, return_list=False)

fluidSurr2 = FluidSurrog(reTrainThres=reTrainThres, maxLen=maxLen)
fluidSurr2.train(
    dispData_forFluid, loadConvData_forFluid, loadData_forFluid,
    rank_pres=RANK, smoothing=1e-5, kernel="polyC", degree=1,
    norm=[False, True], norm_regr=None, normalization=["max", "norm"],
    weights=True, solidReduc=solidReducSaved, multiple_param_regressor=False,
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


# ===========================================================================
# 9.  Model: Global static basis, nonlinear (quadratic) regression
# ===========================================================================
print("Training Model: Global static basis (nonlinear) ...")
fluidSurr2Nln = FluidSurrog(reTrainThres=reTrainThres, maxLen=maxLen)
fluidSurr2Nln.train(
    dispData_forFluid, loadConvData_forFluid, loadData_forFluid,
    rank_pres=RANK, smoothing=1e-5, kernel="polyC", degree=2,
    norm=[False, True], norm_regr=None, normalization=["max", "norm"],
    weights=True, solidReduc=solidReducSaved, multiple_param_regressor=False,
)
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


# ===========================================================================
# 10.  Model: Local static basis, linear interpolation (no basis update)
# ===========================================================================
print("Training Model: Local static basis (no update) ...")
loadData_forFluid, dispData_forFluid, loadConvData_forFluid, size_forFluid, _, _, _, _ = \
    load_train_data(names, remove_dts, return_list=True)

fluidSurr3 = TrackedFluidSurrog(
    reTrainThres=reTrainThres, maxLen=maxLen, updateBasis=True,
    updateThres=2_000_000, updateOmega=True,
)
fluidSurr3.train(
    dispData_forFluid, loadConvData_forFluid, loadData_forFluid,
    rank_pres=RANK, smoothing=1e-5, kernel="polyC", degree=1,
    norm=[False, True], norm_regr=None, normalization=["max", "norm"],
    params=PARAM_INS, weights=True, solidReduc=solidReducSaved,
    multiple_param_regressor=True, cleanup=True,
)
fluidSurr3.initialize_predictions(np.array([[1.0], [1.0]]))
del loadData_forFluid, dispData_forFluid, loadConvData_forFluid

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

# ===========================================================================
# 11.  Online prediction with the proposed model + optional projection errors
# ===========================================================================
print("Running online prediction (Proposed - GROUSE, linear) ...")
results = np.empty_like(loadTestData_Fl)

computeOnlineAngle = not args.skip_online_angle_comp
computeTotalAngle = not args.skip_total_angle_comp
computeProjectionErrs = not args.skip_projection_errors

if computeTotalAngle:
    firstPredictedBasis = fluidSurr.cloneBasis.copy()
    totalAngle = []

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
    np.save(f"{OUTPUT_DIR}/relGlobalOrthNorms_turek.npy", np.array(relGlobalOrthNorms))
    np.save(f"{OUTPUT_DIR}/relOrthNormsNoUpdate_turek.npy", np.array(relOrthNormsNoUpdate))
    np.save(f"{OUTPUT_DIR}/relOrthNorms_turek.npy", np.array(relOrthNorms))

    print("Generating figCompareProjErrorsTurek.pdf ...")
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    ax.set_prop_cycle(PROJ_LINE_CYCLER)
    w = 10
    ax.semilogy(relGlobalOrthNorms, label="\\textnormal{Global basis}")
    ax.semilogy(relOrthNormsNoUpdate, label="\\textnormal{Local - w/o update}")
    raw = relOrthNorms
    smoothed = smooth(raw, w)
    ax.semilogy(raw, "k", lw=0.3, alpha=0.35)
    ax.semilogy(smoothed, "k", lw=1.2, label="\\textnormal{Local - Grouse}")
    ax.set_ylabel("\\textnormal{Relative orthogonal projection error}", fontsize=11)
    ax.set_xlabel("\\textnormal{Online steps}", fontsize=11)
    ax.grid()
    ax.legend(fontsize=11, loc="best")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/figCompareProjErrorsTurek.pdf", bbox_inches="tight")
    plt.close(fig)
else:
    print("Skipping projection error computation and figure.")

if computeTotalAngle and computeOnlineAngle:
    print("Generating combinedAngleShiftTurek.pdf ...")
    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(6.3, 2.4))
    ax1.plot(np.array([np.linalg.norm(i) for i in fluidSurr.recursive_angles]),
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
    fig.savefig(f"{OUTPUT_DIR}/combinedAngleShiftTurek.pdf", bbox_inches="tight")
    plt.close(fig)

# ===========================================================================
# 12.  Model: Proposed - Local basis with GROUSE update, nonlinear regression
# ===========================================================================
print("Training Model: Proposed (GROUSE, nonlinear) ...")
loadData_forFluid, dispData_forFluid, loadConvData_forFluid, size_forFluid, _, _, _, _ = \
    load_train_data(names, remove_dts, return_list=True)

t0 = time.time()
fluidSurrNln = TrackedFluidSurrog(
    reTrainThres=reTrainThres, maxLen=maxLen, updateBasis=True,
    updateThres=updateThres, updateOmega=True,
)
fluidSurrNln.train(
    dispData_forFluid, loadConvData_forFluid, loadData_forFluid,
    rank_pres=RANK, smoothing=1e-5, kernel="polyC", degree=2,
    norm=[False, True], norm_regr=None, normalization=["max", "norm"],
    params=PARAM_INS, weights=True, solidReduc=solidReducSaved,
    multiple_param_regressor=True, cleanup=True,
)
print(f"  Training time [Nln regression]: {time.time() - t0:.3f} s")

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
print(
    f"Total prediction and update time for the proposed method"
    f" [Nln regression]: "
    f"{time.time() - t0:.3f} s"
)

trueNorms = np.linalg.norm(loadTestData_Fl, axis=0)
errsNln = relative_error(resultsNln, loadTestData_Fl, trueNorms)[:-4]


# ===========================================================================
# 13.  Model: Global static basis, parametric regression
# ===========================================================================
print("Training Model: Global static basis (parametric) ...")
del loadData_forFluid, dispData_forFluid, loadConvData_forFluid

loadData_forFluid, dispData_forFluid, loadConvData_forFluid, size_forFluid, _, _, _, _ = \
    load_train_data(names, remove_dts, return_list=False)
loadData_forFluid_list, _, _, _, _, _, _, _ = load_train_data(names, remove_dts, return_list=True)

fluidSurr4 = FluidSurrog(reTrainThres=reTrainThres, maxLen=maxLen)
fluidSurr4.train(
    dispData_forFluid, loadConvData_forFluid, loadData_forFluid,
    rank_pres=RANK, smoothing=1e-5, kernel="polyC", degree=1,
    norm=[False, True], norm_regr=None, normalization=["max", "norm"],
    weights=True, solidReduc=solidReducSaved, multiple_param_regressor=False,
    params=np.repeat(PARAM_INS, [a.shape[1] for a in loadData_forFluid_list], axis=1),
)

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
del loadData_forFluid, dispData_forFluid, loadConvData_forFluid

# ===========================================================================
# 14.  rDMDc baseline
# ===========================================================================
print("Training rDMDc ...")
loadData_forFluid, dispData_forFluid, loadConvData_forFluid, size_forFluid, _, _, _, _ = \
    load_train_data(names, remove_dts, return_list=False)

n_dmd = min(5000, loadConvData_forFluid.shape[1])
fluidSurrRdmd = RDMDC(epsilon=0.01, lambdaForgetBasis=0.95, lambdaForget=0.95)
fluidSurrRdmd.decompose(
    loadConvData_forFluid[:, :n_dmd], rank=RANK, Y=loadData_forFluid[:, :n_dmd],
    u_input=solidROM.reducLoad.encode(dispData_forFluid[:, :n_dmd]),
    precomp_std=fluidSurr.reducLoad.rom.snap_norms,
    precomp_mean=fluidSurr.reducLoad.rom.mean_flow,
    precomputed_modes=fluidSurr3.reducLoad.pod.modes.copy(),
)

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
print("Total prediction and update time for rDMDC: ", time.time() - t0, " s")
del loadData_forFluid, dispData_forFluid, loadConvData_forFluid

# ===========================================================================
# 15.  FIGURE : Online relative error, all models
# ===========================================================================
print("Generating figTurekCompareAccuracies.pdf ...")
trueField = loadTestData_Fl.copy()
trueNorms = np.linalg.norm(trueField, axis=0)

errs = moving_average(relative_error(results, trueField, trueNorms)[:-4], 10)
errs2 = moving_average(relative_error(results2, trueField, trueNorms)[:-4], 10)
errs3 = moving_average(relative_error(results3, trueField, trueNorms)[:-4], 10)
errs4 = moving_average(relative_error(results4, trueField, trueNorms)[:-4], 10)
errsRdmd = moving_average(relative_error(resultsRdmd, trueField, trueNorms)[:-4], 10)

np.save(f"{OUTPUT_DIR}/errs_turek.npy", errs)
np.save(f"{OUTPUT_DIR}/errs2_turek.npy", errs2)
np.save(f"{OUTPUT_DIR}/errs3_turek.npy", errs3)
np.save(f"{OUTPUT_DIR}/errs4_turek.npy", errs4)
np.save(f"{OUTPUT_DIR}/errsNln_turek.npy", errsNln)
np.save(f"{OUTPUT_DIR}/errs2Nln_turek.npy", errs2Nln)
np.save(f"{OUTPUT_DIR}/errsRdmd_turek.npy", errsRdmd)

fig, ax = plt.subplots(figsize=(6.5, 3.4))
ax.set_prop_cycle(ERROR_LINE_CYCLER)
ax.semilogy(errs2, label="\\textnormal{Global $\\mathcal{V}$  - Linear $\\mathcal{I}$}", linewidth=1)
ax.semilogy(errs2Nln, label="\\textnormal{Global $\\mathcal{V}$  - Nln $\\mathcal{I}$}", linewidth=1)
ax.semilogy(errs4, label="\\textnormal{Global $\\mathcal{V}$  - Parametric $\\mathcal{I}$}", linewidth=1)
ax.semilogy(errsRdmd, label="\\textnormal{rDMDc}", linewidth=1)
ax.semilogy(errs, label="\\textnormal{Current - Linear $\\mathcal{I}$}", linewidth=1)
ax.semilogy(errs3, label="\\textnormal{Local static $\\mathcal{V}$ - Linear $\\mathcal{I}$}", linewidth=1)
ax.semilogy(errsNln, "k", label="\\textnormal{Current - Nln $\\mathcal{I}$}", linewidth=1)
ax.set_xlabel("\\textnormal{Online steps}", fontsize=9)
ax.set_ylabel("\\textnormal{Relative error} $e^n$", fontsize=9)
ax.tick_params(axis="both", which="major", labelsize=9)
ax.tick_params(axis="both", which="minor", labelsize=9)
ax.legend(shadow=False, fontsize=7.5, ncol=2, loc="best")
ax.grid()
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/figTurekCompareAccuracies.pdf", bbox_inches="tight")
plt.close(fig)

print("\nAll figures/arrays saved to:", OUTPUT_DIR)
print("Done.")
