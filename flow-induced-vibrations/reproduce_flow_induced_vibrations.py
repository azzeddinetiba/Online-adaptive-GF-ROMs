"""
Reproduce results and figures for the flow-induced vibrations benchmark (oscillating cylinder).

This script trains and evaluates several surrogate models for online fluid field prediction:
  - Global static basis with linear interpolation (FluidSurrog)
  - Global static basis with nonlinear regression
  - Global static basis with parametric regression
  - Local static basis with linear interpolation (TrackedFluidSurrog, no basis update)
  - Proposed: Local basis with GROUSE update + linear interpolation
  - Proposed: Local basis with GROUSE update + nonlinear regression
  - Local basis with PAST update + nonlinear regression
  - Recursive DMDc (rDMDc)

Figures generated (matching the paper):
  - figReducedSnaps.pdf        : Phase-space view in the latent space before/after calibration
  - figReducedSnaps3D.pdf      : 3-D phase-space view in the latent space
  - figReducedPredictionStep.pdf : Combining local regressions in the latent space
  - combinedAngleShift.pdf : Evolution of the subspace rotation
  - figCompareProjErrors.pdf   : Orthogonal projection error comparison
  - figCompareFieldsComplete.pdf : Online relative error comparison (all models)
  - compareGrouseAndPast.pdf   : GROUSE vs PAST in the latent space
  - xVelocitySignalMiddleInlet.pdf : x-velocity signal at the middle inlet point
  - modesFieldsComparison.pdf    : Initial vs final basis mode velocity-norm fields

All figures are saved in the results_figures_arrays/ subdirectory.
"""

import argparse
import os
import time

import h5py
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reproduce the flow-induced vibrations benchmark results."
    )
    parser.add_argument(
        "--dataset-file",
        default="data/oscillating_cylinder_benchmark_dataset_v2.mat",
        help="Path to the MATLAB dataset file.",
    )
    parser.add_argument(
        "--output-dir",
        default="results_figures_arrays",
        help="Directory where figures and arrays will be written.",
    )
    parser.add_argument(
        "--skip-projection-errors",
        action="store_true",
        help="Skip the expensive orthogonal projection error computation.",
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
    DATASET_FILE = args.dataset_file
    OUTPUT_DIR = args.output_dir
    COMPUTE_PROJECTION_ERRORS = not args.skip_projection_errors
    COMPUTE_ONLINE_ANGLES = not args.skip_online_angle_comp
    COMPUTE_TOTAL_ANGLES = not args.skip_total_angle_comp
else:
    DATASET_FILE = "oscillating_cylinder_benchmark_dataset_v2.mat"
    OUTPUT_DIR = "results_figures_arrays"
    COMPUTE_PROJECTION_ERRORS = True
    COMPUTE_ONLINE_ANGLES = True
    COMPUTE_TOTAL_ANGLES = True

from rom_am.fluid_surrogate import FluidSurrog
from rom_am.tracked_fluid_surrogate import TrackedFluidSurrog
from rom_am.rdmdc import RDMDC
from rom_am.utils import rank1_update

# ---------------------------------------------------------------------------
# Matplotlib / LaTeX settings
# ---------------------------------------------------------------------------
plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname("./coSimData/"), exist_ok=True)

# ---------------------------------------------------------------------------
# 1.  Load training data
#     Training parameters: f = {0.5, 0.7, 1.1, 1.3} Hz  (0.9 Hz is the test case)
# ---------------------------------------------------------------------------
FIELD_NAMES  = ["Vx_grid", "Vy_grid", "p_grid"]
N_GRID       = 6405            # spatial grid points per field
N_FIELDS     = len(FIELD_NAMES)
RANK         = 64              # maximum of 99.9% energy criterion-based ranks

train_params   = ["f0p50_A0p20", "f0p70_A0p20", "f1p10_A0p20", "f1p30_A0p20"]
train_lens     = [800, 880, 1080, 1160]   # total snapshots per parameter
param_ins      = np.array([[0.5, 0.7, 1.1, 1.3]])   # shape (1, n_params)

fieldsData_pre  = []   # x^n   (state at time n)
fieldsData_post = []   # x^{n+1} (state at time n+1)
yData           = []   # solid displacement / parameter signal

with h5py.File(DATASET_FILE, "r") as f:
    for j, param in enumerate(train_params):
        L   = train_lens[j]
        mid = L // 2
        snaps = np.empty((N_GRID * N_FIELDS, 800))

        for i, field in enumerate(FIELD_NAMES):
            snaps[i * N_GRID:(i + 1) * N_GRID, :] = \
                f["data_structure"]["monosines"][param][field][:, mid - 400:mid + 400]

        fieldsData_pre.append(snaps[:, :-1].copy())
        fieldsData_post.append(snaps[:, 1:].copy())
        yData.append(
            f["data_structure"]["monosines"][param]["y"][[0], mid - 400:mid + 400 - 1]
        )

# ---------------------------------------------------------------------------
# 2.  Load test data  (f = 0.9 Hz, full sequence)
# ---------------------------------------------------------------------------
TEST_PARAM = "f0p90_A0p20"
LEN_TST    = 1000

fieldsData_pre_tst  = []
fieldsData_post_tst = []
yData_tst           = []

with h5py.File(DATASET_FILE, "r") as f:
    snaps_tst = np.empty((N_GRID * N_FIELDS, LEN_TST))
    for i, field in enumerate(FIELD_NAMES):
        snaps_tst[i * N_GRID:(i + 1) * N_GRID, :] = \
            f["data_structure"]["monosines"][TEST_PARAM][field][:]

    fieldsData_pre_tst.append(snaps_tst[:, :-1].copy())
    fieldsData_post_tst.append(snaps_tst[:, 1:].copy())
    yData_tst.append(f["data_structure"]["monosines"][TEST_PARAM]["y"][[0], :-1])

    timeTst = f["data_structure"]["monosines"][TEST_PARAM]["time"][[0], :-1][0]
    grid_points = f["data_structure"]["monosines"][TEST_PARAM]["grid_points"][:, :].T

N_TST = LEN_TST - 1   # number of prediction steps

# ===========================================================================
# Helper: common training keyword arguments (shared across most models)
# ===========================================================================
COMMON_TRAIN_KWARGS = dict(
    rank_pres      = RANK,
    smoothing      = "auto",
    kernel         = "polyC",
    degree         = 1,
    norm           = [False, True],
    norm_regr      = "max",
    normalization  = ["max", "norm"],
    weights        = True,
    solidReduc     = None,
)
REGRESSION_FREQUENCY = 80
BASIS_FREQUENCY      = 170

def relative_error(pred, true):
    """Column-wise relative L2 error."""
    return np.linalg.norm(pred - true, axis=0) / np.linalg.norm(true, axis=0)

# ===========================================================================
# 3.  Model A – Global static basis, linear interpolation
# ===========================================================================
print("Training Model A: Global static basis (linear) ...")
fluidSurr2 = FluidSurrog(reTrainThres=REGRESSION_FREQUENCY, maxLen=1000)
fluidSurr2.train(
    np.hstack(yData),
    np.hstack(fieldsData_pre),
    np.hstack(fieldsData_post),
    **COMMON_TRAIN_KWARGS,
    alg                     = "snap",
    multiple_param_regressor = False,
)

results2 = np.empty_like(fieldsData_post_tst[0])
for i in range(N_TST):
    results2[:, i] = fluidSurr2.predict(
        yData_tst[0][:, [i]], fieldsData_pre_tst[0][:, [i]], solidReduc=None
    ).ravel()
    fluidSurr2.augmentData(
        yData_tst[0][:, [i]], fieldsData_pre_tst[0][:, [i]],
        fieldsData_post_tst[0][:, [i]], solidReduc=None,
    )
errs2 = relative_error(results2, fieldsData_post_tst[0])

# ===========================================================================
# 4.  Model B – Global static basis, nonlinear (cubic) regression
# ===========================================================================
print("Training Model B: Global static basis (nonlinear) ...")
fluidSurr2Nln = FluidSurrog(reTrainThres=REGRESSION_FREQUENCY, maxLen=1000)
fluidSurr2Nln.train(
    np.hstack(yData),
    np.hstack(fieldsData_pre),
    np.hstack(fieldsData_post),
    **{**COMMON_TRAIN_KWARGS, "smoothing": "auto", "kernel": "cubic"},
    multiple_param_regressor = False,
)

results2Nln = np.empty_like(fieldsData_post_tst[0])
for i in range(N_TST):
    results2Nln[:, i] = fluidSurr2Nln.predict(
        yData_tst[0][:, [i]], fieldsData_pre_tst[0][:, [i]], solidReduc=None
    ).ravel()
    fluidSurr2Nln.augmentData(
        yData_tst[0][:, [i]], fieldsData_pre_tst[0][:, [i]],
        fieldsData_post_tst[0][:, [i]], solidReduc=None,
    )
errs2Nln = relative_error(results2Nln, fieldsData_post_tst[0])

# ===========================================================================
# 5.  Model C – Global static basis, parametric regression
# ===========================================================================
print("Training Model C: Global static basis (parametric) ...")
# Repeat parameter value for each snapshot in each training batch
params_repeated = np.repeat(
    param_ins, [a.shape[1] for a in fieldsData_post], axis=1
)

fluidSurr4 = FluidSurrog(reTrainThres=REGRESSION_FREQUENCY, maxLen=1000)
fluidSurr4.train(
    np.hstack(yData),
    np.hstack(fieldsData_pre),
    np.hstack(fieldsData_post),
    **COMMON_TRAIN_KWARGS,
    multiple_param_regressor = False,
    params                   = params_repeated,
)

results4 = np.empty_like(fieldsData_post_tst[0])
for i in range(N_TST):
    results4[:, i] = fluidSurr4.predict(
        yData_tst[0][:, [i]], fieldsData_pre_tst[0][:, [i]],
        solidReduc=None, params=np.array([[0.9]])
    ).ravel()
    fluidSurr4.augmentData(
        yData_tst[0][:, [i]], fieldsData_pre_tst[0][:, [i]],
        fieldsData_post_tst[0][:, [i]], solidReduc=None,
        params=np.array([[0.9]]),
    )
errs4 = relative_error(results4, fieldsData_post_tst[0])

# ===========================================================================
# 6.  Model D – Local static basis, linear interpolation (no basis update)
# ===========================================================================
print("Training Model D: Local static basis (no update) ...")
fluidSurr3 = TrackedFluidSurrog(
    reTrainThres=REGRESSION_FREQUENCY, maxLen=1000,
    updateBasis=True, updateThres=500000,
    automatic_weight=True, eps_for_automatic_weight=4.,
    automatic_weight_at_retrain=True, updateOmega=False,
    output_folder=OUTPUT_DIR,
)
fluidSurr3.train(
    yData, fieldsData_pre, fieldsData_post,
    **COMMON_TRAIN_KWARGS,
    params                   = param_ins,
    multiple_param_regressor = True,
)
fluidSurr3.initialize_predictions(np.array([[0.9]]))

results3 = np.empty_like(fieldsData_post_tst[0])
for i in range(N_TST):
    results3[:, i] = fluidSurr3.predict(
        yData_tst[0][:, [i]], fieldsData_pre_tst[0][:, [i]],
        solidReduc=None, params=np.array([[0.9]])
    ).ravel()
    fluidSurr3.augmentData(
        yData_tst[0][:, [i]], fieldsData_pre_tst[0][:, [i]],
        fieldsData_post_tst[0][:, [i]], solidReduc=None,
        params=np.array([[0.9]]),
    )
errs3 = relative_error(results3, fieldsData_post_tst[0])

# ===========================================================================
# 7.1  PROPOSED – Local basis with GROUSE update, linear interpolation
# ===========================================================================
print("Training Model E (Proposed - GROUSE, linear) ...")
t0 = time.time()
fluidSurr = TrackedFluidSurrog(
    reTrainThres=REGRESSION_FREQUENCY, maxLen=1000,
    updateBasis=True, updateThres=BASIS_FREQUENCY,
    automatic_weight=True, eps_for_automatic_weight=4.,
    automatic_weight_at_retrain=True, updateOmega=False,
    output_folder=OUTPUT_DIR,
)
fluidSurr.omega0         = 0.01
fluidSurr._omega_terms   = (1 - fluidSurr.omega0, fluidSurr.omega0)

fluidSurr.train(
    yData, fieldsData_pre, fieldsData_post,
    **COMMON_TRAIN_KWARGS,
    params                   = param_ins,
    multiple_param_regressor = True,
    cleanup                  = False,
    alg                      = "snap",
)
print(f"  Training time [Linear regression]: {time.time() - t0:.3f} s")


# Pre-prediction phase
t0 = time.time()
fluidSurr.initialize_predictions(np.array([[0.9]]))
t1 = time.time()
print(f"  Pre-prediction time: {t1 - t0:.3f} s")
print(f"  Initial weights: {fluidSurr.reducLoad.weights}")


# ===========================================================================
# 7.2.  FIGURE 0 – Combining local regressions in the latent space
# ===========================================================================
print("Generating figReducedPredictionStep.pdf ...")

fig, ax = plt.subplots(1, 2, figsize=(6.3, 3.2), sharey=False)
lc_mid = cycler(color=["#009E73", "#0072B2", "#D55E00", "#CC79A7",
                        "#56B4E9", "#F0E442", "#E69F00"])
ax[0].set_prop_cycle(lc_mid)
ax[1].set_prop_cycle(lc_mid)

k_lat = 1;  j_lat = 4
strtIncrems_lat = [390, 390, 390, 400]
strtIncremPred  = 400

for i in range(fluidSurr._p):
    visInput  = np.vstack((fluidSurr.reducedDispData[i], fluidSurr.reducedPrevLoadData[i]))
    visOutput = fluidSurr.reducedLoadData[i]
    ax[0].plot(visInput[k_lat, strtIncrems_lat[i]:], visOutput[j_lat, strtIncrems_lat[i]:],
               "-", linewidth=0.7, alpha=1,
               label=r"${\boldsymbol{x}}_{" + str(i + 1) + r"}^{n, r}$")

new_input  = np.vstack((yData_tst[0], fluidSurr.reducLoad.encode(fieldsData_pre_tst[0])))
true_output = fluidSurr.reducLoad.encode(fieldsData_post_tst[0])
ax[1].plot(new_input[k_lat, strtIncremPred:], true_output[j_lat, strtIncremPred:],
           "k", linewidth=1.0, label=r"$\textnormal{True }  \boldsymbol{x}_{*}^{n, r}$")

new_preds = np.empty((fluidSurr._p, fluidSurr.reducLoad.latent_dim, yData_tst[0].shape[1]))
for i in range(fluidSurr._p):
    xTest = np.vstack((
        yData_tst[0],
        fluidSurr.reducLoadLocals[i].encode(fieldsData_pre_tst[0]),
    ))
    xTestCalibed = xTest.copy()
    xTestCalibed[-fluidSurr.reducLoadLocals[i].pod.kept_rank:, :] = \
        fluidSurr.calibrationQs[i] @ xTestCalibed[-fluidSurr.reducLoadLocals[i].pod.kept_rank:, :]

    new_preds[i] = fluidSurr.regressor[i].predict(xTest)
    new_preds[i] = fluidSurr.calibrationQs[i] @ new_preds[i]
    ax[1].plot(xTestCalibed[k_lat, strtIncrems_lat[i]:], new_preds[i, j_lat, strtIncrems_lat[i]:],
               "-", linewidth=0.7, alpha=0.6,
               label=r"$\widehat{\boldsymbol{x}}_{" + str(i + 1) + r"}^{n, r}$")

predicted_output = np.dot(new_preds.T, fluidSurr.reducLoad.weights).T
ax[1].plot(new_input[k_lat, strtIncremPred:], predicted_output[j_lat, strtIncremPred:],
           "--", linewidth=1.0, alpha=1, color="palegoldenrod",
           label=r"$\textnormal{Predicted} \boldsymbol{x}_{*}^{n, r}$")

ax[0].set_xlabel(r"$" + str(k_lat + 1) + r"^{nd}$ \textnormal{Comp. of} $\boldsymbol{z}^{n, r}_{k}$", fontsize=8)
ax[1].set_xlabel(r"$" + str(k_lat + 1) + r"^{nd}$ \textnormal{Comp. of} $\boldsymbol{z}^{n, r}_{*}$", fontsize=8)
ax[0].set_ylabel(r"$" + str(j_lat + 1) + r"^{th}$ \textnormal{Comp. of} $\boldsymbol{x}^{n, r}_{k}$", fontsize=8)
ax[1].legend(loc="best", ncol=2, fontsize=6.5)
ax[0].legend(loc="best", ncol=2, fontsize=6.5)
for a in ax:
    a.tick_params(axis="both", which="major", labelsize=8)
    a.grid(alpha=0.3)
fig.tight_layout(w_pad=1.0)
fig.savefig(f"{OUTPUT_DIR}/figReducedPredictionStep.pdf", bbox_inches="tight")
plt.close(fig)


# ===========================================================================
# 7.2.2  FIGURE 1 – Phase-space view in the latent space (2-D)
#        Before vs after calibration for each training parameter
# ===========================================================================
print("Generating figReducedSnaps.pdf ...")

fig, ax = plt.subplots(2, 2, figsize=(9, 7.5), sharex="col", sharey="row")
lc_local = cycler(color=["#ffa600", "#58508d", "#bc5090", "#ff6361", "#ffa600"])
k = 0;  j = 5
strtIncrems = [200, 200, 390, 400]
poss = [(0, 0), (0, 1), (1, 0), (1, 1)]

for i in range(fluidSurr._p0):
    ax[poss[i]].set_prop_cycle(lc_local)
    visInput  = fluidSurr.reducedLoadData[i].copy()
    visOutput = fluidSurr.reducedLoadData[i].copy()
    visOutputCalibed = fluidSurr.calibrationQs[i] @ visOutput

    ax[poss[i]].plot(visInput[k, strtIncrems[i]:], visOutput[j, strtIncrems[i]:],
                     "-", linewidth=0.8, alpha=0.8,
                     label=r"$\boldsymbol{f}_" + str(i) + r"^r$")
    ax[poss[i]].plot(visOutputCalibed[k, strtIncrems[i]:], visOutputCalibed[j, strtIncrems[i]:],
                     "-", linewidth=0.8, alpha=0.8,
                     label=r"$\widehat{\boldsymbol{f}}_" + str(i) + r"^r$")
    ax[poss[i]].legend(loc="best")

    if i in (0, 1):
        ax[poss[i]].tick_params(bottom=False)
    else:
        ax[poss[i]].set_xlabel(
            r"$" + str(k + 1) + r"^{st}$ \textnormal{Component}", fontsize=12)
    if i in (1, 3):
        ax[poss[i]].tick_params(left=False)
    else:
        ax[poss[i]].set_ylabel(
            r"$" + str(j + 1) + r"^{st}$ \textnormal{Component}", fontsize=12)

fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/figReducedSnaps.pdf", bbox_inches="tight")
plt.close(fig)

# ===========================================================================
# 7.2.3  FIGURE 2 – Phase-space view in the latent space (3-D)
# ===========================================================================
print("Generating figReducedSnaps3D.pdf ...")

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

fig, ax = plt.subplots(2, 2, figsize=(6.3, 5.0), subplot_kw={"projection": "3d"})
strtIncrems = [20, 20, 39, 40]
k = 0;  v = 1;  j = 5

for i in range(4):
    ax[poss[i]].set_prop_cycle(lc_local)
    ax[poss[i]].view_init(elev=23, azim=120)

    visInput  = fluidSurr.reducedLoadData[i].copy()
    visOutput = fluidSurr.reducedLoadData[i].copy()
    visOutputCalibed = fluidSurr.calibrationQs[i] @ visOutput

    t0_s = strtIncrems[i]
    ax[poss[i]].plot(visInput[k, t0_s:], visInput[v, t0_s:], visOutput[j, t0_s:],
                     linewidth=0.7, alpha=0.8,
                     label=rf"$\boldsymbol{{x}}_{i+1}^r$")
    ax[poss[i]].plot(visOutputCalibed[k, t0_s:], visOutputCalibed[v, t0_s:], visOutputCalibed[j, t0_s:],
                     linewidth=0.7, alpha=0.8,
                     label=rf"$\widehat{{\boldsymbol{{x}}}}_{i+1}^r$")

    ax[poss[i]].legend(fontsize=7, loc="best")
    ax[poss[i]].grid(False)
    ax[poss[i]].set_xlabel(rf"${k+1}^{{st}}$ component", fontsize=8, labelpad=2)
    ax[poss[i]].set_ylabel(rf"${v+1}^{{nd}}$ component", fontsize=8, labelpad=2)
    ax[poss[i]].set_zlabel(rf"${j+1}^{{th}}$ component", fontsize=8, labelpad=2)
    ax[poss[i]].tick_params(axis="both", which="major", labelsize=7, pad=1)

fig.tight_layout(pad=0.5, w_pad=0.3, h_pad=0.3)
fig.savefig(f"{OUTPUT_DIR}/figReducedSnaps3D.pdf", bbox_inches="tight")
plt.close(fig)


# ===========================================================================
# 7.3  PROPOSED – Local basis with GROUSE update, linear interpolation
# ===========================================================================

computeTotalAngle  = COMPUTE_TOTAL_ANGLES
computeOnlineAngle = COMPUTE_ONLINE_ANGLES
if computeTotalAngle:
    firstPredictedBasis = fluidSurr.cloneBasis.copy()
    totalAngle = []
    from rom_am.utils import angles

t0 = time.time()
results = np.empty_like(fieldsData_post_tst[0])
for i in range(N_TST):
    results[:, i] = fluidSurr.predict(
        yData_tst[0][:, [i]], fieldsData_pre_tst[0][:, [i]],
        solidReduc=None, params=np.array([[0.9]])
    ).ravel()
    fluidSurr.augmentData(
        yData_tst[0][:, [i]], fieldsData_pre_tst[0][:, [i]],
        fieldsData_post_tst[0][:, [i]], solidReduc=None,
        params=np.array([[0.9]]), stepsize=None, computeAngle=computeOnlineAngle,
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
errs = relative_error(results, fieldsData_post_tst[0])

print("GROUSE orthonormality residual: ",
      np.linalg.norm((fluidSurr.reducLoad.pod.modes.T
                      @ fluidSurr.reducLoad.pod.modes)
                     - np.eye(fluidSurr.reducLoad.latent_dim)))


# ===========================================================================
# 7.4  Angle Shift Figure
# ===========================================================================
print("Generating combinedAngleShift.pdf ...")

if computeOnlineAngle and computeTotalAngle:
    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(6.3, 2.4))
    
    # Left subplot: dual y-axes combining first two figures
    ax1.plot(timeTst, np.array([np.linalg.norm(i) for i in fluidSurr.recursive_angles]),
             color='darkslategray', linewidth=0.75)
    ax1.set_xlabel("\\textnormal{Time [s] (Online)}", fontsize=9)
    ax1.set_ylabel("\\textnormal{Online recursive distance}", fontsize=9, color='darkslategray')
    ax1.tick_params(axis='both', which='major', labelsize=9)
    ax1.tick_params(axis='y', labelcolor='darkslategray')
    ax1.grid(alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.plot(timeTst, np.array([np.linalg.norm(i[0]) for i in totalAngle]),
             color='darkcyan', linewidth=1)
    ax2.set_ylabel("\\textnormal{Total distance}", fontsize=9, color='darkcyan')
    ax2.tick_params(axis='y', labelcolor='darkcyan', labelsize=9)
    
    # Right subplot: third figure
    ax3.plot(timeTst, np.rad2deg(np.array([i[0].max() for i in totalAngle])),
             color='darkslategray', linewidth=1)
    ax3.set_xlabel("\\textnormal{Time [s] (Online)}", fontsize=9)
    ax3.set_ylabel("\\textnormal{Maximum total angle (\N{DEGREE SIGN})}", fontsize=9)
    ax3.tick_params(axis='both', which='major', labelsize=9)
    ax3.grid(alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/combinedAngleShift.pdf", bbox_inches="tight")
    plt.close(fig)

# ===========================================================================
# 8.  PROPOSED – Local basis with GROUSE update, nonlinear regression
# ===========================================================================
print("Training Model F (Proposed - GROUSE, nonlinear) ...")
t0 = time.time()
fluidSurrNln = TrackedFluidSurrog(
    reTrainThres=REGRESSION_FREQUENCY, maxLen=1000,
    updateBasis=True, updateThres=BASIS_FREQUENCY,
    automatic_weight=True, eps_for_automatic_weight=4.,
    automatic_weight_at_retrain=True, updateOmega=False,
    output_folder=OUTPUT_DIR,
)
fluidSurrNln.omega0       = 0.01
fluidSurrNln._omega_terms = (1 - fluidSurrNln.omega0, fluidSurrNln.omega0)

fluidSurrNln.train(
    yData, fieldsData_pre, fieldsData_post,
    **{**COMMON_TRAIN_KWARGS, "smoothing": 1e-1, "kernel": "cubic"},
    params                   = param_ins,
    multiple_param_regressor = True,
    cleanup                  = True,
    alg                      = "snap",
)
print(f"  Training time [Nln regression]: {time.time() - t0:.3f} s")
fluidSurrNln.initialize_predictions(np.array([[0.9]]))

t0 = time.time()
resultsNln = np.empty_like(fieldsData_post_tst[0])
for i in range(N_TST):
    resultsNln[:, i] = fluidSurrNln.predict(
        yData_tst[0][:, [i]], fieldsData_pre_tst[0][:, [i]],
        solidReduc=None, params=np.array([[0.9]])
    ).ravel()
    fluidSurrNln.augmentData(
        yData_tst[0][:, [i]], fieldsData_pre_tst[0][:, [i]],
        fieldsData_post_tst[0][:, [i]], solidReduc=None,
        params=np.array([[0.9]]), stepsize=None, computeAngle=False,
    )
print(
    f"Total prediction and update time for the proposed method"
    f" [Nln regression]: "
    f"{time.time() - t0:.3f} s"
)
errsNln = relative_error(resultsNln, fieldsData_post_tst[0])

# ===========================================================================
# 9.  Local basis with PAST update, nonlinear regression
# ===========================================================================
print("Training Model G (PAST, nonlinear) ...")
fluidSurrPast = TrackedFluidSurrog(
    reTrainThres=REGRESSION_FREQUENCY, maxLen=1000,
    updateBasis=True, updateThres=BASIS_FREQUENCY,
    automatic_weight=True, eps_for_automatic_weight=4.,
    automatic_weight_at_retrain=True, updateOmega=False,
    output_folder=OUTPUT_DIR,
)
fluidSurrPast.omega0       = 0.01
fluidSurrPast._omega_terms = (1 - fluidSurrPast.omega0, fluidSurrPast.omega0)

fluidSurrPast.train(
    yData, fieldsData_pre, fieldsData_post,
    **{**COMMON_TRAIN_KWARGS, "smoothing": "auto", "kernel": "cubic"},
    params                   = param_ins,
    multiple_param_regressor = True,
    cleanup                  = True,
    alg                      = "snap",
)
fluidSurrPast.initialize_predictions(np.array([[0.9]]))

resultsPast = np.empty_like(fieldsData_post_tst[0])
for i in range(N_TST):
    resultsPast[:, i] = fluidSurrPast.predict(
        yData_tst[0][:, [i]], fieldsData_pre_tst[0][:, [i]],
        solidReduc=None, params=np.array([[0.9]])
    ).ravel()
    fluidSurrPast.augmentData(
        yData_tst[0][:, [i]], fieldsData_pre_tst[0][:, [i]],
        fieldsData_post_tst[0][:, [i]], solidReduc=None,
        params=np.array([[0.9]]), stepsize=None, computeAngle=False,
        update_method=4,
    )
errsPast = relative_error(resultsPast, fieldsData_post_tst[0])

print("PAST orthonormality residual (expected > 0): ",
      np.linalg.norm((fluidSurrPast.reducLoad.pod.modes.T
                      @ fluidSurrPast.reducLoad.pod.modes)
                     - np.eye(fluidSurrPast.reducLoad.latent_dim)))

# ===========================================================================
# 10. rDMDc baseline
# ===========================================================================
print("Training rDMDc ...")
fluidSurrRdmd = RDMDC(epsilon=0.01, lambdaForgetBasis=0.95, lambdaForget=0.95)
fluidSurrRdmd.decompose(
    np.hstack(fieldsData_pre), rank=RANK,
    Y        = np.hstack(fieldsData_post),
    u_input  = np.hstack(yData),
    precomp_std   = fluidSurr.reducLoad.rom.snap_norms,
    precomp_mean  = fluidSurr.reducLoad.rom.mean_flow,
    precomputed_modes = fluidSurr3.reducLoad.pod.modes.copy(), # Pre initialized with 
                                                               # the interpolated modes
)

print("online streaming rDMDc ...")
t0 = time.time()
resultsRdmd = np.empty_like(fieldsData_post_tst[0])
for i in range(N_TST):
    resultsRdmd[:, i] = fluidSurrRdmd.predict(
        fieldsData_pre_tst[0][:, [i]], yData_tst[0][:, [i]]
    ).ravel()
    fluidSurrRdmd.update(
        fieldsData_post_tst[0][:, [i]],
        fieldsData_pre_tst[0][:, [i]],
        yData_tst[0][:, [i]],
    )
t1 = time.time()
print("Total prediction and update time for rDMDC: ", t1 - t0, " s")
errsRdmd = relative_error(resultsRdmd, fieldsData_post_tst[0])

# ===========================================================================
# 11.  Orthogonal projection errors
#      (uses a static "romTracked" model and a separate RDMDC for comparison)
#      ATTENTION: Costly part
# ===========================================================================
if COMPUTE_PROJECTION_ERRORS:
    print("Computing orthogonal projection errors ...")

    romTracked = TrackedFluidSurrog(
        reTrainThres=200000, maxLen=1000,
        updateBasis=True, updateThres=500000,
        automatic_weight=True, eps_for_automatic_weight=4.,
        automatic_weight_at_retrain=True, updateOmega=False,
        output_folder=OUTPUT_DIR,
    )
    romTracked.train(
        yData, fieldsData_pre, fieldsData_post,
        **COMMON_TRAIN_KWARGS,
        params                   = param_ins,
        multiple_param_regressor = True,
        cleanup                  = True,
        alg                      = "snap",
    )
    romTracked.initialize_predictions(np.array([[0.9]]))

    romRdmdcInt = RDMDC(epsilon=0.01, lambdaForgetBasis=0.95, lambdaForget=0.95)
    romRdmdcInt.decompose(
        np.hstack(fieldsData_pre), rank=RANK,
        Y        = np.hstack(fieldsData_post),
        u_input  = np.hstack(yData),
        precomp_std   = romTracked.reducLoad.rom.snap_norms,
        precomp_mean  = romTracked.reducLoad.rom.mean_flow,
        precomputed_modes = romTracked.reducLoad.pod.modes.copy(),
    )

    relGlobalOrthNorms     = []
    relOrthNormsNoUpdate   = []
    relRdmdcInitializedNorm = []
    localGrouseNorm        = []

    for i in range(N_TST):
        trueField     = fieldsData_post_tst[0][:, [i]]
        trueFieldNorm = np.linalg.norm(trueField)

        relGlobalOrthNorms.append(
            np.linalg.norm(trueField
                           - fluidSurr2.reducLoad.decode(fluidSurr2.reducLoad.encode(trueField)))
            / trueFieldNorm
        )
        relOrthNormsNoUpdate.append(
            np.linalg.norm(trueField
                           - fluidSurr3.reducLoad.decode(fluidSurr3.reducLoad.encode(trueField)))
            / trueFieldNorm
        )
        relRdmdcInitializedNorm.append(
            np.linalg.norm(
                trueField - romRdmdcInt.rom.decenter(romRdmdcInt.rom.denormalize(
                    romRdmdcInt.pod.inverse_project(romRdmdcInt.pod.project(
                        romRdmdcInt.rom.normalize(romRdmdcInt.rom.center(trueField))
                    ))))
            ) / trueFieldNorm
        )
        localGrouseNorm.append(
            np.linalg.norm(
                trueField - romTracked.reducLoad.rom.decenter(
                    romTracked.reducLoad.rom.denormalize(
                        romTracked.cloneBasis @ romTracked.cloneBasis.T
                        @ romTracked.reducLoad.rom.normalize(
                            romTracked.reducLoad.rom.center(trueField)
                        )))
            ) / trueFieldNorm
        )

        romRdmdcInt.pod.update(romRdmdcInt.rom.normalize(romRdmdcInt.rom.center(trueField)))
        rank1_update(
            romTracked.cloneBasis,
            romTracked.reducLoad.rom.normalize(romTracked.reducLoad.rom.center(trueField)),
            None,
        )

    # Convert to arrays for saving / plotting
    relGlobalOrthNorms      = np.array(relGlobalOrthNorms)
    relOrthNormsNoUpdate    = np.array(relOrthNormsNoUpdate)
    relRdmdcInitializedNorm = np.array(relRdmdcInitializedNorm)
    localGrouseNorm         = np.array(localGrouseNorm)

    # Persist projection-error arrays so they can be reloaded independently
    np.save(f"{OUTPUT_DIR}/relGlobalOrthNorms.npy",      relGlobalOrthNorms)
    np.save(f"{OUTPUT_DIR}/relOrthNormsNoUpdate.npy",    relOrthNormsNoUpdate)
    np.save(f"{OUTPUT_DIR}/relRdmdcInitializedNorm.npy", relRdmdcInitializedNorm)
    np.save(f"{OUTPUT_DIR}/localGrouseImprovedNorm.npy", localGrouseNorm)

    # Persist prediction error arrays
    np.save(f"{OUTPUT_DIR}/errs.npy",     errs)
    np.save(f"{OUTPUT_DIR}/errsNln.npy",  errsNln)
    np.save(f"{OUTPUT_DIR}/errs2.npy",    errs2)
    np.save(f"{OUTPUT_DIR}/errs3.npy",    errs3)
    np.save(f"{OUTPUT_DIR}/errs4.npy",    errs4)
    np.save(f"{OUTPUT_DIR}/errsRdmd.npy", errsRdmd)
    np.save(f"{OUTPUT_DIR}/errsPast.npy", errsPast)


    # ===========================================================================
    # 12.  FIGURE 3 – Orthogonal projection error comparison
    # ===========================================================================
    print("Generating figCompareProjErrors.pdf ...")

    lc_proj = cycler(color=["#E69F00", "#56B4E9", "#F0E442"])
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    ax.set_prop_cycle(lc_proj)
    ax.semilogy(timeTst, relGlobalOrthNorms,      label=r"\textnormal{Global basis}")
    ax.semilogy(timeTst, relOrthNormsNoUpdate,    label=r"\textnormal{Local - w/o update}")
    ax.semilogy(timeTst, relRdmdcInitializedNorm, label=r"\textnormal{Local - PAST}")
    ax.semilogy(timeTst, localGrouseNorm, "k",    label=r"\textnormal{Local - Grouse}")
    ax.set_ylabel(r"\textnormal{Relative orthogonal projection error}", fontsize=11)
    ax.set_xlabel(r"\textnormal{Time [s] (Online)}", fontsize=11)
    ax.grid()
    ax.legend(fontsize=11, loc="best")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/figCompareProjErrors.pdf", bbox_inches="tight")
    plt.close(fig)
else:
    print("Skipping orthogonal projection error computation. Re-run without --skip-projection-errors to include it.")

# ===========================================================================
# 13.  FIGURE 4 – Online relative error: all models
# ===========================================================================
print("Generating figCompareFieldsComplete.pdf ...")

lc_err = cycler(color=["#88CCEE", "#44AA99", "#CC79A7", "#DDCC77",
                        "#332288", "#ffa600", "#C7DD77", "#882255", "#AA4499"])
fig, ax = plt.subplots(figsize=(6.5, 3.4))
ax.set_prop_cycle(lc_err)
ax.semilogy(timeTst, errs2,    linewidth=0.9, label=r"\textnormal{Global static $\mathcal{V}$ - Linear $\mathcal{I}$}")
ax.semilogy(timeTst, errs2Nln, linewidth=0.9, label=r"\textnormal{Global static $\mathcal{V}$ - Nln $\mathcal{I}$}")
ax.semilogy(timeTst, errs4,    linewidth=0.9, label=r"\textnormal{Global static $\mathcal{V}$ - Parametric $\mathcal{I}$}")
ax.semilogy(timeTst, errsRdmd, linewidth=0.9, label=r"\textnormal{rDMDc}")
ax.semilogy(timeTst, errs,     linewidth=0.9, label=r"\textnormal{Current - Linear $\mathcal{I}$}")
ax.semilogy(timeTst, errs3,    linewidth=0.9, label=r"\textnormal{Local static $\mathcal{V}$ - Linear $\mathcal{I}$}")
ax.semilogy(timeTst, errsPast, linewidth=0.9, label=r"\textnormal{Current (PAST) - Nln $\mathcal{I}$}")
ax.semilogy(timeTst, errsNln,  "k", alpha=0.8, linewidth=0.9, label=r"\textnormal{Current - Nln $\mathcal{I}$}")
ax.legend(shadow=False, fontsize=7.5, ncol=2, loc="best")
ax.set_xlabel(r"\textnormal{Time [s]}", fontsize=9)
ax.set_ylabel(r"\textnormal{Relative error} $e^n$", fontsize=9)
ax.tick_params(axis="both", which="major", labelsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/figCompareFieldsComplete.pdf", bbox_inches="tight")
plt.close(fig)

# ===========================================================================
# 14.  FIGURE 5 – GROUSE vs PAST in the latent space
# ===========================================================================
print("Generating compareGrouseAndPast.pdf ...")

fig, ax = plt.subplots(1, 2, figsize=(6.3, 3.2), sharey=False)
ax[0].set_prop_cycle(lc_mid)
ax[1].set_prop_cycle(lc_mid)

k_gp = 14;  j_gp = 28;  strtIncremGP = 900

new_input_nln  = np.vstack((yData_tst[0], fluidSurrNln.reducLoad.encode(fieldsData_pre_tst[0])))
true_out_nln   = fluidSurrNln.reducLoad.encode(fieldsData_post_tst[0])
new_input_past = np.vstack((yData_tst[0], fluidSurrPast.reducLoad.encode(fieldsData_pre_tst[0])))
true_out_past  = fluidSurrPast.reducLoad.encode(fieldsData_post_tst[0])

ax[0].plot(new_input_nln[k_gp, strtIncremGP:],  true_out_nln[j_gp, strtIncremGP:],
           "k", linewidth=1.3, label=r"\textnormal{True}  $\boldsymbol{x}_{*}^{n, r}$")
ax[1].plot(new_input_past[k_gp, strtIncremGP:], true_out_past[j_gp, strtIncremGP:],
           "k", linewidth=1.3, label=r"\textnormal{True}  $\boldsymbol{x}_{*}^{n, r}$")

new_preds_nln  = np.empty((fluidSurrNln._p,  fluidSurrNln.reducLoad.latent_dim,  yData_tst[0].shape[1]))
new_preds_past = np.empty((fluidSurrPast._p, fluidSurrPast.reducLoad.latent_dim, yData_tst[0].shape[1]))

for i in range(fluidSurrNln._p):
    xTest = np.vstack((
        yData_tst[0],
        fluidSurrNln.reducLoadLocals[i].encode(fieldsData_pre_tst[0]),
    ))
    xTestCalibed = xTest.copy()
    xTestCalibed[-RANK:, :] = fluidSurrNln.calibrationQs[i] @ xTestCalibed[-RANK:, :]
    new_preds_nln[i] = fluidSurrNln.regressor[i].predict(xTest)
    new_preds_nln[i] = fluidSurrNln.calibrationQs[i] @ new_preds_nln[i]
    if i < fluidSurrNln._p - 3:
        ax[0].plot(xTestCalibed[k_gp, strtIncremGP:], new_preds_nln[i, j_gp, strtIncremGP:],
                   "-", linewidth=0.6, alpha=0.4,
                   label=r"$\widehat{\boldsymbol{x}}_{" + str(i + 1) + r"}^{n, r}$")

new_input_nln2 = np.vstack((yData_tst[0], fluidSurrNln.reducLoad.encode(fieldsData_pre_tst[0])))
ax[0].plot(new_input_nln2[k_gp, strtIncremGP:],
           (np.dot(new_preds_nln.T, fluidSurrNln.reducLoad.weights).T)[j_gp, strtIncremGP:],
           "-", linewidth=0.8, alpha=0.8,
           label=r"$\textnormal{Predicted }\widehat{\boldsymbol{x}}_{*}^{n, r}$")

for i in range(fluidSurrPast._p):
    xTest = np.vstack((
        yData_tst[0],
        fluidSurrPast.reducLoadLocals[i].encode(fieldsData_pre_tst[0]),
    ))
    xTestCalibed = xTest.copy()
    xTestCalibed[-RANK:, :] = fluidSurrPast.calibrationQs[i] @ xTestCalibed[-RANK:, :]
    new_preds_past[i] = fluidSurrPast.regressor[i].predict(xTest)
    new_preds_past[i] = fluidSurrPast.calibrationQs[i] @ new_preds_past[i]
    if i < fluidSurrPast._p - 3:
        ax[1].plot(xTestCalibed[k_gp, strtIncremGP:], new_preds_past[i, j_gp, strtIncremGP:],
                   "-", linewidth=0.6, alpha=0.4,
                   label=r"$\widehat{\boldsymbol{x}}_{" + str(i + 1) + r"}^{n, r}$")

ax[1].plot(new_input_past[k_gp, strtIncremGP:],
           (np.dot(new_preds_past.T, fluidSurrPast.reducLoad.weights).T)[j_gp, strtIncremGP:],
           "-", linewidth=0.8, alpha=0.8,
           label=r"$\textnormal{Predicted }\widehat{\boldsymbol{x}}_{*}^{n, r}$")

ax[0].set_xlabel(r" $" + str(k_gp) + r"^{th}$ \textnormal{Component of} $\boldsymbol{z}^{n, r}_{*}$", fontsize=8)
ax[1].set_xlabel(r" $" + str(k_gp) + r"^{th}$ \textnormal{Component of} $\boldsymbol{z}^{n, r}_{*}$", fontsize=8)
ax[0].set_ylabel(r" $" + str(j_gp + 1) + r"^{th}$ \textnormal{Component of} $\boldsymbol{x}^{n, r}_{k}$", fontsize=8)
ax[0].set_title(r"$\textnormal{Using GROUSE}$", fontsize=8)
ax[1].set_title(r"$\textnormal{Using PAST}$",   fontsize=8)
for a in ax:
    a.legend(loc="best", ncol=2, fontsize=6.5)
    a.tick_params(axis="both", which="major", labelsize=8)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/compareGrouseAndPast.pdf", bbox_inches="tight")
plt.close(fig)

# ===========================================================================
# 15.  FIGURE 8 – x-velocity signal at the middle inlet point
# ===========================================================================
print("Generating xVelocitySignalMiddleInlet.pdf ...")

# Find the grid point at the minimum x (inlet) closest to mid-y
min_x = np.min(grid_points[:, 0])
candidate_idx = np.where(grid_points[:, 0] == min_x)[0]
candidates = grid_points[candidate_idx]
y_mid = (candidates[:, 1].min() + candidates[:, 1].max()) / 2
local_idx = np.argmin(np.abs(candidates[:, 1] - y_mid))
inlet_idx = candidate_idx[local_idx]

fig, ax = plt.subplots(figsize=(6.3, 2.2))
ax.plot(timeTst, fieldsData_post_tst[0][inlet_idx, :],
        color="darkslategray", linewidth=0.7)
ax.set_xlabel(r"\textnormal{Time [s]}", fontsize=9)
ax.set_ylabel(r"$v_y$ \textnormal{[m/s]}", fontsize=9)
ax.tick_params(axis="both", which="major", labelsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/xVelocitySignalMiddleInlet.pdf", bbox_inches="tight")
plt.close(fig)

# ===========================================================================
# 16.  FIGURE 9 – Initial vs final basis mode absolute x-velocity fields
#      Identifies the two modes whose velocity direction changes most
#      (lowest scalar product between initial and evolved basis vectors)
# ===========================================================================
print("Generating modesFieldsComparison.pdf ...")

# Scalar product between initial-basis and GROUSE-evolved-basis mode norms
scalarP = []
for i in range(fluidSurr.reducLoad.latent_dim):
    vec1x    = fluidSurr.reducLoadLocals[fluidSurr._p0].pod.modes[:N_GRID, i]
    vec1Norm = np.abs(vec1x)
    vec1 = vec1Norm / np.linalg.norm(vec1Norm)

    vec2x = fluidSurr.cloneBasis[:N_GRID, i]
    vec2Norm = np.abs(vec2x)
    vec2 = vec2Norm / np.linalg.norm(vec2Norm)

    scalarP.append(np.dot(vec1, vec2))

ids_ = np.argsort(scalarP)
mode_indices = [ids_[1], ids_[0]]          # two most-changed modes
mode_labels  = ["1st", "2nd"]
row_labels   = [r"\textnormal{Initial basis}", r"\textnormal{Final basis}"]
basis_sources = [
    lambda midx: fluidSurr.reducLoadLocals[fluidSurr._p0].pod.modes[:, midx],
    lambda midx: fluidSurr.cloneBasis[:, midx],
]

fig, axes = plt.subplots(2, 2, figsize=(6.3, 3.4))
for row in range(2):
    for col, (midx, mlabel) in enumerate(zip(mode_indices, mode_labels)):
        state   = basis_sources[row](midx)
        Vx      = state[:N_GRID]
        V_shown = np.abs(Vx)

        ax = axes[row, col]
        vmin, vmax = V_shown.min(), V_shown.max()
        c0 = ax.tricontourf(grid_points[:, 0], grid_points[:, 1], V_shown,
                            cmap="RdYlBu_r", vmin=vmin, vmax=vmax, levels=100)
        fig.colorbar(c0, ax=ax, fraction=0.046, pad=0.04)
        ax.set_aspect("equal")

        if row == 0:
            ax.set_title(f"\\textnormal{{{mlabel} mode}}", fontsize=8)
        if col == 0:
            ax.set_ylabel(row_labels[row], fontsize=8)
            ax.tick_params(axis="y", which="both", labelsize=7)
        else:
            ax.set_yticks([])
        if row == 1:
            ax.tick_params(axis="x", which="both", labelsize=7)
        else:
            ax.set_xticks([])

fig.tight_layout(h_pad=0.4, w_pad=0.6)
fig.savefig(f"{OUTPUT_DIR}/modesFieldsComparison.pdf", bbox_inches="tight")
plt.close(fig)

print("\nAll figures saved to:", OUTPUT_DIR)
print("Done.")
