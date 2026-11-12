"""
Two ablations on the Turek-Hron FSI benchmark, reusing the existing data (no new simulation).

  W. Weighting of the local regressions. The Grassmann-distance weights are compared with weights computed
     in the (range-normalized) parameter space (inverse-distance, nearest neighbour, uniform) on the
     pre-trained model (no online data), for the benchmark parameter (1, 1). The four training parameters
     are the corners of the square around it, so the three parameter-space weightings coincide.
     Reported as a table (printed and saved), no figure.

  O. Use of online data. For the benchmark parameter (1, 1), the pre-trained model (neither regression
     retraining nor basis update) is compared to regression retraining only and to the full proposed model
     (regression retraining and basis update), all with the same nonlinear regression.

Outputs are saved in the output directory (default: results_figures_arrays/).
"""

import argparse
import copy
import os
import shutil
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

sys.path.append("./trainScripts/")

from importData import importData
from rom_am.fluid_surrogate import FluidSurrog
from rom_am.tracked_fluid_surrogate import TrackedFluidSurrog

DT_NAME = "dt01"
DT = 0.01
REMOVE_DTS_SINGLE = int(8 / 0.008)
CUTOFF_INCR = int(20 / DT) - 1
REMOVED_DOFS = [896, 897, 900, 901]
RANK = 24
TRAIN_CODES = ["vel0-9rho0-9", "vel0-9rho1-1", "vel1-1rho1-1", "vel1-1rho0-9"]
PARAM_INS = np.array([
    [0.9, 0.9, 1.1, 1.1],
    [0.9, 1.1, 1.1, 0.9],
])
PARAM_RANGE = PARAM_INS.max(axis=1) - PARAM_INS.min(axis=1)
BENCHMARK = np.array([[1.0], [1.0]])
TEST_CODE = "vel1-1rho1-0-longer"  # name of the 40 s benchmark run in trainData/
RETRAIN_THRES = 60
MAX_LEN = 5000
UPDATE_THRES = 200
NEVER = 10**9
# Number of online steps of the weighting study (None: the whole test sequence)
WEIGHTING_STEPS = None
SIDE_FOLDER = "results_figures_arrays/coSimData_ablation"
SCHEMES = ["Grassmann", "parameter IDW", "nearest (parameter)", "uniform"]


def parse_args():
    parser = argparse.ArgumentParser(description="Weighting and online-data ablations (Turek-Hron).")
    parser.add_argument("--data-root", default="./trainData")
    parser.add_argument("--output-dir", default="results_figures_arrays")
    parser.add_argument("--skip-weighting", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    return parser.parse_args()


def relative_error(pred, true):
    return np.linalg.norm(pred - true, axis=0) / np.linalg.norm(true, axis=0)


def import_kwargs():
    return dict(remove_noise=True, remove_corners=True, cutoff_incr=[CUTOFF_INCR], removed_dofs=REMOVED_DOFS)


def train_names(data_root):
    return [f"{data_root}/{DT_NAME}/{code}/coSimData/" for code in TRAIN_CODES]


def setup_solid_rom(names):
    """Solid ROM as in reproduce_turek_case.py, refreshed against the interpolated fluid reducer."""
    remove_dts = [REMOVE_DTS_SINGLE] * len(names)
    (_, _, _, _, loadSol, dispSol, dispConvSol, _) = importData(names, remove_dts, **import_kwargs())
    solidROM = FluidSurrog(maxLen=11000, reTrainThres=780)
    solidROM.train(loadSol, dispConvSol, dispSol, rank_pres=18, rank_disp=3, smoothing=1e-6,
                   kernel="polyC", degree=1, norm=[True, True], center=[True, True],
                   normalization=["max", "max"], norm_regr="max", solidReduc=None)
    solidReducSaved = copy.deepcopy(solidROM.reducLoad)
    return loadSol, dispSol, dispConvSol, solidReducSaved


def refresh_solid_rom(solid_data, solidReducSaved, fluid_reduc):
    loadSol, dispSol, dispConvSol = solid_data
    solidROM = FluidSurrog(maxLen=11000, reTrainThres=780)
    solidROM.train(loadSol, dispConvSol, dispSol, rank_pres=18, rank_disp=3, smoothing=1e-6,
                   kernel="polyC", degree=1, norm=[True, True], center=[True, True],
                   normalization=["max", "max"], norm_regr="max", solidReduc=fluid_reduc,
                   precomputedReducLoad=solidReducSaved)
    return solidROM


def build(names, solidReducSaved, reTrainThres, updateThres, updateBasis):
    remove_dts = [REMOVE_DTS_SINGLE] * len(names)
    loadD, dispD, loadConvD, *_ = importData(names, remove_dts, return_list=True, **import_kwargs())
    model = TrackedFluidSurrog(
        reTrainThres=reTrainThres, maxLen=MAX_LEN, updateBasis=updateBasis, updateThres=updateThres,
        updateOmega=True, output_folder=SIDE_FOLDER)
    model.train(dispD, loadConvD, loadD, rank_pres=RANK, smoothing=1e-5, kernel="polyC", degree=2,
                norm=[False, True], norm_regr="std", normalization=["max", "norm"], params=PARAM_INS,
                weights=True, solidReduc=solidReducSaved, multiple_param_regressor=True, cleanup=True)
    return model


def load_test(data_root):
    (loadT, dispT, loadConvT, *_) = importData(
        [f"{data_root}/{DT_NAME}/{TEST_CODE}/coSimData/"], [REMOVE_DTS_SINGLE],
        remove_noise=True, remove_corners=True, cutoff_incr=[int(40 / DT) - 1], removed_dofs=REMOVED_DOFS)
    return loadT, dispT, loadConvT


def run_online(model, solidROM, test, num_steps=None):
    loadT, dispT, loadConvT = test
    n = loadT.shape[1] if num_steps is None else num_steps
    model.initialize_predictions(BENCHMARK)
    out = np.empty_like(loadT[:, :n])
    for i in range(n):
        out[:, i] = model.predict(dispT[:, [i]], loadConvT[:, [i]], solidReduc=solidROM.reducLoad,
                                  params=BENCHMARK).ravel()
        model.augmentData(dispT[:, [i]], loadConvT[:, [i]], loadT[:, [i]], solidReduc=solidROM.reducLoad,
                          params=BENCHMARK, stepsize="complete", computeAngle=False)
    return relative_error(out, loadT[:, :n])


def parameter_weights(scheme):
    d = np.linalg.norm((PARAM_INS - BENCHMARK) / PARAM_RANGE[:, None], axis=0)
    if scheme == "parameter IDW":
        w = d ** -2.0
    elif scheme == "nearest (parameter)":
        w = np.isclose(d, d.min()).astype(float)
    else:
        w = np.ones(PARAM_INS.shape[1])
    return w / w.sum()


def prepare(data_root):
    names = train_names(data_root)
    loadSol, dispSol, dispConvSol, solidReducSaved = setup_solid_rom(names)
    ref_model = build(names, solidReducSaved, NEVER, NEVER, False)
    ref_model.initialize_predictions(BENCHMARK)
    solidROM = refresh_solid_rom((loadSol, dispSol, dispConvSol), solidReducSaved, ref_model.reducLoad)
    return names, solidReducSaved, solidROM, ref_model


def study_weighting(data_root, out_dir):
    print("=" * 70)
    print("Study W: weighting of the local regressions (pre-trained model)")
    print("=" * 70)
    names, solidReducSaved, solidROM, model = prepare(data_root)
    test = load_test(data_root)
    grassmann_w = model.reducLoad.weights.copy()
    med, mean = np.zeros(len(SCHEMES)), np.zeros(len(SCHEMES))
    for b, s in enumerate(SCHEMES):
        model.reducLoad.weights = grassmann_w.copy() if s == "Grassmann" else parameter_weights(s)
        err = run_online(model, solidROM, test, WEIGHTING_STEPS)[:-4]
        med[b], mean[b] = np.median(err), err.mean()
        print(f"  {s:20s} weights={np.round(model.reducLoad.weights, 3)} median={med[b]:.3e} mean={mean[b]:.3e}")
    np.savez(f"{out_dir}/ablation_weighting_turek.npz", med=med[None], mean=mean[None], schemes=SCHEMES)


def study_online_data(data_root, out_dir):
    print("=" * 70)
    print("Study O: use of online data (benchmark parameter)")
    print("=" * 70)
    names, solidReducSaved, solidROM, _ = prepare(data_root)
    test = load_test(data_root)
    configs = [
        ("pre-trained only", NEVER, NEVER, False),
        ("regression retraining only", RETRAIN_THRES, NEVER, False),
        ("regression retraining + basis update (proposed)", RETRAIN_THRES, UPDATE_THRES, True),
    ]
    errs = {}
    for name, rt, ut, ub in configs:
        t0 = time.time()
        model = build(names, solidReducSaved, rt, ut, ub)
        err = run_online(model, solidROM, test)[:-4]
        errs[name] = err
        print(f"  {name:50s} median={np.median(err):.3e} mean={err.mean():.3e} "
              f"last-quarter median={np.median(err[3 * len(err) // 4:]):.3e}  [{time.time() - t0:.0f} s]")
        np.savez(f"{out_dir}/ablation_online_data_turek.npz", names=list(errs.keys()),
                 **{f"e{i}": e for i, e in enumerate(errs.values())})
    return errs


def make_figure(out_dir, o):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    styles = [("#999999", 0.9), ("#0072B2", 0.9), ("k", 1.1)]
    window = 20
    for (name, e), (col, lw) in zip(o.items(), styles):
        ax.semilogy(np.convolve(e, np.ones(window) / window, mode="valid"), color=col, lw=lw, label=name)
    ax.set_xlabel("online steps")
    ax.set_ylabel("relative error")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/figAblationsTurek.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.skip_weighting:
        study_weighting(args.data_root, args.output_dir)
    o = None if args.skip_ablation else study_online_data(args.data_root, args.output_dir)
    if o is None and os.path.exists(f"{args.output_dir}/ablation_online_data_turek.npz"):
        d = np.load(f"{args.output_dir}/ablation_online_data_turek.npz")
        o = {n: d[f"e{i}"] for i, n in enumerate(list(d["names"]))}
    if o is not None:
        make_figure(args.output_dir, o)
    shutil.rmtree(SIDE_FOLDER, ignore_errors=True)
    print("Done.")
