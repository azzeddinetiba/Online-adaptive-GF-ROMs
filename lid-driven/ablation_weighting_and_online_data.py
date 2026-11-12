"""
Two ablations on the FSI lid-driven cavity (dt = 0.3 s), reusing the existing data (no new simulation).

  W. Weighting of the local regressions. The Grassmann-distance weights are compared with weights computed
     in the (range-normalized) parameter space (inverse-distance, nearest neighbour, uniform) on the
     pre-trained model (no online data), for the benchmark parameter (1, 1) with all 15 training sets.
     Reported as a table (printed and saved), no figure.

  O. Use of online data. For the benchmark parameter (1, 1), the pre-trained model (neither regression
     retraining nor basis update) is compared to regression retraining only and to the full proposed model
     (regression retraining and basis update), all with the same nonlinear regression.

Outputs are saved in the output directory (default: results_figures_arrays/).
"""

import argparse
import os
import shutil
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

sys.path.append("./dataLoadingScripts/")

from importData import importData
from rom_am.fluid_surrogate import FluidSurrog
from rom_am.tracked_fluid_surrogate import TrackedFluidSurrog

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
P = PARAM_INS.shape[1]
PARAM_RANGE = PARAM_INS.max(axis=1) - PARAM_INS.min(axis=1)
BENCHMARK = np.array([[1.0], [1.0]])
RANK = 8
RETRAIN_THRES = 50
MAX_LEN = 2100
UPDATE_THRES = 120
NEVER = 10**9
REMOVE_DTS = [94]
CUTOFF_INCR = [299]
SIDE_FOLDER = "results_figures_arrays/coSimData_ablation"
SCHEMES = ["Grassmann", "parameter IDW", "nearest (parameter)", "uniform"]

TRAIN_KWARGS = dict(
    rank_pres=RANK, smoothing="auto", kernel="polyC", degree=2, norm=[False, True], norm_regr="max",
    normalization=["max", "norm"], weights=True, multiple_param_regressor=True, cleanup=True
)


def parse_args():
    parser = argparse.ArgumentParser(description="Weighting and online-data ablations (lid-driven cavity).")
    parser.add_argument("--data-root", default="./fomData")
    parser.add_argument("--output-dir", default="results_figures_arrays")
    parser.add_argument("--skip-weighting", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    return parser.parse_args()


def relative_error(pred, true):
    return np.linalg.norm(pred - true, axis=0) / np.linalg.norm(true, axis=0)


def category(k):
    i, j = divmod(k, 3)
    on_i, on_j = i in (0, 4), j in (0, 2)
    if on_i and on_j:
        return "corner"
    return "edge" if (on_i or on_j) else "interior"


def import_kwargs():
    return dict(remove_noise=True, remove_corners=True, cutoff_incr=CUTOFF_INCR)


def train_solid_rom(names):
    (_, _, _, _, loadSol, dispSol, dispConvSol, _) = importData(
        names, REMOVE_DTS * len(names), remove_corners=True, remove_noise=True, cutoff_incr=CUTOFF_INCR)
    solidROM = FluidSurrog(maxLen=11000, reTrainThres=780)
    solidROM.train(loadSol, dispConvSol, dispSol, rank_pres=0.999999, rank_disp=20, smoothing=1e-6,
                   kernel="polyC", degree=1, norm=[True, True], center=[True, True],
                   normalization=["max", "max"], norm_regr="max", solidReduc=None)
    return solidROM


def build(fluid_lists, params, solidROM, reTrainThres, updateThres, updateBasis, refP=0):
    loadD, dispD, loadConvD = fluid_lists
    model = TrackedFluidSurrog(
        reTrainThres=reTrainThres, maxLen=MAX_LEN, updateBasis=updateBasis, updateThres=updateThres,
        updateOmega=True, output_folder=SIDE_FOLDER)
    model.train(dispD, loadConvD, loadD, params=params, solidReduc=solidROM.reducLoad, **TRAIN_KWARGS, refP=refP)
    return model


def run_online(model, solidROM, disp, load_conv, load, theta, num_snaps=None):
    if num_snaps is None:
        num_snaps = load.shape[1]
    model.initialize_predictions(theta)
    out = np.empty_like(load[:, :num_snaps])
    for i in range(num_snaps):
        out[:, i] = model.predict(disp[:, [i]], load_conv[:, [i]], solidReduc=solidROM.reducLoad,
                                  params=theta).ravel()
        model.augmentData(disp[:, [i]], load_conv[:, [i]], load[:, [i]], solidReduc=solidROM.reducLoad,
                          params=theta, stepsize=None, computeAngle=False)
    return relative_error(out, load[:, :num_snaps])


def parameter_weights(scheme, train_params, theta):
    d = np.linalg.norm((train_params - theta) / PARAM_RANGE[:, None], axis=0)
    if scheme == "parameter IDW":
        w = d ** -2.0
    elif scheme == "nearest (parameter)":
        w = np.isclose(d, d.min()).astype(float)
    else:
        w = np.ones(train_params.shape[1])
    return w / w.sum()


def weighting_cases(model, solidROM, disp, load_conv, load, train_params, theta):
    """Median and mean error of the pre-trained model for each weighting scheme."""
    model.initialize_predictions(theta)
    grassmann_w = model.reducLoad.weights.copy()
    med, mean = np.zeros(len(SCHEMES)), np.zeros(len(SCHEMES))
    for b, s in enumerate(SCHEMES):
        model.reducLoad.weights = grassmann_w.copy() if s == "Grassmann" else parameter_weights(
            s, train_params, theta)
        err = run_online(model, solidROM, disp, load_conv, load, theta, UPDATE_THRES) # Only the snapshots before the first basis update
        med[b], mean[b] = np.median(err), err.mean()
    return med, mean


def study_weighting(data_root, names, out_dir):
    print("=" * 70)
    print("Study W: weighting of the local regressions (pre-trained model)")
    print("=" * 70)
    loadD, dispD, loadConvD, *_ = importData(
        names, REMOVE_DTS * P, return_list=True, **import_kwargs())
    med = np.zeros((1, len(SCHEMES)))
    mean = np.zeros_like(med)

    # Leave-one-out over all training parameters (disabled)
    # med = np.zeros((P + 1, len(SCHEMES)))
    # mean = np.zeros_like(med)
    # for k in range(P):
    #     t0 = time.time()
    #     rest = [i for i in range(P) if i != k]
    #     names_rest = [names[i] for i in rest]
    #     solidROM = train_solid_rom(names_rest)
    #     model = build(([loadD[i] for i in rest], [dispD[i] for i in rest], [loadConvD[i] for i in rest]),
    #                   PARAM_INS[:, rest], solidROM, NEVER, NEVER, False)
    #     theta = PARAM_INS[:, [k]]
    #     med[k], mean[k] = weighting_cases(model, solidROM, dispD[k], loadConvD[k], loadD[k],
    #                                       PARAM_INS[:, rest], theta)
    #     print(f"  held-out {k:2d} {category(k):9s} theta=({theta[0, 0]:.3f}, {theta[1, 0]:.2f}) "
    #           + "  ".join(f"{s}={m:.2e}" for s, m in zip(SCHEMES, med[k])) + f"  [{time.time() - t0:.0f} s]")

    # Benchmark parameter with the 15 training sets
    solidROM = train_solid_rom(names)
    model = build((loadD, dispD, loadConvD), PARAM_INS, solidROM, NEVER, NEVER, False)
    (loadT, dispT, loadConvT, *_) = importData(
        [f"{data_root}/dt03/10mu10/coSimData/"], REMOVE_DTS, **import_kwargs())
    med[0], mean[0] = weighting_cases(model, solidROM, dispT[:, :-4], loadConvT[:, :-4], loadT[:, :-4],
                                      PARAM_INS, BENCHMARK)
    print("  benchmark (1, 1)  " + "  ".join(f"{s}={m:.2e}" for s, m in zip(SCHEMES, med[0])))
    np.savez(f"{out_dir}/ablation_weighting_lid.npz", med=med, mean=mean, schemes=SCHEMES)
    return med, mean


def study_online_data(data_root, names, out_dir):
    print("=" * 70)
    print("Study O: use of online data (benchmark parameter)")
    print("=" * 70)
    loadD, dispD, loadConvD, *_ = importData(names, REMOVE_DTS * P, return_list=True, **import_kwargs())
    solidROM = train_solid_rom(names)
    (loadT, dispT, loadConvT, *_) = importData(
        [f"{data_root}/dt03/10mu10/coSimData/"], REMOVE_DTS, **import_kwargs())

    configs = [
        ("pre-trained only", NEVER, NEVER, False),
        ("regression retraining only", RETRAIN_THRES, NEVER, False),
        ("regression retraining + basis update (proposed)", RETRAIN_THRES, UPDATE_THRES, True),
    ]
    errs = {}
    for name, rt, ut, ub in configs:
        t0 = time.time()
        model = build((loadD, dispD, loadConvD), PARAM_INS, solidROM, rt, ut, ub)
        err = run_online(model, solidROM, dispT, loadConvT, loadT, BENCHMARK)[:-4]
        errs[name] = err
        print(f"  {name:50s} median={np.median(err):.3e} mean={err.mean():.3e} "
              f"last-quarter median={np.median(err[3 * len(err) // 4:]):.3e}  [{time.time() - t0:.0f} s]")
    np.savez(f"{out_dir}/ablation_online_data_lid.npz", names=list(errs.keys()),
             **{f"e{i}": e for i, e in enumerate(errs.values())})
    return errs


def make_figure(out_dir, o):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    styles = [("#999999", 0.9), ("#0072B2", 0.9), ("k", 1.1)]
    for (name, e), (col, lw) in zip(o.items(), styles):
        ax.semilogy(np.convolve(e, np.ones(3) / 3, mode="valid"), color=col, lw=lw, label=name)
    ax.set_xlabel("online steps")
    ax.set_ylabel("relative error")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/figAblationsLid.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    names = [f"{args.data_root}/train_dt03/{code}/coSimData/" for code in F_MU_CODES]

    if not args.skip_weighting:
        study_weighting(args.data_root, names, args.output_dir)
    o = None if args.skip_ablation else study_online_data(args.data_root, names, args.output_dir)
    if o is None and os.path.exists(f"{args.output_dir}/ablation_online_data_lid.npz"):
        d = np.load(f"{args.output_dir}/ablation_online_data_lid.npz")
        o = {n: d[f"e{i}"] for i, n in enumerate(list(d["names"]))}
    if o is not None:
        make_figure(args.output_dir, o)
    shutil.rmtree(SIDE_FOLDER, ignore_errors=True)
    print("Done.")
