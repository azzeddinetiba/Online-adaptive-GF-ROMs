"""
Sensitivity of the Grassmann interpolation to the reference subspace (lid-driven cavity, dt = 0.3 s).

Two studies, both reusing the existing training data (no new simulation required):

  A. Reference sweep on the benchmark test parameter: each of the p = 15 training bases is used in turn
     as the reference of the tangent space. Reports the projection error of the initial predicted basis,
     the online prediction error of the proposed model (nonlinear regression), and the subspace distance
     between the bases obtained with different references (initially and after the online adaptation).

  B. Leave-one-out over the training grid, for every possible reference of the remaining 14 bases.
     Removed corners lie outside the convex hull of the remaining parameters (extrapolation), removed
     edge points on its boundary, and removed interior points inside it. The projection error of the
     held-out snapshots onto the predicted basis is compared to that of the global static basis.

Outputs are saved in the output directory (default: results_figures_arrays/).
"""

import argparse
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["text.latex.preamble"] = " \\usepackage{amsmath}"
plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

sys.path.append("./dataLoadingScripts/")

from importData import importData
from rom_am.fluid_surrogate import FluidSurrog
from rom_am.tracked_fluid_surrogate import TrackedFluidSurrog
from rom_am.dimreducers.rom_am.podReducer import PodReducer
from rom_am.dimreducers.rom_am.manifReducer import ManifInterpReducer
from rom_am.utils import angles

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
RANK = 8
M_WEIGHT = 2  # exponent z of the Grassmann-distance weights

REMOVE_DTS = [94] * P
CUTOFF_INCR = [299]


def parse_args():
    parser = argparse.ArgumentParser(description="Reference-subspace sensitivity study.")
    parser.add_argument("--data-root", default="./fomData")
    parser.add_argument("--output-dir", default="results_figures_arrays")
    parser.add_argument("--skip-online", action="store_true",
                        help="Skip study A (reference sweep with online runs).")
    parser.add_argument("--skip-loo", action="store_true", help="Skip study B (leave-one-out).")
    return parser.parse_args()


def principal_angles(A, B):
    angs, _ = angles(A, B)
    return angs


def projection_errors(reducer, X):
    """Column-wise relative error of the projection of X on the reducer's subspace."""
    return np.linalg.norm(X - reducer.decode(reducer.encode(X)), axis=0) / np.linalg.norm(X, axis=0)


def relative_error(pred, true):
    return np.linalg.norm(pred - true, axis=0) / np.linalg.norm(true, axis=0)


def load_train(names, return_list=True):
    return importData(names, REMOVE_DTS, remove_noise=True, return_list=return_list,
                      remove_corners=True, cutoff_incr=CUTOFF_INCR)


def category(k):
    i, j = divmod(k, 3)
    on_i = i in (0, 4)
    on_j = j in (0, 2)
    if on_i and on_j:
        return "corner (extrapolation)"
    if on_i or on_j:
        return "edge (hull boundary)"
    return "interior"


# ---------------------------------------------------------------------------
# Study B: leave-one-out
# ---------------------------------------------------------------------------
def run_loo(names, out_dir):
    print("=" * 70)
    print("Study B: leave-one-out over the training grid")
    print("=" * 70)
    fluid, *_ = load_train(names, return_list=True)
    rows = []
    e_pred = np.full((P, P - 1), np.nan)
    ang_pred = np.full((P, P - 1), np.nan)
    e_oracle = np.empty(P)
    e_global = np.empty(P)

    for k in range(P):
        rest = [i for i in range(P) if i != k]
        data_rest = [fluid[i] for i in rest]
        prm_rest = PARAM_INS[:, rest]
        mean_ = np.mean([d.mean(axis=1) for d in data_rest], axis=0)
        std_ = np.max(np.abs(np.hstack(data_rest)), axis=1)

        def local(data):
            red = PodReducer(latent_dim=RANK)
            red.train(data.copy(), precomp_mean=mean_, normalization="max", normalize=False,
                      precomp_std=std_, to_copy=False)
            return red

        locals_ = [local(d) for d in data_rest]
        oracle = local(fluid[k])
        glob = PodReducer(latent_dim=RANK)
        glob.train(np.hstack(data_rest), normalize=False, center=True, to_copy=True)

        X = fluid[k]
        e_oracle[k] = np.median(projection_errors(oracle, X))
        e_global[k] = np.median(projection_errors(glob, X))

        for j in range(P - 1):
            manif = ManifInterpReducer(RANK, m=M_WEIGHT)
            manif.train(locals_, prm_rest, precomp_mean=mean_, precomp_std=std_, refP=j)
            manif.predictNewModes(PARAM_INS[:, [k]], locals_)
            e_pred[k, j] = np.median(projection_errors(manif, X))
            ang_pred[k, j] = np.rad2deg(principal_angles(manif.pod.modes, oracle.pod.modes).max())

        rows.append(k)
        print(f"  held-out {k:2d} {category(k):24s} theta=({PARAM_INS[0, k]:.3f}, {PARAM_INS[1, k]:.2f})"
              f"  oracle={e_oracle[k]:.2e} global={e_global[k]:.2e}"
              f"  predicted: min={e_pred[k].min():.2e} med={np.median(e_pred[k]):.2e} max={e_pred[k].max():.2e}"
              f"  max-angle-to-oracle [deg]: {ang_pred[k].min():.2f}-{ang_pred[k].max():.2f}")

    np.savez(f"{out_dir}/ref_sensitivity_loo.npz", e_pred=e_pred, ang_pred=ang_pred,
             e_oracle=e_oracle, e_global=e_global, params=PARAM_INS)
    return e_pred, ang_pred, e_oracle, e_global


# ---------------------------------------------------------------------------
# Study A: reference sweep on the benchmark test parameter
# ---------------------------------------------------------------------------
def run_reference_sweep(data_root, names, out_dir):
    print("=" * 70)
    print("Study A: reference sweep on the benchmark test parameter")
    print("=" * 70)
    test_names = [f"{data_root}/dt03/10mu10/coSimData/"]
    test_remove_dts = [94]

    (_, _, _, _, loadSol, dispSol, dispConvSol, _) = importData(
        names, REMOVE_DTS, remove_corners=True, remove_noise=True, cutoff_incr=CUTOFF_INCR)
    solidROM = FluidSurrog(maxLen=11000, reTrainThres=780)
    solidROM.train(loadSol, dispConvSol, dispSol, rank_pres=0.999999, rank_disp=20, smoothing=1e-6,
                   kernel="polyC", degree=1, norm=[True, True], center=[True, True],
                   normalization=["max", "max"], norm_regr="max", solidReduc=None)

    (loadTest, dispTest, loadConvTest, *_) = importData(
        test_names, test_remove_dts, remove_noise=True, remove_corners=True, cutoff_incr=CUTOFF_INCR)

    static_err = np.empty(P)
    online_med = np.empty(P)
    online_mean = np.empty(P)
    online_last = np.empty(P)
    phi0, phiN = [], []

    for ref in range(P):
        t0 = time.time()
        loadD, dispD, loadConvD, *_ = load_train(names, return_list=True)
        fluidSurr = TrackedFluidSurrog(
            reTrainThres=50, maxLen=2100, updateBasis=True, updateThres=120, updateOmega=True,
            output_folder="results_figures_arrays/coSimData_refsweep")
        fluidSurr.train(
            dispD, loadConvD, loadD, rank_pres=RANK, smoothing="auto", kernel="polyC", degree=2,
            norm=[False, True], norm_regr="max", normalization=["max", "norm"],
            params=PARAM_INS, weights=True, solidReduc=solidROM.reducLoad,
            multiple_param_regressor=True, cleanup=True, refP=ref)
        fluidSurr.initialize_predictions(np.array([[1.0], [1.0]]))
        phi0.append(fluidSurr.cloneBasis.copy())
        static_err[ref] = np.median(projection_errors(fluidSurr.reducLoad, loadTest)[-204:-4])

        results = np.empty_like(loadTest)
        for i in range(dispTest.shape[1]):
            results[:, i] = fluidSurr.predict(
                dispTest[:, [i]], loadConvTest[:, [i]],
                solidReduc=solidROM.reducLoad, params=np.array([[1.0], [1.0]])).ravel()
            fluidSurr.augmentData(
                dispTest[:, [i]], loadConvTest[:, [i]], loadTest[:, [i]],
                solidReduc=solidROM.reducLoad, params=np.array([[1.0], [1.0]]),
                stepsize=None, computeAngle=False)
        phiN.append(fluidSurr.cloneBasis.copy())
        err = relative_error(results, loadTest)[-204:-4] # The error on the last 200 steps
        online_med[ref], online_mean[ref], online_last[ref] = np.median(err), err.mean(), np.median(err[-50:])
        print(f"  ref={ref:2d}  static e_perp (median)={static_err[ref]:.3e}  "
              f"online error: median={online_med[ref]:.3e} mean={online_mean[ref]:.3e}  "
              f"[{time.time() - t0:.0f} s]")

    ang0 = np.array([np.rad2deg(principal_angles(phi0[0], phi0[r]).max()) for r in range(P)])
    angN = np.array([np.rad2deg(principal_angles(phiN[0], phiN[r]).max()) for r in range(P)])
    print("  max angle [deg] between initial bases (ref r vs ref 0):", np.round(ang0, 3))
    print("  max angle [deg] between final bases   (ref r vs ref 0):", np.round(angN, 3))
    np.savez(f"{out_dir}/ref_sensitivity_sweep.npz", static_err=static_err, online_med=online_med,
             online_mean=online_mean, online_last=online_last, ang0=ang0, angN=angN)
    return static_err, online_med, online_mean, ang0, angN


def make_figure(out_dir, loo, sweep):
    ncols = (loo is not None) + (sweep is not None)
    fig, axs = plt.subplots(1, ncols, figsize=(6.5 * ncols, 3.6), squeeze=False)
    axs = axs[0]
    c = 0
    if loo is not None:
        e_pred, _, _, e_global = loo
        order = sorted(range(P), key=lambda k: (["interior", "edge (hull boundary)",
                                                  "corner (extrapolation)"].index(category(k)), k))
        ax = axs[c]
        c += 1
        x = np.arange(P)
        for pos, k in enumerate(order):
            if category(k).startswith("corner"):
                ax.axvspan(pos - 0.5, pos + 0.5, color="#CC79A7", alpha=0.2, lw=0,
                           label="$\\textnormal{outside the convex hull}$" if k == order[-4] else None)
            ax.semilogy([pos] * e_pred.shape[1], e_pred[k], ".", color="0.45", alpha=0.7,
                        label="$\\textnormal{predicted basis, every reference}$" if pos == 0 else None)
        ax.semilogy(x, e_global[order], "x--", color="#D55E00", lw=0.8, ms=5, label="$\\textnormal{global static basis}$")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{PARAM_INS[0, k]:g},{PARAM_INS[1, k]:g}" for k in order], rotation=70, fontsize=10)
        ax.set_xlabel("$\\textnormal{held-out} (\\rho, \\mu \\times10^{-2})\\textnormal{: interior | edge | corner}$", fontsize=10)
        ax.set_ylabel("$\\textnormal{median relative projection error}$", fontsize=12)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=10)
    if sweep is not None:
        static_err, online_med, _, _, _ = sweep
        ax = axs[c]
        r = np.arange(1, P + 1)
        ax.semilogy(r, static_err, "o-", color="#0072B2", ms=4, lw=0.8, label="$\\textnormal{initial basis, projection error}$")
        ax.semilogy(r, online_med, "s-", color="k", ms=4, lw=0.8, label="$\\textnormal{online prediction error}$")
        ax.set_xlabel("$\\textnormal{reference basis index}$", fontsize=14)
        ax.set_ylabel("$\\textnormal{median relative error}$", fontsize=14)
        ax.set_xticks(r)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/figReferenceSensitivityLid.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    names = [f"{args.data_root}/train_dt03/{code}/coSimData/" for code in F_MU_CODES]

    loo = None if args.skip_loo else run_loo(names, args.output_dir)
    sweep = None if args.skip_online else run_reference_sweep(args.data_root, names, args.output_dir)

    # A skipped study is plotted from its saved results, if any
    if loo is None and os.path.exists(f"{args.output_dir}/ref_sensitivity_loo.npz"):
        d = np.load(f"{args.output_dir}/ref_sensitivity_loo.npz")
        loo = (d["e_pred"], d["ang_pred"], d["e_oracle"], d["e_global"])
    if sweep is None and os.path.exists(f"{args.output_dir}/ref_sensitivity_sweep.npz"):
        d = np.load(f"{args.output_dir}/ref_sensitivity_sweep.npz")
        sweep = (d["static_err"], d["online_med"], d["online_mean"], d["ang0"], d["angN"])
    make_figure(args.output_dir, loo, sweep)
    print("Done.")
