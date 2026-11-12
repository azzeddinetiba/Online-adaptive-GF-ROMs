"""
Two ablations on the flow-induced vibrations benchmark (no new simulation required).

  W. Weighting of the local regressions. The Grassmann-distance weights are compared with weights computed
     in the parameter space (inverse-distance, nearest neighbour, uniform) on the pre-trained model
     (no online data), for the benchmark test frequency f = 0.9 (the other four frequencies are the
     training set). Reported as a table (printed and saved), no figure.

  O. Use of online data. For the benchmark test frequency f = 0.9, the pre-trained model (neither
     regression retraining nor basis update) is compared to regression retraining only and to the full
     proposed model (regression retraining and basis update), all with the same nonlinear regression.

Outputs are saved in the output directory (default: results_figures_arrays/).
"""

import argparse
import os
import shutil
import time

import h5py
import numpy as np
import matplotlib.pyplot as plt

from rom_am.tracked_fluid_surrogate import TrackedFluidSurrog

FIELD_NAMES = ["Vx_grid", "Vy_grid", "p_grid"]
N_GRID = 6405
N_FIELDS = len(FIELD_NAMES)
RANK = 64
REGRESSION_FREQUENCY = 80
BASIS_FREQUENCY = 170
NEVER = 10**9

FREQS = [0.5, 0.7, 0.9, 1.1, 1.3]
# Held-out frequencies of the weighting study. Leave-one-out over all frequencies (disabled):
# WEIGHTING_FREQS = FREQS
WEIGHTING_FREQS = [0.9]
LENGTHS = {0.5: 800, 0.7: 880, 0.9: 1000, 1.1: 1080, 1.3: 1160}
SIDE_FOLDER = "results_figures_arrays/coSimData_ablation"

TRAIN_KWARGS = dict(
    rank_pres=RANK, smoothing=1e-1, kernel="cubic", degree=1, norm=[False, True], norm_regr="max",
    normalization=["max", "norm"], weights=True, solidReduc=None,
    multiple_param_regressor=True, cleanup=True, alg="snap",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Weighting and online-data ablations (VIV benchmark).")
    parser.add_argument("--dataset-file", default="data/oscillating_cylinder_benchmark_dataset_v2.mat")
    parser.add_argument("--output-dir", default="results_figures_arrays")
    parser.add_argument("--skip-weighting", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    return parser.parse_args()


def key(freq):
    return f"f{freq:.2f}".replace(".", "p") + "_A0p20"


def load_train(h5, freq):
    mid = LENGTHS[freq] // 2
    g = h5["data_structure"]["monosines"][key(freq)]
    snaps = np.empty((N_GRID * N_FIELDS, 800))
    for i, field in enumerate(FIELD_NAMES):
        snaps[i * N_GRID:(i + 1) * N_GRID, :] = g[field][:, mid - 400:mid + 400]
    return snaps[:, :-1].copy(), snaps[:, 1:].copy(), g["y"][[0], mid - 400:mid + 400 - 1]


def load_test(h5, freq):
    g = h5["data_structure"]["monosines"][key(freq)]
    n = LENGTHS[freq]
    snaps = np.empty((N_GRID * N_FIELDS, n))
    for i, field in enumerate(FIELD_NAMES):
        snaps[i * N_GRID:(i + 1) * N_GRID, :] = g[field][:]
    return snaps[:, :-1].copy(), snaps[:, 1:].copy(), g["y"][[0], :-1], g["time"][[0], :-1][0]


def relative_error(pred, true):
    return np.linalg.norm(pred - true, axis=0) / np.linalg.norm(true, axis=0)


def build(train_freqs, h5, reTrainThres, updateThres, updateBasis):
    pre, post, y = zip(*[load_train(h5, f) for f in train_freqs])
    model = TrackedFluidSurrog(
        reTrainThres=reTrainThres, maxLen=1000, updateBasis=updateBasis, updateThres=updateThres,
        automatic_weight=True, eps_for_automatic_weight=4., automatic_weight_at_retrain=True,
        updateOmega=False, output_folder=SIDE_FOLDER)
    model.omega0 = 0.01
    model._omega_terms = (1 - model.omega0, model.omega0)
    model.train(list(y), list(pre), list(post), params=np.array([train_freqs]), **TRAIN_KWARGS)
    return model


def run_online(model, test, freq):
    pre, post, y, _ = test
    theta = np.array([[freq]])
    model.initialize_predictions(theta)
    out = np.empty_like(post)
    for i in range(post.shape[1]):
        out[:, i] = model.predict(y[:, [i]], pre[:, [i]], solidReduc=None, params=theta).ravel()
        model.augmentData(y[:, [i]], pre[:, [i]], post[:, [i]], solidReduc=None, params=theta,
                          stepsize=None, computeAngle=False)
    return relative_error(out, post)


def parameter_weights(scheme, train_freqs, freq):
    d = np.abs(np.array(train_freqs) - freq)
    if scheme == "parameter IDW":
        w = d ** -2.0
    elif scheme == "nearest (parameter)":
        w = np.isclose(d, d.min()).astype(float)
    else:
        w = np.ones(len(d))
    return w / w.sum()


SCHEMES = ["Grassmann", "parameter IDW", "nearest (parameter)", "uniform"]


def study_weighting(h5, out_dir):
    print("=" * 70)
    print("Study W: weighting of the local regressions (pre-trained model)")
    print("=" * 70)
    med = np.zeros((len(WEIGHTING_FREQS), len(SCHEMES)))
    mean = np.zeros_like(med)
    for a, f in enumerate(WEIGHTING_FREQS):
        train_freqs = [g for g in FREQS if g != f]
        test = load_test(h5, f)
        model = build(train_freqs, h5, NEVER, NEVER, False)
        model.initialize_predictions(np.array([[f]]))
        grassmann_w = model.reducLoad.weights.copy()
        print(f"  held-out f={f}: training {train_freqs}")
        for b, s in enumerate(SCHEMES):
            model.reducLoad.weights = grassmann_w.copy() if s == "Grassmann" else parameter_weights(s, train_freqs, f)
            w = model.reducLoad.weights
            err = run_online(model, test, f)
            med[a, b], mean[a, b] = np.median(err), err.mean()
            print(f"    {s:20s} weights={np.round(w, 3)}  median error={med[a, b]:.3e}  mean={mean[a, b]:.3e}")
    np.savez(f"{out_dir}/ablation_weighting_viv.npz", med=med, mean=mean, freqs=WEIGHTING_FREQS, schemes=SCHEMES)
    return med, mean


def study_online_data(h5, out_dir):
    print("=" * 70)
    print("Study O: use of online data (test frequency 0.9)")
    print("=" * 70)
    f = 0.9
    train_freqs = [0.5, 0.7, 1.1, 1.3]
    test = load_test(h5, f)
    configs = [
        ("pre-trained only", NEVER, NEVER, False),
        ("regression retraining only", REGRESSION_FREQUENCY, NEVER, False),
        ("regression retraining + basis update (proposed)", REGRESSION_FREQUENCY, BASIS_FREQUENCY, True),
    ]
    errs = {}
    for name, rt, ut, ub in configs:
        t0 = time.time()
        model = build(train_freqs, h5, rt, ut, ub)
        err = run_online(model, test, f)
        errs[name] = err
        print(f"  {name:50s} median={np.median(err):.3e} mean={err.mean():.3e} "
              f"last-quarter median={np.median(err[3 * len(err) // 4:]):.3e}  [{time.time() - t0:.0f} s]")
    np.savez(f"{out_dir}/ablation_online_data_viv.npz", time=test[3], **{f"e{i}": e for i, e in enumerate(errs.values())},
             names=list(errs.keys()))
    return test[3], errs


def make_figure(out_dir, o):
    t, errs = o
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    styles = [("#999999", 0.9), ("#0072B2", 0.9), ("k", 1.1)]
    for (name, e), (col, lw) in zip(errs.items(), styles):
        ax.semilogy(t[:len(e) - 9], np.convolve(e, np.ones(10) / 10, mode="valid"), color=col, lw=lw, label=name)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("relative error")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/figAblationsViv.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    o = None
    with h5py.File(args.dataset_file, "r") as h5:
        if not args.skip_weighting:
            study_weighting(h5, args.output_dir)
        if not args.skip_ablation:
            o = study_online_data(h5, args.output_dir)
    if o is None and os.path.exists(f"{args.output_dir}/ablation_online_data_viv.npz"):
        d = np.load(f"{args.output_dir}/ablation_online_data_viv.npz")
        names = list(d["names"])
        o = (d["time"], {n: d[f"e{i}"] for i, n in enumerate(names)})
    if o is not None:
        make_figure(args.output_dir, o)
    shutil.rmtree(SIDE_FOLDER, ignore_errors=True)
    print("Done.")
