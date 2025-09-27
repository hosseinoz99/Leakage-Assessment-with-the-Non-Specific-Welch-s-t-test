# -*- coding: utf-8 -*-
"""Assignment 10: Non-Specific Welch's t-test up to 3rd order (univariate per time sample)
Generates:
- plot_points_t_order_1/2/3.png
- plot_nrtraces_t_order_1/2/3.png
"""
import os, json, numpy as np, matplotlib.pyplot as plt
from .reader import iter_blocks
from .t_test import update_group_sums, welch_t_from_sums

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def run(traces_path, out_dir, plots_dir, points_per_trace=1000, block_size=2000, schedule=None):
    ensure_dir(out_dir); ensure_dir(plots_dir)
    orders = (1,2,3)
    # Initialize running sums
    P = points_per_trace
    sums = {
        'n0': 0, 'n1': 0,
        'sum0': {m: np.zeros((P,), dtype=np.int64) for m in orders},
        'sum1': {m: np.zeros((P,), dtype=np.int64) for m in orders},
        'sum20': {m: np.zeros((P,), dtype=np.int64) for m in orders},
        'sum21': {m: np.zeros((P,), dtype=np.int64) for m in orders},
    }
    # For Deliverable 2: max |t| vs #traces for each order
    checkpoints = list(schedule or [])
    maxabs_t_by_order = {m: [] for m in orders}
    n_seen_list = []

    n_seen = 0
    next_idx = 0
    for traces, ids in iter_blocks(traces_path, points_per_trace=P, block_size=block_size):
        update_group_sums(traces, ids, sums, orders=orders)
        n_seen += traces.shape[0]
        # If we just crossed a checkpoint, compute t and record max |t|
        while next_idx < len(checkpoints) and n_seen >= checkpoints[next_idx]:
            n_seen_list.append(int(n_seen))
            for m in orders:
                tvals = welch_t_from_sums(sums, order=m)
                maxabs_t_by_order[m].append(float(np.max(np.abs(tvals))))
            next_idx += 1

    # Final t-values for Deliverable 1
    for m in orders:
        tvals = welch_t_from_sums(sums, order=m)
        plt.figure()
        xs = np.arange(P)
        plt.plot(xs, tvals)
        plt.xlabel("Time sample index (0..{})".format(P-1))
        plt.ylabel("Welch t-value")
        plt.title("t-values over time (order {})".format(m))
        plt.savefig(os.path.join(plots_dir, "plot_points_t_order_{}.png".format(m)), dpi=200, bbox_inches="tight")
        plt.close()

    # Deliverable 2: max |t| vs #traces
    if n_seen_list:
        for m in orders:
            plt.figure()
            plt.plot(np.array(n_seen_list, dtype=np.int64), np.array(maxabs_t_by_order[m], dtype=np.float64))
            plt.xlabel("#traces considered")
            plt.ylabel("Absolute max t-value")
            plt.title("Abs max t vs #traces (order {})".format(m))
            plt.savefig(os.path.join(plots_dir, "plot_nrtraces_t_order_{}.png".format(m)), dpi=200, bbox_inches="tight")
            plt.close()

    # Write a short description
    with open(os.path.join(out_dir, "description.txt"), "w") as f:
        f.write("Assignment 10 — Non-specific Welch t-test up to 3rd order.\n")
        f.write("We compare fixed (id=0) vs random (id=1) groups for each point.\n")
        f.write("Order m uses y = x**m (raw moments).\n")
        if n_seen_list:
            f.write("Checkpoints: {}\n".format(n_seen_list))
        f.write("Outputs: 3× plot_points_t_order_X.png, 3× plot_nrtraces_t_order_X.png.\n")

    return plots_dir

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Assignment 10 — Non-specific Welch t-test (orders 1..3)")
    parser.add_argument("--traces_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--plots_dir", type=str, default="outputs/figs")
    parser.add_argument("--points_per_trace", type=int, default=1000)
    parser.add_argument("--block_size", type=int, default=2000)
    parser.add_argument("--schedule", type=int, nargs="*", default=[1000,2000,5000,10000,20000,50000,100000])
    args = parser.parse_args()
    run(args.traces_path, args.out_dir, args.plots_dir, args.points_per_trace, args.block_size, args.schedule)
