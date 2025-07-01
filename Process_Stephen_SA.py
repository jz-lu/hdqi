"""
`Process_Stephen_SA.py`
Code file to grab a bunch of numpy files and output them as one .npy file and as a CSV.
Useful for gathering distributed compute outputs into an amalgamated results file for SA analysis.
Then do some simple plotting on the file.
"""
import os
import re
import numpy as np
import csv
import matplotlib.pyplot as plt

def load_sastephen(root, out_np="output.npy", out_csv="entries.csv"):
    # 1) compile the filename pattern
    pattern = re.compile(
        r"^SAStephen_TYPE(?P<type>[^_]+)"
        r"_m(?P<m>\d+)n(?P<n>\d+)k(?P<k>\d+)"
        r"_(?P<trial>\d+)\.npy$"
    )

    entries = []
    # 2) scan and parse
    for fname in os.listdir(root):
        match = pattern.match(fname)
        if not match:
            continue
        gd = match.groupdict()
        m = int(gd["m"])
        n = int(gd["n"])
        k = int(gd["k"])
        trial = int(gd["trial"])
        ratio = m // n
        t = gd["type"]
        path = os.path.join(root, fname)
        entries.append((t, n, ratio, k, trial, path))

    # 3) sort lexicographically by the first five fields
    entries.sort(key=lambda e: (e[0], e[1], e[2], e[3], e[4]))

    # 4) extract the unique, sorted values along each axis
    types   = sorted({e[0] for e in entries})
    ns      = sorted({e[1] for e in entries})
    ratios  = sorted({e[2] for e in entries})
    ks      = sorted({e[3] for e in entries})
    trials  = sorted({e[4] for e in entries})

    # 5) build lookup maps to indices
    ti = {t:i for i,t in enumerate(types)}
    ni = {v:i for i,v in enumerate(ns)}
    ri = {v:i for i,v in enumerate(ratios)}
    ki = {v:i for i,v in enumerate(ks)}
    xi = {v:i for i,v in enumerate(trials)}

    # 6) allocate the 5D array and fill it
    shape = (len(types), len(ns), len(ratios), len(ks), len(trials))
    arr = np.empty(shape, dtype=object)  # or change dtype as needed

    for t, n, ratio, k, trial, path in entries:
        data = np.load(path)
        arr[ti[t], ni[n], ri[ratio], ki[k], xi[trial]] = data

    # 7) save the numpy array
    np.save(out_np, arr)
    print(f"Saved 5D array to {out_np}, shape = {arr.shape}")

    # 8) write CSV of the parameter tuples
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "n", "ratio", "k", "trial"])
        for t, n, ratio, k, trial, _ in entries:
            writer.writerow([t, n, ratio, k, trial])
    print(f"Wrote parameter tuples to {out_csv}")

    return arr, {
        "types": types,
        "ns": ns,
        "ratios": ratios,
        "ks": ks,
        "trials": trials
    }


def extract_ns_ratios_ks(data):
    """
    data: tuple (arr, meta), where meta is the dict returned by load_sastephen
    Returns:
      ns:      sorted list of n-values
      ratios:  sorted list of m//n ratios
      ks:      sorted list of k-values
    """
    _, meta = data
    return meta["ns"], meta["ratios"], meta["ks"]


def plot_ratio(arr, types, ns, ratios, ks, r, out_png="ratio_plot.png"):
    """
    arr.shape == (len(types), len(ns), len(ratios), len(ks), num_trials)
    types:   list of the two type identifiers (e.g. [1,2])
    ns:      sorted list of n values
    ratios:  sorted list of m//n values
    ks:      sorted list of k values
    r:       the ratio to filter on (must be in ratios)
    out_png: filename to save the figure to
    """
    # 1) find the index of the requested ratio
    try:
        ridx = ratios.index(r)
    except ValueError:
        raise ValueError(f"ratio {r} not found in ratios list")
    
    # 2) prepare markers & linestyles per type
    marker_map = {types[0]: "o",   # circle for type 1
                  types[1]: "^"}   # triangle for type 2
    linestyle_map = {types[0]: "-",   # solid for type 1
                     types[1]: ":"}   # dotted for type 2

    # 3) get a color cycle
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots()
    ax.set_xlabel("n")
    ax.set_ylabel("f(n)")
    ax.set_ylim(0.4, 0.8)

    # 4) for each k, pick a color, then for each type plot
    for ki, k in enumerate(ks):
        color = color_cycle[ki % len(color_cycle)]

        for ti, t in enumerate(types):
            # slice out shape (len(ns), num_trials)
            data_trials = arr[ti, :, ridx, ki, :]
            # compute mean & std over the last axis (trials)
            means = np.mean(data_trials.astype(float), axis=1)
            stds  = np.std (data_trials.astype(float), axis=1)

            # plot with errorbars
            ax.errorbar(
                ns, means, yerr=stds,
                label=f"type={t}, k={k}",
                marker=marker_map[t],
                linestyle=linestyle_map[t],
                color=color,
                capsize=3
            )

    ax.legend(title="(type, k)", bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    print(f"Saved plot to {out_png}")




# Example usage:
# load_sastephen("ROOT")

