"""
`scale_test.py`

Test a Pauli k-sparse rejection sampler's scaling numerically.
Choose a sampler, and then find the count scaling with respect to n, keeping m = c*n 
fixed with a constant choice of c (we recommend 1 < c < 2).
The parameter k is also fixed. We recommend 2 <= k <= 6 (2 to be nontrivial, 6 due to square-expo scaling).

This currently only works for Types 1-3. Type 4 is NOT supported.
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
from generate_commuting import sample_commuting_Paulis

def main(args):
    c = args.c
    k = args.k
    num_trials = args.trials
    sampling_type = args.type
    ROOT = args.save
    save_data = not args.nosave
    make_plot = not args.noplot
    n_start = args.nstart
    n_end = args.nend
    num_ns = args.npts
    ns = np.linspace(n_start, n_end, num_ns)
    ns = np.array(ns, dtype=int)
    assert n_end > n_start, f"n_start = {n_start} must be smaller than n_end = {n_end}"
    print(f"Values of n = {ns}")

    data = np.zeros((num_ns, 2, num_trials))

    for i, n in enumerate(ns):
        print(f"Working on n = {n}")
        m = int(c * n)
        iter_data = np.zeros((num_trials, m))
        for trial in range(num_trials):
            _, iter_data[trial] = sample_commuting_Paulis(m, n, k, sampling_type=sampling_type)
            if trial % 10 == 0 and trial > 0:
                print(f"Starting trial {trial}")
        last_data = iter_data[:,-m//3:]
        data[i, 0] = np.mean(last_data, axis=1)
        data[i, 1] = np.std(last_data, axis=1)
        
        if make_plot:
            IDENTIFIER = f"TYPE{sampling_type}_m{m}n{n}k{k}_t{num_trials}"
            INDICES = np.arange(m)
            plt.clf()
            plt.errorbar(INDICES, iter_data.mean(axis=0), yerr=iter_data.std(axis=0), fmt='none', 
                    ecolor='gray', alpha=0.5, capsize=3)
            plt.scatter(INDICES, iter_data.mean(axis=0), color='royalblue', alpha=1)
            plt.xlabel(f"Step number")
            plt.ylabel(f"Number of trials until success")
            plt.title(rf"$m = ${m}, $n = ${n}, $k = ${k} ({num_trials} trials)")
            plt.savefig(f"{ROOT}/IterationsCommuting_{IDENTIFIER}.png")
    
    IDENTIFIER = f"TYPE{sampling_type}_c{c}k{k}_t{num_trials}"
    if save_data:
        np.save(f"{ROOT}/Scaling_{IDENTIFIER}.npz", data)

    plt.clf()
    for i in range(num_ns):
        x_val = ns[i]
        means = data[i, 0, :]
        stds = data[i, 1, :]
        for j in range(num_trials):
            plt.errorbar(x_val, means[j], yerr=stds[j], fmt='o', capsize=3)

    plt.xlabel(r'$n$')
    plt.ylabel('Rejections before success')
    plt.xticks(ns)
    plt.title(rf"$c = ${c}, $k = ${k} ({num_trials} trials)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f"ScaleTest_{IDENTIFIER}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scaling tests as a function of n for commuting Hamiltonians"
    )

    parser.add_argument(
        "--c", "-c",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--k", "-k",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of trials per n"
    )
    parser.add_argument(
        "--nstart",
        type=int,
        default=50,
        help="Starting n (inclusive)"
    )
    parser.add_argument(
        "--nend",
        type=int,
        default=500,
        help="Ending n (inclusive)"
    )
    parser.add_argument(
        "--npts",
        type=int,
        default=10,
        help="Number of n's to evaluate"
    )
    parser.add_argument(
        "--type",
        type=int,
        default=1,
        help="Sampling type",
        choices=[1, 2, 3]
    )
    parser.add_argument(
        "--noplot",
        action="store_true",
        help="Flag if you don't want instance plots"
    )
    parser.add_argument(
        "--nosave",
        action="store_true",
        help="Flag if you don't want to save your scaling data"
    )
    parser.add_argument(
        "--save", "-s",
        type=str,
        default=".",
        help="Directory in which data will be saved"
    )
    args = parser.parse_args()

    main(args)
