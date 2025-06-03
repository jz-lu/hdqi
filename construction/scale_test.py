"""
`scale_test.py`

Test a Pauli k-sparse rejection sampler's scaling numerically.
Choose a sampler, and then find the count scaling with respect to n, keeping m = c*n 
fixed with a constant choice of c (we recommend 1 < c < 2).
The parameter k is also fixed. We recommend 2 <= k <= 6 (2 to be nontrivial, 6 due to square-expo scaling).
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
from paulis4 import sample_commuting_Paulis


def main():
    parser = argparse.ArgumentParser(
        description="Process parameters"
    )

    parser.add_argument(
        "--c", "-c",
        type=float,
        default=1.5,
        help="Parameter c"
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

    args = parser.parse_args()
    c = args.c
    k = args.k
    num_trials = args.trials
    num_ns = 10
    ns = (1 + np.arange(num_ns))*50
    print(f"Values of n = {ns}")

    data = np.zeros((num_ns, 2, num_trials))

    for i, n in enumerate(ns):
        print(f"Working on n = {n}")
        m = int(c * n)
        iter_data = np.zeros((num_trials, m))
        for trial in range(num_trials):
            _, iter_data[trial] = sample_commuting_Paulis(m, n, k)
            if trial % 10 == 0 and trial > 0:
                print(f"Starting trial {trial}")
        last_data = iter_data[:,-m//3:]
        data[i, 0] = np.mean(last_data, axis=1)
        data[i, 1] = np.std(last_data, axis=1)
        
        INDICES = np.arange(m)
        plt.clf()
        plt.errorbar(INDICES, iter_data.mean(axis=0), yerr=iter_data.std(axis=0), fmt='none', 
                ecolor='gray', alpha=0.5, capsize=3)
        plt.scatter(INDICES, iter_data.mean(axis=0), color='royalblue', alpha=1)
        plt.xlabel(f"Step number")
        plt.ylabel(f"Number of trials until success")
        plt.title(rf"$m = ${m}, $n = ${n}, $k = ${k} ({num_trials} trials)")
        plt.savefig(f"speed_TYPE4_m{m}n{n}k{k}_t{num_trials}.png")
    
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
    plt.savefig(f"ScaleTest_TYPE4_c{c}k{k}_t{num_trials}.png")


if __name__ == "__main__":
    main()
