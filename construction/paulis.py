"""
Sampling k-sparse symplectic matrices by rejection sampling

NOTE: all symplectic representations are in the XZ convention.
"""
import numpy as np
import argparse
import matplotlib.pyplot as plt
import time

def symp2Pauli(x, n):
    """Return a sign-free Pauli string representation of the length 2`n` symplectic vector `x`"""
    vec = []
    for i in range(n):
        char = 'I'
        if x[i] == 0 and x[i+n] == 1:
            char = 'Z'
        elif x[i] == 1 and x[i+n] == 1:
            char = 'Y'
        elif x[i] == 1 and x[i+n] == 0:
            char = 'X'
        vec.append(char)
    return ''.join(vec)

def matrix_symp2Pauli(M, n):
    """Turn a symplectic matrix into a dense Pauli string representation"""
    m = M.shape[0]
    strings = []
    for vec in M:
        strings.append(symp2Pauli(vec, n))
    return strings

def sym_inner_prod(x, y, n):
    """Return the symplectic inner product, mod 2, of two 2`n`-vectors `x` and `y`"""
    return ((x[:n] @ y[n:]) % 2) ^ ((y[:n] @ x[n:]) % 2)

def matrix_sym_inner_prod(M, x, n):
    """
    Return the symplectic inner product between every row of M
    and x, as a column vector.

    Input:
        * M: m x 2n binary matrix
        * x: 2n binary vector
    Return:
        * y: m binary vector where y[i] = sym_inner_prod(M[i], x, n)
    """

    # Define `z` which is just x but with the symplectic twirl
    z = np.zeros(2*n)
    z[:n] = x[n:]
    z[n:] = x[:n]

    return (M @ z) % 2

def rand_k_sparse_vec(n, k):
    """Generate a length-`n` bitstring which is random and `k`-sparse"""
    bits = np.zeros(n, dtype=int)
    indices = np.random.choice(n, size=k, replace=False)
    bits[indices] = 1
    return bits

def sample_commuting_Paulis(m, n, k):
    """
    Sample a list of `m` `n`-qubit Paulis which all commute and are `k`-sparse, via rejection sampling.
    At each step, choose a random k-sparse X vector, and a random k-sparse Z-vector.
    Combine to a form a random 2k-sparse XZ-symplectic vector.
    If this new vector commutes with all previous ones, then add it to the list. 
    Otherwise, reject and try again.

    Returns:
        * `Paulis`: a m x 2n matrix where the ith row is the ith Pauli in symplectic representation.
        * `num_iters`: a length-m integer vector counting how many trials it took to get the ith Pauli.
    """
    counter = 1

    Paulis = np.zeros((m, 2*n), dtype=int) # The rows of `Paulis` are the Paulis we want
    Paulis[0] = np.concatenate([rand_k_sparse_vec(n, k), rand_k_sparse_vec(n, k)])
    scale = int(2 ** ((k ** 2) * m / n))

    # Count the number of iterations it took to find each next commuting Pauli
    num_iters = np.zeros(m)
    num_iters[0] = 1

    while counter < m:
        loop_ctr = 1
        found = False

        while not found:
            if loop_ctr % scale == 0:
                factor = loop_ctr // scale
                print(f"[Ctr={counter}] WARNING: Loop counter is at {factor} * scale steps")
                if factor > 10:
                    print(f"[n={n}, m={m}, k={k}] [Ctr={counter}] I give up. Here's your cursed Pauli:")
                    print(matrix_symp2Pauli(Paulis[:counter], n))
                    print(f"And your trial Pauli: {symp2Pauli(Pauli_new, n)}")
                    print(f"Their SIP is {matrix_sym_inner_prod(Paulis[:counter], Pauli_new, n)}")
                    exit()

            x_new = rand_k_sparse_vec(n, k)
            z_new = rand_k_sparse_vec(n, k)
            Pauli_new = np.concatenate([x_new, z_new])
            sips = matrix_sym_inner_prod(Paulis[:counter], Pauli_new, n)
            if np.all(sips == 0):
                Paulis[counter] = Pauli_new
                found = True
                num_iters[counter] = loop_ctr
                counter += 1
            else:
                loop_ctr += 1
        
    
    return Paulis, num_iters

def main():
    parser = argparse.ArgumentParser(
        description="Process parameters m, n, and k."
    )

    parser.add_argument(
        "--m", "-m",
        type=int,
        default=-1,
        help="Parameter m"
    )
    parser.add_argument(
        "--n", "-n",
        type=int,
        default=100,
        help="Parameter n (default: 100)"
    )
    parser.add_argument(
        "--k", "-k",
        type=int,
        default=5,
        help="Parameter k (default: 5)"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Number of trials (default 100)"
    )

    args = parser.parse_args()
    m = args.m; n = args.n; k = args.k
    num_trials = args.trials
    if m == -1:
        m = 15 * n
    
    print(f"m = {args.m}, n = {args.n}, k = {args.k}")
    iter_data = np.zeros((num_trials, m))

    for trial in range(num_trials):
        start = time.perf_counter()
        _, iter_data[trial] = sample_commuting_Paulis(m, n, k)
        end = time.perf_counter()
        minutes, seconds = divmod(end - start, 60)
        print(f"[Trial {trial}] Execution took {minutes} min {round(seconds, 5)} sec")
    
    INDICES = np.arange(m)
    plt.errorbar(INDICES, iter_data.mean(axis=0), yerr=iter_data.std(axis=0), fmt='none', 
             ecolor='gray', alpha=0.5, capsize=3)
    plt.scatter(INDICES, iter_data.mean(axis=0), color='royalblue', alpha=1)
    plt.xlabel(f"Step number")
    plt.ylabel(f"Number of trials until success")
    plt.title(rf"$m = ${m}, $n = ${n}, $k = ${k} ({num_trials} trials)")
    plt.savefig(f"speed_m{m}n{n}k{k}_t{num_trials}.png")

if __name__ == "__main__":
    main()

