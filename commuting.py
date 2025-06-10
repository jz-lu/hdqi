"""
`commuting.py`
Code file to sample local commuting Pauli Hamiltonians via rejection sampling.
See `README.md` for usage details.
"""
import numpy as np
import argparse
import matplotlib.pyplot as plt
import time
from helper import rand_k_sparse_vec, matrix_sym_inner_prod, \
                   symp2Pauli, matrix_symp2Pauli, rand_k_local_Pauli, \
                   many_random_indices, rand_k_indices, check_overlap_parities, \
                   diagonalize_commuting_Paulis

def sample_a_Pauli(n, k, sampling_type):
    """
    Sample a Pauli according to type. See `type123_sampler` for details.
    
    Params:
        * n (int): number of qubits.
        * k (int): locality parameter of the Paulis.
        * sampling_type (int): 1, 2, or 3
    
    Returns:
        * length-2n vector which is the symplectic representation of a Pauli sampled from its type
    """
    if sampling_type == 1:
        return rand_k_local_Pauli(n, k)
    elif sampling_type == 2:
        return np.concatenate([rand_k_sparse_vec(n, k), rand_k_sparse_vec(n, k)])
    elif sampling_type == 3:
        return rand_k_sparse_vec(2*n, k)
    else:
        assert False, "Impressive...you should have never made it here."

def type123_sampler(m, n, k, sampling_type):
    """
    Sampling type 1: Random k-local Pauli. 
    Choose k random locations to place a non-identity Pauli, then choose a 
    random Pauli (X, Y, Z) with equal probability for each.
    Locality guarantee: k.

    Sampling type 2: Random k-local X's and independent random k-local Z's. 
    Choose k random places to put X's, and k random places to put Z's. 
    Places that have X and Z instead get a Y. This corresponds to a CSS-type XZ-decoupled error model.
    Locality guarantee: 2 * k.

    Sampling type 3: Random k-sparse symplectic vectors. Choose a uniformly random weight-k binary vector of length 2n. 
    There is not really a natural/physical corresponding model.
    Locality guarantee: k.

    Params:
        * m (int): number of Paulis to sample. Should be at least n and no more than around 2n to be reasonable.
        * n (int): number of qubits.
        * k (int): locality parameter of the Paulis.
        * sampling_type (int): 1, 2, or 3

    Returns:
        * `Paulis`: a m x 2n matrix where the ith row is the ith Pauli in symplectic representation.
        * `num_iters`: a length-m integer vector counting how many rejections it took to get the ith Pauli.
    """
    assert sampling_type in [1, 2, 3], f"Sampling type {sampling_type} invalid!"
    counter = 1 # index of which Pauli we are on

    Paulis = np.zeros((m, 2*n), dtype=int) # The rows of `Paulis` are the Paulis we want
    Paulis[0] = sample_a_Pauli(n, k, sampling_type) # Pick the first one

    # Roughly the magnitude scale of how many rejections we expect
    scale = int(2 ** (((2*k) ** 2) * m / n)) if sampling_type == 2 else int(2 ** ((k ** 2) * m / n))

    # Count the number of iterations it took to find each next commuting Pauli
    num_iters = np.zeros(m)

    while counter < m:
        loop_ctr = 1
        found = False

        while not found:
            # Provide an exit condition in case the amount of iterations we need is much
            # larger than what we expect
            if loop_ctr % scale == 0:
                factor = loop_ctr // scale
                print(f"[Ctr={counter}] WARNING: Loop counter is at {factor} * scale steps")
                if factor > 10:
                    print(f"[n={n}, m={m}, k={k}] [Ctr={counter}] I give up. Here's your cursed Pauli tableau:")
                    print(matrix_symp2Pauli(Paulis[:counter], n))
                    print(f"And your last trial Pauli: {symp2Pauli(Pauli_new, n)}")
                    print(f"Their SIPs are {matrix_sym_inner_prod(Paulis[:counter], Pauli_new, n)}")
                    raise RuntimeError("Took too long to sample. You should not ever see this message. If you do, you are probably choosing way too large of m, or you are ridiculously unlucky.")

            Pauli_new = sample_a_Pauli(n, k, sampling_type)

            sips = matrix_sym_inner_prod(Paulis[:counter], Pauli_new, n) # evaluate symplectic inner products
            if np.all(sips == 0): # if the Pauli commutes with all previous, mark found and add to list
                Paulis[counter] = Pauli_new
                found = True
                num_iters[counter] = loop_ctr
                counter += 1
            else: # else, add 1 to the rejection loop counter and try again
                loop_ctr += 1
        
    return Paulis, num_iters

def type4_sampler(m1, m2, n, k):
    """
    Type 4: X first, then Z. Choose `m1` k-local Pauli X strings randomly. 
    Then choose `m2` k-local Pauli Z strings randomly, subject to them commuting with the X strings. 
    This also lacks a natural physical interpretation.
    Warning: do not use this type with `m1 >> n`. 
    If you do this, then with high probability the X's will form an independent basis of all X-type strings, 
    and there will not exist any Z-type string which commutes with them. 

    Params:
        * m1 (int): number of X Paulis to sample.
        * m2 (int): number of Z Paulis to sample.
        * n (int): number of qubits.
        * k (int): locality parameter of the Paulis.

    Returns:
        * `Paulis`: a m x 2n matrix where the ith row is the ith Pauli in symplectic representation.
        * `num_iters`: a length-m integer vector counting how many rejections it took to get the ith Pauli.
    """
    m = m1 + m2
    X_list = many_random_indices(m1, n, k)
    Z_list = np.empty((m2, k), dtype=int)
    num_iters = np.zeros(m2, dtype=int)
    
    # Roughly the magnitude scale of how many rejections we expect
    scale = int(2 ** ((k ** 2) * m1 / n))

    for i in range(m2):
        found = False
        local_counter = 1
        while not found:
            Z_candidate = np.array(rand_k_indices(n, k))
            good = check_overlap_parities(X_list, Z_candidate)
            if good:
                found = True
                Z_list[i] = Z_candidate
            else:
                local_counter += 1
            
            if local_counter % scale == 0:
                excess_factor = local_counter // scale
                print(f"[Iteration {i}, attempt {local_counter}] WARNING: Excess factor reached {excess_factor}")
                if excess_factor > 1e3:
                    raise TimeoutError("Too slow, better luck next time :(")
                
        num_iters[i] = local_counter

    Paulis = np.zeros((m, 2*n), dtype=int)
    for i in range(m1):
        Paulis[i, X_list[i]] = 1
    for i in range(m2):
        Paulis[i+m1, Z_list[i]+n] = 1
    return Paulis, num_iters

def sample_commuting_Paulis(m, n, k, sampling_type=1, m1=-1, m2=-1):
    """
    Wrapper function calling a sampling function depending on the input `sampling_type` specified, defaulted to Type 1.
    In each case, we sample a list of `m` `n`-qubit Paulis which are local, either k-local or 2k-local 
    depending on the model.

    Params:
        * m (int): number of Paulis to sample. Should be at least n and no more than around 2n to be reasonable.
        * n (int): number of qubits.
        * k (int): locality parameter of the Paulis.
        * sampling_type (1, 2, 3, or 4): Choice of sampling type (see README.md)

    Returns:
        * `Paulis`: a m x 2n matrix where the ith row is the ith Pauli in symplectic representation.
        * `num_iters`: a length-m integer vector counting how many rejections it took to get the ith Pauli.
    """
    if sampling_type == 4:
        assert m1 > 0 and m2 > 0, f"You must specify m1 and m2 to use Type 4"
        return type4_sampler(m1, m2, n, k)
    else:
        return type123_sampler(m, n, k, sampling_type=sampling_type)

def main(args):
    ROOT = args.save
    SAVE_DATA = not args.nosave
    MAKE_PLOT = not args.noplot
    DIAGONALIZE = args.diagonalize
    m = args.m; n = args.n; k = args.k
    m1 = args.m1; m2 = args.m2
    NUM_TRIALS = args.trials
    if m == -1:
        m = int(1.5 * n)  # default choice of m
    sampling_type = args.type
    assert sampling_type != 4 or (m1 > 0 and m2 > 0), "If you use sampling type 4, m1 and m2 must be provided."
    iter_data = None
    IDENTIFIER = ""
    
    if sampling_type == 4:
        print(f"m1 = {m1}, m2 = {m2}, n = {n}, k = {k}")
        m = m1 + m2
        iter_data = np.zeros((NUM_TRIALS, m2))
        IDENTIFIER = f"TYPE{sampling_type}_m1{m1}m2{m2}n{n}k{k}_t{NUM_TRIALS}"
    else:
        print(f"m = {m}, n = {n}, k = {k}")
        iter_data = np.zeros((NUM_TRIALS, m))
        IDENTIFIER = f"TYPE{sampling_type}_m{m}n{n}k{k}_t{NUM_TRIALS}"
        
    data = np.zeros((NUM_TRIALS, m, 2*n), dtype=np.uint8)

    for trial in range(NUM_TRIALS):
        start = time.perf_counter()
        data[trial], iter_data[trial] = sample_commuting_Paulis(m, n, k, sampling_type=sampling_type, m1=m1, m2=m2)
        end = time.perf_counter()
        minutes, seconds = divmod(end - start, 60)
        print(f"[Trial {trial}] Execution took {minutes} min {round(seconds, 4)} sec")

    if SAVE_DATA or DIAGONALIZE:
        # Transpose so data is num_trials x 2n x m as promised
        data = np.transpose(data, (0, 2, 1))

    if SAVE_DATA:
        np.save(f"{ROOT}/Commuting_{IDENTIFIER}.npy", data)

    if DIAGONALIZE:
        start = time.perf_counter()
        diags = [diagonalize_commuting_Paulis(data[i], m, n, inplace=False) for i in range(NUM_TRIALS)]
        end = time.perf_counter()
        minutes, seconds = divmod((end - start) / NUM_TRIALS, 60)
        print(f"Diagonalization took {minutes} min {round(seconds, 4)} sec on average per trial")
        diags = np.stack(diags, axis=0)

        if SAVE_DATA:
            np.save(f"{ROOT}/Diagonalized_{IDENTIFIER}.npy", data)
    
    if MAKE_PLOT:
        INDICES = np.arange(m2) if sampling_type == 4 else np.arange(m)
        plt.errorbar(INDICES, iter_data.mean(axis=0), yerr=iter_data.std(axis=0), fmt='none', 
                ecolor='gray', alpha=0.5, capsize=3)
        plt.scatter(INDICES, iter_data.mean(axis=0), color='royalblue', alpha=1)
        plt.xlabel(f"Step number")
        plt.ylabel(f"Number of trials until success")
        plt.title(rf"$m = ${m}, $n = ${n}, $k = ${k} ({NUM_TRIALS} trials)")
        plt.savefig(f"{ROOT}/IterationsCommuting_{IDENTIFIER}.png")
        np.save(f"{ROOT}/IterationsCommuting_{IDENTIFIER}.npy", iter_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample commuting Paulis"
    )

    parser.add_argument(
        "--m", "-m",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--m1",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--m2",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--n", "-n",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--k", "-k",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of instances you want to generate"
    )
    parser.add_argument(
        "--type",
        type=int,
        default=1,
        help="Sampling type",
        choices=[1, 2, 3, 4]
    )
    parser.add_argument(
        "--noplot",
        action="store_true",
        help="Flag if you don't want instance plots"
    )
    parser.add_argument(
        "--nosave",
        action="store_true",
        help="Flag if you don't want to save your data"
    )
    parser.add_argument(
        "--save", "-s",
        type=str,
        default=".",
        help="Directory in which data will be saved"
    )
    parser.add_argument(
        "--diagonalize", '-d',
        action="store_true",
        help="Flag if you want to also diagonalize the Paulis"
    )

    args = parser.parse_args()
    main(args)

