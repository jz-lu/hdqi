import numpy as np
import random
import argparse
import time
import matplotlib.pyplot as plt

def generate_X_idxs(m, n, k):
    """
    Generate an m x k matrix where each row consists of k unique numbers
    sampled without replacement from the range 1 to n (inclusive).
    
    Parameters:
    - m (int): Number of rows.
    - n (int): Maximum number in the sampling range (1 to n).
    - k (int): Number of samples per row.
    
    Returns:
    - np.ndarray: An m x k array of sampled numbers.
    """
    # Generate the matrix
    matrix = np.empty((m, k), dtype=int)
    for i in range(m):
        matrix[i] = random.sample(range(n), k)
    return matrix

def generate_Z_idx(n, k):
    """
    Generate a single set of indices for rejection sampling.
    
    Parameters:
    - n (int): Maximum number in the sampling range (1 to n).
    - k (int): Number of samples per row.
    
    Returns:
    - np.ndarray: An length-k array of sampled numbers.
    """
    return random.sample(range(n), k)

def list_overlap_parities(X_list, Z_candidate):
    """
    Calculate the `Z_candidate` overlap parities with the `X_list` input.

    Parameters:
    - X_list (numpy matrix): m x k matrix of indices for X placements.
    - Z_candidate (numpy vector): length-k vector of indices for Z placements.
    
    Returns:
    - np.ndarray vector of length m indicating parities of each.
    """
    # Convert to NumPy array for consistent indexing
    M_arr = np.array(X_list)
    v_arr = np.array(Z_candidate)
    
    # Convert v to a set for fast membership testing
    v_set = set(v_arr.tolist())
    
    # Initialize result
    m = M_arr.shape[0]
    result = np.empty(m, dtype=int)
    
    # Compute parity for each row
    for i in range(m):
        # Count shared elements
        shared_count = sum(1 for x in M_arr[i] if x in v_set)
        # Store parity (0 if even, 1 if odd)
        result[i] = shared_count % 2
    
    return result

def check_overlap_parities(X_list, Z_candidate):
    """
    Boolean function indicating if all the parities between `Z_candidate` and 
    `X_list` are zero.
    """
    return np.all(list_overlap_parities(X_list, Z_candidate) == 0)

def find_cutoff(m2, n, k):
    """
    Determine roughly how many iterations we expect to need before we find a valid Z.
    """
    c = m2 / n
    return int(np.exp(c * k**2))

def generate_sample(m1, m2, n, k, count=False, trial=0):
    """
    Sampling procedure

    1) Pick X's completely independently.
    2) Pick Z's by rejection sampling.

    Parameters:
    - m1 (int): number of X-type Paulis
    - m2 (int): number of Z-type Paulis
    - n (int): number of qubits
    - k (int): weight of checks
    - count (bool): indicator flag for whether to count the number of iterations made
    """
    X_list = generate_X_idxs(m1, n, k)
    print(f"X list = \n{X_list}")
    Z_list = np.empty((m2, k), dtype=int)
    counter = np.zeros(m2, dtype=int)
    cutoff = find_cutoff(m2, n, k)

    for i in range(m2):
        found = False
        local_counter = 1
        while not found:
            Z_candidate = np.array(generate_Z_idx(n, k))
            good = check_overlap_parities(X_list, Z_candidate)
            if good:
                found = True
                Z_list[i] = Z_candidate
                # print(np.array(Z_candidate > n/2, dtype=int))
            elif count:
                local_counter += 1
            
            if local_counter % cutoff == 0:
                excess_factor = local_counter // cutoff
                print(f"[Trial {trial}, iteration {i}, attempt {local_counter}] WARNING: Excess factor reached {excess_factor}")
                if excess_factor > 10:
                    raise TimeoutError("Too slow, better luck next time :(")
        if count:
            counter[i] = local_counter

    if count:
        return X_list, Z_list, counter
    else:
        return X_list, Z_list

def plot_trials(data):
    """
    Plot the number of iterations it took to get each Z, for each trial.
    """
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    num_trials = data.shape[0]
    INDICES = np.arange(data.shape[-1])
    plt.errorbar(INDICES, mean, yerr=std, fmt='none', 
             ecolor='royalblue', alpha=0.5, capsize=3)
    plt.scatter(INDICES, mean, color='royalblue', alpha=1)
    plt.xlabel(f"Index in Z-list")
    plt.ylabel(f"Number of trials until success")
    plt.title(rf"$m_1 = ${m1}, $m_2 = ${m2}, $n = ${n}, $k = ${k} ({num_trials} trials)")
    plt.savefig("speed2.png")

def main(m1, m2, n, k, num_trials=1):
    """
    Main execution
    """
    iterations = np.zeros((num_trials, m2), dtype=int)

    for trial in range(num_trials):
        start = time.perf_counter()
        X_list, Z_list, counter = generate_sample(m1, m2, n, k, count=True, trial=trial)
        end = time.perf_counter()
        minutes, seconds = divmod(end - start, 60)
        print(f"[Trial {trial}] Execution took {minutes} min {seconds} sec")
        iterations[trial] = counter
    
    plot_trials(iterations)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process parameters m1, m2, n, and k."
    )

    parser.add_argument(
        "--m1",
        type=int,
        default=-1,
        help="Parameter m1"
    )
    parser.add_argument(
        "--m2",
        type=int,
        default=1,
        help="Parameter m2"
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
        default=3,
        help="Parameter k (default: 5)"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Number of trials (default 100)"
    )

    args = parser.parse_args()
    CONST = 2
    m1 = args.m1; m2 = args.m2; n = args.n; k = args.k
    if m1 == -1:
        m1 = CONST * n
    if m2 == -1:
        m2 = CONST * n
    m = m1 + m2
    num_trials = args.trials
    main(m1, m2, n, k, num_trials=num_trials)

