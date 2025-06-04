"""
`helper.py`
Helper file containing many functions relevant to sampling Hamiltonians.
These include:
    * Transformations between Pauli strings and symplectic vectors
    * Evaluating symplectic inner products
    * Sampling random sparse vectors
"""
import numpy as np

def symp2Pauli(x, n):
    """
    Return a sign-free Pauli string representation of the length 2`n` symplectic vector `x`.

    Input:
        * n (int): number of qubits.
        * x (numpy.ndarray): binary vector of length 2n, symplectically representing a n-qubit Pauli string.
    
    Returns:
        * length-n string over {I, X, Y, Z} where the ith character is the Pauli on the ith qubit
    """
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
    """
    Turns a list of m = M.shape[0] symplectic vectors into their Pauli strings all at once,
    by calling `symp2Pauli()`.
    
    Input:
        * M (np.ndarray): m x 2n arrayof bits, where each row is a symplectic vector representing a Pauli
        * n (int): number of qubits
    
    Returns:
        * list of m strings of length n over {I, X, Y, Z} which are the Paulis in M
    """
    m = M.shape[0]
    strings = []
    for vec in M:
        strings.append(symp2Pauli(vec, n))
    return strings

def sym_inner_prod(x, y, n):
    """
    Return the symplectic inner product, mod 2, of two 2`n`-vectors `x` and `y`
    
    Input:
        * x (np.ndarray): binary vector of length 2n
        * y (np.adarray): binary vector of length 2n
        * n (int): number of qubits
    
    Returns:
        * a bit indicating the symplectic inner product parity of x and y
    """
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
    """
    Generate a length-`n` bitstring which is random and `k`-sparse

    Input:
        * n (int): length of bitstring/vector
        * k (int <= n): number of ones in the bitstring
    
    Returns:
        * random length-n bitstring represented as a numpy.ndarray with k 1's and n-k 0's
    """
    bits = np.zeros(n, dtype=int)
    indices = np.random.choice(n, size=k, replace=False)
    bits[indices] = 1
    return bits

def rand_k_indices(n, k):
    """
    Generates a list of k indices, sampled without replacement, out of n
    
    Input:
        * n (int): cutoff for sampling, i.e. sampling space is S := {1, ..., n}
        * k (int <= n): number of numbers to sample
    
    Returns:
        * length-k vector of numbers between 1 and n, which are sampled from S without replacement
    """
    return np.random.choice(n, size=k, replace=False)

def rand_k_local_Pauli(n, k):
    """
    Generate a uniformly random Pauli of weight k

    Input:
        * n (int): number of qubits
        * k (int <= n): support of Pauli
    
    Returns:
        * length-2n binary vector symplectically representing a uniformly random weight-k Pauli
    """
    # First choose the support indices, then sample the Paulis on each support index
    idxs = rand_k_indices(n, k)
    pauli_choices = np.random.choice([1, 2, 3], size=k, replace=True) # 1 = X, 2 = Y, 3 = Z

    # Write down the symplectic vector representing the sampled Pauli
    Xs = np.zeros(n, dtype=int)
    Zs = np.zeros(n, dtype=int)

    for i, idx in enumerate(idxs):
        if pauli_choices[i] <= 2: # if Pauli is X or Y put a 1 in the X part of the vector
            Xs[idx] = 1
        if pauli_choices[i] >= 2: # if Pauli is Y or Z put a 1 in the Z part of the vector
            Zs[idx] = 1
    return np.concatenate([Xs, Zs]) # concatenate to build the symplectic vector

def many_random_indices(m, n, k):
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
        matrix[i] = rand_k_indices(n, k)
    return matrix

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