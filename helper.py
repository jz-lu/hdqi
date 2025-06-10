"""
`helper.py`
Helper file containing many functions relevant to sampling Hamiltonians.
These include:
    * Transformations between Pauli strings and symplectic vectors
    * Evaluating symplectic inner products
    * Sampling random sparse vectors
"""
import numpy as np

"""
SECTION I: Manipulating symplectic vectors and matrices.
"""

def standard_Pauli_tableau(n, include_y=False):
    """
    Generate a symplectic matrix representing the standard Paulis.
    There will always be 2n rows, and the number of columns depends on
    `include_y`. If True, then there are 3n columns. If False, there
    are 2n columns.

    Input:
        * n (int): number of qubits.
        * include_y (bool): indicator on whether to include Y-type Paulis.

    Returns:
        * Symplectic matrix (np.ndarray) where each columns represents a standard
          Pauli. If `include_y`, one column for each X_i, Z_i, Y_i. Otherwise, 
          one column for each X_i and Z_i.
    """
    tableau = np.eye(2*n, dtype=np.uint8) # X's and Z's only
    if include_y:
        eye = np.eye(n, dtype=np.uint8)
        ys = np.vstack([eye, eye])
        return np.hstack([tableau, ys])
    else:
        return tableau

def pauli_to_sym(pauli_str):
    """
    Convert a string of Pauli operators to binary vector (x|z)
    'I' -> (0, 0), 'X' -> (1, 0), 'Z' -> (0, 1), 'Y' -> (1, 1)
    """
    x = []
    z = []
    for p in pauli_str:
        if p == 'I':
            x.append(0)
            z.append(0)
        elif p == 'X':
            x.append(1)
            z.append(0)
        elif p == 'Z':
            x.append(0)
            z.append(1)
        elif p == 'Y':
            x.append(1)
            z.append(1)
        else:
            raise ValueError(f"Invalid Pauli operator: {p}")
    return np.array(x + z, dtype=np.uint8)  # Concatenate X | Z parts

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

def apply_Clifford_gate(matrix, op, n, i, j=None, right=False):
    """
    Apply the given Clifford operation to the matrix `matrix`.
    Return the transformed Pauli matrix.

    Input:
        * matrix (np.ndarray): can either be a 2n x m matrix of Paulis in symplectic form, 
          or a 2n x 2n symplectic matrix representing a Clifford.
        * op (str): description of the operator. Choices are ['H', 'S', 'CX', 'CZ', 'SWAP'].
        * n (int): number of qubits.
        * i (int): index of first qubit operator acts on (i < n).
        * j (int, optional): if `op` is a 2-qubit operator, specify the second qubit (j < n).
        * right (bool): flag on whether to act on the right or the left. If acting on the right, 
          the input matrix must be a Clifford. The operation is otherwise ill-defined.
    
    Returns:
        * 2n x m np.ndarray representing the transformed version of `Paulis`
    """
    if right:
        assert matrix.shape[1] == 2*n, f"You cannot act on the right unless the input matrix is a 2n x 2n Clifford"
        if op == 'H':
            # Swap columns i and i+n
            matrix[:, [i, i+n]] = matrix[:, [i+n, i]]
        elif op == 'S':
            # Adds column i+n to column i
            matrix[:, i] = (matrix[:, i] + matrix[:, i+n]) % 2
        elif op == 'CX':
            # Adds column j to column i, and column i+n to column j+n
            matrix[:, i] = (matrix[:, i] + matrix[:, j]) % 2
            matrix[:, j+n] = (matrix[:, j+n] + matrix[:, i+n]) % 2
        elif op == 'CZ':
            # Adds column j+n to column i and add column i+n to column j
            matrix[:, i] = (matrix[:, i] + matrix[:, j+n]) % 2
            matrix[:, j] = (matrix[:, j] + matrix[:, i+n]) % 2
        elif op == 'SWAP':
            # Swap columns i and j, and swap columns i+n and j+n
            matrix[:, [i, j]] = matrix[:, [j, i]]
            matrix[:, [i+n, j+n]] = matrix[:, [j+n, i+n]]
        else:
            raise ValueError(f"Specified operator {op} is not a valid Clifford. Choices = ['H', 'S', 'CX', 'CZ', 'SWAP'].")
    else:
        if op == 'H':
            # Swap rows i and i+n
            matrix[[i, i+n]] = matrix[[i+n, i]]
        elif op == 'S':
            # Adds row i to row i+n
            matrix[i+n] = (matrix[i] + matrix[i+n]) % 2
        elif op == 'CX':
            # Adds row i to row j, and adds row j+n to row i+n
            matrix[j] = (matrix[i] + matrix[j]) % 2
            matrix[i+n] = (matrix[j+n] + matrix[i+n]) % 2
        elif op == 'CZ':
            # Adds row i to row j+n and add row j to row i+n
            matrix[j+n] = (matrix[i] + matrix[j+n]) % 2
            matrix[i+n] = (matrix[j] + matrix[i+n]) % 2
        elif op == 'SWAP':
            # Swap rows i and j, and swap rows i+n and j+n
            matrix[[i, j]] = matrix[[j, i]]
            matrix[[i+n, j+n]] = matrix[[j+n, i+n]]
        else:
            raise ValueError(f"Specified operator {op} is not a valid Clifford. Choices = ['H', 'S', 'CX', 'CZ', 'SWAP'].")
    
    return matrix

def check_symplectic_consistency(Paulis, m, n):
    """
    Verify that all columns of a 2n x m matrix `Paulis` are all symplectically orthogonal.

    Input:
        * Paulis (np.ndarray): 2n x m binary matrix, each column is the symplectic representation of a n-qubit Pauli.
          This function expects that every pair of columns is symplectically orthogonal.
        * m (int): number of Paulis.
        * n (int): number of qubits.
    
    Returns:
        * boolean indicating whether all pairs of columns are symplectically orthogonal.
    """
    return np.all([sym_inner_prod(Paulis[:,i], Paulis[:,j], n) == 0 for i in range(m) for j in range(m)])

def find_diagonalizing_Clifford(Paulis, m, n):
    """
    Find a symplectic matrix (a Clifford) which diagonalizes a list of commuting Paulis.
    The diagonalization is performed by Gottesman's algorithm (see "Suriving as a quantum
    computer in a classical world" by D. Gottesman, Chapter 6.)
    The actual output is not a symplectic matrix or unitary matrix describing the Clifford, 
    but rather a list of Clifford gates.
    
    Input:
        * Paulis (numpy.ndarray): 2n x m binary matrices representing commuting Paulis in symplectic
          form. Assumes that all Paulis are commuting (i.e. columns are orthogonal in SIP).
        * m (int): number of Paulis.
        * n (int): number of qubits.

    Returns: (clifford, diag_rep)
        * clifford (list): list of tuples (gate, qubit(s)) that describe the Clifford, e.g.
          [('H', 12), ('CX', 1, 3)]. This describes a Clifford circuit acting from left to
          right, i.e. you first apply H, then CX.
    """
    circuit = [] # list of operators applied in the Clifford circuit
    
    # assert check_symplectic_consistency(Paulis, m, n)

    itr = 0 # number of independent columns we've looked at
    i = 0 # Column index
    while i < m:
        # If the entire column is zero, then we have a dependent Pauli.
        # Move to the next column.
        if np.all(Paulis[:,i] == 0):
            print(f"Skipped {i}")
            i += 1
            continue
    
        # Now, somewhere in the column there is a 1. Using left multiplication by
        # H and SWAP, we can put this 1 in the index of `itr`.
        idx = np.where(Paulis[:,i] == 1)[0][0]
        if idx != itr:
            if idx >= n:
                # Apply H on qubit (idx-n)
                idx -= n
                circuit.append(('H', idx))
                Paulis[[idx, idx+n]] = Paulis[[idx+n, idx]]
                # assert check_symplectic_consistency(Paulis, m, n)
            # Apply SWAP on index and 0
            circuit.append(('SWAP', itr, idx))
            Paulis[[idx, itr]] = Paulis[[itr, idx]]
            Paulis[[idx+n, itr+n]] = Paulis[[itr+n, idx+n]]
            # assert check_symplectic_consistency(Paulis, m, n)

        # Do column reduction on the current column of the top half: Use left multiplication by CX to add the 1 in the
        # upper left corner to any other row of in the top half with a 1 in the current column.
        # This zeros out all but the current element in top half of the current column.
        # Do the same for the bottom half, using S gates and CZ gates.
        ones_cols_idxs = np.where(Paulis[:,i] == 1)[0][1:]
        for j in ones_cols_idxs:
            if j < n:
                # Top half: use CX gates
                circuit.append(('CX', itr, j))
                Paulis[j] = (Paulis[itr] + Paulis[j]) % 2
                Paulis[itr+n] = (Paulis[j+n] + Paulis[itr+n]) % 2
                # assert check_symplectic_consistency(Paulis, m, n)
            else:
                # Bottom half: use S for index n and CZ for everything else
                if j == n:
                    circuit.append(('S', itr))
                    Paulis[itr+n] = (Paulis[itr] + Paulis[itr+n]) % 2
                    # assert check_symplectic_consistency(Paulis, m, n)
                else:
                    idx = j - n
                    circuit.append(('CZ', itr, idx))
                    Paulis[idx+n] = (Paulis[itr] + Paulis[idx+n]) % 2
                    Paulis[itr+n] = (Paulis[idx] + Paulis[itr+n]) % 2
                    # assert check_symplectic_consistency(Paulis, m, n)

        # Do row reduction by multiplying other stabilizers by the first 
        # stabilizer if the stabilizer has a 1 in its first entry. This zeros out
        # the first row except for the first entry.
        ones_rows_idxs = np.where(Paulis[itr] == 1)[0][1:]
        for j in ones_rows_idxs:
            Paulis[:,j] = (Paulis[:,i] + Paulis[:,j]) % 2
            # assert check_symplectic_consistency(Paulis, m, n)

        # At this point, we should find that the first row of the bottom half is also zero,
        # due to the symplectic orthogonality of columns.
        # assert np.all(Paulis[i+n] == 0), f"{itr}th row of bottom half should be 0, but is\n{Paulis[itr+n]}"
        itr += 1
        i += 1
    
    assert np.all(Paulis[n:] == 0), f"Bottom half of matrix should be 0, but is actually\n{Paulis[n:]}"

    return circuit

def apply_Clifford_circuit(clifford, Paulis, n):
    """
    Conjugate a Clifford circuit onto a list of Paulis in symplectic reprsentation.

    Input: 
        * clifford (list): list of tuples (gate, qubit(s)) that describe the Clifford, e.g.
          [('H', 12), ('CX', 1, 3)]. This describes a Clifford circuit acting from left to
          right, i.e. you first apply H, then CX.
        * Paulis (np.ndarray): 2n x m binary matrices representing m n-qubit Paulis 
          in symplectic form.
        * n (int): number of qubits.
    
    Returns:
        * 2n x m binary matrix representing `Paulis` after transformation under conjugation
          by the circuit given by `clifford`.
    """
    for gate in clifford:
        gatelen = len(gate) # should be 2 or 3
        if gatelen == 2:
            op, qubit = gate
            Paulis = apply_Clifford_gate(Paulis, op, n, qubit)
        elif gatelen == 3:
            op, q1, q2 = gate
            Paulis = apply_Clifford_gate(Paulis, op, n, q1, q2)
        else:
            raise SyntaxError(f"Gate should either be of form (op, i) or (op, i, j), but was {gate}")
    return Paulis

"""
SECTION II: Sampling random vectors and sets.
"""

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


"""
SECTION III: Pertinent test cases.
"""

def toric_code_symplectic_matrix(L):
    """
    Make the stabilizer tableau (including 2 redundancies) in symplectic form of the LxL toric code.
    Useful for testing functions on commuting Paulis.

    Input:
        * L (int): side length of torus.

    Returns:
        * np.ndarray of shape (2*n, n), where n = 2L^2, where each column is a stabilizer of the code.
    """
    n = 2 * L * L  # number of qubits (edges)
    num_stabilizers = n  # total stabilizers: L^2 star + L^2 plaquette

    # Qubit indexing:
    # - Each vertex has a horizontal and vertical edge starting from it.
    # - Horizontal edges (X-direction): index 0 to L^2 - 1
    # - Vertical edges (Y-direction): index L^2 to 2L^2 - 1
    def hor_edge(i, j):
        return (i % L) * L + (j % L)

    def ver_edge(i, j):
        return L * L + (i % L) * L + (j % L)

    stabilizers = []

    # Star operators (X-type): act on edges around a vertex
    for i in range(L):
        for j in range(L):
            indices = [
                hor_edge(i, j),
                ver_edge(i, j),
                hor_edge(i, j - 1),
                ver_edge(i - 1, j)
            ]
            x = np.zeros(n, dtype=int)
            z = np.zeros(n, dtype=int)
            for idx in indices:
                x[idx] = 1
            stabilizers.append((x, z))

    # Plaquette operators (Z-type): act on edges around a plaquette
    for i in range(L):
        for j in range(L):
            indices = [
                hor_edge(i, j),
                ver_edge(i, j + 1),
                hor_edge(i + 1, j),
                ver_edge(i, j)
            ]
            x = np.zeros(n, dtype=int)
            z = np.zeros(n, dtype=int)
            for idx in indices:
                z[idx] = 1
            stabilizers.append((x, z))

    # Build the 2n x n matrix: each stabilizer is a column
    symplectic_matrix = np.zeros((2 * n, num_stabilizers), dtype=int)

    for col, (x, z) in enumerate(stabilizers):
        symplectic_matrix[:n, col] = x
        symplectic_matrix[n:, col] = z

    return symplectic_matrix

def five_qubit_stabilizer_tableau():
    # Stabilizer generators (standard form for the 5-qubit code)
    stabilizers = [
        "X Z Z X I",
        "I X Z Z X",
        "X I X Z Z",
        "Z X I X Z"
    ]

    # Remove spaces for consistency
    stabilizers = [s.replace(" ", "") for s in stabilizers]

    # Convert to binary tableau
    tableau = np.array([pauli_to_sym(p) for p in stabilizers]).T  # shape (2n, m)
    return tableau

if __name__ == "__main__":
    """
    Use this space to build and run any pertinent test cases
    """


    exit(0)