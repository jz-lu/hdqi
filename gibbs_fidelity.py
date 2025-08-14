import numpy as np
from scipy.linalg import eigh, sqrtm
from functools import reduce
import matplotlib.pyplot as plt

def random_Z_hamiltonian(n, m, k, seed=None):
    """
    Generate an n-qubit Hamiltonian H = sum_{i=1}^m s_i * P_i,
    where each P_i is a k-local Z-type Pauli string and
    each s_i is a random sign +/-1.

    Args:
        n     (int): number of qubits
        m     (int): number of terms to sum
        k     (int): locality (number of Z’s in each term)
        seed  (int, optional): RNG seed for reproducibility

    Returns:
        H (ndarray): 2^n x 2^n complex Hamiltonian matrix
    """
    if seed is not None:
        np.random.seed(seed)

    # Single-qubit operators
    Z = np.array([[1, 0],
                  [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)

    H = np.zeros((2**n, 2**n), dtype=complex)

    # Support matrix
    supp = np.zeros((m, n), dtype=np.int8)

    for i in range(m):
        # 1) pick which k qubits get a Z
        active_qubits = np.random.choice(n, size=k, replace=False)
        supp[i,active_qubits] = 1

        # 2) build the tensor-product string
        ops = [Z if i in active_qubits else I for i in range(n)]
        term = reduce(np.kron, ops)

        # 3) random sign ±1
        sign = 1 if np.random.rand() < 0.5 else -1

        H += sign * term

    return H, supp

def random_X_perturbation(base_matrix, n, k, seed=None):
    """
    Generate an n-qubit Hamiltonian H = sum_{i=1}^m s_i * P_i,
    where each P_i is a k-local Z-type Pauli string and
    each s_i is a random sign +/-1.

    Args:
        n     (int): number of qubits
        m     (int): number of terms to sum
        k     (int): locality (number of Z’s in each term)
        seed  (int, optional): RNG seed for reproducibility

    Returns:
        H (ndarray): 2^n x 2^n complex Hamiltonian matrix
    """
    if seed is not None:
        np.random.seed(seed)

    # Single-qubit operators
    X = np.array([[0, 1],
                  [1, 0]], dtype=complex)
    I = np.eye(2, dtype=complex)

    H = np.zeros((2**n, 2**n), dtype=complex)

    # Support matrix
    allowed_idxs = set(np.arange(n))

    while True:
        # Sample a perturbative term and add it to the matrix
        sample = np.sort(np.random.choice(list(allowed_idxs), size=k, replace=False))
        print(f"Adding X on {sample}")
        ops = [X if i in sample else I for i in range(n)]
        term = reduce(np.kron, ops)
        sign = 1 if np.random.rand() < 0.5 else -1
        H += sign * term

        # Cut out the qubits which overlap with the perturbation
        bad = set()
        for Z_term in base_matrix:
            if np.max(Z_term[sample]) == 1:
                bad = bad | set(np.flatnonzero(Z_term))
            
        allowed_idxs = allowed_idxs.difference(bad)
        print(f"Cut indices {bad}; remaining: {allowed_idxs}")
        if len(allowed_idxs) < k:
            break

    return H


def gibbs_state(H, beta):
    # diagonalize H
    E, U = eigh(H)
    # Boltzmann weights
    w = np.exp(-beta * E)
    Z = np.sum(w)
    rho = (U * w) @ U.conj().T / Z
    return rho

def fidelity(rho1, rho2):
    # sqrt of rho1
    sr = sqrtm(rho1)
    # intermediate product
    prod = sr @ rho2 @ sr
    # fidelity
    return np.real(np.trace(sqrtm(prod)))**2

n = 10
m = 3*n
k = 3
beta = 10

betas = [0.1, 1, 5, 10, 20]

H, supp = random_Z_hamiltonian(n, m, k)
perturbation = random_X_perturbation(supp, n, k)

G = gibbs_state(H, beta)
Gprime = gibbs_state(H + perturbation, beta)

F = fidelity(G, Gprime)
print(f"[beta={beta}] Fidelity = {F}")
