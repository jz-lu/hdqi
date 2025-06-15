import numpy as np
import itertools
from math import factorial, gcd
import sympy as sp

# Define the symbol once
x = sp.symbols('x')

def reduce_fraction(a: int, b: int) -> str:
    """
    Reduce the fraction a/b to lowest terms and return as a string "num/den".
    Raises ZeroDivisionError if b == 0.
    """
    if b == 0:
        raise ZeroDivisionError("Denominator cannot be zero")
    # Compute greatest common divisor
    g = gcd(a, b)
    # Divide out gcd
    num, den = a // g, b // g
    # Ensure the denominator is positive
    if den < 0:
        num, den = -num, -den
    return f"{num}/{den}"

def f_of_G(adj_matrix):
    """
    Compute f(G) = (1/n!) * sum_pi sgn_G(pi)
    where sgn_G(pi) = (-1)^k and k is the number of nearest-neighbor swaps 
    along edges needed to sort pi to identity via bubble sort.
    """
    n = adj_matrix.shape[0]
    total = 0
    for pi in itertools.permutations(range(n)):
        perm = list(pi)
        swaps_on_edges = 0
        # Bubble-sort-like procedure to bring perm to [0,1,2,...,n-1]
        perm_copy = perm.copy()
        for target in range(n):
            # Find the current position of the target element
            idx = perm_copy.index(target)
            # Bubble it down to position 'target' via adjacent swaps
            while idx > target:
                i, j = idx - 1, idx
                # If the swap is along an edge in G, count it
                if adj_matrix[perm_copy[i], perm_copy[j]] == 1:
                    swaps_on_edges += 1
                # Perform the swap
                perm_copy[i], perm_copy[j] = perm_copy[j], perm_copy[i]
                idx -= 1
        total += (-1) ** swaps_on_edges
    if total != 0:
        return reduce_fraction(total, factorial(n))
    else:
        return 0

def closed_form_chain(n):
    """
    Compute the closed‐form f(P_n):
      f(P_n) = 0                     if n is even,
      f(P_n) = (-1)^((n-1)/2) * E_n/n! if n is odd,
    where E_n are the 'tangent' Euler numbers from tan(x) = sum E_n x^n/n!.
    """
    n = int(n)
    # Even n gives zero
    if n % 2 == 0:
        return sp.Integer(0)
    # Expand tan(x) series up to x^n
    series_tan = sp.series(sp.tan(x), x, 0, n+1).removeO()
    # Coefficient of x^n in the series is E_n/n!
    coeff = sp.simplify(series_tan.coeff(x, n))
    # Apply the closed‐form factor
    sign = (-1)**((n - 1)//2)
    return sp.simplify(sign * coeff)

def closed_form_star(n):
    return "0" if n%2 == 0 else f"1/{n}"

def chain_graph(n):
    """
    Generate the adjacency matrix of an undirected chain graph on n nodes:
    edges between (0,1), (1,2), ..., (n-2, n-1).
    """
    G = np.zeros((n, n), dtype=int)
    for i in range(n - 1):
        G[i, i + 1] = G[i + 1, i] = 1
    return G

def star_graph(n):
    """
    Adjacency matrix of the n‑node star graph:
      – v0 is the center, connected to v1,…,v_{n-1}.
      – No other edges.
    """
    G = np.zeros((n, n), dtype=int)
    for i in range(1, n):
        G[0, i] = G[i, 0] = 1
    return G

def cycle_graph(n):
    """
    Builds the adjacency matrix of the undirected n‐cycle:
    edges (0,1), (1,2), ..., (n-2, n-1), (n-1,0).
    
    Returns: n x n matrix of 0/1
    """
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        j = (i+1) % n
        adj[i,j] = adj[j,i] = 1
    return adj

# Test f_of_G on cycle graphs
print("===== CYCLE =====")
for n in range(1, 11):
    G_star = cycle_graph(n)
    val = f_of_G(G_star)
    claimed = 0
    print(f"n = {n} \t\t a = {val}")

# # Test f_of_G on star graphs
# print("===== STAR =====")
# for n in range(1, 10):
#     G_star = star_graph(n)
#     val = f_of_G(G_star)
#     claimed = closed_form_star(n)
#     print(f"n = {n} \t\t a = {val} \t\t\t expected = {claimed}")

# Test f(G) on chain graphs
print("===== CHAIN =====")
for n in range(1, 11):
    G_chain = chain_graph(n)
    val = f_of_G(G_chain)
    claimed = closed_form_chain(n)
    print(f"n = {n} \t\t a = {val} \t\t\t expected = {claimed}")


