from math import comb
from fractions import Fraction

def f_K_ab_closed(a: int, b: int) -> Fraction:
    """
    Closed‑form for f(K_{a,b}) via a single alternating sum:
      f = 1/C(a+b,a) * sum_{k=0..a} (-1)^k * C(a,k) * C(a+b-k-1, b-1).
    """
    n = a + b
    total = 0
    for k in range(a+1):
        total += (-1)**k * comb(a, k) * comb(n - k - 1, b - 1)
    return Fraction(total, comb(n, a))

# --- Self‑check against brute force for a,b up to 6 ---
if __name__ == "__main__":
    import numpy as np, itertools, math
    from fractions import Fraction

    def f_bruteforce(a, b):
        n = a + b
        # build K_{a,b}
        G = np.zeros((n, n), int)
        for i in range(a):
            for j in range(a, n):
                G[i, j] = G[j, i] = 1
        total = 0
        for pi in itertools.permutations(range(n)):
            perm = list(pi)
            swaps = 0
            for t in range(n):
                idx = perm.index(t)
                while idx > t:
                    if G[perm[idx-1], perm[idx]]:
                        swaps += 1
                    perm[idx-1], perm[idx] = perm[idx], perm[idx-1]
                    idx -= 1
            total += (-1)**swaps
        return Fraction(total, math.factorial(n))

    mismatch = False
    for a in range(1, 7):
        for b in range(1, 7):
            c = f_K_ab_closed(a, b)
            d = f_bruteforce(a, b)
            if c != d:
                print(f"Mismatch at a={a}, b={b}: closed={c}, brute={d}")
                mismatch = True
    if not mismatch:
        print("All tests passed for 1 <= a,b <= 6!")
