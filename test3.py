def idx2params(idx):
    """
    Compute a parameter tuple from idx.
    We have a finite number of parameters, given by some n's, some m/n's and some k's.
    """
    ns = [800]
    m_over_ns = [3, 6, 10]
    ks = [3, 4, 5, 6]

    n_idx = idx % len(ns)
    idx = idx // len(ns)
    m_idx = idx % len(m_over_ns)
    idx = idx // len(m_over_ns)
    k_idx = idx % len(ks)
    n = ns[n_idx]
    return (n, n * m_over_ns[m_idx], ks[k_idx])

for i in range(12):
    print(idx2params(i))