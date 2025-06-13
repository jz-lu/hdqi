import numpy as np
from pyldpc import make_ldpc, encode, decode

def make_random_ldpc_H(n, m, k, seed=None):
    rng = np.random.default_rng(seed)
    H = np.zeros((n, m), dtype=int)
    for i in range(n):
        ones = rng.choice(m, size=k, replace=False)
        H[i, ones] = 1
    return H

def estimate_bler_bsc_pyldpc(H, p_list, snr, max_frames=10000, min_errors=10, maxiter=500000):
    # build code
    print(np.sum(H.sum(axis=0)))
    print(f"d_c = {np.mean(H.sum(axis=1))}")
    print(f"d_v = {np.mean(H.sum(axis=0))}")
    H, G = make_ldpc(500,
                     d_v=2, 
                     d_c=5,
                     systematic=True,
                     sparse=True)                   # :contentReference[oaicite:0]{index=0}
    n = G.shape[0]
    bler = []

    for p in p_list:
        blk_err = 0
        frames = 0

        while frames < max_frames and blk_err < min_errors:
            # message = all‑zero → v = zeros(k)
            k = G.shape[1]
            v = np.zeros(k, dtype=int)
            # encode → real BPSK y = +1/−1
            y = encode(G, v, snr)            # :contentReference[oaicite:1]{index=1}

            # flip p‑fraction of bits: choose random indices
            flips = np.random.rand(n) < p
            y[flips] *= -1

            # decode
            d_hat = decode(H, y, snr, maxiter=maxiter)

            # check syndrome H·d_hat %2 == 0
            if np.any((H @ d_hat) % 2):
                blk_err += 1
            frames += 1

        bler.append(blk_err/frames)
        print(f"p={p:.4f} → BLER={bler[-1]:.3%} after {frames} frames")

    return np.array(bler)

if __name__ == "__main__":
    # LDPC dims & weight
    n, m, k_chk = 50, 500, 5
    H = make_random_ldpc_H(n, m, k_chk)

    # flip‑probabilities and SNR setting
    p_list = np.linspace(0.001, 0.01, 20)
    snr = 5.0   # in dB, choose something moderate

    bler = estimate_bler_bsc_pyldpc(H, p_list, snr)

    idx = np.where(bler > 0.01)[0]
    if idx.size:
        print(f"\nBLER crosses 1% at p ≈ {p_list[idx[0]]:.4f} (BLER={bler[idx[0]]:.3%})")
    else:
        print("\nBLER never exceeded 1% in the tested range.")
