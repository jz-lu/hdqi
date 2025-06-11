"""
`classical.py`

Code file to generate classical max-k-XOR-SAT instances.
"""
import numpy as np
import argparse
from constants import CLASSICAL_FILE_PREFIX, generate_classical_identifier
from helper import rand_k_sparse_matrix

def main(args):
    ROOT = args.save
    SAVE_DATA = not args.nosave
    m = args.m; n = args.n; k = args.k
    NUM_TRIALS = args.trials

    data = np.stack([rand_k_sparse_matrix(m, n, k) for _ in range(NUM_TRIALS)], axis=0)

    if SAVE_DATA:
        IDENTIFIER = generate_classical_identifier(m, n, k, NUM_TRIALS)
        np.save(f"{ROOT}/{CLASSICAL_FILE_PREFIX}_{IDENTIFIER}.npy", data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample max-k-XOR-SAT instances"
    )

    parser.add_argument(
        "--m", "-m",
        type=int,
    )
    parser.add_argument(
        "--n", "-n",
        type=int,
    )
    parser.add_argument(
        "--k", "-k",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of instances you want to generate"
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

    args = parser.parse_args()
    main(args)
