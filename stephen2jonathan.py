import os
import re
import numpy as np
from helper import check_symplectic_consistency, find_diagonalizing_Clifford, apply_Clifford_circuit, \
                    transform_standard_Paulis

def load_cham_matrix(path: str) -> np.ndarray:
    """
    Load a binary Cham matrix from a TSV file with naming pattern `cham_<n>_<m>_<k>.tsv`.

    The file is expected to have m+1 rows (including a header row), where each of the next m rows
    contains a list of integer indices (1-based) between 1 and 2*n, separated by tabs. The function
    discards the header, reads the m data rows, and constructs a binary NumPy array of shape (m, 2*n).
    The returned matrix is the transpose of this matrix.

    Args:
        path: Path to the TSV file, whose basename matches `cham_<n>_<m>_<k>.tsv`.

    Returns:
        A NumPy array of shape (2*n, m), with ones at positions specified in each column and zeros elsewhere.

    Raises:
        ValueError: If the filename does not match the expected pattern or if the number of rows
                    read does not match m.
    """
    # Extract n and m from filename
    basename = os.path.basename(path)
    match = re.match(r'cham_(\d+)_(\d+)_(\d+)\.tsv$', basename)
    if not match:
        raise ValueError(f"Filename '{basename}' does not match pattern 'cham_<n>_<m>_<k>.tsv'.")

    n = int(match.group(1))
    m = int(match.group(2))
    k = int(match.group(3))

    # Prepare the output matrix
    matrix = np.zeros((m, 2 * n), dtype=np.int8)

    with open(path, 'r') as f:
        # Discard header row
        _ = f.readline()

        # Process each data row
        for i in range(m):
            line = f.readline()
            if not line:
                raise ValueError(f"Expected {m} data rows, but file ended after {i} rows.")
            # Split on tabs, convert to integers
            indices = [int(x) for x in line.strip().split('\t') if x]
            for idx in indices:
                if idx < 0 or idx >= 2 * n:
                    raise ValueError(f"Index {idx} out of bounds for n={n}.")
                matrix[i, idx] = 1

        # Check for extra unexpected rows
        extra = f.readline()
        if extra:
            raise ValueError(f"File has more than {m} data rows.")
        matrix = matrix.T

        # assert check_symplectic_consistency(matrix, m, n)

    return matrix, n, m, k


def find_cham_files(directory: str) -> list[str]:
    """
    Search a directory for files matching the pattern `cham_<n>_<m>_<k>.tsv` and return their paths.

    Args:
        directory: Path to the directory to search.

    Returns:
        A list of full file paths for every file in `directory` whose basename matches
        `cham_<n>_<m>_<k>.tsv`.
    """
    pattern = re.compile(r'^cham_(\d+)_(\d+)_(\d+)\.tsv$')
    matched_files = []

    try:
        for fname in os.listdir(directory):
            if pattern.match(fname):
                matched_files.append(os.path.join(directory, fname))
    except FileNotFoundError:
        raise ValueError(f"Directory '{directory}' does not exist.")
    except NotADirectoryError:
        raise ValueError(f"Path '{directory}' is not a directory.")

    return matched_files

if __name__ == "__main__":
    IN_PATH = "/Users/jzlu/Dropbox/data_hdqi/Stephen_in"
    OUT_PATH = "/Users/jzlu/Dropbox/data_hdqi/Stephen_out"
    files = find_cham_files(IN_PATH)
    for file in files:
        matrix, n, m, k = load_cham_matrix(file)
        this_out = f"{OUT_PATH}/Stephen_{m}_{n}_{k}.npy"
        np.save(this_out, matrix)
        print(f"Saved {this_out}")

        clifford = find_diagonalizing_Clifford(matrix, m, n)
        moves = transform_standard_Paulis(clifford, n, inverse=True, include_y=True)
        assert moves.shape == (2*n, 3*n)
        apply_Clifford_circuit(clifford, matrix, n, inplace=True)
        matrix = matrix[:n, :] # cut off the zero part of each matrix
        assert matrix.shape == (n, m), f"Matrix shape should be {(n, m)} but is {matrix.shape}"
        this_out = f"{OUT_PATH}/DiagStephen_{m}_{n}_{k}.npy"
        moves_out = f"{OUT_PATH}/MovesStephen_{m}_{n}_{k}.npy"
        
        np.save(moves_out, moves)
        print(f"Saved {moves_out}")
        np.save(this_out, matrix)
        print(f"Saved {this_out}")


