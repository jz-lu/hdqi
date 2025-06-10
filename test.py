import numpy as np

def gf2_rank(M):
    """Compute the rank over GF(2)."""
    M = M.copy()
    nrows, ncols = M.shape
    rank = 0
    for col in range(ncols):
        pivot_row = None
        for row in range(rank, nrows):
            if M[row, col]:
                pivot_row = row
                break
        if pivot_row is not None:
            if pivot_row != rank:
                M[[rank, pivot_row]] = M[[pivot_row, rank]]
            for row in range(rank + 1, nrows):
                if M[row, col]:
                    M[row] ^= M[rank]
            rank += 1
    return rank

def extract_independent_columns_and_dependencies(M):
    M = M.copy()
    M = M % 2  # Ensure binary
    nrows, ncols = M.shape

    # Track column indices of S
    S_cols = []
    S = []

    # Keep track of combinations used to form dependent columns
    dependent_columns = []

    # Matrix to track combinations (over GF(2)) of S columns
    comb_matrix = []

    # Use reduced row echelon form (RREF) over GF(2)
    A = np.zeros((0, 0), dtype=int)  # current basis

    for j in range(ncols):
        col = M[:, j]
        if len(S_cols) == 0:
            # First column is always independent
            S.append(col)
            S_cols.append(j)
            A = np.column_stack([col])
            comb_matrix.append(np.array([1]))
            continue

        # Try to solve A x = col over GF(2)
        A_mat = np.array(S).T % 2  # shape (2n x k)
        try:
            # Use least-squares mod 2: A x = col
            # Build augmented matrix [A | col]
            aug = np.column_stack((A_mat, col))
            aug = aug.astype(np.uint8)
            reduced = gaussian_elimination_gf2(aug)

            # If last row of reduced matrix has 0...0 | 1, then no solution
            if not is_consistent_gf2(reduced):
                # Independent
                S.append(col)
                S_cols.append(j)
            else:
                # Dependent: extract solution
                solution = back_substitution_gf2(reduced)
                support = {S_cols[i] for i, v in enumerate(solution) if v}
                dependent_columns.append((j, support))
        except Exception as e:
            raise RuntimeError(f"Error solving linear system for column {j}: {e}")

    S_mat = np.column_stack(S)
    return S_mat, dependent_columns


def gaussian_elimination_gf2(A):
    """Gaussian elimination over GF(2) on augmented matrix A."""
    A = A.copy() % 2
    nrows, ncols = A.shape
    rank = 0
    for col in range(ncols - 1):  # exclude last column (augmented part)
        pivot = None
        for row in range(rank, nrows):
            if A[row, col] == 1:
                pivot = row
                break
        if pivot is None:
            continue
        if pivot != rank:
            A[[rank, pivot]] = A[[pivot, rank]]
        for row in range(nrows):
            if row != rank and A[row, col] == 1:
                A[row] ^= A[rank]
        rank += 1
    return A


def is_consistent_gf2(R):
    """Check if the reduced system is consistent over GF(2)."""
    for row in R:
        if not np.any(row[:-1]) and row[-1] == 1:
            return False
    return True


def back_substitution_gf2(R):
    """Perform back substitution over GF(2) to get solution."""
    nrows, ncols = R.shape
    nvars = ncols - 1
    x = np.zeros(nvars, dtype=int)

    for i in reversed(range(nrows)):
        row = R[i, :-1]
        rhs = R[i, -1]
        nz = np.nonzero(row)[0]
        if len(nz) == 0:
            if rhs == 1:
                raise ValueError("Inconsistent system")
            else:
                continue
        pivot = nz[0]
        sum_others = sum(x[j] for j in nz[1:]) % 2
        x[pivot] = (rhs - sum_others) % 2

    return x

# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    M = np.random.randint(0, 2, size=(8, 10), dtype=int)
    S, dependencies = extract_independent_columns_and_dependencies(M)

    print("Independent matrix S (shape {}):".format(S.shape))
    print(S)
    print("\nDependencies:")
    for dep_col, dep_set in dependencies:
        print(f"Column {dep_col} = sum of columns {sorted(dep_set)} in S")
