import pandas as pd
import numpy as np
import argparse

def semicircle_law(m, e):
    """
    Compute r(m, e) = 1/2 * (sqrt(e/m) + sqrt(1 - e/m))^2.

    Parameters:
    - m: array-like or scalar
    - e: array-like or scalar

    Returns:
    - r: array-like or scalar
    """
    return np.round((0.5 * (np.sqrt(e / m) + np.sqrt(1 - e / m)) ** 2) * 100, 2)


def main():
    parser = argparse.ArgumentParser(description="Compute r column from CSV with m,n,k,e columns.")
    parser.add_argument("--input", '-i', help="Path to input CSV file")
    parser.add_argument("--output", '-o', default="output.csv", help="Path to output CSV file (default: output.csv)")
    args = parser.parse_args()

    # Read the CSV into a pandas DataFrame
    df = pd.read_csv(args.input)

    # Compute the r column
    df['r'] = semicircle_law(df['m'], df['e'])
    df['m/n'] = df['m'] / df['n']
    df = df.drop(columns=['m'])
    cols = ['n', 'm/n', 'k', 'e', 'r']
    df = df[cols]

    # Save to output CSV
    df.to_csv(args.output, index=False)
    print(f"Computed 'r' and saved to {args.output}")


if __name__ == "__main__":
    main()
