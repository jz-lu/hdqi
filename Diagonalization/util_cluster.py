"""
`util_cluster.py`

Code file to perform some basic data manipulations for interactions with slurm-type
HPC hardware. The main function is zipping together stuff.
"""
import os
import re
import numpy as np
from pathlib import Path
from constants import COMMUTE_FILE_PREFIX, MOVES_FILE_PREFIX, DIAG_FILE_PREFIX

def concat_keyed_files(directory, pfx):
    """
    If you're parallelizing trials over the cluster, you might want to combine them back 
    into a single file when you're done. This function automates this task completely.

    Scan a directory for files matching pattern:
      <stuff>_TYPE1_m<m>n<n>k<k>_t<t>_KEY<KEY>.npy
    Group by TYPE, m, n, k, sum t's, load arrays, concatenate along axis=0,
    and save as:
      <stuff>_TYPE1_m<m>n<n>k<k>_t<total_t>.npy

    Parameters
    ----------
    directory : str or Path
        Path to folder containing the .npy files.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    # Regex to capture TYPE, m, n, k, t
    pattern = re.compile(
        rf"^({pfx}_[^_]+_m(\d+)n(\d+)k(\d+))_t(\d+)_KEY.*\.npy$"
    )

    # Group files by (prefix, m, n, k)
    groups = {}
    for file in directory.iterdir():
        if file.suffix.lower() != ".npy":
            continue
        m = pattern.match(file.name)
        if not m:
            continue
        prefix, m_val, n_val, k_val, t_val = m.groups()
        key = (prefix, int(m_val), int(n_val), int(k_val))
        groups.setdefault(key, []).append((file, int(t_val)))

    # Process each group
    for (prefix, m_val, n_val, k_val), items in groups.items():
        # Sort by t to maintain order
        items.sort(key=lambda x: x[1])

        arrays = []
        total_t = 0
        for file, t in items:
            try:
                arr = np.load(file)
            except Exception as e:
                print(f"Warning: failed loading {file.name}: {e}")
                continue
            arrays.append(arr)
            total_t += t

        if not arrays:
            print(f"No valid arrays for {prefix}, skipping")
            continue

        try:
            concatenated = np.concatenate(arrays, axis=0)
        except ValueError as e:
            print(f"Error concatenating group {prefix}: {e}")
            continue

        out_name = f"{prefix}_t{total_t}.npy"
        out_path = directory / out_name
        try:
            np.save(out_path, concatenated)
            print(f"Saved: {out_name}")
        except Exception as e:
            print(f"Error saving {out_name}: {e}")

if __name__ == "__main__":
    for pfx in [COMMUTE_FILE_PREFIX, MOVES_FILE_PREFIX, DIAG_FILE_PREFIX]:
        concat_keyed_files("/Users/jzlu/Dropbox/data_hdqi", pfx)