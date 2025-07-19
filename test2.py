import os
import re
import shutil

def copy_files_with_n_greater_than_800(ROOT, OUT_DIR):
    """
    Searches for files under ROOT matching the pattern *_<m>_<n>_<k>.*.
    If n > 800, copies them to OUT_DIR.
    """
    # Regular expression to match the filename pattern
    pattern = re.compile(r'^.*_(\d+)_(\d+)_(\d+)\..*$')

    # Ensure output directory exists
    os.makedirs(OUT_DIR, exist_ok=True)

    # Walk through all files in the directory tree
    for dirpath, _, filenames in os.walk(ROOT):
        for fname in filenames:
            match = pattern.match(fname)
            if match:
                m, n, k = map(int, match.groups())
                if n > 800:
                    src_path = os.path.join(dirpath, fname)
                    dst_path = os.path.join(OUT_DIR, fname)
                    shutil.copy2(src_path, dst_path)
                    print(f"Copied: {src_path} -> {dst_path}")

copy_files_with_n_greater_than_800("../data_hdqi/Stephen_out/all_instances/", "../data_hdqi/Stephen_out")