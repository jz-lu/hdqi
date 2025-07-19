import os
import re
import shutil
import csv


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


def extract_emax_to_csv(ROOT, CSV_PATH):
    """
    Searches for files of the form log_<n>_<m>_<k>.txt under ROOT.
    Extracts (n, m, k, e) where e is parsed from the last line of each file.
    Writes the results to CSV_PATH.
    """
    log_pattern = re.compile(r'^log_(\d+)_(\d+)_(\d+)\.txt$')
    emax_pattern = re.compile(r'culled emax = ([\d.+-eE]+)')

    results = []

    for dirpath, _, filenames in os.walk(ROOT):
        for fname in filenames:
            match = log_pattern.match(fname)
            if match:
                n, m, k = map(int, match.groups())
                full_path = os.path.join(dirpath, fname)

                try:
                    with open(full_path, 'r') as f:
                        lines = f.readlines()
                        if not lines:
                            continue
                        last_line = lines[-1].strip()
                        e_match = emax_pattern.search(last_line)
                        if e_match:
                            e = float(e_match.group(1))
                            new_result = (n, m, k, e)
                            results.append(new_result)
                            print(f"Found {new_result}")
                        else:
                            print(f"Warning: No emax found in {full_path}")
                except Exception as ex:
                    print(f"Error reading {full_path}: {ex}")

    # Write results to CSV
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    with open(CSV_PATH, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['n', 'm', 'k', 'e'])
        writer.writerows(results)
    
    print(f"Saved {len(results)} entries to {CSV_PATH}")


extract_emax_to_csv("/Users/jzlu/Downloads/big2", "/Users/jzlu/Downloads/big2large.csv")