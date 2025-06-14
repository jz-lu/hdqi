import os
import argparse
import subprocess
import glob

# Parameter grids for generation jobs
grid_ns = [100, 200, 400, 800]
grid_rs = [3, 6, 10]
grid_ks = [3, 4, 5, 6]

def determine_trials(n, r, k):
    m = n * r
    if k <= 4:
        if m <= 2000:
            return 50
        elif m <= 4000:
            return 25
        else:
            return 10
    elif k == 5:
        if m <= 2000:
            return 10
        elif m <= 4000:
            return 5
        else:
            return 2
    else:  # k == 6
        return 1

# Template for commuting.py batch files (gen)
TEMPLATE_GEN = """#!/bin/bash
#SBATCH -n 4  # number of cores
#SBATCH -t 0-12:00  # Runtime DAYS-HRS:MIN
#SBATCH -p sched_mit_hill  # Node partition, don't change this
#SBATCH --mem-per-cpu=4000  # Request 4G of memory per CPU
#SBATCH -o out_%A_%a.txt  # Redirect output to output_JOBID.txt
#SBATCH -e err_%A_%a.txt  # Redirect errors to error_JOBID.txt
#SBATCH --mail-type=END  # Mail when job ends
#SBATCH --mail-user=lujz@mit.edu  # Email recipient

source activate myenv
echo "Hello World"
python commuting.py -m {m} -n {n} -k {k} --trials {trials} -s {root} -d --key $SLURM_ARRAY_TASK_ID
echo "Done!"
"""

# Template for SA.py batch files (opt)
TEMPLATE_OPT = """#!/bin/bash
#SBATCH -n 4  # number of cores
#SBATCH -t 0-8:00  # Runtime DAYS-HRS:MIN
#SBATCH -p sched_mit_hill  # Node partition, don't change this
#SBATCH --mem-per-cpu=4000  # Request 4G of memory per CPU
#SBATCH -o out_%A_%a.txt  # Redirect output to output_JOBID.txt
#SBATCH -e err_%A_%a.txt  # Redirect errors to error_JOBID.txt
#SBATCH --mail-type=END  # Mail when job ends
#SBATCH --mail-user=lujz@mit.edu  # Email recipient

source activate myenv
echo "Starting SA optimization"
python SA.py -m {m} -n {n} -k {k} --trials {trials} --root {root} -o {root}
echo "Done optimization!"
"""

def create_gen(root):
    for n in grid_ns:
        for r in grid_rs:
            for k in grid_ks:
                m = n * r
                trials = determine_trials(n, r, k)
                filename = f"BAT_{n}_{r}_{k}"
                with open(os.path.join(root, filename), "w") as f:
                    f.write(TEMPLATE_GEN.format(n=n, m=m, k=k, trials=trials, root=root))
                print(f"Created batch file: {filename} in {root}")


def submit_gen(root):
    batch_files = glob.glob(os.path.join(root, "BAT_*_*_*"))
    for bf in batch_files:
        base = os.path.basename(bf)
        try:
            _, n_str, r_str, k_str = base.split('_')
            n, r, k = int(n_str), int(r_str), int(k_str)
        except ValueError:
            print(f"Skipping unrecognized file: {base}")
            continue
        trials = determine_trials(n, r, k)
        if trials == 100:
            cmd = ["sbatch", bf]
        else:
            if trials % 100 != 0:
                raise ValueError(f"Trials {trials} for {base} is not divisible by 100")
            p = trials // 100
            cmd = ["sbatch", f"--array=0-{p-1}", bf]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


def create_opt(root):
    # DiagCommuting_TYPE1_m<m>n<n>k<k>_t<trials>.npy
    pattern = os.path.join(root, "DiagCommuting_TYPE1_m*")
    files = glob.glob(pattern.replace('m*', 'm*'))
    for fn in files:
        parts = os.path.basename(fn).rstrip('.npy').split('_')
        try:
            mrk = parts[2]
            tpart = parts[3]
            m = int(mrk.split('m')[1].split('n')[0])
            n = int(mrk.split('n')[1].split('k')[0])
            k = int(mrk.split('k')[1])
            trials = int(tpart.split('t')[1])
        except Exception as e:
            print(f"Skipping file due to parse error {fn}: {e}")
            continue
        filename = f"BATSA_{m}_{n}_{k}_{trials}"
        with open(os.path.join(root, filename), 'w') as f:
            f.write(TEMPLATE_OPT.format(m=m, n=n, k=k, trials=trials, root=root))
        print(f"Created SA batch file: {filename} in {root}")


def submit_opt(root):
    files = glob.glob(os.path.join(root, "BATSA_*_*_*_*"))
    for bf in files:
        print(f"Submitting {os.path.basename(bf)}")
        subprocess.run(["sbatch", bf], check=True)


def main():
    parser = argparse.ArgumentParser(description="Generate or submit SLURM batch jobs for commuting and optimization tasks.")
    parser.add_argument(
        "--mode", choices=["create", "submit"], required=True,
        help="Mode: create batch files or submit them."
    )
    parser.add_argument(
        "--type", choices=["gen", "opt"], required=True,
        help="Type: 'gen' for commuting.py jobs, 'opt' for SA optimization jobs."
    )
    parser.add_argument(
        "--root", default='.',
        help="Root directory for batch file creation and submission (default: current directory)."
    )
    args = parser.parse_args()

    # Validate root path
    if not os.path.isdir(args.root):
        parser.error(f"Root path '{args.root}' is not a valid directory.")

    if args.type == 'gen':
        if args.mode == 'create':
            create_gen(args.root)
        else:
            submit_gen(args.root)
    else:  # opt
        if args.mode == 'create':
            create_opt(args.root)
        else:
            submit_opt(args.root)

if __name__ == '__main__':
    main()
