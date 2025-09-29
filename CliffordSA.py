"""
`CliffordSA.py`: a simulated annealing-type algorithm with moves given by random local Cliffords.
This heuristic beats HDQI+BP+Semicircle law on random local commuting Hamiltonians.

Code & Algorithm by NS, adapted for multi-trial SLURM compute by JZL.
"""

import re
import os
import stim
import numpy as np
import time
import argparse
import glob
import subprocess
import sys

def list_cham_files(dirpath):
    pattern = os.path.join(dirpath, "cham_*_*_*.tsv")
    files = glob.glob(pattern)
    return sorted(files)


def parse_cham_filename(filename):
    # get just the filename in case a path is included
    base = os.path.basename(filename)
    
    if not base.startswith("cham_") or not base.endswith(".tsv"):
        raise ValueError(f"Invalid filename: {filename}")
    
    parts = base[:-4].split("_")  # remove ".tsv" and split
    if len(parts) != 4:  # ["cham", n, m, k]
        raise ValueError(f"Unexpected format: {filename}")
    
    n, m, k = map(int, parts[1:])
    return n, m, k


def parse_hamiltonian(filepath):
    """
    Parses a TSV file describing Pauli operators and returns them as a list
    of stim.PauliString objects. Assumes 'cham_<n>_<m>_<k>.tsv' format.
    
    Returns:
        A tuple containing:
        - A list of stim.PauliString objects representing the Hamiltonian.
        - The number of qubits (n).
    """
    filename = os.path.basename(filepath)
    n = int(re.search(r'cham_(\d+)', filename).group(1))
    hamiltonian = []

    with open(filepath, 'r') as f:
        next(f)  # Skip the header line.
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            indices = {int(x) for x in line.split()}
            
            pauli_string = ["_"] * n
            for i in range(n):
                is_x = i in indices
                is_z = (i + n) in indices
                
                if is_x and is_z:
                    pauli_string[i] = 'Y'
                elif is_x:
                    pauli_string[i] = 'X'
                elif is_z:
                    pauli_string[i] = 'Z'
            
            hamiltonian.append(stim.PauliString("".join(pauli_string)))
            
    return hamiltonian, n


def build_connectivity_map(hamiltonian, n):
    """
    Creates a map from each qubit to the Hamiltonian terms that act on it.

    Returns:
        A dict where keys are qubit indices and values are lists of the
        hamiltonian terms (stim.PauliString) that act on that qubit.
    """
    connectivity = {i: [] for i in range(n)}
    for term in hamiltonian:
        for i in range(n):
            # stim.PauliString convention is that 0 corresponds to Identity
            if term[i] != 0:
                connectivity[i].append(term)
    return connectivity


def calculate_energy(simulator, hamiltonian):
    """Calculates the total expectation value of a Hamiltonian for a given state."""
    total_energy = 0.0
    for term in hamiltonian:
        total_energy += simulator.peek_observable_expectation(term)
    return total_energy

def calculate_sat_frac(m, E):
    return (m - E) / (2*m)


def simulated_annealing_search(hamiltonian, n, max_sweeps=1000, initial_beta=0.0, beta_step=0.01):
    """
    Performs simulated annealing to find a low-energy stabilizer state by applying
    single-qubit Clifford gates.

    Args:
        hamiltonian (list[stim.PauliString]): The Hamiltonian to minimize.
        n (int): The number of qubits.
        max_sweeps (int): The number of annealing sweeps to perform.
        initial_beta (float): The starting inverse temperature.
        beta_step (float): How much to increase beta after each sweep.
    
    Returns:
        A tuple containing:
        - The final stim.TableauSimulator representing the optimized state.
        - The minimized energy.
    """
    connectivity_map = build_connectivity_map(hamiltonian, n)
    
    simulator = stim.TableauSimulator()
    # Start in a random-ish product state by applying random single-qubit Cliffords.
    for i in range(n):
        if np.random.random() < 0.5:
            simulator.h(i)
        if np.random.random() < 0.5:
            simulator.s(i)
        if np.random.random() < 0.5:
            simulator.x(i)
        if np.random.random() < 0.5:
            simulator.z(i)
        
    current_energy = calculate_energy(simulator, hamiltonian)
    print(f"Initial state energy: {current_energy:.4f}")

    beta = initial_beta
    single_qubit_cliffords = ["X", "Y", "Z", "H", "S"]

    # Create a list of all possible single-qubit Clifford moves.
    all_possible_moves = []
    for qubit_idx in range(n):
        for op_name in single_qubit_cliffords:
            all_possible_moves.append((op_name, qubit_idx))

    # Main annealing loop
    for sweep in range(max_sweeps):
        start = time.time()
        # Shuffle the order of moves to try in each sweep
        np.random.shuffle(all_possible_moves)

        for op_name, qubit_idx in all_possible_moves:
            affected_terms = connectivity_map[qubit_idx]
            if not affected_terms:
                continue

            # Calculate the energy change (delta) that this move would cause.
            local_energy_before = calculate_energy(simulator, affected_terms)
            
            temp_simulator = simulator.copy()
            getattr(temp_simulator, op_name.lower())(qubit_idx)
            local_energy_after = calculate_energy(temp_simulator, affected_terms)
            
            delta_energy = local_energy_after - local_energy_before

            # Decide whether to accept the move
            accept_move = False
            if delta_energy < 0:
                # Always accept moves that lower the energy
                accept_move = True
            else:
                # Accept moves that raise the energy with a certain probability
                acceptance_prob = np.exp(-delta_energy * beta)
                if np.random.random() < acceptance_prob:
                    accept_move = True
            
            if accept_move:
                # If accepted, apply the move to the main simulator and update energy
                getattr(simulator, op_name.lower())(qubit_idx)
                current_energy += delta_energy

        current_sat_frac = calculate_sat_frac(len(hamiltonian), current_energy)
        # print(f"Sweep {sweep + 1}/{max_sweeps}: Beta = {beta:.2f}, Energy = {current_energy:.4f} <s> / m = {current_sat_frac:.4f}")
        # Increase inverse temperature for the next sweep
        beta += beta_step
        end = time.time()
        print(f"\t[Sweep {sweep}] Took {round(end-start, 2)}s")

    # Sanity-check the energy
    final_energy_recalculated = calculate_energy(simulator, hamiltonian)
    assert final_energy_recalculated == current_energy

    print(f"\nAnnealing finished after {max_sweeps} sweeps.")
    print(f"Final minimized energy: {current_energy:.4f}")
    return current_energy


def main():
    """
    Main execution function to parse arguments and run the simulated annealing.
    """
    parser = argparse.ArgumentParser(description="Find low-energy stabilizer states using simulated annealing.")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "stim"])

    # --- Required Argument ---
    parser.add_argument("slurm_idx", type=int, help="Index into which file to run")

    # --- Optional Annealing Arguments with Defaults ---
    parser.add_argument(
        "--max-sweeps",
        type=int,
        default=80,
        help="Total number of sweeps to perform."
    )
    parser.add_argument(
        "--beta-step",
        type=float,
        default=0.025,
        help="Inverse temperature increment per sweep."
    )
    parser.add_argument(
        '--initial-beta',
        type=float, 
        default=1,
        help="Starting inverse temperature beta."
    )
    parser.add_argument(
        '--trials',
        type=int, 
        default=20,
        help="Number of experimental trials."
    )
    parser.add_argument(
        '--filepath',
        type=str,
        default='/Users/jzlu/Dropbox/data_hdqi/Stephen_in/',
        help="Filepath with cham files"
    )
    
    args = parser.parse_args()
    file_list = list_cham_files(args.filepath)

    filepath = file_list[args.slurm_idx]
    n, m, k = parse_cham_filename(filepath)
    print(f"Optimizing {filepath} (n={n}, m={m}, k={k}) for {args.trials} trials.")

    start_time = time.time()
    hamiltonian, n_qubits = parse_hamiltonian(filepath)
    parsing_time = time.time() - start_time
    print(f"Parsed Hamiltonian with {len(hamiltonian)} terms on {n_qubits} qubits in {parsing_time:.2f}s.")

    sat_fracs = np.zeros(args.trials)

    for trial in range(args.trials):
        start_time = time.time()
        energy = simulated_annealing_search(
            hamiltonian,
            n_qubits,
            max_sweeps=args.max_sweeps,
            initial_beta=args.initial_beta,
            beta_step=args.beta_step
        )
        search_time = time.time() - start_time
        minutes = int(search_time // 60)
        seconds = int(search_time % 60)
        sat_fracs[trial] = calculate_sat_frac(len(hamiltonian), energy)
        print(f"[{trial+1}] Satisfied {round(sat_fracs[trial] * 100, 2)}% in {minutes}m {seconds}s.")
    
    np.save(f"CSA_{n}_{m}_{k}.npy", sat_fracs)
    

if __name__ == '__main__':
    main()
