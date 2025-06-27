"""
`Stephen_SA.py`

Code file to perform simple simulated annealing (SA) experiments on 
Stephen's instances.
"""
from simanneal import Annealer
from random import randint
import numpy as np
import argparse
import time
import os
import re
from constants import generate_identifier, DIAG_FILE_PREFIX, MOVES_FILE_PREFIX, \
                      generate_classical_identifier, CLASSICAL_FILE_PREFIX


def find_starting_temp(m, prob):
    """
    Find a starting temperature such that Pr[accept] >= prob
    for all trial steps. Pr[x -> x'] = exp((g(x) - g(x')) / T)
    when g(x') > g(x). Doing some algebra, we find that 
    T >= 2*m / log(1/prob).
    """
    assert 0 < prob < 1, f"Probability must be in (0, 1), but is {prob}"
    return 2*m / np.log(1/prob)


def find_ending_temp(prob):
    """
    Find an ending temperature such that Pr[accept] <= prob
    for all trial steps. 
    Doing some algebra, we find that T <= 1/log(1/prob) is a conservative estimate.
    """
    assert 0 < prob < 1, f"Probability must be in (0, 1), but is {prob}"
    return 1 / np.log(1/prob)


class Max_XOR_SAT_Annealer(Annealer):
    def __init__(self, state, clauses, target, move_space, num_moves):
        """
        Set up a SA instance for max-XOR-SAT. Want to find x such that |Bx - v| is minimized.

        Params:
            * state (np.ndarray): initial choice for the solution. Vector of length N. (initial x)
            * clauses (np.ndarray): M x N matrix specifying M equations of length N. (B)
            * target (np.ndarray): length M vector specifying desired value. (v)
            * move_space (np.ndarray): P x N matrix specifying P possible moves at each step.
              Here a "move" means adding a chosen vector to the current state (mod 2).
            * num_moves (int): P as defined above.
        
        Returns:
            * None
        """
        super(Max_XOR_SAT_Annealer, self).__init__(state)
        self.clauses = clauses # B
        self.target = target # v
        self.move_space = move_space
        # print("Possible moves:")
        # for move in move_space:
        #     print(move)
        self.num_moves = num_moves

    def move(self):
        """
        Pick a random element of the move space and go there.
        """
        i = randint(0, self.num_moves - 1)
        self.state ^= self.move_space[i]

    def energy(self):
        """
        For max-XOR-SAT, the energy function is g(x) = -f(x) = #UNSAT - #SAT.
        We implement it by first calculating t = Bx + v (mod 2).
        Then t_i = 0 if SAT and 1 if UNSAT. So we map t -> 2t - 1 so
        that now t_i = 1 if UNSAT and -1 if SAT. Then we return sum_i t_i = g(x).
        """
        diff = np.mod(np.mod(self.clauses @ self.state, 2) + self.target, 2) # |Bx - v|
        g = np.sum(2*diff - 1)
        return g


def run_annealer(clauses, initial_state, target, \
                 move_space, num_moves, Tmin, Tmax, \
                 num_steps, return_solution=False, auto=False, wait_time=1):
    """
    Run the simulated annealer on a given instance.

    Params:
        * clauses (np.ndarray): M x N matrix specifying M equations of length N. (B)
        * initial_state (np.ndarray): initial choice for the solution. Vector of length N. (initial x)
        * target (np.ndarray): length M vector specifying desired value. (v)
        * move_space (np.ndarray): P x N matrix specifying P possible moves at each step.
          Here a "move" means adding a chosen vector to the current state (mod 2).
        * num_moves (int): P as defined above.
        * Tmin (float): minimum temperature of the process. This should be large (e.g. 10000).
        * Tmax (float): maximum temperature of the process. This should be small (e.g. 0.01).
        * num_steps (int): number of steps taken by the annealer. This should be large (e.g. 1e5).
        * return_solution (bool): flag on whether to return the solution found.
        * auto (bool): flag on whether to use the automatic scheduler. If this is on, 
          `initial_state`, `num_moves`, `Tmin`, and `Tmax` are irrelevant.
        * wait_time (int): how many minutes you're willing to wait (only used if `auto` is on).
    
    Returns:
        * if `return_solution`, returns pair (solution, energy). Otherwise, returns just the 
          energy of the solution found at the end of the process.
    """
    annealer = Max_XOR_SAT_Annealer(initial_state, clauses, target, move_space, num_moves)
    annealer.steps = num_steps
    annealer.Tmin, annealer.Tmax = Tmin, Tmax
    annealer.updates = 100   # Number of updates (by default an update prints to stdout)
    if auto:
        auto_schedule = annealer.auto(minutes=wait_time) 
        annealer.set_schedule(auto_schedule)
    if return_solution:
        return annealer.anneal()
    else:
        _, best_energy = annealer.anneal()
        return best_energy


def find_files(directory: str) -> list[str]:
    """
    Search a directory for files matching the pattern `DiagStephen_<n>_<m>_<k>.tsv` and return their paths.

    Args:
        directory: Path to the directory to search.

    Returns:
        A list of full file paths for every file in `directory` whose basename matches
        `DiagStephen_<n>_<m>_<k>.tsv`.
    """
    pattern = re.compile(r'^DiagStephen_(\d+)_(\d+)_(\d+)\.npy$')
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


def save_csv(array, filepath):
    """
    Save a 2D NumPy array to a CSV file.

    Parameters:
        array (np.ndarray): The 2D array to save.
        filepath (str): The path to the output CSV file.
    """
    if array.ndim != 2:
        raise ValueError("Input array must be 2-dimensional.")

    np.savetxt(filepath, array, delimiter=",", fmt="%.18e")


def main(args):
    # Process arguments
    ROOT = args.root
    OUTDIR = args.out
    if OUTDIR is None:
        OUTDIR = ROOT
    TYPE = args.type
    AUTO = args.auto
    WAIT_TIME = args.waittime
    print(f"Executing type {TYPE} SA with {'auto' if AUTO else 'manual'} scheduling.")

    filenames = find_files(ROOT)
    results = np.zeros((len(filenames), 4))
    for idx, filename in enumerate(filenames):
        instance = np.load(filename).T
        basename = os.path.basename(filename)
        match = re.match(r'^DiagStephen_(\d+)_(\d+)_(\d+)\.npy$', basename)
        if not match:
            raise ValueError(f"'{basename}' does not match pattern 'DiagStephen_<m>_<n>_<k>.npy'.")

        m = int(match.group(1))
        n = int(match.group(2))
        k = int(match.group(3))
        assert instance.shape == (m, n), f"Instanace shape should be {(m, n)} but is {instance.shape}"

        # Write down the starting and ending temperatures for annealing
        Tmin = 1e-2
        Tmax = find_starting_temp(m, 0.98)
        NUM_STEPS = 200_000

        if TYPE == 1:
            # Move space consists of local bit flips
            MOVE_SPACE = np.eye(n, dtype=np.int8)
        elif TYPE == 2:
            # Move space consists of Pauli flips
            raise SyntaxError("Not implemented yet.")
        else:
            raise ValueError(f"Invalid TYPE = {TYPE}")
        
        initial_state = np.random.randint(0, 2, size=n, dtype=np.int8) # random start
        target = np.random.randint(0, 2, size=m, dtype=np.int8) # random target
        NUM_MOVES = MOVE_SPACE.shape[0]

        start = time.perf_counter()
        energy = run_annealer(instance, initial_state, target, \
                 MOVE_SPACE, NUM_MOVES, Tmin, Tmax, \
                 NUM_STEPS, return_solution=False, auto=AUTO, wait_time=WAIT_TIME)
        end = time.perf_counter()
        minutes, seconds = divmod((end - start), 60)
        print(f"[m={m}, n={n}, k={k}] SA took {minutes} min {round(seconds, 4)}\n")
        
        # `energy`` is #UNSAT - #SAT. Proportion satisfied is #SAT / m.
        # We compute this using #UNSAT + #SAT = m, so #SAT = (m - `energy`)/2
        ratio = (m - energy) / (2 * m)
        print(f"[m={m}, n={n}, k={k}] Ratio: {ratio}\n")
        results[idx] = np.array([m, n, k, ratio])
        np.save(f"{OUTDIR}/SA_Stephen.npy", results)
        save_csv(f"{OUTDIR}/SA_Stephen.csv", results)
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run simulated annealing on Stephen's instances"
    )

    parser.add_argument(
        "--type",
        type=int,
        default=1,
        choices=[1, 2],
        help="1 for Z-local flips, 2 for natural local flips",
    )

    parser.add_argument(
        "--root", "-i",
        type=str,
        help="Import file directory",
        default='.'
    )

    parser.add_argument(
        "--out", "-o",
        type=str,
        help="Output file directory, defaults to import file directory",
    )

    parser.add_argument(
        "--auto", "-a",
        action="store_true",
        help="Use automatic scheduler instead of manual specs",
    )

    parser.add_argument(
        "--waittime", "-w",
        type=int,
        help="How many minutes you're willing to wait (if using automatic scheduler)",
        default=1
    )

    args = parser.parse_args()
    main(args)


