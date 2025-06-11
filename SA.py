"""
`SA.py`

Code file to perform simple simulated annealing (SA) experiments on 
max-k-XOR-SAT problems.
"""
from simanneal import Annealer
from random import randint
import numpy as np
import argparse
import time
from constants import generate_identifier, DIAG_FILE_PREFIX, MOVES_FILE_PREFIX


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
        that now t_i = 1 if UNSAT and -1 if SAT. Then we return \sum_i t_i = g(x).
        """
        diff = np.mod(np.mod(self.clauses @ self.state, 2) + self.target, 2) # |Bx - v|
        return np.sum(2*diff - 1)


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
    if auto:
        auto_schedule = annealer.auto(minutes=wait_time) 
        annealer.set_schedule(auto_schedule)
    if return_solution:
        return annealer.anneal()
    else:
        _, best_energy = annealer.anneal()
        return best_energy


def main(args):
    # Process arguments
    m = args.numclauses
    n = args.size
    k = args.locality
    NUM_TRIALS = args.numtrials
    ROOT = args.root
    TYPE = args.type
    print(f"Executing tyoe {TYPE} SA with {'auto' if AUTO else 'manual'} scheduling.")
    AUTO = args.auto
    WAIT_TIME = args.waittime

    # Import file of instances (assume they are already generated)
    IDENTIFIER = generate_identifier(m, n, k, NUM_TRIALS, sampling_type=1)
    INSTANCES = None
    MOVES = None
    LOCAL_MOVE_SPACE = np.eye(n, dtype=np.uint8)
    NUM_STEPS = 200_000 # kind of arbitrary

    try:
        INSTANCES = np.load(f"{ROOT}/{DIAG_FILE_PREFIX}_{IDENTIFIER}.npy")
        MOVES = np.load(f"{ROOT}/{MOVES_FILE_PREFIX}_{IDENTIFIER}.npy")
    except Exception as e:
        print(e)
        print("======= Hint =======")
        print(f"No generated instances of m = {m}, n = {n}, k = {k}, t = {NUM_TRIALS} found in {ROOT}.")
        print("You must generate instances before you can run algorithms on them.")
    assert INSTANCES.shape == (NUM_TRIALS, m, n), f"Expected INSTANCES shape to be {(NUM_TRIALS, m, n)} but got {INSTANCES.shape}"
    assert MOVES.shape == (NUM_TRIALS, 3*n, n), f"Expected MOVES shape to be {(NUM_TRIALS, 3*n, n)} but got {MOVES.shape}"

    # Write down the starting and ending temperatures for annealing
    Tmin = 1e-2
    Tmax = find_starting_temp(m, 0.98)
    
    ratios = np.zeros(NUM_TRIALS)

    # Run the annealing on each trial
    for trial in range(NUM_TRIALS):
        print(f"STARTING TRIAL {trial}...")
        CLAUSES = INSTANCES[trial] 
        MOVE_SPACE = None

        if TYPE == 1:
            # Move space consists of local bit flips
            MOVE_SPACE = LOCAL_MOVE_SPACE
        elif TYPE == 2:
            # Move space consists of Pauli flips
            MOVE_SPACE = MOVES[trial]
        else:
            raise ValueError(f"Invalid TYPE = {TYPE}")
        
        initial_state = np.random.randint(0, 2, size=n, dtype=np.uint8) # random start
        target = np.random.randint(0, 2, size=m, dtype=np.uint8) # random target
        NUM_MOVES = MOVE_SPACE.shape[0]

        start = time.perf_counter()
        energy = run_annealer(CLAUSES, initial_state, target, \
                 MOVE_SPACE, NUM_MOVES, Tmin, Tmax, \
                 NUM_STEPS, return_solution=False, auto=AUTO, wait_time=WAIT_TIME)
        end = time.perf_counter()
        minutes, seconds = divmod((end - start) / NUM_TRIALS, 60)
        print(f"[Trial {trial}] SA took {minutes} min {round(seconds, 4)}")
        
        # `energy`` is #UNSAT - #SAT. Proportion satisfied is #SAT / m.
        # We compute this using #UNSAT + #SAT = m, so #SAT = (m - `energy`)/2
        ratios[trial] = (m - energy) / (2 * m)
        print(f"[Trial {trial}] Ratios:\n{ratios}")
    
    np.save(f"SA_TYPE{TYPE}_{IDENTIFIER}.npy", ratios)
    print(f"Average ratio: {np.mean(ratios)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run simulated annealing on XOR-SAT problems"
    )

    parser.add_argument(
        "--type",
        type=int,
        default=1,
        choices=[1, 2],
        help="1 for Z-local flips, 2 for natural local flips"
    )

    parser.add_argument(
        "--size", "-n",
        type=int,
        help="Size of solution"
    )

    parser.add_argument(
        "--numclauses", "-m",
        type=int,
        help="Number of clauses"
    )

    parser.add_argument(
        "--locality", "-k",
        type=int,
        help="Locality of clauses"
    )

    parser.add_argument(
        "--numtrials", "-t",
        type=int,
        help="Number of trials"
    )

    parser.add_argument(
        "--root", "-i",
        type=str,
        help="Import file directory",
        default='.'
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


