"""
`SA.py`

Code file to perform simple simulated annealing (SA) experiments on 
max-k-XOR-SAT problems.
"""
from simanneal import Annealer
from random import randint

class XORMaxSatAnnealer(Annealer):
    def __init__(self, state, clauses):
        super().__init__(state)
        self.clauses = clauses

    def move(self):
        i = randint(0, len(self.state)-1)
        self.state[i] ^= 1

    def energy(self):
        unsat = 0
        for vars_, parity in self.clauses:
            s = sum(self.state[i] for i in vars_) % 2
            if s != parity:
                unsat += 1
        return unsat

# Build your problem
n = 1000
clauses = [
    ([0, 5, 10], 1),
    ([3, 4, 7], 0),
    ([2, 8, 9], 1),
    ([1, 6, 11], 0),
]

# CORRECT: no `self` here
init = [randint(0, 1) for _ in range(n)]

sa = XORMaxSatAnnealer(init, clauses)
sa.steps = 200_000
sa.Tmin, sa.Tmax = 1e-3, 1e2
best_state, best_energy = sa.anneal()
print(f"Best state = {best_state}")
print(f"Best energy = {best_energy}")


