# Construction

We consider two types of Pauli Hamiltonian (i.e. sum of signed Pauli strings) instances. The first is commuting Hamiltonians, and the second is "almost commuting" Hamiltoninans, for an appropriate definition of almost commuting. These are organized in the files `commuting.py` and `almost.py`.

## How to sample some instances
If you wish to sample some instances, the basic way to do it is to specify some parameters in either `commuting.py` or `almost.py` and then tell the file to save your data to some directory. The data will be saved as a `numpy` array whose dimension is described below.

The parameters of choice are as follows:
* `n`: number of qubits
* `m`: number of Pauli terms
* `k`: locality of each term
* `trials`: number of instances you would like to generate

In addition, there may be different sampling algorithms you wish to implement. You have the option to choose between them when you call the function. The specifics of this, alongside examples, are given below for each file. 

### Commuting
There are 4 types of sampling algorithms you might want. All of them sample some kind of local commuting Hamiltonian via rejection sampling, i.e. keep drawing the next Pauli until it commutes with the previous Paulis, then add it to the list.

1. Type 1: Random k-local Pauli. Choose k random locations to place a non-identity Pauli, then choose a random Pauli (X, Y, Z) with equal probability for each. This is analogous to depolarizing noise. *This is the most physical and natural model, so the default type is set to this one.*
2. Type 2: Random k-sparse symplectic vectors. Choose a uniformly random weight-k binary vector of length 2n. There is not really a natural/physical corresponding model.
3. Type 3: Random k-local X's and independent random k-local Z's. Choose k random places to put X's, and k random places to put Z's. Places that have X and Z instead get a Y. This corresponds to a CSS-type XZ-decoupled error model.
4. Type 4: X first, then Z. Choose `m1` k-local Pauli X strings randomly. Then choose `m2` k-local Pauli Z strings randomly, subject to them commuting with the X strings. This also lacks a natural physical interpretation. *Warning*: do not use this type with `m1 >> n`. If you do this, then with high probability the X's will form an independent basis of all X-type strings, and there will not exist any Z-type string which commutes with them. 

In general, use Type 1 or Type 2 unless you feel a strong reason to work with a strange model. 
Some careful mathematical analysis shows that rejection sampling terminates quickly for Type 1 and Type 2, with high probability. 
The number of trials needed to find the next Pauli string scales as `O(exp(k^2 * m / n))` for Type 1 and the same, but with `k -> 2k` on Type 2. 
Therefore, if `k` is very small and `m` is a small linear function of `n`, then the scaling is a large but manageable constant. 
In general, we recommend setting `2 <= k <= 6` for Type 1 and `2 <= k <= 4` for Type 2 (since the actual weight is doubled) if you want to finish very quickly. 
If you use larger compute resources, you can probably get `k` to be a little bit larger, but the scaling is very bad in terms of `k`.
Wisdom from the Gilbert-Varshamov bound suggests that to have linear distance while maintaining linear rate, one should set the rate to be a small constant certainly no larger than 1/2.
So we recommend setting `m = c*n` where `c` is a constant satisfying `1 < c < 2`. 
(Note that the rate is given by (c-1)/c.)

**Remark**: Sometimes we say "number of trials" in reference to the number of rejections before we successfully get the next commuting Pauli. Other times, we mean the number of Hamiltonians we want to sample. 
We rely on context to tell the difference, e.g. the latter is a parameter that the user sets.

By default, the code will save a plot of how many tries it took to get each new column, averaged over the trials. If you wish not to have this plot, the command-line flag `--noplot` will do the trick.

To save the instances do a directory, include the command-line flag `--save <DIRECTORY>`. By default, the instances will be saved to the current working directory. The saved file is of the form `Commuting_TYPE<TYPE>_m<m>n<n>k<k>_t<num_trials>.npz`, which you can load using `np.load()`. The array has shape `(num_trials, 2*n, m)` where for each trial we save a `2n x m` parity check matrix constructed. The columns are symplectic representations of the Paulis.

Example: `python commuting.py --type 1 -m 150 -n 100 -k 4 --trials 75 --save .`

### Almost Commuting

**TBD**


## Scaling tests
In the commuting case, we also give the option to run numerics for how the number of rejections scales with `n` for fixed `c`, where `m = c*n` and fixed `k`.
This is because we sample our Hamiltonians by a rejection sampling procedure that has no fixed termination time.
This is contained in `scale_test.py`. To run it, specify `c` instead of `m` and roughly everything else will be the same as above, except how you specify `n`. 
By default, the code will run tests for `n` from 50 to 500, going in increments of 50, i.e. 50, 100, 150, ..., 500.
You may also specify the starting `n`, the ending `n`, and the number of points you want. The code will use `np.linspace()` to automatically interpolate the points. Also, if `c*n` is not an integer, we set `m = int(c*n)`.

Example: `python scale_test.py --type 1 -c 1.5 --nstart 50 --nend 500 --npts 10 -k 4 --trials 75 --save .` This would run a scaling test from `n = 50` to `n = 500`, with 10 points in between (so here, incrementing by 50).

The code will periodically let you know which trial it is on so you are aware of progress (this test could be quite slow depending on your numbers).  

The scaling tests do **not** by default save the instances. (The `--save` flag is for saving the scaling plots.) 
If you wish to save the instances (not recommended unless for debugging), use the `--keepinstances` flag in the command line.
