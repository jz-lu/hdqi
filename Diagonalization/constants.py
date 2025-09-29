"""
`constants.py`

File consisting only of constants and constant generators.
"""

CLASSICAL_FILE_PREFIX = "Classical"

COMMUTE_FILE_PREFIX = "Commuting"
DIAG_FILE_PREFIX = "DiagCommuting"
CLIFF_FILE_PREFIX = "Cliff"
MOVES_FILE_PREFIX = "PauliMoves"
ITRPLT_FILE_PREFIX = "ItrComPlt"
ITRDATA_FILE_PREFIX = "ItrComData"

def generate_identifier(m, n, k, num_trials, sampling_type=1, m1=None, m2=None, key=None):
    KEY_STR = "" if key is None else f"_KEY{key}"
    if sampling_type == 4:
        return f"TYPE{sampling_type}_m1{m1}m2{m2}n{n}k{k}_t{num_trials}{KEY_STR}"
    else:
        return f"TYPE{sampling_type}_m{m}n{n}k{k}_t{num_trials}{KEY_STR}"
    
def generate_classical_identifier(m, n, k, num_trials):
    return f"m{m}n{n}k{k}_t{num_trials}"