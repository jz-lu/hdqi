"""
`constants.py`

File consisting only of constants and constant generators.
"""

COMMUTE_FILE_PREFIX = "Commuting"
DIAG_FILE_PREFIX = "DiagCommuting"
MOVES_FILE_PREFIX = "PauliMoves"
ITRPLT_FILE_PREFIX = "ItrComPlt"
ITRDATA_FILE_PREFIX = "ItrComData"

def generate_identifier(m, n, k, num_trials, sampling_type=1, m1=None, m2=None):
    if sampling_type == 4:
        return f"TYPE{sampling_type}_m1{m1}m2{m2}n{n}k{k}_t{num_trials}"
    else:
        return f"TYPE{sampling_type}_m{m}n{n}k{k}_t{num_trials}"