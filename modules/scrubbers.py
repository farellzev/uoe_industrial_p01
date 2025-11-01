import numpy as np
import scipy.sparse

def hello_world():
    return "You imported the modules! :)"

def add_scrubbers(coeff_A, dt, dx, k_s, pos, w = 0.1):
    """  
    Reconstructs the coefficient matrix for the forward Euler system
    by adding scrubbers.

    Input:
        coeff_A (matrix): the current coefficient matrix
        dt (float): current time-step
        dx (float): current space-step
        k_s (float): relative scrubber strength
        pos (float): the position of the scrubber
        w (float): length of interval around the scrubber, default 0.1
    """
    C2 = dt * k_s

    # absolute difference from -1
    absolute_diff = np.abs(pos - -1) if pos < 0 else pos + 1
    x_index = int(absolute_diff // dx)
    n_plusminus = int(w // dx)

    # need n to be even for symmetry
    if (n_plusminus % 2 != 0):
        n_plusminus += 1
    
    # add the sink terms around the scrubber
    start_point = x_index - n_plusminus // 2
    diag_index = start_point

    # A cannot be in sparse format if editing; so convert and convert back after
    A_edit = coeff_A.copy().toarray()
    for i in range(n_plusminus):
        A_edit[diag_index][diag_index] += C2
        diag_index += 1

    return scipy.sparse.csr_matrix(A_edit)
