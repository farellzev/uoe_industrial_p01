import numpy as np

def find_x_index(x, loc):
    """ 
    Finds the best index for some spatial point in a discretised array of the space [-1, 1].
    Input:
        x (ndarray): an array of space points
        loc (float): a position in space
    Output:
        x_index (int): the best index in `x` that represents `loc`
    """

    # Reused code from earlier. Same logic applies:
    # if location is < 0, then take absolute difference between that and -1. If >= 0, just add 1.
    absolute_diff = np.abs(loc - -1) if loc < 0 else loc + 1
    dx = x[1] - x[0]
    x_index = int(absolute_diff // dx)

    return x_index

# ----

def find_first_t(sol, x_star, C_star):
    """ 
    Finds the first time that a 1D-diffusion system exceeds some critical concentration at some given point.
    Input:
        sol (ndarray): the solution matrix to the 1d-diffusion system
        x_star (float): the critical point x_star in [-1, 1]
        C_star (float): the critical concentration C_star >= 0
    Output:
        t_star (int): the *index* of the time step
        real_time (float): the associated 'real' time
    """

    # Num. of rows/columns in solution matches the space/time steps
    xs = np.linspace(-1, 1, sol.shape[1])
    ts = np.linspace(start_time, T, sol.shape[0])

    # Find all concentrations at your point, i.e. at every time
    x_index = find_x_index(xs, x_star)
    t_star = -1    

    print(f'Using an x-index {x_index}.')
    for t in range(sol.shape[0]):
        sol_at_x = sol[t, x_index]
        if (sol_at_x >= C_star):
            t_star = t
            break;

    # Time never found
    if (t_star == -1):
        print(f'Critical time never found for C* = {C_star} and x* = {x_star}.')
        return None, None

    real_time = ts[t_star]
    return t_star, real_time

# ----

def find_largest_x(sol, t_star, C_star, center = 0):
    """ 
    Given some t^* and C^*, finds the 'largest' x-values (furthest from origin) that satisfy
        C(t^*, x^*) >= C^* 
    
    Input: 
        sol (ndarray): solution matrix for the 1D system
        t_star (int): the particular time index
        C_star (float): the critical concentration
    Output:
        best_xs (ndarray): the array of 'largest' x-values.

    """ 

    xs = np.linspace(-1, 1, sol.shape[1])

    # One liner - all xs that satisfy condition, use enumerate for index
    valid_xs = np.array([x for i, x in enumerate(xs) if sol[t_star][i] >= C_star])
    max_distance = np.max(np.abs(valid_xs - center))

    # NumPy's isclose(a, d) is boolean for points being `d` away from `a`,
    # so this acts as a mask on this condition. (Allows symmetry.)
    best_xs = valid_xs[np.isclose(np.abs(valid_xs), max_distance)]
    return best_xs
