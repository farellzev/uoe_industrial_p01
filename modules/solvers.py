import numpy as np
from scipy.sparse import diags_array
from scipy.sparse.linalg import spsolve

markers = ['x', 's', 'd', '+', 'o', '.']
colours = ['xkcd:salmon', 'xkcd:golden rod', 'xkcd:teal', 'xkcd:purple', 'xkcd:slate',' xkcd:silver']    

def analytical_sol(x, t, N):   
    '''
    Gives the exact solution as outlined above.
    Input:
        x (list): the entire space of the system
        t (float): the particular time to evaluate the solution at
        N (int): number of terms to approximate the infinite series by
        
    '''
    
    exact_sol = np.zeros_like(x, dtype = float)
    exact_sol += 1/2

    for n in range(1, N):
        exact_sol += np.cos(n*np.pi*x) * np.exp(-1*((n*np.pi)**2)*t)
    
    return exact_sol

# -----

def construct_original_A(M, C):
    # Constructs the coefficient matrix A given the parameters.
    main_diag = np.array(([1+2*C] * (M+1)))
    super_diag = np.array([-2*C] + ([-C] * (M-1)))
    sub_diag = np.array(([-C]*(M-1)) + [-2*C])

    A = diags_array([main_diag, super_diag, sub_diag], offsets = [0, 1, -1], format = 'csr')
    return A

# ----

def solve_diffusion(N, M, coeff_A, start_time = 0.01, analytic_truncate = int(1e4)):
    """ 
    Solves the diffusion equation using the implicit (backwards) Euler method
        AU^{n+1} = U^{n} .  

    Recall again that M is square with dimension M+1,
    and U has dimension (N+1)x(M+1).

    Input:
        N (int): number of timesteps 
        M (int): number of space steps
        coeff_A (sparse matrix): the coefficient matrix, sparse.

    Output:
        U_sol (np matrix): the solution matrix
    """
    if (coeff_A.shape[0] != M + 1):
        raise ValueError(f'Need dimensions of coefficient matrix and space steps to match. Have {coeff_A.shape[0]} and {M+1}.')
    
    x = np.linspace(-1, 1, M + 1)
    
    # add some validation that A is a sparse matrix: if (A. != "sparse")
    
    # initialise solution matrix and set initial condition
    U_sol = np.zeros((N+1, M+1))
    U_sol[0] = analytical_sol(x, start_time, analytic_truncate)

    # spsolve needs A to be a sparse matrix (SciPy)
    for n in range(0, N):
        pseudo_inv = spsolve(coeff_A, U_sol[n])
        U_sol[n + 1] = pseudo_inv

    return U_sol    

# -----

def plot_numeric(x, sol, ax, times = [0, 2, 10], title = ""):
    """ 
    Simple plotter function to plot the numerical solution to the diffusion equation.
    Plots every two steps for efficiency - minimal information lost.

    Input:
        x (np array): the array of x values to plot over
        sol (np array): the solution matrix,
        ax (plt axis): the pyplot axis to plot onto
        times (np array): optional times (in timesteps) to plot at, default are 0, 2, 10 timesteps
        title (str): optional title for the axis

    Output:
        ax (plt axis): the axis object, stops additional plotting.
    """
    
    for i, time in enumerate(times):
            colour = "black" if i > len(colours) else colours[i]
            ax.scatter(x[::2], sol[time][::2], label = f'{time} steps, numeric', color = colour, marker = '.')

    ax.set_xlabel("Distance")
    ax.set_ylabel("Gas Concentration")

    if (title):
         ax.set_title(title)

    ax.legend(loc = "best")
    return ax

def plot_analytic(x, t, func, ax, times = [0, 2, 10], title = "", analytic_truncate = int(1e4)):
    """ 
    Simple plotter function to plot the analytic solution to the diffusion equation.
    Plots every two steps for efficiency.

    Input:
        x (np array): array of x values to plot over
        t (np array): array of t values to plot over
        func (function): function for the analytic solution
        ax (plt axis): the pyplot axis to plot onto
        times (np array): optional times (IN TIMESTEPS) to plot at, default 0, 2, 10
        title (str): optional plot tile

    Output:
        ax (plt axis): the axis object, stops additional plotting.
    """

    for i, time in enumerate(times):
            # need actual time not timestep
            real_time = t[time]
            colour = "black" if i > len(colours) else colours[i]

            ax.plot(x[::2], func(x[::2], real_time, analytic_truncate), label = f'{time} steps, analytic', color = colour)

    ax.set_xlabel("Distance")
    ax.set_ylabel("Gas Concentration")

    if (title):
         ax.set_title(title)

    ax.legend(loc = "best")
    return ax
    