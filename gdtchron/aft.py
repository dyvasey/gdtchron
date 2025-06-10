"""Module for forward modeling of the apatite fission track system.

This code follows the workflow from Ketcham (2005) and supports producing ages 
and distributions of c-axis projected lengths. This code uses the fanning 
curvilinear (FC) model from Ketcham et al. (1999). 
"""

import numpy as np
from scipy.stats import norm

#############
# Constants #
#############

# Constants for the fanning curvilinear model from Ketcham et al. (1999)
KETCHAM_99_FC = {
    "c0": -19.844,
    "c1": 0.38951,
    "c2": -51.253,
    "c3": -7.6423,
    "alpha": -0.12327,
    "beta": -11.988,
    "r_kappa_sum": 1.,
    "l_slope": 0.35,        # Value taken from HeFTy
    "l_intercept": 15.72    # Value taken from HeFTy
}

# Other constants
SECONDS_PER_YEAR = 365.2422 * 24 * 60 * 60 

#######################
# Annealing Functions #
#######################


def g(r, constants=KETCHAM_99_FC):
    """Implement the length transform from Ketcham et al. (1999) (Equation 6).

    Note: g is undefined for reduced lengths of 0 and may throw overflow
    exceptions for reduced lengths that are very close to 0. With the Ketcham et
    al. (1999) constants, this exception can occur for reduced lengths below 
    0.0007.

    Parameters
    ----------
    r : float or NumPy array of floats
        Reduced length (unitless)
    constants : dict
        Dictionary of constants associated with annealing model being used
        (default: KETCHAM_99_FC)
    
    Returns
    -------
    float or NumPy array of floats:
        Result of length transform
    
    """
    alpha = constants["alpha"]
    beta = constants["beta"]

    return (((1 - r ** beta) / beta) ** alpha - 1) / alpha


def f(temperature, time_annealed, constants=KETCHAM_99_FC):
    """Calculate f following Equation 4 from Ketcham et al. (1999).

    Parameters
    ----------
    temperature : float
        Temperature (K)
    time_annealed : float or NumPy array of floats
        How long the crystal annealed at a given temperature (s)
    constants : dict
        Dictionary of constants associated with annealing model being used
        (default: KETCHAM_99_FC)

    Returns
    -------
    float or NumPy array of floats: 
        Value(s) of f for each value of t

    """
    c0 = constants["c0"]
    c1 = constants["c1"]
    c2 = constants["c2"]
    c3 = constants["c3"]

    return c0 + c1 * \
        ((np.log(time_annealed) - c2) / (np.log(1 / temperature) - c3))


def get_equiv_time(r_initial, temperature, constants=KETCHAM_99_FC):
    """Calculate time it takes to reach a reduced length at given temperature.
     
    This function solves Equation 5 from Ketcham (2005) for t (time).

    Parameters
    ----------
    r_initial : NumPy array of floats
        Reduced length (unitless)
    temperature : float
        Temperature (K)
    constants : dict   
        Dictionary of constants associated with annealing model being used
        (default: KETCHAM_99_FC)

    Returns
    -------
    NumPy array of floats:
        Time(s) (in seconds) it would take to anneal to the given reduced 
        length(s) if temperature remained constant

    """
    c0 = constants["c0"]
    c1 = constants["c1"]
    c2 = constants["c2"]
    c3 = constants["c3"]

    exponent = ((g(r_initial, constants) - c0) / c1) * \
        (np.log(1 / temperature) - c3) + c2

    return np.exp(exponent)


def get_next_r(temperature, time_annealed, constants=KETCHAM_99_FC):
    """Calculate reduced lengths after annealing over a given time period.
     
    This function solves Equation 5 from Ketcham (2005) for r (reduced length).

    Parameters
    ----------
    temperature : float
        Temperature (K)
    time_annealed : NumPy array of floats
        How long the crystal annealed at a given temperature (s)
    constants : dict
        Dictionary of constants associated with annealing model being used
        (default: KETCHAM_99_FC)

    Returns
    -------
    NumPy array of floats:
        Mean reduced length(s) of fission tracks that annealed
        at the given temperature for the given period(s) of time

    """
    alpha = constants["alpha"]
    beta = constants["beta"]

    inner_base = alpha * f(temperature, time_annealed, constants) + 1
    
    # Anywhere the inner base is negative has high enough temperatures that it
    # got fully annealed
    # Any number that would cause an integer overflow (i.e, any number below
    # approximately 0.00002 for Ketcham 1999 constants) is also excluded (and
    # effectively corresponds to all tracks being annealed)
    fully_annealed = inner_base < 0.00002

    inner_root = np.zeros(np.size(inner_base))

    # Only take root of inner base if not fully annealed
    # If fully annealed, leave inner root as 0 (and ultimately make r = 0)
    inner_root[~fully_annealed] = inner_base[~fully_annealed] ** (1 / alpha)

    outer_base = (1 - beta * inner_root)

    r = np.zeros(np.size(inner_root))
    r[~fully_annealed] = outer_base[~fully_annealed] ** (1 / beta)
    r[fully_annealed] = 0

    return r


def calc_annealing(r_initial, temperature, start, end, next_nan_index, 
                   constants=KETCHAM_99_FC):
    """Calculate the annealing of fission tracks across a timestep.

    Parameters
    ----------
    r_initial : NumPy array of floats
        Initial mean reduced lengths (unitless) of fission tracks at start 
        of timestep. The value at index 0 corresponds to fission tracks
        produced at the first timestep, the value at index 1 corresponds to
        fission tracks produced at the second timstep, and so on. np.nan
        should be stored at indices corresponding to fission tracks
        produced at the current timestep or at future timesteps.
    temperature : float
        Temperature (K)
    start : float
        Start time of timestep (yrs BP)
    end : float
        End time of timestep (yrs BP)
    next_nan_index : int
        First index of r_initial to have a value of np.nan. This index
        corresponds to fission tracks that will be produced at the current
        timestep.
    constants : dict
        Dictionary of constants associated with annealing model being used
        (default: KETCHAM_99_FC)

    Returns
    -------
    NumPy array of floats: 
        Updated mean reduced length(s) of fission tracks at the end of this 
        timestep. 
        
    """
    # Convert timesteps to seconds
    start *= SECONDS_PER_YEAR
    end *= SECONDS_PER_YEAR

    # Getting equivalent time it would have taken to reach current reduced
    # lengths if the temperature had always been at its current value
    # Note - we can't call get_equiv_time on r_initial = 0 (or very small r),
    # so we need to check for that
    fully_annealed = r_initial < 0.0007
    t_before = np.zeros(np.size(r_initial))
    t_before[~fully_annealed] = \
        get_equiv_time(r_initial=r_initial[~fully_annealed], 
                       temperature=temperature, constants=constants)
    t_before[next_nan_index] = 0  # Accounting for FTs formed at this timestep

    # Adding the duration of the current timestep to get the new cumulative
    # duration of annealing
    cumulative_t = t_before + start - end

    # Calculating next r
    r = np.zeros(np.size(r_initial))
    r[~fully_annealed] = get_next_r(temperature=temperature, 
                                    time_annealed=cumulative_t[~fully_annealed],
                                    constants=constants)
    return r

#############################
# Age Calculation Functions #
#############################


def dpar_conversion(r_mr, dpar, constants=KETCHAM_99_FC):
    """Convert reduced lengths for one apatite to another apatite.

    This function converts from the reduced lengths of a more 
    resistant apatite to reduced lengths of a less resistant apatite
    following Equations 7, 8, and 9a of Ketcham (2005).

    Parameters
    ----------
    r_mr : NumPy array of floats
        Mean reduced lengths (unitless) of fission tracks for the more
        resitant apatite
    dpar : float
        Etch figure length (micrometers) for the apatite
    constants : dict
        Dictionary of constants associated with annealing model being used
        (default: KETCHAM_99_FC)

    Returns
    -------
    NumPy array of floats:
        Mean reduced lengths (unitless) of fission tracks for the more resitant 
        apatite.
    
    """
    # Calculate r_mr0 via Equation 9a
    r_mr0 = 1 - np.exp(0.647 * (dpar - 1.75) - 1.834)

    # Calculate kappa via Equation 8
    kappa = constants["r_kappa_sum"] - r_mr0

    # Calculate r_lr via Equation 7
    base = (r_mr - r_mr0) / (1 - r_mr0)

    # Anywhere r_mr < r_mr0 should be fully annealed and have r = 0
    base[r_mr < r_mr0] = 0

    # Returning r_lr
    return base ** kappa


def r_to_rho(r):
    """Convert final reduced lengths to fission track densities.

    This function uses Equation 13 from Ketcham (2005).

    Parameters
    ----------
    r : NumPy array of floats
        Mean reduced lengths (unitless) of fission tracks for the specific
        apatite for each point on the apatite's time-temperature path. These 
        values should already be converted to account for dpar variations.

    Returns
    -------
    NumPy array of floats:
        Normalized fission track densities corresponding to each interval
        on the time-temperature path. Any reduced lengths below 0.13 can't be
        observed, so for intervals with a mean reduced length below 0.13,
        their corresponding fission track densities are set to 0.
    
    """
    # Adding in r = 1 for FTs formed in the present
    r = np.append(r, np.array(1))

    # Using the r's at the midpoint between timesteps
    # Done following Ketcham 2000 to prevent bias toward younger ages
    midpoints = (r[1:] + r[:-1]) / 2

    # Calculating densities following Equation 13 of Ketcham (2005)
    # r >= 0.765 case (Equation 13a)
    rho = 1.600 * midpoints - 0.600

    # r < 0.765 case (Equation 13b)
    low_indices = np.where(midpoints < 0.765)       
    rho[low_indices] = (9.205 * (midpoints[low_indices] ** 2) - 
                        9.157 * midpoints[low_indices] + 2.269)
    
    # r below 0.13 can't be observed and are effectively 0
    zero_indices = np.where(midpoints < 0.13)
    rho[zero_indices] = 0
    
    return rho


def calc_aft_age(r_final, tsteps, rho_st=0.893):
    """Calculate AFT age based on present-day reduced lengths.

    This function uses Equations 13-14 from Ketcham (2005). Note that, if
    rho_st < 1, then an un-annealed apatite will have an age older than
    the earliest timestep. This behavior is reproduced in HeFTy when
    using both the Ketcham et al. (1999) and Ketcham et al. (2007)
    parameters.

    Parameters
    ----------
    r_final : NumPy array of floats
        Mean reduced lengths (unitless) of fission tracks produced at each 
        point on the apatite's time-temperature path. These values should 
        already be converted to account for dpar variations.

    tsteps : NumPy array of floats
        Array of times (yrs BP) that each set of fission tracks was produced
        at. This array should be in descending (i.e., chronological) order. The
        final time in this array should be 0. and should not have a
        corresponding reduced length in r_final.

    rho_st : float
        Fission track density reduction in the age standard
        (default: 0.893, the value for the Durango apatite)

    Returns
    -------
    float:
        AFT age (Myrs BP)
    
    """
    # Calculate FT densities via Equation 13
    rho = r_to_rho(r_final)

    # Calculate durations of each timestep
    delta_t = tsteps[:-1] - tsteps[1:]

    # Calculate ages from densities via Equation 14 
    # Also ensure age can't be negative
    # Division by 1e6 converts from yrs to Myrs
    return max(np.sum(rho * delta_t) / rho_st, 0) / 1e6

################################
# Length Calculation Functions #
################################


def l_conversion(r, dpar, constants=KETCHAM_99_FC):
    """Convert reduced length r to mean c-axis projected length l_c.

    Parameters
    ----------
    r : float (or NumPy array of floats)
            reduced length (unitless)
    dpar : float
        Etch figure length (micrometers) for the apatite
    constants : dict
        Dictionary of constants associated with annealing model being used
        (default: KETCHAM_99_FC)

    Returns
    -------
        float or NumPy array of floats: 
        Mean c-axis projected length(s) (micrometers) 
        corresponding to each r value

    """
    # Calculate initial c-axis projected fission track length
    l0 = constants["l_slope"] * dpar + constants["l_intercept"]

    # Calculate current c-axis projected length from reduced and initial lengths
    l_c = r * l0
    return l_c


def get_l_stdev(l_c):
    """Get standard deviation of length distribution based on its mean length.
    
    This function follows equation from Figure 6b in Ketcham (2005).
    Assumes lengths are normally distributed around mean.

    Parameters
    ----------
    l_c : float (or NumPy array of floats)
        Mean c-axis projected length(s) (micrometers)

    Returns
    -------
    float (or NumPy array of floats): 
        Standard deviation(s) (micrometers) corresponding to each l value

    """
    return 0.010 * l_c * l_c - 0.2827 * l_c + 2.501


def calc_weights(r, tsteps, lamb=1.551e-4):
    """Calculate weights to apply to length distributions when summing together.

    This function uses Equations 12 and 13 in Ketcham (2005) to determine the 
    weights to apply to the length distributions associated with each timestep. 
    These weights are used when summing the length distributions together to
    create a mixed distribution.

    Parameters
    ----------
    r : NumPy array of floats with length n
        Mean reduced lengths (unitless) of fission tracks produced at each 
        point on the apatite's time-temperature path. These values should 
        already be converted to account for dpar variations.
    tsteps : NumPy array of floats with length n + 1, where n > 1
        Array of times (yrs BP) that each set of fission tracks was produced
        at. This array should be in descending (i.e., chronological) order. The
        first time is the start of the first timestep, last time is end of last 
        timestep.
    lamb : float
        Total U238 decay constant (lambda) (Myr^-1) (default: 1.551e-4)

    Returns
    -------
    NumPy array of floats with length n:
        Weights for each timestep
    
    """
    starts = tsteps[:-1] / 1e6  # t2 (Myr)
    ends = tsteps[1:] / 1e6     # t1 (Myr)
    w1 = (np.exp(lamb * starts) - np.exp(lamb * ends)) / lamb
    w2 = r_to_rho(r)
    w = w1 * w2
    return w


def combine_dists(means, stdevs, w, make_graph=False, x_num=100):
    """Calculate weighted sum of normal distributions.
     
    This funcion calculates a weighted sum of normal distributions based on 
    their means and distributions. Disregards means of zero and negative values.
    Used to create mixed distribution of lengths

    Parameters
    ----------
    means : NumPy array of floats
        Means of each distribution
    stdevs : NumPy array of floats
        Standard deviations of each distribution
    w : NumPy array of floats
        Weights to apply to each distribution when combining
    make_graph : bool
        Boolean indicating whether or not to return x and frequency
        series for plotting (default: False)
    x_num : int
        Nonnegative integer specifying how many x values to use for the 
        frequency series and internal calculations (default: 100). A value of 
        at least 50 is recommended for calculating the standard deviation and
        mean of the mixed distribution; larger values produce smoother curves
        when plotting the frequency series. 

    Returns
    -------
    mean : float
        Mean of mixed distribution
    stdev : float
        Standard deviation of mixed distribution
    x (optional) : NumPy array of floats
        Array of x values. Only returned if make_graph = True
    freqs (optional) : NumPy array of floats
        Array of frequencies at which each x value is observed. 
        Only returned if make_graph = True
    """
    # Removing means of zero and associated weights/stdevs
    valid_indices = np.where(means > 0)
    means = means[valid_indices]
    stdevs = stdevs[valid_indices]
    w = w[valid_indices]

    # Getting x values to calculate 
    # Note: Expanding the bounds here has a negligible effect on the mean
    #       and standard deviation of the resulting length distribution for
    #       any of the tests from Ketcham (2005)
    x_lower = max(np.min(means) - np.max(stdevs) * 2, 0)
    x_upper = np.max(means) + np.max(stdevs) * 2
    x = np.linspace(x_lower, x_upper, x_num)
    
    # Transposing x so norm.pdf broadcasts correctly
    x_grid = x[:, np.newaxis]

    # Getting distributions
    dists = norm.pdf(x_grid, means, stdevs)

    # Summing by weights
    mixed_dist = np.dot(dists, w)

    # Normalizing
    mixed_dist /= np.sum(mixed_dist)

    # Finding mean
    mean = np.dot(x, mixed_dist)

    # Finding stdev
    var = np.sum(mixed_dist * (x - mean) ** 2)
    stdev = var ** 0.5

    # Returning relevant values
    if make_graph:
        return mean, stdev, x, mixed_dist
    else:
        return mean, stdev
    

def calc_l_dist(r, dpar, tsteps, constants=KETCHAM_99_FC, make_graph=False, 
                l_num=100):
    """Calculate length distribution from reduced lengths.
    
    This function takes the reduced lengths of fission
    tracks produced at each timestep, calculates the mean and standard
    deviation of the length distributions for each timestep, and combines
    these distributions to find the overall fission track length
    distribution

    Parameters
    ----------
    r : NumPy array of floats with length n
            Reduced lengths of fission tracks produced at each timestep 
            (unitless)
    dpar : float
        Etch figure length (micrometers)
    tsteps : NumPy array of floats with length n + 1
        Array of times (yrs BP) that each set of fission tracks was produced
        at. This array should be in descending (i.e., chronological) order. The
        first time is the start of the first timestep, last time is end of last 
        timestep.
    constants : dict
        Dictionary of constants associated with annealing model being used
        (default: KETCHAM_99_FC)
    make_graph : bool
        Boolean indicating whether or not to return x and frequency
        series for plotting (default: False)
    l_num : int
        Nonnegative integer specifying how many lengths to use for the 
        frequency series and internal calculations (default: 100). A value of 
        at least 50 is recommended for calculating the standard deviation and
        mean of the mixed distribution; larger values produce smoother curves
        when plotting the frequency series. 

    Returns
    -------
    results : tuple containing the following values:
        mean : float
            Mean of mixed distribution
        stdev : float
            Standard deviation of mixed distribution
        l_c (optional) : NumPy array of floats
            Array of length values. Only returned if make_graph = True
        freqs (optional) : NumPy array of floats
            Array of frequencies at which each x value is observed. 
            Only returned if make_graph = True
        
    """
    # Get descriptors of each distribution 
    l_c = l_conversion(r=r, dpar=dpar, constants=constants)
    stdevs = get_l_stdev(l_c=l_c)
    w = calc_weights(r=r, tsteps=tsteps)

    # Combine distributions
    results = combine_dists(means=l_c, 
                            stdevs=stdevs, 
                            w=w, 
                            make_graph=make_graph,
                            x_num=l_num)
    return results


def forward_model_aft(temps, tsteps, dpar, constants=KETCHAM_99_FC, 
                      get_lengths=False, make_graph=False, l_num=100):
    """Conduct forward modeling of the apatite fission track system.

    This function runs the forward model of the AFT system from Ketcham (2005)
    based on given time-temperature series. Calculates thermochronological age
    and (optionally) mean fission track length

    Parameters
    ----------
    temps : NumPy array of floats with length n, where n > 1
        Temperatures (K) at each time in tsteps
    tsteps : NumPy array of floats with length n, where n > 1
        Array of times (yrs BP) in chronological (descending) order. First 
        time is start of first timestep, last time is end of last timestep. 
        Each pair of adjacent times composes a timestep
    dpar : float
        Etch figure length (micrometers)
    constants : dict
        Dictionary of constants associated with annealing model being used
        (default: KETCHAM_99_FC)
    get_lengths: bool
        Boolean indicating whether or not to also calculate distribution of
        c-axis projected fission track lengths within an apatite that
        experienced this time-temperature history
    make_graph : bool
        Boolean indicating whether or not to return length and frequency
        series for plotting length distribution (default: False). Only relevant
        if get_lengths is True.
    l_num : int
        Nonnegative integer specifying how many lengths to use for the 
        frequency series and internal calculations (default: 100). A value of 
        at least 50 is recommended for calculating the standard deviation and
        mean of the mixed distribution; larger values produce smoother curves
        when plotting the frequency series. Only relevant if get_lengths is True

    Returns
    -------
    age : float
        Model age (Myrs BP) of apatite that experienced the given 
        time-temperature history. If get_lengths is False, only age is
        returned.
    length_results (optional) : tuple
        Tuple describing the length distribution produced by calc_l_dist.
        Only included if get_lengths is True.
        Contains (in order) the mean (float) and standard deviation 
        (float) of the distribution. If make_graph is also true, includes
        lengths (NumPy array of floats) and frequencies at which
        each length occurs in the distribution (NumPy array of floats)
    """
    # Getting average temperatures for each interval
    avg_temps = np.convolve(temps, np.ones(2), 'valid') / 2.
    # Setting up r arrray
    r = np.full(shape=np.size(avg_temps), fill_value=np.nan)
    
    # Calculate annealing for each timestep
    for i in range(0, len(avg_temps)):
        r = calc_annealing(r_initial=r, 
                           constants=constants, 
                           temperature=avg_temps[i], 
                           start=tsteps[i],
                           end=tsteps[i + 1], 
                           next_nan_index=i)
        
    # Adjusting based on Dpar
    r = dpar_conversion(r_mr=r, dpar=dpar, constants=constants)

    # Calculating final age/length distribution
    if get_lengths:
        return (calc_aft_age(r_final=r, tsteps=tsteps), 
                calc_l_dist(r=r, 
                            dpar=dpar, 
                            tsteps=tsteps,
                            constants=constants,
                            make_graph=make_graph,
                            l_num=l_num))
    else:
        return calc_aft_age(r_final=r, tsteps=tsteps)