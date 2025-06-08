"""Test file for aft.py."""

import numpy as np
import pytest

from gdtchron import aft

# Dummy constants used for some tests
TEST_CONSTS = {
    "c0": -1.,
    "c1": 1. / 3.,
    "c2": -1.,
    "c3": -2.,
    "alpha": -2.,
    "beta": -1.,
    "r_kappa_sum": 1.,
    "l_slope": 0.5,
    "l_intercept": 15.
}

# This Dpar yields r_mr0 = 0.5
TEST_DPAR = (1.834 - np.log(2)) / 0.647 + 1.75


def test_g():
    """Unit tests for g (length transform)."""
    # Confirm NaNs stay NaN
    assert np.isnan(aft.g(np.nan))

    # Confirm calculations performed properly for np array
    # Using Ketcham et al. (1999) fanning curvilinear model:
    assert aft.g(np.array([np.nan, 0.9, 0.1]))[1:] == \
            pytest.approx(np.array([-1.711884, 7.745533]))
    assert np.isnan(aft.g(np.array([np.nan, 0.9, 0.1]))[0])

    # Confirm calculations work for dummy constants
    assert aft.g(0.5, TEST_CONSTS) == 0.


def test_f():
    """Unit tests for f (right-hand side of main annealing equation)."""
    # Confirm NaNs stay NaN
    assert np.isnan(aft.f(temperature=450., time_annealed=np.nan))

    # Confirm calculations performed properly for np array
    # Using Ketcham et al. (1999) fanning curvilinear model:
    assert aft.f(temperature=450, 
                 time_annealed=np.array([np.nan, 1e10, 1e11]))[1:] == \
                    pytest.approx(np.array([-0.971615, -0.386586]))
    assert np.isnan(aft.f(temperature=450, 
                          time_annealed=np.array([np.nan, 0.9, 0.1]))[0])

    # Confirm calculations work for dummy constants
    assert aft.f(temperature=np.e, 
                 time_annealed=np.e ** 2, 
                 constants=TEST_CONSTS) == 0.


def test_get_equiv_time():
    """Unit tests for get_equiv_time."""
    # Confirm calculations performed properly for np array with NaNs
    # Using Ketcham et al. (1999) fanning curvilinear model:
    equiv_times = aft.get_equiv_time(np.array([np.nan, 0.9, 0.1]), 
                                     temperature=450.)[1:]
    assert equiv_times == pytest.approx(np.array([5.428073e8, 7.950111e24]))
    assert np.isnan(aft.get_equiv_time(np.array([np.nan, 0.9, 0.1]), 
                                       temperature=450.)[0])
    
    # Confirming that Equation 5 of Ketcham (2005) still holds
    assert aft.g(np.array([0.9, 0.1])) == \
        pytest.approx(aft.f(temperature=450., time_annealed=equiv_times))

    # Confirm calculations work for dummy constants
    assert aft.get_equiv_time(np.array([0.5]), 
                              temperature=np.e, 
                              constants=TEST_CONSTS) \
        == pytest.approx(np.array([np.e ** 2]))


def test_get_next_r():
    """Unit tests for get_next_r."""
    # Confirm calculations performed properly for np array with NaNs
    # Using Ketcham et al. (1999) fanning curvilinear model:
    rs = aft.get_next_r(450., np.array([np.nan, 1e10, 1e11]))[1:]
    assert rs == pytest.approx(np.array([0.863753, 0.830875]))
    assert np.isnan(aft.get_next_r(450., np.array([np.nan, 1e10, 1e11]))[0])
    
    # Confirming that Equation 5 of Ketcham (2005) still holds
    assert aft.g(rs) == pytest.approx(aft.f(temperature=450., 
                                            time_annealed=np.array([1e10, 
                                                                    1e11])))

    # Confirm calculations work for dummy constants
    assert aft.get_next_r(temperature=np.e, 
                          time_annealed=np.e ** 2, 
                          constants=TEST_CONSTS) == 0.5
    
    # Confirm code works for fully annealed tracks
    assert aft.get_next_r(temperature=550., 
                          time_annealed=np.array([1.87165e19])) == \
                            np.array([0.])
    assert aft.get_next_r(temperature=550., time_annealed=np.array([1e21])) == \
        np.array([0.])
    

def test_calc_annealing():
    """Unit tests for calc_annealing."""
    # Using Ketcham et al. (1999) fanning curvilinear model

    # Confirm calculations performed properly for initial timestep
    assert aft.calc_annealing(r_initial=np.array([np.nan, np.nan]),
                              temperature=450.,
                              start=1e10 / aft.SECONDS_PER_YEAR,
                              end=0.,
                              next_nan_index=0)[0] == pytest.approx(0.863753)
    assert np.isnan(aft.calc_annealing(r_initial=np.array([np.nan, np.nan]),
                                       temperature=450.,
                                       start=1e10 / aft.SECONDS_PER_YEAR,
                                       end=0.,
                                       next_nan_index=0)[1])
    
    # Confirm calculations performed properly for intermediate timestep
    r1 = aft.calc_annealing(r_initial=np.array([np.nan, np.nan, np.nan]),
                            temperature=450.,
                            start=1e11 / aft.SECONDS_PER_YEAR,
                            end=1e10 / aft.SECONDS_PER_YEAR,
                            next_nan_index=0)
    r2 = aft.calc_annealing(r_initial=r1,
                            temperature=450.,
                            start=1e10 / aft.SECONDS_PER_YEAR,
                            end=0. / aft.SECONDS_PER_YEAR,
                            next_nan_index=1)
    assert r2[:2] == pytest.approx(np.array([0.830875, 0.863753]))
    assert np.isnan(r2[2])
    
    # Confirm calculations performed properly for final timestep
    # Also confirm that full annealing of some (but not all tracks) works
    r1 = aft.calc_annealing(r_initial=np.array([np.nan, np.nan]),
                            temperature=650.,
                            start=1.8e19 / aft.SECONDS_PER_YEAR,
                            end=1e19 / aft.SECONDS_PER_YEAR,
                            next_nan_index=0)
    assert aft.calc_annealing(r_initial=r1,
                              temperature=550.,
                              start=1e19 / aft.SECONDS_PER_YEAR,
                              end=0.,
                              next_nan_index=1) \
                                == pytest.approx(np.array([0., 0.0625308]))
    
    # Confirm calculations work when all tracks fully anneal at end
    assert aft.calc_annealing(r_initial=r1,
                              temperature=950.,
                              start=1e19 / aft.SECONDS_PER_YEAR,
                              end=0.,
                              next_nan_index=1) \
                                == pytest.approx(np.array([0., 0.]))

    # Confirm calculations work for dummy constants
    assert aft.calc_annealing(r_initial=np.array([np.nan]),
                              temperature=np.e,
                              start=(np.e ** 2) / aft.SECONDS_PER_YEAR,
                              end=0.,
                              next_nan_index=0,
                              constants=TEST_CONSTS)[0] == pytest.approx(0.5)
    

def test_dpar_conversion():
    """Unit tests for dpar_conversion."""
    # Test that dpar conversion works for the following cases:
    #   Normal conversion (r_mr = 0.625)
    #   When r_mr = r_mr0 (r_mr = 0.5)
    #   When r_mr < r_mr0 (r_mr = 0.2)
    #   When r_mr = 0
    assert aft.dpar_conversion(r_mr=np.array([0.625, 0.5, 0.2, 0.]),
                               dpar=TEST_DPAR) == \
        pytest.approx(np.array([0.5, 0., 0., 0.]), abs=1e-6)
    

def test_r_to_rho():
    """Unit tests for r_to_rho."""
    # Test that r_to_rho works for the following cases:
    #   r = 0
    #   0 < r < 0.13
    #   r < 0.765
    #   r = 0.765
    #   r > 0.765
    #   r = 1
    assert aft.r_to_rho(np.array([0., 0., 0.1, 0.73, 0.8, 1.])) == \
        pytest.approx(np.array([0., 0., 0.054176125, 0.624, 0.84, 1.]))
    

def test_calc_aft_age():
    """Unit tests for calc_aft_age."""
    # Test when no annealing occurs
    assert aft.calc_aft_age(r_final=np.array([1., 1.]),
                            tsteps=np.array([0.893, 0.4, 0.])) == 1e-6
    
    # Test when len(r_final) == 1
    assert aft.calc_aft_age(r_final=np.array([0.8]),
                            tsteps=np.array([1., 0.]),
                            rho_st=0.84) == pytest.approx(1e-6)
    
    # Test when len(r_final) != 1, some annealing occurs, and age != 0
    assert aft.calc_aft_age(r_final=np.array([0.8, 0.9]),
                            tsteps=np.array([30., 20., 0.]),
                            rho_st=1.) == pytest.approx(26e-6)
    
    # Test when r = 0
    assert aft.calc_aft_age(r_final=np.array([0.]),
                            tsteps=np.array([30., 0.])) == pytest.approx(0.)
    

def test_l_conversion():
    """Unit tests for l_conversion."""
    # Test default constants
    assert aft.l_conversion(r=np.array([1.0, 0.5, 0.]),
                            dpar=2.) == \
                                pytest.approx(np.array([16.42, 8.21, 0.]))
    
    # Test dummy constants and non-array r
    assert aft.l_conversion(r=0.5,
                            dpar=2.,
                            constants=TEST_CONSTS) == 8.

    
def test_get_l_stdev():
    """Unit tests for get_l_stdev."""
    # Test np array
    assert aft.get_l_stdev(np.array([1.0, 0.5, 0.])) == \
                            pytest.approx(np.array([2.2283, 2.36215, 2.501]))
    
    # Test just a float
    assert aft.get_l_stdev(0.9) == \
                                pytest.approx(2.25467)
    

def test_calc_weights():
    """Unit tests for calc_weights."""
    # Test smallest possible arrays
    assert aft.calc_weights(r=np.array([0.5]), 
                            tsteps=np.array([1e6 * np.log(5), 0]), 
                            lamb=1) == pytest.approx(np.array([2.31625]))
    # Test longer arrays (including with un-annealed tracks)
    assert aft.calc_weights(r=np.array([0.6, 1.]),
                            tsteps=1e10 * np.array([np.log(5), np.log(2), 0]), 
                            lamb=1e-4) == pytest.approx(np.array([2.04e4, 1e4]))


def test_combine_dists():
    """Unit tests for combine_dists."""
    # Test array of length 1
    # Note: "Combined" dist is the same but with everything beyond 2 standard
    # deviations cut off (hence the lower standard deviation)
    mean, sd, x, freqs = aft.combine_dists(means=np.array([1.]),
                                           stdevs=np.array([0.5]),
                                           w=np.array([50.]),
                                           make_graph=True)
    assert mean == pytest.approx(1)
    assert sd == pytest.approx(0.441888)
    assert x[0] == pytest.approx(0.)
    assert x[99] == pytest.approx(2.)
    assert np.argmax(freqs) == 49 or np.argmax(freqs) == 50
    assert np.argmin(freqs) == 0 or np.argmin(freqs) == 99

    # Test that negative/zero mean distributions are excluded
    mean, sd = aft.combine_dists(means=np.array([1., 0., -1.]),
                                 stdevs=np.array([0.5, 0.5, 0.5]),
                                 w=np.array([5., 70., 50.]),
                                 make_graph=False)
    assert mean == pytest.approx(1)
    assert sd == pytest.approx(0.441888)

    # Test making a proper mixed distribution
    mean, sd, x, freqs = aft.combine_dists(means=np.array([1., 5.]),
                                           stdevs=np.array([0.5, 2.]),
                                           w=np.array([3., 1.]),
                                           make_graph=True,
                                           x_num=91)
    assert mean == pytest.approx(2., rel=1e-2)
    assert sd < 2.046     # What actual stdev would be if we didn't cut of dists
    assert sd == pytest.approx(2.046, rel=1e-1)
    assert x[0] == pytest.approx(0.)
    assert x[90] == pytest.approx(9.)
    assert np.argmax(freqs) == 10
    assert np.argmin(freqs) == 90

    # Note: Cutting off the distributions has much less of an effect on the
    # final mean/stdev when combining larger numbers of distributions (which is
    # the main use case here). 


def test_calc_l_dist():
    """Unit tests for combine_dists."""
    # Test array of length 1
    mean, sd, x, freqs = aft.calc_l_dist(r=np.array([0.5]),
                                         dpar=2.,
                                         tsteps=np.array([1e5, 0]),
                                         make_graph=True)
    assert mean == pytest.approx(8.21)
    assert sd < 0.854074  # stdev of un-truncated distribution
    assert sd == pytest.approx(0.854074, abs=0.1)
    assert x[0] == pytest.approx(8.21 - 0.854074 * 2)
    assert x[99] == pytest.approx(8.21 + 0.854074 * 2)
    assert np.argmax(freqs) == 49 or np.argmax(freqs) == 50
    assert np.argmin(freqs) == 0 or np.argmin(freqs) == 99

    # Test longer array
    mean, sd, x, freqs = aft.calc_l_dist(r=np.array([0., 15. / 16., 1.]),
                                         dpar=2.,
                                         tsteps=(1e6 / 1.551e-4) *
                                            np.array([np.log(10),
                                                      np.log(5), 
                                                      np.log(2), 
                                                      0]),
                                         make_graph=True,
                                         constants=TEST_CONSTS,
                                         l_num=150)
    
    assert mean == pytest.approx(15.27, rel=1e-3)
    assert sd < 0.674733     # What stdev would be if we didn't cut of dists
    assert sd == pytest.approx(0.674733, rel=1e-1)
    assert x[0] == pytest.approx(15. - 0.5378 * 2)
    assert x[149] == pytest.approx(16. + 0.5378 * 2)
    assert np.argmin(freqs) == 149


def test_forward_model_aft():
    """Unit tests for forward_model_aft.
    
    The time-temperature series for these tests are taken from Ketcham (2005)
    Figure 7, and the comparison data comes from HeFTy v1.9.3.

    These tests compare ages and means, standard deviations, and modes of 
    the length distributions. Modes may be off by an index because the
    discretization of the lengths varies between HeFTy and our model.
    Standard deviations tend to align less well between models, so the
    tolerance on most tests is more generous for standard deviations than
    for means.
    """
    # Figure 7a (Fast Cooling)
    # Creating time and temperature series
    initial_temps = np.linspace(190.1, 35, 20, endpoint=False)
    initial_tsteps = np.linspace(93, 89.1, 20, endpoint=False)

    second_temps = np.linspace(35, 20, 96 * 5, endpoint=True)
    second_tsteps = np.linspace(89.1, 0, 96 * 5, endpoint=True)

    # Converting temps to K and times to yrs
    temps = np.concatenate((initial_temps, second_temps), axis=None)
    temps += 273.15
    tsteps = np.concatenate((initial_tsteps, second_tsteps), axis=None)
    tsteps *= 1e6

    age, (mean, stdev, l_c, freqs) = aft.forward_model_aft(temps, 
                                                           tsteps, 
                                                           dpar=1.75, 
                                                           get_lengths=True, 
                                                           make_graph=True)

    assert age == pytest.approx(89.4, rel=5e-3)             # Age
    # Stdevs do not align well for this test, so they're skipped here
    assert mean == pytest.approx(15.03, rel=5e-3)           # Mean length
    assert np.argmax(freqs) == pytest.approx(np.argmin(np.abs(l_c - 15.)),
                                             abs=1)         # Mode length

    # Figure 7b (Constant Cooling)
    temps = np.linspace(190.1, 20, 500, endpoint=True)
    temps += 273.15

    tsteps = np.linspace(93, 0, 500, endpoint=True)
    tsteps *= 1e6

    age, (mean, stdev, l_c, freqs) = aft.forward_model_aft(temps, 
                                                           tsteps, 
                                                           dpar=1.75, 
                                                           get_lengths=True, 
                                                           make_graph=True)

    assert age == pytest.approx(39.8, rel=5e-3)             # Age
    assert mean == pytest.approx(14.24, rel=5e-3)           # Mean length
    assert stdev == pytest.approx(1.35, rel=0.1)            # Length stdev
    assert np.argmax(freqs) == pytest.approx(np.argmin(np.abs(l_c - 14.75)),
                                             abs=1)         # Mode length

    # Figure 7c (Reheating)
    first_temps = np.linspace(190.1, 19.8, 11 * 5, endpoint=False)
    first_tsteps = np.linspace(93, 79.5, 11 * 5, endpoint=False)

    second_temps = np.linspace(19.8, 92, 45 * 5, endpoint=False)
    second_tsteps = np.linspace(79.5, 37.9, 45 * 5, endpoint=False)

    third_temps = np.linspace(92, 20, 43 * 5, endpoint=True)
    third_tsteps = np.linspace(37.9, 0, 43 * 5, endpoint=True)

    temps = np.concatenate((first_temps, second_temps, third_temps), axis=None)
    temps += 273.15
    tsteps = np.concatenate((first_tsteps, second_tsteps, third_tsteps), 
                            axis=None)
    tsteps *= 1e6

    age, (mean, stdev, l_c, freqs) = aft.forward_model_aft(temps, 
                                                           tsteps, 
                                                           dpar=1.75, 
                                                           get_lengths=True, 
                                                           make_graph=True)

    assert age == pytest.approx(63.3, rel=5e-3)             # Age
    assert mean == pytest.approx(13.38, rel=5e-3)           # Mean length
    assert stdev == pytest.approx(1.56, rel=0.1)            # Length stdev
    assert np.argmax(freqs) == pytest.approx(np.argmin(np.abs(l_c - 12.125)),
                                             abs=1)         # Mode length

    # Figure 7d (Cooling at varying rates with 2 Dpar values)
    initial_temps = np.linspace(190.1, 121.7, 13 * 5, endpoint=False)
    initial_tsteps = np.linspace(93, 81, 13 * 5, endpoint=False)

    second_temps = np.linspace(121.7, 101.1, 75 * 5, endpoint=False)
    second_tsteps = np.linspace(81, 10.9, 75 * 5, endpoint=False)

    third_temps = np.linspace(101.1, 20, 13 * 5, endpoint=True)
    third_tsteps = np.linspace(10.9, 0, 13 * 5, endpoint=True)

    temps = np.concatenate((initial_temps, second_temps, third_temps), 
                           axis=None)
    temps += 273.15
    tsteps = np.concatenate((initial_tsteps, second_tsteps, third_tsteps), 
                            axis=None)
    tsteps *= 1e6

    age, (mean, stdev, l_c, freqs) = aft.forward_model_aft(temps, 
                                                           tsteps, 
                                                           dpar=1.75, 
                                                           get_lengths=True, 
                                                           make_graph=True)

    assert age == pytest.approx(13.0, abs=0.25)             # Age
    assert mean == pytest.approx(13.95, rel=5e-3)           # Mean length
    assert stdev == pytest.approx(1.70, rel=5e-2)           # Length stdev
    assert np.argmax(freqs) == pytest.approx(np.argmin(np.abs(l_c - 14.875)),
                                             abs=1)         # Mode length

    age, (mean, stdev, l_c, freqs) = aft.forward_model_aft(temps, 
                                                           tsteps, 
                                                           dpar=2.50, 
                                                           get_lengths=True, 
                                                           make_graph=True)

    assert age == pytest.approx(51.1, rel=1.1e-2)           # Age
    assert mean == pytest.approx(12.91, rel=5e-3)           # Mean length
    assert stdev == pytest.approx(1.52, rel=5e-2)           # Length stdev
    assert np.argmax(freqs) == pytest.approx(np.argmin(np.abs(l_c - 12.375)),
                                             abs=1)         # Mode length

    # Testing not getting length distributions
    age = aft.forward_model_aft(temps, tsteps, 2.50)
    assert age == pytest.approx(51.1, rel=1.1e-2)