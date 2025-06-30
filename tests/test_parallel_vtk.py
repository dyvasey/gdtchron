"""Tests for parallel_vtk module."""
import os
import shutil
from contextlib import suppress

import numpy as np
import pytest
import pyvista as pv

from gdtchron import aft, he, run_tt_paths, run_vtk

# Constants for t-T series in run_vtk
NUM_VTU_FILES = 10  # Must be at least 3
NUM_PARTICLES = 16
X_DIM = 4
Y_DIM = 4
TIME_INTERVAL = 0.2  # Myr
MAX_TEMP = 400.
DELTA_TEMP = 10.

# TODO: Test overwriting
# TODO: Test particle functions


def test_run_vtk():
    """Unit tests for basic functionality of run_vtk."""
    # Generate dummy VTK files for testing
    filenames = []
    for x in range(NUM_VTU_FILES):
        # Create very small mesh with 16 points and assign each an id and temp 
        mesh = pv.ImageData(dimensions=(X_DIM, 
                                        Y_DIM, 
                                        1)).cast_to_unstructured_grid()
        mesh['id'] = np.arange(NUM_PARTICLES)
        # Make temperature the same for all points so it cools over time
        mesh['T'] = MAX_TEMP - (x * DELTA_TEMP * np.ones(NUM_PARTICLES))

        filename = 'file_' + str(x) + '.vtu'
        mesh.save(filename)
        filenames.append(filename)

    # Define times (Ma) and temps (K) for the 10 files
    times = np.arange(start=TIME_INTERVAL * (NUM_VTU_FILES - 1), 
                      stop=-0.5 * TIME_INTERVAL, 
                      step=-TIME_INTERVAL)
    temps = np.arange(start=MAX_TEMP, 
                      stop=MAX_TEMP - (NUM_VTU_FILES - 0.5) * DELTA_TEMP, 
                      step=-DELTA_TEMP)
    
    # Test using the same mesh for data from multiple systems
    run_vtk(files=filenames,
            system='AHe',
            time_interval=TIME_INTERVAL,
            file_prefix='meshes_tchron',
            overwrite=True)
    
    run_vtk(files=filenames,
            system='ZHe',
            time_interval=TIME_INTERVAL,
            file_prefix='meshes_tchron',
            overwrite=True)

    run_vtk(files=filenames,
            system='AFT',
            time_interval=TIME_INTERVAL,
            file_prefix='meshes_tchron',
            overwrite=True)
    
    for i in range(NUM_VTU_FILES):
        prefix = 'meshes_tchron'
        suffix = '_' + str(i).zfill(3) + '.vtu'
        mesh = pv.read(os.path.join('./' + prefix, prefix + suffix))

        # Test all systems
        for sys in ['AHe', 'ZHe', 'AFT']:
            if i == 0:
                assert np.array(mesh[sys]) == \
                    pytest.approx(np.ones(NUM_PARTICLES) * 0.)
            else:
                if sys[1:] == 'He':
                    expected_age = he.forward_model_he(temps=temps[:i + 1],
                                                tsteps=times[:i + 1],
                                                system=sys,
                                                u=100,
                                                th=100,
                                                radius=50)
                else:
                    expected_age = aft.forward_model_aft(temps=temps[:i + 1],
                                                         tsteps=times[:i + 1],
                                                         dpar=1.75)
                assert np.array(mesh[sys]) == \
                    pytest.approx(np.ones(NUM_PARTICLES) * expected_age, 
                                  rel=1e-3)
                

def test_run_vtk_interpolate_no_overwrite():
    """Unit tests for interpolation and not using overwriting with run_vtk."""
    # Delete old directories
    prefix = 'meshes_'
    for sys in ['AHe', 'ZHe', 'AFT']:
        with suppress(Exception):
            shutil.rmtree('./' + prefix + sys + '_int')

    # Generate dummy VTK files for testing
    filenames = []
    for x in range(NUM_VTU_FILES):
        # Create very small mesh with 16 points and assign each an id and temp 
        mesh = pv.ImageData(dimensions=(4, 4, 1)).cast_to_unstructured_grid()
        if x == 3:
            mesh['id'] = np.arange(NUM_PARTICLES)
        else:
            mesh['id'] = np.arange(NUM_PARTICLES, NUM_PARTICLES * 2)
        # Make temperature the same for all but one point
        mesh['T'] = np.append(MAX_TEMP - (x * DELTA_TEMP * np.ones(15)), 
                              MAX_TEMP)

        filename = 'file_interp_' + str(x) + '.vtu'
        mesh.save(filename)
        filenames.append(filename)

    # Define times (Ma) and temps (K) for the 10 files
    times = np.arange(start=TIME_INTERVAL * (NUM_VTU_FILES - 1), 
                      stop=-0.5 * TIME_INTERVAL, 
                      step=-TIME_INTERVAL)
    temps = np.arange(start=MAX_TEMP, 
                      stop=MAX_TEMP - (NUM_VTU_FILES - 0.5) * DELTA_TEMP, 
                      step=-DELTA_TEMP)
    
    # Temps for the last particle
    last_temps = np.ones(NUM_VTU_FILES) * MAX_TEMP
    
    # Test interpolation (with overwriting)
    run_vtk(files=filenames[:NUM_VTU_FILES // 2],
            system='AHe',
            time_interval=TIME_INTERVAL,
            file_prefix='meshes_AHe_int',
            overwrite=True)
    
    run_vtk(files=filenames[:NUM_VTU_FILES // 2],
            system='ZHe',
            time_interval=TIME_INTERVAL,
            file_prefix='meshes_ZHe_int',
            overwrite=True)

    run_vtk(files=filenames[:NUM_VTU_FILES // 2],
            system='AFT',
            time_interval=TIME_INTERVAL,
            file_prefix='meshes_AFT_int',
            overwrite=True)
    
    for i in range(NUM_VTU_FILES // 2):
        suffix = '_int_' + str(i).zfill(3) + '.vtu'

        # Test all systems
        for sys in ['AHe', 'ZHe', 'AFT']:
            mesh = pv.read(os.path.join('./' + prefix + sys + '_int', 
                                        prefix + sys + suffix))
            if i == 0:
                assert np.array(mesh[sys]) == \
                    pytest.approx(np.ones(NUM_PARTICLES) * 0.)
            else:
                if sys[1:] == 'He':
                    expected_age = he.forward_model_he(temps=temps[:i + 1],
                                                       tsteps=times[:i + 1],
                                                       system=sys,
                                                       u=100,
                                                       th=100,
                                                       radius=50)
                    last_age = he.forward_model_he(temps=last_temps[:i + 1],
                                                   tsteps=times[:i + 1],
                                                   system=sys,
                                                   u=100,
                                                   th=100,
                                                   radius=50)
                else:
                    expected_age = aft.forward_model_aft(temps=temps[:i + 1],
                                                         tsteps=times[:i + 1],
                                                         dpar=1.75)
                    last_age = aft.forward_model_aft(temps=last_temps[:i + 1],
                                                     tsteps=times[:i + 1],
                                                     dpar=1.75)
                assert np.array(mesh[sys])[:-1] == \
                    pytest.approx(np.ones(NUM_PARTICLES - 1) * expected_age, 
                                  rel=1e-3)
                assert np.array(mesh[sys][-1]) == pytest.approx(last_age, 
                                                                rel=5e-2)


def test_run_tt_paths():
    """Unit tests for run_tt_paths."""
    # Test (U-Th) / He system
    he_times = np.arange(60, -0.001, -0.1)
    
    # Constants for test 1
    early_temps_1 = np.linspace(120, 20, 101)
    late_temps_1 = np.linspace(20, 20, 500)
    temps_1 = np.append(early_temps_1, late_temps_1) + 273

    # Constants for test 2
    temps_2 = np.linspace(120, 20, 601) + 273

    # Constants for test 3
    early_temps_3 = np.linspace(120, 65, 576) + 273
    late_temps_3 = np.linspace(65, 20, 26) + 273
    temps_3 = np.append(early_temps_3, late_temps_3[1:])

    temp_paths = [temps_1, temps_2, temps_3]

    ages = run_tt_paths(temp_paths=temp_paths, 
                        tsteps=he_times, 
                        system='AHe',
                        radius=100,
                        processes=3)

    assert ages[0] == pytest.approx(54.4, rel=5e-3)
    assert ages[1] == pytest.approx(26.3, rel=5e-2)
    assert ages[2] == pytest.approx(6.83, rel=1e-1)
    assert len(ages) == 3

    # Test AFT system (and a singleton list of temps)
    temps = np.linspace(190.1, 20, 500, endpoint=True)
    temps += 273.15
    ft_times = np.linspace(93, 0, 500, endpoint=True)

    age_list = run_tt_paths(temp_paths=[temps], tsteps=ft_times, system='AFT')
    assert age_list[0] == pytest.approx(39.8, rel=5e-3)
    assert len(age_list) == 1