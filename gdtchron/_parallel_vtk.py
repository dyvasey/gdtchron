"""Module for running forward models across multiple T-t paths and VTK meshes.

This module includes a function for running forward models across multiple
user-provided t-T paths (run_tt_paths) and a function to run forward models
across the T-t paths experienced by different particles across a series of
VTK meshes (run_vtk)
"""

import gc
import os
import shutil
import warnings
from contextlib import suppress

import numpy as np
import pyvista as pv
from joblib import Parallel, delayed
from scipy.spatial import KDTree
from tqdm import tqdm

from gdtchron import aft, he


def run_particle_he(particle_id, inputs, calc_age, interpolate_profile,
                        dtype=np.float32):  
    """Calculate profile of x values for a particular ASPECT particle.

    Function to calculate the profile of x values (He concentrations times
    node positions) across all nodes within a hypothetical grain found in 
    a given particle. This function's primary purpose is to be run in
    parallel by run_vtk.

    Parameters
    ----------
    particle_id : int
        ID corresponding to the particle to get the He profile of
    inputs : tuple
        k : any
            Unused parameter (included here for symmetry with the inputs of
            run_particle_aft)
        positions : pyvista.core.pyvista_ndarray.pyvista_ndarray
            x, y, z coordinates of each particle
        tree : scipy.spatial._kdtree.KDTree or None
            K-d tree containing the positions of particles from the previous
            timestep. Unused (and typically set to None) if interpolate_profile
            is False.
        ids : pyvista.core.pyvista_ndarray.pyvista_ndarray
            IDs for all particles from the current timestep
        old_ids : pyvista.core.pyvista_ndarray.pyvista_ndarray
            IDs for all particles from the previous timestep
        tree_ids : pyvista.core.pyvista_ndarray.pyvista_ndarray or None
            IDs for all particles with profiles from the previous timestep. Not
            used and typically set to None if interpolate_profile is True
        temps : pyvista.core.pyvista_ndarray.pyvista_ndarray
            Temperatures for all particles from the current timestep
        old_temps : pyvista.core.pyvista_ndarray.pyvista_ndarray
            Temperatures for all particles from the previous timestep
        old_profiles : NumPy ndarray of NumPy arrays of floats
            Profiles of x values for all particles.
        time_interval : float
            Time elapsed between mesh files (Myr)
        system : string
            Isotopic system to model. Valid options are 'AHe' (apatite He) or
            'ZHe' (zircon He)
        num_nodes : int
            Number of nodes to use within each profile
        model_inputs : tuple
            u : float
                U concentration (ppm). 
            th : float
                Th concentration (ppm).
            radius : float
                Radius of the grain (micrometers).
    calc_age : bool
        Boolean indicating whether to calculate age of particle. If False,
        age is returned as np.nan. Note that setting this to False will not
        substantially improve the speed of calculations for this function.
    interpolate_profile : bool
        Boolean indicating whether to interpolate He data from nearest neighbor
        of the particle if the particle itself lacks He data. If False and the
        particle is missing He data, an age of np.nan and a profile filled with 
        np.inf are returned.
    dtype : type
        Type of numbers used for calculations (default: np.float32). 32-bit 
        floats are preferred to save memory.

    Returns
    -------
    age : float
        Age of the particle. This equals np.nan if calc_age is False or there
        is an issue obtaining the age of the particle
    x : NumPy array of floats
        Matrix x solved for using Equation 21 in Ketcham (2005) (in that paper,
        x is referred to as u).
        Equivalent to the He concentration (mol / g) times the node position
        (micrometers).

    """
    # Unpack inputs
    (k, positions, tree, ids, old_ids, tree_ids, temps, old_temps, old_profiles,
     time_interval, system, num_nodes, (u, th, radius)) = inputs
    
    # Get old profile and temperature for current particle if present
    array = old_profiles[particle_id == old_ids]
    particle_start_temp = old_temps[particle_id == old_ids]

    # Create variable to track if missing old data for particle
    missing = False
     
    # If array is empty, assign np.nan
    # If the initial array is filled with np.inf (i.e, we have a bugged
    # particle), then return (NaN, initial array)
    # Otherwise, assign new value from old profile
    if array.size == 0:
        profile = np.empty(num_nodes, dtype=dtype)
        profile.fill(np.nan)
        missing = True
    elif array[0][0] == dtype(np.inf):
        age = np.nan
        return (age, array[0])
    else:
        profile = array 
    
    # Get particle temperature
    particle_end_temp = temps[ids == particle_id]
       
    # If particle not found, don't attempt to calculate profile or age
    if particle_end_temp.size == 0:            
        x = np.empty(num_nodes, dtype=dtype)
        x.fill(dtype(np.inf))
        age = np.nan
        return (age, x)
    
    # Use previous He from nearest neighbor in previous timestep if none present
    
    if missing:
        
        if interpolate_profile:
        
            # Get particle position
            particle_position = positions[ids == particle_id]
            
            # Find closest particle
            distance, index = tree.query(particle_position)
            
            # Get id of closest particle
            neighbor_id = tree_ids[index]

            # Get profile of closest particle
            try:
                profile = old_profiles[neighbor_id == old_ids]
            except Exception:
                warnings.warn("Warning: likely multi-dimensional id", 
                              stacklevel=2)
                x = np.empty(num_nodes, dtype=dtype)
                x.fill(dtype(np.inf))
                age = np.nan
                return (age, x)
            
            # Get temp of closest particle
            particle_start_temp = old_temps[neighbor_id == old_ids]
        
        # If turned off, return original profile of np.inf
        elif not interpolate_profile:
            x = np.empty(num_nodes, dtype=dtype)
            x.fill(dtype(np.inf))
            age = np.nan
            return (age, x)
    
    # Double checking that interpolated particle isn't bugged
    if profile[0][0] == dtype(np.inf):
        age = np.nan
        return (age, profile[0])
        
    # passing start and end temperatures to forward model
    particle_temps = np.array([particle_start_temp[0], particle_end_temp[0]])
    particle_tsteps = np.array([time_interval, 0])

    age, age_unc, he_tot, pos, v, x = \
        he.forward_model_he(temps=particle_temps, 
                            tsteps=particle_tsteps,
                            system=system,
                            u=u,
                            th=th,
                            radius=radius,
                            nodes=num_nodes,
                            initial_x=profile.flatten(),
                            return_all=True)
    
    if calc_age:
        return (age, x)
    else:
        age = np.nan
        return (age, x)
    

def run_particle_ft(particle_id, inputs, calc_age, interpolate_vals):  
    """Calculate FT reduced lengths for a particular ASPECT particle.

    Function to calculate the reduced lengths (unitless) within a hypothetical 
    grain found in a given particle. This function's primary purpose is to be 
    run in parallel by run_vtk.

    Parameters
    ----------
    particle_id : int
        ID corresponding to the particle to get the reduced lengths of
    inputs : tuple
        k : int
            Index of the current timestep/mesh being processed.
        positions : pyvista.core.pyvista_ndarray.pyvista_ndarray
            x, y, z coordinates of each particle
        tree : scipy.spatial._kdtree.KDTree or None
            K-d tree containing the positions of particles from the previous
            timestep. Unused (and typically set to None) if interpolate_vals
            is False.
        ids : pyvista.core.pyvista_ndarray.pyvista_ndarray
            IDs for all particles from the current timestep
        old_ids : pyvista.core.pyvista_ndarray.pyvista_ndarray
            IDs for all particles from the previous timestep
        tree_ids : pyvista.core.pyvista_ndarray.pyvista_ndarray or None
            IDs for all particles with profiles from the previous timestep. Not
            used (and typically set to False) if interpolate_vals is True.
        temps : pyvista.core.pyvista_ndarray.pyvista_ndarray
            Temperatures for all particles from the current timestep
        old_temps : pyvista.core.pyvista_ndarray.pyvista_ndarray
            Temperatures for all particles from the previous timestep
        old_annealing_arrays : NumPy ndarray of NumPy arrays of floats
            r values for all particles from the previous timestep
        time_interval : float
            Time elapsed between mesh files (Myr)
        system : string
            Isotopic system to model. Not used for FT system (but included as a
            parameter for symmetry with run_particle_he)
        r_length : int
            Length of the r arrays for particles that have them
        model_inputs : tuple       
            dpar : float
                Etch figure length (micrometers).
            annealing_model : string
                Annealing model to use. Currently, the only acceptable value is
                'Ketcham99', which corresponds to the fanning curvilinear model 
                from Ketcham et al. (1999).
    calc_age : bool
        Boolean indicating whether to calculate age of particle. If False,
        age is returned as np.nan.
    interpolate_vals : bool
        Boolean indicating whether to interpolate FT data from nearest neighbor
        of the particle if the particle itself lacks FT data. If False and the
        particle is missing FT data, an age of np.nan and a profile filled with 
        np.inf are returned.

    Returns
    -------
    age : float
        Age of the particle. This equals np.nan if calc_age is False or there
        is an issue obtaining the age of the particle
    r : NumPy array of floats
        Updated reduced lengths (unitless) for a hyppothetical grain located 
        within this particle.

    """
    # Use float64 to match nans
    dtype = np.float64
    
    # Unpack inputs
    (k, positions, tree, ids, old_ids, tree_ids, temps, old_temps, 
     old_annealing_arrays, time_interval, system, 
     r_length, (dpar, annealing_model)) = inputs
    
    ft_constants = {'Ketcham99': aft.KETCHAM_99_FC}
    
    # Get old profile and temperature for current particle if present
    r_initial = old_annealing_arrays[particle_id == old_ids]
    particle_start_temp = old_temps[particle_id == old_ids]

    # Create variable to track if missing old data for particle
    missing = False
     
    # If array is empty, assign np.nan
    # If r_initial is filled with np.inf (i.e, we have a bugged particle), 
    # return (NaN, r_initital)
    if r_initial.size == 0:
        r_initial = np.empty(r_length, dtype=dtype)
        r_initial.fill(np.nan) 
        missing = True
    elif r_initial[0][0] == dtype(np.inf):
        age = np.nan
        return (age, r_initial[0])
    else:
        r_initial = r_initial[0]
    
    # Get final particle temperature
    particle_end_temp = temps[ids == particle_id]
       
    # If particle not found, don't attempt to calculate profile or age
    if particle_end_temp.size == 0:            
        x = np.empty(r_length, dtype=dtype)
        x.fill(dtype(np.inf))
        age = np.nan
        return (age, x)
    
    # Use annealing from nearest neighbor in previous timestep if none present
    
    if missing:
        
        if interpolate_vals:
        
            # Get particle position
            particle_position = positions[ids == particle_id]
            
            # Find closest particle
            distance, index = tree.query(particle_position)
            
            # Get id of closest particle
            neighbor_id = tree_ids[index]
            
            # Get profile of closest particle
            try:
                r_initial = old_annealing_arrays[neighbor_id == old_ids][0]
            except Exception:
                warnings.warn("Warning: likely multi-dimensional id", 
                              stacklevel=2)
                x = np.empty(r_length, dtype=dtype)
                x.fill(dtype(np.inf))
                age = np.nan
                return (age, x)

            # Get temp of closest particle
            particle_start_temp = old_temps[neighbor_id == old_ids]
        
        # If turned off, return np.nan
        elif not interpolate_vals:
            x = np.empty(r_length, dtype=dtype)
            x.fill(dtype(np.inf))
            age = np.nan
            return (age, x)
    
    # Double checking that interpolated particle isn't bugged
    if r_initial[0] == dtype(np.inf):
        age = np.nan
        return (age, r_initial)
        
    # Getting average temperature
    particle_temp = (particle_start_temp[0] + particle_end_temp[0]) / 2
    
    # For basic annealing calculations, absolute time doesn't matter -
    # we just need to maintain difference between start and end times
    # (start time > end time because the function measures time in yrs BP)
    r = aft.calc_annealing(r_initial, particle_temp, start=time_interval, 
                           end=0, next_nan_index=k - 1, 
                           constants=ft_constants[annealing_model])

    if calc_age:
        r_so_far = aft.dpar_conversion(r_mr=r[~np.isnan(r)], 
                                       dpar=dpar, 
                                       constants=ft_constants[annealing_model])
        tsteps = np.arange(start=k * time_interval, 
                           stop=-0.5 * time_interval, 
                           step=-1 * time_interval)
        age = aft.calc_aft_age(r_so_far, tsteps)
        return (age, r)
        
    else:    
        age = np.nan
        return (age, r)
    

def run_vtk(files, system, time_interval, 
            u=100, th=100, radius=50, num_nodes=513,
            dpar=1.75, annealing_model='Ketcham99',
            file_prefix='meshes_tchron', path='./', 
            temp_dir='~/dump',
            batch_size=100, processes=None, interpolate_vals=True, 
            all_timesteps=True, overwrite=False):
    """Perform parallel He or FT forward modeling of ASPECT VTK data.

    This code performs forward modeling of the AHe, ZHe, or AFT systems
    across ASPECT VTK data. Data is output as .vtu folders in a new
    directory, with data for every timestep given. 
    
    Parameters
    ----------
    files : list of strings
        List of paths to vtu files to run forward model on. Files are
        processed in the order they are given in the list.
    system : string
        Isotopic system to model. Current options are:
            'AHe': Apatite (U-Th) / He
            'ZHe': Zircon (U-Th) / He
            'AFT': Apatite Fission Track
    time_interval : float
        Interval (Myrs) between times when each mesh was produced
    u : float, optional
        U concentration (ppm). Default is 100 ppm. Only used if system is 'AHe'
        or 'ZHe'.
    th : float, optional
        Th concentration (ppm). Default is 100 ppm. Only used if system is 'AHe'
        or 'ZHe'.
    num_nodes : int 
        Number of nodes within the grain (for He) (default: 513). Unused for FT 
        system
    radius : float, optional
        Radius of the grain (micrometers). Default is 50 micrometers. Only used 
        if system is 'AHe' or 'ZHe'.
    dpar : float, optional
        Etch figure length (micrometers). Default is 1.75 micrometers. Only used
        if system is 'AFT'.
    annealing_model : string, optional
        Annealing model to use. Currently, the only acceptable value is
        'Ketcham99', which corresponds to the fanning curvilinear model from
        Ketcham et al. (1999). Default is 'Ketcham99'. Only used if system is
        'AFT'.
    file_prefix : string
        Prefix to give output files (default: 'meshes_tchron')
    path : string
        Path to output directory (default: './')
    temp_dir : string
        Path to temporary output directory
    batch_size : int or 'auto', optional
        Number of batches to be dispatched to each worker at a time during 
        parallel computation (default: 100). If set to 'auto', this value is
        dynamically adjusted during computations to try to optimize efficiency.
    processes : int or None, optional
        Maximum number of processes that can run concurrently. If None, this
        parameter is internally set to two less than the number of CPUs on the
        user's system (default: None).
    interpolate_vals : bool
        Boolean indicating whether to interpolate particle data from nearest 
        neighbor if the particle itself lacks He data. If False and the
        particle is missing data, an age of np.nan is returned for that 
        particle. (default: True)
    all_timesteps : bool
        Boolean indicating whether to calculate ages at each tstep 
        (default: True)
    overwrite : bool
        Boolean indicating whether to overwrite old meshes that already have
        thermochronometric data for this system for a given timestep. If False, 
        this function skips timesteps that already have data for this system
        and uses that data and uses for calculations in subsequent meshes. 
        (default: False)
        TODO: This doesn't currently work when False

    Returns
    -------
    This function does not return any values.

    """
    dtype = np.float32
    
    if processes is None:
        processes = os.cpu_count() - 2
    
    pre_dispatch = 2 * processes if batch_size == 'auto' else 2 * batch_size
    
    particle_fn = {'AHe': run_particle_he,
                'ZHe': run_particle_he,
                'AFT': run_particle_ft}
    
    # Path for temporary memory dumps
    temp_dir = os.path.expanduser(temp_dir)
    
    with suppress(Exception):
        shutil.rmtree(temp_dir)
    
    os.makedirs(temp_dir)
    
    new_dir = os.path.join(path, file_prefix)
    os.makedirs(new_dir, exist_ok=True)
    
    # Path for dump of cached internal values
    cache_path = os.path.join(new_dir, 'cache_internal_vals.npy')
    
    with Parallel(n_jobs=processes,
                  batch_size=batch_size,
                  pre_dispatch=pre_dispatch,
                  temp_folder=temp_dir) as parallel:
    
        # Loop through timesteps
        for k, file in enumerate(files):  
            
            filename = file_prefix + '_' + str(k).zfill(3) + '.vtu'
            filepath = os.path.join(new_dir, filename)
            
            # Check if target mesh already exists
            if os.path.exists(filepath):
                original_mesh = pv.read(filepath)

                # If the mesh only has data from other systems, set that mesh
                # as our output mesh (to avoid overwriting other system data)
                if system not in original_mesh.point_data:
                    mesh = original_mesh
                else:
                    if not overwrite:            
                        # Check for the next target mesh
                        next_filename = file_prefix + '_' + \
                            str(k + 1).zfill(3) + '.vtu'
                        next_filepath = os.path.join(new_dir, next_filename)
                        
                        # If next mesh does not exist or does not have the 
                        # system, load values from cache
                        next_mesh_exists = os.path.exists(next_filepath)
                        if (not next_mesh_exists) or \
                            (system not in pv.read(next_filepath).point_data):
                            ids = original_mesh['id']
                            positions = original_mesh.points
                            temps = original_mesh['T']
                            new_internal_vals = np.load(cache_path)
                        continue
                
                    mesh = pv.read(file)
            
            else:
                mesh = pv.read(file)
            
            if k == 0:

                num_particles = len(mesh['T'])

                # Set up empty arrays for first timestep
                # For AFT system, can use number of files to determine array len
                # -1 is to account for missing tstep from using averages
                # For He system, we just use the input num_nodes
                internal_len = len(files) - 1 if system == "AFT" else num_nodes

                new_internal_vals = np.empty((num_particles, internal_len), 
                                             dtype=dtype)
                new_internal_vals.fill(np.nan)
                np.save(cache_path, new_internal_vals)

                # Publish ages at 0
                ages = np.zeros(num_particles)
                mesh[system] = ages
                mesh.save(filepath)
            elif k > 0:  
                old_internal_vals = new_internal_vals

                old_ids = ids  # Get ids for previous internal values
                old_positions = positions
                old_temps = temps

            temps = mesh['T']
            
            gc.collect()
            if k in np.arange(5, len(files), 5):
                shutil.rmtree(temp_dir)
                os.makedirs(temp_dir)
            
            ids = mesh['id']
            positions = mesh.points

            # Running the forward model if we've seen at least 2 timesteps
            if k > 0:
            
                # Set up KDTree for timestep if doing interpolation
                if interpolate_vals:
                    
                    # Get particle ids of particles with internal_vals
                    has_vals = ~np.isnan(old_internal_vals).all(axis=1)
                    tree_ids = old_ids[has_vals]
                    
                    # Get positions of other particles
                    other_positions = old_positions[has_vals]

                    # Note: At k=0, all particles have internal_vals but
                    #       they're all set to NaN, so we need a special case
                    if k == 1:
                        tree_ids = old_ids
                        other_positions = old_positions
                    
                    # Set up KDTree to find closest particle
                    tree = KDTree(other_positions)
                    
                else:
                    tree = None
                    tree_ids = None

                if system == 'AFT':
                    model_inputs = (dpar, annealing_model)
                else:
                    model_inputs = (u, th, radius)
                    
                inputs = (k, positions, tree, ids, old_ids, tree_ids, 
                          temps, old_temps, old_internal_vals,
                          time_interval, system, internal_len, model_inputs)
                    
                # Calculate ages if indicated or on last timestep
                calc_age = all_timesteps or k == len(files) - 1
                
                prog_bar_txt = "Timestep " + str(k)
                
                output = parallel(
                    delayed(particle_fn[system])
                    (particle, inputs, calc_age, interpolate_vals)
                    for particle in tqdm(ids, position=0, desc=prog_bar_txt)
                    )
                
                ages, new_internal_vals = zip(*output)
            
                # Convert new_internal_vals to array and save for reload
                new_internal_vals = np.array(new_internal_vals, dtype=dtype)
                np.save(cache_path, new_internal_vals)
            
                # Assign ages to mesh
                mesh.point_data[system] = np.array(ages, dtype=dtype)
            
                # Save new mesh
                mesh.save(filepath)
                
                # Purge the temp folder
                with suppress(Exception):
                    shutil.rmtree(temp_dir)
                os.makedirs(temp_dir)
    
    # Delete cached values when all finished
    os.remove(cache_path)
    gc.collect()

    return
    

def run_tt_paths(temp_paths, tsteps, system, 
                 u=100, th=100, radius=50,
                 dpar=1.75, annealing_model='Ketcham99',
                 batch_size=100, processes=None,
                 **kwargs):
    """Run forward model of a given isotopic system across multiple t-T paths.

    Parameters
    ----------
    temp_paths : list of NumPy arrays of floats
        List of NumPy arrays of floats containing the temperatures (K) at each
        timestep in tsteps. Each array corresponds to a different grain
        to obtain a thermochronometric age for.
    tsteps : Numpy array of floats
        Array of times (Ma) in chronological (descending) order. First 
        time is start of first timestep, last time is end of last timestep. 
        Each pair of adjacent times composes a timestep. The time at a given
        index i corresponds to the temperatures at index i of each of the NumPy
        arrays in temp_paths.
    system : string
        Isotopic system to model. Current options are:
            'AHe': Apatite (U-Th) / He
            'ZHe': Zircon (U-Th) / He
            'AFT': Apatite Fission Track
    u : float, optional
        U concentration (ppm). Default is 100 ppm. Only used if system is 'AHe'
        or 'ZHe'.
    th : float, optional
        Th concentration (ppm). Default is 100 ppm. Only used if system is 'AHe'
        or 'ZHe'.
    radius : float, optional
        Radius of the grain (micrometers). Default is 50 micrometers. Only used 
        if system is 'AHe' or 'ZHe'.
    dpar : float, optional
        Etch figure length (micrometers). Default is 1.75 micrometers. Only used
        if system is 'AFT'.
    annealing_model : string, optional
        Annealing model to use. Currently, the only acceptable value is
        'Ketcham99', which corresponds to the fanning curvilinear model from
        Ketcham et al. (1999). Default is 'Ketcham99'. Only used if system is
        'AFT'.
    batch_size : int or 'auto', optional
        Number of batches to be dispatched to each worker at a time during 
        parallel computation (default: 100). If set to 'auto', this value is
        dynamically adjusted during computations to try to optimize efficiency.
    processes : int or None, optional
        Maximum number of processes that can run concurrently. If None, this
        parameter is internally set to two less than the number of CPUs on the
        user's system (default: None).
    **kwargs : optional
        Additional arguments to pass to the forward model function of the
        corresponding isotopic system

    Returns
    -------
    ages : list of floats
        Thermochronometric ages for the given isotopic system for grains that 
        experienced each of the provided time series. All (U-Th) / He ages 
        returned are corrected ages. 
    
    """
    if processes is None:
        processes = os.cpu_count() - 2

    # Setting how many tasks to initially dispatch to workers
    pre_dispatch = 2 * processes if batch_size == 'auto' else 2 * batch_size

    model_fn = {'AHe': he.forward_model_he,
                'ZHe': he.forward_model_he,
                'AFT': aft.forward_model_aft}
    ft_constants = {'Ketcham99': aft.KETCHAM_99_FC}

    he_inputs = (tsteps, system, u, th, radius)
    ft_inputs = (tsteps, dpar, ft_constants[annealing_model])

    model_inputs = {'AHe': he_inputs,
                    'ZHe': he_inputs,
                    'AFT': ft_inputs}
    
    output = Parallel(n_jobs=processes,
                  batch_size=batch_size,
                  pre_dispatch=pre_dispatch)(
                    delayed(model_fn[system])(path, 
                                              *model_inputs[system], 
                                              **kwargs)
                    for path in tqdm(temp_paths, position=0))
    
    return output
