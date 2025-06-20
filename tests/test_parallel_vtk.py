"""Tests for parallel_vtk module."""
import numpy as np
import pyvista as pv

# Generate dummy VTK files for testing
for x in range(10):
    # Create very small mesh with 16 points and assign each an id and temp 
    mesh = pv.ImageData(dimensions=(4, 4, 1)).cast_to_unstructured_grid()
    mesh['id'] = np.arange(16)
    # Make temperature the same for all points so it cools over time
    mesh['T'] = 100 - (x * 10 * np.ones(16))

    mesh.save('file_' + str(x) + '.vtu')

# Define times (Ma) for the 10 files
times = np.arange(9, -1, 1)