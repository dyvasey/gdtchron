"""Tests for visualization module."""
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from gdtchron import plot_vtk_2d

# Create very small mesh of 9 cells with 16 points and assign each a value
mesh = pv.ImageData(dimensions=(4, 4, 1)).cast_to_unstructured_grid()
mesh['sample_field'] = np.arange(9)

def test_plot_vtk_2d():
    """Test plot_vtk_2d function."""
    # Create Matplotlib figure/axes
    fig,ax = plt.subplots(1)

    # Attempt to plot the mesh on the axes, colored by the sample field and with
    # bounds restricted to 0-2 on the x and y axes.
    ax = plot_vtk_2d(mesh,'sample_field',bounds=[0,2,0,2],
                                       ax=ax)

    # Test that the figure contains 1 axes
    assert len(fig.get_axes()) == 1
    
    # Test that the axes contains 1 image
    assert len(ax.get_images()) == 1
    
    # Test that the axes x limits are correct
    assert ax.get_xlim() == (0,2)

    # Test that the axes y limits are correct
    assert ax.get_ylim() == (0,2)