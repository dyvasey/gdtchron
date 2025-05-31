"""Tests for visualization module."""
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from gdtchron import add_comp_field, plot_vtk_2d

# Create very small mesh with 16 points and assign each a value
mesh = pv.ImageData(dimensions=(4, 4, 1)).cast_to_unstructured_grid()
mesh['sample_field'] = np.arange(16)


def test_plot_vtk_2d():
    """Test plot_vtk_2d function."""
    # Test that sample_field in the mesh
    assert 'sample_field' in mesh.point_data

    # Create Matplotlib figure/axes
    fig, ax = plt.subplots(1)

    # Attempt to plot the mesh on the axes, colored by the sample field and with
    # bounds restricted to 0-2 on the x and y axes.
    ax = plot_vtk_2d(mesh, 'sample_field', bounds=[0, 2, 0, 2],
                                       ax=ax)

    # Test that the figure contains 1 axes
    assert len(fig.get_axes()) == 1
    
    # Test that the axes contains 1 image
    assert len(ax.get_images()) == 1
    
    # Test that the axes x limits are correct
    assert ax.get_xlim() == (0, 2)

    # Test that the axes y limits are correct
    assert ax.get_ylim() == (0, 2)


def test_add_comp_field():
    """Test add_comp_field function."""
    # Test basic functionality using sample field
    mesh_comp_field = add_comp_field(mesh, 'sample_field')

    # Ensure comp_field is created
    assert 'comp_field' in mesh_comp_field.point_data

    # Both sample_field and null should be present
    assert 1 in mesh_comp_field['comp_field']
    assert 0 in mesh_comp_field['comp_field']

    # Test functionality with default fields

    # Create random arrays between 0 and 1 for each field
    rng = np.random.default_rng(seed=15)
    mesh['crust_upper'] = rng.random(16)
    mesh['crust_lower'] = rng.random(16)
    mesh['mantle_lithosphere'] = rng.random(16)
    
    # Ensure one point will be assigned null value
    mesh['crust_upper'][-1] = 0
    mesh['crust_lower'][-1] = 0
    mesh['mantle_lithosphere'][-1] = 0

    # Run function with defaults
    mesh_comp_defaults = add_comp_field(mesh)

    # Ensure comp_field is created
    assert 'comp_field' in mesh_comp_defaults.point_data

    # All field and null should be present
    assert 3 in mesh_comp_defaults['comp_field']
    assert 2 in mesh_comp_defaults['comp_field']
    assert 1 in mesh_comp_defaults['comp_field']
    assert 0 in mesh_comp_defaults['comp_field']



