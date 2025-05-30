"""GDTchron.

GDTchron is a Python package for using the outputs of geodynamic models to predict
thermochronometric ages.

Modules:
    visualization: Tools for visualizing VTK files and model thermochronometric ages.
"""

from gdtchron.visualization import add_comp_field, plot_vtk_2d

__all__ = ["plot_vtk_2d", "add_comp_field"]
