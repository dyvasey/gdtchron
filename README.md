<div align="center">
    <img src="https://raw.githubusercontent.com/dyvasey/gdtchron/main/media/logo.png" alt="GDTchron Logo" width="300">
</div>

# GDTchron: Geodynamic Thermochronology
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/dyvasey/gdtchron/HEAD)
[![Online Documentation](https://readthedocs.org/projects/gdtchron/badge/?version=latest)](https://gdtchron.readthedocs.io/en/latest/)

## About
GDTchron is a Python package for using the outputs of geodynamic models to predict thermochronometric ages.

Current authors:
* Dylan Vasey
* Peter Scully
* John Naliboff

Source code: https://github.com/dyvasey/gdtchron

Online documentation: https://gdtchron.readthedocs.io/en/latest/

The documentation consists of Jupyter Notebooks demonstrating use of the code and the full API for the code.
## Installation
The latest release of GDTchron can be installed from PyPI or conda-forge using `pip` or `conda` as package managers:
```
# PyPI
pip install gdtchron
```
```
# conda-forge
conda install -c conda-forge gdtchron
```

For the latest development version, you can clone and install the GitHub repository with the source code. This repository also includes all the tests and Jupyter Notebooks.
```
git clone https://github.com/dyvasey/gdtchron.git
cd gdtchron
pip install .
```
Once installed, GDTchron can be used like any other Python package in scripts or Jupyter Notebooks.
## Running GDTchron with Binder
Clicking the Binder badge at the top of this README will launch an interactive JupyterLab environment hosted by Binder with GDTchron installed. This is a good way to try out the functionality of GDTchron without needing to deal with a local Python installation. Note that the Binder environment does not have ASPECT installed.

## Running GDTchron with ASPECT via Docker
Included in this repository is a Dockerfile allowing you to create an interactive JupyterLab environment that can run both ASPECT and GDTchron in Jupyter Notebooks. This environment allows you to fully run an accompanying ASPECT uplift model, process it using GDTChron, and plot the results using the Jupyter Notebooks in the `aspect` directory. Note that fully replicating this process may take several hours.

See here for how to install Docker: https://docs.docker.com/get-started/

To build the environment, first ensure Docker is running. Then, from the repository root directory run:
```
docker build -f aspect/Dockerfile -t aspect-docker .
```
This may take a few minutes. To then run the environment:
```
docker run --rm --name aspect-docker -d -p 8888:8888 aspect-docker
```
The build and run commands are also provided in the shell script `aspect/aspect_docker.sh`

Once the environment is running, navigate to http://localhost:8888 in your browser.

To stop the environment, run:
```
docker stop aspect-docker
```
Note that the `--rm` flag in `docker run` means that the environment (including all files saved in it) will be removed once stopped (including if the session running the environment is ended). You can make it possible to restart the environment by omitting this flag, but this means that you will have to remove the environment manually when you are done with it.

To replicate the ASPECT uplift model and its results, run in order `run_aspect_model.ipynb`, `process_model.ipynb`, and `figures_model.ipynb`. The outputs of these are currently saved in the notebooks and viewable in the documentation.

## Jupyter Notebooks
There are additional Jupyter Notebooks in the `notebooks` directory and displayed in the documentation demonstrating the functionality of GDTchron. Some of these notebooks can be directly replicated with just a GDTchron install (or in the Binder environment), whereas others depend on local output model files that are too large to include in this repository and are displayed as demonstrations only.

### Fully Reproducible with GDTchron Install or Binder
* `tchron_demo.ipynb`
* `scaling_test.ipynb`

### Not Reproducible without Large Size Local Files
* `process_riftinversion.ipynb`
* `figure_riftinversion.ipynb`
* `interpolation_comparison.ipynb`

The ASPECT parameter files needed to reproduce the rift inversion model for these notebooks are available in the `ri_prms` directory.

## Contributing to GDTchron
GDTchron is designed to be a community-driven, open-source Python package. If you have code you would like to contribute, please see the [contributing guidelines](CONTRIBUTING.md).
