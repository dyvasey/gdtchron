<div align="center">
    <img src="media/logo.png" alt="GDTchron Logo" width="300">
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

Online documentation is available at https://gdtchron.readthedocs.io/en/latest/

## Installation
GDTchron is not yet hosted on PyPI, so it can currently only be installed by cloning this repository and installing with pip:
```
git clone https://github.com/dyvasey/gdtchron.git
cd gdtchron
pip install .
```

## Running GDTchron with Binder
Clicking the Binder badge at the top of this README will launch an interactive JupyterLab environment hosted by Binder with GDTchron installed. This is a good way to try out the functionality of GDTchron without needing to deal with a local Python installation. Note that the Binder environment does not have ASPECT installed.

## Running GDTchron with ASPECT via Docker
Included in this repository is a Dockerfile allowing you to create an interactive JupyterLab environment that can run both ASPECT and GDTchron in Jupyter Notebooks.

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

## Contributing to GDTchron
GDTchron is designed to be a community-driven, open-soure Python package. If you have code you would like to contribute, please see the [contributing guidelines](CONTRIBUTING.md).
