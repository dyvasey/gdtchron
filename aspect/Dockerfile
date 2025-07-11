FROM condaforge/miniforge3 AS miniforge
WORKDIR /scratch
COPY ./aspect/environment.yml .
COPY ./gdtchron ./gdtchron
COPY ./pyproject.toml .
COPY ./README.md .
RUN apt-get update -y && apt-get install -y build-essential
RUN mamba env create -f environment.yml \
     && mamba clean --all --yes

FROM geodynamics/aspect:v3.0.0
USER root

RUN apt-get update -y && apt-get install -y libgl1-mesa-glx xvfb
COPY --from=miniforge /opt/conda/envs/aspect-docker /opt/conda/envs/aspect-docker
ENV PATH=/opt/conda/envs/aspect-docker/bin:$PATH

WORKDIR /workspace
COPY ./aspect/*.ipynb ./aspect/
COPY ./aspect/uplift_box.prm ./aspect/
COPY ./notebooks/*.ipynb ./notebooks/
EXPOSE 8888

ENV OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

CMD jupyter lab --ip=0.0.0.0 --port=8888 --allow-root  --no-browser --NotebookApp.token='' --NotebookApp.disable_check_xsrf=True
