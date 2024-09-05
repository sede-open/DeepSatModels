# Take the baseimage from microsoft instead of the pytorch image provided by mmrotate
FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04:latest

# Take rest of dockerfile from https://github.com/open-mmlab/mmrotate/blob/main/docker/Dockerfile
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV FORCE_CUDA="1"

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python Dependencies
# Note AzureML starts `sh`, not `bash`
COPY deepsatmodels_env.yml .
RUN conda env create -f deepsatmodels_env.yml && \
    conda clean --all --yes && \
    echo "source activate deepsatmodels" >> ~/.bashrc && \
    echo "source activate deepsatmodels" >> ~/.profile && \
    echo "source activate deepsatmodels" >> /etc/profile.d/conda.sh

# Ensure that bash is the default shell and bashrc is sourced
ENV PATH /opt/miniconda/envs/deepsatmodels/bin:$PATH
ENV AZUREML_CONDA_ENVIRONMENT_PATH /opt/miniconda/envs/deepsatmodels
ENV LD_LIBRARY_PATH /opt/miniconda/envs/deepsatmodels/lib:$LD_LIBRARY_PATH
ENV CONDA_DEFAULT_ENV deepsatmodels
ENV CONDA_PREFIX /opt/miniconda/envs/deepsatmodels

# set command
CMD ["conda", "run", "--no-capture-output", "-n", "deepsatmodels", "bash"]
