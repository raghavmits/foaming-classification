# # Use an official Python runtime as a parent image
# FROM python:3.9-slim

# # Set the working directory in the container
# WORKDIR /app

# # Install dependencies (including system packages for Conda and libraries like OpenBLAS, JPEG, PNG)
# RUN apt-get update && apt-get install -y \
#     libopenblas-dev \
#     libjpeg-dev \
#     libpng-dev \
#     wget \
#     bzip2 \
#     && rm -rf /var/lib/apt/lists/*

# # Install Miniconda (needed to manage environments)
# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
#     bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
#     rm Miniconda3-latest-Linux-x86_64.sh

# # Set environment variable for Conda
# ENV PATH /opt/conda/bin:$PATH

# # Copy the environment.yml into the container
# COPY environment.yml /app/

# # Create the Conda environment and install dependencies
# RUN conda env create -f environment.yml && \
#     conda clean --all -f -y

# # Copy the current directory contents into the container at /app
# COPY . /app/

# # Set environment variable to use the Conda environment
# ENV PATH /opt/conda/envs/foaming-backend/bin:$PATH

# # Expose port 8000 to the outside world
# EXPOSE 8000

# # Run FastAPI app with Uvicorn
# CMD ["uvicorn", "model_backend:app", "--host", "0.0.0.0", "--port", "8000"]

# Use an official Miniconda base image to ensure the Python environment is isolated
FROM continuumio/miniconda3:4.12.0

# Set the working directory in the container
WORKDIR /app

# Copy only the environment.yml file to take advantage of Docker caching
COPY environment.yml /app/

# Create the Conda environment (be sure to activate it explicitly here)
RUN conda env create -f environment.yml && \
    conda clean --all -f -y

# Set the environment path to the Conda environment
# This ensures that the 'foaming-backend' environment is automatically activated for future commands
ENV PATH /opt/conda/envs/foaming-backend/bin:$PATH

# Copy the application code into the container
COPY . /app/

# Expose port 8000 for the FastAPI app (ensure the app listens on this port)
EXPOSE 8000

# Use `conda run` to run the app in the Conda environment. This avoids needing a shell.
CMD ["bash", "-c", "source activate foaming-backend && uvicorn model_backend:app --host 0.0.0.0 --port 8000"]