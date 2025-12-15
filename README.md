# GPU-Accelerated-Batch-Image-Filtering-Using-CUDA

## Overview
This project demonstrates GPU-accelerated batch image processing using CUDA.
The pipeline processes hundreds of small images in a single execution, applying a Gaussian blur using NVIDIA NPP followed by Sobel edge detection using a custom CUDA kernel.

The focus of this project is batch throughput on the GPU, rather than single-image acceleration.

## Dataset
The project uses a subset of the MNIST handwritten digits dataset, consisting of grayscale images.
A batch of 500 images is processed per run to demonstrate GPU parallelism on many small inputs.

Dataset Link: https://archive-beta.ics.uci.edu/dataset/683/mnist+database+of+handwritten+digits

## GPU Computation
The following GPU operations are performed:
- Gaussian blur using NVIDIA NPP (nppiFilterGauss)
- Custom CUDA Sobel edge detection kernel
- Batch processing entirely on GPU memory

All filtering operations are executed on the GPU, with minimal host–device transfers.

## How to Build
make

## How to Run
./run.sh

Note: This project was executed and tested in a Google Colab CUDA environment.
The run.sh script documents the expected execution flow.

## Output
Processed images are written to data/output/.
Execution logs and timing information are saved in logs/run_log.txt.

## Design Decisions and Lessons Learned

This project was designed to demonstrate batch GPU image processing
rather than single-image acceleration. MNIST was chosen because it
contains hundreds of small images, which allows efficient batching
and highlights GPU throughput advantages.

A CUDA-accelerated pipeline was implemented using CuPy to:
- Transfer image batches to GPU memory
- Apply element-wise and convolution-style operations in parallel
- Return results back to host for visualization

Key challenges included:
- Ensuring GPU memory transfers were batched efficiently
- Verifying GPU execution in a non-native environment (Colab)
- Balancing simplicity with meaningful computation

This project reinforced the importance of batching workloads on GPUs
and understanding memory transfer costs relative to computation.

## Proof of GPU Execution

All experiments were executed on Google Colab with an NVIDIA GPU.
Screenshots showing GPU availability, code execution, and image filtering
results are available in the `screenshots/` directory.

The project processes hundreds of MNIST images in a single execution
using CUDA-accelerated operations via CuPy.

Although only one sample image is visualized, the program processes
the full MNIST dataset batch (500+ images) in a single GPU execution.
