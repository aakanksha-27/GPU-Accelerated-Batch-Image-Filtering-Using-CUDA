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
Filtered images are written to: data/output/
Execution logs and timing information are saved in: logs/run_log.txt

Although only one representative output image is visualized, the pipeline processes** all images in the batch** during execution.

## Design Decisions and Lessons Learned

This project was designed to demonstrate **batch GPU image processing**, rather than single-image acceleration. MNIST was selected because it contains hundreds of small images, which allows efficient batching and highlights GPU throughput advantages.

A CUDA-accelerated pipeline was implemented using CuPy to:
- Performing Gaussian filtering using NVIDIA NPP to leverage optimized GPU primitives
- Implementing a custom CUDA kernel for Sobel edge detection to demonstrate kernel-level GPU programming
- Batching image transfers to reduce PCIe overhead

Key challenges included:
- Ensuring GPU memory transfers were batched efficiently between host and device
- Verifying GPU execution in a non-native environment (Google Colab)
- Balancing simplicity with meaningful GPU computation

This project reinforced the importance of **batching workloads on GPUs** and understanding the performance trade-offs between computation and memory transfer.

## Proof of GPU Execution

All experiments were executed on **Google Colab with an NVIDIA GPU**.
The repository includes screenshots showing:
- GPU availability (nvidia-smi)
- Successful CUDA execution
- Input and output image filtering results
These artifacts are available in the screenshots/ directory.

Although only one sample image is visualized, the program processes **entire batch of 500+ MNIST images** in a single GPU execution using CUDA-accelerated operations.
