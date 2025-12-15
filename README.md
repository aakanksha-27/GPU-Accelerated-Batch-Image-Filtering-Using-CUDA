# GPU-Accelerated-Batch-Image-Filtering-Using-CUDA

## Overview
This project demonstrates GPU-accelerated image processing on hundreds of small images using CUDA. The pipeline applies Gaussian blur using NVIDIA NPP and Sobel edge detection using a custom CUDA kernel.

## Dataset
We use a subset of the MNIST handwritten digits dataset (grayscale images, 256x256), processing 500 images in a single execution.

Link: https://archive-beta.ics.uci.edu/dataset/683/mnist+database+of+handwritten+digits

## GPU Computation
- NPP Gaussian blur (nppiFilterGauss)
- Custom CUDA Sobel edge detection kernel
- Batch processing on GPU memory

## How to Build
make

## How to Run
./run.sh

## Output
Processed images are written to data/output/.
Execution logs and timing information are saved in logs/run_log.txt.

## Lessons Learned
- Managing device memory for batch image processing
- Combining CUDA libraries with custom kernels
- Performance benefits of GPU parallelism for image workloads

## Proof of GPU Execution

All experiments were executed on Google Colab with an NVIDIA GPU.
Screenshots showing GPU availability, code execution, and image filtering
results are available in the `screenshots/` directory.

The project processes hundreds of MNIST images in a single execution
using CUDA-accelerated operations via CuPy.
