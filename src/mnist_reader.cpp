#include "mnist_reader.h"
#include <fstream>
#include <iostream>
#include <cstdint>

static uint32_t read_be_uint32(std::ifstream& f) {
    uint32_t x = 0;
    f.read(reinterpret_cast<char*>(&x), 4);
    return __builtin_bswap32(x);
}

MNISTImages load_mnist_images(const std::string& filename, int max_images) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open MNIST file");
    }

    uint32_t magic = read_be_uint32(file);
    uint32_t num_images = read_be_uint32(file);
    uint32_t rows = read_be_uint32(file);
    uint32_t cols = read_be_uint32(file);

    if (magic != 2051) {
        throw std::runtime_error("Invalid MNIST image file");
    }

    int images_to_read = std::min((int)num_images, max_images);

    MNISTImages result;
    result.num_images = images_to_read;
    result.rows = rows;
    result.cols = cols;
    result.data.resize(images_to_read * rows * cols);

    for (int i = 0; i < images_to_read; i++) {
        for (int j = 0; j < rows * cols; j++) {
            unsigned char pixel;
            file.read(reinterpret_cast<char*>(&pixel), 1);
            result.data[i * rows * cols + j] = pixel / 255.0f;
        }
    }

    std::cout << "Loaded " << images_to_read
              << " MNIST images (" << rows << "x" << cols << ")\n";

    return result;
}
