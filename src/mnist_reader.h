#pragma once
#include <vector>
#include <string>

struct MNISTImages {
    int num_images;
    int rows;
    int cols;
    std::vector<float> data;  // normalized [0,1]
};

MNISTImages load_mnist_images(
    const std::string& filename,
    int max_images
);
