#include "mnist_reader.h"
#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char** argv) {
    std::string mnist_path = "data/mnist/train-images-idx3-ubyte";
    int max_images = 5;

    MNISTImages images = load_mnist_images(mnist_path, max_images);

    int H = images.rows;
    int W = images.cols;

    for (int i = 0; i < images.num_images; i++) {
        cv::Mat img(H, W, CV_32F,
            images.data.data() + i * H * W);

        cv::Mat img8u;
        img.convertTo(img8u, CV_8U, 255.0);

        std::string filename =
            "data/output/img_" + std::to_string(i) + ".png";

        cv::imwrite(filename, img8u);
    }

    std::cout << "Saved PNG images\n";
    return 0;
}
