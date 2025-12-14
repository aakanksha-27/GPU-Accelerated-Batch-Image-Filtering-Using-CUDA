NVCC = nvcc
CXXFLAGS = -std=c++17
OPENCV = `pkg-config opencv4 --cflags --libs`

all:
	$(NVCC) src/main.cu src/mnist_reader.cpp \
	-o mnist_gpu $(CXXFLAGS) $(OPENCV)

clean:
	rm -f mnist_gpu data/output/*.png
