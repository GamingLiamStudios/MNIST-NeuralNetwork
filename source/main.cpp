#include <iostream>

#include "network.h"
#include "MNIST/mnist.h"

int main()
{
    // Read Training Data
    std::vector<std::pair<std::vector<float>, std::vector<float>>> training_data;
    {
        int  noi, soi, nol;
        auto images = MNIST::read_mnist_images("MNIST/train-images.idx3-ubyte", noi, soi);
        auto labels = MNIST::read_mnist_labels("MNIST/train-labels.idx1-ubyte", nol);

        for (int i = 0; i < noi; i++)
        {
            std::vector<float> images_vec, labels_vec;

            for (int j = 0; j < soi; j++) images_vec.push_back(images[i][j] / 255.0f);
            for (int j = 0; j < 10; j++) labels_vec.push_back(float(labels[i] == j));

            training_data.push_back(std::make_pair(std::move(images_vec), std::move(labels_vec)));
            delete[] images[i];
        }
        delete[] images;
        delete[] labels;
    }

    // Read Test Data
    std::vector<std::pair<std::vector<float>, std::vector<float>>> test_data;
    {
        int  noi, soi, nol;
        auto images = MNIST::read_mnist_images("MNIST/t10k-images.idx3-ubyte", noi, soi);
        auto labels = MNIST::read_mnist_labels("MNIST/t10k-labels.idx1-ubyte", nol);

        for (int i = 0; i < noi; i++)
        {
            std::vector<float> images_vec, labels_vec;

            for (int j = 0; j < soi; j++) images_vec.push_back(images[i][j] / 255.0f);
            for (int j = 0; j < 10; j++) labels_vec.push_back(float(labels[i] == j));

            test_data.push_back(std::make_pair(std::move(images_vec), std::move(labels_vec)));

            delete[] images[i];
        }
        delete[] images;
        delete[] labels;
    }
    std::cout << "Loaded MNIST\n";

    auto net = Network({ 784, 30, 10 });
    std::cout << "After 0 Epochs: " << net.evaluate(test_data) << " / " << test_data.size() << "\n";
    net.SGD(training_data, 30, 10, 3.0f, test_data);
    std::cout << "After 30 Epochs: " << net.evaluate(test_data) << " / " << test_data.size()
              << "\n";

    return 0;
}