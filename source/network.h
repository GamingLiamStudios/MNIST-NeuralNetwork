#pragma once

#include <Eigen/Dense>
#include <initializer_list>
#include <vector>
#include <tuple>
#include <optional>
#include <span>

class Network
{
private:
    size_t                          num_layers;
    std::vector<int>                sizes;
    std::vector<Eigen::RowVectorXf> biases;
    std::vector<Eigen::MatrixXf>    weights;

public:
    Network(std::initializer_list<int> sizes);

    Eigen::VectorXf feedforward(Eigen::VectorXf x);
    size_t evaluate(std::span<std::pair<std::vector<float>, std::vector<float>>> test_data);

    std::pair<std::vector<Eigen::VectorXf>, std::vector<Eigen::MatrixXf>>
         backprop(std::vector<float> x, std::vector<float> y);
    void update_mini_batch(
      std::span<std::pair<std::vector<float>, std::vector<float>>> mini_batch,
      float                                                        eta);
    void SGD(
      std::vector<std::pair<std::vector<float>, std::vector<float>>> &            training_data,
      size_t                                                                      epochs,
      size_t                                                                      mini_batch_size,
      float                                                                       eta,
      std::optional<std::span<std::pair<std::vector<float>, std::vector<float>>>> test_data);
};

inline float activation(float x);
inline float activation_prime(float x);
