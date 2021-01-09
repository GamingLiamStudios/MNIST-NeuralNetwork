#include "network.h"

#include <algorithm>
#include <iostream>

// Activation Function
float activation(float x)
{
    return 0.5 * (x / (1 + abs(x)) + 1);
}
float activation_prime(float x)
{
    return activation(x) * (1 - activation(x));
}

Network::Network(std::initializer_list<int> sizes)
{
    // Initalize variables
    this->sizes      = std::vector<int>(sizes);
    this->num_layers = sizes.size();
    this->biases.resize(num_layers - 1);
    this->weights.resize(num_layers - 1);

    // Create Matrices
    for (size_t i = 1; i < num_layers; i++)
    {
        biases.at(i - 1)  = Eigen::VectorXf::Random(this->sizes[i]);
        weights.at(i - 1) = Eigen::MatrixXf::Random(this->sizes[i - 1], this->sizes[i]);
    }
}

Eigen::VectorXf Network::feedforward(Eigen::VectorXf a)
{
    for (size_t i = 0; i < this->num_layers - 1; i++)
        a = ((this->weights[i] * a) + this->biases[i]).unaryExpr(std::ptr_fun(::activation));
    return a;
}

size_t Network::evaluate(std::span<std::pair<std::vector<float>, std::vector<float>>> test_data)
{
    size_t          sum;
    Eigen::Index    index;
    Eigen::VectorXf x;
    for (size_t i = 0; i < test_data.size(); i++)
    {
        index = 0;
        x     = Eigen::VectorXf::Zero(test_data[i].first.size());
        memcpy(&x(0), test_data[i].first.data(), test_data[i].first.size());

        Eigen::VectorXf ff = feedforward(x);
        ff.maxCoeff(&index);
        sum += test_data[i].second[index];
    }
    return sum;
}

void Network::SGD(
  std::vector<std::pair<std::vector<float>, std::vector<float>>> &            training_data,
  size_t                                                                      epochs,
  size_t                                                                      mini_batch_size,
  float                                                                       eta,
  std::optional<std::span<std::pair<std::vector<float>, std::vector<float>>>> test_data)
{
    for (size_t i = 0; i < epochs; i++)
    {
        std::random_shuffle(training_data.begin(), training_data.end());
        for (size_t j = 0; j < training_data.size(); j += mini_batch_size)
            update_mini_batch(
              std::span<std::pair<std::vector<float>, std::vector<float>>>(
                training_data.begin() + j,
                training_data.begin() + std::min(j + mini_batch_size, training_data.size())),
              eta);
        if (test_data.has_value())
            std::cout << "Epoch " << i << ": " << evaluate(test_data.value()) << " / "
                      << test_data.value().size() << "\n";
        else
            std::cout << "Epoch " << i << " complete\n";
    }
}

void Network::update_mini_batch(
  std::span<std::pair<std::vector<float>, std::vector<float>>> mini_batch,
  float                                                        eta)
{
    std::vector<Eigen::VectorXf> nabla_b;
    std::vector<Eigen::MatrixXf> nabla_w;

    nabla_b.resize(this->num_layers - 1);
    nabla_w.resize(this->num_layers - 1);

    for (size_t i = 0; i < this->num_layers - 1; i++)
    {
        nabla_b[i] = Eigen::VectorXf::Zero(this->biases[i].size());
        nabla_w[i] = Eigen::MatrixXf::Zero(this->weights[i].rows(), this->weights[i].cols());
    }

    for (std::pair<std::vector<float>, std::vector<float>> pair : mini_batch)
    {
        auto delta_nabla = backprop(pair.first, pair.second);
        for (size_t i = 0; i < this->num_layers - 1; i++)
        {
            nabla_b[i] = nabla_b[i] + delta_nabla.first[i];
            nabla_w[i] = nabla_w[i] + delta_nabla.second[i];
        }
    }

    for (size_t i = 0; i < this->num_layers - 1; i++)
    {
        this->weights[i] =
          this->weights[i] - ((eta / mini_batch.size()) * nabla_w[i].array()).matrix();
        this->biases[i] =
          this->biases[i] - ((eta / mini_batch.size()) * nabla_b[i].array()).matrix();
    }
}

std::pair<std::vector<Eigen::VectorXf>, std::vector<Eigen::MatrixXf>>
  Network::backprop(std::vector<float> x, std::vector<float> y)
{
    std::vector<Eigen::VectorXf> nabla_b;
    std::vector<Eigen::MatrixXf> nabla_w;

    nabla_b.resize(this->num_layers - 1);
    nabla_w.resize(this->num_layers - 1);

    // Feed Forward
    Eigen::VectorXf activation = Eigen::VectorXf::Zero(x.size());
    memcpy(&activation(0), x.data(), x.size());
    std::vector<Eigen::VectorXf> activations = { activation };
    std::vector<Eigen::VectorXf> zs;

    for (size_t i = 0; i < this->num_layers - 1; i++)
    {
        zs.push_back(((this->weights[i] * activation) + this->biases[i]));
        activation = zs.back().unaryExpr(std::ptr_fun(::activation));
        activations.push_back(activation);
    }

    // Backward Pass
#define delta activation    // Rename activation as delta
    delta = Eigen::VectorXf::Zero(y.size());
    memcpy(&delta(0), y.data(), y.size());    // delta = y
    delta = ((activations.back() - delta).array() *
             zs.back().unaryExpr(std::ptr_fun(::activation_prime)).array())
              .matrix();
    nabla_b.back() = delta;
    nabla_w.back() = delta * activations[this->num_layers - 2].transpose();
    for (size_t l = 2; l < num_layers; l++)
    {
        delta = ((this->weights[num_layers - (l + 1)].transpose() * delta).array() *
                 zs[num_layers - l].unaryExpr(std::ptr_fun(::activation_prime)).array())
                  .matrix();
        nabla_b[num_layers - 1 - l] = delta;
        nabla_w[num_layers - 1 - l] = delta * activations[num_layers - (l - 1)].transpose();
    }
#undef delta

    return std::make_pair(std::move(nabla_b), std::move(nabla_w));
}
