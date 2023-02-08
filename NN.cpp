#include <iostream>
#include <vector>
#include <cmath>
#include <random>

class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<int>& topology) : topology(topology) {
        int num_layers = topology.size();
        for (int i = 0; i < num_layers; i++) {
            layers.emplace_back(topology[i]);
            if (i != 0) {
                //weights.emplace_back(topology[i], std::vector<std::vector<double>>(topology[i - 1]));
                weights.emplace_back(topology[i], std::vector<double>(topology[i - 1]));
                biases.emplace_back(topology[i]);
            }
        }
        init_weights();
        init_biases();
    }

    void train(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets, int num_epochs) {
        for (int e = 0; e < num_epochs; e++) {
			auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < inputs.size(); i++) {
                std::vector<std::vector<double>> activations = feedforward(inputs[i]);
                std::vector<std::vector<double>> errors = backprop(activations, targets[i]);
                update_weights(errors, activations);
                update_biases(errors);
            }
			auto end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
			std::cout << "epoch # " << e + 1 << " complete (" << duration.count() << " sec.)" << std::endl;
        }
    }

    std::vector<double> predict(const std::vector<double>& input) {
        std::vector<double> output = feedforward(input).back();
        double sum = 0;
        for (int i = 0; i < output.size(); i++) {
            output[i] = exp(output[i]);
            sum += output[i];
        }
        for (int i = 0; i < output.size(); i++) {
            output[i] /= sum;
        }
        return output;
    }

private:
    std::vector<int> topology;
    std::vector<std::vector<double>> layers;
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> biases;
    std::mt19937 generator;

    void init_weights() {
        std::normal_distribution<double> distribution(0, 1);
        for (int i = 0; i < weights.size(); i++) {
            for (int j = 0; j < weights[i].size(); j++) {
                for (int k = 0; k < weights[i][j].size(); k++) {
                    weights[i][j][k] = distribution(generator);
                }
            }
        }
    }

    void init_biases() {
        std::normal_distribution<double> distribution(0, 1);
        for (int i = 0; i < biases.size(); i++) {
            for (int j = 0; j < biases[i].size(); j++) {
                biases[i][j] = distribution(generator);
            }
        }
    }

    std::vector<std::vector<double>> feedforward(const std::vector<double>& input)
    {
        layers[0] = input;
        for (int i = 1; i < topology.size(); i++) {
            for (int j = 0; j < layers[i].size(); j++) {
                double sum = biases[i - 1][j];
                for (int k = 0; k < layers[i - 1].size(); k++) {
                    sum += layers[i - 1][k] * weights[i - 1][j][k];
                }
                layers[i][j] = sigmoid(sum);
            }
        }
        return layers;
    }

    std::vector<std::vector<double>> backprop(const std::vector<std::vector<double>> &activations, const std::vector<double> &target) {
        std::vector<std::vector<double>> errors(topology.size());
        int output_layer = topology.size() - 1;
        for (int i = 0; i < topology[output_layer]; i++) {
            errors[output_layer].emplace_back(activations[output_layer][i] - target[i]);
        }
        for (int i = output_layer - 1; i > 0; i--) {
            for (int j = 0; j < topology[i]; j++) {
                double sum = 0;
                for (int k = 0; k < topology[i + 1]; k++) {
                    sum += errors[i + 1][k] * weights[i][k][j];
                }
                errors[i].emplace_back(sum * dsigmoid(activations[i][j]));
            }
        }
        return errors;
    }

    void update_weights(const std::vector<std::vector<double>> &errors, const std::vector<std::vector<double>> &activations) {
        double learning_rate = 0.02;
        for (int i = 0; i < weights.size(); i++) {
            for (int j = 0; j < weights[i].size(); j++) {
                for (int k = 0; k < weights[i][j].size(); k++) {
                    double delta = -learning_rate * errors[i + 1][j] * activations[i][k];
                    weights[i][j][k] += delta;
                }
            }
        }
    }

    void update_biases(const std::vector<std::vector<double>> &errors) {
        double learning_rate = 0.1;
        for (int i = 0; i < biases.size(); i++) {
            for (int j = 0; j < biases[i].size(); j++) {
                double delta = -learning_rate * errors[i + 1][j];
                biases[i][j] += delta;
            }
        }
    }

    double sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }

    double dsigmoid(double x) {
        return x * (1 - x);
    }
};
