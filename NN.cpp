#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <sstream>
#include <fstream>
#include <stdexcept>

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
		learningRate = 0.02;
    }

    void train(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets, int num_epochs) {
        for (int e = 0; e < num_epochs; e++) {
            for (int i = 0; i < inputs.size(); i++) {
                std::vector<std::vector<double>> activations = feedforward(inputs[i]);
                std::vector<std::vector<double>> errors = backprop(activations, targets[i]);
                update_weights(errors, activations);
                update_biases(errors);
            }
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

	[[nodiscard]] double getLearningRate() const
	{
		return learningRate;
	}

	void setLearningRate(double newLearningRate)
	{
		if (learningRate <= 0)
			return;
		learningRate = newLearningRate;
	}

	void saveWeight(double accuracy, int epoch)
	{
		std::string fileName = getFileNameForWeights(accuracy, epoch);
		std::ofstream weightsFile (fileName);
		if (! weightsFile.is_open())
		{
			std::cerr << "Error open file " << fileName << " for save weight";
			return;
		}
		for (int layerSize: topology) {
			weightsFile << layerSize << " ";
		}
		weightsFile << "\n";
		//Save weights
		for (auto & table : weights){
			for (auto & row : table) {
				int i = 0;
				for (double weight : row) {
					if (i++ > 0)
						weightsFile << " ";
					weightsFile << weight;
				}
				weightsFile << "\n";
			}
			weightsFile << "\n";
		}
		//Save biases
		for (auto & layer: biases){
			int i = 0;
			for (double bias: layer){
				if (i++ > 0)
					weightsFile << " ";
				weightsFile << bias;
			}
			weightsFile << "\n";
		}
		weightsFile.close();
	}

	void loadWeight(std::string fileName)
	{
		std::ifstream weightsFile(fileName);
		std::string line;
		getline(weightsFile, line);
		if (!CheckTopology(line))
			throw std::runtime_error("Wrong saved weight topology");
		//TODO here.
	}

private:
    std::vector<int> topology;
    std::vector<std::vector<double>> layers;
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> biases;
	double learningRate;
	std::mt19937 generator;


	static std::vector<std::string> splitString(const std::string& str, char sep)
	{
		std::vector<std::string> result(0);
		std::string tmp;
		std::stringstream ss(str);
		while (getline(ss, tmp, sep))
			result.push_back(tmp);
		return result;
	}

	bool CheckTopology(const std::string& strTopology)
	{
		std::vector<std::string> strTopologyArr = splitString(strTopology, ' ');
		if (strTopologyArr.size() != topology.size())
			return false;
		for (int i = 0; i < strTopologyArr.size(); ++i)
		{
			if (std::stoi(strTopologyArr[i]) != topology[i])
				return false;
		}
		return true;

	}

	std::string getFileNameForWeights(double  accuracy, int epoch)
	{
		std::ostringstream fileName;
		fileName << "NN_weights_";
		for (int i = 0; i < topology.size(); ++i)
		{
			fileName << topology[i];
			if (i < topology.size() - 1)
				fileName << "-";
		}
		fileName << "_epoch-";
		fileName << epoch;
		fileName << "_accuracy-";
		fileName << accuracy;
		return fileName.str();
	}

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
        for (int i = 0; i < weights.size(); i++) {
            for (int j = 0; j < weights[i].size(); j++) {
                for (int k = 0; k < weights[i][j].size(); k++) {
                    double delta = -learningRate * errors[i + 1][j] * activations[i][k];
                    weights[i][j][k] += delta;
                }
            }
        }
    }

    void update_biases(const std::vector<std::vector<double>> &errors) {
        for (int i = 0; i < biases.size(); i++) {
            for (int j = 0; j < biases[i].size(); j++) {
                double delta = -learningRate * errors[i + 1][j];
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
