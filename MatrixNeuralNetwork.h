#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include "NeuralNetworkBase.h"

#ifndef NN_NN_H
#define NN_NN_H

class MatrixNeuralNetwork: public NeuralNetworkBase
{
public:
	explicit MatrixNeuralNetwork(const std::vector<int>& Topology);
	double Train(const std::vector<std::vector<double>>& inputs,
				 const std::vector<std::vector<double>>& targets, int numEpochs) override;
	std::vector<double> Predict(const std::vector<double>& input) override;
	void SaveWeight(double accuracy, int epoch) override;
	void LoadWeight(std::string fileName) override;

	std::vector<std::vector<std::vector<double>>> weights;
	std::vector<std::vector<double>> biases;

private:
	std::vector<int> topology;
	std::vector<std::vector<double>> layers;

	std::mt19937 generator;

	static std::vector<std::string> SplitString(const std::string& str, char sep);
	bool CheckTopology(const std::string& strTopology);
	std::string GetFileNameForWeights(double  accuracy, int epoch);
	void InitWeights();
	void InitBiases();
	std::vector<std::vector<double>> ForwardFeed(const std::vector<double>& input);
	std::vector<std::vector<double>> BackProp(const std::vector<std::vector<double>> &activations,
											  const std::vector<double> &target);
	void UpdateWeights(const std::vector<std::vector<double>> &errors, const std::vector<std::vector<double>> &activations);
	void UpdateBiases(const std::vector<std::vector<double>> &errors);

	double GetMeanError(std::vector<double> & errors) const;

};


#endif //NN_NN_H
