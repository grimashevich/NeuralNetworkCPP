#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include "NnBase.h"

#ifndef NN_NN_H
#define NN_NN_H

class NeuralNetwork: public NnBase
{
public:
	explicit NeuralNetwork(const std::vector<int>& Topology);
	void Train(const std::vector<std::vector<double>>& inputs,
			   const std::vector<std::vector<double>>& targets, int numEpochs) override;
	std::vector<double> Predict(const std::vector<double>& input) override;
	void SaveWeight(double accuracy, int epoch) override;
	void LoadWeight(std::string fileName) override;

private:
	std::vector<int> topology;
	std::vector<std::vector<double>> layers;
	std::vector<std::vector<std::vector<double>>> weights;
	std::vector<std::vector<double>> biases;
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

};


#endif //NN_NN_H