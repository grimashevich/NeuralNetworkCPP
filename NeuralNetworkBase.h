#ifndef NN_NEURALNETWORKBASE_H
#define NN_NEURALNETWORKBASE_H

#include <vector>
#include <random>
#include <iostream>
#include <string>
#include <cstdio>
#include <ctime>
#include <mutex>

class NeuralNetworkBase
{
public:
	virtual double Train(const std::vector<std::vector<double>>& inputs,
						 const std::vector<std::vector<double>>& targets,
						 int numEpochs) = 0;

	virtual std::vector<double> Predict(const std::vector<double>& input) = 0;

    virtual std::vector<double> PredictMT(const std::vector<double>& input,
                                          std::vector<std::vector<double>> & layersCopy) = 0;
    virtual std::vector<std::vector<double>> ForwardFeedMT(
            const std::vector<double>& input,
            std::vector<std::vector<double>> & layersCopy) = 0;

	virtual void SaveWeights(double accuracy, int epoch, std::string fName) = 0;
	virtual void LoadWeight(std::string fileName) = 0;

	void SetLearningRate(double newLearningRate);
	static double Sigmoid(double x);
	static double DSigmoid(double x);

	[[nodiscard]] double GetLearningRate() const;

	virtual ~NeuralNetworkBase();
	static const std::string currentDateTime();

    std::vector<std::vector<double>> layers;

protected:
	double learningRate{0.02};

};

#endif //NN_NEURALNETWORKBASE_H
