#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <mutex>
#include <thread>
#include "NeuralNetworkBase.h"
#include "StopWatch.h"

#ifndef NN_NN_H
#define NN_NN_H

class MatrixNeuralNetwork: public NeuralNetworkBase
{
public:
	explicit MatrixNeuralNetwork(const std::vector<int>& Topology);
	double Train(const std::vector<std::vector<double>>& inputs,
				 const std::vector<std::vector<double>>& targets, int numEpochs) override;
    double TrainOneBatch(const std::vector<std::vector<double>> &inputs,
                         const std::vector<std::vector<double>> &targets, int batchStart,
                         int batchSize, std::mutex &m) override;
	std::vector<double> Predict(const std::vector<double>& input) override;
	void SaveWeights(double accuracy, int epoch, std::string fName) override;
	void LoadWeight(std::string fileName) override;

    std::vector<double> PredictMT(const std::vector<double>& input,
                                  std::vector<std::vector<double>> & layersCopy);
    std::vector<std::vector<double>> ForwardFeedMT(const std::vector<double>& input,
                                                   std::vector<std::vector<double>> & layersCopy);

	std::vector<std::vector<std::vector<double>>> weights;
	std::vector<std::vector<double>> biases;
	//std::vector<std::vector<double>> layers;

private:
	std::vector<int> topology;

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

    void addVectorValues(std::vector<std::vector<double>> &source, std::vector<std::vector<double>> &summation);

    void divideEachElement(std::vector<std::vector<double>> &source, double divider);

    void ForwardFeedMTBatch(const std::vector<double> &input, std::vector<std::vector<double>> &sumLayersValue);

    double
    trainWithMiniBatches(const std::vector<std::vector<double>> &inputs,
                         const std::vector<std::vector<double>> &targets,
                         double batchSize, int threadsCount);

    void TrainOneBatchLauncher(const std::vector<std::vector<double>> &inputs,
                          const std::vector<std::vector<double>> &targets,
                          int startFrom, int batchSize, int batchCount, std::mutex &m);
};


#endif //NN_NN_H
