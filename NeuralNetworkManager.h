#ifndef NN_NEURALNETWORKMANAGER_H
#define NN_NEURALNETWORKMANAGER_H


#include "NeuralNetworkBase.h"
#include "MatrixNeuralNetwork.h"
#include "DataSet.h"
#include "StopWatch.h"
#include <filesystem>
#include <thread>
#include <mutex>
#include <map>

class NeuralNetworkManager
{
public:
	//TODO move to private
    DataSet *trainingSet = nullptr;

	NeuralNetworkManager();
	void LoadMatrixNN(const std::vector<int> &topology);
	void LoadGraphNN();
	double GetAccuracy() const;
	double getPrecision() const;
	double getRecall() const;
	double getFMeasure() const;
	double getError() const;
	float GetValidationPartOfTrainingDataset() const;
	void SetValidationPartOfTrainingDataset(float newValue);
	void LoadTrainSet(std::string & fileName, size_t inputSize, size_t outputSize, size_t objectLimit = 0);
	void Train(int numEpochs, double learningRate);
    void trainWithMiniBatches(double learningRate, double batchSize, int threadsCount);
	void LoadWeightToNetwork(const std::string& fileName);
	void SaveWeightFromNetwork(double curAccuracy, size_t epochNum, const std::string& alterFileName);
	size_t Predict(std::vector<double> inputSignal, int answerOffset, bool needNormalize);
	void CalculateMetricsForTestSet(const std::vector<std::vector<double>> &testInputs,
									const std::vector<std::vector<double>> &testTargets,
                                    size_t threadsNum = 2);
	static void PredictMT(const std::vector<std::vector<double>> &inputs,
						  const std::vector<std::vector<double>> &targets,
						  size_t fromIndex, size_t toIndex, int answerOffset,
						  bool needNormalize, std::vector<std::vector<size_t>> & result,
						  NeuralNetworkBase *nn,
                          std::mutex & m);

    void CrossValidation(size_t folds_count);
    ~NeuralNetworkManager();

private:
	int inputSizeNN = 0;
	int outputSizeNN = 0;

	//Neural network and datasets
	NeuralNetworkBase *neuralNetwork = nullptr;

	DataSet *testSet = nullptr;

	//Training and dataset setting
	float validationPartOfTrainingDataset = 0.2;
	size_t trainDatasetObjectLimit = 0;

	// Metrics
	double accuracy = 0;
	double precision = 0;
	double recall = 0;
	double fMeasure = 0;
	double error = 0;

    std::vector<std::vector<size_t>> predict_matrix;

    static bool fileExistAndReadable(const std::string &fileName);
	void CrutchNormalzation(std::vector<double> & signal);
	static std::vector<int> getTopologyFromWeightsFile(const std::string &weightsFileName);

  static std::map<std::string, double> GetConfusionMatrix(const std::vector<std::vector<size_t>> &test_matrix);

  static void print_matrix(const std::vector<std::vector<size_t>> &vec);
};


#endif //NN_NEURALNETWORKMANAGER_H
