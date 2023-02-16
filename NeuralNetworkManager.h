#ifndef NN_NEURALNETWORKMANAGER_H
#define NN_NEURALNETWORKMANAGER_H


#include "NeuralNetworkBase.h"
#include "MatrixNeuralNetwork.h"
#include "DataSet.h"
#include "StopWatch.h"
#include <filesystem>

class NeuralNetworkManager
{
public:
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
	void LoadWeightToNetwork(const std::string& fileName);
	void SaveWeightFromNetwork(double curAccuracy, size_t epochNum, const std::string& alterFileName);
	size_t Predict(std::vector<double> inputSignal, int answerOffset, bool needNormalize);
	~NeuralNetworkManager();


	DataSet *trainingSet = nullptr;

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

    static bool fileExistAndReadable(const std::string &fileName);

	void CrutchNormalzation(std::vector<double> & signal);
};


#endif //NN_NEURALNETWORKMANAGER_H
