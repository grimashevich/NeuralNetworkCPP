#ifndef NN_NEURALNETWORKMANAGER_H
#define NN_NEURALNETWORKMANAGER_H


#include "NeuralNetworkBase.h"
#include "MatrixNeuralNetwork.h"
#include "DataSet.h"
#include "StopWatch.h"

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

	void Train(int numEpochs);

	~NeuralNetworkManager();

private:
	int inputSizeNN = 0;
	int outputSizeNN = 0;

	//Neural network and datasets
	NeuralNetworkBase *neuralNetwork = nullptr;
	DataSet *trainingSet = nullptr;
	DataSet *testSet = nullptr;

	//Training and dataset setting
	float validationPartOfTrainingDataset = 0.2;


private:
	size_t trainDatasetObjectLimit = 0;

	// Metrics
	double accuracy = 0;
	double precision = 0;
	double recall = 0;
	double fMeasure = 0;
	double error = 0;
};


#endif //NN_NEURALNETWORKMANAGER_H
