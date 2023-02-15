#ifndef NN_NEURALNETWORKMANAGER_H
#define NN_NEURALNETWORKMANAGER_H


#include "NeuralNetworkBase.h"
#include "MatrixNeuralNetwork.h"
#include "DataSet.h"
#include "StopWatch.h"

class NeuralNetworkManager
{
private:
	int inputSizeNN = 0;
	int outputSizeNN = 0;
	NeuralNetworkBase *neuralNetwork = nullptr;
	DataSet *trainingSet = nullptr;
	DataSet *testSet = nullptr;
	double precision = 0;
	double recall = 0;
	double fMeasure = 0;

public:
	NeuralNetworkManager();
	void LoadMatrixNN(const std::vector<int> &topology);
	void LoadGraphNN();

	double getPrecision() const;
	double getRecall() const;
	double getFMeasure() const;

	~NeuralNetworkManager();
};


#endif //NN_NEURALNETWORKMANAGER_H
