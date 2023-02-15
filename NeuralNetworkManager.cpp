//
// Created by Elyas Clown on 14.02.2023.
//

#include "NeuralNetworkManager.h"

NeuralNetworkManager::NeuralNetworkManager()
{

}

double NeuralNetworkManager::getPrecision() const
{
	return precision;
}

double NeuralNetworkManager::getRecall() const
{
	return recall;
}

double NeuralNetworkManager::getFMeasure() const
{
	return fMeasure;
}

void NeuralNetworkManager::LoadMatrixNN(const std::vector<int> &topology)
{
	if (topology.size() < 3)
		throw std::invalid_argument("Neural network must have minimum 3 layers");
	delete neuralNetwork;
	neuralNetwork = new MatrixNeuralNetwork(topology);
	inputSizeNN = topology[0];
	outputSizeNN = topology[topology.size() - 1];
	if (trainingSet->GetInputSize() != inputSizeNN || trainingSet->GetOutputSize() != outputSizeNN)
	{
		delete trainingSet;
		delete testSet;
		trainingSet = nullptr;
		testSet = nullptr;
	}
}


NeuralNetworkManager::~NeuralNetworkManager()
{
	delete neuralNetwork;
	delete trainingSet;
	delete testSet;
}
