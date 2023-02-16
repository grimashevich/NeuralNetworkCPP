//
// Created by Elyas Clown on 14.02.2023.
//

#include "NeuralNetworkManager.h"

NeuralNetworkManager::NeuralNetworkManager()
{

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

double NeuralNetworkManager::GetAccuracy() const
{
	return accuracy;
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

double NeuralNetworkManager::getError() const
{
	return error;
}

NeuralNetworkManager::~NeuralNetworkManager()
{
	delete neuralNetwork;
	delete trainingSet;
	delete testSet;
}

void NeuralNetworkManager::Train(int numEpochs)
{
	if (neuralNetwork == nullptr)
		throw std::runtime_error("No neural network loaded for training");
	if (trainingSet == nullptr || trainingSet->trainInputs.empty())
		throw std::runtime_error("No training set loaded for training");
	if ((int)((float)trainingSet->trainInputs.size() * validationPartOfTrainingDataset) <= 0)
		throw std::runtime_error("There is not enough data in the training set to create a validation set.");

	trainingSet->SetValidationPartRatio(validationPartOfTrainingDataset);
	trainingSet->Shuffle();
	double meanError = neuralNetwork->Train(trainingSet->trainInputs, trainingSet->trainTargets, numEpochs);
	error = meanError;
	//todo update metrics
}

float NeuralNetworkManager::GetValidationPartOfTrainingDataset() const
{
	return validationPartOfTrainingDataset;
}

void NeuralNetworkManager::SetValidationPartOfTrainingDataset(float newValue)
{
	if (newValue < 0 || newValue > 1)
		throw std::invalid_argument("ValidationPartOfTrainingDataset value must be between 0 and 1");
	NeuralNetworkManager::validationPartOfTrainingDataset = newValue;
}

