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

void NeuralNetworkManager::Train(int numEpochs, double learningRate)
{
	if (neuralNetwork == nullptr)
		throw std::runtime_error("No neural network loaded for training");
	if (trainingSet == nullptr || trainingSet->trainInputs.empty())
		throw std::runtime_error("No training set loaded for training");
	if ((int)((float)trainingSet->trainInputs.size() * validationPartOfTrainingDataset) <= 0)
		throw std::runtime_error("There is not enough data in the training set to create a validation set.");

	neuralNetwork->SetLearningRate(learningRate);
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

void NeuralNetworkManager::CrutchNormalzation(std::vector<double> &signal)
{
	//TODO Заменить на метод из Датасета
	for (double & item: signal)
	{
		if (item > 64)
			item = 1;
		else
			item = 0;
	}
}


size_t NeuralNetworkManager::Predict(std::vector<double> inputSignal, int answerOffset, bool needNormalize)
{
	size_t result;
	std::vector<double> networkAnswer;

	if (needNormalize)
		CrutchNormalzation(inputSignal);
	networkAnswer = neuralNetwork->Predict(inputSignal);
	result = std::max_element(networkAnswer.begin(),networkAnswer.end()) - networkAnswer.begin();
	return result + answerOffset;
}

void NeuralNetworkManager::LoadTrainSet(std::string & fileName, size_t inputSize, size_t outputSize, size_t objectLimit)
{
	if (inputSize && outputSize == 0)
		throw std::invalid_argument("inputSize and outputSize should be positive");
	delete trainingSet;
	trainingSet = new DataSet(inputSize, outputSize);
	trainingSet->SetValidationPartRatio(validationPartOfTrainingDataset);
	trainingSet->LoadFromCSV(fileName, ',', objectLimit, false);
	trainingSet->MoveToTestSet(validationPartOfTrainingDataset);
}

void NeuralNetworkManager::LoadWeightToNetwork(const std::string& fileName)
{
	if (neuralNetwork == nullptr)
		throw std::runtime_error("Neural network is not loaded");
	neuralNetwork->LoadWeight(fileName);
}

void NeuralNetworkManager::SaveWeightFromNetwork(double curAccuracy, size_t epochNum, const std::string& alterFileName)
{
	if (neuralNetwork == nullptr)
		throw std::runtime_error("Neural network is not loaded");
	neuralNetwork->SaveWeights(curAccuracy, epochNum, alterFileName);
}
