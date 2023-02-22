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
	if (trainingSet != nullptr && (trainingSet->GetInputSize() != inputSizeNN || trainingSet->GetOutputSize() != outputSizeNN))
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

void NeuralNetworkManager::trainWithMiniBatches(double learningRate, double batchSize, int threadsCount)
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
	double meanError = neuralNetwork->trainWithMiniBatches(trainingSet->trainInputs,
                                                           trainingSet->trainTargets,
                                                           batchSize, threadsCount);
	error = meanError;
	//todo update metrics
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
	if (!fileExistAndReadable(fileName))
		throw std::invalid_argument("Dataset file not found " + fileName);
	delete trainingSet;
	trainingSet = new DataSet(inputSize, outputSize);
	trainingSet->SetValidationPartRatio(validationPartOfTrainingDataset);
	trainingSet->LoadFromCSV(fileName, ',', objectLimit, false);
    trainingSet->MoveToValidationSet(validationPartOfTrainingDataset);
}

void NeuralNetworkManager::LoadWeightToNetwork(const std::string& fileName)
{
    if (!fileExistAndReadable(fileName))
        throw std::invalid_argument("Weights file not found " + fileName);
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

bool NeuralNetworkManager::fileExistAndReadable(const std::string &fileName)
{
    bool fileIsOk;
    fileIsOk = std::filesystem::exists(fileName);
    fileIsOk = fileIsOk && std::filesystem::is_regular_file(fileName);
    if (fileIsOk)
    {
        std::ifstream file(fileName);
        if (file.is_open())
        {
            file.close();
            return true;
        }

    }
    return false;
}

void NeuralNetworkManager::CalculateMetricsForTestSet(const std::vector<std::vector<double>> & testInputs,
													  const std::vector<std::vector<double>> & testTargets,
													  size_t threadsNum)
{
	size_t dataSetSize = testInputs.size();
	std::vector<std::thread> threads;
	std::vector<std::vector<std::vector<size_t>>> results(
			threadsNum,std::vector<std::vector<size_t>>(
					testTargets[0].size(), std::vector<size_t>(2)));

    std::mutex m;
	size_t pieceSize = dataSetSize / threadsNum;
	for (int i = 0; i < threadsNum; ++i)
	{
		size_t fromIndex = (i * pieceSize);
		size_t toIndex = ((i + 1) * pieceSize); //will be not included
		if (i == threadsNum - 1)
			toIndex = dataSetSize;

		threads.emplace_back(PredictMT,
                             testInputs,
                             testTargets,
                             fromIndex,
                             toIndex,
                             0,
                             false,
                             std::ref(results[i]),
                             neuralNetwork,
                             std::ref(m));
	}
	for (int i = 0; i < threadsNum; ++i)
		threads[i].join();

	std::vector<std::vector<size_t>> finalResult(std::vector<std::vector<size_t>>(testTargets[0].size(), std::vector<size_t>(2)));
	size_t totalRight = 0;
	size_t totalWrong = 0;
	for (int i = 0; i < results.size(); ++i)
	{
		for (int j = 0; j < results[i].size(); ++j)
		{
			finalResult[j][0] += results[i][j][0];
			finalResult[j][1] += results[i][j][1];
			totalRight += results[i][j][0];
			totalWrong += results[i][j][1];
		}
	}
    accuracy = (float)totalRight / (float)(totalRight + totalWrong);
/*    std::cout << "Total wrong: " << totalWrong << std::endl;
    std::cout << "Total right: " << totalRight << std::endl;
    std::cout << "Accuracy: " << ((float)totalRight / (float)(totalRight + totalWrong)) * 100 << "%" << std::endl;*/
}

void NeuralNetworkManager::PredictMT(const std::vector<std::vector<double>> &inputs,
									 const std::vector<std::vector<double>> &targets,
									 size_t fromIndex,
									 size_t toIndex,
									 int answerOffset,
									 bool needNormalize,
									 std::vector<std::vector<size_t>> & result,
									 NeuralNetworkBase *nn,
                                     std::mutex & m)
{
    float right = 0;
    float wrong = 0;
    auto newLayers = nn->layers;
	for (size_t i = fromIndex; i < toIndex; ++i)
	{
        std::vector<double> answerVector = nn->PredictMT(inputs[i], newLayers);

        size_t answer = std::max_element(answerVector.begin(),answerVector.end()) - answerVector.begin();
		size_t true_answer = std::max_element(targets[i].begin(), targets[i].end()) - targets[i].begin();

		if (answer == true_answer)
        {
			result[true_answer][0]++;
            right++;
        }
		else
        {
			result[true_answer][1]++;
            wrong++;
        }
	}
/*    m.lock();
    std::cout << "right/wrong " << right << "/" << wrong << " " <<
            (right / (right + wrong)) * 100 << "%" <<std::endl;
    m.unlock();*/
}

