//
// Created by Elyas Clown on 14.02.2023.
//

#include "NeuralNetworkManager.h"

NeuralNetworkManager::NeuralNetworkManager() {

}

void NeuralNetworkManager::LoadMatrixNN(const std::vector<int> &topology) {
    if (topology.size() < 3)
        throw std::invalid_argument("Neural network must have minimum 3 layers");
    delete neuralNetwork;
    neuralNetwork = new MatrixNeuralNetwork(topology);
    inputSizeNN = topology[0];
    outputSizeNN = topology[topology.size() - 1];
    if (trainingSet != nullptr && (trainingSet->GetInputSize() != inputSizeNN || trainingSet->GetOutputSize() != outputSizeNN)) {
        delete trainingSet;
        delete testSet;
        trainingSet = nullptr;
        testSet = nullptr;
    }
}


void NeuralNetworkManager::Train(int numEpochs, double learningRate) {
    if (neuralNetwork == nullptr)
        throw std::runtime_error("No neural network loaded for training");
    if (trainingSet == nullptr || trainingSet->trainInputs.empty())
        throw std::runtime_error("No training set loaded for training");
    if ((int) ((float) trainingSet->trainInputs.size() * validationPartOfTrainingDataset) <= 0)
        throw std::runtime_error("There is not enough data in the training set to create a validation set.");

    neuralNetwork->SetLearningRate(learningRate);
    trainingSet->SetValidationPartRatio(validationPartOfTrainingDataset);
    trainingSet->Shuffle();
    double meanError = neuralNetwork->Train(trainingSet->trainInputs, trainingSet->trainTargets, numEpochs);
    error = meanError;
    CalculateMetricsForTestSet(trainingSet->validationInputs, trainingSet->validationTargets);
}

float NeuralNetworkManager::GetValidationPartOfTrainingDataset() const {
    return validationPartOfTrainingDataset;
}

void NeuralNetworkManager::SetValidationPartOfTrainingDataset(float newValue) {
    if (newValue < 0 || newValue > 1)
        throw std::invalid_argument("ValidationPartOfTrainingDataset value must be between 0 and 1");
    NeuralNetworkManager::validationPartOfTrainingDataset = newValue;
}

void NeuralNetworkManager::CrutchNormalzation(std::vector<double> &signal) {
    //TODO Заменить на метод из Датасета
    for (double &item : signal) {
        if (item > 64)
            item = 1;
        else
            item = 0;
    }
}


size_t NeuralNetworkManager::Predict(std::vector<double> inputSignal, int answerOffset, bool needNormalize) {
    size_t result;
    std::vector<double> networkAnswer;

    if (needNormalize)
        CrutchNormalzation(inputSignal);
    networkAnswer = neuralNetwork->Predict(inputSignal);
    result = std::max_element(networkAnswer.begin(), networkAnswer.end()) - networkAnswer.begin();
    return result + answerOffset;
}

void NeuralNetworkManager::LoadTrainSet(std::string &fileName,
                                        size_t inputSize,
                                        size_t outputSize,
                                        size_t objectLimit) {
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

void NeuralNetworkManager::LoadWeightToNetwork(const std::string &fileName) {
    if (!fileExistAndReadable(fileName))
        throw std::invalid_argument("Weights file not found: " + fileName);
    LoadMatrixNN(getTopologyFromWeightsFile(fileName));
    neuralNetwork->LoadWeight(fileName);
}

std::vector<int> NeuralNetworkManager::getTopologyFromWeightsFile(const std::string &weightsFileName) {
    std::ifstream weightsFile(weightsFileName);
    std::string line;
    std::vector<int> topology = std::vector<int>();
    if (getline(weightsFile, line)) {
        std::vector<std::string> weightsStr = MatrixNeuralNetwork::SplitString(line, ' ');
        for (int i = 0; i < weightsStr.size(); ++i) {
            topology.emplace_back(std::stoi(weightsStr[i]));
        }
    } else {
        throw std::runtime_error("Can't read from file: " + weightsFileName);
    }
    return topology;
}

void NeuralNetworkManager::SaveWeightFromNetwork(double curAccuracy,
                                                 size_t epochNum,
                                                 const std::string &alterFileName) {
    if (neuralNetwork == nullptr)
        throw std::runtime_error("Neural network is not loaded");
    neuralNetwork->SaveWeights(curAccuracy, epochNum, alterFileName);
}

bool NeuralNetworkManager::fileExistAndReadable(const std::string &fileName) {
    bool fileIsOk;
    fileIsOk = std::filesystem::exists(fileName);
    fileIsOk = fileIsOk && std::filesystem::is_regular_file(fileName);
    if (fileIsOk) {
        std::ifstream file(fileName);
        if (file.is_open()) {
            file.close();
            return true;
        }

    }
    return false;
}

void NeuralNetworkManager::CalculateMetricsForTestSet(const std::vector<std::vector<double>> &testInputs,
                                                      const std::vector<std::vector<double>> &testTargets,
                                                      size_t threadsNum) {
    if (threadsNum <= 0) {
        threadsNum = std::thread::hardware_concurrency();
    }
    size_t dataSetSize = testInputs.size();
    std::vector<std::thread> threads;
    std::vector<std::vector<std::vector<size_t>>> results(
            threadsNum, std::vector<std::vector<size_t>>(
                    testTargets[0].size(), std::vector<size_t>(testTargets[0].size())));

    std::mutex m;
    size_t pieceSize = dataSetSize / threadsNum;
    for (int i = 0; i < threadsNum; ++i) {
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

    std::vector<std::vector<size_t>> finalResult(std::vector<std::vector<size_t>>(testTargets[0].size(),
                                                                                  std::vector<size_t>(testTargets[0].size(),
                                                                                                      0)));
    for (int i = 0; i < results.size(); ++i) //threads result
    {
        for (int j = 0; j < results[i].size(); ++j) // true classes
        {
            for (int k = 0; k < results[i][j].size(); ++k) //predicted classes
            {
                finalResult[j][k] += results[i][j][k];
            }
        }
    }
    predict_matrix = finalResult;
    print_matrix(finalResult);

    std::map<std::string, double> mean_metrics = std::map<std::string, double>();
    mean_metrics.emplace("accuracy", 0);
    mean_metrics.emplace("precision", 0);
    mean_metrics.emplace("recall", 0);
    mean_metrics.emplace("f-measure", 0);
    metrics.clear();
    for (int i = 0; i < predict_matrix.size(); ++i) {
        std::map<std::string, double> confusion_matrix = GetConfusionMatrix(finalResult, 0);
        std::map<std::string, double> metricsMap = getMetricsMap(confusion_matrix);
        metrics.emplace_back(metricsMap);
        mean_metrics["accuracy"] += metricsMap["accuracy"];
        mean_metrics["precision"] += metricsMap["precision"];
        mean_metrics["recall"] += metricsMap["recall"];
        mean_metrics["f-measure"] += metricsMap["f-measure"];
    }
    mean_metrics["accuracy"] /= (double) predict_matrix.size();
    mean_metrics["precision"] /= (double) predict_matrix.size();
    mean_metrics["recall"] /= (double) predict_matrix.size();
    mean_metrics["f-measure"] /= (double) predict_matrix.size();
}



void NeuralNetworkManager::print_matrix(const std::vector<std::vector<size_t>> &vec) {
    size_t n = vec.size(); // размерность вектора

    size_t WIDTH = 4;

    // Вывод заголовка таблицы
    std::cout << std::setw(WIDTH) << " ";
    for (size_t j = 0; j < n; ++j) {
        std::cout << std::setw(WIDTH) << (char) (j + 65);
    }
    std::cout << std::setw(6) << "Sum";
    std::cout << std::setw(4) << "TP";
    std::cout << std::setw(4) << "FP" << std::endl;

    // Вывод таблицы
    for (size_t i = 0; i < n; ++i) {
        size_t TP = 0;
        std::cout << std::setw(WIDTH) << (char) (i + 65);
        size_t row_sum = 0;
        for (size_t j = 0; j < n; ++j) {
            std::cout << std::setw(WIDTH) << vec[i][j];
            row_sum += vec[i][j];
            if (i == j)
                TP = vec[i][j];
        }
        std::cout << std::setw(6) << row_sum;
        std::cout << std::setw(4) << TP;
        std::cout << std::setw(4) << row_sum - TP << std::endl;
    }
}

std::map<std::string, double> NeuralNetworkManager::GetConfusionMatrix(
        const std::vector<std::vector<size_t>> &test_matrix, int class_index)
{

    std::map<std::string, double> result = std::map<std::string, double>();
    double matrixSum = 0;

    result["TP"] = (double) test_matrix[class_index][class_index];
    result["TN"] = 0;
    result["FP"] = 0;
    result["FN"] = 0;
    for (int i = 0; i < test_matrix.size(); ++i) {
        if (i != class_index)
        {
            result["TN"] += (double) test_matrix[i][i];
            result["FP"] += (double) test_matrix[i][class_index];
            result["FN"] += (double) test_matrix[class_index][i];
        }
    }

    return result;
}

std::map<std::string, double> NeuralNetworkManager::getMetricsMap(std::map<std::string, double> & confusion_matrix) {
    std::map<std::string, double> metricsMap = std::map<std::string, double>();
    double TP = confusion_matrix["TP"];
    double TN = confusion_matrix["TN"];
    double FP = confusion_matrix["FP"];
    double FN = confusion_matrix["FN"];

    metricsMap.emplace("accuracy", (TP + TN) / (TP + TN + FP + FN));
    metricsMap.emplace("precision", TP / (TP + FP));
    metricsMap.emplace("recall", TP / (TP + FN));
    metricsMap.emplace("f-measure", 2 * metricsMap["precision"] * metricsMap["recall"]
            / (metricsMap["precision"] + metricsMap["recall"]));
    return metricsMap;
}

void NeuralNetworkManager::PredictMT(const std::vector<std::vector<double>> &inputs,
                                     const std::vector<std::vector<double>> &targets,
                                     size_t fromIndex,
                                     size_t toIndex,
                                     int answerOffset,
                                     bool needNormalize,
                                     std::vector<std::vector<size_t>> &result,
                                     NeuralNetworkBase *nn,
                                     std::mutex &m) {
    auto newLayers = nn->layers;
    for (size_t i = fromIndex; i < toIndex; ++i) {
        std::vector<double> answerVector = nn->PredictMT(inputs[i], newLayers);

        size_t nn_answer = std::max_element(answerVector.begin(), answerVector.end()) - answerVector.begin();
        size_t true_answer = std::max_element(targets[i].begin(), targets[i].end()) - targets[i].begin();
        result[true_answer][nn_answer]++;
    }
}

double NeuralNetworkManager::GetAccuracy() const {
    return accuracy;
}

double NeuralNetworkManager::getPrecision() const {
    return precision;
}

double NeuralNetworkManager::getRecall() const {
    return recall;
}

double NeuralNetworkManager::getFMeasure() const {
    return fMeasure;
}

double NeuralNetworkManager::getError() const {
    return error;
}

NeuralNetworkManager::~NeuralNetworkManager() {
    delete neuralNetwork;
    delete trainingSet;
    delete testSet;
}

void NeuralNetworkManager::CrossValidation(size_t folds_count, double learning_rate, double learning_rate_ratio)
{
    if (trainingSet == nullptr)
        throw std::runtime_error("Training set is not loaded");
    if (folds_count < 2)
        throw std::runtime_error("folds_count must be >= 2");

    trainingSet->ReturnTestSetToTrainSet();
    neuralNetwork->SetLearningRate(learning_rate);
    double mean_accuracy = 0;
    double mean_precision = 0;
    double mean_recall = 0;
    double mean_fMeasure = 0;
    for (int i = 0; i < folds_count; ++i)
    {
        size_t fold_size = trainingSet->Size() / folds_count + (i < trainingSet->Size() % folds_count);
        trainingSet->MoveToValidationSet(0, fold_size);
        neuralNetwork->Train(trainingSet->trainInputs, trainingSet->trainTargets, 1);
        CalculateMetricsForTestSet(trainingSet->validationInputs, trainingSet->validationTargets);
        mean_accuracy += accuracy;
        mean_precision += precision;
        mean_recall += recall;
        mean_fMeasure += fMeasure;
        neuralNetwork->SetLearningRate(neuralNetwork->GetLearningRate() * learning_rate_ratio);
        trainingSet->ReturnTestSetToTrainSet();
        //TODO REMOVE IT
        PrintMetrics();
        std::cout.flush();
        //TODO END REMOVE IT
    }
    //TODO REMOVE IT
    std::cout << std::endl;
    //TODO END REMOVE IT
    accuracy = mean_accuracy / (double) folds_count;
    precision = mean_precision / (double) folds_count;
    recall = mean_recall / (double) folds_count;
    fMeasure = mean_fMeasure / (double) folds_count;
}

void NeuralNetworkManager::PrintMetrics() const {
    std::ios_base::fmtflags old_flags = std::cout.flags();
    std::cout << std::fixed << std::setprecision(4) << std::setfill('0');

    std::cout << "accuracy: " << accuracy << " precision: " << precision << " recall: " <<
        recall << " f-measure: " << fMeasure << std::endl;

    std::cout.flags(old_flags);
}


