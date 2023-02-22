#include "MatrixNeuralNetwork.h"

MatrixNeuralNetwork::MatrixNeuralNetwork(const std::vector<int> &Topology): topology(Topology), generator(std::random_device{}())
{
	size_t numLayers = Topology.size();
	for (int i = 0; i < numLayers; i++) {
		layers.emplace_back(Topology[i]);
		if (i != 0) {
			weights.emplace_back(Topology[i], std::vector<double>(Topology[i - 1]));
			biases.emplace_back(Topology[i]);
		}
	}
	InitWeights();
	InitBiases();
}

double MatrixNeuralNetwork::Train(const std::vector<std::vector<double>>& inputs,
								  const std::vector<std::vector<double>>& targets, int numEpochs)
{
	std::vector<std::vector<double>> errors;
	double netError = 0;
	for (int e = 0; e < numEpochs; e++) {
		for (int i = 0; i < inputs.size(); i++) {
			std::vector<std::vector<double>> activations = ForwardFeed(inputs[i]);
			errors = BackProp(activations, targets[i]);
			netError += GetMeanError(errors[errors.size() - 1]);
			UpdateWeights(errors, activations);
			UpdateBiases(errors);
		}
	}
	return netError;
}

double MatrixNeuralNetwork::trainWithMiniBatches(const std::vector<std::vector<double>> &inputs,
                                                 const std::vector<std::vector<double>> &targets,
                                                 double batchSize, int threadsCount = 0)
{
    int bathSizeInt = static_cast<int>(batchSize);
    if (batchSize < 0)
        throw std::invalid_argument("Batch size must be >= 0");
    else if (batchSize == 0)
        bathSizeInt = static_cast<int>(inputs.size());
    else if (batchSize  < 1)
        bathSizeInt = std::floor(static_cast<double>(inputs.size()) * batchSize);
    if (threadsCount <= 0)
        threadsCount = static_cast<int>(std::thread::hardware_concurrency());
    int batchCount = std::ceil(static_cast<double>(inputs.size()) / static_cast<double>(bathSizeInt));

    std::vector<std::thread> threads;
    std::mutex m;
    int curStart = 0;
    for (int i = 0; i < threadsCount; ++i)
    {
        int batchCountInOneThread = batchCount / threadsCount;
        if (i < batchCount % threadsCount)
            batchCountInOneThread++;
        threads.emplace_back(&MatrixNeuralNetwork::TrainOneBatchLauncher, this, inputs, targets, curStart, bathSizeInt, batchCountInOneThread, std::ref(m));
        curStart += batchCountInOneThread * (int) batchSize;
    }
    for (int i = 0; i < threadsCount; ++i)
        threads[i].join();

    return 0;
}

void MatrixNeuralNetwork::TrainOneBatchLauncher(const std::vector<std::vector<double>> &inputs,
                                                const std::vector<std::vector<double>> &targets,
                                                int startFrom, int batchSize, int batchCount, std::mutex &m)
{
    for (int i = 0; i < batchCount; ++i)
        TrainOneBatch(inputs, targets, startFrom + batchSize * i, batchSize, m);
}

// Train one batch for multi thread method
double MatrixNeuralNetwork::TrainOneBatch(const std::vector<std::vector<double>> &inputs,
                                          const std::vector<std::vector<double>> &targets,
                                          int batchStart, int batchSize, std::mutex &m)
{
    std::vector<std::vector<double>> meanErrors = std::vector<std::vector<double>>();
    std::vector<std::vector<double>> meanActivation = std::vector<std::vector<double>>();
    auto newLayers = layers;
    int batchUpperLimit = std::min(batchStart + batchSize, (int) inputs.size());
    for (int i = batchStart; i < batchUpperLimit; ++i)
    {
        std::vector<std::vector<double>> activations = ForwardFeedMT(inputs[i], newLayers);
        auto errors = BackProp(activations, targets[i]);
        addVectorValues(meanActivation, activations);
        addVectorValues(meanErrors, errors);
    }
    divideEachElement(meanActivation, (double) batchSize);
    //divideEachElement(meanErrors, (double) batchSize);
    m.lock();
    UpdateWeights(meanErrors, meanActivation);
    UpdateBiases(meanErrors);
    m.unlock();
    return 0; //TODO calculate batch error
}

void MatrixNeuralNetwork::addVectorValues(std::vector<std::vector<double>> &source, std::vector<std::vector<double>> &summation)
{
    if (source.empty())
        source = summation;
    for (int i = 0; i < source.size(); ++i){
        for (int j = 0; j < source[i].size(); ++j){
            source[i][j] += summation[i][j];
        }

    }
}

void MatrixNeuralNetwork::divideEachElement(std::vector<std::vector<double>> &source, double divider)
{
    for (int i = 0; i < source.size(); ++i) {
        for (int j = 0; j < source[i].size(); ++j) {
            source[i][j] /= divider;
        }
    }
}

std::vector<double> MatrixNeuralNetwork::Predict(const std::vector<double>& input)
{
	std::vector<double> output = ForwardFeed(input).back();
	double sum = 0;
	for (int i = 0; i < output.size(); i++)
	{
		output[i] = exp(output[i]);
		sum += output[i];
	}
	for (int i = 0; i < output.size(); i++)
	{
		output[i] /= sum;
	}
	return output;
}

std::vector<double> MatrixNeuralNetwork::PredictMT(const std::vector<double>& input,
                                                   std::vector<std::vector<double>> & layersCopy)
{
    std::vector<double> output = ForwardFeedMT(input, layersCopy).back();
    double sum = 0;
    for (int i = 0; i < output.size(); i++)
    {
        output[i] = exp(output[i]);
        sum += output[i];
    }
    for (int i = 0; i < output.size(); i++)
    {
        output[i] /= sum;
    }
    return output;
}

std::vector<std::vector<double>> MatrixNeuralNetwork::ForwardFeedMT(const std::vector<double>& input,
                                                                    std::vector<std::vector<double>> & layersCopy)
{
    layersCopy[0] = input;
    for (int i = 1; i < topology.size(); i++) {
        for (int j = 0; j < layersCopy[i].size(); j++) {
            double sum = biases[i - 1][j];
            for (int k = 0; k < layersCopy[i - 1].size(); k++) {
                sum += layersCopy[i - 1][k] * weights[i - 1][j][k];
            }
            layersCopy[i][j] = Sigmoid(sum);
        }
    }
    return layersCopy;
}

void MatrixNeuralNetwork::ForwardFeedMTBatch(const std::vector<double>& input,
                                             std::vector<std::vector<double>> & sumLayersValue)
{
    sumLayersValue[0] = input;
    for (int i = 1; i < topology.size(); i++) {
        for (int j = 0; j < sumLayersValue[i].size(); j++) {
            double sum = biases[i - 1][j];
            for (int k = 0; k < sumLayersValue[i - 1].size(); k++) {
                sum += sumLayersValue[i - 1][k] * weights[i - 1][j][k];
            }
            sumLayersValue[i][j] += Sigmoid(sum);
        }
    }
}

void MatrixNeuralNetwork::SaveWeights(double accuracy, int epoch, std::string fName)
{
	std::string fileName;
	if (fName.empty())
		fileName = GetFileNameForWeights(accuracy, epoch);
	else
		fileName = fName;
	std::ofstream weightsFile (fileName);
	if (! weightsFile.is_open())
	{
		std::cerr << "Error open file " << fileName << " for save weight";
		return;
	}
	for (int layerSize: topology) {
		weightsFile << layerSize << " ";
	}
	weightsFile << "\n";
	//Save weights
	for (auto & table : weights){
		for (auto & row : table) {
			int i = 0;
			for (double weight : row) {
				if (i++ > 0)
					weightsFile << " ";
				weightsFile << weight;
			}
			weightsFile << "\n";
		}
		weightsFile << "\n";
	}
	//Save biases
	for (auto & layer: biases){
		int i = 0;
		for (double bias: layer){
			if (i++ > 0)
				weightsFile << " ";
			weightsFile << bias;
		}
		weightsFile << "\n";
	}
	weightsFile.close();
}

void MatrixNeuralNetwork::LoadWeight(std::string fileName)
{
	std::ifstream weightsFile(fileName);
	std::string line;
	if (getline(weightsFile, line))
	if (!CheckTopology(line))
		throw std::runtime_error("Wrong saved weight topology");

	//load weights
	int i = 0;
	for (auto & table : weights)
	{
		if (i++ > 0) // Skip blank line between tables
			getline(weightsFile, line);
		for (auto & row : table)
		{
			getline(weightsFile, line);
			std::stringstream stream(line);
			std::string weightStr;
			for (double & weight : row)
			{
				getline(stream, weightStr, ' ');
				weight = std::stod(weightStr);
			}
		}
	}
	//load biases
	getline(weightsFile, line);
	for (auto & layer: biases)
	{
		getline(weightsFile, line);
		std::stringstream stream(line);
		std::string biasStr;
		for (double & bias: layer)
		{
			getline(stream, biasStr, ' ');
			bias = std::stod(biasStr);
		}
	}
	weightsFile.close();
}

std::vector<std::string> MatrixNeuralNetwork::SplitString(const std::string& str, char sep)
{
	std::vector<std::string> result(0);
	std::string tmp;
	std::stringstream ss(str);
	while (getline(ss, tmp, sep))
		result.push_back(tmp);
	return result;
}

bool MatrixNeuralNetwork::CheckTopology(const std::string& strTopology)
{
	std::vector<std::string> strTopologyArr = SplitString(strTopology, ' ');
	if (strTopologyArr.size() != topology.size())
		return false;
	for (int i = 0; i < strTopologyArr.size(); ++i)
	{
		if (std::stoi(strTopologyArr[i]) != topology[i])
			return false;
	}
	return true;

}

std::string MatrixNeuralNetwork::GetFileNameForWeights(double  accuracy, int epoch)
{
	std::ostringstream fileName;
	fileName << currentDateTime() << "_";
	for (int i = 0; i < topology.size(); ++i)
	{
		fileName << topology[i];
		if (i < topology.size() - 1)
			fileName << "-";
	}
	fileName << "_epoch-";
	fileName << epoch;
	fileName << "_accuracy-";
	fileName << accuracy;
	fileName << ".weights";
	return fileName.str();
}

void MatrixNeuralNetwork::InitWeights()
{
	std::normal_distribution<double> distribution(0, 1);
	for (int i = 0; i < weights.size(); i++) {
		for (int j = 0; j < weights[i].size(); j++) {
			for (int k = 0; k < weights[i][j].size(); k++) {
				weights[i][j][k] = distribution(generator);
			}
		}
	}
}

void MatrixNeuralNetwork::InitBiases()
{
	std::normal_distribution<double> distribution(0, 1);
	for (int i = 0; i < biases.size(); i++) {
		for (int j = 0; j < biases[i].size(); j++) {
			biases[i][j] = distribution(generator);
		}
	}
}

std::vector<std::vector<double>> MatrixNeuralNetwork::ForwardFeed(const std::vector<double>& input)
{
	layers[0] = input;
	for (int i = 1; i < topology.size(); i++) {
		for (int j = 0; j < layers[i].size(); j++) {
			double sum = biases[i - 1][j];
			for (int k = 0; k < layers[i - 1].size(); k++) {
				sum += layers[i - 1][k] * weights[i - 1][j][k];
			}
			layers[i][j] = Sigmoid(sum);
		}
	}
	return layers;
}

std::vector<std::vector<double>> MatrixNeuralNetwork::BackProp(const std::vector<std::vector<double>> &activations,
                                                               const std::vector<double> &target)
{
	std::vector<std::vector<double>> errors(topology.size());
	int output_layer = topology.size() - 1;
	for (int i = 0; i < topology[output_layer]; i++) {
		errors[output_layer].emplace_back(activations[output_layer][i] - target[i]);
	}
	for (int i = output_layer - 1; i > 0; i--) {
		for (int j = 0; j < topology[i]; j++) {
			double sum = 0;
			for (int k = 0; k < topology[i + 1]; k++) {
				sum += errors[i + 1][k] * weights[i][k][j];
			}
			errors[i].emplace_back(sum * DSigmoid(activations[i][j]));
		}
	}
	return errors;
}

void MatrixNeuralNetwork::UpdateWeights(const std::vector<std::vector<double>> &errors,
                                        const std::vector<std::vector<double>> &activations)
{
	for (int i = 0; i < weights.size(); i++) {
		for (int j = 0; j < weights[i].size(); j++) {
			for (int k = 0; k < weights[i][j].size(); k++) {
				double delta = -learningRate * errors[i + 1][j] * activations[i][k];
				weights[i][j][k] += delta;
			}
		}
	}
}

void MatrixNeuralNetwork::UpdateBiases(const std::vector<std::vector<double>> &errors)
{
	for (int i = 0; i < biases.size(); i++) {
		for (int j = 0; j < biases[i].size(); j++) {
			double delta = -learningRate * errors[i + 1][j];
			biases[i][j] += delta;
		}
	}
}

double MatrixNeuralNetwork::GetMeanError(std::vector<double> & errors) const
{
	double result = 0;

	if ( errors.empty())
		return 0;

	for (int i = 0; i < errors.size(); ++i)
		result += std::abs(errors[i]);
	result /= static_cast<double>(errors.size());
	return result;
}

