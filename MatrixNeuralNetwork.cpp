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

std::vector<std::vector<double>> MatrixNeuralNetwork::BackProp(const std::vector<std::vector<double>> &activations, const std::vector<double> &target)
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

void MatrixNeuralNetwork::UpdateWeights(const std::vector<std::vector<double>> &errors, const std::vector<std::vector<double>> &activations)
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
		result += abs(errors[i]);
	result /= static_cast<double>(errors.size());
	return result;
}

