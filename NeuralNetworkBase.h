#ifndef NN_NEURALNETWORKBASE_H
#define NN_NEURALNETWORKBASE_H

#include <vector>
#include <random>

class NeuralNetworkBase
{
public:
	virtual void Train(const std::vector<std::vector<double>>& inputs,
					   const std::vector<std::vector<double>>& targets,
					   int numEpochs) = 0;
	virtual std::vector<double> Predict(const std::vector<double>& input) = 0;
	virtual void SaveWeight(double accuracy, int epoch) = 0;
	virtual void LoadWeight(std::string fileName) = 0;

	void SetLearningRate(double newLearningRate);
	static double Sigmoid(double x);
	static double DSigmoid(double x);

	[[nodiscard]] double GetLearningRate() const;

	virtual ~NeuralNetworkBase();

protected:
	double learningRate{0.02};

};

#endif //NN_NEURALNETWORKBASE_H
