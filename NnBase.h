#ifndef NN_NNBASE_H
#define NN_NNBASE_H

#include <vector>
#include <random>

class NnBase
{
public:
	virtual void Train(const std::vector<std::vector<double>>& inputs,
					   const std::vector<std::vector<double>>& targets,
					   int numEpochs) = 0;
	virtual std::vector<double> Predict(const std::vector<double>& input) = 0;
	virtual void SaveWeight(double accuracy, int epoch) = 0;
	virtual void LoadWeight(std::string fileName) = 0;

	static double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
	static double dSigmoid(double x) { return x * (1 - x); }


};

#endif //NN_NNBASE_H
