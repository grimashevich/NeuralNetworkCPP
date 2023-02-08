#include <cfloat>
#include "NN.cpp"
#include <chrono>
#include "cmake-build-debug/TrainingSet.h"

void AddItemToTrainSet(double x, double y, double answer, std::vector<std::vector<double>> &inputs, std::vector<std::vector<double>> &targets)
{
	std::vector<double> newInput = { x, y };
	inputs.push_back(newInput);
	std::vector<double> newAnswer(4);
	newAnswer[answer] = 1;
	targets.push_back(newAnswer);
}


void GenerateTrainSet(std::vector<std::vector<double>>& inputs, std::vector<std::vector<double>>& targets, int recordsCount, int mp = 1)
{
	double minX = -1000 * mp;
	double minY = -1000 * mp;
	double maxX = 1000 * mp;
	double maxY = 1000 * mp;

	double answer;
	for (int i = 0; i < recordsCount; i++)
	{
		double x = (double)rand() / RAND_MAX;
		x = minX + x * (maxX - minX);
		double y = (double)rand() / RAND_MAX;
		y = minY + y * (maxY - minY);

		if (x < 0 && y < 0)
			answer = 2;
		else if (x >= 0 && y >= 0)
			answer = 1;
		else if (x < 0 && y >= 0)
			answer = 0;
		else
			answer = 3;
		AddItemToTrainSet(x, y, answer, inputs, targets);
	}


}


void printVector(std::vector<double> input, std::vector<double> answer)
{
	for (size_t i = 0; i < input.size(); i++)
	{
		std::cout << input[i] << " ";
	}
	std::cout << std::endl;

	for (size_t i = 0; i < answer.size(); i++)
	{
		std::cout << answer[i] << " ";
	}
	std::cout << std::endl << std::endl;

}


int getIndexOfMaxEl(std::vector<double> vect)
{
	double curMin = DBL_MIN;
	int curIndex = -1;
	for (size_t i = 0; i < vect.size(); i++)
	{
		if (vect[i] > curMin)
		{
			curMin = vect[i];
			curIndex = i;
		}
	}
	return curIndex;
}

void CheckTestSet(std::vector<std::vector<double>>& TestInputs, std::vector<std::vector<double>>& TestTargets, NeuralNetwork &nn)
{
	int rightCount = 0;
	int wrongCount = 0;
	for (size_t i = 0; i < TestInputs.size(); i++)
	{
		std::vector<double> input = TestInputs[i];
		std::vector<double> netAnswer = nn.predict(input);
		if (getIndexOfMaxEl(netAnswer) == getIndexOfMaxEl(TestTargets[i]))
			rightCount++;
		else
			wrongCount++;
	}
	std::cout << "right: " << rightCount << " wrong: " << wrongCount << " (" << (rightCount / (double) TestInputs.size()) * 100 << "%)" << std::endl;
}


void SynteticTest()
{

	srand(std::time(nullptr));
	std::vector<std::vector<double>> inputs = std::vector<std::vector<double>>();
	std::vector<std::vector<double>> targets = std::vector<std::vector<double>>();

	std::vector<std::vector<double>> TestInputs = std::vector<std::vector<double>>();
	std::vector<std::vector<double>> TestTargets = std::vector<std::vector<double>>();

	GenerateTrainSet(inputs, targets, 1000);
	GenerateTrainSet(TestInputs, TestTargets, 1000, 100);

	std::vector<int> topology = { 2, 12, 6, 4 };
	NeuralNetwork nn = NeuralNetwork(topology);
	nn.train(inputs, targets, 1);
	CheckTestSet(TestInputs, TestTargets, nn);
	nn.train(inputs, targets, 1);
	CheckTestSet(TestInputs, TestTargets, nn);
	nn.train(inputs, targets, 1);
	CheckTestSet(TestInputs, TestTargets, nn);
	nn.train(inputs, targets, 1);
	CheckTestSet(TestInputs, TestTargets, nn);
	nn.train(inputs, targets, 1);
	CheckTestSet(TestInputs, TestTargets, nn);
}

int main()
{
	//SynteticTest();



	TrainingSet ts = TrainingSet(784, 26);
	std::string fileName = std::string("/Users/eclown/Desktop/projects/NN/emnist-letters-train.csv");
	ts.answerOffset = -1;
	ts.LoadFromCSV(fileName, ',', 1000);
	std::cout << "train set load ok " << std::endl;
	ts.MoveToTestSet(0.2);
	std::cout << "train set split ok"  << std::endl;
	std::vector<int> topology = { 784, 100, 50, 26 };
	NeuralNetwork nn = NeuralNetwork(topology);
	CheckTestSet(ts.testSetInputSignals, ts.testSetAnswers, nn);
	auto start = std::chrono::high_resolution_clock::now();
	nn.train(ts.inputSignals, ts.answers, 1);
	CheckTestSet(ts.testSetInputSignals, ts.testSetAnswers, nn);
	nn.train(ts.inputSignals, ts.answers, 1);
	CheckTestSet(ts.testSetInputSignals, ts.testSetAnswers, nn);
	nn.train(ts.inputSignals, ts.answers, 1);
	CheckTestSet(ts.testSetInputSignals, ts.testSetAnswers, nn);
	nn.train(ts.inputSignals, ts.answers, 1);
	CheckTestSet(ts.testSetInputSignals, ts.testSetAnswers, nn);
	nn.train(ts.inputSignals, ts.answers, 1);
	CheckTestSet(ts.testSetInputSignals, ts.testSetAnswers, nn);
	nn.train(ts.inputSignals, ts.answers, 1);
	CheckTestSet(ts.testSetInputSignals, ts.testSetAnswers, nn);
	nn.train(ts.inputSignals, ts.answers, 1);
	CheckTestSet(ts.testSetInputSignals, ts.testSetAnswers, nn);
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
	std::cout << "learning time " << duration.count() << " seconds" << std::endl;
	CheckTestSet(ts.testSetInputSignals, ts.testSetAnswers, nn);

}