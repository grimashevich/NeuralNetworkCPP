#include <cfloat>
#include "NN.cpp"
#include <chrono>
#include "TrainingSet.h"
#include "StopWatch.h"
#include <thread>

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

double CheckTestSet(std::vector<std::vector<double>>& TestInputs, std::vector<std::vector<double>>& TestTargets, NeuralNetwork &nn)
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
	//std::cout << "right: " << rightCount << " wrong: " << wrongCount << " (" << (rightCount / (double) TestInputs.size()) * 100 << "%)" << std::endl;
	double result = (rightCount / (double) TestInputs.size()) * 100;
	std::cout << result << "%" << std::endl;
	return result;
}


void SyntheticTest()
{
	std::vector<std::vector<double>> inputs = std::vector<std::vector<double>>();
	std::vector<std::vector<double>> targets = std::vector<std::vector<double>>();

	std::vector<std::vector<double>> TestInputs = std::vector<std::vector<double>>();
	std::vector<std::vector<double>> TestTargets = std::vector<std::vector<double>>();

	GenerateTrainSet(inputs, targets, 1000, false);
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
	srand(std::time(nullptr));
	//SyntheticTest();

	TrainingSet ts = TrainingSet(784, 26);
	StopWatch swLoadSet = StopWatch();
	swLoadSet.Start();
	std::string fileName = std::string("/Users/eclown/Desktop/projects/NN/emnist-letters-train.csv");
	ts.answerOffset = -1;
	ts.LoadFromCSV(fileName, ',', 0, false);
	std::cout << swLoadSet.Restart() << " data set loaded" << std::endl;

	//std::this_thread::sleep_for(std::chrono::milliseconds(20000));

	ts.MoveToTestSet(0.1);
	std::cout << swLoadSet.Restart() << " move complete" << std::endl;

	std::vector<int> topology = { 784, 100, 50, 26 };
	NeuralNetwork nn = NeuralNetwork(topology);
	std::cout << swLoadSet.Stop() << " NN init complete" << std::endl;
	std::cout << "BEFORE Learning:" << std::endl;
	CheckTestSet(ts.testSetInputSignals, ts.testSetAnswers, nn);
	std::cout << std::endl;
	double learningRateRatio = 0.7;
	for (int i = 0; i < 50; ++i)
	{
		swLoadSet.Start();
		nn.train(ts.inputSignals, ts.answers, 1);
		std::cout << "epoch " << i << " done in " << swLoadSet.Stop() << " ";
		CheckTestSet(ts.testSetInputSignals, ts.testSetAnswers, nn);
		ts.Shuffle();
		nn.setLearningRate(nn.getLearningRate() * learningRateRatio);
	}

}