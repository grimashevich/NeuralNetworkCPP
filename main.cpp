#include <cfloat>
#include "MatrixNeuralNetwork.h"
#include <chrono>
#include "DataSet.h"
#include "StopWatch.h"
#include <thread>

void PrintTopology(std::vector<int> topology)
{
    std::cout << "{";
    for (int i = 0; i < topology.size(); ++i)
    {
        std::cout << topology[i];
        if (i < topology.size() - 1)
            std::cout << " ";
    }
    std::cout << "}" << std::endl;
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

double CheckTestSet(std::vector<std::vector<double>>& TestInputs,
					std::vector<std::vector<double>>& TestTargets,
					NeuralNetworkBase *nn)
{
	int rightCount = 0;
	int wrongCount = 0;
	for (size_t i = 0; i < TestInputs.size(); i++)
	{
		std::vector<double> input = TestInputs[i];
		std::vector<double> netAnswer = nn->Predict(input);
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

void checkTestSet()
{
	std::string weightFileName = "/Users/eclown/Desktop/projects/NeuralNetworkCPP/cmake-build-debug/NN_weights_784-151-75-26_epoch-4_accuracy-80.7658";

	//std::string testSetFileName = "/Users/eclown/Desktop/projects/NeuralNetworkCPP/emnist-letters-train.csv";

	std::string testSetFileName = "/Users/eclown/Desktop/projects/NeuralNetworkCPP/emnist-letters-test.csv";
	DataSet testSet = DataSet(784, 26);
	testSet.LoadFromCSV(testSetFileName, ',', 0, false);
	std::vector<int> topology = { 784, 151, 75, 26 };
	NeuralNetworkBase *nn = new MatrixNeuralNetwork(topology);
	nn->LoadWeight(weightFileName);

	CheckTestSet(testSet.inputSignals, testSet.answers, nn);
}


void testSaveAndLoadWeight()
{
	std::vector<int> topology = { 5, 4, 3, 2};
	MatrixNeuralNetwork *nn = new MatrixNeuralNetwork(topology);

	int i = 0;
	for (auto & table: nn->weights)
	{
		for (auto & row: table)
		{
			for (double & weight: row)
			{
				weight = i++;
			}
		}
	}
	i = 0;
	for (auto & layer: nn->biases)
	{
		for (double & bias: layer)
		{
			bias = i++;
		}
	}
	nn->SaveWeight(0, 0);
	delete nn;
	nn = new MatrixNeuralNetwork(topology);
	nn->LoadWeight("/Users/eclown/Desktop/projects/NeuralNetworkCPP/cmake-build-debug/NN_weights_5-4-3-2_epoch-0_accuracy-0");
	nn->Sigmoid(1);
}

int main(int argc, char *argv[])
{

	//testSaveAndLoadWeight();
	//checkTestSet();
	//return 0;


    if (argc < 3)
    {
        std::cout << "Learning rate ratio required" << std::endl;
        return 1;
    }

    double learningRate = std::stod(argv[1]);
    double learningRateRatio = std::stod(argv[2]);

	srand(std::time(nullptr));
	//SyntheticTest();


	StopWatch sw = StopWatch();
	sw.Start();
	std::string fileName = std::string("/Users/eclown/Desktop/projects/NeuralNetworkCPP/emnist-letters-train.csv");
	if (argc >= 4)
		fileName = argv[3];
	//std::string fileName = std::string("/Users/user/Desktop/projects/NN/emnist-letters-Train.csv");

	std::vector<int> topology = { 784, 59, 39, 26 };
	NeuralNetworkBase *nn = new MatrixNeuralNetwork(topology);
	//neuralNetwork.LoadWeight("NN_weights_5-4-3-2_epoch-0_accuracy-0");
	//neuralNetwork.SaveWeight(0, 0);
	nn->SetLearningRate(learningRate);
	std::cout << "Neural network initialized in " <<  sw.Restart() << std::endl;


	std::string testSetFileName = "/Users/eclown/Desktop/projects/NeuralNetworkCPP/emnist-letters-test.csv";
	DataSet testSet = DataSet(784, 26);
	testSet.LoadFromCSV(testSetFileName, ',', 0, false);
	std::cout << "Test set loaded in ... " <<  sw.Restart() << std::endl;

	DataSet ts = DataSet(784, 26);
	ts.LoadFromCSV(fileName, ',', 0, false);
	ts.SetTestSetSizeRatio(0.0);
	ts.Shuffle();
	std::cout << "Train set loaded in ... " <<  sw.Restart() << std::endl;

	auto testSetInput = testSet.inputSignals;
	auto testSetAnswers = testSet.answers;


    std::cout << "- - - - - - - - - - - -" << std::endl;
    std::cout << "Topology: ";
    PrintTopology(topology);
    std::cout << "Train set size: " << ts.answers.size() << std::endl;
    std::cout << "Test set size: " << testSetInput.size() << std::endl;
    std::cout << "Learning rate: " << nn->GetLearningRate() << std::endl;
    std::cout << "Learning rate ratio: " << learningRateRatio << std::endl;
    std::cout << "- - - - - - - - - - - -" << std::endl;

    std::cout << "BEFORE Learning:" << " ";
    CheckTestSet(testSetInput, testSetAnswers, nn);
    std::cout << std::endl;
    ts.Shuffle();
	for (int i = 0; i < 50; ++i)
	{
		sw.Start();
		nn->Train(ts.inputSignals, ts.answers, 1);
		std::cout << "epoch " << i << " done in " << sw.Stop() << " ";
		double accuracy = CheckTestSet(testSetInput, testSetAnswers, nn);
		if (accuracy > 50.0)
			nn->SaveWeight(accuracy, i);
		ts.Shuffle();
		nn->SetLearningRate(nn->GetLearningRate() * learningRateRatio);
	}

}