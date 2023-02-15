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


int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cout << "Learning rate ratio required" << std::endl;
        return 1;
    }

    double learningRate = std::stod(argv[1]);
    double learningRateRatio = std::stod(argv[2]);

	srand(std::time(nullptr));
	//SyntheticTest();


	StopWatch swLoadSet = StopWatch();
	swLoadSet.Start();
	std::string fileName = std::string("/Users/eclown/Desktop/projects/NN/emnist-letters-Train.csv");
	if (argc >= 4)
		fileName = argv[3];
	//std::string fileName = std::string("/Users/user/Desktop/projects/NN/emnist-letters-Train.csv");


	std::vector<int> topology = { 784, 333, 222, 26 };
	NeuralNetworkBase *nn = new MatrixNeuralNetwork(topology);
	//neuralNetwork.LoadWeight("NN_weights_5-4-3-2_epoch-0_accuracy-0");
	//neuralNetwork.SaveWeight(0, 0);
	nn->SetLearningRate(learningRate);

	DataSet ts = DataSet(784, 26);
	ts.LoadFromCSV(fileName, ',', 0, false);
	//std::cout << swLoadSet.Restart() << " data set loaded" << std::endl;
    ts.SetTestSetSizeRatio(0.1);
    ts.Shuffle();


    std::cout << "- - - - - - - - - - - -" << std::endl;
    std::cout << "Topology: ";
    PrintTopology(topology);
    std::cout << "Train set size: " << ts.answers.size() << std::endl;
    std::cout << "Test set size: " << ts.testSetAnswers.size() << std::endl;
    std::cout << "Learning rate: " << nn->GetLearningRate() << std::endl;
    std::cout << "Learning rate ratio: " << learningRateRatio << std::endl;
    std::cout << "- - - - - - - - - - - -" << std::endl;

    std::cout << "BEFORE Learning:" << " ";
    CheckTestSet(ts.testSetInputSignals, ts.testSetAnswers, nn);
    std::cout << std::endl;
    ts.Shuffle();
	for (int i = 0; i < 50; ++i)
	{
		swLoadSet.Start();
		nn->Train(ts.inputSignals, ts.answers, 1);
		std::cout << "epoch " << i << " done in " << swLoadSet.Stop() << " ";
		double accuracy = CheckTestSet(ts.testSetInputSignals, ts.testSetAnswers, nn);
		if (accuracy > 70.0)
			nn->SaveWeight(accuracy, i);
		ts.Shuffle();
		nn->SetLearningRate(nn->GetLearningRate() * learningRateRatio);
	}

}