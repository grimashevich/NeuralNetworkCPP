#include <cfloat>
#include "MatrixNeuralNetwork.h"
#include <chrono>
#include "DataSet.h"
#include "StopWatch.h"
#include <thread>
#include "NeuralNetworkManager.h"

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
	std::string weightFileName = "NN_weights_784-59-39-26_epoch-11_accuracy-78.1892";

	//std::string testSetFileName = "/Users/eclown/Desktop/projects/NeuralNetworkCPP/emnist-letters-train.csv";

	std::string testSetFileName = "../emnist-letters-test.csv";
	DataSet testSet = DataSet(784, 26);
	testSet.LoadFromCSV(testSetFileName, ',', 0, false);
	std::vector<int> topology = { 784, 59, 39, 26 };
	NeuralNetworkBase *nn = new MatrixNeuralNetwork(topology);
	nn->LoadWeight(weightFileName);

	CheckTestSet(testSet.trainInputs, testSet.trainTargets, nn);
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
	nn->SaveWeights(0, 0, "");
	delete nn;
	nn = new MatrixNeuralNetwork(topology);
	nn->LoadWeight("/Users/eclown/Desktop/projects/NeuralNetworkCPP/cmake-build-debug/NN_weights_5-4-3-2_epoch-0_accuracy-0");
	nn->Sigmoid(1);
}

int main2(int argc, char *argv[])
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
	std::string fileName = std::string("../emnist-letters-train.csv");
	if (argc >= 4)
		fileName = argv[3];
	//std::string fileName = std::string("/Users/user/Desktop/projects/NN/emnist-letters-Train.csv");

	std::vector<int> topology = { 784, 151, 75, 26 };
	NeuralNetworkBase *nn = new MatrixNeuralNetwork(topology);
	//neuralNetwork.LoadWeight("NN_weights_5-4-3-2_epoch-0_accuracy-0");
	//neuralNetwork.SaveWeights(0, 0);
	nn->SetLearningRate(learningRate);
	std::cout << "Neural network initialized in " <<  sw.Restart() << std::endl;


//	std::string testSetFileName = "../emnist-letters-test.csv";
//	DataSet testSet = DataSet(784, 26);
//	testSet.LoadFromCSV(testSetFileName, ',', 0, false);
//	std::cout << "Test set loaded in ... " <<  sw.Restart() << std::endl;

	DataSet ts = DataSet(784, 26);
	ts.LoadFromCSV(fileName, ',', 1000, false);
	ts.SetValidationPartRatio(0.9);
	ts.Shuffle();
	std::cout << "Train set loaded in ... " <<  sw.Restart() << std::endl;

	auto testSetInput = ts.validationInputs;
	auto testSetAnswers = ts.validationTargets;


    std::cout << "- - - - - - - - - - - -" << std::endl;
    std::cout << "Topology: ";
    PrintTopology(topology);
    std::cout << "Train set size: " << ts.trainTargets.size() << std::endl;
    std::cout << "Test set size: " << testSetInput.size() << std::endl;
    std::cout << "Learning rate: " << nn->GetLearningRate() << std::endl;
    std::cout << "Learning rate ratio: " << learningRateRatio << std::endl;
    std::cout << "- - - - - - - - - - - -" << std::endl;

    std::cout << "BEFORE Learning:" << " ";
    CheckTestSet(testSetInput, testSetAnswers, nn);
    std::cout << std::endl;
    ts.Shuffle();
	for (int i = 1; i <= 50; ++i)
	{
		sw.Start();
		nn->Train(ts.trainInputs, ts.trainTargets, 1);
		std::cout << "epoch " << i << " done in " << sw.Stop() << " ";
		double accuracy = CheckTestSet(testSetInput, testSetAnswers, nn);
		if (accuracy > 50.0)
			nn->SaveWeights(accuracy, i, "");
		ts.Shuffle();
		nn->SetLearningRate(nn->GetLearningRate() * learningRateRatio);
	}
	return 0;
}

int main()
{
	NeuralNetworkManager nnm = NeuralNetworkManager();

	std::string  trainSetFileName = "../emnist-letters-train.csv";
	nnm.LoadTrainSet(trainSetFileName, 784, 26, 10000);

	std::vector<int> topology = { 784, 151, 75, 26 };
	nnm.LoadMatrixNN(topology);
	nnm.SetValidationPartOfTrainingDataset(0.2);


	nnm.LoadWeightToNetwork("NN_weights_784-151-75-26_epoch-1_accuracy-62.2905");

	/*
	//ОБУЧЕНИЕ
	StopWatch sw;
	sw.Start();


	std::vector<double> learningRatios {0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.025, 0.02, 0.015};
	//std::vector<double> learningRatios {0.05, 0.04, 0.03, 0.025, 0.02, 0.015};
	for (int i = 0; i <learningRatios.size(); ++i)
	{
		nnm.Train(1, learningRatios[i]);
		std::cout << "Error: " << nnm.getError() << std::endl;
	}


	nnm.SaveWeightFromNetwork(0, 0, "NN_weights_784-500-405-26_epoch-17_accuracy-93.4234");
	std::cout << "Training complete in " << sw.Stop()  << std::endl;
	*/

	size_t result;
	size_t trueAnswer;
	int totalTests = 1000;
	int rightAnswers = 0;
	for (int i = 0; i < totalTests; ++i)
	{
		result = nnm.Predict(nnm.trainingSet->trainInputs[i], 1, false);
		auto _tmp = nnm.trainingSet->trainTargets[i];
		trueAnswer = std::max_element(_tmp.begin(),_tmp.end()) - _tmp.begin() + 1;
		std::cout << "Predicted: " << result << " | True: " << trueAnswer << std::endl;
		if (trueAnswer == result)
			rightAnswers++;
	}
	std::cout << "- - - - -" << std::endl << "right: " << rightAnswers << " from " << totalTests << std::endl;


}