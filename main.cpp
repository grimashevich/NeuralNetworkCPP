#include <cfloat>
#include "MatrixNeuralNetwork.h"
#include <chrono>
#include "DataSet.h"
#include "StopWatch.h"
#include <thread>
#include "NeuralNetworkManager.h"

void RandomCrossVal()
{
    int max_neuron_count = 500;
    int min_neuron_count = 26;
    std::stringstream weight_file_name;
    weight_file_name << "cv_";

    std::vector<int> topology = { 784 };
    DataSet rnd = DataSet(1, 1);
    int hiddenLayersCount = rnd.GetRandomNumber(2, 5);

    int last_layer = max_neuron_count;
    for (int i = 0; i < hiddenLayersCount; ++i) {
        last_layer = rnd.GetRandomNumber(min_neuron_count * (5 - i), last_layer);
        topology.emplace_back(last_layer);
    }
    topology.emplace_back(26);
    double learning_rate = rnd.GetRandomNumber(10, 100) / 1000.0;
    double learning_rate_ratio = rnd.GetRandomNumber(50, 100) / 100.0;
    std::cout << "{ ";
    for (int i : topology) {
        std::cout << i << " ";
        weight_file_name << i << "_";
    }
    std::cout << "} ";
    std::cout << "LR:" << learning_rate << " LRR: " << learning_rate_ratio << std::endl;
    std::cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - -" << std::endl << std::endl;

    NeuralNetworkManager network_manager = NeuralNetworkManager();
    network_manager.LoadMatrixNN(topology);
    network_manager.LoadTrainSet("../emnist-letters-train.csv", 28 * 28, 26, 0);

    network_manager.CrossValidation(10, learning_rate, learning_rate_ratio);

    weight_file_name << "_f-score_" << network_manager.getFscore() << ".weights";
    network_manager.SaveWeightFromNetwork(0,0,weight_file_name.str());
    std::cout << "weights_saved :" << weight_file_name.str() << std::endl;
}


void metrics_test() {
    NeuralNetworkManager nnm = NeuralNetworkManager();
    std::vector<std::vector<size_t>> testVector = std::vector<std::vector<size_t>>();
    testVector.emplace_back(std::vector<size_t>() = {6, 2, 5});
    testVector.emplace_back(std::vector<size_t>() = {9, 3, 4});
    testVector.emplace_back(std::vector<size_t>() = {1, 6, 8});

    nnm.CalculateMetrics(testVector);
    exit(0);
}

std::string getLetter(std::vector<double> & vector) {
    int i = 0;
    while (vector[i] != 1.0) {
        i++;
    }
    char c = i + 65;
    std::string s = std::string(1, c);
    return s;
}

void paintLetterFromDataSet()
{
    NeuralNetworkManager nnm = NeuralNetworkManager();
    std::string  trainSetFileName = "../emnist-letters-train.csv";
    nnm.SetValidationPartOfTrainingDataset(0.0);
    nnm.LoadTrainSet(trainSetFileName, 784, 26, 100);

    for (int i = 0; i < nnm.trainingSet->trainInputs.size(); ++i) {
        for (int j = 0; j < nnm.trainingSet->trainInputs[i].size(); ++j) {
            if (j % 28 == 0)
                std::cout << std::endl;
            std::string s = "  ";
            if (nnm.trainingSet->trainInputs[i][j] == 1.0)
                s = "██";
            std::cout << s << " ";
        }
        std::cout << std::endl;
        std::cout << "THIS IS: " << getLetter(nnm.trainingSet->trainTargets[i]) << std::endl;
        std::cout << std::endl << "- - - - - " << std::endl;
    }
}


int main()
{
    //metrics_test();


    for (int i = 0; i < 1000; ++i) {
        RandomCrossVal();
    }
    return 0;

    //paintLetterFromDataSet();

    NeuralNetworkManager nnm = NeuralNetworkManager();


	//trainSetFileName = "../emnist-letters-test.csv";s


	//std::vector<int> topology = { 784, 150, 75, 26 };
	//nnm.LoadMatrixNN(topology);
	//nnm.LoadWeightToNetwork("../NN_weights_784-500-405-26_epoch-17_accuracy-93.weights");
	nnm.LoadWeightToNetwork("../NN_weights_784-151-75-26_epoch-3_accuracy-73.4865");

    std::string  trainSetFileName = "../emnist-letters-train.csv";
    nnm.SetValidationPartOfTrainingDataset(0.0);
	nnm.LoadTrainSet(trainSetFileName, 784, 26, 0);




    nnm.CalculateMetricsForTestSet(nnm.trainingSet->trainInputs, nnm.trainingSet->trainTargets);
    nnm.printMetrics("");
    return 0;
    nnm.printMetrics("");
    nnm.CrossValidation(10, 0.05, 0.9);
    std::cout << "Mean metrics:" << std::endl;
    nnm.printMetrics("");

    std::cout << "Actual metrics:" << std::endl;
    nnm.trainingSet->Shuffle();
    nnm.CalculateMetricsForTestSet(nnm.trainingSet->validationInputs, nnm.trainingSet->validationTargets);
    nnm.printMetrics("");
    return 0;

	StopWatch sw;
	sw.Start();
	nnm.CalculateMetricsForTestSet(nnm.trainingSet->trainInputs,
								   nnm.trainingSet->trainTargets,0);
	std::cout << sw.Stop() << std::endl;
	return 0;



    nnm.CalculateMetricsForTestSet(nnm.trainingSet->trainInputs,
                                   nnm.trainingSet->trainTargets,4);

    std::cout << "Accuracy before learning: " << nnm.getAccuracy() << std::endl;

	//ОБУЧЕНИЕ
	sw.Start();
	//std::vector<double> learningRatios {0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.025, 0.02, 0.015};
	std::vector<double> learningRatios {0.05, 0.04, 0.03};
	for (int i = 0; i <learningRatios.size(); ++i)
	{
		nnm.Train(1, learningRatios[i]);
        //nnm.trainWithMiniBatches(learningRatios[i], 1, 16);

        nnm.CalculateMetricsForTestSet(nnm.trainingSet->trainInputs,
                                       nnm.trainingSet->trainTargets,4);

		std::cout << "Accuracy: " << nnm.getAccuracy() << " in " << sw.Restart() << std::endl;
	}


	nnm.SaveWeightFromNetwork(0, 0, "NN_weights_784-500-405-26_epoch-17_accuracy-93.weights");
	std::cout << "Training complete in " << sw.Stop()  << std::endl;

	sw.Start();
	size_t result;
	size_t trueAnswer;
	size_t totalTests = nnm.trainingSet->trainInputs.size();
	int rightAnswers = 0;
	for (int i = 0; i < totalTests; ++i)
	{
		result = nnm.Predict(nnm.trainingSet->trainInputs[i], 1, false);
		auto _tmp = nnm.trainingSet->trainTargets[i];
		trueAnswer = std::max_element(_tmp.begin(),_tmp.end()) - _tmp.begin() + 1;
		//std::cout << "Predicted: " << result << " | True: " << trueAnswer << std::endl;
		if (trueAnswer == result)
			rightAnswers++;
	}
	std::cout << "- - - - -" << std::endl << "right: " << rightAnswers << " from " << totalTests << std::endl;
	std::cout <<  "Done in " << sw.Stop() << std::endl;



    /*
       A   B   C   D   E   F   G   H   I   J   K   L   M   N   O   P   Q   R   S   T   U   V   W   X   Y   Z   Sum  TP  FP
   A  71   1   0   0   0   0   1   2   0   0   1   0   2   6   6   0   7   0   0   1   1   0   0   0   0   1   100  71  29
   B   5  77   0   0   0   0   1   0   0   0   0   0   0   0   3   0   3   0   0   1   0   0   0   0   1   2    93  77  16
   C   0   3  95   0   3   0   0   0   0   0   0   0   0   0   2   0   0   2   0   3   1   0   0   0   0   2   111  95  16
   D   2   4   0  80   0   0   1   0   0   1   0   0   0   1   4   1   1   0   0   1   1   0   0   0   2   0    99  80  19
   E   6   3  15   0  65   1   0   0   0   0   0   0   0   3   2   0   0   1   0   1   0   1   0   0   0   2   100  65  35
   F   2   1   1   0   0  65   2   0   1   0   0   0   0   0   0   9   0   2   0   6   0   0   1   0   0   2    92  65  27
   G   4   3   1   2   0   0  51   0   1   0   0   0   0   1   1   2  10   1   1   2   0   0   0   0   6   0    86  51  35
   H   4   4   0   1   0   0   0  74   2   0   1   0   0  10   0   0   0   2   0   1   3   0   1   0   1   1   105  74  31
   I   1   1   1   1   0   0   0   0  71   1   0  16   0   0   0   0   0   0   0   0   0   0   0   1   0   1    94  71  23
   J   0   1   1   3   0   0   2   0   4  77   0   0   0   0   0   0   0   0   1   5   0   0   0   0   2   1    97  77  20
   K   2   0   1   1   0   0   0   3   3   0  70   0   0   4   0   1   0   2   0   1   4   0   1   1   0   0    94  70  24
   L   0   2   3   0   0   0   0   0  47   0   0  61   0   0   0   0   0   1   0   1   1   0   0   1   0   0   117  61  56
   M   4   0   0   0   0   0   0   0   0   0   1   0  84  12   0   0   1   0   0   1   2   0   0   0   0   1   106  84  22
   N   5   0   0   0   0   0   1   1   0   0   3   0   0  78   0   0   0   0   0   0   6   2   0   1   2   0    99  78  21
   O   2   1   0   2   0   0   0   0   0   0   0   0   0   1  90   0   2   0   0   0   1   0   0   0   0   0    99  90   9
   P   1   0   0   0   0   2   1   0   0   0   0   0   0   1   0  76   1   3   0   0   0   0   0   0   3   0    88  76  12
   Q  16   0   1   1   0   4  19   0   1   0   1   0   0   0   3   2  46   0   0   0   1   0   0   0   7   2   104  46  58
   R  11   1   0   0   3   1   0   1   0   0   2   0   0   0   1   1   1  69   0   2   0   0   0   2   4   2   101  69  32
   S   6   4   0   0   1   1  10   0   0   1   0   0   0   0   1   0   2   0  83   0   0   0   0   0   0   0   109  83  26
   T   0   1   0   0   0   1   0   0   1   0   0   0   0   0   0   0   1   1   0  97   0   0   0   0   2   1   105  97   8
   U   1   0   0   0   0   0   1   0   0   0   2   0   0   3   0   0   0   0   0   0 106   3   0   0   2   0   118 106  12
   V   0   0   0   1   0   0   0   1   0   0   3   0   1   2   0   0   0   0   0   3   6  82   0   0   6   0   105  82  23
   W   2   0   1   1   0   0   0   0   1   0   2   0   0   4   1   0   0   0   0   0   3   2  91   0   0   0   108  91  17
   X   3   0   0   0   0   0   0   0   0   0   2   0   0   0   0   0   0   0   0   0   0   2   0  66   3   2    78  66  12
   Y   1   1   0   0   0   0   2   0   2   0   0   0   1   0   0   0   0   1   0   3   0   3   0   0  86   0   100  86  14
   Z   1   2   1   0   0   0   1   0   0   0   1   0   0   0   0   0   2   0   0   2   0   0   0   1   0  81    92  81  11
     * */


}