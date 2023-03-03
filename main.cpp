#include <cfloat>
#include "MatrixNeuralNetwork.h"
#include <chrono>
#include "DataSet.h"
#include "StopWatch.h"
#include <thread>
#include "NeuralNetworkManager.h"

void RandomCrossVal()
{
}

int main()
{
/*    sandBox();
    return 0;*/


    NeuralNetworkManager nnm = NeuralNetworkManager();

	std::string  trainSetFileName = "../emnist-letters-train.csv";
	//trainSetFileName = "../emnist-letters-test.csv";s


	std::vector<int> topology = { 784, 150, 75, 26 };
	nnm.LoadMatrixNN(topology);
	//nnm.LoadWeightToNetwork("../NN_weights_784-500-405-26_epoch-17_accuracy-93.weights");
	//nnm.LoadWeightToNetwork("../NN_weights_784-151-75-26_epoch-3_accuracy-73.4865");

	nnm.SetValidationPartOfTrainingDataset(0.2);
	nnm.LoadTrainSet(trainSetFileName, 784, 26, 1000);

    nnm.CalculateMetricsForTestSet(nnm.trainingSet->trainInputs, nnm.trainingSet->trainTargets);
    nnm.PrintMetrics();
    nnm.CrossValidation(5, 0.05, 0.9);
    nnm.PrintMetrics();
    return 0;

	StopWatch sw;
	sw.Start();
	nnm.CalculateMetricsForTestSet(nnm.trainingSet->trainInputs,
								   nnm.trainingSet->trainTargets,0);
	std::cout << sw.Stop() << std::endl;
	return 0;



    nnm.CalculateMetricsForTestSet(nnm.trainingSet->trainInputs,
                                   nnm.trainingSet->trainTargets,4);

    std::cout << "Accuracy before learning: " << nnm.GetAccuracy() << std::endl;

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

		std::cout << "Accuracy: " << nnm.GetAccuracy() << " in " << sw.Restart() << std::endl;
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


}