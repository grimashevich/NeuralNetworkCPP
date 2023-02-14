#ifndef NN_NNMANAGER_H
#define NN_NNMANAGER_H

#include "NnBase.h"
#include "NN.h"
#include "TrainingSet.h"
#include "StopWatch.h"

class NnManager
{
private:
	int inputSizeNN = 0;
	int outputSizeNN = 0;
	NnBase *nn = nullptr;
	TrainingSet *trainingSet = nullptr;
	TrainingSet *testSet = nullptr;
	double precision = 0;
	double recall = 0;
	double fMeasure = 0;

public:
	NnManager();
	void LoadMatrixNN();
	void LoadGraphNN();

	[[nodiscard]] double getPrecision() const;
	[[nodiscard]] double getRecall() const;
	[[nodiscard]] double getFMeasure() const;
	[[nodiscard]] int getInputSizeNn() const;
	[[nodiscard]] int getOutputSizeNn() const;
};


#endif //NN_NNMANAGER_H
