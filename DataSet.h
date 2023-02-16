#ifndef NN_DATASET_H
#define NN_DATASET_H

#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <string>
#include <algorithm>
#include <random>

class DataSet
{
public:
	std::vector<std::vector<double>> trainInputs;
	std::vector<std::vector<double>> trainTargets;
	std::vector<std::vector<double>> validationInputs;
	std::vector<std::vector<double>> validationTargets;

	explicit DataSet(int inputSize, int answerSize);
	size_t Size() const;
	void LoadFromCSV(std::string& filePath, char delimiter, int lineLimit = 0, bool skipFirstLine = true);
	void MoveToTestSet(float movePercentage);
	void ReturnTestSetToTrainSet();
	int answerOffset = -1; // Смещение класса ответов в выборке (-1, если для 0-го класса в выборке ответ 1)
	void Shuffle();
    int GetRandomNumber(int min, int max);

	int GetInputSize() const;
	int GetOutputSize() const;
	float GetValidationPartRatio() const;
	void SetValidationPartRatio(float newTestSetSizeRatio);

private:
	int inputSize;
	int outputSize;
    float testSetSizeRatio;
	std::mt19937 rng;

    [[nodiscard]] std::vector<double> GetVectorAnswer(int rightClassNum) const;
	static double normalizeInput(double n, double limit);


};


#endif //NN_DATASET_H
