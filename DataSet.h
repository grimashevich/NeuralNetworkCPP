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
	std::vector<std::vector<double>> inputSignals;
	std::vector<std::vector<double>> answers;
	std::vector<std::vector<double>> testSetInputSignals;
	std::vector<std::vector<double>> testSetAnswers;

	explicit DataSet(int inputSize, int answerSize);
	size_t Size() const;
	void LoadFromCSV(std::string& filePath, char delimiter, int lineLimit = 0, bool skipFirstLine = true);
	void MoveToTestSet(float movePercentage);
	void ReturnTestSetToTrainSet();
	int answerOffset = -1; // Смещение класса ответов в выборке (-1, если для 0-го класса в выборке ответ 1)
	void Shuffle();
    int GetRandomNumber(int min, int max);

private:
public:
	int GetInputSize() const;

	int GetOutputSize() const;

private:
	int inputSize;
	int outputSize;
    float testSetSizeRatio;
public:
	float GetTestSetSizeRatio() const;

	void SetTestSetSizeRatio(float newTestSetSizeRatio);

private:
	std::mt19937 rng;

    [[nodiscard]] std::vector<double> GetVectorAnswer(int rightClassNum) const;
	static double normalizeInput(double n, double limit);


};


#endif //NN_DATASET_H
