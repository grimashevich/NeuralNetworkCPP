#ifndef NN_TRAININGSET_H
#define NN_TRAININGSET_H

#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <string>
#include <algorithm>
#include <random>

class TrainingSet {
public:
	std::vector<std::vector<double>> inputSignals;
	std::vector<std::vector<double>> answers;
	std::vector<std::vector<double>> testSetInputSignals;
	std::vector<std::vector<double>> testSetAnswers;

	explicit TrainingSet(int inputSize, int answerSize);
	size_t Size() const;
	void LoadFromCSV(std::string& filePath, char delimiter, int lineLimit = 0, bool skipFirstLine = true);
	void MoveToTestSet(float movePercentage);
	void ReturnTestSetToTrainSet();
	int answerOffset; // Смещение класса ответов в выборке (-1, если для 0-го класса в выборке ответ 1)
	void Shuffle();
    float getTestSetSizePerc() const;
    void setTestSetSizePerc(float testSetSizePerc);
    int getRandomNumber(int min, int max);
private:
	int inputSize;
	int answerSize;
    float testSetSizePerc;
    std::mt19937 rng;

    [[nodiscard]] std::vector<double> GetVectorAnswer(int rightClassNum) const;
	static double normalizeInput(double n, double limit);


};


#endif //NN_TRAININGSET_H
