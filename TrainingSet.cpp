#include "TrainingSet.h"

TrainingSet::TrainingSet(int inputSize, int answerSize)
{
	this->inputSize = inputSize;
	this->answerSize = answerSize;
	answerOffset = 0;

}

size_t TrainingSet::Size()
{
	return inputSignals.size();
}

void TrainingSet::LoadFromCSV(std::string& filePath, char delimiter, int lineLimit, bool skipFirstLine)
{
	//TODO Check is file exist and readable
	std::ifstream csvFile(filePath);
	std::string line;
	int lineCount = 0;

	inputSignals.clear();
	answers.clear();

	if (skipFirstLine)
		getline(csvFile, line);
	while (getline(csvFile, line))
	{
		std::string answer, other;
		std::vector<double> inputSignal = std::vector<double>();
		int answerInt;
		std::stringstream stream(line);

		getline(stream, answer, delimiter);
		answerInt = std::stoi(answer) + answerOffset;
		answers.push_back(GetVectorAnswer(answerInt));


		for (int i = 0; i < inputSize; ++i)
		{
			getline(stream, other, delimiter);
			inputSignal.push_back(normalizeInput(std::stod(other), 64));

		}
		inputSignals.push_back(inputSignal);
		lineCount++;
		if (lineLimit > 0 && lineCount >= lineLimit)
			break;
	}
	csvFile.close();
}

std::vector<double> TrainingSet::GetVectorAnswer(int rightClassNum) const
{
	std::vector<double> answer(answerSize, 0);
	answer[rightClassNum] = 1;
	return answer;
}

void TrainingSet::MoveToTestSet(float movePercentage)
{
	int countToMove = (int)((float) answers.size() * movePercentage);
	if (movePercentage >= 1.0)
		countToMove = (int) answers.size();
	for (int i = 0; i < countToMove; ++i)
	{
		testSetInputSignals.push_back(inputSignals[inputSignals.size() - 1]);
		inputSignals.pop_back();
		testSetAnswers.push_back(answers[answers.size() - 1]);
		answers.pop_back();
	}
}

double TrainingSet::normalizeInput(double n, double limit)
{
	if (n >= limit)
		return 1;
	return 0;
}
