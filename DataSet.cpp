#include "DataSet.h"
#include "StopWatch.h"

DataSet::DataSet(int inputSize, int answerSize): rng(std::random_device{}())
{
	this->inputSize = inputSize;
	this->outputSize = answerSize;
	answerOffset = 0;
	testSetSizeRatio = 0;
	std::srand(std::time(nullptr));
}

size_t DataSet::Size() const
{
	return inputSignals.size();
}

void DataSet::LoadFromCSV(std::string& filePath, char delimiter, int lineLimit, bool skipFirstLine)
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

std::vector<double> DataSet::GetVectorAnswer(int rightClassNum) const
{
	std::vector<double> answer(outputSize, 0);
	answer[rightClassNum] = 1;
	return answer;
}

void DataSet::MoveToTestSet(float movePercentage)
{
	testSetSizeRatio = movePercentage;
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

/*	for (int i = 0; i < countToMove; ++i)
	{
		testSetInputSignals.push_back(inputSignals[i]);
		testSetAnswers.push_back(answers[i]);
	}
	inputSignals.erase(inputSignals.begin(), inputSignals.begin() + countToMove);
	answers.erase(answers.begin(), answers.begin() + countToMove);*/
}

double DataSet::normalizeInput(double n, double limit)
{
	if (n >= limit)
		return 1;
	return 0;
}



void DataSet::Shuffle()
{
	//auto sw = StopWatch();
	//sw.Start();
	ReturnTestSetToTrainSet();

	//std::cout <<  sw.Restart() << " return complete" << std::endl;

	size_t setSize, rnd1, rnd2;
	setSize = inputSignals.size();
	for (size_t i = 0; i < setSize * 2; ++i)
	{
		rnd1 = GetRandomNumber(0, static_cast<int>(setSize) - 1);
		rnd2 = GetRandomNumber(0, static_cast<int>(setSize) - 1);
		inputSignals[rnd1].swap(inputSignals[rnd2]);
		answers[rnd1].swap(answers[rnd2]);
	}

	//std::cout <<  sw.Restart() << " shuffle complete" << std::endl;

	MoveToTestSet(testSetSizeRatio);

	//std::cout <<  sw.Restart() << " move to test set complete" << std::endl;
}

void DataSet::ReturnTestSetToTrainSet()
{
	for (size_t i = 0; i < testSetInputSignals.size(); ++i)
	{
		inputSignals.push_back(testSetInputSignals[i]);
		answers.push_back(testSetAnswers[i]);
	}
	testSetInputSignals.clear();
	testSetAnswers.clear();
}


int DataSet::GetRandomNumber(int min, int max)
{
    std::uniform_int_distribution<int> dist(min, max);
    return dist(rng);
}

float DataSet::GetTestSetSizeRatio() const
{
	return testSetSizeRatio;
}

void DataSet::SetTestSetSizeRatio(float newTestSetSizeRatio)
{
	if (newTestSetSizeRatio >= 0 && newTestSetSizeRatio <= 1)
		testSetSizeRatio = newTestSetSizeRatio;
}

int DataSet::GetInputSize() const
{
	return inputSize;
}

int DataSet::GetOutputSize() const
{
	return outputSize;
}