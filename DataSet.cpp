#include "DataSet.h"
#include "StopWatch.h"

DataSet::DataSet(size_t inputSize, size_t answerSize): rng(std::random_device{}())
{
	this->inputSize = inputSize;
	this->outputSize = answerSize;
	testSetSizeRatio = 0;
	std::srand(std::time(nullptr));
}

size_t DataSet::Size() const
{
	return trainInputs.size();
}

void DataSet::LoadFromCSV(std::string& filePath, char delimiter, size_t lineLimit, bool skipFirstLine)
{
	//TODO Check is file exist and readable
	std::ifstream csvFile(filePath);
	std::string line;
	int lineCount = 0;

	trainInputs.clear();
	trainTargets.clear();

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
		trainTargets.push_back(GetVectorAnswer(answerInt));


		for (int i = 0; i < inputSize; ++i)
		{
			getline(stream, other, delimiter);
			inputSignal.push_back(normalizeInput(std::stod(other), 64));

		}
		trainInputs.push_back(inputSignal);
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
	int countToMove = (int)((float) trainTargets.size() * movePercentage);
	if (movePercentage >= 1.0)
		countToMove = (int) trainTargets.size();
	for (int i = 0; i < countToMove; ++i)
	{
		validationInputs.push_back(trainInputs[trainInputs.size() - 1]);
		trainInputs.pop_back();
		validationTargets.push_back(trainTargets[trainTargets.size() - 1]);
		trainTargets.pop_back();
	}

/*	for (int i = 0; i < countToMove; ++i)
	{
		validationInputs.push_back(trainInputs[i]);
		validationTargets.push_back(trainTargets[i]);
	}
	trainInputs.erase(trainInputs.begin(), trainInputs.begin() + countToMove);
	trainTargets.erase(trainTargets.begin(), trainTargets.begin() + countToMove);*/
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
	setSize = trainInputs.size();
	for (size_t i = 0; i < setSize * 2; ++i)
	{
		rnd1 = GetRandomNumber(0, static_cast<int>(setSize) - 1);
		rnd2 = GetRandomNumber(0, static_cast<int>(setSize) - 1);
		trainInputs[rnd1].swap(trainInputs[rnd2]);
		trainTargets[rnd1].swap(trainTargets[rnd2]);
	}

	//std::cout <<  sw.Restart() << " shuffle complete" << std::endl;

	MoveToTestSet(testSetSizeRatio);

	//std::cout <<  sw.Restart() << " move to test set complete" << std::endl;
}

void DataSet::ReturnTestSetToTrainSet()
{
	for (size_t i = 0; i < validationInputs.size(); ++i)
	{
		trainInputs.push_back(validationInputs[i]);
		trainTargets.push_back(validationTargets[i]);
	}
	validationInputs.clear();
	validationTargets.clear();
}


int DataSet::GetRandomNumber(int min, int max)
{
    std::uniform_int_distribution<int> dist(min, max);
    return dist(rng);
}

float DataSet::GetValidationPartRatio() const
{
	return testSetSizeRatio;
}

void DataSet::SetValidationPartRatio(float newTestSetSizeRatio)
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