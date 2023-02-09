#include "TrainingSet.h"
#include "StopWatch.h"

TrainingSet::TrainingSet(int inputSize, int answerSize)
{
	this->inputSize = inputSize;
	this->answerSize = answerSize;
	answerOffset = 0;
	testSetSizePerc = 0;
	std::srand(std::time(nullptr));
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
	testSetSizePerc = movePercentage;
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

double TrainingSet::normalizeInput(double n, double limit)
{
	if (n >= limit)
		return 1;
	return 0;
}



void TrainingSet::Shuffle()
{
	//auto sw = StopWatch();
	//sw.Start();
	ReturnTestSetToTrainSet();

	//std::cout <<  sw.Restart() << " return complete" << std::endl;

	size_t setSize, rnd1, rnd2;
	setSize = inputSignals.size();
	for (size_t i = 0; i < setSize * 2; ++i)
	{
		rnd1 = rand() % setSize;
		rnd2 = rand() % setSize;
		inputSignals[rnd1].swap(inputSignals[rnd2]);
		answers[rnd1].swap(answers[rnd2]);
	}

	//std::cout <<  sw.Restart() << " shuffle complete" << std::endl;

	MoveToTestSet(testSetSizePerc);

	//std::cout <<  sw.Restart() << " move to test set complete" << std::endl;
}

void TrainingSet::ReturnTestSetToTrainSet()
{
	for (size_t i = 0; i < testSetInputSignals.size(); ++i)
	{
		inputSignals.push_back(testSetInputSignals[i]);
		answers.push_back(testSetAnswers[i]);
	}
	testSetInputSignals.clear();
	testSetAnswers.clear();
}
