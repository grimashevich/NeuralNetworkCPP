#include "NeuralNetworkBase.h"

//
// Created by Elyas Clown on 15.02.2023.
//
double NeuralNetworkBase::GetLearningRate() const
{
	return learningRate;
}

double NeuralNetworkBase::Sigmoid(double x)
{
	return 1.0 / (1.0 + exp(-x));
}

double NeuralNetworkBase::DSigmoid(double x)
{
	return x * (1 - x);
}

void NeuralNetworkBase::SetLearningRate(double newLearningRate)
{
	if (learningRate > 0)
		learningRate = newLearningRate;
}

const std::string NeuralNetworkBase::currentDateTime()
{
	time_t     now = time(0);
	struct tm  tstruct;
	char       buf[80];
	tstruct = *localtime(&now);
	// Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
	// for more information about date/time format
	strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

	return buf;
}


NeuralNetworkBase::~NeuralNetworkBase()
= default;
