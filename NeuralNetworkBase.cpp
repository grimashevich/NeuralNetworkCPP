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


NeuralNetworkBase::~NeuralNetworkBase()
= default;
