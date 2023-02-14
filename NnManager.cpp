//
// Created by Elyas Clown on 14.02.2023.
//

#include "NnManager.h"

NnManager::NnManager()
{

}

double NnManager::getPrecision() const
{
	return precision;
}

double NnManager::getRecall() const
{
	return recall;
}

double NnManager::getFMeasure() const
{
	return fMeasure;
}

int NnManager::getInputSizeNn() const
{
	return inputSizeNN;
}

int NnManager::getOutputSizeNn() const
{
	return outputSizeNN;
}

