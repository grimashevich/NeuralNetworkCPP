#ifndef NEURALNETWORKCPP_STOPWATCH_H
#define NEURALNETWORKCPP_STOPWATCH_H

#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>

class StopWatch
{
private:
	std::chrono::time_point<std::chrono::steady_clock> start;
	std::chrono::time_point<std::chrono::steady_clock> end;
	long long duration;
public:
	[[nodiscard]] long long int getDuration() const;

public:
	StopWatch();
	void Start();
	std::string Stop();

};


#endif //NEURALNETWORKCPP_STOPWATCH_H
