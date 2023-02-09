#include "StopWatch.h"

StopWatch::StopWatch()
{

}

void StopWatch::Start()
{
	duration = 0;
	start = std::chrono::high_resolution_clock::now();
}

std::string StopWatch::Stop()
{
	double result;
	std::string units;
	std::ostringstream strResult;
	end = std::chrono::high_resolution_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	duration = elapsed.count();
	if (duration > 1000000)
	{
		result = static_cast<double>(static_cast<double>(duration) / 1000000.0);
		result = std::round(result / 0.001) * 0.001;
		units = "s";
	}
	else if (duration > 1000)
	{
		result = static_cast<double>(static_cast<double>(duration) / 1000.0);
		result = std::round(result / 0.001) * 0.001;
		units = "ms";
	}
	else
	{
		result = static_cast<double>(duration);
		units = "microseconds.";
	}

	strResult << result << units;
	return strResult.str();
}

long long int StopWatch::getDuration() const
{
	return duration;
}

std::string StopWatch::Restart()
{
	std::string tmp = Stop();
	Start();
	return tmp;
}
