#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <vector>
#include <string>

class Timer {
public:
	using FloatT = double;
	struct Options {
		bool print_results = false;
		bool store_history = false;
		uint32_t print_period = 0xffffffffu;
	};

	static Options options;

	Timer(const std::string& name);
	~Timer();

	void start();
	void stop();
	// Adds a dummy value to the history to identify specific events later on.
	void sentinel();

	FloatT get_total_time() const { return m_total_time; }
private:
	std::string m_name;

	using Clock = std::chrono::high_resolution_clock;
	FloatT m_total_time;
	uint32_t m_count;
	std::chrono::time_point<Clock> m_start;
	std::vector<FloatT> m_timings;
};


#endif