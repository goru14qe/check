#include "Timer.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>

Timer::Options Timer::options;

Timer::Timer(const std::string& name)
	: m_name(name)
	, m_total_time(0.0)
	, m_count(0) {}

Timer::~Timer() {
	if (options.store_history) {
		const std::string file_name = m_name + "_timings.txt";
		std::cout << "Writing full timing log to file " << file_name << "\n";
		std::ofstream file(file_name);
		file.precision(12);
		for (FloatT t : m_timings)
			file << t << "\n";
	}
	if (options.print_results) {
		std::cout << std::fixed << std::setprecision(6)
				  << m_name << " avg " << m_total_time / m_count
				  << " total " << m_total_time << "\n";
	}
}

void Timer::start() {
	m_start = Clock::now();
}

void Timer::stop() {
	auto end = Clock::now();
	const FloatT dur = std::chrono::duration<FloatT>(end - m_start).count();
	m_total_time += dur;
	++m_count;

	if (options.store_history){
		m_timings.push_back(dur);
	}

	if (m_count % options.print_period == 0 && options.print_results) {
		std::cout << m_name << " cur " << dur
				  << " avg " << m_total_time / m_count
				  << " total " << m_total_time << "\n";
	}
}

void Timer::sentinel() {
	m_timings.push_back(-1.0);
}