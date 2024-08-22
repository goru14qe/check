#include "Config_utils.h"
#include <iostream>
#include <stdio.h>
#include <dirent.h>
#include <unistd.h>

void find_line_after_header(std::ifstream& input_file, const std::string& header) {
	std::string Line1;
	input_file.clear();
	input_file.seekg(0, std::ios::beg);
	while (std::getline(input_file, Line1)) {
		if (Line1.find(header) != std::string::npos) {
			return;
		}
	}
	std::cerr << "[Warning] Could not find the requested header \""
			  << header << "\" in the input file.\n";
}

void find_line_after_comment(std::ifstream& input_file) {
	char comment_indicator;
	std::string Line1;

	do {
		int beg_line = input_file.tellg();
		input_file >> comment_indicator;
		if (comment_indicator != '#') {
			input_file.seekg(beg_line);
			return;
		}
	} while (std::getline(input_file, Line1, '\n'));

	std::cerr << "[Warning] Reached the end of the input file looking for a non-comment line.\n";
}

std::string get_current_working_directory() {
#if defined _WIN64 || _WIN32
	char* cwd = _getcwd(0, 0);  // **** microsoft specific ****
#endif
#if defined __linux__ || __APPLE__
	char* cwd = getcwd(0, 0);
#endif
	std::string working_directory(cwd);
	std::free(cwd);
	return working_directory;
}
