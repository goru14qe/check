#ifndef CONFIG_UTILS_H
#define CONFIG_UTILS_H

#include <fstream>

void find_line_after_header(std::ifstream& input_file, const std::string& header);
void find_line_after_comment(std::ifstream& input_file);

constexpr int COLUMN_WIDTH = 40;

std::string get_current_working_directory();

#endif  // CONFIG_UTILS_H