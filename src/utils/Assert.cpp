#include "Assert.h"
#include <iostream>

namespace details {
void assert_handler(const std::string& file, long line, const std::string& function_name,
                    const std::string& condition_name, const std::string& error_message) {
	std::string message("Assertion failed in file \"" + file + "\" line " + std::to_string(line) + " (function \"" + function_name + "\"): " + condition_name + "\n" + error_message);
	std::cerr << "[Error] " << message << std::endl;
}
}  // namespace details