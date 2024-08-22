#ifndef ASSERT_H
#define ASSERT_H

#include "Debug_trap.h"
#include <string>

namespace details {
/// Intern handler for assertions.
void assert_handler(const std::string& file, long line, const std::string& function_name,
                    const std::string& condition_name, const std::string& error_message);
}  // namespace details

#ifndef NDEBUG
/// Default assert macro for all our needs. Use this instead of <cassert>
#define ASSERT(condition)                                                                \
	do {                                                                                 \
		if (!static_cast<bool>(condition)) {                                             \
			::details::assert_handler(__FILE__, __LINE__, __FUNCTION__, #condition, ""); \
			psnip_trap();                                                                \
		}                                                                                \
	} while (false)
#define ASSERT_EXT(condition, error_message)                                                          \
	do {                                                                                              \
		if (!static_cast<bool>(condition)) {                                                          \
			::details::assert_handler(__FILE__, __LINE__, __FUNCTION__, #condition, (error_message)); \
			psnip_trap();                                                                             \
		}                                                                                             \
	} while (false)
#else
/// Default assert macro for all our needs. Use this instead of <cassert>
#define ASSERT(condition) \
	do {                  \
	} while (false)
#define ASSERT_EXT(condition, errorMessage) \
	do {                                    \
	} while (false)
#endif

#endif