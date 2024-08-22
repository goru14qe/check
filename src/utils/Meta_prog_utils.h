#ifndef META_PROG_UTILS_H
#define META_PROG_UTILS_H

namespace detail {

template <class T, std::size_t N, class... Args>
struct get_element_idx_from_tuple_by_type_impl {
	static constexpr auto value = N;
};

template <class T, std::size_t N, class... Args>
struct get_element_idx_from_tuple_by_type_impl<T, N, T, Args...> {
	static constexpr auto value = N;
};

template <class T, std::size_t N, class U, class... Args>
struct get_element_idx_from_tuple_by_type_impl<T, N, U, Args...> {
	static constexpr auto value = get_element_idx_from_tuple_by_type_impl<T, N + 1, Args...>::value;
};

}  // namespace detail

// Alternative for std::get.
// Access of tuple elements by type is only available in C++14.
template <class T, class... Args>
T& get_element_by_type(std::tuple<Args...>& t) {
	return std::get<detail::get_element_idx_from_tuple_by_type_impl<T, 0, Args...>::value>(t);
}

#endif