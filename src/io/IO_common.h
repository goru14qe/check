#ifndef IO_COMMON_H
#define IO_COMMON_H

#include "../Tensor.h"
#include <string>
#include <vector>

enum struct OPEN_MODE {
	WRITE,
	READ
};

struct IO_dataset_info {
	std::string name;  // name of the field
	double scaling;    // scaling parameter for S.I. units
};

class IO_exception : public std::exception {
public:
	IO_exception(const std::string& msg)
		: message(msg) {}

	const char* what() const noexcept override { return message.c_str(); }

private:
	std::string message;
};

template <typename Tensor_expr>
struct Domain_map {
	Slice_view<Tensor_expr> dest;
	typename Tensor_expr::Index_vec src_offset;
};

// Decomposes a tensor placed onto a periodic domain such that each chunk is inside that domain.
// @param offset global offset of the slice (can be negative)
// @param global_size actual size of the source
// @return List of slices together with the valid source area on the domain.
template <typename Tensor_expr>
std::vector<Domain_map<Tensor_expr>> decompose_periodic(const Tensor_expr& expr,
                                                        const Index_vec3& offset,
                                                        const Index_vec3& global_size);

// ******************************************************************* //
//  Implementation
// ******************************************************************* //
namespace details {

template <typename Tensor_expr>
void decompose_dim(std::vector<Domain_map<Tensor_expr>>& slices, const Index_vec3& global_size, int dim) {
	using Index_vec = typename Tensor_expr::Index_vec;
	const size_t n = slices.size();

	for (size_t i = 0; i < n; ++i) {
		Index_vec sizes = slices[i].dest.sizes();
		if (slices[i].src_offset[dim] < 0) {
			// add new slice
			Index_vec new_offset = slices[i].src_offset;
			new_offset[dim] = global_size[dim] + slices[i].src_offset[dim];
			Index_vec new_size = sizes;
			new_size[dim] = -slices[i].src_offset[dim];
			slices.push_back({slice(slices[i].dest.base_expr(), slices[i].dest.offset(), new_size), new_offset});

			// truncate old slice
			Index_vec dst_offset = slices[i].dest.offset();
			dst_offset[dim] = new_size[dim];
			sizes[dim] -= new_size[dim];
			slices[i].dest.resize(dst_offset, sizes);
			slices[i].src_offset[dim] = 0;
		}

		if (slices[i].src_offset[dim] + sizes[dim] > global_size[dim]) {
			// add new slice
			Index_vec new_offset = slices[i].src_offset;
			new_offset[dim] = 0;
			Index_vec new_size = sizes;
			new_size[dim] = (slices[i].src_offset[dim] + sizes[dim]) - global_size[dim];
			Index_vec dst_offset = slices[i].dest.offset();
			dst_offset[dim] += sizes[dim] - new_size[dim];
			slices.push_back({slice(slices[i].dest.base_expr(), dst_offset, new_size), new_offset});

			// truncate old slice
			Index_vec dst_size = slices[i].dest.sizes();
			dst_size[dim] -= new_size[dim];
			slices[i].dest.resize(slices[i].dest.offset(), dst_size);
		}
	}
}
}  // namespace details

// @param offset global offset of the slice (can be negative)
// @param global_size actual size of the source
// @return
template <typename Tensor_expr>
std::vector<Domain_map<Tensor_expr>> decompose_periodic(const Tensor_expr& expr,
                                                        const Index_vec3& offset,
                                                        const Index_vec3& global_size) {
	std::vector<Domain_map<Tensor_expr>> slices;
	typename Tensor_expr::Index_vec zero_offset = {};

	auto ext_offset = zero_offset;
	for (int i = 0; i < 3; ++i)
		ext_offset[i] = offset[i];
	slices.push_back({slice(expr, zero_offset, expr.sizes()), ext_offset});
	
	details::decompose_dim(slices, global_size, 0);
	details::decompose_dim(slices, global_size, 1);
	details::decompose_dim(slices, global_size, 2);

	return slices;
}

#endif