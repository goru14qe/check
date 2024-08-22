#ifndef LATTICE_OPS_H
#define LATTICE_OPS_H

#include <unordered_set>
#include "Tensor.h"
#include "Parallel.h"
#include "Geometry.h"
#include "utils/Assert.h"

class Non_solid_update {
public:
	void compute_intervals(const Solid_field& is_solid, const Parallel_MPI& parallel_MPI);

	// Invoke fn with signature void operator()(Flat_index idx)
	// on each non solid cell of the lattice.
	template <typename Fn>
	void update(Fn fn) const {
		for (int interval = 0; interval + 1 < non_solid_intervals.size(); interval += 2) {
#pragma omp simd
			for (Flat_index idx = non_solid_intervals[interval]; idx < non_solid_intervals[interval + 1]; ++idx) {
				fn(idx);
			}
		}
	}

	// In some cases vectorization can degrade performance.
	// An example for that is 2D mode, because vectorization happens along the z-axis.
	template <typename Fn>
	void update_no_simd(Fn fn) const {
		for (int interval = 0; interval + 1 < non_solid_intervals.size(); interval += 2) {
			for (Flat_index idx = non_solid_intervals[interval]; idx < non_solid_intervals[interval + 1]; ++idx) {
				fn(idx);
			}
		}
	}

private:
	std::vector<Flat_index> non_solid_intervals;
};

namespace interpolation {

struct Stencil {
	Stencil() = default;
	// @param base_idx point that is excluded from the interpolation
	// @param pos local non-dimensional position for which values are interpolated
	// @param radius non dimensional radius of influence
	// @param is_2d consider only nodes on the xy-plane
	// @param exclude set of lattice points to exclude in the stencil
	Stencil(const Index_vec3& base_idx, const Vec3& pos, const Solid_field& is_solid, double radius, bool is_2d, const std::unordered_set<Index_vec3>& exclude = {});

	bool is_valid() const { return !nodes.empty(); }
	double interpolate(const Scalar_field& field) const;
	double interpolate(const Vector_field& field, int component) const;

	struct Node {
		Flat_index flat_idx;
		double weight;
	};

	std::vector<Node> nodes;
};

}  // namespace interpolation

#endif