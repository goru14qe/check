#include "Lattice_ops.h"
#include "ALBORZ_Macros.h"

//the function identifies intervals of non-solid regions in the computational domain along the X, Y, and Z dimensions.
void Non_solid_update::compute_intervals(const Solid_field& is_solid, const Parallel_MPI& parallel_MPI) {
	if (parallel_MPI.is_master()) {
		return;
	}

	bool prev_solid = true;
	for (int X = parallel_MPI.start_XYZ2[0]; X <= parallel_MPI.end_XYZ2[0]; ++X)
		for (int Y = parallel_MPI.start_XYZ2[1]; Y <= parallel_MPI.end_XYZ2[1]; ++Y) {
			for (int Z = parallel_MPI.start_XYZ2[2]; Z <= parallel_MPI.end_XYZ2[2]; ++Z) {
				const Flat_index idx = is_solid.flat_index({X, Y, Z});
				const bool cur_solid = !(is_solid[idx] == FALSE);
				if (cur_solid != prev_solid) {
					non_solid_intervals.push_back(idx);
					prev_solid = cur_solid;
				}
			}
			// ghost nodes are always skipped
			if (prev_solid == false) {
				non_solid_intervals.push_back(is_solid.flat_index({X, Y, static_cast<Index>(parallel_MPI.end_XYZ2[2] + 1)}));
				prev_solid = true;
			}
		}
}

namespace interpolation {

Stencil::Stencil(const Index_vec3& base_idx, const Vec3& pos, const Solid_field& is_solid, double radius, bool is_2d, const std::unordered_set<Index_vec3>& exclude) {
	std::array<int, 3> start;
	std::array<int, 3> end;

	const int max_dim = is_2d ? 2 : 3;
	for (int d = 0; d < max_dim; ++d) {
		start[d] = std::ceil(pos[d] - radius);
		end[d] = std::floor(pos[d] + radius) + 1;
		ASSERT(start[d] >= 0 && "sufficient number of ghost nodes for interpolation");
		ASSERT(end[d] <= is_solid.sizes()[d] && "sufficient number of ghost nodes for interpolation");
	}
	if(is_2d){
		start[2] = base_idx[2];
		end[2] = base_idx[2] + 1;
		ASSERT(std::round(pos[2]) - base_idx[2] < 1e-7 && "position lies on the z plane");
		ASSERT(start[2] >= 0 && start[2] <= is_solid.sizes()[2] && "z-plane is valid");
	}

	for (int x = start[0]; x < end[0]; ++x) {
		for (int y = start[1]; y < end[1]; ++y) {
			for (int z = start[2]; z < end[2]; ++z) {
				const Index_vec3 idx{x, y, z};
				if (is_solid[idx] != FALSE || exclude.find(idx) != exclude.end()) {
					continue;
				}
				const Vec3 diff = pos - Vec3{static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)};
				const double dist = sqrt(dot(diff, diff));
				// special case: (almost) exact hit
				// The threshold eps is chosen very conservative. 
				// It should depend on the weight function w(r,x) below, where r - radius and x - dist.
				// The worst cast for ratio of discarded values is then w(r,eps) / w(r,1-eps)
				constexpr double eps = 1e-5;
				if (dist < eps){
					nodes.clear();
					nodes.push_back({is_solid.flat_index(idx), 1.0});
					return;
				}
				if (dist < radius) {
					nodes.push_back({is_solid.flat_index(idx), dist});
				}
			}
		}
	}

	if(nodes.empty()){
		return;
	}

//	ASSERT(!nodes.empty());
/*	auto max_it = std::max_element(nodes.begin(), nodes.end(), [](const Stencil::Node& lhs, const Stencil::Node& rhs){
		return lhs.weight < rhs.weight; 
	});
	const double max_dist = max_it->weight;*/
	const double max_dist = radius;
	constexpr double weight_exp = 2.0;
	// compute inverse distance weights
	double w_total = 0.0;
	for(Node& node : nodes) {
		node.weight = std::pow((max_dist - node.weight) / (max_dist * node.weight), weight_exp);
		w_total += node.weight;
	}

	// normalize to get sum 1
	for(Node& node: nodes) {
		node.weight /= w_total;
	}
}

double Stencil::interpolate(const Scalar_field& field) const {
	double v = 0.0;
	for(const Node& node : nodes) {
		v += node.weight * field[node.flat_idx];
	}
	return v;
}

double Stencil::interpolate(const Vector_field& field, int component) const{
	double v = 0.0;
	for(const Node& node : nodes) {
		v += node.weight * field(node.flat_idx, component);
	}
	return v;
}

}  // namespace interpolation