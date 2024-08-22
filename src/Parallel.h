#ifndef PARALLEL_H
#define PARALLEL_H

#include "Tensor.h"
#include "ALBORZ_GlobalVariables.h"
#include "ALBORZ_Macros.h"
#include "mpi.h"
#include "Vec.h"
#include <vector>
#include <cmath>
#include <iostream>  // for the use of 'cout'
#include <iomanip>   // std::setw
#include <time.h>    /* clock_t, clock, CLOCKS_PER_SEC */

class Parallel_MPI;

class Data_exchange_group {
public:
	Data_exchange_group() = default;
	Data_exchange_group(const Parallel_MPI& parallel_MPI);
	~Data_exchange_group();

	// Add a field with the global settings for the ghost nodes (Parallel_MPI::buffer_size).
	template <typename T, int N>
	void add_field(Tensor<T, N>& tensor);

	// More fine grained control over the slice being exchanged.
	// @param size The number of layers which are exchanged.
	// @param offset The start of the slice is shifted towards the ghost nodes.
	template <typename T, int N>
	void add_field(Tensor<T, N>& tensor, int size, int offset = 0);

	// Add a population tensor for which only the needed directions are exchanged.
	void add_population(Population_field& tensor, const std::vector<std::vector<int>>& c_alpha);

	// Exchange all tensors registered with this group.
	void exchange_data();

private:
	struct Exchange_info {
		std::array<MPI_Datatype, 6> send_datatypes;
		std::array<MPI_Datatype, 6> receive_datatypes;

		void* data() const { return data_ptr_fn(tensor_ptr); }
		void* tensor_ptr;
		using Get_data_ptr = void* (*)(void*);
		Get_data_ptr data_ptr_fn;
	};

	struct Exchange_pop_info {
		Population_field* tensor_ptr;
		std::array<std::vector<int>, 6> directions;
		std::array<Flat_index, 6> msg_sizes;
	};

	const Parallel_MPI* parallel_MPI = nullptr;
	std::vector<Exchange_info> exchange_infos;
	std::vector<Exchange_pop_info> exchange_pop_infos;
	std::vector<double> send_buffer;
	std::vector<double> receive_buffer;

	void exchange_pop_data();
	template <int D1, int D2, int D3, int MSG_IDX, int SEND_IDX, int RECEIVE_IDX>
	void exchange_pop_data_impl(const Exchange_pop_info& info, Index src_offset, Index dest_offset);

	template <typename T, int N, Storage_order S>
	MPI_Datatype make_MPI_datatype(const Slice_view<Tensor<T, N, S>>& view, MPI_Datatype datatype) const;
	template <typename T, int N>
	Exchange_info make_exchange_info(Tensor<T, N>& tensor, int size, int offset, const std::array<MPI_Datatype, 6>& datatypes) const;

	template <typename T, int N>
	static void* get_data_ptr(void* tensor) {
		auto tensor_ptr = reinterpret_cast<Tensor<T, N>*>(tensor);
		return tensor_ptr->data();
	}
};

template <typename Tensor_expr>
class Global_slice : public Slice_view<Tensor_expr> {
public:
	static constexpr int Order = Tensor_expr::Order;
	using Value_type = typename Tensor_expr::Value_type;
	using Index_vec = typename Tensor_expr::Index_vec;

	const Index_vec& global_offset() const { return m_global_offset; }
	const Index_vec& global_sizes() const { return m_global_size; }

private:
	friend class Parallel_MPI;

	Global_slice(const Tensor_expr& expr, const Index_vec& offset, const Index_vec& sizes,
	             const Index_vec& global_offset, const Index_vec& global_size)
		: Slice_view<Tensor_expr>(expr, offset, sizes)
		, m_global_offset(global_offset)
		, m_global_size(global_size) {
	}

	Index_vec m_global_offset;
	Index_vec m_global_size;
};

class Parallel_MPI {
public:
	unsigned int Np_X, Np_Y, Np_Z;
	unsigned int buffer_size;  // number of ghost node layers which are synchronized
	int processor_id, num_processors;

	std::vector<std::vector<std::vector<unsigned int>>> proc_arrangement;
	std::vector<unsigned> proc_neighbours;
	unsigned int proc_position[3];

	unsigned int dev_start[3], dev_end[3];		
	unsigned int avg_nod_per_process[3];
	std::vector<unsigned int> start_XYZ;       /* First node absolute position */
	std::vector<unsigned int> end_XYZ;         /* Last node absolute position */
	std::vector<unsigned int> actual_rows_XYZ; /* Number of non-ghost nodes */
	std::vector<unsigned int> start_XYZ2;      /* First node relative position */
	std::vector<unsigned int> end_XYZ2;        /* Last node relative position */
	Vec3 center; // set by stl_import::Initialize_geometry

	clock_t t0;
	double t_current = -1;
	double t_old = -1;
	double delta_t = -1;
	unsigned int step_current, step_old;

	Parallel_MPI();
	Parallel_MPI(int argc, char** argv);
	virtual ~Parallel_MPI();
	void Initialize_MPI(int argc, char** argv);
	void Domain_Decomp(unsigned int Dimension, const std::string& filename);
	void Sync_Master();
	void Time_monitor(unsigned int, unsigned int, unsigned int);
	void Onscreen_Report(int, int);
	// Computes coordinates of a node based on the center and node spacing
	void get_coordinates(double, double, double, double, double, double, double&, double&, double&) const;
	Vec3 get_coordinates(const Index_vec3& idx, const Vec3& center) const;
	Vec3 get_coordinates(const Vec3& pos, const Vec3& center) const;
	// uses Parallel_MPI::center
	Vec3 get_coordinates(const Index_vec3& idx) const;
	Vec3 get_coordinates(const Vec3& pos) const;
	// determine if the node is inside the compute domain of the current processor
	bool is_inside_compute_domain(const Index_vec3& idx) const;

	bool is_master() const;

	template <typename Tensor_expr>
	Global_slice<Tensor_expr> make_global_slice(const Tensor_expr& expr, const typename Tensor_expr::Index_vec& offset, const typename Tensor_expr::Index_vec& size) const;

protected:
private:
};

#define ERROR_ABORT(message)                             \
	do {                                                 \
		std::cerr << "[Error] " << message << std::endl; \
		MPI_Abort(MPI_COMM_WORLD, 1);                    \
	} while (false)
// ******************************************************************* //
//  Implementation
// ******************************************************************* //
template <typename T>
struct to_MPI_type;

template <>
struct to_MPI_type<int> {
	static MPI_Datatype value() { return MPI_INT; }
};

template <>
struct to_MPI_type<float> {
	static MPI_Datatype value() { return MPI_FLOAT; }
};

template <>
struct to_MPI_type<double> {
	static MPI_Datatype value() { return MPI_DOUBLE; }
};

template <>
struct to_MPI_type<uint64_t> {
	static MPI_Datatype value() { return MPI_UINT64_T; }
};

template <typename T, int N>
void Data_exchange_group::add_field(Tensor<T, N>& tensor) {
	add_field(tensor, parallel_MPI->buffer_size, 0);
}

template <typename T, int N>
void Data_exchange_group::add_field(Tensor<T, N>& tensor, int size, int offset) {
	ASSERT(parallel_MPI);
	std::array<MPI_Datatype, 6> datatypes;
	datatypes.fill(to_MPI_type<T>::value());
	exchange_infos.push_back(make_exchange_info(tensor, size, offset, datatypes));
}

template <typename T, int N>
Data_exchange_group::Exchange_info Data_exchange_group::make_exchange_info(Tensor<T, N>& tensor,
                                                                           int size, int offset, const std::array<MPI_Datatype, 6>& datatypes) const {
	Exchange_info exchange_info;
	exchange_info.tensor_ptr = &tensor;
	exchange_info.data_ptr_fn = &get_data_ptr<T, N>;

	// synchronisation always happens in 3D
	for (int dim = 0; dim < 3; ++dim) {
		const int dest = 2 * dim;
		Base_index_vec<N> sizes = tensor.sizes();
		sizes[dim] = size;
		Base_index_vec<N> offsets{};
		//+1 because end_XYZ2 is a closed interval
		offsets[dim] = parallel_MPI->end_XYZ2[dim] - size + 1 + offset;
		exchange_info.send_datatypes[dest] = make_MPI_datatype(slice(tensor, offsets, sizes), datatypes[dest]);
		offsets[dim] = parallel_MPI->start_XYZ2[dim] - size + offset;
		exchange_info.receive_datatypes[dest] = make_MPI_datatype(slice(tensor, offsets, sizes), datatypes[dest]);

		offsets[dim] = parallel_MPI->start_XYZ2[dim] - offset;
		exchange_info.send_datatypes[dest + 1] = make_MPI_datatype(slice(tensor, offsets, sizes), datatypes[dest + 1]);
		offsets[dim] = parallel_MPI->end_XYZ2[dim] + 1 - offset;
		exchange_info.receive_datatypes[dest + 1] = make_MPI_datatype(slice(tensor, offsets, sizes), datatypes[dest + 1]);
	}

	for (int i = 0; i < 6; ++i) {
		MPI_Type_commit(&exchange_info.send_datatypes[i]);
		MPI_Type_commit(&exchange_info.receive_datatypes[i]);
	}

	return exchange_info;
}

template <typename T, int N, Storage_order S>
MPI_Datatype Data_exchange_group::make_MPI_datatype(const Slice_view<Tensor<T, N, S>>& view, MPI_Datatype datatype) const {
	static_assert(std::is_same<Index, int>::value, "expects int as tensor index type");
	static_assert(S == Storage_order::ROW_MAJOR, "only row major order is supported for data exchange");

	MPI_Datatype mpi_array_t;
	// if a custom type is given we assume that it replaces the last dimension
	MPI_Type_create_subarray(datatype == to_MPI_type<T>::value() ? N : N - 1,
	                         view.base_expr().sizes().data(),
	                         view.sizes().data(),
	                         view.offset().data(),
	                         MPI_ORDER_C,
	                         datatype,
	                         &mpi_array_t);
	MPI_Type_commit(&mpi_array_t);

	return mpi_array_t;
}

// ******************************************************************* //
template <typename Tensor_expr>
Global_slice<Tensor_expr> Parallel_MPI::make_global_slice(const Tensor_expr& expr, const typename Tensor_expr::Index_vec& offset, const typename Tensor_expr::Index_vec& sizes) const {
	constexpr int N = Tensor_expr::Order;
	using Index_vec = Base_index_vec<N>;

	Index_vec local_size;
	Index_vec local_offset;
	Index_vec global_offset;

	if (is_master()) {
		for (int d = 0; d < 3; ++d) {
			local_size[d] = 0;
			local_offset[d] = 0;
		}
	} else {
		for (int d = 0; d < 3; ++d) {
			const Index start = offset[d] - static_cast<Index>(start_XYZ[d]);
			const Index start_clamp = std::max(start, 0);
			local_offset[d] = start_clamp + static_cast<Index>(start_XYZ2[d]);
			const Index end = (offset[d] + sizes[d]) - static_cast<Index>(start_XYZ[d]);
			const Index end_clamp = std::min(end, static_cast<Index>(actual_rows_XYZ[d]));
			local_size[d] = std::max(0, end_clamp - start_clamp);
			global_offset[d] = std::max(0, -start);
		}
	}
	for (int d = 3; d < N; ++d) {
		local_offset[d] = offset[d];
		local_size[d] = sizes[d];
		global_offset[d] = offset[d];
	}

	return Global_slice<Tensor_expr>(expr, local_offset, local_size, global_offset, sizes);
}

#endif  // PARALLEL_H
