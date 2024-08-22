#ifndef H5ALB_H
#define H5ALB_H

#ifdef WITH_HDF5
#include "../Parallel.h"
#include "../Flow_solver.h"
#include "../Phase_Field.h"
#include "../ALBORZ_GlobalVariables.h"
#include "../Tensor.h"
#include "IO_common.h"
#include <mpi.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <time.h>

class stl_import;

namespace details {

// Maps a C++ type to the respective hdf5 type id.
template <typename T>
struct to_hdf_type;
// value has to be a function here because h5 initializes the type ids at runtime.
template <>
struct to_hdf_type<float> {
	static const hid_t value() { return H5T_NATIVE_FLOAT; }
};
template <>
struct to_hdf_type<double> {
	static const hid_t value() { return H5T_NATIVE_DOUBLE; }
};
template <>
struct to_hdf_type<char> {
	static hid_t value() { return H5T_NATIVE_CHAR; }
};
template <>
struct to_hdf_type<signed char> {
	static hid_t value() { return H5T_NATIVE_SCHAR; }
};
template <>
struct to_hdf_type<int16_t> {
	static hid_t value() { return H5T_NATIVE_INT16; }
};
template <>
struct to_hdf_type<int32_t> {
	static hid_t value() { return H5T_NATIVE_INT32; }
};
template <>
struct to_hdf_type<int64_t> {
	static hid_t value() { return H5T_NATIVE_INT64; }
};
template <>
struct to_hdf_type<uint16_t> {
	static hid_t value() { return H5T_NATIVE_UINT16; }
};
template <>
struct to_hdf_type<uint32_t> {
	static hid_t value() { return H5T_NATIVE_UINT32; }
};
template <>
struct to_hdf_type<uint64_t> {
	static hid_t value() { return H5T_NATIVE_UINT64; }
};

template <typename Function_type, typename... Args>
auto _H5_call(const char* _functionName, Function_type _function, Args&&... _args)
	-> decltype(_function(std::forward<Args>(_args)...)) {
	auto ret = _function(std::forward<Args>(_args)...);
	if (ret < 0) {
		// throw an exception so that the program can be stopped
		// in a context that is aware of MPI
		throw IO_exception(std::string("Call to \"") + _functionName + "\" returned an error.");
	}
	return ret;
}
}  // namespace details

#if defined(__GNUC__) || defined(__MINGW32__)
#define H5_call(_function, ...) details::_H5_call(#_function, _function, ##__VA_ARGS__)
#else
#define H5_call(_function, ...) details::_H5_call(#_function, _function, __VA_ARGS__)
#endif

class H5_ALB {
public:
	hsize_t dimsfile[3];
	hsize_t dimsmem[3];
	hsize_t offset[3];
	hsize_t stride[3];
	hsize_t block[3];
	hsize_t count[3];

	double origin[3];

	int ax0_st, ax0_en;
	int ax1_st, ax1_en;
	int ax2_st, ax2_en;
	bool is_master;  //< stored because the master process needs special handling

	H5_ALB() = default;
	// @param vtk_compatible Add metadata such that the files are valid vtkhdf
	H5_ALB(const stl_import& geo, const Parallel_MPI& MPI_parallel, bool vtk_compatible = true, bool with_compression = false);

	void initialize(const stl_import* Geo, const Parallel_MPI* MPI_parallel);

	// new interface
	void open_file(const std::string& filename, OPEN_MODE mode, const Index_vec3& domain_start = {}, const Index_vec3& domain_end = {});
	void close_file();

	// these functions should only be called while a file is opened
	template <typename Tensor_expr>
	void write(const Tensor_expr& tensor, const IO_dataset_info& info);
	template <typename Tensor_expr>
	void read(Tensor_expr& tensor, const IO_dataset_info& info);

	template <typename T>
	void write_scalar(T scalar, const std::string& name);
	template <typename T>
	void read_scalar(T& scalar, const std::string& name);

private:
	static bool is_filter_available(int h5_filter_type);
	static bool is_compression_available();

	// write necessary infos for the vtkhdf-reader
	void write_vtk_info(const Index_vec3& domain_start, const Index_vec3& domain_end);

	// alternative for H5LTset_attribute_string that does not store the terminating null
	// which confuses vtkhdf
	static void set_attribute_string(hid_t loc_id, const char* obj_name, const char* attr_name, const char* attr_data);

	// determines the maximum possible chunk size across all processes
	std::array<hsize_t, 3> get_max_chunk_size(std::array<hsize_t, 3> block_size) const;

	// reduces the chunk size if it is too large
	template <typename T, size_t N>
	static std::array<hsize_t, N> shrink_chunk_size(std::array<hsize_t, N> sizes);

	template <int N>
	struct H5_space_size {
		hsize_t dimsmem[N];
		hsize_t dimsfile[N];
		hsize_t block[N];
		hsize_t offset[N];
		hsize_t count[N];
	};

	template <typename Tensor_expr>
	struct Size_descriptor {
		static constexpr int N = Tensor_expr::Order;
		using T = typename Tensor_expr::Value_type;

		static Slice_view<Transpose_view<Tensor_expr>> make_hyperslap_view(const Tensor_expr& view, const H5_ALB& h5_alb) {
			using Index_vec = Base_index_vec<N>;

			const Index_vec transpose_order = get_transpose_order<N>();

			// dimsmem is already transposed
			Index_vec local_size;
			local_size[0] = h5_alb.dimsmem[0];
			local_size[1] = h5_alb.dimsmem[1];
			local_size[2] = h5_alb.dimsmem[2];

			for (int i = 3; i < N; ++i) {
				local_size[i] = view.sizes()[transpose_order[i]];
			}

			Index_vec local_offset{};
			local_offset[0] = h5_alb.ax0_st;
			local_offset[1] = h5_alb.ax1_st;
			local_offset[2] = h5_alb.ax2_st;

			return slice(transpose(view, transpose_order),
			             local_offset, local_size);
		}

		static H5_space_size<N> compute_h5_sizes(const Tensor_expr& view, const H5_ALB& h5_alb) {
			H5_ALB::H5_space_size<N> result;
			constexpr int domain_end = 3;

			// non spatial dimensions are not transposed
			for (int i = domain_end; i < N; ++i) {
				result.dimsmem[i] = view.sizes()[i];
				result.dimsfile[i] = view.sizes()[i];
				result.block[i] = view.sizes()[i];
				result.offset[i] = 0;
				result.count[i] = 1;
			}

			// spatial dimensions are already transposed
			for (int i = 0; i < domain_end; ++i) {
				result.dimsmem[i] = h5_alb.dimsmem[i];
				result.dimsfile[i] = h5_alb.dimsfile[i];
				result.block[i] = h5_alb.block[i];
				result.offset[i] = h5_alb.offset[i];
				result.count[i] = 1;
			}

			return result;
		}

		static std::array<hsize_t, N> compute_chunk_size(const Tensor_expr& view, const H5_ALB& h5_alb){
			std::array<hsize_t, N> chunk_size;
			// use precomputed minimum across all processes
			for(int i = 0; i < 3; ++i){
				chunk_size[i] = h5_alb.chunk_size[i];
			}
			for(int i = 3; i < N; ++i){
				chunk_size[i] = view.sizes()[i];
			}
			return shrink_chunk_size<T>(chunk_size);
		}
	};

	template <typename Tensor_expr>
	struct Size_descriptor<Global_slice<Tensor_expr>> {
		static constexpr int N = Tensor_expr::Order;
		using T = typename Tensor_expr::Value_type;

		static Transpose_view<Global_slice<Tensor_expr>> make_hyperslap_view(const Global_slice<Tensor_expr>& view, const H5_ALB& h5_alb) {
			using Index_vec = Base_index_vec<N>;

			const Index_vec transpose_order = get_transpose_order<N>();
			return transpose(view, transpose_order);
		}

		static H5_space_size<N> compute_h5_sizes(const Global_slice<Tensor_expr>& view, const H5_ALB& h5_alb) {
			H5_ALB::H5_space_size<N> result;
			constexpr int domain_end = 3;

			// non spatial dimensions are not transposed
			for (int i = domain_end; i < N; ++i) {
				result.dimsmem[i] = view.sizes()[i];
				result.dimsfile[i] = view.global_sizes()[i];
				result.block[i] = view.sizes()[i];
				result.offset[i] = view.global_offset()[i];
				result.count[i] = 1;
			}

			// spatial dimensions need to be transposed
			for (int i = 0; i < domain_end; ++i) {
				const int j = domain_end - 1 - i;
				result.dimsmem[i] = view.sizes()[j];
				result.dimsfile[i] = view.global_sizes()[j];
				result.block[i] = view.sizes()[j];
				result.offset[i] = view.global_offset()[j];
				result.count[i] = 1;
			}

			return result;
		}

		static std::array<hsize_t, N> compute_chunk_size(const Global_slice<Tensor_expr>& view, const H5_ALB& h5_alb) {
			// since we work with the base view, the size still needs to be transposed
			std::array<hsize_t, 3> block_size{view.sizes()[2], view.sizes()[1], view.sizes()[0]};
			block_size = h5_alb.get_max_chunk_size(block_size);

			std::array<hsize_t, N> chunk_size;
			for(int i = 0; i < 3; ++i){
				chunk_size[i] = block_size[i];
			}
			for(int i = 3; i < N; ++i){
				chunk_size[i] = view.sizes()[i];
			}

			return shrink_chunk_size<T>(chunk_size);
		}
	};

	//	template <typename Tensor_expr>
	//	friend class Size_descriptor;

	template <size_t N>
	static std::array<hsize_t, N> to_hsize(const Base_index_vec<N>& sizes);

	template <size_t N>
	static Base_index_vec<N> get_transpose_order();

	bool m_vtk_compatible = false;
	hid_t m_file_id = -1;
	hid_t m_root_group_id = -1;

	bool use_compression = false;
	std::array<hsize_t, 3> chunk_size;
	size_t num_elem_chunk;
};

// ******************************************************************* //
//  Implementation
// ******************************************************************* //
template <typename T, size_t N>
std::array<hsize_t, N> H5_ALB::shrink_chunk_size(std::array<hsize_t, N> sizes) {
	size_t byte_size = sizeof(T);
	for (int i = 0; i < N; ++i) {
		byte_size *= sizes[i];
	}

	// shrink if larger than 1gb (maximum allowed size is 4gb)
	// no synchronization with other processes is necessary since chunk_size is the same
	constexpr size_t max_bytes = 2ull << 30ull;
	while (byte_size > max_bytes) {
		if (sizes[0] >= sizes[1] && sizes[0] >= sizes[2]) {
			sizes[0] /= 2;
		} else if (sizes[1] >= sizes[0] && sizes[1] >= sizes[2]) {
			sizes[1] /= 2;
		} else {
			sizes[2] /= 2;
		}
		byte_size /= 2;
	}
	return sizes;
}

template <typename Tensor_expr>
void H5_ALB::write(const Tensor_expr& tensor, const IO_dataset_info& info) {
	using T = typename Tensor_expr::Value_type;
	constexpr int N = Tensor_expr::Order;
	using Descriptor = Size_descriptor<Tensor_expr>;

	static_assert(N > 2, "Tensors need to be atleast of order 3 due to the domain decomposition.");
	ASSERT(m_file_id != H5I_INVALID_HID);

	Tensor<T, N> data;
	// currently this special case is needed for 2 reasons:
	// 1. dimsmem is larger than 0 leading to a non-zero view on a nullptr
	// 2. zero-sized tensors are not well tested (but should work in principle)
	if (!is_master && tensor.num_elem()) {
		data = Descriptor::make_hyperslap_view(tensor, *this);
		if (info.scaling != 1.0) {
			data *= info.scaling;
		}
	}

	const auto h5_sizes = Descriptor::compute_h5_sizes(tensor, *this);
	const char* dset_name = info.name.c_str();
	const hid_t file_space = H5_call(H5Screate_simple, N, h5_sizes.dimsfile, nullptr);
	const hid_t type_id = details::to_hdf_type<T>::value();

	// compression
	hid_t dcpl_id = H5P_DEFAULT;
	if (use_compression) {
		dcpl_id = H5_call(H5Pcreate, H5P_DATASET_CREATE);
		H5_call(H5Pset_shuffle, dcpl_id);
		H5_call(H5Pset_deflate, dcpl_id, 6);
		std::array<hsize_t, N> local_chunk_size = Descriptor::compute_chunk_size(tensor, *this);
		H5_call(H5Pset_chunk, dcpl_id, N, local_chunk_size.data());
	}

	const hid_t dset_id = H5_call(H5Dcreate2, m_root_group_id, dset_name, type_id, file_space, H5P_DEFAULT, dcpl_id, H5P_DEFAULT);
	H5Sclose(file_space);

	hid_t mem_space;
	const hid_t file_space_slab = H5_call(H5Dget_space, dset_id);
	if (is_master || !tensor.num_elem()) {
		static constexpr hsize_t zero_shape[N] = {};
		mem_space = H5_call(H5Screate_simple, N, zero_shape, nullptr);
		H5_call(H5Sselect_none, file_space_slab);
	} else {
		mem_space = H5_call(H5Screate_simple, N, h5_sizes.dimsmem, nullptr);
		H5_call(H5Sselect_hyperslab, file_space_slab, H5S_SELECT_SET, h5_sizes.offset, nullptr, h5_sizes.count, h5_sizes.block);
	}

	const hid_t plist_data_id = H5_call(H5Pcreate, H5P_DATASET_XFER);
	H5_call(H5Pset_dxpl_mpio, plist_data_id, H5FD_MPIO_COLLECTIVE);
	H5_call(H5Dwrite, dset_id, type_id, mem_space, file_space_slab, plist_data_id, data.data());

	if (dcpl_id) {
		H5_call(H5Pclose, dcpl_id);
	}
	H5_call(H5Pclose, plist_data_id);
	H5_call(H5Sclose, file_space_slab);
	H5_call(H5Sclose, mem_space);
	H5_call(H5Dclose, dset_id);
}

template <typename Tensor_expr>
void H5_ALB::read(Tensor_expr& tensor, const IO_dataset_info& info) {
	using T = typename Tensor_expr::Value_type;
	constexpr int N = Tensor_expr::Order;
	using Descriptor = Size_descriptor<Tensor_expr>;

	ASSERT(m_file_id != H5I_INVALID_HID);

	// open dataset
	const hid_t dataset_id = H5_call(H5Dopen, m_root_group_id, info.name.c_str(), H5P_DEFAULT);
	if (dataset_id < 0) {
		std::cout << "[Warning] Reading dataset " << info.name << " failed.\n";
		return;
	}
	//	const hid_t dcpl_id = H5Dget_create_plist (dset);
	//	const int num_filters = H5Pget_nfilters(dcpl);

	const hid_t file_space = H5_call(H5Dget_space, dataset_id);
	const hid_t type_id = details::to_hdf_type<T>::value();

	// retrieve size
	/*	const int ndims = H5Sget_simple_extent_ndims(dspace);
	    std::vector<hsize_t> dims(ndims, 0);
	    H5Sget_simple_extent_dims(dspace, dims.data(), nullptr);
	    // check that the target tensor is of the correct size
	    if (ndims != N){
	        std::cerr << "[Error] Number of dimensions does not match for tensor " << field.name
	            << ". Found " << ndims << " but expected " << N << "\n";
	    }
	    for (int i = 0; i < N; ++i){
	        if (dims[i] != field.data->sizes()[i]){
	            std::cerr << "[Error] Dimension size does not match for tensor " << field.name
	                << ". Found " << dims[i] << " but expected " << field.data->sizes()[i] << "\n";
	        }
	    }*/

	if (!is_master) {
		const auto h5_sizes = Descriptor::compute_h5_sizes(tensor, *this);
		auto chunks = decompose_periodic(tensor,
		                                 {static_cast<Index>(offset[2]) - ax2_st,
		                                  static_cast<Index>(offset[1]) - ax1_st,
		                                  static_cast<Index>(offset[0]) - ax0_st},
		                                 {static_cast<Index>(dimsfile[2]),
		                                  static_cast<Index>(dimsfile[1]),
		                                  static_cast<Index>(dimsfile[0])});
		for (auto& chunk : chunks) {
			const auto transpose_order = get_transpose_order<N>();
			auto dst_tensor = transpose(chunk.dest, transpose_order);
			hsize_t src_offset[N];
			for (int i = 0; i < N; ++i) {
				src_offset[i] = chunk.src_offset[transpose_order[i]];
			}
			const auto sizes = to_hsize(dst_tensor.sizes());

			H5_call(H5Sselect_hyperslab, file_space, H5S_SELECT_SET, src_offset, nullptr, h5_sizes.count, sizes.data());
			Tensor<T, N> data(dst_tensor.sizes());

			const hid_t mem_space = H5_call(H5Screate_simple, N, sizes.data(), nullptr);
			const hid_t plist_data_id = H5_call(H5Pcreate, H5P_DATASET_XFER);
			H5_call(H5Dread, dataset_id, type_id, mem_space, file_space, plist_data_id, data.data());
			if (info.scaling != 1.0) {
				data *= 1.0 / info.scaling;
			}

			assign_view(dst_tensor, data);

			H5_call(H5Pclose, plist_data_id);
			H5_call(H5Sclose, mem_space);
		}
	}

	H5_call(H5Sclose, file_space);
	H5_call(H5Dclose, dataset_id);
}

template <typename T>
void H5_ALB::write_scalar(T scalar, const std::string& name) {
	const hid_t type_id = details::to_hdf_type<T>::value();
	const hid_t dataspace_id = H5_call(H5Screate, H5S_SCALAR);

	const hid_t attr_id = H5_call(H5Acreate, m_file_id, name.c_str(), type_id, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
	H5_call(H5Awrite, attr_id, type_id, &scalar);

	H5_call(H5Aclose, attr_id);
	H5_call(H5Sclose, dataspace_id);
}

template <typename T>
void H5_ALB::read_scalar(T& scalar, const std::string& name) {
	const hid_t type_id = details::to_hdf_type<T>::value();
	const hid_t attr_id = H5_call(H5Aopen_name, m_file_id, name.c_str());
	H5_call(H5Aread, attr_id, type_id, &scalar);
	/*	const hid_t stored_type_id = H5Aget_type(attr_id);
	    if (stored_type_id == type_id) {
	        H5Aread(attr_id, type_id, &scalar);
	    } else {
	        std::cout << "[Warning] Could not read the attribute \"" << name << "\""
	                  << " because of a type mismatch. Expected type id is " << type_id
	                  << " but found " << stored_type_id << " in the file.\n";
	    }*/
	H5_call(H5Aclose, attr_id);
}

template <size_t N>
std::array<hsize_t, N> H5_ALB::to_hsize(const Base_index_vec<N>& sizes) {
	std::array<hsize_t, N> result;
	for (int i = 0; i < N; ++i) {
		ASSERT(sizes[i] >= 0);
		result[i] = sizes[i];
	}
	return result;
}

template <size_t N>
std::array<Index, N> H5_ALB::get_transpose_order() {
	std::array<Index, N> ord;
	ord[0] = 2;
	ord[1] = 1;
	ord[2] = 0;

	for (Index i = 3; i < static_cast<Index>(N); ++i) {
		ord[i] = i;
	}

	return ord;
}

#endif  // WITH_HDF5
#endif  // H5ALB_H
