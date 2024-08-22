#ifndef IO_INTERFACE_H
#define IO_INTERFACE_H

#include "../Tensor.h"
#include "H5_ALB.h"
#include "VTK_ALB.h"
#include "IO_common.h"
#include <vector>
#include <string>

// Stores registered fields and implementation details for a specific format.
// For format agnostic IO see IO_interface below.
template <typename Impl>
class IO_backend {
public:
	IO_backend(IO_backend&& oth)
		: m_impl(std::move(oth.m_impl))
		, m_datasets(std::move(oth.m_datasets)) {}

	// This constructor just forwards everything to the actual implementation.
	template <typename... Args>
	explicit IO_backend(Args&&... args)
		: m_impl(std::forward<Args>(args)...) {
	}

	template <typename... Args>
	void write(const std::string& filename, Args&&... args);
	void read(const std::string& filename);

	template <typename Tensor_expr>
	void add_field(const Tensor_expr& tensor, const IO_dataset_info& info);

	template <typename T>
	void add_scalar(T& scalar, const std::string& name);

	Impl& get_impl() { return m_impl; }

private:
	Impl m_impl;
	using IO_op = void(Impl&, const IO_dataset_info&);
	struct IO_dataset {
		IO_dataset_info info;
		std::function<IO_op> read;
		std::function<IO_op> write;
	};
	std::vector<IO_dataset> m_datasets;
};

class IO_interface {
public:
	IO_interface(const stl_import& geo, const Parallel_MPI& parallel_MPI,
	             bool write_vtk, bool write_h5, const double* physical_time = nullptr,
	             const std::string& base_name = "",
	             const Index_vec3& offset = {}, const Index_vec3& sizes = {},
	             bool use_compression = false);

	// Register a field for read and write operations.
	// Only a reference is stored which needs to remain valid for the whole lifetime
	// of the IO_interface.
	// @param scaling Scalar factor applied to the whole field
	// @mask If provided, values where mask[{X,Y,Z}] != -1 will be set to 0 in the output.
	//    In DEBUG_MODE the mask is ignored.
	template <typename Tensor_expr>
	void add_field(const Tensor_expr& tensor, const std::string& name, double scaling = 1.0, const Solid_field* mask = nullptr);
	template <typename T>
	void add_field(T*** tensor, const std::string& name, double scaling = 1.0, const Solid_field* mask = nullptr);
	template <typename T>
	void add_field(T**** tensor, const std::string& name, double scaling = 1.0, const Solid_field* mask = nullptr);

	// Register a scalar value for read and write operations.
	// As with fields the address is expected to remain valid for the whole lifetime
	// of the IO_interface.
	template <typename T>
	void add_scalar(T& scalar, const std::string& name);

	// Constructs views assuming the standard grid size for local data.
	// If you just want to register a field as-is, use the specialized version of add_field instead.
	template <typename T>
	Ptr3_view<T> make_std_view(T*** tensor);
	template <typename T>
	Ptr4_view<T> make_std_view(T**** tensor);

	// Register a custom functor that is invoked on read or write operations.
	// @param  The base path to the target file that should be accessed by the custom op.
	// @param  The current time-step.
	using Custom_IO_fn = std::function<void(const std::string&, int)>;
	void add_custom_read(Custom_IO_fn read);
	void add_custom_write(Custom_IO_fn write);

	// Write all registered objects to a file.
	// @param tm The current time-step. A value <0 means that no time is appended to the name.
	void write(const std::string& filename, int tm = -1);
	// Attempt to read values for the registered objects from a file.
	// Reading is currently only supported for h5 and possibly custom operations.
	void read(const std::string& filename, int tm = -1);

private:
	template <typename Tensor_expr>
	void add_field_impl(const Tensor_expr& tensor, const std::string& name, double scaling);

	const Parallel_MPI& m_parallel_MPI;
	Index_vec3 m_std_size;
	Index_vec3 m_offset;
	Index_vec3 m_sizes;
	Index_vec3 m_domain_start;
	Index_vec3 m_domain_end;
	bool m_is_slice;
	bool m_write_vtk;
	bool m_write_h5;
	std::vector<Custom_IO_fn> m_custom_writes;
	std::vector<Custom_IO_fn> m_custom_reads;
	IO_backend<VTK_ALB> m_vtk_backend;
#ifdef WITH_HDF5
	IO_backend<H5_ALB> m_h5_backend;
#endif
	std::unordered_set<std::string> m_field_names;
};

// ******************************************************************* //
//  Implementation
// ******************************************************************* //
template <typename Impl>
template <typename... Args>
void IO_backend<Impl>::write(const std::string& filename, Args&&... args) {
	try {
		m_impl.open_file(filename, OPEN_MODE::WRITE, std::forward<Args>(args)...);

		for (const auto& dataset : m_datasets) {
			dataset.write(m_impl, dataset.info);
		}

		m_impl.close_file();
	} catch (const std::exception& e) {
		std::cerr << "[Error] Could not write \"" << filename << "\": " << e.what() << std::endl;
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
}

template <typename Impl>
void IO_backend<Impl>::read(const std::string& filename) {
	try {
		m_impl.open_file(filename, OPEN_MODE::READ);

		for (auto& dataset : m_datasets) {
			dataset.read(m_impl, dataset.info);
		}
	} catch (const std::exception& e) {
		std::cerr << "[Error] Could not read \"" << filename << "\": " << e.what() << std::endl;
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	m_impl.close_file();
}

template <typename Impl>
template <typename Tensor_expr>
void IO_backend<Impl>::add_field(const Tensor_expr& tensor, const IO_dataset_info& info) {
	m_datasets.push_back({info,
	                      [tensor](Impl& impl, const IO_dataset_info& info) {
							  impl.read(tensor, info);
						  },
	                      [tensor](Impl& impl, const IO_dataset_info& info) {
							  impl.write(tensor, info);
						  }});
}

template <typename Impl>
template <typename T>
void IO_backend<Impl>::add_scalar(T& scalar, const std::string& name) {
	m_datasets.push_back({IO_dataset_info{name, 1.0},
	                      [&scalar](Impl& impl, const IO_dataset_info& info) {
							  impl.read_scalar(scalar, info.name);
						  },
	                      [&scalar](Impl& impl, const IO_dataset_info& info) {
							  impl.write_scalar(scalar, info.name);
						  }});
}

// ******************************************************************* //
template <typename Tensor_expr>
void IO_interface::add_field(const Tensor_expr& tensor, const std::string& name, double scaling, const Solid_field* mask) {
	if (m_field_names.find(name) != m_field_names.end()) {
		std::cout << "[Warning] Skipping registration of the field \"" << name << "\" because a field of this name already exists in this task.\n";
		return;
	}
	m_field_names.insert(name);
	
#if !defined DEBUG_MODE
	using Value_type = typename Tensor_expr::Value_type;
	using Index_vec = typename Tensor_expr::Index_vec;
	if (mask) {
		auto apply_mask = [mask](Value_type v, const Index_vec& idx) {
			// if this value is changed to non-zero, than scaling also needs to be applied in the mask
			// instead of by the backend
			constexpr Value_type masked_value = static_cast<Value_type>(0);
			return (*mask)[{idx[0], idx[1], idx[2]}] == -1 ? v : masked_value;
		};
		add_field_impl(filter_view(tensor, apply_mask), name, scaling);
		return;
	}
#endif
	add_field_impl(tensor, name, scaling);
}

template <typename T>
void IO_interface::add_scalar(T& scalar, const std::string& name) {
#ifdef WITH_HDF5
	m_h5_backend.add_scalar(scalar, name);
#endif
	m_vtk_backend.add_scalar(scalar, name);
}

template <typename T>
void IO_interface::add_field(T*** tensor, const std::string& name, double scaling, const Solid_field* mask) {
	add_field(make_std_view(tensor), name, scaling, mask);
}

template <typename T>
void IO_interface::add_field(T**** tensor, const std::string& name, double scaling, const Solid_field* mask) {
	add_field(make_std_view(tensor), name, scaling, mask);
}

template <typename T>
Ptr3_view<T> IO_interface::make_std_view(T*** tensor) {
	return ptr_view(tensor, m_std_size);
}

template <typename T>
Ptr4_view<T> IO_interface::make_std_view(T**** tensor) {
	return ptr_view(tensor, {m_std_size[0], m_std_size[1], m_std_size[2], 3});
}

template <typename Tensor_expr>
void IO_interface::add_field_impl(const Tensor_expr& tensor, const std::string& name, double scaling) {
	details::Tensor_expr_t<Tensor_expr> tensor_ref(tensor);
	const IO_dataset_info info{name, scaling};

	if (m_write_vtk && m_is_slice && m_parallel_MPI.is_master()) {
		std::cerr << "[Warning] Attempting to add a slice to vtk outputs which is currently not supported. Consider creating a IO task with hdf5 only.\n";
	}

#ifdef WITH_HDF5
	using Index_vec = typename Tensor_expr::Index_vec;
	if (m_is_slice) {
		Index_vec offset = {};
		Index_vec sizes = tensor.sizes();
		for (int d = 0; d < 3; ++d) {
			offset[d] = m_offset[d];
			sizes[d] = m_sizes[d];
		}
		m_h5_backend.add_field(m_parallel_MPI.make_global_slice(tensor_ref, offset, sizes), info);
	} else {
		m_h5_backend.add_field(tensor_ref, info);
	}
#endif
	m_vtk_backend.add_field(tensor_ref, info);
}

#endif