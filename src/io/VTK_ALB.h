#ifndef VTKALB_H
#define VTKALB_H
#include "IO_common.h"
#include "../Parallel.h"
#include "../Flow_solver.h"
#include "../ALBORZ_GlobalVariables.h"
#include "../ALBORZ_Macros.h"
#include "../utils/Assert.h"
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <time.h>
#include <functional>

class stl_import;

class VTK_ALB {
public:
	std::vector<unsigned int> begin_vtk;
	std::vector<unsigned int> end_vtk;
	std::vector<unsigned int> begin_vtk_gen_co;
	std::vector<unsigned int> end_vtk_gen_co;
	std::vector<unsigned int> length_vtk;

	std::string out_filename;

	VTK_ALB() = default;
	VTK_ALB(VTK_ALB&&) = default;
	VTK_ALB(const std::string& filename_out, const stl_import& Geo, const Parallel_MPI& MPI_parallel,
	        const double* physical_time);

	void Initialize(const std::string& filename_out, const stl_import* Geo, const Parallel_MPI* MPI_parallel);

	static void swap_endian(char* buffer, size_t size);
	static void swap_endian(const char* buffer, char* buffer2, size_t size);
	static void write_bigendian(std::ofstream& file, const char* buffer, size_t count, size_t size);
	template <typename T>
	static void write_bigendian(std::ofstream& file, const T* buffer, size_t count);

	// new interface for IO_backend --------------------------------------
	void open_file(const std::string& filename, OPEN_MODE mode);
	void close_file();

	// these functions should only be called while a file is opened
	template <typename Tensor_expr>
	void write(const Tensor_expr& tensor, const IO_dataset_info& info);
	// dummy function to work with IO_backend
	template <typename Tensor_expr>
	void read(Tensor_expr& tensor, const IO_dataset_info& info);

	template <typename T>
	void write_scalar(T scalar, const std::string& name);
	template <typename T>
	void read_scalar(T& scalar, const std::string& name);

private:
	// prepares file contents that depend only on the geometry
	void prepare_file(const Parallel_MPI& MPI_parallel, const stl_import& geo);
	static std::string make_filename(const std::string& name, int proc_id);
	void update_pvd(const std::string& file_to_add) const;

	std::ofstream m_file;
	std::size_t m_pointer_position;
	using data_write = void(std::ofstream& file);
	std::vector<std::function<data_write>> m_data_write_ops;
	// these parts only depend on the domain decomposition and do not change based on the fields
	std::string m_file_header;
	std::vector<std::string> m_piece_extends;
	int m_processor_id;
	std::string m_filename;
	const double* m_physical_time = nullptr;
};

// ******************************************************************* //
//  Implementation
// ******************************************************************* //
template <typename T>
void VTK_ALB::write_bigendian(std::ofstream& file, const T* buffer, size_t count) {
	char _buf[sizeof(T)];
	for (size_t i = 0; i < count; i++) {
		swap_endian(reinterpret_cast<const char*>(buffer + i), _buf, sizeof(T));
		file.write(_buf, sizeof(T));
	}
}

template <typename Tensor_expr>
void VTK_ALB::write(const Tensor_expr& tensor, const IO_dataset_info& info) {
	constexpr int Order = Tensor_expr::Order;
	using T = typename Tensor_expr::Value_type;
	ASSERT(Order > 2 && Order < 5);
	ASSERT(m_file.is_open());
	constexpr const char* datatype = "appended";

	if (m_processor_id == MASTER) {
		m_file << "<PDataArray type=\"Float64\" Name=\"" << info.name;
		if (Order > 3) {
			m_file << "\" NumberOfComponents=\"" << tensor.sizes()[3];
		}
		m_file << "\" format=\"" << datatype
			   << "\">\n"
			   << "</PDataArray>\n";
	} else {
		m_file << "<DataArray type=\"Float64\" Name=\"" << info.name;
		if (Order > 3) {
			m_file << "\" NumberOfComponents=\"" << tensor.sizes()[3];
		}
		m_file << "\" format=\"" << datatype
			   << "\" offset=\"" << m_pointer_position
			   << "\">\n"
			   << "</DataArray>\n";
		int64_t data_size = length_vtk[0] * length_vtk[1] * length_vtk[2] * sizeof(T);
		for (int i = 3; i < Order; ++i) {
			data_size *= tensor.sizes()[i];
		}
		m_pointer_position += data_size + sizeof(data_size);

		const auto scaling = info.scaling;
		using Index_vec = typename Tensor_expr::Index_vec;
		Index_vec local_size = tensor.sizes();
		local_size[0] = length_vtk[0];
		local_size[1] = length_vtk[1];
		local_size[2] = length_vtk[2];
		Index_vec offset{};
		offset[0] = begin_vtk[0];
		offset[1] = begin_vtk[1];
		offset[2] = begin_vtk[2];
		// xyz are flipped but the vector dimension remains the same
		Index_vec transpose_order;
		transpose_order[0] = 2;
		transpose_order[1] = 1;
		transpose_order[2] = 0;
		for (int i = 3; i < Order; ++i) {
			transpose_order[i] = i;
		}
		const auto view = transpose(slice(tensor, offset, local_size), transpose_order);
		// contents can not be written yet as they are all part of the appendix
		m_data_write_ops.emplace_back([view, data_size, scaling](std::ofstream& file) {
			write_bigendian(file, &data_size, 1);
			for (size_t i = 0; i < view.num_elem(); ++i) {
				const auto idx = view.index(i);
				T value = scaling * view[idx];
				write_bigendian(file, &value, 1);
			}
		});
	}
}

template <typename Tensor_expr>
void VTK_ALB::read(Tensor_expr& tensor, const IO_dataset_info& info) {
	std::cerr << "[Error] Reading from vtk files is not implemented.\n";
	MPI_Abort(MPI_COMM_WORLD, 1);
}

template <typename T>
void VTK_ALB::write_scalar(T scalar, const std::string& name) {
	std::cerr << "[Error] Writing scalars to vtk files is not (yet) implemented.\n";
	MPI_Abort(MPI_COMM_WORLD, 1);
}

template <typename T>
void VTK_ALB::read_scalar(T& scalar, const std::string& name) {
	std::cerr << "[Error] Reading from vtk files is not implemented.\n";
	MPI_Abort(MPI_COMM_WORLD, 1);
}

#endif