#include "IO_interface.h"

IO_interface::IO_interface(const stl_import& geo, const Parallel_MPI& parallel_MPI,
                           bool write_vtk, bool write_h5, const double* physical_time,
                           const std::string& base_name,
                           const Index_vec3& offset, const Index_vec3& sizes,
                           bool use_compression)
	: m_parallel_MPI(parallel_MPI)
	, m_std_size{static_cast<int>(parallel_MPI.dev_end[0]),
                 static_cast<int>(parallel_MPI.dev_end[1]),
                 static_cast<int>(parallel_MPI.dev_end[2])}
	, m_offset{offset}
	, m_sizes{sizes[0] ? sizes[0] : global_parameters.Nx,
              sizes[1] ? sizes[1] : global_parameters.Ny,
              sizes[2] ? sizes[2] : global_parameters.Nz}
	, m_domain_start{m_offset}
	, m_domain_end{m_offset[0] + m_sizes[0] - 1,
                   m_offset[1] + m_sizes[1] - 1,
                   m_offset[2] + m_sizes[2] - 1}
	, m_is_slice(sizes != Index_vec3{})
	, m_write_vtk(write_vtk)
	, m_write_h5(write_h5)
	, m_vtk_backend(base_name, geo, parallel_MPI, physical_time)
#ifdef WITH_HDF5
	, m_h5_backend(geo, parallel_MPI, true, use_compression)
#endif
{
	if (parallel_MPI.is_master()) {
		// master thread does not hold any data but dev_end stores the full domain size
		m_std_size = {0, 0, 0};

		if (m_sizes[0] == 1 || m_sizes[1] == 1 || m_sizes[2] == 1) {
			std::cout << "[Warning] Output slice of size " << m_sizes
					  << " has at least one dimension of size 1. Paraview might not be able to open such outputs. Consider setting the size to 2 instead."
					  << std::endl;
		}
	}

	// check whether the selected region is valid
	if (m_domain_start[0] < 0 || m_domain_start[0] > m_domain_end[0] || m_domain_end[0] >= global_parameters.Nx
	    || m_domain_start[1] < 0 || m_domain_start[1] > m_domain_end[1] || m_domain_end[1] >= global_parameters.Ny
	    || m_domain_start[2] < 0 || m_domain_start[2] > m_domain_end[2] || m_domain_end[2] >= global_parameters.Nz) {
		if (parallel_MPI.is_master()) {
			std::cerr << "[Error] Output slice is invalid with nodes from [" << m_domain_start
					  << "] to [" << m_domain_end << "] (inclusive)." << std::endl;
		}
		std::abort();
	}
}

void IO_interface::add_custom_write(Custom_IO_fn _write) {
	m_custom_writes.emplace_back(std::move(_write));
}

void IO_interface::add_custom_read(Custom_IO_fn _read) {
	m_custom_reads.emplace_back(std::move(_read));
}

void IO_interface::write(const std::string& filename, int tm) {
	for (auto& custom_write : m_custom_writes) {
		custom_write(filename, tm);
	}
	const std::string filename_tm = tm >= 0 ? filename + "_t" + std::to_string(tm) : filename;
	if (m_write_vtk) {
		m_vtk_backend.write(filename_tm);
	}
#ifdef WITH_HDF5
	if (m_write_h5) {
		m_h5_backend.write(filename_tm, m_domain_start, m_domain_end);
	}
#endif
}

void IO_interface::read(const std::string& filename, int tm) {
#ifdef WITH_HDF5
	const std::string filename_tm = tm >= 0 ? filename + "_t" + std::to_string(tm) : filename;
	if (m_write_h5) {
		m_h5_backend.read(filename_tm);
	}
#endif
	for (auto& custom_read : m_custom_reads) {
		custom_read(filename, tm);
	}
}