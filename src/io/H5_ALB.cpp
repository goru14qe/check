#include "H5_ALB.h"
#include "../Geometry.h"
#include "../ALBORZ_Macros.h"
#include "../ALBORZ_GlobalVariables.h"
#include <fstream>  // file stream
#include <iostream>
#include <sstream>
#include <memory>
#include <cstring>

static constexpr const char* vtk_group_name = "VTKHDF";
static constexpr const char* point_data_group = "PointData";

H5_ALB::H5_ALB(const stl_import& geo, const Parallel_MPI& MPI_parallel, bool vtk_compatible, bool with_compression)
	: m_vtk_compatible(vtk_compatible)
	, use_compression(with_compression)
	, chunk_size{}
	, num_elem_chunk(0) {
	initialize(&geo, &MPI_parallel);
}

void H5_ALB::initialize(const stl_import* Geo, const Parallel_MPI* MPI_parallel) {
	dimsfile[2] = global_parameters.Nx;
	dimsfile[1] = global_parameters.Ny;
	dimsfile[0] = global_parameters.Nz;

	dimsmem[2] = MPI_parallel->end_XYZ[0] - MPI_parallel->start_XYZ[0] + 1;
	dimsmem[1] = MPI_parallel->end_XYZ[1] - MPI_parallel->start_XYZ[1] + 1;
	dimsmem[0] = MPI_parallel->end_XYZ[2] - MPI_parallel->start_XYZ[2] + 1;

	offset[2] = MPI_parallel->start_XYZ[0];
	offset[1] = MPI_parallel->start_XYZ[1];
	offset[0] = MPI_parallel->start_XYZ[2];

	ax0_st = MPI_parallel->start_XYZ2[2];
	ax0_en = MPI_parallel->end_XYZ2[2];

	ax1_st = MPI_parallel->start_XYZ2[1];
	ax1_en = MPI_parallel->end_XYZ2[1];

	ax2_st = MPI_parallel->start_XYZ2[0];
	ax2_en = MPI_parallel->end_XYZ2[0];
	//	}

	//////////// Required dimensions for writing the output files ////////////////////

	stride[0] = 1;
	stride[1] = 1;
	stride[2] = 1;

	count[0] = 1;
	count[1] = 1;
	count[2] = 1;

	block[0] = dimsmem[0];
	block[1] = dimsmem[1];
	block[2] = dimsmem[2];

	origin[0] = Geo->x_center;
	origin[1] = Geo->y_center;
	origin[2] = Geo->z_center;

	////////////////////////////////////////////////////////////////////////////////////
	is_master = MPI_parallel->is_master();

	if (is_master && use_compression && !is_compression_available()) {
		std::cerr << "[Error] Attempting to use compression which is not supported by this version of hdf5 (>=1.10 is required)." << std::endl;
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	if (use_compression) {
		chunk_size = get_max_chunk_size({block[0], block[1], block[2]});
		num_elem_chunk = chunk_size[0] * chunk_size[1] * chunk_size[2];
	}
}

bool H5_ALB::is_filter_available(int h5_filter_type) {
	htri_t avail = H5Zfilter_avail(h5_filter_type);
	if (!avail) {
		return false;
	}

	unsigned filter_info;
	herr_t status = H5Zget_filter_info(h5_filter_type, &filter_info);
	if (status < 0
	    || !(filter_info & H5Z_FILTER_CONFIG_ENCODE_ENABLED)
	    || !(filter_info & H5Z_FILTER_CONFIG_DECODE_ENABLED)) {
		return false;
	}
	return true;
}

bool H5_ALB::is_compression_available() {
	return is_filter_available(H5Z_FILTER_DEFLATE) && is_filter_available(H5Z_FILTER_SHUFFLE);
}

void H5_ALB::open_file(const std::string& filename, OPEN_MODE mode, const Index_vec3& domain_start, const Index_vec3& domain_end) {
	const hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
	H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

	constexpr const char* file_ending = ".h5";
	constexpr const char* file_ending_vtk = ".hdf";

	const std::string full_name = filename + (m_vtk_compatible ? file_ending_vtk : file_ending);
	if (mode == OPEN_MODE::WRITE) {
		// Create a new file collectively
		m_file_id = H5_call(H5Fcreate, full_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
		if (m_vtk_compatible) {
			write_vtk_info(domain_start, domain_end);
		} else {
			m_root_group_id = m_file_id;
		}
	} else {
		m_file_id = H5_call(H5Fopen, full_name.c_str(), H5F_ACC_RDONLY, plist_id);
		if (m_vtk_compatible) {
			const hid_t vtk_group_id = H5_call(H5Gopen, m_file_id, vtk_group_name, H5P_DEFAULT);
			m_root_group_id = H5_call(H5Gopen, vtk_group_id, point_data_group, H5P_DEFAULT);
			H5_call(H5Gclose, vtk_group_id);
		} else {
			m_root_group_id = m_file_id;
		}
	}

	H5_call(H5Pclose, plist_id);
}

void H5_ALB::close_file() {
	if (m_root_group_id != m_file_id) {
		H5_call(H5Gclose, m_root_group_id);
	}
	H5_call(H5Fclose, m_file_id);
	m_file_id = -1;  // mark member to make debugging easier
	m_root_group_id = -1;
}

void H5_ALB::write_vtk_info(const Index_vec3& domain_start, const Index_vec3& domain_end) {
	const hid_t vtk_group_id = H5_call(H5Gcreate, m_file_id, vtk_group_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	constexpr int version[] = {1, 0};
	H5_call(H5LTset_attribute_int, vtk_group_id, ".", "Version", version, 2);

	set_attribute_string(vtk_group_id, ".", "Type", "ImageData");

	// whole extend is the global size of the stored field and should start at 0
	const std::array<int, 6> whole_extend = {
		static_cast<int>(0),
		static_cast<int>(domain_end[0] - domain_start[0]),
		static_cast<int>(0),
		static_cast<int>(domain_end[1] - domain_start[1]),
		static_cast<int>(0),
		static_cast<int>(domain_end[2] - domain_start[2]),
	};

	const std::array<double, 3> orig = {
		origin[0] + domain_start[0] * global_parameters.D_x,
		origin[1] + domain_start[1] * global_parameters.D_x,
		origin[2] + domain_start[2] * global_parameters.D_x};
	H5_call(H5LTset_attribute_int, vtk_group_id, ".", "WholeExtent", whole_extend.data(), whole_extend.size());
	H5_call(H5LTset_attribute_double, vtk_group_id, ".", "Origin", orig.data(), 3);

	const std::array<double, 3> spacing = {global_parameters.D_x, global_parameters.D_x, global_parameters.D_x};
	H5_call(H5LTset_attribute_double, vtk_group_id, ".", "Spacing", spacing.data(), spacing.size());

	constexpr std::array<double, 9> direction = {
		1.0, 0.0, 0.0,  // x
		0.0, 1.0, 0.0,  // y
		0.0, 0.0, 1.0   // z
	};
	H5_call(H5LTset_attribute_double, vtk_group_id, ".", "Direction", direction.data(), direction.size());

	m_root_group_id = H5_call(H5Gcreate, vtk_group_id, point_data_group, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	set_attribute_string(vtk_group_id, ".", "Scalars", "Fluid");
	H5_call(H5Gclose, vtk_group_id);
}

// #define USE_H5L_STRING_ATTR
//  see https://github.com/HDFGroup/vol-rest/blob/91a07befb7b28068647a7f2a66a1839661b6808f/hdf5/hl/src/H5LT.c
//  for the H5L implementation
//  unlike H5LTset_attribute_string we do not check whether the attribute already exists
void H5_ALB::set_attribute_string(hid_t loc_id, const char* obj_name, const char* attr_name, const char* attr_data) {
#ifdef USE_H5L_STRING_ATTR
	H5_call(H5LTset_attribute_string, loc_id, obj_name, attr_name, attr_data);
#else

	const hid_t obj_id = H5_call(H5Oopen, loc_id, obj_name, H5P_DEFAULT);

	const hid_t attr_type = H5_call(H5Tcopy, H5T_C_S1);
	// no extra char for null
	const size_t attr_size = strlen(attr_data);

	H5_call(H5Tset_size, attr_type, attr_size);
	H5_call(H5Tset_strpad, attr_type, H5T_STR_NULLTERM);

	const hid_t attr_space_id = H5_call(H5Screate, H5S_SCALAR);

	// Create and write the attribute
	const hid_t attr_id = H5_call(H5Acreate2, obj_id, attr_name, attr_type, attr_space_id, H5P_DEFAULT, H5P_DEFAULT);
	H5_call(H5Awrite, attr_id, attr_type, attr_data);

	H5_call(H5Aclose, attr_id);
	H5_call(H5Sclose, attr_space_id);
	H5_call(H5Tclose, attr_type);
	H5_call(H5Oclose, obj_id);
#endif
}

std::array<hsize_t, 3> H5_ALB::get_max_chunk_size(std::array<hsize_t, 3> block_size) const {
	// the master process does not matter for chunk size so it should be large enough to not distort the results
	if (is_master) {
		block_size[0] = dimsfile[0];
		block_size[1] = dimsfile[1];
		block_size[2] = dimsfile[2];
	} else {
		for (int d = 0; d < 3; ++d) {
			// processes with size 0 are ignored for the purpose of chunk selection
			if (!block_size[d]) {
				block_size[d] = dimsfile[d];
			}
		}
	}

	std::array<hsize_t, 3> result;
	MPI_Allreduce(block_size.data(), result.data(), 3, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);
	return result;
}