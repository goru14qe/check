#include "VTK_ALB.h"
#include "../Geometry.h"
#include "../ALBORZ_GlobalVariables.h"
#include "../utils/Assert.h"
#include <fstream>  // file stream
#include <iostream>
#include <sstream>
#ifdef __linux__
#include <sys/stat.h>  //for mkdir
#endif
#ifdef __APPLE__
#include <sys/stat.h>  //for mkdir
#endif
#if defined _WIN64 || _WIN32
#include "direct.h"  //for mkdir
#endif

VTK_ALB::VTK_ALB(const std::string& filename_out, const stl_import& Geo, const Parallel_MPI& MPI_parallel, const double* physical_time)
	: m_physical_time(physical_time) {
	Initialize(filename_out, &Geo, &MPI_parallel);
}

void VTK_ALB::Initialize(const std::string& filename_out, const stl_import* Geo, const Parallel_MPI* MPI_parallel) {
	out_filename = filename_out;
	begin_vtk.resize(3);
	end_vtk.resize(3);
	begin_vtk_gen_co.resize(3);
	end_vtk_gen_co.resize(3);
	length_vtk.resize(3);
	if (MPI_parallel->processor_id != MASTER) {
		begin_vtk[0] = MPI_parallel->start_XYZ2[0];
		begin_vtk[1] = MPI_parallel->start_XYZ2[1];
		begin_vtk[2] = MPI_parallel->start_XYZ2[2];
		begin_vtk_gen_co[0] = MPI_parallel->start_XYZ[0];
		begin_vtk_gen_co[1] = MPI_parallel->start_XYZ[1];
		begin_vtk_gen_co[2] = MPI_parallel->start_XYZ[2];

		if (MPI_parallel->proc_position[0] < MPI_parallel->Np_X - 1) { end_vtk[0] = MPI_parallel->end_XYZ2[0] + 1; }
		if (MPI_parallel->proc_position[0] == MPI_parallel->Np_X - 1) { end_vtk[0] = MPI_parallel->end_XYZ2[0]; }
		if (MPI_parallel->proc_position[1] < MPI_parallel->Np_Y - 1) { end_vtk[1] = MPI_parallel->end_XYZ2[1] + 1; }
		if (MPI_parallel->proc_position[1] == MPI_parallel->Np_Y - 1) { end_vtk[1] = MPI_parallel->end_XYZ2[1]; }
		if (MPI_parallel->proc_position[2] < MPI_parallel->Np_Z - 1) { end_vtk[2] = MPI_parallel->end_XYZ2[2] + 1; }
		if (MPI_parallel->proc_position[2] == MPI_parallel->Np_Z - 1) { end_vtk[2] = MPI_parallel->end_XYZ2[2]; }

		if (MPI_parallel->proc_position[0] < MPI_parallel->Np_X - 1) { end_vtk_gen_co[0] = MPI_parallel->end_XYZ[0] + 1; }
		if (MPI_parallel->proc_position[0] == MPI_parallel->Np_X - 1) { end_vtk_gen_co[0] = MPI_parallel->end_XYZ[0]; }
		if (MPI_parallel->proc_position[1] < MPI_parallel->Np_Y - 1) { end_vtk_gen_co[1] = MPI_parallel->end_XYZ[1] + 1; }
		if (MPI_parallel->proc_position[1] == MPI_parallel->Np_Y - 1) { end_vtk_gen_co[1] = MPI_parallel->end_XYZ[1]; }
		if (MPI_parallel->proc_position[2] < MPI_parallel->Np_Z - 1) { end_vtk_gen_co[2] = MPI_parallel->end_XYZ[2] + 1; }
		if (MPI_parallel->proc_position[2] == MPI_parallel->Np_Z - 1) { end_vtk_gen_co[2] = MPI_parallel->end_XYZ[2]; }

		length_vtk[0] = end_vtk[0] - begin_vtk[0] + 1;
		length_vtk[1] = end_vtk[1] - begin_vtk[1] + 1;
		length_vtk[2] = end_vtk[2] - begin_vtk[2] + 1;
	}
	m_processor_id = MPI_parallel->processor_id;
	prepare_file(*MPI_parallel, *Geo);
}

void VTK_ALB::swap_endian(char* buffer, size_t size) {
	for (size_t b = 0; b < size / 2; b++) {
		char temp = buffer[b];
		buffer[b] = buffer[size - 1 - b];
		buffer[size - 1 - b] = temp;
	}
}
void VTK_ALB::swap_endian(const char* buffer, char* buffer2, size_t size) {
	for (size_t b = 0; b < size; b++) {
		buffer2[b] = buffer[size - 1 - b];
	}
}
void VTK_ALB::write_bigendian(std::ofstream& file, const char* buffer, size_t count, size_t size) {
	char* _buf = new char[size];
	for (size_t i = 0; i < count; i++) {
		swap_endian(buffer + (size * i), _buf, size);
		file.write(_buf, size);
	}
	delete[] _buf;
}

// ******************************************************************* //
void VTK_ALB::prepare_file(const Parallel_MPI& parallel_MPI, const stl_import& geo) {
	std::stringstream file_content;
	if (m_processor_id == MASTER) {
		/* Write VTI XML header */
		file_content << "<?xml version=\"1.0\"?>\n";
		file_content << "<VTKFile type=\"PImageData\" version=\"0.1\">\n";
		file_content << "<PImageData WholeExtent=\"";
		file_content << std::setprecision(14) << parallel_MPI.start_XYZ[0] << " " << global_parameters.Nx - 1 << " ";
		file_content << std::setprecision(14) << parallel_MPI.start_XYZ[1] << " " << global_parameters.Ny - 1 << " ";
		file_content << std::setprecision(14) << parallel_MPI.start_XYZ[2] << " " << global_parameters.Nz - 1 << " "
					 << "\" ";
		file_content << std::setprecision(14) << "Origin=\" " << geo.x_center << " " << geo.y_center << " " << geo.z_center << "\" Spacing=\"" << global_parameters.D_x << " " << global_parameters.D_x << " " << global_parameters.D_x << "\" GhostLevel=\"0\">\n";
		file_content << "<PPointData Scalars=\"Fluid\">\n";
		m_file_header = file_content.str();

		file_content.str("");
		for (int i = 1; i < parallel_MPI.num_processors; i++) {
			file_content.str("");
			file_content << "<Piece Extent=\"";
			begin_vtk[0] = parallel_MPI.start_XYZ[0 + 3 * i];
			begin_vtk[1] = parallel_MPI.start_XYZ[1 + 3 * i];
			begin_vtk[2] = parallel_MPI.start_XYZ[2 + 3 * i];
			end_vtk[0] = parallel_MPI.end_XYZ[0 + 3 * i] + 1;
			end_vtk[1] = parallel_MPI.end_XYZ[1 + 3 * i] + 1;
			end_vtk[2] = parallel_MPI.end_XYZ[2 + 3 * i] + 1;
			if (parallel_MPI.end_XYZ[0 + 3 * i] == global_parameters.Nx - 1) end_vtk[0] -= 1;
			if (parallel_MPI.end_XYZ[1 + 3 * i] == global_parameters.Ny - 1) end_vtk[1] -= 1;
			if (parallel_MPI.end_XYZ[2 + 3 * i] == global_parameters.Nz - 1) end_vtk[2] -= 1;

			file_content << std::setprecision(14) << begin_vtk[0] << " " << end_vtk[0] << " ";
			file_content << std::setprecision(14) << begin_vtk[1] << " " << end_vtk[1] << " ";
			file_content << std::setprecision(14) << begin_vtk[2] << " " << end_vtk[2] << "\" ";
			m_piece_extends.emplace_back(file_content.str());
		}
	} else {
		/* Write VTI XML header */
		file_content << "<?xml version=\"1.0\"?>\n";
		file_content << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"BigEndian\" header_type=\"UInt64\">\n";
		file_content << "<ImageData WholeExtent=\"";
		file_content << std::setprecision(14) << begin_vtk_gen_co[0] << " " << end_vtk_gen_co[0] << " ";
		file_content << std::setprecision(14) << begin_vtk_gen_co[1] << " " << end_vtk_gen_co[1] << " ";
		file_content << std::setprecision(14) << begin_vtk_gen_co[2] << " " << end_vtk_gen_co[2] << "\" ";
		file_content << std::setprecision(14) << "Origin=\" " << geo.x_center << " " << geo.y_center << " " << geo.z_center << "\" Spacing=\"" << global_parameters.D_x << " " << global_parameters.D_x << " " << global_parameters.D_x << "\">\n";
		file_content << std::setprecision(14) << "<Piece Extent=\"" << begin_vtk_gen_co[0] << " " << end_vtk_gen_co[0] << " ";
		file_content << std::setprecision(14) << begin_vtk_gen_co[1] << " " << end_vtk_gen_co[1] << " ";
		file_content << std::setprecision(14) << begin_vtk_gen_co[2] << " " << end_vtk_gen_co[2] << "\">\n";
		file_content << "<PointData Scalars=\"Fluid\">\n";

		m_file_header = file_content.str();
	}
}

void VTK_ALB::open_file(const std::string& filename, OPEN_MODE mode) {
	ASSERT_EXT(mode == OPEN_MODE::WRITE, "Only writing is implemented for vtk.");
	// the actual name part is needed later for the master file but without path
	auto start = filename.find_last_of('/');
	start = start == std::string::npos ? 0 : start + 1;
	m_filename = filename.substr(start);

	m_pointer_position = 0;
	m_data_write_ops.clear();
	const std::string full_name = make_filename(filename, m_processor_id);
	m_file.open(full_name, std::ios::binary | ios::out);
	if (!m_file){
		throw IO_exception(std::string("Could not open file \"") + full_name + "\" for writing.");
	}

	if (m_processor_id == MASTER) {
		update_pvd(m_filename + ".pvti");
	}

	m_file << m_file_header;
}

void VTK_ALB::close_file() {
	if (m_processor_id == MASTER) {
		m_file << "</PPointData>\n";
		for (int i = 0; i < static_cast<int>(m_piece_extends.size()); i++) {
			m_file << m_piece_extends[i]
				   << "Source=\"" << make_filename(m_filename, i + 1) << "\"/>\n";
		}
		m_file << "</PImageData>\n"
			   << "</VTKFile>\n";
	} else {
		m_file << "</PointData>\n"
			   << "</Piece>\n"
			   << "</ImageData>\n";

		m_file << "<AppendedData encoding=\"raw\">\n"
			   << "_";

		for (auto& write_op : m_data_write_ops) {
			write_op(m_file);
		}

		m_file << "</AppendedData>\n"
			   << "</VTKFile>\n";
	}

	m_file.close();
}

std::string VTK_ALB::make_filename(const std::string& name, int proc_id) {
	std::stringstream full_name;
	if (proc_id == MASTER) {
		full_name << name << ".pvti";
	} else {
		full_name << name << "_" << proc_id << ".vti";
	}
	return full_name.str();
}

void VTK_ALB::update_pvd(const std::string& file_to_add) const {
	const std::string pvd_name = out_filename + ".pvd";

	// read in current file skipping closing statements
	std::vector<std::string> lines;
	{
		ifstream pvd_file(pvd_name);
		std::string line;
		while (std::getline(pvd_file, line)) {
			if (line.find("</Collection>") == std::string::npos
			    && line.find("</VTKFile>") == std::string::npos) {
				lines.push_back(line);
			}
		}
	}
	ofstream pvd_file(pvd_name, std::fstream::out);
	if (lines.empty()) {
		/// Write PVD XML header
		pvd_file << "<?xml version=\"1.0\"?>\n";
		pvd_file << "<VTKFile type=\"Collection\" version=\"0.1\"\n";
		pvd_file << "byte_order=\"LittleEndian\"\n";
		pvd_file << "compressor=\"vtkZLibDataCompressor\">\n";
		pvd_file << "<Collection>\n";
	}

	for (const std::string& line : lines) {
		pvd_file << line << "\n";
	}

	// add current step
	const double t = m_physical_time ? *m_physical_time : -1.0;
	pvd_file << "<DataSet timestep=\"" << t << "\" group=\"\" part=\"0\"\n";
	pvd_file << "file=\"" << file_to_add << "\"/>\n";

	/// close PVD XML statements
	pvd_file << "</Collection>\n";
	pvd_file << "</VTKFile>\n";
}
