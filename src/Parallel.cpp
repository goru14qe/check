#include "Parallel.h"
#include "ALBORZ_GlobalVariables.h"
#include "ALBORZ_Macros.h"
#include "utils/Config_utils.h"
#include <fstream>  // file stream
#include <sstream>

Parallel_MPI::Parallel_MPI() {
}

Parallel_MPI::Parallel_MPI(int argc, char** argv) {
	Initialize_MPI(argc, argv);
}

Parallel_MPI::~Parallel_MPI() {
	MPI_Finalize();
}
/// ***************************************************** ///
/// INITIALIZE MPI ENGINE                                 ///
/// ***************************************************** ///
void Parallel_MPI::Initialize_MPI(int argc, char** argv) {
	/// **************************
	/// MPI INITIALIZATION
	/// **************************
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &processor_id);
	MPI_Comm_size(MPI_COMM_WORLD, &num_processors);
}
/// ***************************************************** ///
/// DOMAIN DECOMPOSITION: DISTRIBUTION OF DOMAIN          ///
/// ***************************************************** ///
void Parallel_MPI::Domain_Decomp(unsigned int Dimension, const std::string& filename) {
	/// -----------------------> Domain decomposition (definition of start and end of indexes)
	///    In this first section the number of processing units in each direction are defined
	///    for a 1-D or 2-D domain decomposition the values of the non-defined directions shall
	///    be set to 1
	if (processor_id == MASTER) {
		/// Open input file
		std::ifstream input_file(filename + ".dat", std::ios::binary);
		find_line_after_header(input_file, "c\tParallel Processing");
		find_line_after_comment(input_file);
		input_file >> Np_X >> Np_Y >> Np_Z;

		input_file.seekg(0);
		find_line_after_header(input_file, "c\tFlow Field Boundary Conditions");
		find_line_after_comment(input_file);
		int dummy;
		bool curved_boundary = false;
		input_file >> dummy >> curved_boundary;
		if (curved_boundary) {
			double radius = 2.0;
			if (!(input_file >> radius)) {
				std::cerr << "[Error] Could not read stencil radius for curved boundaries." << std::endl;
				MPI_Abort(MPI_COMM_WORLD, 1);
			}
			// worst case: First img point is the first ghost node, then radius >=2 would require
			// buffer_size == 3 to access all points in range.
			// However, the influence at max radius is 0, so this point is excluded.
			// The second img point is one grid point further, so we still need +1.
			buffer_size = std::max(2.0, std::floor(radius) + 1.0);
		} else {
			buffer_size = 2;
		}
		input_file.close();
		if (buffer_size > global_parameters.Nx || buffer_size > global_parameters.Ny 
			|| (Dimension == 3 &&  buffer_size > global_parameters.Nz)) {
			std::cerr << "[Error] The number of ghost nodes required (" << buffer_size
					  << ") exceeds the size of the domain." << std::endl;
			MPI_Abort(MPI_COMM_WORLD, 1);
		}

		const unsigned num_processors_exp = Np_X * Np_Y * Np_Z + 1;
		if (num_processors_exp != num_processors) {
			std::cerr << "[Error] Launched " << num_processors
					  << " processes but the domain decomposition requires "
					  << num_processors_exp << ".\n";
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	}

	// num_processors
	MPI_Bcast(&Np_X, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&Np_Y, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&Np_Z, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&buffer_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
	proc_arrangement.resize(Np_X);
	for (int i = 0; i < proc_arrangement.size(); i++) {
		proc_arrangement[i].resize(Np_Y);
		for (int j = 0; j < proc_arrangement[i].size(); j++) {
			proc_arrangement[i][j].resize(Np_Z);
			for (int k = 0; k < Np_Z; k++) {
				proc_arrangement[i][j][k] = k + Np_Z * j + Np_Z * Np_Y * i + 1;
				if (proc_arrangement[i][j][k] == processor_id) {
					proc_position[0] = i;
					proc_position[1] = j;
					proc_position[2] = k;
				}
			}
		}
	}

	if (processor_id != MASTER) {
		proc_neighbours.resize(6);
		proc_neighbours[0] = proc_arrangement[(proc_position[0] + 1) % Np_X][proc_position[1]][proc_position[2]];
		proc_neighbours[1] = proc_arrangement[(proc_position[0] + Np_X - 1) % Np_X][proc_position[1]][proc_position[2]];
		proc_neighbours[2] = proc_arrangement[proc_position[0]][(proc_position[1] + 1) % Np_Y][proc_position[2]];
		proc_neighbours[3] = proc_arrangement[proc_position[0]][(proc_position[1] + Np_Y - 1) % Np_Y][proc_position[2]];
		proc_neighbours[4] = proc_arrangement[proc_position[0]][proc_position[1]][(proc_position[2] + 1) % Np_Z];
		proc_neighbours[5] = proc_arrangement[proc_position[0]][proc_position[1]][(proc_position[2] + Np_Z - 1) % Np_Z];
	}

	/// -------------------> Output configuration
	if (processor_id == MASTER) {
		std::stringstream out_buf;
		out_buf << "Parallel Processing\n=====================\n";
		out_buf << std::setw(COLUMN_WIDTH) << std::left
				<< "exchanged ghost nodes = " << buffer_size << "\n";

		for (int j = Np_Y - 1; j > -1; j--) {
			out_buf << "|";
			for (int i = 0; i < Np_X; i++) {
				out_buf << "---|";
			}
			out_buf << "\n";
			out_buf << "|";
			for (int i = 0; i < Np_X; i++) {
				out_buf << std::setw(3);
				out_buf << proc_arrangement[i][j][0] << "|";
			}
			out_buf << "\n";
		}
		out_buf << "|";
		for (int i = 0; i < Np_X; i++) {
			out_buf << "---|";
		}
		std::cout << out_buf.str() << std::endl;
	}
	avg_nod_per_process[0] = floor(global_parameters.Nx / Np_X);
	avg_nod_per_process[1] = floor(global_parameters.Ny / Np_Y);
	avg_nod_per_process[2] = floor(global_parameters.Nz / Np_Z);

	dev_start[0] = 0;
	dev_start[1] = 0;
	dev_start[2] = 0;
	dev_end[0] = avg_nod_per_process[0] + 2 * buffer_size;
	dev_end[1] = avg_nod_per_process[1] + 2 * buffer_size;
	dev_end[2] = avg_nod_per_process[2] + 2 * buffer_size;

	if (processor_id == MASTER) {
		// check validity
		for (int d = 0; d < 3; ++d) {
			if (avg_nod_per_process[d] == 0) {
				std::cerr << "[Error] Bad domain decomposition. Most processes are assigned no nodes because of dimension "
						  << d << " with size 0." << std::endl;
				MPI_Abort(MPI_COMM_WORLD, 1);
			}
		}

		dev_end[0] = global_parameters.Nx;
		dev_end[1] = global_parameters.Ny;
		dev_end[2] = global_parameters.Nz;
		start_XYZ.resize(3 * num_processors);
		end_XYZ.resize(3 * num_processors);
		start_XYZ2.resize(3 * num_processors);
		end_XYZ2.resize(3 * num_processors);
		actual_rows_XYZ.resize(3 * num_processors);
	}
	if (processor_id != MASTER) {
		if (proc_position[0] == Np_X - 1 && Np_X > 1) dev_end[0] = global_parameters.Nx - avg_nod_per_process[0] * (Np_X - 1) + 2 * buffer_size;
		if (proc_position[1] == Np_Y - 1 && Np_Y > 1) dev_end[1] = global_parameters.Ny - avg_nod_per_process[1] * (Np_Y - 1) + 2 * buffer_size;
		if (proc_position[2] == Np_Z - 1 && Np_Z > 1) dev_end[2] = global_parameters.Nz - avg_nod_per_process[2] * (Np_Z - 1) + 2 * buffer_size;
		start_XYZ.resize(3);
		end_XYZ.resize(3);
		start_XYZ2.resize(3);
		end_XYZ2.resize(3);
		actual_rows_XYZ.resize(3);
		/// ---------------------> X-direction
		start_XYZ[0] = proc_position[0] * avg_nod_per_process[0];
		end_XYZ[0] = start_XYZ[0] + avg_nod_per_process[0] - 1;
		if (proc_position[0] == Np_X - 1) end_XYZ[0] = global_parameters.Nx - 1;
		actual_rows_XYZ[0] = end_XYZ[0] - start_XYZ[0] + 1;
		start_XYZ2[0] = buffer_size;
		end_XYZ2[0] = actual_rows_XYZ[0] + buffer_size - 1;
		/// ---------------------> Y-direction
		start_XYZ[1] = proc_position[1] * avg_nod_per_process[1];
		end_XYZ[1] = start_XYZ[1] + avg_nod_per_process[1] - 1;
		if (proc_position[1] == Np_Y - 1) end_XYZ[1] = global_parameters.Ny - 1;
		actual_rows_XYZ[1] = end_XYZ[1] - start_XYZ[1] + 1;
		start_XYZ2[1] = buffer_size;
		end_XYZ2[1] = actual_rows_XYZ[1] + buffer_size - 1;
		/// ---------------------> Z-direction
		start_XYZ[2] = proc_position[2] * avg_nod_per_process[2];
		end_XYZ[2] = start_XYZ[2] + avg_nod_per_process[2] - 1;
		if (proc_position[2] == Np_Z - 1) end_XYZ[2] = global_parameters.Nz - 1;
		actual_rows_XYZ[2] = end_XYZ[2] - start_XYZ[2] + 1;
		start_XYZ2[2] = buffer_size;
		end_XYZ2[2] = actual_rows_XYZ[2] + buffer_size - 1;
	}
	/// ---------------------> Transfer data from slave threads to master thread
	std::vector<unsigned int> buffer(3 * num_processors);
	MPI_Gather(&start_XYZ[0], 3, MPI_INT, &buffer[0], 3, MPI_INT, MASTER, MPI_COMM_WORLD);
	if (processor_id == MASTER) start_XYZ = buffer;

	MPI_Gather(&end_XYZ[0], 3, MPI_INT, &buffer[0], 3, MPI_INT, MASTER, MPI_COMM_WORLD);
	if (processor_id == MASTER) end_XYZ = buffer;
	MPI_Gather(&start_XYZ2[0], 3, MPI_INT, &buffer[0], 3, MPI_INT, MASTER, MPI_COMM_WORLD);
	if (processor_id == MASTER) start_XYZ2 = buffer;
	MPI_Gather(&end_XYZ2[0], 3, MPI_INT, &buffer[0], 3, MPI_INT, MASTER, MPI_COMM_WORLD);
	if (processor_id == MASTER) end_XYZ2 = buffer;
	MPI_Gather(&actual_rows_XYZ[0], 3, MPI_INT, &buffer[0], 3, MPI_INT, MASTER, MPI_COMM_WORLD);
	if (processor_id == MASTER) actual_rows_XYZ = buffer;
}
/// ***************************************************** ///
/// COMPUTE CALCULATION TIME                              ///
/// ***************************************************** ///
void Parallel_MPI::Time_monitor(unsigned int tm, unsigned int t_time, unsigned int tstart) {
	if (tm == tstart) {
		MPI_Barrier(MPI_COMM_WORLD);
		t0 = MPI_Wtime();
		t_current = 0;
		t_old = 0;
		step_current = tm;
		step_old = tm;
	} else if (tm % t_time == 0) {
		double t_max;
		t_old = t_current;
		t_current = MPI_Wtime() - t0;
		step_old = step_current;
		step_current = tm;
		delta_t = ((double)t_current - t_old + 1e-13) / (step_current - step_old + 1e-13);
		// since synchronization happens in every step they should all be roughly equal
		// and limited by the slowest one in the end
		MPI_Reduce(&delta_t, &t_max, 1, MPI_DOUBLE, MPI_MAX, MASTER, MPI_COMM_WORLD);

		if (is_master()) {
			const unsigned cells = global_parameters.Nx * global_parameters.Ny * global_parameters.Nz;
			const double mlups = cells / static_cast<double>(num_processors - 1) / t_max * 1e-6;
			std::cout << "AVG STEP-TIME[s]: " << std::setw(8) << t_max << "\tMLUPS:" << std::setw(8) << mlups << "\n";
		}
		MPI_Barrier(MPI_COMM_WORLD);
		t_current = MPI_Wtime() - t0;
	}
}
/// ***************************************************** ///
/// WRITE STEP INFO ON SCREEN                             ///
/// ***************************************************** ///
void Parallel_MPI::Onscreen_Report(int tm, int t_num) {
	/// ***************************
	///  REPORT ON COMMAND SCREEN
	/// ***************************
	/// Report end of time step
	if (tm % t_info == 0) {
		Sync_Master();
		if (processor_id == MASTER) {
			std::cout << "Completed Time Step " << tm << " out of " << t_num << std::endl;
		}
	}
}
void Parallel_MPI::Sync_Master() {
	if (processor_id != MASTER) {
		int mtype = FROM_WORKER;
		int status_Worker = 1;
		MPI_Isend(&status_Worker, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &request);
	}
	if (processor_id == MASTER) {
		int mtype = FROM_WORKER;
		int* status_Master;
		status_Master = new int[num_processors - 1];
		for (int i = 1; i < num_processors; i++) {
			MPI_Recv(&status_Master[i - 1], 1, MPI_INT, i, mtype, MPI_COMM_WORLD, &status);
		}
		delete[] status_Master;
	}
}
/// ***************************************************** ///
/// GET PHYSICAL COORDINATES                              ///
/// ***************************************************** ///
Vec3 Parallel_MPI::get_coordinates(const Index_vec3& idx, const Vec3& center) const {
	return {
		(idx[0] - start_XYZ2[0] + start_XYZ[0] + global_parameters.Nx) % global_parameters.Nx * global_parameters.D_x + center[0],
		(idx[1] - start_XYZ2[1] + start_XYZ[1] + global_parameters.Ny) % global_parameters.Ny * global_parameters.D_x + center[1],
		(idx[2] - start_XYZ2[2] + start_XYZ[2] + global_parameters.Nz) % global_parameters.Nz * global_parameters.D_x + center[2]};
}

Vec3 Parallel_MPI::get_coordinates(const Vec3& pos, const Vec3& center) const {
	return {
		fmod(pos[0] - start_XYZ2[0] + start_XYZ[0] + global_parameters.Nx, global_parameters.Nx) * global_parameters.D_x + center[0],
		fmod(pos[1] - start_XYZ2[1] + start_XYZ[1] + global_parameters.Ny, global_parameters.Ny) * global_parameters.D_x + center[1],
		fmod(pos[2] - start_XYZ2[2] + start_XYZ[2] + global_parameters.Nz, global_parameters.Nz) * global_parameters.D_x + center[2]};
}

Vec3 Parallel_MPI::get_coordinates(const Index_vec3& pos) const {
	return get_coordinates(pos, center);
}

Vec3 Parallel_MPI::get_coordinates(const Vec3& pos) const {
	return get_coordinates(pos, center);
}

void Parallel_MPI::get_coordinates(double X, double Y, double Z,
                                   double ref_x, double ref_y, double ref_z,
                                   double& xx, double& yy, double& zz) const {
	xx = (X - double(start_XYZ2[0]) + double(start_XYZ[0])) * global_parameters.D_x + ref_x;
	yy = (Y - double(start_XYZ2[1]) + double(start_XYZ[1])) * global_parameters.D_x + ref_y;
	zz = (Z - double(start_XYZ2[2]) + double(start_XYZ[2])) * global_parameters.D_x + ref_z;
}

bool Parallel_MPI::is_master() const {
	return processor_id == MASTER;
}

/// ***************************************************** ///
Data_exchange_group::Data_exchange_group(const Parallel_MPI& parallel_MPI)
	: parallel_MPI(&parallel_MPI) {
}

Data_exchange_group::~Data_exchange_group() {
	if (!parallel_MPI) return;

	for (auto& info : exchange_infos) {
		for (int i = 0; i < 6; ++i) {
			MPI_Type_free(&info.send_datatypes[i]);
			MPI_Type_free(&info.receive_datatypes[i]);
		}
	}
}

/* For each exchange information, it performs data exchange with its six neighboring processes
(assuming a 3D Cartesian grid structure) using MPI's MPI_Sendrecv function. which exchanges data between
 two neighboring processes. It sends data from the current process to the neighbor specified by proc_neighbours[i] 
 and receives data from the neighbor specified by proc_neighbours[i + 1]. This is done twice (LTAG and RTAG) for each pair of neighbors.
*/
void Data_exchange_group::exchange_data() {
	ASSERT(!parallel_MPI->is_master());

	for (auto& info : exchange_infos) {
		for (int i = 0; i < 6; i += 2) {
			const auto data = info.data();
			MPI_Sendrecv(data, 1, info.send_datatypes[i], parallel_MPI->proc_neighbours[i], LTAG,
			             data, 1, info.receive_datatypes[i], parallel_MPI->proc_neighbours[i + 1], LTAG,
			             MPI_COMM_WORLD, &status);
			MPI_Sendrecv(data, 1, info.send_datatypes[i + 1], parallel_MPI->proc_neighbours[i + 1], RTAG,
			             data, 1, info.receive_datatypes[i + 1], parallel_MPI->proc_neighbours[i], RTAG,
			             MPI_COMM_WORLD, &status);
		}
	}
	exchange_pop_data();
}

void Data_exchange_group::add_population(Population_field& tensor, const std::vector<std::vector<int>>& c_alpha) {
	ASSERT(parallel_MPI);

	Exchange_pop_info info;
	info.tensor_ptr = &tensor;
	// neighbours expect a different ordering
	// static constexpr std::array<int, 3> DIM_PERMUTATION = {1,0,2};
	for (int dim = 0; dim < 3; ++dim) {
		for (int alpha = 0; alpha < c_alpha.size(); alpha++) {
			if (c_alpha[alpha][dim] > 0) { info.directions[dim * 2].push_back(alpha); }
			if (c_alpha[alpha][dim] < 0) { info.directions[dim * 2 + 1].push_back(alpha); }
		}
		size_t msg_size = 1;
		for (int dim2 = 0; dim2 < 3; ++dim2) {
			if (dim != dim2) {
				msg_size *= parallel_MPI->dev_end[dim2];
			}
		}
		info.msg_sizes[dim * 2] = msg_size * info.directions[dim * 2].size();
		info.msg_sizes[dim * 2 + 1] = msg_size * info.directions[dim * 2 + 1].size();
		const size_t max_size = std::max(info.msg_sizes[dim * 2], info.msg_sizes[dim * 2 + 1]);
		if (max_size > send_buffer.size()) {
			send_buffer.resize(max_size);
			receive_buffer.resize(max_size);
		}
	}
	exchange_pop_infos.push_back(info);
}
// Alternative population exchange using nested MPI_types.
// This version does not require a custom exchange routine, but is ~6 times slower than
// the manual version with OpenMPI.
/*
template <typename T>
void Data_exchange_group::add_population(Tensor<T, 4>& tensor, const std::vector<std::vector<int>>& c_alpha) {
    std::vector<int> block_lengths(c_alpha.size(), 1);
    std::vector<MPI_Datatype> types(c_alpha.size(), details::to_MPI_type<T>::value());
    std::array<MPI_Datatype, 6> datatypes;

    for (int dim = 0; dim < 3; ++dim) {
        std::vector<MPI_Aint> pop_to0;
        std::vector<MPI_Aint> pop_to1;

        // find list of indicies to be communicated
        for (int alpha = 0; alpha < c_alpha.size(); alpha++) {
            if (c_alpha[alpha][dim] > 0) { pop_to0.push_back(alpha * sizeof(T)); }
            if (c_alpha[alpha][dim] < 0) { pop_to1.push_back(alpha * sizeof(T)); }
        }

        MPI_Datatype base_datatype0;
        MPI_Datatype base_datatype1;
        MPI_Type_create_hindexed_block(pop_to0.size(), block_lengths.front(), pop_to0.data(), types.front(), &base_datatype0);
        MPI_Type_create_hindexed_block(pop_to1.size(), block_lengths.front(), pop_to1.data(), types.front(), &base_datatype1);
    //	MPI_Type_create_struct(pop_to0.size(), block_lengths.data(), pop_to0.data(), types.data(), &base_datatype0);
    //	MPI_Type_create_struct(pop_to1.size(), block_lengths.data(), pop_to1.data(), types.data(), &base_datatype1);
        // resize to full length so that array strides are correct
        MPI_Type_create_resized(base_datatype0, 0, tensor.sizes()[3] * sizeof(T), &base_datatype0);
        MPI_Type_create_resized(base_datatype1, 0, tensor.sizes()[3] * sizeof(T), &base_datatype1);
        MPI_Type_commit(&base_datatype0);
        MPI_Type_commit(&base_datatype1);
        datatypes[dim * 2] = base_datatype0;
        datatypes[dim * 2 + 1] = base_datatype1;
    }

    exchange_infos.push_back(make_exchange_info(tensor, 1, 1, datatypes));
}*/

// D1,D2 dimensions to iterate along
// D3 slice dimension
template <int D1, int D2, int D3, int MSG_IDX, int SEND_IDX, int RECEIVE_IDX>
void Data_exchange_group::exchange_pop_data_impl(const Exchange_pop_info& info, Index src_offset, Index dest_offset) {
	static_assert(D1 != D2 && D1 != D3 && D2 != D3, "each dimension can only appear once");
	Population_field& tensor = *info.tensor_ptr;
	const size_t num_dirs = info.directions[MSG_IDX].size();

	// Prepare messages to be sent
	Base_index_vec<4> pop_idx{0, 0, 0, 0};
	pop_idx[D3] = src_offset;
	for (Index X1 = 0; X1 < parallel_MPI->dev_end[D1]; X1++) {
		pop_idx[D1] = X1;
		for (Index X2 = 0; X2 < parallel_MPI->dev_end[D2]; X2++) {
			pop_idx[D2] = X2;
			const Flat_index dest_idx_flat = num_dirs * X2 + num_dirs * parallel_MPI->dev_end[D2] * X1;
			const Flat_index src_idx_flat = tensor.flat_index(pop_idx);
			for (Index i = 0; i < num_dirs; i++) {
				send_buffer[dest_idx_flat + i] = tensor[src_idx_flat + info.directions[MSG_IDX][i]];
			}
		}
	}
	MPI_Sendrecv(send_buffer.data(), info.msg_sizes[MSG_IDX], MPI_DOUBLE, parallel_MPI->proc_neighbours[SEND_IDX], LTAG,
	             receive_buffer.data(), info.msg_sizes[MSG_IDX], MPI_DOUBLE, parallel_MPI->proc_neighbours[RECEIVE_IDX], LTAG,
	             MPI_COMM_WORLD, &status);

	pop_idx[D3] = dest_offset;
	for (Index X1 = 0; X1 < parallel_MPI->dev_end[D1]; X1++) {
		pop_idx[D1] = X1;
		for (Index X2 = 0; X2 < parallel_MPI->dev_end[D2]; X2++) {
			pop_idx[D2] = X2;
			const Flat_index src_idx_flat = num_dirs * X2 + num_dirs * parallel_MPI->dev_end[D2] * X1;
			const Flat_index dest_idx_flat = tensor.flat_index(pop_idx);
			for (Index i = 0; i < num_dirs; i++) {
				tensor[dest_idx_flat + info.directions[MSG_IDX][i]] = receive_buffer[src_idx_flat + i];
			}
		}
	}
}

void Data_exchange_group::exchange_pop_data() {
	ASSERT(parallel_MPI->processor_id != MASTER);

	for (auto& info : exchange_pop_infos) {
		exchange_pop_data_impl<1, 2, 0, 0, 0, 1>(info, parallel_MPI->end_XYZ2[0] + 1, parallel_MPI->start_XYZ2[0]);
		exchange_pop_data_impl<1, 2, 0, 1, 1, 0>(info, parallel_MPI->start_XYZ2[0] - 1, parallel_MPI->end_XYZ2[0]);

		exchange_pop_data_impl<0, 2, 1, 2, 2, 3>(info, parallel_MPI->end_XYZ2[1] + 1, parallel_MPI->start_XYZ2[1]);
		exchange_pop_data_impl<0, 2, 1, 3, 3, 2>(info, parallel_MPI->start_XYZ2[1] - 1, parallel_MPI->end_XYZ2[1]);

		exchange_pop_data_impl<0, 1, 2, 4, 4, 5>(info, parallel_MPI->end_XYZ2[2] + 1, parallel_MPI->start_XYZ2[2]);
		exchange_pop_data_impl<0, 1, 2, 5, 5, 4>(info, parallel_MPI->start_XYZ2[2] - 1, parallel_MPI->end_XYZ2[2]);
	}
}