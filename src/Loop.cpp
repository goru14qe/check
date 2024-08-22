#define _USE_MATH_DEFINES
#include <cmath>     // mathematical library
#include <iostream>  // for the use of 'cout'
#include <fstream>   // file stream
#include <sstream>   // string streams
#include <cstdlib>   // standard library
#include <iomanip>   // For set precision. From ver 28
#include <chrono>
#ifdef __linux__
#include <sys/stat.h>  //for mkdir
#endif
#ifdef __APPLE__
#include <sys/stat.h>  //for mkdir
#endif
#if defined _WIN64 || _WIN32
#include "direct.h"  //for mkdir
#endif

#include "ALBORZ_Macros.h"
#include "ALBORZ_SETTINGS.h"
#include "ALBORZ_GlobalVariables.h"
#include "Loop.h"

Loop::Loop(Parallel_MPI& _parallel_MPI, const Output_options& options)
	: parallel_MPI(_parallel_MPI)
	, output_options(options)
	, m_recovery_task(nullptr) {
	stencil_list.Initialize();
}

void Loop::init(const std::string& config_file_path) {
	initialize(config_file_path);
	prepare_recovery_task();

	// dummy task to create the debug outputs folder
	add_io_task(0, "debug", "debug", Output_types::NO, std::numeric_limits<int>::max());

	// call after all tasks have been created
	prepare_output_dirs();
}

void Loop::run() {
	if (parallel_MPI.processor_id == MASTER + 1) {
		cout << "\n";
		cout << "\t|-------------------------------------|\n";
		cout << "\t|     ... STARTING SIMULATION ...     |\n";
		cout << "\t|-------------------------------------|\n";
		// todo: remove or add loop specific description
		//	cout << "\t|  FLOW : weakly compressible solver  |\n";
		//	cout << "\t|  TEMPERATURE : LB solver            |\n";
		//	cout << "\t|  SPECIES : none                     |\n";
		//	cout << "\t|-------------------------------------|\n";
		cout << endl;
		srand(1);
	}

	// the master process might not be finished with the cleanup yet
	MPI_Barrier(MPI_COMM_WORLD);
	auto start_time = std::chrono::high_resolution_clock::now();

	// write outputs for step 0
	if (recovery_step == 0) {
		for (auto& io_task : m_io_tasks) {
			if (io_task->t_start == 0) {
				io_task->interface.write(make_path(*io_task), 0);
			}
		}
	}

	int steps_computed = 0;
	for (int tm = recovery_step + 1; tm <= t_num; tm++) {
		step(tm);
		++steps_computed;

		/// ***************************
		///  WRITE OUTPUTS
		/// ***************************
		for (auto& io_task : m_io_tasks) {
			if (tm >= io_task->t_start && tm % io_task->t_period == 0) {
				io_task->interface.write(make_path(*io_task), tm);
			}
		}

		/// ***************************
		///  REPORT TIME ON SCREEN
		/// ***************************
		parallel_MPI.Time_monitor(tm, t_time, 1 + recovery_step);
		parallel_MPI.Onscreen_Report(tm, t_num);

		// check custom stop criteria
		if (has_converged(tm)) {
			if (parallel_MPI.is_master()) {
				std::cout << "simulation has converged at step " << tm << "\n";
			}
			break;
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
	if (parallel_MPI.is_master()) {
		auto end_time = std::chrono::high_resolution_clock::now();
		const float total_time = std::chrono::duration<float>(end_time - start_time).count();
		std::cout << "\t|-------------------------|\n";
		std::cout << "\t|   END OF SIMULATION ... |\n";
		std::cout << "\t|-------------------------|\n";

		const unsigned cells = global_parameters.Nx * global_parameters.Ny * global_parameters.Nz;
		std::cout << "total simulation time: " << total_time << "s"
				  << ", MLUPS: " << cells / static_cast<double>(parallel_MPI.num_processors - 1) * steps_computed / total_time * 1e-6
				  << "\n";
	}
}

IO_interface& Loop::add_io_task(int t_period, const std::string& path, const std::string& name,
                                Output_types out_types, int t_start,
                                const Index_vec3& offset, const Index_vec3& sizes) {
	if (out_types == Output_types::AUTO) {
		out_types = output_options.types;
	}
	const bool use_vtk = out_types == Output_types::VTK || out_types == Output_types::BOTH;
	const bool use_h5 = out_types == Output_types::H5 || out_types == Output_types::BOTH;
#ifndef WITH_HDF5
	if (parallel_MPI.is_master() && use_h5) {
		std::cout << "[Warning] The io task \"" << path << "/" << name
				  << "\" requests format h5 but this binary was build without hdf5 support.\n";
	}
#endif
	m_io_tasks.emplace_back(new IO_task{t_period, t_start, path, name,
	                                    IO_interface(geo_stl, parallel_MPI, use_vtk, use_h5, &flow_field.physical_time, output_options.output_path + '/' + path + '/' + name, offset, sizes, output_options.use_compression)});
	return m_io_tasks.back()->interface;
}

void Loop::recover_state() {
	prepare_recovery_task();
	m_recovery_task->interface.read(make_path(*m_recovery_task), recovery_step);
}

void Loop::prepare_recovery_task() {
	// already initialized
	if (m_recovery_task) return;

	// outputs need to be ready before the recovery task is created
	// because of Average_field
	register_outputs();

	// start at step 1 because the initial state can be recovery from just the config
	IO_interface& recovery_interface = add_io_task(t_recovery, "recovery", "recover",
	                                               Output_types::H5, 1);
	recovery_interface.add_scalar(global_parameters.D_t, "global_D_t");
	register_recovery(recovery_interface);
	m_recovery_task = m_io_tasks.back().get();
}

void Loop::prepare_output_dirs() {
	if (parallel_MPI.is_master()) {
		if (output_options.reset_output_dir) {
#if defined __linux__ || __APPLE__
			constexpr const char* command = "rm -rfv ";
#elif defined _WIN64 || _WIN32
			constexpr const char* command = "rmdir ";
#endif
			const std::string full_command = command + output_options.output_path;
			if (system(full_command.c_str())) {
				std::cout << "[Warning] Could not delete directory \""
						  << output_options.output_path << "\".\n";
			}
		}

		// ensures that the directory exists and is empty for a new simulation
		auto update_dir = [](const std::string& path) {
#if defined __linux__ || __APPLE__
			mkdir(path.c_str(), 0777);
			constexpr const char* command = "rm -f ";
#elif defined _WIN64 || _WIN32
			_mkdir(path.c_str(), 0777);
			constexpr const char* command = "del /Q ";
#endif
			if (recovery_step == 0) {
				const std::string full_command = command + path + "/*.*";
				if (system(full_command.c_str())) {
					std::cout << "[Warning] Something went wrong cleaning up the results directory \""
							  << path << "\".\n";
				}
			}
		};
		// root directory first
		update_dir(output_options.output_path);

		for (const auto& io_task : m_io_tasks) {
			const std::string full_path = output_options.output_path + '/' + io_task->path;
			update_dir(full_path);
		}
	}
}

std::string Loop::make_path(const IO_task& task) const {
	return output_options.output_path + '/'
	       + task.path + '/' + task.name;
}