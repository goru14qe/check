#include "simulations/Loop_flow_particle.h"
#include "simulations/Loop_flow_heat_full_lb.h"
#include "simulations/Loop_flow_crystal.h"
#include "simulations/Loop_flow_reactive.h"
#include <unordered_map>

#ifndef NDEBUG
// enable to debug with mpi
#define DEBUG_MPI
#endif
#ifdef DEBUG_MPI
#include <thread>
#endif

using namespace std;  // permanently use the standard namespace

int main(int argc, char** argv) {
#ifdef DEBUG_MPI
	// wait so that it is possible to attach a debugger before entering critical sections
	std::this_thread::sleep_for(8s);
#endif
	Parallel_MPI parallel_MPI;
	parallel_MPI.Initialize_MPI(argc, argv);

	string filename;
	// read config file name from command line if given
	if (argc > 1) {
		filename = argv[1];
	} else {
		// otherwise look for a file
		constexpr const char* dummy_filename = "input.dat";
		ifstream name_file(dummy_filename, ios::binary);
		name_file >> filename;
	}
	const string complete_filename = filename + ".dat";
	// check that the config file exists
	if (!ifstream(complete_filename).is_open()) {
		if (parallel_MPI.is_master()) {
			std::cerr << "[Error] Could not open the parameter file \""
					  << complete_filename << "\".\n";
		}
		return 1;
	}

	std::string simulation_type_name = "flow_reactive";
	if (argc > 2) {
		simulation_type_name = argv[2];
	}

	Output_options out_opts;
	if (argc > 3) {
		if (argv[3] == "--use_compression"s) {
			if (parallel_MPI.is_master()) {
				std::cout << "Using compression." << std::endl;
			}
			out_opts.use_compression = true;
		}
	}

	// add new loops here to make them available
	std::unordered_map<std::string, std::function<Loop*()>> simulation_types = {
		{"flow_heat_full_lb", [&]() { return new Loop_flow_heat_full_lb(parallel_MPI, out_opts); }},
		{"flow_particle", [&]() { return new Loop_flow_particle(parallel_MPI, out_opts); }},
#ifdef WITH_CANTERA
		{"flow_reactive", [&]() { return new Loop_flow_reactive(parallel_MPI, out_opts);}},
#endif
		{"flow_crystal_lb", [&]() { return new Loop_flow_crystal_LB(parallel_MPI, out_opts); }},
		{"flow_crystal_fd", [&]() { return new Loop_flow_crystal_FD(parallel_MPI, out_opts); }}};

	auto it = simulation_types.find(simulation_type_name);
	if (it != simulation_types.end()) {
		if (parallel_MPI.is_master()) {
			std::cout << "Running simulation of type \"" << simulation_type_name << "\".\n";
		}
		// create simulation loop of the selected type
		std::unique_ptr<Loop> loop(it->second());
		loop->init(filename);
		loop->run();
	} else {
		if (parallel_MPI.is_master()) {
			std::cerr << "[Error] Unknown simulation type \"" << simulation_type_name << "\".\n"
					  << "Available types are: \n";
			for (auto& sim_type : simulation_types) {
				std::cout << sim_type.first << "\n";
			}
		}
	}

	return 0;
}  // end of MAIN function
