#ifndef LOOP_H_INCLUDED
#define LOOP_H_INCLUDED

#include "Geometry.h"
#include "Flow_solver.h"
#include "Thermal_solver.h"
#include "Species_solver.h"
#include "Fluid_read_write.h"
#include "Parallel.h"
#include "io/IO_interface.h"
#include <vector>

enum struct Output_types {
	NO,  // neither means just custom operations
	VTK,
	H5,
	BOTH,
	AUTO
};

struct Output_options {
	Output_types types = Output_types::VTK;
	// root path for all output directories
	std::string output_path = "Alborz_Results";
	// usually only files in directories with associated output tasks are deleted
	// if this this true, everything inside output_path is deleted
	bool reset_output_dir = false;
	// enable compression for hdf5 outputs and recovery files
	bool use_compression = false;
};

// The loop combines the different solvers to a simulation and manages time-steps and IO.
class Loop {
public:
	// the parallel enviornment can not be part of the loop because it needs to be
	// initialized before reading in the config file
	Parallel_MPI& parallel_MPI;
	Flow_solver flow_field;
	Thermal_solver temperature_field;
	Species_solver species_field;
	Geometry geo;
	stl_import geo_stl;
	DdQq stencil_list;
	Fluid_read_write fluid_read_write;

	Output_options output_options;

	// @param _parallel_MPI The parallel enviornment used in for all operations.
	//	                    It needs to live atleast as long as the loop.
	// todo: move output options into the config file?
	Loop(Parallel_MPI& _parallel_MPI, const Output_options& options = {});
	// Declare virtual destructor so that derived Loop-classes are properly destroyed.
	virtual ~Loop() = default;

	void init(const std::string& config_file_path);

	// Run the simulation for the time specified in the config.
	void run();

protected:
	// Initialize the solvers with the given config file.
	virtual void initialize(const std::string& config_file_path) = 0;
	// Perform a simulation step.
	virtual void step(int tm) = 0;
	virtual bool has_converged(int tm) { return false; }
	// Register data of interest that will be written to files in regular intervals.
	virtual void register_outputs() = 0;
	// Register all data that is necessary to restore the simulation state.
	virtual void register_recovery(IO_interface& io_interface) = 0;

	struct IO_task {
		int t_period;      // period after which a write should happen
		int t_start;       // first step where this task is considered
		std::string path;  // local part of the path relative to output_path
		std::string name;  // identifier that is part of the actual filename
		IO_interface interface;
	};

	// Returns a reference to the interface to register data.
	// @param path   - Relative path starting from Output_options::output_path for outputs.
	// @param name   - Prefix for files written by this task.
	// @param offset - Global start of the region of fields to be included in nodes.
	// @param sizes  - Number of nodes to include in the region, starting from offset.
	//				   Value 0 is interpreted as the whole domain (i.e. run_parameters.D{x,y,z}).
	//				   Paraview has trouble with dimensions of size 1, so it is better to always
	//				   save at least 2 layers.
	IO_interface& add_io_task(int t_period, const std::string& path, const std::string& name,
	                          Output_types out_types = Output_types::AUTO,
	                          int t_start = 0,
	                          const Index_vec3& offset = {}, const Index_vec3& sizes = {});

	// Restores a previous simulation state by reading the recovery task.
	void recover_state();

private:
	void prepare_recovery_task();
	// Creates output directories and cleans up from previous runs.
	void prepare_output_dirs();

	std::string make_path(const IO_task& task) const;

	bool should_write_vtk(Output_types out_types) const;
	bool should_write_h5(Output_types out_types) const;

	std::vector<std::unique_ptr<IO_task>> m_io_tasks;
	IO_task* m_recovery_task;
};

#endif  // LOOP_H_INCLUDED