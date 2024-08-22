#include "Loop_flow_heat_full_lb.h"
#include "../utils/Timer.hpp"

Loop_flow_heat_full_lb::~Loop_flow_heat_full_lb() {
	// we look for the slowest process to get relevant measurements
	// other processes will be inaccurate due to waiting
	const double t_lbm = lbm_timer->get_total_time();
	const double t_bc = bc_timer->get_total_time();
	double max_t_lbm = 0.0;
	double max_t_bc = 0.0;
	MPI_Allreduce(&t_lbm, &max_t_lbm, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(&t_bc, &max_t_bc, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	if (t_lbm == max_t_lbm || t_bc == max_t_bc) {
		Timer::options.print_results = true;
	}
}

void Loop_flow_heat_full_lb::initialize(const std::string& config_file_path) {
	fluid_read_write.get_sim_data(&geo, &geo_stl, config_file_path, &parallel_MPI);
	flow_field.General_data_input(config_file_path, &parallel_MPI);
	temperature_field.General_data_input(config_file_path, &parallel_MPI);
	parallel_MPI.Domain_Decomp(flow_field.Dimension, config_file_path);
	flow_field.Memory_allocation(stencil_list.get(flow_field.Dimension, flow_field.Discrete_Velocity),
	                             &parallel_MPI);
	temperature_field.Memory_allocation(stencil_list.get(temperature_field.Dimension, temperature_field.Discrete_Velocity),
	                                    &parallel_MPI);
	geo_stl.Initialize_geometry(global_parameters.Nx, global_parameters.Ny, global_parameters.Nz, &parallel_MPI);
	if (recovery_step == 0) {
		temperature_field.initialize_field(&geo, &geo_stl, (Temperature_Ini)Inline_User_Defined, &flow_field, &parallel_MPI, config_file_path);  // allocate memory and initialize variable
		flow_field.initialize_field(&geo, &geo_stl, (Flow_Ini)USERDEFINED, &temperature_field, &species_field, &parallel_MPI, config_file_path);
		flow_field.initialize_pop_eq(&parallel_MPI, &temperature_field, &species_field, config_file_path);  // allocate memory and initialize variables
		temperature_field.initialize_pop_eq(&flow_field, &parallel_MPI, config_file_path);                  // allocate memory and initialize variable
	} else {
		recover_state();
	}
	geo_stl.initialize_boundary_nodes(flow_field.is_solid, parallel_MPI);
	flow_field.initialize_BC(&geo, &geo_stl, &parallel_MPI, config_file_path);
	if (flow_field.curved_bounce_back) {
		flow_field.initialize_curved_boundaries(geo_stl, parallel_MPI);
	}
	temperature_field.initialize_BC(&geo, &geo_stl, &parallel_MPI, config_file_path);

	Timer::Options opts;
	opts.print_results = parallel_MPI.processor_id == 1;
	lbm_timer.reset(new Timer("lbm_" + std::to_string(parallel_MPI.processor_id)));
	data_exchange_timer.reset(new Timer("data_exchange_" + std::to_string(parallel_MPI.processor_id)));
	bc_timer.reset(new Timer("bc_" + std::to_string(parallel_MPI.processor_id)));
	moments_timer.reset(new Timer("moments_" + std::to_string(parallel_MPI.processor_id)));
}

void Loop_flow_heat_full_lb::step(int tm) {
	flow_field.Update_physical_time(tm, t_recovery, &parallel_MPI);
	flow_field.Diffusion_Coefficient_const_viscosity(&temperature_field, &species_field, &parallel_MPI);

	lbm_timer->start();
	flow_field.LBM_CM_MRT(tm, &temperature_field, &species_field, &geo, &parallel_MPI);
	lbm_timer->stop();
	/// ******************************
	///  DATA EXCHANGE BETWEEN CORES
	/// ******************************
	data_exchange_timer->start();
	flow_field.Data_Exchange(&parallel_MPI);
	data_exchange_timer->stop();
	/// ******************************
	///  BOUNDARY CONDITIONS
	/// ******************************
	bc_timer->start();
	flow_field.BC(tm, &temperature_field, &species_field, &parallel_MPI);
	bc_timer->stop();
	/// ******************************
	///  MOMENTS EVALUATION
	/// ******************************
	moments_timer->start();
	flow_field.momenta(tm, &parallel_MPI, &temperature_field, &species_field);
	moments_timer->stop();
	data_exchange_timer->start();
	flow_field.Data_Exchange_Macroscopic(&parallel_MPI);
	data_exchange_timer->stop();

	if (tm % 100 == 0) {
		check_L2_residual(flow_field.velocity_magnitude, flow_field.previous_velocity_magnitude, flow_field.is_solid, fluid_read_write.residual_flow, parallel_MPI, tm, "velocity_magnitude");
		
	}
}

void Loop_flow_heat_full_lb::register_outputs() {
	IO_interface& flow_out = add_io_task(t_vtk, "vtk_fluid", "fluid", Output_types::H5);

	//	flow_out.add_field(flow_field.viscosity, "Boundary");
	flow_out.add_field(flow_field.density, "Density", flow_field.rho_0, &flow_field.is_solid);
	flow_out.add_field(flow_field.pressure, "Pressure", flow_field.rho_0 * sqr(global_parameters.D_x / global_parameters.D_t), &flow_field.is_solid);

	auto adapt_velocity = [this](double v, const Tensor_base<4>::Index_vec& idx) {
#if defined Shan_Chen || defined Kupershtokh
		v = v + 0.5 * flow_field.temp_force[idx] / flow_field.density[{idx[0], idx[1], idx[2]}];
#endif
		return v;
	};
	flow_out.add_field(filter_view(flow_field.velocity, adapt_velocity),
	                   "Velocity", global_parameters.D_x / global_parameters.D_t, &flow_field.is_solid);

#if defined DEBUG_MODE
	flow_out.add_field(flow_field.alpha_entropic, "alpha_entropic");
#endif
}

void Loop_flow_heat_full_lb::register_recovery(IO_interface& io_interface) {
	flow_field.register_recovery(io_interface, parallel_MPI);
	temperature_field.register_recovery(io_interface);
}