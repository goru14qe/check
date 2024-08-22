#include "Loop_flow_particle.h"

void Loop_flow_particle::initialize(const std::string& config_file_path) {
	/// **************************
	/// INPUT FILE READING
	/// **************************
	fluid_read_write.get_sim_data(&geo, &geo_stl, config_file_path, &parallel_MPI);
	flow_field.General_data_input(config_file_path, &parallel_MPI);
	temperature_field.General_data_input(config_file_path, &parallel_MPI);
	particle_sim.General_data_input(&temperature_field, config_file_path, &parallel_MPI);
	parallel_MPI.Domain_Decomp(flow_field.Dimension, config_file_path);
	/// **************************
	/// MEMORY ALLOCATION FLOW
	/// **************************
	flow_field.Memory_allocation(stencil_list.get(flow_field.Dimension, flow_field.Discrete_Velocity),
	                             &parallel_MPI);
	average.Memory_allocation(config_file_path, parallel_MPI);
	/// **************************
	/// MEMORY ALLOCATION ENERGY
	/// **************************
	temperature_field.Memory_allocation(stencil_list.get(temperature_field.Dimension, temperature_field.Discrete_Velocity),
	                                    &parallel_MPI);
	/// **************************
	/// GEOMETRY INITIALIZATION
	/// **************************
	geo_stl.Initialize_geometry(global_parameters.Nx, global_parameters.Ny, global_parameters.Nz, &parallel_MPI);
	/// **************************
	/// PARTICLE INITIALIZATION
	/// **************************
	particle_sim.Particle_initialize(config_file_path, &geo_stl, &parallel_MPI, recovery_step);
	/// **************************
	/// FIELD INITIALIZATION
	/// **************************
	if (recovery_step == 0) {
		temperature_field.initialize_field(&geo, &geo_stl, (Temperature_Ini)Inline_User_Defined, &flow_field, &parallel_MPI, config_file_path);  // allocate memory and initialize variable
		flow_field.initialize_field(&geo, &geo_stl, (Flow_Ini)USERDEFINED, &temperature_field, &species_field, &parallel_MPI, config_file_path);
		flow_field.initialize_pop_eq(&parallel_MPI, &temperature_field, &species_field, config_file_path);  // allocate memory and initialize variables
		temperature_field.initialize_pop_eq(&flow_field, &parallel_MPI, config_file_path);                               // allocate memory and initialize variable
	} else {
		recover_state();
	}
	/// **************************
	/// BOUNDARIES INITIALIZATION
	/// **************************
	geo_stl.initialize_boundary_nodes(flow_field.is_solid, parallel_MPI);
	flow_field.initialize_BC(&geo, &geo_stl, &parallel_MPI, config_file_path);
	if (flow_field.curved_bounce_back) {
		flow_field.initialize_curved_boundaries(geo_stl, parallel_MPI);
	}
	temperature_field.initialize_BC(&geo, &geo_stl, &parallel_MPI, config_file_path);
}

void Loop_flow_particle::step(int tm) {
	flow_field.Update_physical_time(tm, t_recovery, &parallel_MPI);
	flow_field.Diffusion_Coefficient_const_viscosity(&temperature_field, &species_field, &parallel_MPI);
	flow_field.LBM_CM_MRT(tm, &temperature_field, &species_field, &geo, &parallel_MPI);
	/// ******************************
	///  DATA EXCHANGE BETWEEN CORES
	/// ******************************
	flow_field.Data_Exchange(&parallel_MPI);
	/// ******************************
	///  TIME-DEPENDENT BOUNDARY CONDITIONS
	/// ******************************
#if defined TIME_DEPENDENT_BC
	BC_t.set_values(tm, &flow_field, &parallel_MPI);
#endif
	/// ******************************
	///  BOUNDARY CONDITIONS
	/// ******************************
	flow_field.BC(tm, &temperature_field, &species_field, &parallel_MPI);
	/// ******************************
	///  MOMENTS EVALUATION
	/// ******************************
	flow_field.momenta(tm, &parallel_MPI, &temperature_field, &species_field);
	flow_field.Data_Exchange_Macroscopic(&parallel_MPI);

	particle_sim.Data_Slave_to_Master_rcv(particle_sim.particle, tm, &parallel_MPI);
	particle_sim.Tasks_Before_Newton(particle_sim.particle, &temperature_field, &flow_field, &parallel_MPI, tm);
	particle_sim.update_location(particle_sim.particle, tm, particle_sim.Particle_IN, &parallel_MPI);
	particle_sim.Tasks_After_Newton(particle_sim.particle, tm, &parallel_MPI);

	if (tm >= average.start_sampling) {
		average.add_data(parallel_MPI, tm);
	}
}

void Loop_flow_particle::register_outputs() {
	/// ******************************
	/// Flow fields
	/// ******************************
	IO_interface& flow_out = add_io_task(t_vtk, "vtk_fluid", "fluid");
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

	/// ******************************
	/// average field
	/// ******************************
	IO_interface& average_out = add_io_task(t_vtk, "vtk_fluid", "fluid_average", Output_types::AUTO, average.start_sampling);
	average.add_field(temperature_field.temperature, "av_T", average_out, 
		temperature_field.T_0, &temperature_field.solid_thermal_type);
	average.add_field(flow_field.velocity, "av_u", average_out, 
		global_parameters.D_x / global_parameters.D_t, &flow_field.is_solid);

	auto compute_reynolds_stress = [this](Tensor<double, 4>& reynolds_stress, Index X, Index Y, Index Z) {
		reynolds_stress[{X, Y, Z, 0}] += sqr(flow_field.velocity[{X, Y, Z, 0}]);
		reynolds_stress[{X, Y, Z, 1}] += sqr(flow_field.velocity[{X, Y, Z, 1}]);
		reynolds_stress[{X, Y, Z, 2}] += sqr(flow_field.velocity[{X, Y, Z, 2}]);
		reynolds_stress[{X, Y, Z, 3}] += flow_field.velocity[{X, Y, Z, 0}] * flow_field.velocity[{X, Y, Z, 1}];
		reynolds_stress[{X, Y, Z, 4}] += flow_field.velocity[{X, Y, Z, 1}] * flow_field.velocity[{X, Y, Z, 2}];
		reynolds_stress[{X, Y, Z, 5}] += flow_field.velocity[{X, Y, Z, 0}] * flow_field.velocity[{X, Y, Z, 2}];
	};
	auto re_sizes = flow_field.velocity.sizes();
	re_sizes[3] = 6;
	average.add_field(compute_reynolds_stress, re_sizes, "Re_tensor", average_out, 
		sqr(global_parameters.D_x / global_parameters.D_t), &flow_field.is_solid);

	/// ******************************
	// particle
	IO_interface& particle_out = add_io_task(t_data, "Data_particle", "data_particle", Output_types::NO);
	particle_out.add_custom_write([this](const std::string& base_path, int tm) {
		for (int np = 0; np < particle_sim.num_particles; ++np) {
			if (particle_sim.Particle_IN[np] == TRUE) {
				particle_sim.write_particle_data(base_path, tm, particle_sim.particle[np], np);
			}
		}
	});

	IO_interface& particle_out_vtk = add_io_task(t_vtk, "vtk_particle", "particle", Output_types::NO);
	particle_out_vtk.add_custom_write([this](const std::string& base_path, int tm) {
		for (int np = 0; np < particle_sim.num_particles; ++np) {
			if (particle_sim.Particle_IN[np] == TRUE) {
				particle_sim.write_particle_vtk(base_path, tm, particle_sim.particle[np], np);
			}
		}
	});
}

void Loop_flow_particle::register_recovery(IO_interface& io_interface) {
	flow_field.register_recovery(io_interface, parallel_MPI);
	temperature_field.register_recovery(io_interface);
	average.register_recovery(io_interface);
	particle_sim.register_recovery(io_interface, parallel_MPI);
}