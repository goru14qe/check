#include "Loop_flow_crystal.h"
#include "../utils/Timer.hpp"

void Loop_flow_crystal_LB::initialize(const std::string& config_file_path) {
	/// **************************
	/// INPUT FILE READING
	/// **************************
	fluid_read_write.get_sim_data(&geo, &geo_stl, config_file_path, &parallel_MPI);
	flow_field.General_data_input(config_file_path, &parallel_MPI);
	temperature_field.General_data_input(config_file_path, &parallel_MPI);
	phase_field.General_data_input(config_file_path, &parallel_MPI);
	species_field.General_data_input(config_file_path, &parallel_MPI);
	parallel_MPI.Domain_Decomp(flow_field.Dimension, config_file_path);
	/// **************************
	/// MEMORY ALLOCATION FLOW
	/// **************************
	flow_field.Memory_allocation(stencil_list.get(flow_field.Dimension, flow_field.Discrete_Velocity),
	                             &parallel_MPI);
	/// **************************
	/// MEMORY ALLOCATION ENERGY
	/// **************************
	temperature_field.Memory_allocation(stencil_list.get(temperature_field.Dimension, temperature_field.Discrete_Velocity),
	                                    &parallel_MPI);
	/// **************************
	/// MEMORY ALLOCATION PHASE
	/// **************************
	phase_field.Memory_allocation(stencil_list.get(phase_field.Dimension, phase_field.Discrete_Velocity),
	                              &parallel_MPI);
	/// **************************
	/// MEMORY ALLOCATION PHASE
	/// **************************
	species_field.Memory_allocation(stencil_list.get(species_field.Dimension, species_field.Discrete_Velocity),
	                                &parallel_MPI);
	/// **************************
	/// GEOMETRY INITIALIZATION
	/// **************************
	geo_stl.Initialize_geometry(global_parameters.Nx, global_parameters.Ny, global_parameters.Nz, &parallel_MPI);
	/// **************************
	/// FIELD INITIALIZATION
	/// **************************
	if (recovery_step == 0) {
		temperature_field.initialize_field(&geo, &geo_stl, (Temperature_Ini)Inline_User_Defined, &flow_field, &parallel_MPI, config_file_path);
		flow_field.initialize_field(&geo, &geo_stl, (Flow_Ini)USERDEFINED, &temperature_field, &species_field, &parallel_MPI, config_file_path);
		flow_field.Diffusion_Coefficient_const_viscosity(&temperature_field, &species_field, &parallel_MPI);
		phase_field.initialize_p(&geo, &geo_stl, (Phase_Ini)Inline_User_Defined_p, &parallel_MPI, config_file_path);
		species_field.initialize_field(&geo, &geo_stl, (Species_Ini)Inline_User_Defined, &flow_field, &parallel_MPI, config_file_path);
		temperature_field.initialize_pop_eq_crystal(&flow_field, &parallel_MPI, config_file_path);
		flow_field.initialize_pop_eq(&parallel_MPI, &temperature_field, &species_field, config_file_path);
		phase_field.initialize_pop_eq_p(&parallel_MPI, config_file_path);
		species_field.initialize_pop_eq_snow(&flow_field, &phase_field, &parallel_MPI, config_file_path);
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
	phase_field.initialize_BC_p(&geo, &geo_stl, &parallel_MPI, config_file_path);
	species_field.initialize_BC(&geo, &geo_stl, &parallel_MPI, config_file_path);
}

void Loop_flow_crystal_LB::step(int tm) {
	flow_field.Update_physical_time(tm, t_recovery, &parallel_MPI);
	flow_field.Diffusion_Coefficient_const_viscosity(&temperature_field, &species_field, &parallel_MPI);

	/// ******************************
	///  COMPUTE FORCES FROM PHASE-FIELD ON FLOW
	/// ******************************
	phase_field.Force_on_Fluid(tm, &flow_field, &parallel_MPI);
	phase_field.Reset_Velocity(tm, &flow_field, &parallel_MPI);

	// phase_field.Crystal_DW_T(tm, &temperature_field,&species_field, &parallel_MPI);
	// phase_field.Crystal_DW_S(tm, &temperature_field,&species_field, &parallel_MPI);
	phase_field.Crystal_DW(tm, &temperature_field, &species_field, &parallel_MPI);
	// phase_field.Diffusion_Coefficient_computation_T(tm, &temperature_field, &parallel_MPI);
	phase_field.Crystal_Heat(tm, &temperature_field, &parallel_MPI);
	// phase_field.Diffusion_Coefficient_computation_S(tm, &species_field, &parallel_MPI);
	phase_field.Crystal_Species(tm, &species_field, &parallel_MPI);

	flow_field.LBM_CM_MRT(tm, &temperature_field, &species_field, &geo, &parallel_MPI);
	phase_field.LBMNONCONS_p(tm, (Crystal_Anisotropy)Hexahedral_2D, &parallel_MPI);  /// Hexahedral_2D/Tetrahedral_2D
	species_field.LBM_SRT_snow(tm, &flow_field, &temperature_field, &phase_field, &parallel_MPI);
	temperature_field.LBM_SRT_crystal(tm, &flow_field, &parallel_MPI);
	/// ******************************
	///  DATA EXCHANGE BETWEEN CORES
	/// ******************************
	temperature_field.Data_Exchange(&parallel_MPI);
	flow_field.Data_Exchange(&parallel_MPI);
	phase_field.Data_Exchange(&parallel_MPI);
	species_field.Data_Exchange(&parallel_MPI);
	/// ******************************
	///  BOUNDARY CONDITIONS
	/// ******************************
	temperature_field.BC(tm, &flow_field, &parallel_MPI);
	flow_field.BC(tm, &temperature_field, &species_field, &parallel_MPI);
	phase_field.BC_p(tm, &flow_field, &parallel_MPI);
	species_field.BC(tm, &flow_field, &temperature_field, &parallel_MPI);
	/// ******************************
	///  MOMENTS EVALUATION
	/// ******************************
	temperature_field.momenta_crystal(&parallel_MPI);
	flow_field.momenta(tm, &parallel_MPI, &temperature_field, &species_field);
	phase_field.momenta_p(&parallel_MPI);
	species_field.momenta(&flow_field, &parallel_MPI);

	/// ******************************
	///  MACRO DATA EXCHANGE BETWEEN CORES
	/// ******************************
	temperature_field.Data_Exchange_Macroscopic(&parallel_MPI);
	flow_field.Data_Exchange_Macroscopic(&parallel_MPI);
	phase_field.Data_Exchange_Macroscopic(&parallel_MPI);
	species_field.Data_Exchange_Macroscopic(&parallel_MPI);
}

void Loop_flow_crystal_LB::register_outputs() {
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

	IO_interface& temperature_out = add_io_task(t_vtk, "vtk_fluid", "temperature");
	temperature_out.add_field(temperature_field.temperature, "Temperature", temperature_field.T_0, &temperature_field.solid_thermal_type);
	temperature_out.add_field(temperature_field.Production, "Production", 1.0 / global_parameters.D_t, &temperature_field.solid_thermal_type);

	IO_interface& species_out = add_io_task(t_vtk, "vtk_fluid", "Species");
	species_out.add_field(species_field.mass_fraction, "Mass_fraction");
	species_out.add_field(species_field.Production, "Production", 1.0 / global_parameters.D_t);

	IO_interface& phase_out = add_io_task(t_vtk, "vtk_fluid", "phase");
	phase_out.add_field(phase_field.phase, "Phase");
}

void Loop_flow_crystal_LB::register_recovery(IO_interface& io_interface) {
	io_interface.add_custom_write([this](const std::string& base_path, int tm){
		if(parallel_MPI.is_master()){
			std::cerr << "[Error] Trying to create a recovery point but recovery is not implemented for this simulation type.\n";
		}
	});
	io_interface.add_custom_read([this](const std::string&, int) {
		if(parallel_MPI.is_master()){
			std::cerr << "[Error] Trying to resume a previous simulation but recovery is not implemented for this simulation type.\n";
		}
	});
	// The implementation is unfinished and the simulation will diverge
	// if resumed from a recovery point.
	// In particular, phase_field and species_field are only tested not to crash so far.
//	flow_field.register_recovery(io_interface, parallel_MPI);
//	temperature_field.register_recovery(io_interface);
//	phase_field.register_recovery(io_interface);
//	species_field.register_recovery(io_interface);
}

// *************************************************** //
void Loop_flow_crystal_FD::initialize(const std::string& config_file_path) {
	/// **************************
	/// INPUT FILE READING
	/// **************************
	fluid_read_write.get_sim_data(&geo, &geo_stl, config_file_path, &parallel_MPI);
	flow_field.General_data_input(config_file_path, &parallel_MPI);
	temperature_field.General_data_input(config_file_path, &parallel_MPI);
	phase_field.General_data_input(config_file_path, &parallel_MPI);
	species_field.General_data_input(config_file_path, &parallel_MPI);
	parallel_MPI.Domain_Decomp(flow_field.Dimension, config_file_path);
	/// **************************
	/// MEMORY ALLOCATION FLOW
	/// **************************
	flow_field.Memory_allocation(stencil_list.get(flow_field.Dimension, flow_field.Discrete_Velocity),
	                             &parallel_MPI);
	/// **************************
	/// MEMORY ALLOCATION ENERGY
	/// **************************
	temperature_field.Memory_allocation_FD(stencil_list.get(temperature_field.Dimension, temperature_field.Discrete_Velocity),
	                                       &parallel_MPI);
	/// **************************
	/// MEMORY ALLOCATION PHASE
	/// **************************
	phase_field.Memory_allocation(stencil_list.get(phase_field.Dimension, phase_field.Discrete_Velocity),
	                              &parallel_MPI);
	/// **************************
	/// MEMORY ALLOCATION PHASE
	/// **************************
	species_field.Memory_allocation_FD(stencil_list.get(species_field.Dimension, species_field.Discrete_Velocity),
	                                   &parallel_MPI);
	/// **************************
	/// GEOMETRY INITIALIZATION
	/// **************************
	geo_stl.Initialize_geometry(global_parameters.Nx, global_parameters.Ny, global_parameters.Nz, &parallel_MPI);
	/// **************************
	/// FIELD INITIALIZATION
	/// **************************
	if (recovery_step == 0) {
		temperature_field.initialize_field_FD(&geo, &geo_stl, (Temperature_Ini)Inline_User_Defined, &flow_field, &parallel_MPI, config_file_path);
		flow_field.initialize_field(&geo, &geo_stl, (Flow_Ini)USERDEFINED, &temperature_field, &species_field, &parallel_MPI, config_file_path);
		flow_field.Diffusion_Coefficient_const_viscosity(&temperature_field, &species_field, &parallel_MPI);
		phase_field.initialize_p(&geo, &geo_stl, (Phase_Ini)Inline_User_Defined_p, &parallel_MPI, config_file_path);
		species_field.initialize_field_FD(&geo, &geo_stl, (Species_Ini)Inline_User_Defined, &flow_field, &temperature_field, &parallel_MPI, config_file_path);
		flow_field.initialize_pop_eq(&parallel_MPI, &temperature_field, &species_field, config_file_path);
		phase_field.initialize_pop_eq_p(&parallel_MPI, config_file_path);
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
	phase_field.initialize_BC_p(&geo, &geo_stl, &parallel_MPI, config_file_path);
	species_field.initialize_BC(&geo, &geo_stl, &parallel_MPI, config_file_path);
}

void Loop_flow_crystal_FD::step(int tm) {
	flow_field.Update_physical_time(tm, t_recovery, &parallel_MPI);
	flow_field.Diffusion_Coefficient_const_viscosity(&temperature_field, &species_field, &parallel_MPI);

	/// ******************************
	///  COMPUTE FORCES FROM PHASE-FIELD ON FLOW
	/// ******************************
	phase_field.Force_on_Fluid(tm, &flow_field, &parallel_MPI);
	phase_field.Reset_Velocity(tm, &flow_field, &parallel_MPI);

	phase_field.Crystal_DW_S(tm, &temperature_field, &species_field, &parallel_MPI);
	phase_field.Crystal_Heat(tm, &temperature_field, &parallel_MPI);
	phase_field.Crystal_Species(tm, &species_field, &parallel_MPI);

	flow_field.LBM_CM_MRT(tm, &temperature_field, &species_field, &geo, &parallel_MPI);
	phase_field.LBMNONCONS_p(tm, (Crystal_Anisotropy)Hexahedral_2D, &parallel_MPI);  /// Hexahedral_2D/Tetrahedral_2D

	phase_field.Diffusion_Coefficient_computation_T(tm, &temperature_field, &parallel_MPI);
	temperature_field.FD_Euler(tm, &flow_field, &species_field, &parallel_MPI);
	temperature_field.FD_Euler_diffusion(tm, &flow_field, &species_field, &parallel_MPI);

	phase_field.Diffusion_Coefficient_computation_S(tm, &species_field, &parallel_MPI);
	species_field.FD_Euler(tm, &flow_field, &temperature_field, &parallel_MPI);
	species_field.FD_Euler_diffusion(tm, &flow_field, &temperature_field, &parallel_MPI, 1);
	/// ******************************
	///  DATA EXCHANGE BETWEEN CORES
	/// ******************************
	flow_field.Data_Exchange(&parallel_MPI);
	phase_field.Data_Exchange(&parallel_MPI);
	/// ******************************
	///  BOUNDARY CONDITIONS
	/// ******************************
	temperature_field.BC(tm, &flow_field, &parallel_MPI);
	flow_field.BC(tm, &temperature_field, &species_field, &parallel_MPI);
	phase_field.BC_p(tm, &flow_field, &parallel_MPI);
	species_field.BC(tm, &flow_field, &temperature_field, &parallel_MPI);
	/// ******************************
	///  MOMENTS EVALUATION
	/// ******************************
	flow_field.momenta(tm, &parallel_MPI, &temperature_field, &species_field);
	phase_field.momenta_p(&parallel_MPI);

	/// ******************************
	///  MACRO DATA EXCHANGE BETWEEN CORES
	/// ******************************
	temperature_field.Data_Exchange_Macroscopic(&parallel_MPI);
	flow_field.Data_Exchange_Macroscopic(&parallel_MPI);
	phase_field.Data_Exchange_Macroscopic(&parallel_MPI);
	species_field.Data_Exchange_Macroscopic(&parallel_MPI);
}