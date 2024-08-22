#include "Loop_flow_reactive.h"
#include "../utils/Timer.hpp"

void Loop_flow_reactive::initialize(const std::string& config_file_path) {
	fluid_read_write.get_sim_data(&geo, &geo_stl, config_file_path, &parallel_MPI);            // read Geometry, general, inp-output, residual data from fluid_read_write.cpp
	flow_field.General_data_input(config_file_path, &parallel_MPI);                            // reads Flow field solver data, from Flow solver.cpp
	temperature_field.General_data_input(config_file_path, &parallel_MPI);                     // reads Temperature Field Solver from Thermal_solver.cpp
	species_field.General_data_input(config_file_path, &parallel_MPI);                         // reads Species Field Solver from species_solver.cpp
	thermo_chemistry_cantera.Initialisation(&species_field, config_file_path, &parallel_MPI);  // Initializes the thermodynamic and chemical properties using input data from ALBORZ file, specifying the Cantera libraries, retrieves the number of Species and reactions.
	// species_field.initialize_reactions(&thermo_chemistry_cantera, &parallel_MPI);              // initializes the reaction rates, reaction rate constants, and reaction stoichiometry for the Species field.
	parallel_MPI.Domain_Decomp(flow_field.Dimension, config_file_path);  // reads Parallel Processing, Flow Field Boundary Conditions from parallel.cpp
	flow_field.Memory_allocation(stencil_list.get(flow_field.Dimension, flow_field.Discrete_Velocity), &parallel_MPI);
	// average.Memory_allocation(config_file_path, parallel_MPI);
	temperature_field.Memory_allocation_FD(stencil_list.get(temperature_field.Dimension, temperature_field.Discrete_Velocity), &parallel_MPI);
	species_field.Memory_allocation_FD(stencil_list.get(species_field.Dimension, species_field.Discrete_Velocity), &parallel_MPI);
	geo_stl.Initialize_geometry(global_parameters.Nx, global_parameters.Ny, global_parameters.Nz, &parallel_MPI);
	// initializes the simulation domain processes STL files, determines the intersections of the STL geometry with lines along the z-axis in each subdomain and assigns indices to the domain tensor based on the parity of these intersections.
	flow_field.initialize_rho_LMNA(&parallel_MPI, &temperature_field, &species_field);  // checks if the Flow has Species or not, and sets the density.
	if (recovery_step == 0) {
		/*Normal case*/
		//// SPECIES INITIALISATION ////
		species_field.initialize_field_FD_TGV_reactive(species_field.mass_fraction, species_field.diffusion_coefficient, species_field.molar_mass_av, species_field.solid_species, &geo, &geo_stl, &flow_field, &temperature_field, &parallel_MPI, config_file_path,  &thermo_chemistry_cantera);
		temperature_field.initialize_field_FD_TGV_temp_reactive(temperature_field.temperature, temperature_field.c_p, temperature_field.thermal_diffusion_coefficient, temperature_field.solid_thermal_type, &geo, &geo_stl, &flow_field, &species_field, &parallel_MPI, config_file_path,  &thermo_chemistry_cantera);
		
		//species_field.initialize_field_FD(&geo, &geo_stl, (Species_Ini)TGV3Dreacting_species, &flow_field, &temperature_field, &parallel_MPI, config_file_path);  // initializes mass fraction, molar_mass_av, etc for TGV 3D cold_species
		// species_field.initialize_field_FD(&geo, &geo_stl, (Species_Ini)TGV3Dcold_species, &flow_field, &temperature_field, &parallel_MPI, config_file_path);  // initializes mass fraction, molar_mass_av, etc for TGV 3D cold_species
		// species_field.initialize_field_FD(&geo, &geo_stl, (Species_Ini)Inline_User_Defined, &flow_field, &temperature_field, &parallel_MPI, config_file_path);  // initializes mass fraction, molar_mass_av, etc TANH OR SHAPE PROFILE
		// initial_field_slice.initialize_profile_vector(species_field.mass_fraction, Ini_s, flow_field.is_solid, config_file_path, parallel_MPI);
		// initial_field_slice.initialize_profile_vector(species_field.diffusion_coefficient, Ini_D, flow_field.is_solid, config_file_path, parallel_MPI);
		//species_field.Data_Exchange_Macroscopic(&parallel_MPI);
		//// TEMPERATURE INITIALISATION ////
		// temperature_field.initialize_field_FD_TGV_temp_reactive(temperature_field.temperature, temperature_field.c_p, temperature_field.thermal_diffusion_coefficient, temperature_field.solid_thermal_type, &geo, &geo_stl, &flow_field, &species_field, &parallel_MPI, config_file_path, &thermo_chemistry_cantera);
		// temperature_field.initialize_field_FD(&geo, &geo_stl, (Temperature_Ini)TGV3Dreacting_thermal, &flow_field, &parallel_MPI, config_file_path);  // initializes cp, diffusion_coefficient, temperature for TGV 3D cold_thermal
		// temperature_field.initialize_field_FD(&geo, &geo_stl, (Temperature_Ini)TGV3Dcold_thermal, &flow_field, &parallel_MPI, config_file_path);                 // initializes cp, diffusion_coefficient, temperature for TGV 3D cold_thermal
		// temperature_field.initialize_field_FD(&geo, &geo_stl, (Temperature_Ini)Inline_User_Defined, &flow_field, &parallel_MPI, config_file_path);  // initializes cp, diffusion_coefficient, temperature   TANH OR SHAPE PROFILE
		// initial_field_slice.initialize_profile_scalar(temperature_field.temperature, Ini_T, temperature_field.solid_thermal_type, config_file_path, parallel_MPI);
		// initial_field_slice.initialize_profile_scalar(temperature_field.c_p, Ini_cp, temperature_field.solid_thermal_type, config_file_path, parallel_MPI);
		// initial_field_slice.initialize_profile_scalar(temperature_field.thermal_diffusion_coefficient, Ini_lambda, temperature_field.solid_thermal_type, config_file_path, parallel_MPI);
		//// FLOW FIELD INITIALISATION ////
		// Sampler.Save_VelPoint(config_file_path, &geo_stl, &parallel_MPI);
		flow_field.initialize_field(&geo, &geo_stl, (Flow_Ini)TAYLORGREEN3D, &temperature_field, &species_field, &parallel_MPI, config_file_path);  // initialization of various fields in the fluid solver based on provided geometries and initial conditions.
		// flow_field.initialize_field(&geo, &geo_stl, (Flow_Ini)USERDEFINED, &temperature_field, &species_field, &parallel_MPI, config_file_path);  // initialization of various fields in the fluid solver based on provided geometries and initial conditions.
		//  flow_field.initialize_profile(flow_field.velocity, ini_vel, flow_field.is_solid, config_file_path, 3, &parallel_MPI);

		/*----------TGV Case--------*/
		flow_field.initialize_pop_eq_LMNA(&parallel_MPI, config_file_path);                  // Initializes population for LMNA, computes equilibrium populations based on den, vel, theta values.
		flow_field.initialize_p_th_LMNA(&parallel_MPI, &temperature_field, &species_field);  // calculates mass_0, sets p_th and p_th_previous to a predefined value p_th_0.
	} else {
		recover_state();
		// flow_field.Recovery_read_physical_time(t_recovery, t_recovery);
		// species_field.Molar_mass_computation(&flow_field, &temperature_field, &parallel_MPI);
		// flow_field.update_density_LMNA(0, &parallel_MPI, &temperature_field, &species_field);
		// flow_field.initialize_pop_eq_LMNA(&parallel_MPI, config_file_path);
		// flow_field.initialize_p_th_LMNA(&parallel_MPI, &temperature_field, &species_field);
	}
	/// **************************
	/// BOUNDARIES INITIALIZATION
	/// **************************
	geo_stl.initialize_boundary_nodes(flow_field.is_solid, parallel_MPI);
	flow_field.initialize_BC(&geo, &geo_stl, &parallel_MPI, config_file_path);  // reads Flow Field Boundary Conditions
	if (flow_field.curved_bounce_back) {
		flow_field.initialize_curved_boundaries(geo_stl, parallel_MPI);
	}
	if (non_uniform_boundary.number_of_BC) {
		non_uniform_boundary.set_values(flow_field, geo_stl, parallel_MPI);
	}
	temperature_field.initialize_BC(&geo, &geo_stl, &parallel_MPI, config_file_path);  // reads Temperature Field Boundary Conditions data
	species_field.initialize_BC(&geo, &geo_stl, &parallel_MPI, config_file_path);      // reads Species Field Boundary Conditions data

	species_field.Molar_mass_computation(&flow_field, &temperature_field, &parallel_MPI);
	flow_field.update_density_LMNA(0, &parallel_MPI, &temperature_field, &species_field);  // initializes density values for both fluid nodes and boundary nodes for time 0.

	temperature_field.BC(0, &flow_field, &parallel_MPI);
	species_field.BC(0, &flow_field, &temperature_field, &parallel_MPI);
	flow_field.BC(0, &temperature_field, &species_field, &parallel_MPI);
}

void Loop_flow_reactive::step(int tm) {
	//	for (int outer_iter = 0; outer_iter < 1; ++outer_iter) {
	//		for (int inner_iter = 0; inner_iter < 5; ++inner_iter) {
	thermo_chemistry_cantera.Thermo_properties(&flow_field, &temperature_field, &species_field, &parallel_MPI);     // calculates Species production rates, molar masses, and specific heat capacities, incorporating Cantera libraries for thermodynamics and chemistry.
	thermo_chemistry_cantera.Transport_properties(&flow_field, &temperature_field, &species_field, &parallel_MPI);  //  calculate and assign transport properties, including Thermal conductivity, viscosity, and Species diffusion coefficients, to fluid nodes.
	thermo_chemistry_cantera.Heat_production(&flow_field, &temperature_field, &species_field, &parallel_MPI);       // dealing with enthalpy and Temperature production

	// flow_field.Sponge_zone_1(&parallel_MPI);
	// flow_field.Sponge_zone_2(0.020936, 0.025936, 0, &geo_stl, &parallel_MPI);
	// temperature_field.Sponge_zone(0.020936, 0.025936, 0, &flow_field, &geo_stl, &parallel_MPI);

	flow_field.swap_divergence_LMNA(tm, &parallel_MPI, &temperature_field, &species_field);  // swapping prevdiv with div
	flow_field.Update_physical_time(tm, t_recovery, &parallel_MPI);                          // generating output data

	temperature_field.Data_Exchange_Macroscopic(&parallel_MPI);
	temperature_field.FD_Euler(tm, &flow_field, &species_field, &parallel_MPI);  //(∂T/∂t + u · ∇T)
	temperature_field.Data_Exchange_Macroscopic(&parallel_MPI);
	temperature_field.FD_Euler_diffusion(tm, &flow_field, &species_field, &parallel_MPI);  //- ∇ · (λ∇T) = ω˙T/ρ * Cp
	temperature_field.Data_Exchange_Macroscopic(&parallel_MPI);
	temperature_field.FD_Euler_species_diffusion_enthalpy(tm, &flow_field, &species_field, &parallel_MPI);  // ρ Σ_k=1^N_sp (cpk * Yk * Vk · ∇T)
	temperature_field.Data_Exchange_Macroscopic(&parallel_MPI);

	species_field.FD_Euler(tm, &flow_field, &temperature_field, &parallel_MPI);  // ∂Yk/∂t + u·∇Yk
	species_field.FD_Euler_Correction_Velocity(tm, &flow_field, &temperature_field, &parallel_MPI);
	species_field.FD_HC_Euler(tm, &flow_field, &temperature_field, &parallel_MPI, false);  // -∇·(Dk∇Yk) = ω˙k/ρ
	// species_field.FD_Fick_Euler(tm, &flow_field, &temperature_field, &parallel_MPI, true);
	species_field.Data_Exchange_Macroscopic(&parallel_MPI);
	temperature_field.Data_Exchange_Macroscopic(&parallel_MPI);
	//		}
	// species_field.Diffusion_Coefficient_computation(&flow_field, &temperature_field, &parallel_MPI);

	// flow_field.LBM_CM_MRT_LMNA(tm, &temperature_field, &species_field, &parallel_MPI);
	// flow_field.LBM_SRT_LMNA(tm, &temperature_field, &species_field, &geo, &parallel_MPI);
	flow_field.LBM_CUMULANT_LMNA(tm, &temperature_field, &species_field, &geo, &parallel_MPI);  // to check
	flow_field.Data_Exchange(&parallel_MPI);

	temperature_field.BC(tm, &flow_field, &parallel_MPI);
	species_field.BC(tm, &flow_field, &temperature_field, &parallel_MPI);
	flow_field.BC(tm, &temperature_field, &species_field, &parallel_MPI);

	flow_field.update_density_LMNA(tm, &parallel_MPI, &temperature_field, &species_field);  // Density is based on : p_th, M_av,R_GAS,T,T_0,rho_0.
	flow_field.Data_Exchange_Macroscopic(&parallel_MPI);
	flow_field.momenta_LMNA(tm, &parallel_MPI, &temperature_field, &species_field);  // updates the velocity field, pressure field, and related quantities in the context of a low Mach number approximation (LMNA) model.
	flow_field.Data_Exchange_Macroscopic(&parallel_MPI);
	// species_field.Check_Mass_Fraction_Conservation(&parallel_MPI);

	// temperature_field.calculateThermalDiffusivity(tm, &temperature_field, &flow_field, &parallel_MPI);
	// species_field.calculateMassDiffusivity(tm, &temperature_field, &flow_field, &parallel_MPI);
	// species_field.calculateLewisNumber(tm, &flow_field, &temperature_field, &species_field, &parallel_MPI);  // Call the calculateLewisNumber function with the correct arguments.
	//	}
	if (tm % 100 == 0) {
		check_L2_residual(flow_field.velocity_magnitude, flow_field.previous_velocity_magnitude, flow_field.is_solid, fluid_read_write.residual_flow, parallel_MPI, tm, "velocity");  // velocity L2 norm
		// Check_Mass_Fraction_Conservation(tm, &species_field, &parallel_MPI);
		//  species_field.Consumption_rate_monitor_each_species(&flow_field, &parallel_MPI, tm, 1);
		//  temperature_field.temp_monitor(tm, &flow_field, parallel_MPI);
		//   species_field.Consumption_rate_monitor_each_species(&flow_field, &parallel_MPI, tm, 1);
		//   temperature_field.temp_monitor(tm, &flow_field, parallel_MPI);
		//    species_field.mass_conservation_report(tm, &flow_field, &temperature_field, &parallel_MPI);                                                                                       // computes and monitors the consumption rates of Species in a simulation
		//   CFL_monitor(&flow_field, &parallel_MPI, tm);
		//   Viscosity_monitor(&flow_field, &temperature_field, &species_field, &parallel_MPI, tm);
		//   Sampler.Out_VelPoint(flow_field.density, &geo_stl, &parallel_MPI, tm);
		fluid_read_write.AverageKineticEnergy(&flow_field, &parallel_MPI, tm);
		fluid_read_write.AverageEnstrophy(&flow_field, &parallel_MPI, tm);
	}
}

void Loop_flow_reactive::register_outputs() {
	IO_interface& flow_out = add_io_task(t_vtk, "vtk_fluid", "fluid", Output_types::VTK);
	flow_out.add_field(flow_field.density, "Density", flow_field.rho_0, &flow_field.is_solid);
	flow_out.add_field(flow_field.pressure, "Pressure", flow_field.rho_0 * sqr(global_parameters.D_x / global_parameters.D_t), &flow_field.is_solid);
	auto adapt_velocity = [](double v, const Tensor_base<4>::Index_vec& idx) {
		return v;
	};
	flow_out.add_field(filter_view(flow_field.velocity, adapt_velocity), "Velocity", global_parameters.D_x / global_parameters.D_t, &flow_field.is_solid);
	flow_out.add_field(temperature_field.temperature, "Temperature", temperature_field.T_0, &temperature_field.solid_thermal_type);
	// flow_out.add_field(temperature_field.Production, "wdot", temperature_field.T_0, &temperature_field.solid_thermal_type);
	flow_out.add_field(temperature_field.c_p, "cp", 1.0, &temperature_field.solid_thermal_type);
	flow_out.add_field(temperature_field.thermal_diffusion_coefficient, "thermal_diffusion_coefficient", 1.0 / (global_parameters.D_t / sqr(global_parameters.D_x)), &temperature_field.solid_thermal_type);
	flow_out.add_field(temperature_field.Production, "HRR", 1.0 / global_parameters.D_t, &temperature_field.solid_thermal_type);

	for (Index i = 0; i < species_field.Nb_spec; ++i) {
		flow_out.add_field(layer_view<3>(species_field.mass_fraction, i),
		                   "Mass_fraction_" + species_field.species_name_RG[i], 1.0, &flow_field.is_solid);
		flow_out.add_field(layer_view<3>(species_field.Production, i),
		                   "Prod_species" + species_field.species_name_RG[i], 1.0 / (flow_field.rho_0 * flow_field.rho_0), &flow_field.is_solid);
		flow_out.add_field(layer_view<3>(species_field.diffusion_coefficient, i),
		                   "Diffusion_coefficient_" + species_field.species_name_RG[i], 1.0 / (global_parameters.D_t / sqr(global_parameters.D_x)), &flow_field.is_solid);
	}

	// here Index_vec3 is for defining the desired plane of the slice (based on grid points Nx, Ny, Nz). Eg: {global_parameters.Nx / 2, 0, 0} is the middle plane of the x-axis.
	// the second Index_vec3 is for defining the thickness of the slice (in grid points). Eg: {2, 0, 0} is the thickness of the slice in the x-axis. It does not work with just 1 grid point.
	// here t_vtk is the time step for the output, "slices" is the folder name, "pl-x" is the file name, Output_types::H5 is the output type,
	// 0 is the processor id, Index_vec3{global_parameters.Nx / 2, 0, 0} is the starting point of the slice, Index_vec3{2, 0, 0} is the thickness of the slice.
	// The add_field function is used to add the fields to the output file. The first argument is the field to be added, the second argument is the name of the field,
	// the third argument is the scaling factor, and the fourth argument is the condition for the field to be added.

	/*IO_interface& slice_out = add_io_task(t_vtk, "slices", "pl-x", Output_types::H5, 0, Index_vec3{global_parameters.Nx / 2, 0, 0}, Index_vec3{2, 0, 0});
	slice_out.add_field(flow_field.velocity, "Velocity", global_parameters.D_x / global_parameters.D_t, &flow_field.is_solid);
	slice_out.add_field(flow_field.is_solid, "is_solid", -1);

	IO_interface& slice_out2 = add_io_task(t_vtk, "slices", "pl-y", Output_types::H5, 0, Index_vec3{0, global_parameters.Ny / 2 - 1, 0}, Index_vec3{0, 2, 0});
	slice_out2.add_field(flow_field.velocity, "Velocity", global_parameters.D_x / global_parameters.D_t, &flow_field.is_solid);
	slice_out2.add_field(flow_field.is_solid, "is_solid", -1);

	IO_interface& slice_out3 = add_io_task(t_vtk, "slices", "pl-z", Output_types::H5, 0, Index_vec3{0, 0, global_parameters.Nz / 2 - 1}, Index_vec3{0, 0, 2});
	slice_out3.add_field(flow_field.velocity, "Velocity", global_parameters.D_x / global_parameters.D_t, &flow_field.is_solid);
	slice_out3.add_field(flow_field.is_solid, "is_solid", -1);
*/
	/*IO_interface& slice_out = add_io_task(1000, "slices", "pl-x", Output_types::H5, 0, Index_vec3{0, 0, global_parameters.Nz / 2}, Index_vec3{0, 0, 2});
	slice_out.add_field(flow_field.velocity, "Velocity_slice", global_parameters.D_x / global_parameters.D_t, &flow_field.is_solid);
	slice_out.add_field(temperature_field.temperature, "Temperature_slice", temperature_field.T_0, &temperature_field.solid_thermal_type);
	slice_out.add_field(flow_field.density, "Density_slice", flow_field.rho_0, &flow_field.is_solid);
	slice_out.add_field(flow_field.pressure, "Pressure_slice", flow_field.rho_0 * sqr(global_parameters.D_x / global_parameters.D_t), &flow_field.is_solid);
	slice_out.add_field(temperature_field.Production, "HRR_slice", 1.0 / global_parameters.D_t, &temperature_field.solid_thermal_type);
	for (Index i = 0; i < species_field.Nb_spec; ++i) {
		slice_out.add_field(layer_view<3>(species_field.mass_fraction, i),
		                    "Mass_fraction_" + species_field.species_name_RG[i], 1.0, &flow_field.is_solid);
	}
	slice_out.add_field(flow_field.is_solid, "is_solid", -1);
	*/

#if defined DEBUG_MODE
	flow_out.add_field(flow_field.alpha_entropic, "alpha_entropic");
#endif

	IO_interface& cantera_out = add_io_task(std::numeric_limits<int>::max(), "Cantera", "", Output_types::NO);
	cantera_out.add_custom_write([this](const std::string& base_path, int tm) {
		thermo_chemistry_cantera.write_report(base_path, species_field, parallel_MPI);
	});

	/*
	/// ******************************
	    /// average field
	    /// ******************************
	    IO_interface& average_out = add_io_task(t_vtk, "vtk_fluid", "fluid_average", Output_types::VTK, average.start_sampling);
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
*/
}

void Loop_flow_reactive::register_recovery(IO_interface& io_interface) {
	flow_field.register_recovery(io_interface, parallel_MPI);
	temperature_field.register_recovery(io_interface);
	//	average.register_recovery(io_interface);
	species_field.register_recovery(io_interface);
}