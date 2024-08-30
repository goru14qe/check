// #include "stdafx.h"
#include <sstream>   // string streams
#include <iostream>  // for the use of 'cout'
#include <cstdlib>
#include <cmath>
#include <string.h>
#include <set>
#include <unordered_set>
#include <filesystem>
#include "mpi.h"
#include "Fluid_read_write.h"
#include "Geometry.h"
#include "Flow_solver.h"
#include "Thermal_solver.h"
#include "Species_solver.h"
#include "ALBORZ_Macros.h"
#include "ALBORZ_SETTINGS.h"
#include "ALBORZ_GlobalVariables.h"
#include "Particle_sim.h"
#include "io/IO_interface.h"
#include "utils/Config_utils.h"
#ifdef __linux__
#include <sys/stat.h>  //for mkdir
#endif
#ifdef __APPLE__
#include <sys/stat.h>  //for mkdir
#endif
#if defined _WIN64 || _WIN32
#include "direct.h"  //for mkdir
#endif
using namespace std;

// Macro DISPATCH_BY_STENCIL is used to dispatch the function to the correct dimension and stencil size.
//  The __VA_ARGS__ represents any additional arguments passed to the macro.
#define DISPATCH_BY_STENCIL(function, ...) \
	do {                                   \
		if (Dimension == 2)                \
			function<2, 9>(__VA_ARGS__);   \
		else if (Dimension == 3)           \
			function<3, 27>(__VA_ARGS__);  \
	} while (false)

Flow_solver::Flow_solver() {
}
/// ***************************************************** ///
/// WRITE AVERAGE VALUES TO FILE TO CHECK DIVERGENCE      ///
/// ***************************************************** ///
FlowResults Flow_solver::average_values(const Parallel_MPI& MPI_parallel) const {
	double pres = 0.0;
	double den = 0.0;
	double vel = 0.0;
	double max_pres = -std::numeric_limits<double>::infinity();
	double max_den = -std::numeric_limits<double>::infinity();
	double max_vel = -std::numeric_limits<double>::infinity();
	size_t nodes = 0;
	if (!MPI_parallel.is_master()) {
		for (int X = MPI_parallel.start_XYZ2[0]; X <= MPI_parallel.end_XYZ2[0]; ++X) {
			for (int Y = MPI_parallel.start_XYZ2[1]; Y <= MPI_parallel.end_XYZ2[1]; ++Y) {
				for (int Z = MPI_parallel.start_XYZ2[2]; Z <= MPI_parallel.end_XYZ2[2]; ++Z) {
					if (is_solid[{X, Y, Z}] == FALSE) {
						pres += pressure[{X, Y, Z}];
						den += density[{X, Y, Z}];
						vel += velocity_magnitude[{X, Y, Z}];
						max_pres = std::max(max_pres, pressure[{X, Y, Z}]);
						max_den = std::max(max_den, density[{X, Y, Z}]);
						max_vel = std::max(max_vel, velocity[0]);
						++nodes;
					}
				}
			}
		}
	}
	double pres_total = pres;
	double den_total = den;
	double vel_total = vel;
	double max_pres_total = max_pres;
	double max_den_total = max_den;
	double max_vel_total = max_vel;
	size_t nodes_total = nodes;

	MPI_Reduce(&pres, &pres_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&den, &den_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&vel, &vel_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&max_pres, &max_pres_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&max_den, &max_den_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&max_vel, &max_vel_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&nodes, &nodes_total, 1, to_MPI_type<size_t>::value(), MPI_SUM, 0, MPI_COMM_WORLD);

	double avg_pres = pres_total / nodes_total;
	double avg_den = den_total / nodes_total;
	double avg_vel = vel_total / nodes_total;

	FlowResults results;
	results.avg_pres = avg_pres;
	results.avg_den = avg_den;
	results.avg_vel = avg_vel;
	results.max_pres = max_pres_total * rho_0;
	results.max_den = max_den_total * rho_0;
	results.max_vel = max_vel_total * (global_parameters.D_t / global_parameters.D_x);

	return results;
}
/// ***************************************************** ///
/// WRITE PHYSICAL TIME TO RECOVERY FILE                  ///
/// ***************************************************** ///
void Flow_solver::Update_physical_time(unsigned int time_step, unsigned int t_recovery, Parallel_MPI* MPI_parallel) {
	physical_time += global_parameters.D_t;
	//const FlowResults average_values = this->average_values(*MPI_parallel);
	if (MPI_parallel->processor_id == MASTER) {
		stringstream output_filename;
		output_filename << "Alborz_Results/debug/time_monitor.dat";
		ofstream output_file;
		/// Open file
		output_file.open(output_filename.str().c_str(), fstream::app);
		// Write header if the file is being created for the first time
		if (output_file.tellp() == 0) {
			//output_file << "Time_Step\tPhysical_Time\tPressure\tDensity\tVelocity" << std::endl;
			output_file << "Time_Step\tPhysical_Time" << std::endl;
		}
		/// Write data
		output_file << time_step << "\t";  // time step
		// output_file << setprecision(30) << physical_time << "\t" << average_values.avg_pres << "\t" << average_values.avg_den << "\t" << average_values.avg_vel << endl;
		output_file << setprecision(20) << physical_time << endl;
		/// Close file
		output_file.close();

		std::stringstream str_line1;
		ofstream output_recovery;
		if (time_step % t_recovery == 0) {
			str_line1 << "Alborz_Results/recovery/recover_time_" << time_step << ".dat";
			std::string strstr_line1 = str_line1.str();
			output_recovery.open(strstr_line1.c_str(), std::ios::out);
			output_recovery << physical_time << "\t" << global_parameters.D_t;
			output_recovery.close();
		}
	}
}
/*void Flow_solver::temp_monitor(unsigned int time_step, const Thermal_solver& Thermal, const Parallel_MPI& MPI_parallel) {
    const double average_temp = Thermal.average_temp(MPI_parallel);
    if (MPI_parallel.is_master()) {
        /// Open file
        ofstream output_file("Alborz_Results/debug/temp_monitor.dat", fstream::app);
        /// Write data
        output_file << time_step << "\t";  // time step
        output_file << setprecision(20) << physical_time << "\t" << average_temp << endl;
    }
}*/
/// ***************************************************** ///
/// READ PHYSICAL TIME FROM RECOVERY FILE                 ///
/// ***************************************************** ///
void Flow_solver::Recovery_read_physical_time(unsigned int time_step, unsigned int t_recovery) {
	std::stringstream str_line1;
	ifstream intput_recovery;
	if (time_step % t_recovery == 0) {
		str_line1 << "Alborz_Results/recovery/recover_time_" << time_step << ".dat";
		std::string strstr_line1 = str_line1.str();
		intput_recovery.open(strstr_line1.c_str(), std::ios::in);
		intput_recovery >> physical_time >> global_parameters.D_t;
		intput_recovery.close();
	}
}
/// ***************************************************** ///
/// READ IN GENERAL FLOW PARAMETERS, I.E. STENCIL AND     ///
/// AND DIMENSION, R0, GRAVITY                            ///
/// ***************************************************** ///
void Flow_solver::General_data_input(const std::string& filename, Parallel_MPI* MPI_parallel) {
	/// Open input file
	ifstream input_file(filename + ".dat", ios::binary);

	find_line_after_header(input_file, "c\tFlow Field Solver");
	find_line_after_comment(input_file);
	/// input_file >> initial_condition_type;
	input_file >> Dimension >> Discrete_Velocity >> rho_0 >> nu_0 >> gravity[0] >> gravity[1] >> gravity[2] >> p_th_0 >> M_av;
	gravity[0] *= (global_parameters.D_t * global_parameters.D_t / global_parameters.D_x);
	gravity[1] *= (global_parameters.D_t * global_parameters.D_t / global_parameters.D_x);
	gravity[2] *= (global_parameters.D_t * global_parameters.D_t / global_parameters.D_x);
	fluid_constant_viscosity = nu_0 * (global_parameters.D_t / sqr(global_parameters.D_x));
	double rel_Tau = 0.5 + (3.0 * nu_0  * global_parameters.D_t / sqr(global_parameters.D_x));
	double Re = global_parameters.D_x * ini_velocity / nu_0; // Reynolds number = rho_0*U_0*L/nu
	double cal_Pressure = (rho_0 * pow(global_parameters.D_x / global_parameters.D_t,2)) / 3;
	input_file.close();

	if (MPI_parallel->processor_id == MASTER + 1) {
		std::cout << "====================================================\n";
		std::cout << "Flow field parameters\n=====================\n";
		std::cout << "Stencil : " << left << "D" << Dimension << "Q" << Discrete_Velocity << "\n";
		std::cout << setw(COLUMN_WIDTH) << left << "rho_0 = " << rho_0 << "\n";
		std::cout << setw(COLUMN_WIDTH) << left << "nu_0 = " << fluid_constant_viscosity << "\n";
		std::cout << "[g_x g_y g_z] : " << gravity[0] << " " << gravity[1] << " " << gravity[2] << "\n";
		std::cout << "Background pressure : " << p_th_0 << endl;
		std::cout << "Relaxation Tau : " << rel_Tau << endl;
		std::cout << "Reynolds number : " << Re << endl;
		std::cout << "Calculated Pressure : " << cal_Pressure << endl;
		std::cout << "====================================================\n";
		/// std::cout << "Simulation type : " << initial_condition_type << endl;
	}
}
/// ***************************************************** ///
/// INITIALIZE STENCIL AND ALLOCATE MEMORY FOR            ///
/// POPULATIONS AND PARAMETERS                            ///
/// ***************************************************** ///
void Flow_solver::Memory_allocation(Stencil_Definition Stencil_Def, Parallel_MPI* MPI_parallel) {
	///  Stencil parameters initialization
	Stencil_Def(Dimension, Discrete_Velocity, weight, c_alpha, alpha_bar, c_s2);

	Stress.fill({});
	/////////////////////////////////////////////////////////////
	/// Allocate memory for the fluid density, velocity, and force

	// for collective IO additional dimensions need to be correct even for empty tensors
	const Index x_size = MPI_parallel->is_master() ? 0 : static_cast<Index>(MPI_parallel->dev_end[0]);
	const Index y_size = MPI_parallel->is_master() ? 0 : static_cast<Index>(MPI_parallel->dev_end[1]);
	const Index z_size = MPI_parallel->is_master() ? 0 : static_cast<Index>(MPI_parallel->dev_end[2]);

	const Scalar_field::Index_vec scalar_sizes{x_size,
	                                           y_size,
	                                           z_size};
	const Vector_field::Index_vec vec_sizes{x_size,
	                                        y_size,
	                                        z_size,
	                                        3};

	density = Scalar_field::zeros(scalar_sizes);
	previous_density = density;
	pressure = Scalar_field::zeros(scalar_sizes);
	previous_pressure = pressure;
	velocity = Vector_field::zeros(vec_sizes);
	previous_velocity = velocity;
	force = Vector_field::zeros(vec_sizes);
	temp_force = force;
	is_solid = Solid_field::ones(scalar_sizes);
	is_solid *= -1;
	velocity_magnitude = Scalar_field::zeros(scalar_sizes);
	previous_velocity_magnitude = velocity_magnitude;
	viscosity = Scalar_field::zeros(scalar_sizes);
#ifdef LMNA_solver
	divU = Scalar_field::zeros(scalar_sizes);
	previous_divU = divU;
#endif  // defined
#ifdef DEBUG_MODE
	alpha_entropic = Scalar_field::zeros(scalar_sizes);
#endif  // DEBUG_MODE
	velocity_corrections = Vector_field::zeros(vec_sizes);

	const Vector_field::Index_vec pop_sizes{
		x_size,
		y_size,
		z_size,
		static_cast<Index>(Discrete_Velocity),
	};

	pop_eq = new double[Discrete_Velocity];
	pop_w = new double[Discrete_Velocity];
	pop_neq = new double[Discrete_Velocity];
	pop_temp = new double[Discrete_Velocity];
	pop = Vector_field::zeros(pop_sizes);
	pop_old = pop;

	c_alpha_offsets.reserve(Discrete_Velocity);
	for (int alpha = 0; alpha < Discrete_Velocity; ++alpha) {
		c_alpha_offsets.push_back(is_solid.flat_index({c_alpha[alpha][0], c_alpha[alpha][1], c_alpha[alpha][2]}));
	}

	if (!MPI_parallel->is_master()) {
		pop_group = Data_exchange_group(*MPI_parallel);
		pop_group.add_population(pop, c_alpha);
		macroscopic_group = Data_exchange_group(*MPI_parallel);
		macroscopic_group.add_field(density);
		macroscopic_group.add_field(velocity);
		macroscopic_group.add_field(pressure);
#ifdef DEBUG_MODE
		macroscopic_group.add_field(alpha_entropic);
#endif
	}
}
/// ***************************************************** ///
/// INITIALIZE FIELD VARIABLES, DENSITY AND VELOCITY AND  ///
/// GEOMETRY                                              ///
/// ***************************************************** ///
void Flow_solver::initialize_field(Geometry* Geo, stl_import* Geo_stl, Flow_Ini Ini_Field, Thermal_solver* Thermal, Species_solver* Species, Parallel_MPI* MPI_parallel, std::string filename) {
	if (MPI_parallel->processor_id != MASTER) {
		if (Geo->flag == TRUE) {
			for (int X = 0; X < global_parameters.Nx; X++) {
				for (int Y = 0; Y < global_parameters.Ny; Y++) {
					for (int Z = 0; Z < global_parameters.Nz; Z++) {
						is_solid[{X, Y, Z}] = Geo->img[X][Y];  // initialise "is_solid" field, where Geo->img is a 2D array
					}
				}
			}
		}
		if (Geo_stl->flag == 1) {
			for (int X = 0; X < MPI_parallel->dev_end[0]; X++) {
				for (int Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
					for (int Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
						is_solid[{X, Y, Z}] = Geo_stl->domain[{X, Y, Z}];  // Initialize the 'is_solid' field based on STL geo assuming Geo_stl->domain is a 3D array indicating solid cells
					}
				}
			}
		}
		Ini_Field(velocity, density, force, viscosity, is_solid, global_parameters.Nx, global_parameters.Ny, global_parameters.Nz, global_parameters.D_x, global_parameters.D_t, rho_0, c_s2, filename, Geo_stl->Source_count, MPI_parallel);
		for (int X = 0; X < MPI_parallel->dev_end[0]; ++X) {
			for (int Y = 0; Y < MPI_parallel->dev_end[1]; ++Y) {
				for (int Z = 0; Z < MPI_parallel->dev_end[2]; ++Z) {
					velocity_magnitude[{X, Y, Z}] = 0;
					pressure[{X, Y, Z}] = 0;
					if (is_solid[{X, Y, Z}] == FALSE) {
#if defined compressible
						const double conv_factor = c_s2 * Thermal->T_0 / sqr(global_parameters.D_x / global_parameters.D_t);  /// theta = T / (kBT0/m0) = T / (dx2/dt2)/3
#if defined Flow_With_Species
						const double r = R_GAS / Species->molar_mass_av[{X, Y, Z}];
#else
						const double r = R_GAS / M_av;
#endif  // defined
						const double theta = r * Thermal->temperature[{X, Y, Z}] * conv_factor;
#else
						constexpr double theta = 1.0;
#endif
						pressure[{X, Y, Z}] = density[{X, Y, Z}] * theta / c_s2;
					}
					previous_pressure[{X, Y, Z}] = pressure[{X, Y, Z}];
					previous_density[{X, Y, Z}] = density[{X, Y, Z}];
#if defined LMNA_solver
					divU[{X, Y, Z}] = 0;
#endif  // defined
					for (int d = 0; d < 3; d++) {
						previous_velocity[{X, Y, Z, d}] = velocity[{X, Y, Z, d}];
						velocity_magnitude[{X, Y, Z}] += sqr(velocity[{X, Y, Z, d}]);
					}
					velocity_magnitude[{X, Y, Z}] = sqrt(velocity_magnitude[{X, Y, Z}]);
					previous_velocity_magnitude[{X, Y, Z}] = velocity_magnitude[{X, Y, Z}];
				}
			}
		}
	}

	non_solid_lattice.compute_intervals(is_solid, *MPI_parallel);
}
/// ***************************************************** ///
/// INITIALIZE POPULATIONS USING ONLY EQUILIBRIUM PART    ///
/// ***************************************************** ///
void Flow_solver::initialize_pop_eq(Parallel_MPI* MPI_parallel, Thermal_solver* Thermal, Species_solver* Species, std::string filename) {
	if (MPI_parallel->processor_id != MASTER) {
		for (int X = 0; X < MPI_parallel->dev_end[0]; ++X) {
			for (int Y = 0; Y < MPI_parallel->dev_end[1]; ++Y) {
				for (int Z = 0; Z < MPI_parallel->dev_end[2]; ++Z) {
#if defined compressible
					/* temperature in SI units */
					double T = Thermal->temperature[{X, Y, Z}] * Thermal->T_0;
					/* rT in SI units */
					double conv = c_s2 / sqr(global_parameters.D_x / global_parameters.D_t);
					const double r = R_GAS / M_av;
					double theta = r * T * conv;
#else
					constexpr double theta = 1. / 3.;
#endif
					// std::cout << theta << " ";
					/* Optimal theta = 1 */
					equilibrium(density[{X, Y, Z}], theta, &velocity[{X, Y, Z, 0}], pop_eq);
					for (int alpha = 0; alpha < Discrete_Velocity; ++alpha) {
						pop[{X, Y, Z, alpha}] = pop_eq[alpha];
						pop_old[{X, Y, Z, alpha}] = pop[{X, Y, Z, alpha}];
					}
				}
			}
		}
	}
}
/// ***************************************************** ///
/// INITIALIZE POPULATIONS USING ONLY EQUILIBRIUM PART    ///
/// FOR THE LOW MACH MODEL                                ///
/// ***************************************************** ///
// initializes populations (pop). computes the equilibrium populations (pop_eq) based on given density, velocity, and theta values.
// The populations are initialized using a combination of equilibrium values and additional terms based on discrete velocities.
void Flow_solver::initialize_pop_eq_LMNA(Parallel_MPI* MPI_parallel, std::string filename) {
	int X, Y, Z, alpha;
	double theta;
	if (MPI_parallel->processor_id != MASTER) {
		for (X = 0; X < MPI_parallel->dev_end[0]; ++X) {
			for (Y = 0; Y < MPI_parallel->dev_end[1]; ++Y) {
				for (Z = 0; Z < MPI_parallel->dev_end[2]; ++Z) {
					theta = 1;
					equilibrium(density[{X, Y, Z}], theta, &velocity[{X, Y, Z, 0}], pop_eq);
					pressure[{X, Y, Z}] = 0;
					for (alpha = 0; alpha < Discrete_Velocity; ++alpha) {
						pop[{X, Y, Z, alpha}] = (1. / c_s2) * pop_eq[alpha] + weight[alpha] * (pressure[{X, Y, Z}] - (1. / c_s2) * density[{X, Y, Z}]);
						pop_old[{X, Y, Z, alpha}] = pop[{X, Y, Z, alpha}];
					}
					/*// Print old and new populations at the first and last nodes
					if (X == 0 && Y == 0 && Z == 0) {
					    std::cout << "First Node - Old Populations: ";
					    for (alpha = 0; alpha < Discrete_Velocity; ++alpha) {
					        std::cout << pop_old[{X, Y, Z, alpha}] << " ";
					    }
					    std::cout << std::endl;

					    std::cout << "First Node - New Populations: ";
					    for (alpha = 0; alpha < Discrete_Velocity; ++alpha) {
					        std::cout << pop[{X, Y, Z, alpha}] << " ";
					    }
					    std::cout << std::endl;
					}

					if (X == MPI_parallel->dev_end[0] - 1 && Y == MPI_parallel->dev_end[1] - 1 && Z == MPI_parallel->dev_end[2] - 1) {
					    std::cout << "Last Node - Old Populations: ";
					    for (alpha = 0; alpha < Discrete_Velocity; ++alpha) {
					        std::cout << pop_old[{X, Y, Z, alpha}] << " ";
					    }
					    std::cout << std::endl;

					    std::cout << "Last Node - New Populations: ";
					    for (alpha = 0; alpha < Discrete_Velocity; ++alpha) {
					        std::cout << pop[{X, Y, Z, alpha}] << " ";
					    }
					    std::cout << std::endl;
					}*/
				}
			}
		}
	}
}
/// ***************************************************** ///
/// INITIALIZE POPULATIONS USING THE EQUILIBRIUM AND FIRST///
///-ORDER NON_EQUILIBRIUM PART FROM CE EXPANSION          ///
/// ***************************************************** ///
void Flow_solver::initialize_pop_grad(Parallel_MPI* MPI_parallel, Thermal_solver* Thermal, Species_solver* Species, std::string filename) {
	int X, Y, Z, alpha;
	double theta;
	if (MPI_parallel->processor_id != MASTER) {
		double omega_eff;
#if defined compressible
		const double conv_factor = c_s2 * Thermal->T_0 / sqr(global_parameters.D_x / global_parameters.D_t);
#endif
		for (X = 0; X < MPI_parallel->dev_end[0]; ++X) {
			for (Y = 0; Y < MPI_parallel->dev_end[1]; ++Y) {
				for (Z = 0; Z < MPI_parallel->dev_end[2]; ++Z) {
					theta = 1;
#if defined compressible
#if defined Flow_With_Species
					const double r = R_GAS / Species->molar_mass_av[{X, Y, Z}];
#else
					const double r = R_GAS / M_av;
#endif  // defined
					theta = r * Thermal->temperature[{X, Y, Z}] * conv_factor;
#endif
					omega_eff = 1. / (c_s2 * viscosity[{X, Y, Z}] / theta + 0.5);
					Stress[0][0] = -density[{X, Y, Z}] * (theta / c_s2) * (velocity[{(X + 1) % MPI_parallel->dev_end[0], Y, Z, 0}] - velocity[{(X - 1 + MPI_parallel->dev_end[0]) % MPI_parallel->dev_end[0], Y, Z, 0}]) / omega_eff;
					Stress[1][1] = -density[{X, Y, Z}] * (theta / c_s2) * (velocity[{X, (Y + 1) % MPI_parallel->dev_end[1], Z, 1}] - velocity[{X, (Y - 1 + MPI_parallel->dev_end[1]) % MPI_parallel->dev_end[1], Z, 1}]) / omega_eff;
					Stress[2][2] = -density[{X, Y, Z}] * (theta / c_s2) * (velocity[{X, Y, (Z + 1) % MPI_parallel->dev_end[2], 2}] - velocity[{X, Y, (Z - 1 + MPI_parallel->dev_end[2]) % MPI_parallel->dev_end[2], 2}]) / omega_eff;

					Stress[1][0] = -0.5 * density[{X, Y, Z}] * (theta / c_s2) * ((velocity[{(X + 1) % MPI_parallel->dev_end[0], Y, Z, 1}] - velocity[{(X - 1 + MPI_parallel->dev_end[0]) % MPI_parallel->dev_end[0], Y, Z, 1}]) + (velocity[{X, (Y + 1) % MPI_parallel->dev_end[1], Z, 0}] - velocity[{X, (Y - 1 + MPI_parallel->dev_end[1]) % MPI_parallel->dev_end[1], Z, 0}])) / omega_eff;
					Stress[0][1] = Stress[1][0];

					Stress[2][0] = -0.5 * density[{X, Y, Z}] * (theta / c_s2) * ((velocity[{(X + 1) % MPI_parallel->dev_end[0], Y, Z, 2}] - velocity[{(X - 1 + MPI_parallel->dev_end[0]) % MPI_parallel->dev_end[0], Y, Z, 2}]) + (velocity[{X, Y, (Z + 1) % MPI_parallel->dev_end[2], 0}] - velocity[{X, Y, (Z - 1 + MPI_parallel->dev_end[2]) % MPI_parallel->dev_end[2], 0}])) / omega_eff;
					Stress[0][2] = Stress[2][0];

					Stress[2][1] = -0.5 * density[{X, Y, Z}] * (theta / c_s2) * ((velocity[{X, (Y + 1) % MPI_parallel->dev_end[1], Z, 2}] - velocity[{X, (Y - 1 + MPI_parallel->dev_end[1]) % MPI_parallel->dev_end[1], Z, 2}]) + (velocity[{X, Y, (Z + 1) % MPI_parallel->dev_end[2], 1}] - velocity[{X, Y, (Z - 1 + MPI_parallel->dev_end[2]) % MPI_parallel->dev_end[2], 1}])) / omega_eff;
					Stress[1][2] = Stress[1][0];

					equilibrium(density[{X, Y, Z}], theta, &velocity[{X, Y, Z, 0}], pop_eq);
					double Hxx, Hyy, Hzz, Hxy, Hxz, Hyz, dS_alpha;
					for (alpha = 0; alpha < Discrete_Velocity; ++alpha) {
						Hxx = c_alpha[alpha][0] * c_alpha[alpha][0] - (1. / c_s2);
						Hyy = c_alpha[alpha][1] * c_alpha[alpha][1] - (1. / c_s2);
						Hzz = c_alpha[alpha][2] * c_alpha[alpha][2] - (1. / c_s2);
						Hxy = c_alpha[alpha][0] * c_alpha[alpha][1];
						Hxz = c_alpha[alpha][0] * c_alpha[alpha][2];
						Hyz = c_alpha[alpha][1] * c_alpha[alpha][2];
						dS_alpha = 0.5 * c_s2 * c_s2 * weight[alpha] * (Hxx * Stress[0][0] + Hyy * Stress[1][1] + Hzz * Stress[2][2] + 2. * (Hxy * Stress[0][1] + Hxz * Stress[0][2] + Hyz * Stress[1][2]));
						pop[{X, Y, Z, alpha}] = pop_eq[alpha] + dS_alpha;
						pop_old[{X, Y, Z, alpha}] = pop[{X, Y, Z, alpha}];
					}
				}
			}
		}
	}
	return;
}
/// ***************************************************** ///
/// GET DENSITY FROM TEMPERATURE AND SPECIES FIELDS FROM   ///
/// IDEAL GAS LAW                                         ///
/// ***************************************************** ///
void Flow_solver::initialize_rho_LMNA(Parallel_MPI* MPI_parallel, Thermal_solver* Thermal, Species_solver* Species) {
	if (MPI_parallel->processor_id != MASTER) {
		int X, Y, Z;
		for (X = 0; X < MPI_parallel->dev_end[0]; ++X) {
			for (Y = 0; Y < MPI_parallel->dev_end[1]; ++Y) {
				for (Z = 0; Z < MPI_parallel->dev_end[2]; ++Z) {
					if (is_solid[{X, Y, Z}] == FALSE) {  /// if a node has at least one fluid neighbor
#if defined Flow_With_Species
						density[{X, Y, Z}] = p_th_0 * Species->molar_mass_av[{X, Y, Z}] / (R_GAS * Thermal->temperature[{X, Y, Z}] * Thermal->T_0 * rho_0);
						// rho = (p_th * M_av_of Species) / (R_GAS * T_species * T_0 * rho_0);
#endif
#if !defined Flow_With_Species
						density[{X, Y, Z}] = p_th_0 * M_av / (R_GAS * Thermal->temperature[{X, Y, Z}] * Thermal->T_0 * rho_0);
						// Fluid density initialization based on ideal gas law and Species considerations
#endif
					}
				}
			}
		}
	}
	return;
}
/// ***************************************************** ///
/// GET DENSITY FROM TEMPERATURE AND SPECIES FIELS FROM   ///
/// IDEAL GAS LAW (COMPRESSIBLE SOLVER)                   ///
/// ***************************************************** ///
void Flow_solver::initialize_rho_compressible(Parallel_MPI* MPI_parallel, Thermal_solver* Thermal, Species_solver* Species) {
	if (MPI_parallel->processor_id != MASTER) {
		int X, Y, Z;
		for (X = MPI_parallel->start_XYZ2[0] - 1; X <= MPI_parallel->end_XYZ2[0] + 1; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1] - 1; Y <= MPI_parallel->end_XYZ2[1] + 1; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2] - 1; Z <= MPI_parallel->end_XYZ2[2] + 1; ++Z) {
					if ((is_solid[{X, Y, Z}]
					     + is_solid[{X + 1, Y, Z}] + is_solid[{X - 1, Y, Z}]
					     + is_solid[{X, Y + 1, Z}] + is_solid[{X, Y - 1, Z}]
					     + is_solid[{X, Y, Z + 1}] + is_solid[{X, Y, Z - 1}])
					    < 7) {  /// if a node has at least one fluid neighbor
						density[{X, Y, Z}] = p_th_0 * Species->molar_mass_av[{X, Y, Z}] / (R_GAS * Thermal->temperature[{X, Y, Z}] * Thermal->T_0 * rho_0);
					}
				}
			}
		}
	}
}
/// ***************************************************** ///
/// INITIAL THERMODYNAMIC PRESSURE FOR LOW MACH SOLVER    ///
/// ***************************************************** ///
void Flow_solver::initialize_p_th_LMNA(Parallel_MPI* MPI_parallel, Thermal_solver* Thermal, Species_solver* Species) {
	int X, Y, Z;
	double mass_temp = 0;
	if (MPI_parallel->processor_id != MASTER) {
		for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (is_solid[{X, Y, Z}] == FALSE) {  /// if a node has at least one fluid neighbor
#if defined Flow_With_Species
						mass_temp += Species->molar_mass_av[{X, Y, Z}] / (Thermal->temperature[{X, Y, Z}] * Thermal->T_0 * R_GAS);
#endif
#if !defined Flow_With_Species
						mass_temp += M_av / (Thermal->temperature[{X, Y, Z}] * Thermal->T_0 * R_GAS);
#endif
					}
				}
			}
		}
	}
	MPI_Allreduce(&mass_temp, &mass_0, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	/// mass_0 = sum_{x,y,z} 1/rT(0)
	p_th = p_th_0;
	p_th_previous = p_th_0;
}
/// ***************************************************** ///
/// INITIALIZE BOUNDARY CONDITIONS                        ///
/// ***************************************************** ///
static unsigned uses_img_point(Fluid_BC_type type) {
	unsigned flags = 0;
	switch (type) {
		case Fluid_BC_type::ZERO_GRAD_2_WEAK:
		case Fluid_BC_type::VELOCITY_NEQ:
		case Fluid_BC_type::PRESSURE_NEQ:
		case Fluid_BC_type::VELOCITY_NEQ_LMNA:
		case Fluid_BC_type::PRESSURE_NEQ_LMNA:
		case Fluid_BC_type::VELOCITY_EQ:
		case Fluid_BC_type::PRESSURE_EQ:
			flags |= Flow_fluid_boundary_node::Img_point_flag::SECOND;
			// no break here because the first point is always needed if we use the second
		case Fluid_BC_type::ZERO_GRAD_1_WEAK:
			flags |= Flow_fluid_boundary_node::Img_point_flag::FIRST;
			break;
		default:
			break;
	};

	return flags;
}

void Flow_solver::initialize_BC(Geometry* Geo, stl_import* Geo_stl, Parallel_MPI* MPI_parallel, const std::string& filename) {
	ifstream input_file(filename + ".dat", ios::binary);
	find_line_after_header(input_file, "c\tFlow Field Boundary Conditions");
	find_line_after_comment(input_file);
	int number_of_BC = 0;
	bool curved_bounce_back;
	input_file >> number_of_BC >> curved_bounce_back;

	// Handle curved boundaries if present
	double stencil_radius = 0.0;
	if (curved_bounce_back) {
		input_file >> stencil_radius;
	}
	std::unordered_map<int, Vec3> flow_rates;

	boundaries.resize(number_of_BC);
	for (int i = 0; i < number_of_BC; i++) {
		find_line_after_comment(input_file);
		unsigned bc_type;
		input_file >> boundaries[i].index >> boundaries[i].in_zone >> boundaries[i].out_zone >> bc_type;
		boundaries[i].type = static_cast<Fluid_BC_type>(bc_type);
		int filtered;
		input_file >> filtered;
		if (filtered == 1) {
			input_file >> boundaries[i].w_c;
			boundaries[i].w_c = boundaries[i].w_c / global_parameters.D_t;
			boundaries[i].pop_filtered = new double[Discrete_Velocity];
		} else {
			boundaries[i].w_c = 0;
		}
		int is_turbulent;
		input_file >> is_turbulent;
		if (is_turbulent) {
			input_file >> boundaries[i].turbulence_intensity;
			boundaries[i].turbulence_intensity *= global_parameters.D_t / global_parameters.D_x;
		}
		switch (boundaries[i].type) {
			case Fluid_BC_type::VELOCITY_WEAK:        // case 2
			case Fluid_BC_type::VELOCITY_NEQ:         // case 6
			case Fluid_BC_type::VELOCITY_EQ:          // case 8
			case Fluid_BC_type::VELOCITY_LMNA:        // case 12
			case Fluid_BC_type::VELOCITY_NEQ_LMNA: {  // case 16
				int is_flow_rate;
				input_file >> boundaries[i].v[0] >> boundaries[i].v[1] >> boundaries[i].v[2] >> is_flow_rate;
				if (is_flow_rate) {
					flow_rates.emplace(i, boundaries[i].v * global_parameters.D_t / pow(global_parameters.D_x, 3));
				} else {
					boundaries[i].v *= global_parameters.D_t / global_parameters.D_x;
				}
				break;
			}
			case Fluid_BC_type::PRESSURE_WEAK:  // case 3
			case Fluid_BC_type::PRESSURE_NEQ:   // case 7
			case Fluid_BC_type::PRESSURE_EQ: {  // case 9
				input_file >> boundaries[i].p;
				boundaries[i].p *= c_s2 / (rho_0 * sqr(global_parameters.D_x / global_parameters.D_t));
				break;
			}
			case Fluid_BC_type::PRESSURE_LMNA: {  // case 13
				input_file >> boundaries[i].p;
				break;
			}
			case Fluid_BC_type::PRESSURE_NEQ_LMNA: {  // case 17
				input_file >> boundaries[i].p;
				boundaries[i].p /= rho_0;
				break;
			}
			// no parameters
			case Fluid_BC_type::WALL_WEAK:               // case 1
			case Fluid_BC_type::ZERO_GRAD_1_WEAK:        // case 4
			case Fluid_BC_type::ZERO_GRAD_2_WEAK:        // case 5
			case Fluid_BC_type::NON_REFLECTING_OUTFLOW:  // case 14
			case Fluid_BC_type::CONVECTIVE_LMNA:         // case 18
			case Fluid_BC_type::SYMMETRICAL_LMNA:        // case 0
				break;
			default: {
				ERROR_ABORT("Flow boundary condition of type " << static_cast<unsigned>(boundaries[i].type)
				                                               << " is not recognized.");
			}
		}
	}
	input_file.close();

	for (int i = 0; i < number_of_BC; i++) {
		// boundary_data represents data associated with a specific boundary condition, such as the type of boundary and the velocity (v) at that boundary.
		auto& boundary_data = boundaries[i];
		// boundary_nodes Stores information about nodes on the boundary, retrieved from Geo_stl based on the specified input zones.
		auto& boundary_nodes = Geo_stl->get_boundary_nodes(boundary_data.in_zone, boundary_data.out_zone);
		size_t num_nodes = boundary_nodes.size();
		size_t num_nodes_total = 0;
		MPI_Reduce(&num_nodes, &num_nodes_total, 1, to_MPI_type<size_t>::value(), MPI_SUM, MASTER, MPI_COMM_WORLD);
		if (MPI_parallel->is_master() && !num_nodes_total) {
			std::cout << "[Warning] Flow field boundary with index " << boundary_data.index
					  << " between meshes " << boundary_data.in_zone << " and " << boundary_data.out_zone
					  << " consists of 0 nodes." << std::endl;
		}
	}

	for (int i = 0; i < number_of_BC; i++) {
		auto& boundary_data = boundaries[i];
		auto& boundary_nodes = Geo_stl->get_boundary_nodes(boundary_data.in_zone, boundary_data.out_zone);

		for (const Boundary_node& node : boundary_nodes) {
			Flow_solid_boundary_node flow_node;  // Create a Flow_solid_boundary_node based on boundary data
			flow_node.idx = node.idx;
			flow_node.flat_idx = node.flat_idx;
			flow_node.n = Geo_stl->compute_simple_normal(node.idx, is_solid, -1);
			flow_node.v = boundary_data.v;

			for (unsigned alpha = 1; alpha < Discrete_Velocity; alpha++) {  // Initialize directions based on fluid neighbors and a velocity value
				const Index Xp = node.idx[0] - c_alpha[alpha][0];           // c_alpha =  array storing velocity directions. Used to compute neighboring fluid nodes.
				const Index Yp = node.idx[1] - c_alpha[alpha][1];
				const Index Zp = node.idx[2] - c_alpha[alpha][2];
				// look only for fluid nodes inside domain
				if (Xp >= MPI_parallel->start_XYZ2[0] && Xp <= MPI_parallel->end_XYZ2[0]
				    && Yp >= MPI_parallel->start_XYZ2[1] && Yp <= MPI_parallel->end_XYZ2[1]
				    && Zp >= MPI_parallel->start_XYZ2[2] && Zp <= MPI_parallel->end_XYZ2[2]
				    && Geo_stl->domain[{Xp, Yp, Zp}] == boundary_data.in_zone) {
					// store indices of the relevant directions
					flow_node.directions.push_back(alpha);
				}
			}
			boundary_data.node_data.emplace_back(std::move(flow_node));
		}

		std::unordered_set<Index_vec3> fluid_nodes_set;
		// Without curved boundaries there is always only one node available
		// so we can't exclude that. However, access of a neighboring BC node,
		// such that this becomes a problem, only happens for very specific grids,
		// i.e. a corner, surrounded by solid nodes.
		if (stencil_radius > 0.0) {
			for (const Flow_solid_boundary_node& node : boundary_data.node_data) {
				for (int alpha : node.directions) {
					const Index_vec3 fluid_idx{node.idx[0] - c_alpha[alpha][0],
					                           node.idx[1] - c_alpha[alpha][1],
					                           node.idx[2] - c_alpha[alpha][2]};
					fluid_nodes_set.insert(fluid_idx);
				}
			}
		}

		// initialize fluid nodes
		std::unordered_map<Index_vec3, size_t> fluid_nodes_idx;
		boundary_data.fluid_node_data.reserve(fluid_nodes_set.size());  // Store information about fluid nodes in boundary_data.fluid_node_data
		size_t ind = 0;

		ASSERT(std::floor(stencil_radius) + 1.0 <= MPI_parallel->buffer_size);
		const unsigned img_points = uses_img_point(boundary_data.type);

		for (Flow_solid_boundary_node& node : boundary_data.node_data) {
			for (int alpha : node.directions) {
				const Index_vec3 fluid_idx{node.idx[0] - c_alpha[alpha][0],
				                           node.idx[1] - c_alpha[alpha][1],
				                           node.idx[2] - c_alpha[alpha][2]};
				// fluid nodes outside of the active area are handled by other processes
				if (fluid_idx[0] < MPI_parallel->start_XYZ2[0] || fluid_idx[0] > MPI_parallel->end_XYZ2[0]
				    || fluid_idx[1] < MPI_parallel->start_XYZ2[1] || fluid_idx[1] > MPI_parallel->end_XYZ2[1]
				    || fluid_idx[2] < MPI_parallel->start_XYZ2[2] || fluid_idx[2] > MPI_parallel->end_XYZ2[2]) {
					continue;
				}

				if (fluid_nodes_idx.find(fluid_idx) == fluid_nodes_idx.end()) {
					Vec3 pos = {fluid_idx[0], fluid_idx[1], fluid_idx[2]};

					// simple normal for straight boundaries and symmetric bc
					const Index_vec3 normal_int = Geo_stl->compute_simple_normal(fluid_idx, is_solid, -1 /*boundary_data.out_zone*/);
					// stencil_radius == 0 only works with regular normals
					const bool curved_boundary_normals = curved_bounce_back && stencil_radius > 0;
					const Vec3 normal = curved_boundary_normals ? -Geo_stl->compute_normal(fluid_idx, *MPI_parallel, boundary_data.in_zone, boundary_data.out_zone)
					                                            : Vec3{static_cast<double>(normal_int[0]), static_cast<double>(normal_int[1]), static_cast<double>(normal_int[2])};
					// straight boundaries with diagonal normals also have len != 1
					// but in that case normal == normal_1, so no scaling is necessary
					const double len = curved_boundary_normals ? std::sqrt(dot(normal, normal)) : 1.0;

					auto compute_image_stencil = [&](const Vec3& pos) {
						auto stencil = interpolation::Stencil(fluid_idx, pos, is_solid, stencil_radius, Dimension == 2, fluid_nodes_set);
						if (!stencil.is_valid()) {
							const Vec3 global_pos = MPI_parallel->get_coordinates(pos, Vec3{Geo_stl->x_center, Geo_stl->y_center, Geo_stl->z_center});
							ERROR_ABORT("Invalid interpolation stencil for boundary node " << fluid_idx << ". No valid node found in radius " << stencil_radius << " from position " << pos << "(" << global_pos << ")\n"
							                                                               << ".");
						}
						return stencil;
					};

					interpolation::Stencil img_stencil;
					if (img_points & Flow_fluid_boundary_node::Img_point_flag::FIRST) {
						img_stencil = compute_image_stencil(pos + normal);
					}
					interpolation::Stencil img_stencil2;
					if (img_points & Flow_fluid_boundary_node::Img_point_flag::SECOND) {
						img_stencil2 = compute_image_stencil(pos + normal + normal / len);
					}

					fluid_nodes_idx.emplace(fluid_idx, boundary_data.fluid_node_data.size());
					boundary_data.fluid_node_data.emplace_back(Flow_fluid_boundary_node{fluid_idx,
					                                                                    is_solid.flat_index(fluid_idx),
					                                                                    normal,
					                                                                    boundary_data.v,
					                                                                    len,
					                                                                    std::move(img_stencil),
					                                                                    std::move(img_stencil2),
					                                                                    normal_int});
				}
				const size_t fluid_node_idx = fluid_nodes_idx[fluid_idx];
				Flow_fluid_boundary_node& fluid_node = boundary_data.fluid_node_data[fluid_node_idx];
				fluid_node.solid_boundary_idx.push_back(ind);
				fluid_node.directions.push_back(alpha_bar[alpha]);
				//	node.fluid_boundary_idx.push_back(fluid_node_idx);
			}
			ind += 1;
		}
		// Sorting the boundaries, likely for consistent treatment, especially at corners.
		for (auto& fluid_node : boundary_data.fluid_node_data) {
			std::sort(fluid_node.directions.begin(), fluid_node.directions.end());
		}
	}

	// enforce order so that interactions at corners are consistent
	std::sort(boundaries.begin(), boundaries.end(), [](const Flow_boundary_data& lhs, Flow_boundary_data& rhs) {
		return static_cast<unsigned>(lhs.type) < static_cast<unsigned>(rhs.type);
	});

	// compute final values for boundaries defined by Flow rates
	for (auto& flow_rate_boundaries : flow_rates) {
		Flow_boundary_data& boundary = boundaries[flow_rate_boundaries.first];
		// Counter number of grid-points to define velocity
		int64_t num_nodes = boundary.fluid_node_data.size();
		int64_t num_nodes_global = 0;
		MPI_Allreduce(&num_nodes, &num_nodes_global, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
		boundary.v = flow_rate_boundaries.second / num_nodes_global;
	}

	initialize_boundary_corners();

	// consistency check for boundary treatment: every node direction should only be handled by one BC
	std::vector<int> directions_intersection;
	for (size_t i = 0; i < boundaries.size(); ++i) {
		for (size_t j = i + 1; j < boundaries.size(); ++j) {
			for (Flow_fluid_boundary_node& node_i : boundaries[i].fluid_node_data) {
				for (Flow_fluid_boundary_node& node_j : boundaries[j].fluid_node_data) {
					if (node_i.flat_idx == node_j.flat_idx) {
						directions_intersection.clear();
						std::set_intersection(node_i.directions.begin(), node_i.directions.end(),
						                      node_j.directions.begin(), node_j.directions.end(),
						                      std::back_inserter(directions_intersection));
						if (!directions_intersection.empty()) {
							std::cout << "[Warning] Directions are treated by multiple boundary conditions at corner node "
									  << node_i.idx << " with types " << static_cast<unsigned>(boundaries[i].type)
									  << " and " << static_cast<unsigned>(boundaries[j].type) << "." << std::endl;
						}
					}
				}
			}
		}
	}
	if (MPI_parallel->processor_id == MASTER + 1) {
		std::filesystem::create_directories("Alborz_Results/Data/");
		std::ofstream BC_output;
		BC_output.open("Alborz_Results/Data/Flow_Boundary_Conditions.dat", fstream::trunc);
		BC_output << "FLOW: Boundary Conditions \n================================ \n";
		BC_output << "Values given in LB units \n";
		BC_output << "dx    : " << global_parameters.D_x << "\n";
		BC_output << "dt    : " << global_parameters.D_t << "\n";
		BC_output << "rho_0 : " << rho_0 << "\n";
		if (curved_bounce_back)
			BC_output << "Curved: "
					  << "\tON\t" << stencil_radius << "\n";
		else
			BC_output << "Curved: "
					  << "\tOFF\n";
		BC_output << "================================ \n";
		for (int i = 0; i < number_of_BC; i++) {
			BC_output << "BOUNDARY INDEX : " << i + 1 << "\t TYPE : " << static_cast<unsigned>(boundaries[i].type) << "\n";
			BC_output << "PRESSURE : " << boundaries[i].p << "\n";
			BC_output << "VELOCITY : " << boundaries[i].v[0] << " " << boundaries[i].v[1] << " " << boundaries[i].v[2] << "\n";
			BC_output << "FILTER : ";
			if (boundaries[i].w_c != 0) {
				BC_output << "\tON\t" << boundaries[i].w_c << "\n";
			} else {
				BC_output << "\tOFF\n";
			}
			BC_output << "TURBULENCE : ";
			if (boundaries[i].turbulence_intensity != 0) {
				BC_output << "\tON\t" << boundaries[i].turbulence_intensity << "\n";
			} else {
				BC_output << "\tOFF\n";
			}
			BC_output << "VOLUME FLOW RATE : ";
			auto it = flow_rates.find(i);
			if (it != flow_rates.end()) {
				BC_output << "\tON\nFLOWRATE : " << it->second << std::endl;
			} else {
				BC_output << "\tOFF\n";
			}
			BC_output << "-------------------------------- \n";
		}
		if (number_of_BC == 0) {
			BC_output << "NO FLOW BOUNDARY CONDITIONS\n";
		}
		BC_output.close();
	}
		if (MPI_parallel->processor_id != MASTER) {
#if defined DEBUG_MODE
		stringstream DB_filename;
		DB_filename << "BC_Check/Flow_Boundary_Conditions_DB_proc_" << MPI_parallel->processor_id << ".dat";
		ofstream BCDEBUG;
		BCDEBUG.open(DB_filename.str().c_str(), fstream::trunc);
		BCDEBUG << "PROCESSOR ID : " << MPI_parallel->processor_id << " OUT OF " << MPI_parallel->num_processors - 1 << "\n";
		BCDEBUG << "BC\tX\tY\tZ\tn_x\tn_y\tn_z\t";
		for (i = 0; i < Discrete_Velocity; i++) {
			BCDEBUG << i << "\t";
		}
		BCDEBUG << "\n";
		for (i = 0; i < Boundaries.size(); i++) {
			BCDEBUG << Boundaries[i].type << "\t" << Boundaries[i].X << "\t" << Boundaries[i].Y << "\t" << Boundaries[i].Z << "\t";
			BCDEBUG << Boundaries[i].n[0] << "\t" << Boundaries[i].n[1] << "\t" << Boundaries[i].n[2] << "\t";
			for (int alpha = 0; alpha < Discrete_Velocity; alpha++) {
				BCDEBUG << Boundaries[i].directions[alpha] << "\t";
			}
			BCDEBUG << "\n";
		}
		BCDEBUG.close();
#endif  // defined
	}
}
void Flow_solver::initialize_boundary_corners() {
	std::vector<size_t> corner_idx_slf;
	std::vector<size_t> corner_idx_oth;

	const size_t num_bc = boundaries.size();
	for (size_t i = 0; i < num_bc; ++i) {
		// put into hash map for constant lookup
		std::unordered_map<Flat_index, size_t> nodes_i;
		for (size_t k = 0; k < boundaries[i].fluid_node_data.size(); ++k) {
			nodes_i.emplace(boundaries[i].fluid_node_data[k].flat_idx, k);
		}

		for (size_t j = i + 1; j < num_bc; ++j) {
			// idx of fluid nodes in boundaries[idx]
			corner_idx_slf.clear();
			corner_idx_oth.clear();

			// check all nodes
			for (size_t k = 0; k < boundaries[j].fluid_node_data.size(); ++k) {
				auto it = nodes_i.find(boundaries[j].fluid_node_data[k].flat_idx);
				if (it != nodes_i.end()) {
					corner_idx_slf.push_back(it->second);
					corner_idx_oth.push_back(k);
				}
			}

			// idx in boundaries
			size_t idx_slf;
			size_t idx_oth;
			// match type independent of order and prepare the corner data
			auto match_type = [&](Fluid_BC_type slf_type, Fluid_BC_type oth_type) {
				if (slf_type == boundaries[i].type && oth_type == boundaries[j].type) {
					idx_slf = i;
					idx_oth = j;
					return true;
				}
				if (oth_type == boundaries[i].type && slf_type == boundaries[j].type) {
					idx_slf = j;
					idx_oth = i;
					std::swap(corner_idx_oth, corner_idx_slf);
					return true;
				}
				return false;
			};

			// wall takes precedence
			if (match_type(Fluid_BC_type::WALL_WEAK, Fluid_BC_type::ZERO_GRAD_1_WEAK)
			    || match_type(Fluid_BC_type::WALL_WEAK, Fluid_BC_type::ZERO_GRAD_2_WEAK)
			    || match_type(Fluid_BC_type::WALL_WEAK, Fluid_BC_type::VELOCITY_NEQ)
			    || match_type(Fluid_BC_type::WALL_WEAK, Fluid_BC_type::VELOCITY_EQ)) {
				//	boundaries[idx_slf].move_nodes(corner_idx_slf, corner_idx_oth, boundaries[idx_oth]);
				boundaries[idx_oth].move_nodes(corner_idx_oth, corner_idx_slf, boundaries[idx_slf]);
			}
		}
	}
}
/// ***************************************************** ///
/// GET DISTANCE FROM LAST SOLID NODE TO STL SURFACE      ///
/// ALONG STENCIL VECTORS FOR CURVED BOUNDARIES           ///
/// ***************************************************** ///
void Flow_solver::initialize_curved_boundaries(const stl_import& geo_stl, const Parallel_MPI& MPI_parallel) {
	if (MPI_parallel.is_master()) {
		return;
	}

	for (auto& boundary : boundaries) {
		for (Flow_solid_boundary_node& node : boundary.node_data) {
			node.distance.resize(node.directions.size(), -1.0);
			for (unsigned i = 0; i < node.directions.size(); ++i) {
				const unsigned alpha = node.directions[i];
				// look for distance to surface from the fluid nodes
				const stl::point O((node.idx[0] - c_alpha[alpha][0] - MPI_parallel.start_XYZ2[0] + MPI_parallel.start_XYZ[0]) * global_parameters.D_x + geo_stl.x_center,
				                   (node.idx[1] - c_alpha[alpha][1] - MPI_parallel.start_XYZ2[1] + MPI_parallel.start_XYZ[1]) * global_parameters.D_x + geo_stl.y_center,
				                   (node.idx[2] - c_alpha[alpha][2] - MPI_parallel.start_XYZ2[2] + MPI_parallel.start_XYZ[2]) * global_parameters.D_x + geo_stl.z_center);

				/// defined the direction vector for the ray (i.e. discrete velocity direction)
				const stl::point D(c_alpha[alpha][0] * global_parameters.D_x,
				                   c_alpha[alpha][1] * global_parameters.D_x,
				                   c_alpha[alpha][2] * global_parameters.D_x);
				for (const stl::stl_data& info : geo_stl.get_stl_data()) {
					for (const stl::triangle& triangle : info.triangles /*geo_stl.get_stl_data()[boundary.in_zone - 1].triangles*/) {
						stl::point intersection_distance;
						/// Check if there is an intersection between the ray in direction c_alpha starting at the boundary node and the stl triangle
						if (stl_import::triangle_intersection(triangle.v1, triangle.v2, triangle.v3, O, D, &intersection_distance)) {
							/// get the distance between the intersection and boundary node
							intersection_distance.x = intersection_distance.x - O.x;
							intersection_distance.y = intersection_distance.y - O.y;
							intersection_distance.z = intersection_distance.z - O.z;
							/// if the distance is smaller than a grid-length store it in memory and stop iteration
							if (std::abs(intersection_distance.x) < global_parameters.D_x
							    && std::abs(intersection_distance.y) < global_parameters.D_x
							    && std::abs(intersection_distance.z) < global_parameters.D_x) {
								node.distance[i] = sqrt(sqr(intersection_distance.x) + sqr(intersection_distance.y) + sqr(intersection_distance.z))
								                   / global_parameters.D_x;
								// distance is only used by curved boundaries which need length normalized by the lattice directions
								node.distance[i] /= sqrt(sqr(c_alpha[alpha][0]) + sqr(c_alpha[alpha][1]) + sqr(c_alpha[alpha][2]));
								break;
							}
						}
					}
					if (node.distance[i] >= 0.0) {
						break;
					}
				}
			}
		}
	}
#if defined DEBUG_MODE
	stringstream DB_filename;
	DB_filename << "Flow_Curved_Boundary_Conditions_DB_proc_" << MPI_parallel.processor_id << ".dat";
	ofstream BCDEBUG;
	BCDEBUG.open(DB_filename.str().c_str(), fstream::trunc);
	BCDEBUG << "PROCESSOR ID : " << MPI_parallel.processor_id << " OUT OF " << MPI_parallel.num_processors - 1 << "\n";
	BCDEBUG << "BC\tX\tY\tZ\t";
	for (i = 0; i < Discrete_Velocity; i++) {
		BCDEBUG << i << "\t";
	}
	BCDEBUG << "\n";
	const int N_x = global_parameters.Nx;
	const int N_y = global_parameters.Ny;
	const int N_z = global_parameters.Nz;
	for (const auto& boundary : boundaries) {
		for (const Flow_solid_boundary_node& node : boundary.node_data) {
			BCDEBUG << Boundaries[i].type << "\t" << fmod(node.idx[0] - MPI_parallel.start_XYZ2[0] + MPI_parallel.start_XYZ[0] + N_x, N_x) * global_parameters.D_x + x_center
					<< "\t" << fmod(node.idx[1] - MPI_parallel.start_XYZ2[1] + MPI_parallel.start_XYZ[1] + N_y, N_y) * global_parameters.D_x + y_center
					<< "\t" << fmod(node.idx[2] - MPI_parallel.start_XYZ2[2] + MPI_parallel.start_XYZ[2] + N_z, N_z) * global_parameters.D_x + z_center << "\t";
			for (int i = 0; i < node.directions.size(); ++i) {
				BCDEBUG << node.directions[i] << " : " << node.distance[i] * global_parameters.D_x << "\t";
			}
			BCDEBUG << "\n";
		}
	}
	BCDEBUG.close();
#endif  // defined
}
/// ***************************************************** ///
/// GET EQUILIBRIUM POPULATIONS AND PUT THEM IN F_EQ      ///
/// ***************************************************** ///
/* Equilibrium function: The function calculates equilibrium populations for a given density, velocity, and temperature.
The equilibrium populations are stored in the array f_eq.
It is calculated with the following formula:
f_eq = w_i * rho * (1 + 3 * c_i * u + 4.5 * (c_i * u)^2 - 1.5 * u^2)
where w_i is the weight of the i-th velocity direction, rho is the density, c_i is the i-th velocity direction, and u is the velocity.
The equilibrium populations are used in the collision step to relax the distribution functions towards local equilibrium.
*/
void Flow_solver::equilibrium(double den, double theta, const double* vel, double* f_eq) {
	const Vec3 eta = {theta + vel[0] * vel[0],
	                  theta + vel[1] * vel[1],
	                  theta + vel[2] * vel[2]};

	for (int i = 0; i < Discrete_Velocity; i++) {
		double eq = den;
		for (int D = 0; D < 3; D++) {
			const double C = c_alpha[i][D];
			const double Psi = (1. - eta[D]) * (1. - fabs(C)) + 0.5 * fabs(C) * (C * vel[D] + eta[D]);
			eq *= Psi;
		}
		f_eq[i] = eq;
	}
}
/// ***************************************************** ///
/// COLLISION AND STREAMING USING HCM-MRT MODEL           ///
/// ***************************************************** ///
void Flow_solver::LBM_CM_MRT(int time, Thermal_solver* Thermal, Species_solver* Species, Geometry* Geo, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		/// Swap populations
		swap(pop_old, pop);

		// select correct implementation for stencil
		DISPATCH_BY_STENCIL(LBM_CM_MRT_impl, *MPI_parallel, *Thermal);
	}
}
/// ***************************************************** ///
/// COLLISION AND STREAMING SRT MODEL FOR LOW MACH        ///
/// ***************************************************** ///
void Flow_solver::LBM_SRT_LMNA(int time, Thermal_solver* Thermal, Species_solver* Species, Geometry* Geo, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		swap(pop_old, pop);
		double velv[3];
		double grad_rho[3], g_eq_alpha, omega_eff, fm, fdir;
		int X, Y, Z, alpha;
		double theta;

		for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (is_solid[{X, Y, Z}] == FALSE) {
						theta = 1.;
						equilibrium(density[{X, Y, Z}], theta, &velocity[{X, Y, Z}], pop_eq);
						omega_eff = 1. / (c_s2 * viscosity[{X, Y, Z}] + 0.5);
						/// Kn_filtering (X, Y, Z, omega_eff);
						grad_rho[0] = 0;
						grad_rho[1] = 0;
						grad_rho[2] = 0;
#if defined FD_CENTRAL || defined FD_UPWIND2 || defined FD_UPWIND
						grad_rho[0] = FD::CENTRALNONCONS(1.0, density[{X - 1, Y, Z}], density[{X + 1, Y, Z}]);
						grad_rho[1] = FD::CENTRALNONCONS(1.0, density[{X, Y - 1, Z}], density[{X, Y + 1, Z}]);
						grad_rho[2] = FD::CENTRALNONCONS(1.0, density[{X, Y, Z - 1}], density[{X, Y, Z + 1}]);
#endif  // defined
#if defined FD_CENTRAL4 || defined FD_WENO3
						grad_rho[0] = FD::CENTRAL4NONCONS(1.0, density[{X - 2, Y, Z}], density[{X - 1, Y, Z}], density[{X, Y, Z}], density[{X + 1, Y, Z}], density[{X + 2, Y, Z}]);
						grad_rho[1] = FD::CENTRAL4NONCONS(1.0, density[{X, Y - 2, Z}], density[{X, Y - 1, Z}], density[{X, Y, Z}], density[{X, Y + 1, Z}], density[{X, Y + 2, Z}]);
						grad_rho[2] = FD::CENTRAL4NONCONS(1.0, density[{X, Y, Z - 2}], density[{X, Y, Z - 1}], density[{X, Y, Z}], density[{X, Y, Z + 1}], density[{X, Y, Z + 2}]);
						if (is_solid[{X + 1, Y, Z}] != -1 || is_solid[{X - 1, Y, Z}] != -1) {
							grad_rho[0] = FD::CENTRALNONCONS(1., density[{X - 1, Y, Z}], density[{X + 1, Y, Z}]);
						}
						if (is_solid[{X, Y + 1, Z}] != -1 || is_solid[{X, Y - 1, Z}] != -1) {
							grad_rho[1] = FD::CENTRALNONCONS(1., density[{X, Y - 1, Z}], density[{X, Y + 1, Z}]);
						}
						if (is_solid[{X, Y, Z + 1}] != -1 || is_solid[{X, Y, Z - 1}] != -1) {
							grad_rho[2] = FD::CENTRALNONCONS(1., density[{X, Y, Z - 1}], density[{X, Y, Z + 1}]);
						}
#endif  // defined
#if defined FD_ISOTROPIC
						grad_rho[0] = 0;
						grad_rho[1] = 0;
						grad_rho[2] = 0;
						for (alpha = 1; alpha < Discrete_Velocity; alpha++) {
							double density_p = density[X + c_alpha[alpha][0]][Y + c_alpha[alpha][1]][Z + c_alpha[alpha][2]];
							grad_rho[0] += c_s2 * (weight[alpha] * c_alpha[alpha][0] * density_p);
							grad_rho[1] += c_s2 * (weight[alpha] * c_alpha[alpha][1] * density_p);
							grad_rho[2] += c_s2 * (weight[alpha] * c_alpha[alpha][2] * density_p);
						}
#endif
						if (Dimension < 2) grad_rho[1] = 0;
						if (Dimension < 3) grad_rho[2] = 0;
						double forcingX = force[{X, Y, Z, 0}] / density[{X, Y, Z}];
						double forcingY = force[{X, Y, Z, 1}] / density[{X, Y, Z}];
						double forcingZ = force[{X, Y, Z, 2}] / density[{X, Y, Z}];

						for (alpha = 0; alpha < Discrete_Velocity; alpha++) {
							g_eq_alpha = (1. / c_s2) * pop_eq[alpha] + weight[alpha] * (pressure[{X, Y, Z}] - density[{X, Y, Z}] / c_s2);
							velv[0] = grad_rho[0] * (pop_eq[alpha] / density[{X, Y, Z}] - weight[alpha]) / c_s2 + forcingX * pop_eq[alpha];
							velv[1] = grad_rho[1] * (pop_eq[alpha] / density[{X, Y, Z}] - weight[alpha]) / c_s2 + forcingY * pop_eq[alpha];
							velv[2] = grad_rho[2] * (pop_eq[alpha] / density[{X, Y, Z}] - weight[alpha]) / c_s2 + forcingZ * pop_eq[alpha];
							fm = velocity[{X, Y, Z, 0}] * velv[0] + velocity[{X, Y, Z, 1}] * velv[1] + velocity[{X, Y, Z, 2}] * velv[2];
							fdir = c_alpha[alpha][0] * velv[0] + c_alpha[alpha][1] * velv[1] + c_alpha[alpha][2] * velv[2]
							       + weight[alpha] * (1. / c_s2) * density[{X, Y, Z}] * divU[{X, Y, Z}];

							pop[{X + c_alpha[alpha][0], Y + c_alpha[alpha][1], Z + c_alpha[alpha][2], alpha}] =
								pop_old[{X, Y, Z, alpha}] * (1. - omega_eff) + g_eq_alpha * omega_eff + (1. - 0.5 * omega_eff) * (fdir - fm);
						}
					}
				}
			}
		}
	}
	return;
}
/// ***************************************************** ///
/// COLLISION AND STREAMING USING ENTROPIC SRT MODEL      ///
/// ***************************************************** ///
void Flow_solver::LBM_CM_MRT_LMNA(int time, Thermal_solver* Thermal, Species_solver* Species, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		/* Swap populations */
		swap(pop_old, pop);
		int X, Y, Z, alpha;
		double theta;
		if (Discrete_Velocity == 9) {
			double omega[Discrete_Velocity];
			double Mstar_i[Discrete_Velocity];
			double f_i[Discrete_Velocity];
			double forcingTerm[Discrete_Velocity];
			for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
				for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
					for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
						if (is_solid[{X, Y, Z}] == FALSE) {
							/* Temperature in SI units */
							theta = 1.;
							double cs2 = 1. / c_s2;
							/* We have three types of relaxations */
							/* omega_shear : Shear viscosity */
							/* omega_bulk : Bulk viscosity */
							/* omega_ghost : dissipation of ghost moments */
							double omega_shear = 1. / (c_s2 * viscosity[{X, Y, Z}] + 0.5);
							// double omega_bulk = 1.;
							double omega_ghost = 1.;
							omega[0] = omega_ghost;
							omega[1] = omega_ghost;
							omega[2] = omega_ghost;
							omega[3] = omega_shear;
							omega[4] = omega_shear;
							omega[5] = omega_ghost;
							omega[6] = omega_ghost;
							omega[7] = omega_ghost;
							omega[8] = omega_ghost;

							double rho = density[{X, Y, Z}];
							double rho_x1p = density[{X + 1, Y, Z}];
							double rho_x1n = density[{X - 1, Y, Z}];
							double rho_x2p = density[{X + 2, Y, Z}];
							double rho_x2n = density[{X - 2, Y, Z}];

							double rho_y1p = density[{X, Y + 1, Z}];
							double rho_y1n = density[{X, Y - 1, Z}];
							double rho_y2p = density[{X, Y + 2, Z}];
							double rho_y2n = density[{X, Y - 2, Z}];

							double rho_z1p = density[{X, Y, Z + 1}];
							double rho_z1n = density[{X, Y, Z - 1}];
							double rho_z2p = density[{X, Y, Z + 2}];
							double rho_z2n = density[{X, Y, Z - 2}];

							double ux = velocity[{X, Y, Z, 0}];
							double uy = velocity[{X, Y, Z, 1}];
							double uz = velocity[{X, Y, Z, 2}];
							double divU_term = divU[{X, Y, Z}];
							// double P = pressure[{X, Y, Z}];

							double grad_rho0, grad_rho1;
							double grad_rho2 = 0;
							
							equilibrium(density[{X, Y, Z}], theta, &velocity[{X, Y, Z, 0}], pop_eq);
							double forcingX = force[{X, Y, Z, 0}];
							double forcingY = force[{X, Y, Z, 1}];
							double forcingZ = 0;

#if defined FD_CENTRAL || defined FD_UPWIND2 || defined FD_UPWIND
							grad_rho0 = FD::CENTRALNONCONS(1., rho_x1n, rho_x1p);
							grad_rho1 = FD::CENTRALNONCONS(1., rho_y1n, rho_y1p);
							grad_rho2 = FD::CENTRALNONCONS(1., rho_z1n, rho_z1p);
							grad_rho2 = 0;
#endif  // defined
#if defined FD_CENTRAL4 || defined FD_WENO3
							grad_rho0 = FD::CENTRAL4NONCONS(1.0, rho_x2n, rho_x1n, rho, rho_x1p, rho_x2p);
							grad_rho1 = FD::CENTRAL4NONCONS(1.0, rho_y2n, rho_y1n, rho, rho_y1p, rho_y2p);
							grad_rho2 = FD::CENTRAL4NONCONS(1.0, rho_z2n, rho_z1n, rho, rho_z1p, rho_z2p);
							if (is_solid[{X + 1, Y, Z}] != -1 || is_solid[{X - 1, Y, Z}] != -1) {
								grad_rho0 = FD::CENTRALNONCONS(1.0, rho_x1n, rho_x1p);
							}
							if (is_solid[{X, Y + 1, Z}] != -1 || is_solid[{X, Y - 1, Z}] != -1) {
								grad_rho1 = FD::CENTRALNONCONS(1.0, rho_y1n, rho_y1p);
							}
							if (is_solid[{X, Y, Z + 1}] != -1 || is_solid[{X, Y, Z - 1}] != -1) {
								grad_rho2 = FD::CENTRALNONCONS(1.0, rho_z1n, rho_z1p);
								grad_rho2 = 0;
							}
#endif  // defined
							for (alpha = 0; alpha < Discrete_Velocity; alpha++) {
								int CX = c_alpha[alpha][0];
								int CY = c_alpha[alpha][1];
								int CZ = c_alpha[alpha][2];

								double gamm = pop_eq[alpha] / rho;
								double fac1 = cs2 * (gamm - weight[alpha]);
								forcingTerm[alpha] = weight[alpha] * rho * cs2 * divU_term
								                     + (CX - ux) * (fac1 * grad_rho0 + gamm * forcingX)
								                     + (CY - uy) * (fac1 * grad_rho1 + gamm * forcingY)
													 + (CZ - uz) * (fac1 * grad_rho2 + gamm * forcingZ);
								/* Tranform populations and add half of force */
								f_i[alpha] = c_s2 * (pop_old[{X, Y, Z, alpha}] + 0.5 * forcingTerm[alpha]) / rho;
							}
/// *************************************************************************************************** ///
///                                       EQUILIBRIUM CENTRAL MOMENTS                                   ///
///                                             MOMENT SPACE :                                          ///
///                                                  H_0,                                               ///
///                                                H_x, H_y,                                            ///
///                                             H_xy, H_xx, H_yy                                        ///
///                                              H_xxy, H_xyy                                           ///
///                                                  H_xxyy                                             ///
/// *************************************************************************************************** ///
/// FIRST STEP: GET CENTRAL HERMITE MOMENTS OF EDF (ASSUMING ALL TERMS SUPPORTED BY THE STENCIL ARE KEPT)
#ifndef PERFORMANCE_MODE
							Mstar_i[0] = (3. * P / rho) * omega[0] - (omega[0] - 1.0) * (f_i[0] + f_i[1] + f_i[2] + f_i[3] + f_i[4] + f_i[5] + f_i[6] + f_i[7] + f_i[8]);
							Mstar_i[1] = -((3. * P * ux - rho * ux) / rho) * omega[1] + (omega[1] - 1.0) * (ux * f_i[0] + ux * f_i[2] + ux * f_i[4] + f_i[1] * (ux - 1.0) + f_i[3] * (ux + 1.0) + f_i[5] * (ux - 1.0) + f_i[6] * (ux + 1.0) + f_i[7] * (ux + 1.0) + f_i[8] * (ux - 1.0));
							Mstar_i[2] = -((3. * P * uy - rho * uy) / rho) * omega[2] + (omega[2] - 1.0) * (uy * f_i[0] + uy * f_i[1] + uy * f_i[3] + f_i[2] * (uy - 1.0) + f_i[4] * (uy + 1.0) + f_i[5] * (uy - 1.0) + f_i[6] * (uy - 1.0) + f_i[7] * (uy + 1.0) + f_i[8] * (uy + 1.0));
							Mstar_i[3] = ((3. * P * uy * ux - rho * uy * ux) / rho) * omega[3] - (omega[3] - 1.0) * (f_i[5] * (ux - 1.0) * (uy - 1.0) + f_i[6] * (ux + 1.0) * (uy - 1.0) + f_i[7] * (ux + 1.0) * (uy + 1.0) + f_i[8] * (ux - 1.0) * (uy + 1.0) + ux * uy * f_i[0] + uy * f_i[1] * (ux - 1.0) + ux * f_i[2] * (uy - 1.0) + uy * f_i[3] * (ux + 1.0) + ux * f_i[4] * (uy + 1.0));
							Mstar_i[4] = ((ux * ux - uy * uy) * (3. * P - rho) / rho) * omega[4] - (omega[4] - 1.0) * (f_i[1] * (pow(ux - 1.0, 2.0) - uy * uy) - f_i[2] * (pow(uy - 1.0, 2.0) - ux * ux) + f_i[3] * (pow(ux + 1.0, 2.0) - uy * uy) - f_i[4] * (pow(uy + 1.0, 2.0) - ux * ux) + f_i[5] * (pow(ux - 1.0, 2.0) - pow(uy - 1.0, 2.0)) + f_i[6] * (pow(ux + 1.0, 2.0) - pow(uy - 1.0, 2.0)) + f_i[7] * (pow(ux + 1.0, 2.0) - pow(uy + 1.0, 2.0)) + f_i[8] * (pow(ux - 1.0, 2.0) - pow(uy + 1.0, 2.0)) + f_i[0] * (ux * ux - uy * uy));
							Mstar_i[5] = ((ux * ux + uy * uy) * (3. * P - rho) / rho) * omega[5] - (omega[5] - 1.0) * (f_i[5] * (cs2 * -2.0 + pow(ux - 1.0, 2.0) + pow(uy - 1.0, 2.0)) + f_i[6] * (cs2 * -2.0 + pow(ux + 1.0, 2.0) + pow(uy - 1.0, 2.0)) + f_i[7] * (cs2 * -2.0 + pow(ux + 1.0, 2.0) + pow(uy + 1.0, 2.0)) + f_i[8] * (cs2 * -2.0 + pow(ux - 1.0, 2.0) + pow(uy + 1.0, 2.0)) + f_i[0] * (cs2 * -2.0 + ux * ux + uy * uy) + f_i[1] * (cs2 * -2.0 + pow(ux - 1.0, 2.0) + uy * uy) + f_i[2] * (cs2 * -2.0 + pow(uy - 1.0, 2.0) + ux * ux) + f_i[3] * (cs2 * -2.0 + pow(ux + 1.0, 2.0) + uy * uy) + f_i[4] * (cs2 * -2.0 + pow(uy + 1.0, 2.0) + ux * ux));
							Mstar_i[6] = -(3. * P * ux * ux * uy / rho) * omega[6] - (omega[6] - 1.0) * (uy * f_i[1] * (cs2 - pow(ux - 1.0, 2.0)) + f_i[2] * (uy - 1.0) * (cs2 - ux * ux) + uy * f_i[3] * (cs2 - pow(ux + 1.0, 2.0)) + f_i[4] * (uy + 1.0) * (cs2 - ux * ux) + f_i[5] * (cs2 - pow(ux - 1.0, 2.0)) * (uy - 1.0) + f_i[6] * (cs2 - pow(ux + 1.0, 2.0)) * (uy - 1.0) + f_i[7] * (cs2 - pow(ux + 1.0, 2.0)) * (uy + 1.0) + f_i[8] * (cs2 - pow(ux - 1.0, 2.0)) * (uy + 1.0) + uy * f_i[0] * (cs2 - ux * ux));
							Mstar_i[7] = -(3. * P * ux * uy * uy / rho) * omega[7] - (omega[7] - 1.0) * (f_i[1] * (ux - 1.0) * (cs2 - uy * uy) + ux * f_i[2] * (cs2 - pow(uy - 1.0, 2.0)) + f_i[3] * (ux + 1.0) * (cs2 - uy * uy) + ux * f_i[4] * (cs2 - pow(uy + 1.0, 2.0)) + f_i[5] * (cs2 - pow(uy - 1.0, 2.0)) * (ux - 1.0) + f_i[6] * (cs2 - pow(uy - 1.0, 2.0)) * (ux + 1.0) + f_i[7] * (cs2 - pow(uy + 1.0, 2.0)) * (ux + 1.0) + f_i[8] * (cs2 - pow(uy + 1.0, 2.0)) * (ux - 1.0) + ux * f_i[0] * (cs2 - uy * uy));
							Mstar_i[8] = (ux * ux * uy * uy * (3. * P + 2. * rho) / rho) * omega[8] - (omega[8] - 1.0) * (f_i[5] * (cs2 - pow(ux - 1.0, 2.0)) * (cs2 - pow(uy - 1.0, 2.0)) + f_i[6] * (cs2 - pow(ux + 1.0, 2.0)) * (cs2 - pow(uy - 1.0, 2.0)) + f_i[7] * (cs2 - pow(ux + 1.0, 2.0)) * (cs2 - pow(uy + 1.0, 2.0)) + f_i[8] * (cs2 - pow(ux - 1.0, 2.0)) * (cs2 - pow(uy + 1.0, 2.0)) + f_i[0] * (cs2 - ux * ux) * (cs2 - uy * uy) + f_i[1] * (cs2 - pow(ux - 1.0, 2.0)) * (cs2 - uy * uy) + f_i[2] * (cs2 - pow(uy - 1.0, 2.0)) * (cs2 - ux * ux) + f_i[3] * (cs2 - pow(ux + 1.0, 2.0)) * (cs2 - uy * uy) + f_i[4] * (cs2 - pow(uy + 1.0, 2.0)) * (cs2 - ux * ux));
#endif
#ifdef PERFORMANCE_MODE
							Mstar_i[0] = 1.;
							Mstar_i[1] = 0.;
							Mstar_i[2] = 0.;
							Mstar_i[3] = 0. * omega[3] - (omega[3] - 1.0) * (pop_old[{X, Y, Z, 5}] * (ux - 1.0) * (uy - 1.0) + pop_old[{X, Y, Z, 6}] * (ux + 1.0) * (uy - 1.0) + pop_old[{X, Y, Z, 7}] * (ux + 1.0) * (uy + 1.0) + pop_old[{X, Y, Z, 8}] * (ux - 1.0) * (uy + 1.0) + ux * uy * pop_old[{X, Y, Z, 0}] + uy * pop_old[{X, Y, Z, 1}] * (ux - 1.0) + ux * pop_old[{X, Y, Z, 2}] * (uy - 1.0) + uy * pop_old[{X, Y, Z, 3}] * (ux + 1.0) + ux * pop_old[{X, Y, Z, 4}] * (uy + 1.0));
							Mstar_i[4] = 0. * omega[4] - (omega[4] - 1.0) * (pop_old[{X, Y, Z, 1}] * (pow(ux - 1.0, 2.0) - uy * uy) - pop_old[{X, Y, Z, 2}] * (pow(uy - 1.0, 2.0) - ux * ux) + pop_old[{X, Y, Z, 3}] * (pow(ux + 1.0, 2.0) - uy * uy) - pop_old[{X, Y, Z, 4}] * (pow(uy + 1.0, 2.0) - ux * ux) + pop_old[{X, Y, Z, 5}] * (pow(ux - 1.0, 2.0) - pow(uy - 1.0, 2.0)) + pop_old[{X, Y, Z, 6}] * (pow(ux + 1.0, 2.0) - pow(uy - 1.0, 2.0)) + pop_old[{X, Y, Z, 7}] * (pow(ux + 1.0, 2.0) - pow(uy + 1.0, 2.0)) + pop_old[{X, Y, Z, 8}] * (pow(ux - 1.0, 2.0) - pow(uy + 1.0, 2.0)) + pop_old[{X, Y, Z, 0}] * (ux * ux - uy * uy));
							Mstar_i[5] = 0.;
							Mstar_i[6] = -ux * ux * uy;
							Mstar_i[7] = -ux * uy * uy;
							Mstar_i[8] = c_s2 * ux * ux * uy * uy;
#endif
							f_i[0] = Mstar_i[8] + ux * Mstar_i[7] * 2.0 + uy * Mstar_i[6] * 2.0 + Mstar_i[5] * (cs2 + (ux * ux) / 2.0 + (uy * uy) / 2.0 - 1.0) - (Mstar_i[4] * (ux * ux - uy * uy)) / 2.0 + ux * uy * Mstar_i[3] * 4.0 + ux * Mstar_i[1] * (cs2 + uy * uy - 1.0) * 2.0 + uy * Mstar_i[2] * (cs2 + ux * ux - 1.0) * 2.0 + Mstar_i[0] * (cs2 + ux * ux - 1.0) * (cs2 + uy * uy - 1.0);
							f_i[1] = Mstar_i[8] * (-1.0 / 2.0) + Mstar_i[4] * (ux / 4.0 + (ux * ux) / 4.0 - (uy * uy) / 4.0 + 1.0 / 4.0) - uy * Mstar_i[6] - Mstar_i[7] * (ux + 1.0 / 2.0) - Mstar_i[5] * (ux / 4.0 + cs2 / 2.0 + (ux * ux) / 4.0 + (uy * uy) / 4.0 - 1.0 / 4.0) - (Mstar_i[0] * (cs2 + uy * uy - 1.0) * (ux + cs2 + ux * ux)) / 2.0 - uy * Mstar_i[3] * (ux * 2.0 + 1.0) - (Mstar_i[1] * (ux * 2.0 + 1.0) * (cs2 + uy * uy - 1.0)) / 2.0 - uy * Mstar_i[2] * (ux + cs2 + ux * ux);
							f_i[2] = Mstar_i[8] * (-1.0 / 2.0) - Mstar_i[4] * (uy / 4.0 - (ux * ux) / 4.0 + (uy * uy) / 4.0 + 1.0 / 4.0) - ux * Mstar_i[7] - Mstar_i[6] * (uy + 1.0 / 2.0) - Mstar_i[5] * (uy / 4.0 + cs2 / 2.0 + (ux * ux) / 4.0 + (uy * uy) / 4.0 - 1.0 / 4.0) - (Mstar_i[0] * (cs2 + ux * ux - 1.0) * (uy + cs2 + uy * uy)) / 2.0 - ux * Mstar_i[3] * (uy * 2.0 + 1.0) - (Mstar_i[2] * (uy * 2.0 + 1.0) * (cs2 + ux * ux - 1.0)) / 2.0 - ux * Mstar_i[1] * (uy + cs2 + uy * uy);
							f_i[3] = Mstar_i[8] * (-1.0 / 2.0) - Mstar_i[4] * (ux / 4.0 - (ux * ux) / 4.0 + (uy * uy) / 4.0 - 1.0 / 4.0) - uy * Mstar_i[6] - Mstar_i[7] * (ux - 1.0 / 2.0) - Mstar_i[5] * (ux * (-1.0 / 4.0) + cs2 / 2.0 + (ux * ux) / 4.0 + (uy * uy) / 4.0 - 1.0 / 4.0) - uy * Mstar_i[3] * (ux * 2.0 - 1.0) - uy * Mstar_i[2] * (-ux + cs2 + ux * ux) - (Mstar_i[1] * (ux * 2.0 - 1.0) * (cs2 + uy * uy - 1.0)) / 2.0 - (Mstar_i[0] * (-ux + cs2 + ux * ux) * (cs2 + uy * uy - 1.0)) / 2.0;
							f_i[4] = Mstar_i[8] * (-1.0 / 2.0) + Mstar_i[4] * (uy / 4.0 + (ux * ux) / 4.0 - (uy * uy) / 4.0 - 1.0 / 4.0) - ux * Mstar_i[7] - Mstar_i[6] * (uy - 1.0 / 2.0) - Mstar_i[5] * (uy * (-1.0 / 4.0) + cs2 / 2.0 + (ux * ux) / 4.0 + (uy * uy) / 4.0 - 1.0 / 4.0) - ux * Mstar_i[3] * (uy * 2.0 - 1.0) - ux * Mstar_i[1] * (-uy + cs2 + uy * uy) - (Mstar_i[2] * (uy * 2.0 - 1.0) * (cs2 + ux * ux - 1.0)) / 2.0 - (Mstar_i[0] * (-uy + cs2 + uy * uy) * (cs2 + ux * ux - 1.0)) / 2.0;
							f_i[5] = Mstar_i[8] / 4.0 + Mstar_i[5] * (ux / 8.0 + uy / 8.0 + cs2 / 4.0 + (ux * ux) / 8.0 + (uy * uy) / 8.0) - Mstar_i[4] * (ux / 8.0 - uy / 8.0 + (ux * ux) / 8.0 - (uy * uy) / 8.0) + Mstar_i[7] * (ux / 2.0 + 1.0 / 4.0) + Mstar_i[6] * (uy / 2.0 + 1.0 / 4.0) + (Mstar_i[0] * (ux + cs2 + ux * ux) * (uy + cs2 + uy * uy)) / 4.0 + (Mstar_i[2] * (uy * 2.0 + 1.0) * (ux + cs2 + ux * ux)) / 4.0 + (Mstar_i[1] * (ux * 2.0 + 1.0) * (uy + cs2 + uy * uy)) / 4.0 + (Mstar_i[3] * (ux * 2.0 + 1.0) * (uy * 2.0 + 1.0)) / 4.0;
							f_i[6] = Mstar_i[8] / 4.0 + Mstar_i[5] * (ux * (-1.0 / 8.0) + uy / 8.0 + cs2 / 4.0 + (ux * ux) / 8.0 + (uy * uy) / 8.0) + Mstar_i[4] * (ux / 8.0 + uy / 8.0 - (ux * ux) / 8.0 + (uy * uy) / 8.0) + Mstar_i[7] * (ux / 2.0 - 1.0 / 4.0) + Mstar_i[6] * (uy / 2.0 + 1.0 / 4.0) + (Mstar_i[2] * (uy * 2.0 + 1.0) * (-ux + cs2 + ux * ux)) / 4.0 + (Mstar_i[1] * (ux * 2.0 - 1.0) * (uy + cs2 + uy * uy)) / 4.0 + (Mstar_i[0] * (-ux + cs2 + ux * ux) * (uy + cs2 + uy * uy)) / 4.0 + (Mstar_i[3] * (ux * 2.0 - 1.0) * (uy * 2.0 + 1.0)) / 4.0;
							f_i[7] = Mstar_i[8] / 4.0 + Mstar_i[5] * (ux * (-1.0 / 8.0) - uy / 8.0 + cs2 / 4.0 + (ux * ux) / 8.0 + (uy * uy) / 8.0) + Mstar_i[4] * (ux / 8.0 - uy / 8.0 - (ux * ux) / 8.0 + (uy * uy) / 8.0) + Mstar_i[7] * (ux / 2.0 - 1.0 / 4.0) + Mstar_i[6] * (uy / 2.0 - 1.0 / 4.0) + (Mstar_i[2] * (uy * 2.0 - 1.0) * (-ux + cs2 + ux * ux)) / 4.0 + (Mstar_i[1] * (ux * 2.0 - 1.0) * (-uy + cs2 + uy * uy)) / 4.0 + (Mstar_i[0] * (-ux + cs2 + ux * ux) * (-uy + cs2 + uy * uy)) / 4.0 + (Mstar_i[3] * (ux * 2.0 - 1.0) * (uy * 2.0 - 1.0)) / 4.0;
							f_i[8] = Mstar_i[8] / 4.0 + Mstar_i[5] * (ux / 8.0 - uy / 8.0 + cs2 / 4.0 + (ux * ux) / 8.0 + (uy * uy) / 8.0) - Mstar_i[4] * (ux / 8.0 + uy / 8.0 + (ux * ux) / 8.0 - (uy * uy) / 8.0) + Mstar_i[7] * (ux / 2.0 + 1.0 / 4.0) + Mstar_i[6] * (uy / 2.0 - 1.0 / 4.0) + (Mstar_i[1] * (ux * 2.0 + 1.0) * (-uy + cs2 + uy * uy)) / 4.0 + (Mstar_i[2] * (uy * 2.0 - 1.0) * (ux + cs2 + ux * ux)) / 4.0 + (Mstar_i[0] * (-uy + cs2 + uy * uy) * (ux + cs2 + ux * ux)) / 4.0 + (Mstar_i[3] * (ux * 2.0 + 1.0) * (uy * 2.0 - 1.0)) / 4.0;
							for (int alpha = 0; alpha < Discrete_Velocity; alpha++) {
								pop[{X + c_alpha[alpha][0], Y + c_alpha[alpha][1], Z + c_alpha[alpha][2], alpha}] = rho * cs2 * f_i[alpha] + 0.5 * forcingTerm[alpha];
							}
						}
					}
				}
			}
		}
	}
}
/// ***************************************************** ///
/// COLLISION AND STREAMING MODEL FOR CUMULANT LMNA   ///
/// ***************************************************** ///
/*
Collision Step:
In the lattice Boltzmann method, the collision step involves updating the distribution functions of particles on a lattice.
For the cumulant model, the distribution functions represent cumulants or moments of the probability distribution function.
The collision step typically uses a collision operator to relax the distribution functions towards local equilibrium.
In the provided function, this step involves updating distribution functions (pop_old) based on local fluid properties like density, velocity, and temperature.
*/
void Flow_solver::LBM_CUMULANT_LMNA(int time, Thermal_solver* Thermal, Species_solver* Species, Geometry* Geo, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		swap(pop_old, pop);
		int X, Y, Z, alpha;
		double theta;
		double forcingTerm[Discrete_Velocity];

		double c1o2 = 1.0 / 2.0;
		double c1o3 = 1.0 / 3.0;
		double c1o6 = 1.0 / 6.0;
		double c2o3 = 2.0 / 3.0;
		double c1o9 = 1.0 / 9.0;
		double c2o9 = 2.0 / 9.0;
		double c4o9 = 4.0 / 9.0;
		double c3o2 = 3.0 / 2.0;
		double c1o18 = 1.0 / 18.0;
		double c1o27 = 1.0 / 27.0;
		double c1o36 = 1.0 / 36.0;

#if defined compressible
		const double conv_factor = c_s2 * Thermal->T_0 / sqr(global_parameters.D_x / global_parameters.D_t);
#endif
		for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (is_solid[{X, Y, Z}] == FALSE) {
						theta = 1;
#if defined compressible
#if defined Flow_With_Species
						const double r = R_GAS / Species->molar_mass_av[{X, Y, Z}];
#else
						const double r = R_GAS / M_av;
#endif  // defined
						theta = r * Thermal->temperature[{X, Y, Z}] * conv_factor;
#endif
						/// Compute equilibrium

						double collFactorM = c_s2 * viscosity[{X, Y, Z}] / theta + 0.5;
						collFactorM = 1.0 / collFactorM;

						double mfbbb = pop_old[{X, Y, Z, 0}];   // REST
						double mfcbb = pop_old[{X, Y, Z, 1}];   // E
						double mfbcb = pop_old[{X, Y, Z, 3}];   // N
						double mfbbc = pop_old[{X, Y, Z, 5}];   // T
						double mfccb = pop_old[{X, Y, Z, 7}];   // NE
						double mfacb = pop_old[{X, Y, Z, 8}];   // NW
						double mfcbc = pop_old[{X, Y, Z, 11}];  // TE
						double mfabc = pop_old[{X, Y, Z, 12}];  // TW
						double mfbcc = pop_old[{X, Y, Z, 15}];  // TN
						double mfbac = pop_old[{X, Y, Z, 16}];  // TS
						double mfccc = pop_old[{X, Y, Z, 19}];  // TNE
						double mfacc = pop_old[{X, Y, Z, 20}];  // TNW
						double mfcac = pop_old[{X, Y, Z, 21}];  // TSE
						double mfaac = pop_old[{X, Y, Z, 22}];  // TSW
						double mfabb = pop_old[{X, Y, Z, 2}];   // W
						double mfbab = pop_old[{X, Y, Z, 4}];   // S
						double mfbba = pop_old[{X, Y, Z, 6}];   // B
						double mfaab = pop_old[{X, Y, Z, 10}];  // SW
						double mfcab = pop_old[{X, Y, Z, 9}];   // SE
						double mfaba = pop_old[{X, Y, Z, 14}];  // BW
						double mfcba = pop_old[{X, Y, Z, 13}];  // BE
						double mfbaa = pop_old[{X, Y, Z, 18}];  // BS
						double mfbca = pop_old[{X, Y, Z, 17}];  // BN
						double mfaaa = pop_old[{X, Y, Z, 26}];  // BSW
						double mfcaa = pop_old[{X, Y, Z, 25}];  // BSE
						double mfaca = pop_old[{X, Y, Z, 24}];  // BNW
						double mfcca = pop_old[{X, Y, Z, 23}];  // BNE

						double rho = density[{X, Y, Z}];

						double rho_x1p = density[{X + 1, Y, Z}];
						double rho_x1n = density[{X - 1, Y, Z}];
						double rho_x2p = density[{X + 2, Y, Z}];
						double rho_x2n = density[{X - 2, Y, Z}];

						double rho_y1p = density[{X, Y + 1, Z}];
						double rho_y1n = density[{X, Y - 1, Z}];
						double rho_y2p = density[{X, Y + 2, Z}];
						double rho_y2n = density[{X, Y - 2, Z}];

						double rho_z1p = density[{X, Y, Z + 1}];
						double rho_z1n = density[{X, Y, Z - 1}];
						double rho_z2p = density[{X, Y, Z + 2}];
						double rho_z2n = density[{X, Y, Z - 2}];

						double ux = velocity[{X, Y, Z, 0}];
						double uy = velocity[{X, Y, Z, 1}];
						double uz = velocity[{X, Y, Z, 2}];
						double divU_term = divU[{X, Y, Z}];
						// double divU_term = 0.0;

						double grad_rho0, grad_rho1, grad_rho2;

#if defined FD_CENTRAL || defined FD_UPWIND2 || defined FD_UPWIND
						grad_rho0 = FD::CENTRALNONCONS(1., rho_x1n, rho_x1p);
						grad_rho1 = FD::CENTRALNONCONS(1., rho_y1n, rho_y1p);
						grad_rho2 = FD::CENTRALNONCONS(1., rho_z1n, rho_z1p);
#endif  // defined
#if defined FD_CENTRAL4 || defined FD_WENO3
						grad_rho0 = FD::CENTRAL4NONCONS(1.0, rho_x2n, rho_x1n, rho, rho_x1p, rho_x2p);
						grad_rho1 = FD::CENTRAL4NONCONS(1.0, rho_y2n, rho_y1n, rho, rho_y1p, rho_y2p);
						grad_rho2 = FD::CENTRAL4NONCONS(1.0, rho_z2n, rho_z1n, rho, rho_z1p, rho_z2p);
						if (is_solid[{X + 1, Y, Z}] != -1 || is_solid[{X - 1, Y, Z}] != -1) {
							grad_rho0 = FD::CENTRALNONCONS(1.0, rho_x1n, rho_x1p);
						}
						if (is_solid[{X, Y + 1, Z}] != -1 || is_solid[{X, Y - 1, Z}] != -1) {
							grad_rho1 = FD::CENTRALNONCONS(1.0, rho_y1n, rho_y1p);
						}
						if (is_solid[{X, Y, Z + 1}] != -1 || is_solid[{X, Y, Z - 1}] != -1) {
							grad_rho2 = FD::CENTRALNONCONS(1.0, rho_z1n, rho_z1p);
						}
#endif  // defined
						double ux2 = ux * ux;
						double uy2 = uy * uy;
						double uz2 = uz * uz;

						double forcingX = rho * gravity[0];
						double forcingY = rho * gravity[1];
						double forcingZ = rho * gravity[2];

						for (alpha = 0; alpha < Discrete_Velocity; alpha++) {
							int CX = c_alpha[alpha][0];
							int CY = c_alpha[alpha][1];
							int CZ = c_alpha[alpha][2];

							double velProd = CX * ux + CY * uy + CZ * uz;
							double gamm = weight[alpha] * (1.0 + 3.0 * velProd + 4.5 * velProd * velProd - 1.5 * (ux2 + uy2 + uz2));
							double fac1 = (gamm - weight[alpha]) / 3.0;
							forcingTerm[alpha] = rho * c1o3 * divU_term * weight[alpha] + (((-ux) * (fac1 * grad_rho0 + gamm * forcingX) + (-uy) * (fac1 * grad_rho1 + gamm * forcingY) + (-uz) * (fac1 * grad_rho2 + gamm * forcingZ)) + (CX) * (fac1 * grad_rho0 + gamm * forcingX) + (CY) * (fac1 * grad_rho1 + gamm * forcingY) + (CZ) * (fac1 * grad_rho2 + gamm * forcingZ));
						}

						mfbbb = 3.0 * (mfbbb + 0.5 * forcingTerm[0]) / rho;   //- (3.0*p1 - rho)*WEIGTH[ZERO];
						mfcbb = 3.0 * (mfcbb + 0.5 * forcingTerm[1]) / rho;   //-(3.0*p1 - rho)*WEIGTH[E  ];
						mfbcb = 3.0 * (mfbcb + 0.5 * forcingTerm[3]) / rho;   //-(3.0*p1 - rho)*WEIGTH[N  ];
						mfbbc = 3.0 * (mfbbc + 0.5 * forcingTerm[5]) / rho;   //-(3.0*p1 - rho)*WEIGTH[T  ];
						mfccb = 3.0 * (mfccb + 0.5 * forcingTerm[7]) / rho;   //-(3.0*p1 - rho)*WEIGTH[NE ];
						mfacb = 3.0 * (mfacb + 0.5 * forcingTerm[8]) / rho;   //-(3.0*p1 - rho)*WEIGTH[NW ];
						mfcbc = 3.0 * (mfcbc + 0.5 * forcingTerm[11]) / rho;  //-(3.0*p1 - rho)*WEIGTH[TE ];
						mfabc = 3.0 * (mfabc + 0.5 * forcingTerm[12]) / rho;  //-(3.0*p1 - rho)*WEIGTH[TW ];
						mfbcc = 3.0 * (mfbcc + 0.5 * forcingTerm[15]) / rho;  //-(3.0*p1 - rho)*WEIGTH[TN ];
						mfbac = 3.0 * (mfbac + 0.5 * forcingTerm[16]) / rho;  //-(3.0*p1 - rho)*WEIGTH[TS ];
						mfccc = 3.0 * (mfccc + 0.5 * forcingTerm[19]) / rho;  //-(3.0*p1 - rho)*WEIGTH[TNE];
						mfacc = 3.0 * (mfacc + 0.5 * forcingTerm[20]) / rho;  //-(3.0*p1 - rho)*WEIGTH[TNW];
						mfcac = 3.0 * (mfcac + 0.5 * forcingTerm[21]) / rho;  //-(3.0*p1 - rho)*WEIGTH[TSE];
						mfaac = 3.0 * (mfaac + 0.5 * forcingTerm[22]) / rho;  //-(3.0*p1 - rho)*WEIGTH[TSW];
						mfabb = 3.0 * (mfabb + 0.5 * forcingTerm[2]) / rho;   //-(3.0*p1 - rho)*WEIGTH[W  ];
						mfbab = 3.0 * (mfbab + 0.5 * forcingTerm[4]) / rho;   //-(3.0*p1 - rho)*WEIGTH[S  ];
						mfbba = 3.0 * (mfbba + 0.5 * forcingTerm[6]) / rho;   //-(3.0*p1 - rho)*WEIGTH[B  ];
						mfaab = 3.0 * (mfaab + 0.5 * forcingTerm[10]) / rho;  //-(3.0*p1 - rho)*WEIGTH[SW ];
						mfcab = 3.0 * (mfcab + 0.5 * forcingTerm[9]) / rho;   //-(3.0*p1 - rho)*WEIGTH[SE ];
						mfaba = 3.0 * (mfaba + 0.5 * forcingTerm[14]) / rho;  //-(3.0*p1 - rho)*WEIGTH[BW ];
						mfcba = 3.0 * (mfcba + 0.5 * forcingTerm[13]) / rho;  //-(3.0*p1 - rho)*WEIGTH[BE ];
						mfbaa = 3.0 * (mfbaa + 0.5 * forcingTerm[18]) / rho;  //-(3.0*p1 - rho)*WEIGTH[BS ];
						mfbca = 3.0 * (mfbca + 0.5 * forcingTerm[17]) / rho;  //-(3.0*p1 - rho)*WEIGTH[BN ];
						mfaaa = 3.0 * (mfaaa + 0.5 * forcingTerm[26]) / rho;  //-(3.0*p1 - rho)*WEIGTH[BSW];
						mfcaa = 3.0 * (mfcaa + 0.5 * forcingTerm[25]) / rho;  //-(3.0*p1 - rho)*WEIGTH[BSE];
						mfaca = 3.0 * (mfaca + 0.5 * forcingTerm[24]) / rho;  //-(3.0*p1 - rho)*WEIGTH[BNW];
						mfcca = 3.0 * (mfcca + 0.5 * forcingTerm[23]) / rho;  //-(3.0*p1 - rho)*WEIGTH[BNE];

						double oMdrho, m0, m1, m2;

						oMdrho = mfccc + mfaaa;
						m0 = mfaca + mfcac;
						m1 = mfacc + mfcaa;
						m2 = mfaac + mfcca;
						oMdrho += m0;
						m1 += m2;
						oMdrho += m1;
						m0 = mfbac + mfbca;
						m1 = mfbaa + mfbcc;
						m0 += m1;
						m1 = mfabc + mfcba;
						m2 = mfaba + mfcbc;
						m1 += m2;
						m0 += m1;
						m1 = mfacb + mfcab;
						m2 = mfaab + mfccb;
						m1 += m2;
						m0 += m1;
						oMdrho += m0;
						m0 = mfabb + mfcbb;
						m1 = mfbab + mfbcb;
						m2 = mfbba + mfbbc;
						m0 += m1 + m2;
						m0 += mfbbb;  // hat gefehlt
						oMdrho = 1. - (oMdrho + m0);
						// oMdrho = rho - (oMdrho + m0);
						////////////////////////////////////////////////////////////////////////////////////
						double wadjust;
						double qudricLimit = 0.01;
						////////////////////////////////////////////////////////////////////////////////////
						// Forward Cumulant conversion
						////////////////////////////////////////////////////////////////////////////////////
						// with 1/36, 1/9, 1/36, 1/9, 4/9, 1/9, 1/36, 1/9, 1/36  conditioning
						////////////////////////////////////////////////////////////////////////////////////
						// Z - Dir
						m2 = mfaaa + mfaac;
						m1 = mfaac - mfaaa;
						m0 = m2 + mfaab;
						mfaaa = m0;
						m0 += c1o36 * oMdrho;
						mfaab = m1 - m0 * uz;
						mfaac = m2 - 2. * m1 * uz + uz2 * m0;
						////////////////////////////////////////////////////////////////////////////////////
						m2 = mfaba + mfabc;
						m1 = mfabc - mfaba;
						m0 = m2 + mfabb;
						mfaba = m0;
						m0 += c1o9 * oMdrho;
						mfabb = m1 - m0 * uz;
						mfabc = m2 - 2. * m1 * uz + uz2 * m0;
						////////////////////////////////////////////////////////////////////////////////////
						m2 = mfaca + mfacc;
						m1 = mfacc - mfaca;
						m0 = m2 + mfacb;
						mfaca = m0;
						m0 += c1o36 * oMdrho;
						mfacb = m1 - m0 * uz;
						mfacc = m2 - 2. * m1 * uz + uz2 * m0;
						////////////////////////////////////////////////////////////////////////////////////
						////////////////////////////////////////////////////////////////////////////////////
						m2 = mfbaa + mfbac;
						m1 = mfbac - mfbaa;
						m0 = m2 + mfbab;
						mfbaa = m0;
						m0 += c1o9 * oMdrho;
						mfbab = m1 - m0 * uz;
						mfbac = m2 - 2. * m1 * uz + uz2 * m0;
						////////////////////////////////////////////////////////////////////////////////////
						m2 = mfbba + mfbbc;
						m1 = mfbbc - mfbba;
						m0 = m2 + mfbbb;
						mfbba = m0;
						m0 += c4o9 * oMdrho;
						mfbbb = m1 - m0 * uz;
						mfbbc = m2 - 2. * m1 * uz + uz2 * m0;
						////////////////////////////////////////////////////////////////////////////////////
						m2 = mfbca + mfbcc;
						m1 = mfbcc - mfbca;
						m0 = m2 + mfbcb;
						mfbca = m0;
						m0 += c1o9 * oMdrho;
						mfbcb = m1 - m0 * uz;
						mfbcc = m2 - 2. * m1 * uz + uz2 * m0;
						////////////////////////////////////////////////////////////////////////////////////
						////////////////////////////////////////////////////////////////////////////////////
						m2 = mfcaa + mfcac;
						m1 = mfcac - mfcaa;
						m0 = m2 + mfcab;
						mfcaa = m0;
						m0 += c1o36 * oMdrho;
						mfcab = m1 - m0 * uz;
						mfcac = m2 - 2. * m1 * uz + uz2 * m0;
						////////////////////////////////////////////////////////////////////////////////////
						m2 = mfcba + mfcbc;
						m1 = mfcbc - mfcba;
						m0 = m2 + mfcbb;
						mfcba = m0;
						m0 += c1o9 * oMdrho;
						mfcbb = m1 - m0 * uz;
						mfcbc = m2 - 2. * m1 * uz + uz2 * m0;
						////////////////////////////////////////////////////////////////////////////////////
						m2 = mfcca + mfccc;
						m1 = mfccc - mfcca;
						m0 = m2 + mfccb;
						mfcca = m0;
						m0 += c1o36 * oMdrho;
						mfccb = m1 - m0 * uz;
						mfccc = m2 - 2. * m1 * uz + uz2 * m0;
						////////////////////////////////////////////////////////////////////////////////////
						////////////////////////////////////////////////////////////////////////////////////
						// with  1/6, 0, 1/18, 2/3, 0, 2/9, 1/6, 0, 1/18 conditioning
						////////////////////////////////////////////////////////////////////////////////////
						// Y - Dir
						m2 = mfaaa + mfaca;
						m1 = mfaca - mfaaa;
						m0 = m2 + mfaba;
						mfaaa = m0;
						m0 += c1o6 * oMdrho;
						mfaba = m1 - m0 * uy;
						mfaca = m2 - 2. * m1 * uy + uy2 * m0;
						////////////////////////////////////////////////////////////////////////////////////
						m2 = mfaab + mfacb;
						m1 = mfacb - mfaab;
						m0 = m2 + mfabb;
						mfaab = m0;
						mfabb = m1 - m0 * uy;
						mfacb = m2 - 2. * m1 * uy + uy2 * m0;
						////////////////////////////////////////////////////////////////////////////////////
						m2 = mfaac + mfacc;
						m1 = mfacc - mfaac;
						m0 = m2 + mfabc;
						mfaac = m0;
						m0 += c1o18 * oMdrho;
						mfabc = m1 - m0 * uy;
						mfacc = m2 - 2. * m1 * uy + uy2 * m0;
						////////////////////////////////////////////////////////////////////////////////////
						////////////////////////////////////////////////////////////////////////////////////
						m2 = mfbaa + mfbca;
						m1 = mfbca - mfbaa;
						m0 = m2 + mfbba;
						mfbaa = m0;
						m0 += c2o3 * oMdrho;
						mfbba = m1 - m0 * uy;
						mfbca = m2 - 2. * m1 * uy + uy2 * m0;
						////////////////////////////////////////////////////////////////////////////////////
						m2 = mfbab + mfbcb;
						m1 = mfbcb - mfbab;
						m0 = m2 + mfbbb;
						mfbab = m0;
						mfbbb = m1 - m0 * uy;
						mfbcb = m2 - 2. * m1 * uy + uy2 * m0;
						////////////////////////////////////////////////////////////////////////////////////
						m2 = mfbac + mfbcc;
						m1 = mfbcc - mfbac;
						m0 = m2 + mfbbc;
						mfbac = m0;
						m0 += c2o9 * oMdrho;
						mfbbc = m1 - m0 * uy;
						mfbcc = m2 - 2. * m1 * uy + uy2 * m0;
						////////////////////////////////////////////////////////////////////////////////////
						////////////////////////////////////////////////////////////////////////////////////
						m2 = mfcaa + mfcca;
						m1 = mfcca - mfcaa;
						m0 = m2 + mfcba;
						mfcaa = m0;
						m0 += c1o6 * oMdrho;
						mfcba = m1 - m0 * uy;
						mfcca = m2 - 2. * m1 * uy + uy2 * m0;
						////////////////////////////////////////////////////////////////////////////////////
						m2 = mfcab + mfccb;
						m1 = mfccb - mfcab;
						m0 = m2 + mfcbb;
						mfcab = m0;
						mfcbb = m1 - m0 * uy;
						mfccb = m2 - 2. * m1 * uy + uy2 * m0;
						////////////////////////////////////////////////////////////////////////////////////
						m2 = mfcac + mfccc;
						m1 = mfccc - mfcac;
						m0 = m2 + mfcbc;
						mfcac = m0;
						m0 += c1o18 * oMdrho;
						mfcbc = m1 - m0 * uy;
						mfccc = m2 - 2. * m1 * uy + uy2 * m0;
						////////////////////////////////////////////////////////////////////////////////////
						////////////////////////////////////////////////////////////////////////////////////
						// with     1, 0, 1/3, 0, 0, 0, 1/3, 0, 1/9            conditioning
						////////////////////////////////////////////////////////////////////////////////////
						// X - Dir
						m2 = mfaaa + mfcaa;
						m1 = mfcaa - mfaaa;
						m0 = m2 + mfbaa;
						mfaaa = m0;
						m0 += 1. * oMdrho;
						mfbaa = m1 - m0 * ux;
						mfcaa = m2 - 2. * m1 * ux + ux2 * m0;
						////////////////////////////////////////////////////////////////////////////////////
						m2 = mfaba + mfcba;
						m1 = mfcba - mfaba;
						m0 = m2 + mfbba;
						mfaba = m0;
						mfbba = m1 - m0 * ux;
						mfcba = m2 - 2. * m1 * ux + ux2 * m0;
						////////////////////////////////////////////////////////////////////////////////////
						m2 = mfaca + mfcca;
						m1 = mfcca - mfaca;
						m0 = m2 + mfbca;
						mfaca = m0;
						m0 += c1o3 * oMdrho;
						mfbca = m1 - m0 * ux;
						mfcca = m2 - 2. * m1 * ux + ux2 * m0;
						////////////////////////////////////////////////////////////////////////////////////
						////////////////////////////////////////////////////////////////////////////////////
						m2 = mfaab + mfcab;
						m1 = mfcab - mfaab;
						m0 = m2 + mfbab;
						mfaab = m0;
						mfbab = m1 - m0 * ux;
						mfcab = m2 - 2. * m1 * ux + ux2 * m0;
						////////////////////////////////////////////////////////////////////////////////////
						m2 = mfabb + mfcbb;
						m1 = mfcbb - mfabb;
						m0 = m2 + mfbbb;
						mfabb = m0;
						mfbbb = m1 - m0 * ux;
						mfcbb = m2 - 2. * m1 * ux + ux2 * m0;
						////////////////////////////////////////////////////////////////////////////////////
						m2 = mfacb + mfccb;
						m1 = mfccb - mfacb;
						m0 = m2 + mfbcb;
						mfacb = m0;
						mfbcb = m1 - m0 * ux;
						mfccb = m2 - 2. * m1 * ux + ux2 * m0;
						////////////////////////////////////////////////////////////////////////////////////
						////////////////////////////////////////////////////////////////////////////////////
						m2 = mfaac + mfcac;
						m1 = mfcac - mfaac;
						m0 = m2 + mfbac;
						mfaac = m0;
						m0 += c1o3 * oMdrho;
						mfbac = m1 - m0 * ux;
						mfcac = m2 - 2. * m1 * ux + ux2 * m0;
						////////////////////////////////////////////////////////////////////////////////////
						m2 = mfabc + mfcbc;
						m1 = mfcbc - mfabc;
						m0 = m2 + mfbbc;
						mfabc = m0;
						mfbbc = m1 - m0 * ux;
						mfcbc = m2 - 2. * m1 * ux + ux2 * m0;
						////////////////////////////////////////////////////////////////////////////////////
						m2 = mfacc + mfccc;
						m1 = mfccc - mfacc;
						m0 = m2 + mfbcc;
						mfacc = m0;
						m0 += c1o9 * oMdrho;
						mfbcc = m1 - m0 * ux;
						mfccc = m2 - 2. * m1 * ux + ux2 * m0;

						////////////////////////////////////////////////////////////////////////////////////
						// Cumulants
						////////////////////////////////////////////////////////////////////////////////////
						double OxxPyyPzz = 1.;  // omega2 or bulk viscosity
						double OxyyPxzz = 1.;   //-s9;//2+s9;//
						double OxyyMxzz = 1.;   // according to normal.... for magic OxyyMxzz = 2.0 +(-collFactor)
						double O4 = 1.;
						double O5 = 1.;
						double O6 = 1.;

						// Cum 4.
						// double CUMcbb = mfcbb - ((mfcaa + c1o3 * oMdrho) * mfabb + 2. * mfbba * mfbab); // till 18.05.2015
						// double CUMbcb = mfbcb - ((mfaca + c1o3 * oMdrho) * mfbab + 2. * mfbba * mfabb); // till 18.05.2015
						// double CUMbbc = mfbbc - ((mfaac + c1o3 * oMdrho) * mfbba + 2. * mfbab * mfabb); // till 18.05.2015

						double CUMcbb = mfcbb - ((mfcaa + c1o3) * mfabb + 2. * mfbba * mfbab);
						double CUMbcb = mfbcb - ((mfaca + c1o3) * mfbab + 2. * mfbba * mfabb);
						double CUMbbc = mfbbc - ((mfaac + c1o3) * mfbba + 2. * mfbab * mfabb);

						double CUMcca = mfcca - ((mfcaa * mfaca + 2. * mfbba * mfbba) + c1o3 * (mfcaa + mfaca) * oMdrho + c1o9 * (oMdrho - 1) * oMdrho);
						double CUMcac = mfcac - ((mfcaa * mfaac + 2. * mfbab * mfbab) + c1o3 * (mfcaa + mfaac) * oMdrho + c1o9 * (oMdrho - 1) * oMdrho);
						double CUMacc = mfacc - ((mfaac * mfaca + 2. * mfabb * mfabb) + c1o3 * (mfaac + mfaca) * oMdrho + c1o9 * (oMdrho - 1) * oMdrho);

						// double CUMcca = mfcca - ((mfcaa * mfaca + 2. * mfbba * mfbba) + c1o3 * (mfcaa + mfaca) * oMdrho + c1o9*(-p1/c1o3)*oMdrho);
						// double CUMcac = mfcac - ((mfcaa * mfaac + 2. * mfbab * mfbab) + c1o3 * (mfcaa + mfaac) * oMdrho + c1o9*(-p1/c1o3)*oMdrho);
						// double CUMacc = mfacc - ((mfaac * mfaca + 2. * mfabb * mfabb) + c1o3 * (mfaac + mfaca) * oMdrho + c1o9*(-p1/c1o3)*oMdrho);

						// Cum 5.
						double CUMbcc = mfbcc - (mfaac * mfbca + mfaca * mfbac + 4. * mfabb * mfbbb + 2. * (mfbab * mfacb + mfbba * mfabc)) - c1o3 * (mfbca + mfbac) * oMdrho;
						double CUMcbc = mfcbc - (mfaac * mfcba + mfcaa * mfabc + 4. * mfbab * mfbbb + 2. * (mfabb * mfcab + mfbba * mfbac)) - c1o3 * (mfcba + mfabc) * oMdrho;
						double CUMccb = mfccb - (mfcaa * mfacb + mfaca * mfcab + 4. * mfbba * mfbbb + 2. * (mfbab * mfbca + mfabb * mfcba)) - c1o3 * (mfacb + mfcab) * oMdrho;

						// Cum 6.
						double CUMccc = mfccc + ((-4. * mfbbb * mfbbb - (mfcaa * mfacc + mfaca * mfcac + mfaac * mfcca) - 4. * (mfabb * mfcbb + mfbab * mfbcb + mfbba * mfbbc) - 2. * (mfbca * mfbac + mfcba * mfabc + mfcab * mfacb)) + (4. * (mfbab * mfbab * mfaca + mfabb * mfabb * mfcaa + mfbba * mfbba * mfaac) + 2. * (mfcaa * mfaca * mfaac) + 16. * mfbba * mfbab * mfabb) - c1o3 * (mfacc + mfcac + mfcca) * oMdrho - c1o9 * oMdrho * oMdrho - c1o9 * (mfcaa + mfaca + mfaac) * oMdrho * (1. - 2. * oMdrho) - c1o27 * oMdrho * oMdrho * (-2. * oMdrho) + (2. * (mfbab * mfbab + mfabb * mfabb + mfbba * mfbba) + (mfaac * mfaca + mfaac * mfcaa + mfaca * mfcaa)) * c2o3 * oMdrho) + c1o27 * oMdrho;

						// 2.
						//  linear combinations
						double mxxPyyPzz = mfcaa + mfaca + mfaac;
						double mxxMyy = mfcaa - mfaca;
						double mxxMzz = mfcaa - mfaac;

						double dxux = -c1o2 * collFactorM * (mxxMyy + mxxMzz) + c1o2 * OxxPyyPzz * (mfaaa - mxxPyyPzz);
						double dyuy = dxux + collFactorM * c3o2 * mxxMyy;
						double dzuz = dxux + collFactorM * c3o2 * mxxMzz;

						// relax
						mxxPyyPzz += OxxPyyPzz * (mfaaa - mxxPyyPzz) - 3. * (1. - c1o2 * OxxPyyPzz) * (ux2 * dxux + uy2 * dyuy + uz2 * dzuz);
						mxxMyy += collFactorM * (-mxxMyy) - 3. * (1. - c1o2 * collFactorM) * (ux2 * dxux - uy2 * dyuy);
						mxxMzz += collFactorM * (-mxxMzz) - 3. * (1. - c1o2 * collFactorM) * (ux2 * dxux - uz2 * dzuz);

						mfabb += collFactorM * (-mfabb);
						mfbab += collFactorM * (-mfbab);
						mfbba += collFactorM * (-mfbba);

						// linear combinations back
						mfcaa = c1o3 * (mxxMyy + mxxMzz + mxxPyyPzz);
						mfaca = c1o3 * (-2. * mxxMyy + mxxMzz + mxxPyyPzz);
						mfaac = c1o3 * (mxxMyy - 2. * mxxMzz + mxxPyyPzz);

						// 3.
						//  linear combinations
						double mxxyPyzz = mfcba + mfabc;
						double mxxyMyzz = mfcba - mfabc;

						double mxxzPyyz = mfcab + mfacb;
						double mxxzMyyz = mfcab - mfacb;

						double mxyyPxzz = mfbca + mfbac;
						double mxyyMxzz = mfbca - mfbac;

						// relax
						wadjust = OxyyMxzz + (1. - OxyyMxzz) * fabs(mfbbb) / (fabs(mfbbb) + qudricLimit);
						mfbbb += wadjust * (-mfbbb);
						wadjust = OxyyPxzz + (1. - OxyyPxzz) * fabs(mxxyPyzz) / (fabs(mxxyPyzz) + qudricLimit);
						mxxyPyzz += wadjust * (-mxxyPyzz);
						wadjust = OxyyMxzz + (1. - OxyyMxzz) * fabs(mxxyMyzz) / (fabs(mxxyMyzz) + qudricLimit);
						mxxyMyzz += wadjust * (-mxxyMyzz);
						wadjust = OxyyPxzz + (1. - OxyyPxzz) * fabs(mxxzPyyz) / (fabs(mxxzPyyz) + qudricLimit);
						mxxzPyyz += wadjust * (-mxxzPyyz);
						wadjust = OxyyMxzz + (1. - OxyyMxzz) * fabs(mxxzMyyz) / (fabs(mxxzMyyz) + qudricLimit);
						mxxzMyyz += wadjust * (-mxxzMyyz);
						wadjust = OxyyPxzz + (1. - OxyyPxzz) * fabs(mxyyPxzz) / (fabs(mxyyPxzz) + qudricLimit);
						mxyyPxzz += wadjust * (-mxyyPxzz);
						wadjust = OxyyMxzz + (1. - OxyyMxzz) * fabs(mxyyMxzz) / (fabs(mxyyMxzz) + qudricLimit);
						mxyyMxzz += wadjust * (-mxyyMxzz);

						// linear combinations back
						mfcba = (mxxyMyzz + mxxyPyzz) * c1o2;
						mfabc = (-mxxyMyzz + mxxyPyzz) * c1o2;
						mfcab = (mxxzMyyz + mxxzPyyz) * c1o2;
						mfacb = (-mxxzMyyz + mxxzPyyz) * c1o2;
						mfbca = (mxyyMxzz + mxyyPxzz) * c1o2;
						mfbac = (-mxyyMxzz + mxyyPxzz) * c1o2;

						// 4.
						CUMacc += O4 * (-CUMacc);
						CUMcac += O4 * (-CUMcac);
						CUMcca += O4 * (-CUMcca);

						CUMbbc += O4 * (-CUMbbc);
						CUMbcb += O4 * (-CUMbcb);
						CUMcbb += O4 * (-CUMcbb);

						// 5.
						CUMbcc += O5 * (-CUMbcc);
						CUMcbc += O5 * (-CUMcbc);
						CUMccb += O5 * (-CUMccb);

						// 6.
						CUMccc += O6 * (-CUMccc);

						// back cumulants to central moments
						// 4.
						// mfcbb = CUMcbb + ((mfcaa + c1o3 * oMdrho) * mfabb + 2. * mfbba * mfbab); // till 18.05.2015
						// mfbcb = CUMbcb + ((mfaca + c1o3 * oMdrho) * mfbab + 2. * mfbba * mfabb); // till 18.05.2015
						// mfbbc = CUMbbc + ((mfaac + c1o3 * oMdrho) * mfbba + 2. * mfbab * mfabb); // till 18.05.2015

						mfcbb = CUMcbb + ((mfcaa + c1o3) * mfabb + 2. * mfbba * mfbab);
						mfbcb = CUMbcb + ((mfaca + c1o3) * mfbab + 2. * mfbba * mfabb);
						mfbbc = CUMbbc + ((mfaac + c1o3) * mfbba + 2. * mfbab * mfabb);

						mfcca = CUMcca + (mfcaa * mfaca + 2. * mfbba * mfbba) + c1o3 * (mfcaa + mfaca) * oMdrho + c1o9 * (oMdrho - 1) * oMdrho;
						mfcac = CUMcac + (mfcaa * mfaac + 2. * mfbab * mfbab) + c1o3 * (mfcaa + mfaac) * oMdrho + c1o9 * (oMdrho - 1) * oMdrho;
						mfacc = CUMacc + (mfaac * mfaca + 2. * mfabb * mfabb) + c1o3 * (mfaac + mfaca) * oMdrho + c1o9 * (oMdrho - 1) * oMdrho;

						// mfcca = CUMcca + (mfcaa * mfaca + 2. * mfbba * mfbba) + c1o3 * (mfcaa + mfaca) * oMdrho + c1o9*(-p1/c1o3)*oMdrho;
						// mfcac = CUMcac + (mfcaa * mfaac + 2. * mfbab * mfbab) + c1o3 * (mfcaa + mfaac) * oMdrho + c1o9*(-p1/c1o3)*oMdrho;
						// mfacc = CUMacc + (mfaac * mfaca + 2. * mfabb * mfabb) + c1o3 * (mfaac + mfaca) * oMdrho + c1o9*(-p1/c1o3)*oMdrho;

						// 5.
						mfbcc = CUMbcc + (mfaac * mfbca + mfaca * mfbac + 4. * mfabb * mfbbb + 2. * (mfbab * mfacb + mfbba * mfabc)) + c1o3 * (mfbca + mfbac) * oMdrho;
						mfcbc = CUMcbc + (mfaac * mfcba + mfcaa * mfabc + 4. * mfbab * mfbbb + 2. * (mfabb * mfcab + mfbba * mfbac)) + c1o3 * (mfcba + mfabc) * oMdrho;
						mfccb = CUMccb + (mfcaa * mfacb + mfaca * mfcab + 4. * mfbba * mfbbb + 2. * (mfbab * mfbca + mfabb * mfcba)) + c1o3 * (mfacb + mfcab) * oMdrho;

						// 6.
						mfccc = CUMccc - ((-4. * mfbbb * mfbbb - (mfcaa * mfacc + mfaca * mfcac + mfaac * mfcca) - 4. * (mfabb * mfcbb + mfbac * mfbca + mfbba * mfbbc) - 2. * (mfbca * mfbac + mfcba * mfabc + mfcab * mfacb)) + (4. * (mfbab * mfbab * mfaca + mfabb * mfabb * mfcaa + mfbba * mfbba * mfaac) + 2. * (mfcaa * mfaca * mfaac) + 16. * mfbba * mfbab * mfabb) - c1o3 * (mfacc + mfcac + mfcca) * oMdrho - c1o9 * oMdrho * oMdrho - c1o9 * (mfcaa + mfaca + mfaac) * oMdrho * (1. - 2. * oMdrho) - c1o27 * oMdrho * oMdrho * (-2. * oMdrho) + (2. * (mfbab * mfbab + mfabb * mfabb + mfbba * mfbba) + (mfaac * mfaca + mfaac * mfcaa + mfaca * mfcaa)) * c2o3 * oMdrho) - c1o27 * oMdrho;

						////////////////////////////////////////////////////////////////////////////////////
						// forcing
						// mfbaa=-mfbaa;
						// mfaba=-mfaba;
						// mfaab=-mfaab;
						//////////////////////////////////////////////////////////////////////////////////////

						////////////////////////////////////////////////////////////////////////////////////
						// back
						////////////////////////////////////////////////////////////////////////////////////
						// with 1, 0, 1/3, 0, 0, 0, 1/3, 0, 1/9   conditioning
						////////////////////////////////////////////////////////////////////////////////////
						// Z - Dir
						m0 = mfaac * c1o2 + mfaab * (uz - c1o2) + (mfaaa + 1. * oMdrho) * (uz2 - uz) * c1o2;
						m1 = -mfaac - 2. * mfaab * uz + mfaaa * (1. - uz2) - 1. * oMdrho * uz2;
						m2 = mfaac * c1o2 + mfaab * (uz + c1o2) + (mfaaa + 1. * oMdrho) * (uz2 + uz) * c1o2;
						mfaaa = m0;
						mfaab = m1;
						mfaac = m2;
						////////////////////////////////////////////////////////////////////////////////////
						m0 = mfabc * c1o2 + mfabb * (uz - c1o2) + mfaba * (uz2 - uz) * c1o2;
						m1 = -mfabc - 2. * mfabb * uz + mfaba * (1. - uz2);
						m2 = mfabc * c1o2 + mfabb * (uz + c1o2) + mfaba * (uz2 + uz) * c1o2;
						mfaba = m0;
						mfabb = m1;
						mfabc = m2;
						////////////////////////////////////////////////////////////////////////////////////
						m0 = mfacc * c1o2 + mfacb * (uz - c1o2) + (mfaca + c1o3 * oMdrho) * (uz2 - uz) * c1o2;
						m1 = -mfacc - 2. * mfacb * uz + mfaca * (1. - uz2) - c1o3 * oMdrho * uz2;
						m2 = mfacc * c1o2 + mfacb * (uz + c1o2) + (mfaca + c1o3 * oMdrho) * (uz2 + uz) * c1o2;
						mfaca = m0;
						mfacb = m1;
						mfacc = m2;
						////////////////////////////////////////////////////////////////////////////////////
						////////////////////////////////////////////////////////////////////////////////////
						m0 = mfbac * c1o2 + mfbab * (uz - c1o2) + mfbaa * (uz2 - uz) * c1o2;
						m1 = -mfbac - 2. * mfbab * uz + mfbaa * (1. - uz2);
						m2 = mfbac * c1o2 + mfbab * (uz + c1o2) + mfbaa * (uz2 + uz) * c1o2;
						mfbaa = m0;
						mfbab = m1;
						mfbac = m2;
						/////////b//////////////////////////////////////////////////////////////////////////
						m0 = mfbbc * c1o2 + mfbbb * (uz - c1o2) + mfbba * (uz2 - uz) * c1o2;
						m1 = -mfbbc - 2. * mfbbb * uz + mfbba * (1. - uz2);
						m2 = mfbbc * c1o2 + mfbbb * (uz + c1o2) + mfbba * (uz2 + uz) * c1o2;
						mfbba = m0;
						mfbbb = m1;
						mfbbc = m2;
						/////////b//////////////////////////////////////////////////////////////////////////
						m0 = mfbcc * c1o2 + mfbcb * (uz - c1o2) + mfbca * (uz2 - uz) * c1o2;
						m1 = -mfbcc - 2. * mfbcb * uz + mfbca * (1. - uz2);
						m2 = mfbcc * c1o2 + mfbcb * (uz + c1o2) + mfbca * (uz2 + uz) * c1o2;
						mfbca = m0;
						mfbcb = m1;
						mfbcc = m2;
						////////////////////////////////////////////////////////////////////////////////////
						////////////////////////////////////////////////////////////////////////////////////
						m0 = mfcac * c1o2 + mfcab * (uz - c1o2) + (mfcaa + c1o3 * oMdrho) * (uz2 - uz) * c1o2;
						m1 = -mfcac - 2. * mfcab * uz + mfcaa * (1. - uz2) - c1o3 * oMdrho * uz2;
						m2 = mfcac * c1o2 + mfcab * (uz + c1o2) + (mfcaa + c1o3 * oMdrho) * (uz2 + uz) * c1o2;
						mfcaa = m0;
						mfcab = m1;
						mfcac = m2;
						/////////c//////////////////////////////////////////////////////////////////////////
						m0 = mfcbc * c1o2 + mfcbb * (uz - c1o2) + mfcba * (uz2 - uz) * c1o2;
						m1 = -mfcbc - 2. * mfcbb * uz + mfcba * (1. - uz2);
						m2 = mfcbc * c1o2 + mfcbb * (uz + c1o2) + mfcba * (uz2 + uz) * c1o2;
						mfcba = m0;
						mfcbb = m1;
						mfcbc = m2;
						/////////c//////////////////////////////////////////////////////////////////////////
						m0 = mfccc * c1o2 + mfccb * (uz - c1o2) + (mfcca + c1o9 * oMdrho) * (uz2 - uz) * c1o2;
						m1 = -mfccc - 2. * mfccb * uz + mfcca * (1. - uz2) - c1o9 * oMdrho * uz2;
						m2 = mfccc * c1o2 + mfccb * (uz + c1o2) + (mfcca + c1o9 * oMdrho) * (uz2 + uz) * c1o2;
						mfcca = m0;
						mfccb = m1;
						mfccc = m2;
						////////////////////////////////////////////////////////////////////////////////////
						////////////////////////////////////////////////////////////////////////////////////
						// with 1/6, 2/3, 1/6, 0, 0, 0, 1/18, 2/9, 1/18   conditioning
						////////////////////////////////////////////////////////////////////////////////////
						// Y - Dir
						m0 = mfaca * c1o2 + mfaba * (uy - c1o2) + (mfaaa + c1o6 * oMdrho) * (uy2 - uy) * c1o2;
						m1 = -mfaca - 2. * mfaba * uy + mfaaa * (1. - uy2) - c1o6 * oMdrho * uy2;
						m2 = mfaca * c1o2 + mfaba * (uy + c1o2) + (mfaaa + c1o6 * oMdrho) * (uy2 + uy) * c1o2;
						mfaaa = m0;
						mfaba = m1;
						mfaca = m2;
						////////////////////////////////////////////////////////////////////////////////////
						m0 = mfacb * c1o2 + mfabb * (uy - c1o2) + (mfaab + c2o3 * oMdrho) * (uy2 - uy) * c1o2;
						m1 = -mfacb - 2. * mfabb * uy + mfaab * (1. - uy2) - c2o3 * oMdrho * uy2;
						m2 = mfacb * c1o2 + mfabb * (uy + c1o2) + (mfaab + c2o3 * oMdrho) * (uy2 + uy) * c1o2;
						mfaab = m0;
						mfabb = m1;
						mfacb = m2;
						////////////////////////////////////////////////////////////////////////////////////
						m0 = mfacc * c1o2 + mfabc * (uy - c1o2) + (mfaac + c1o6 * oMdrho) * (uy2 - uy) * c1o2;
						m1 = -mfacc - 2. * mfabc * uy + mfaac * (1. - uy2) - c1o6 * oMdrho * uy2;
						m2 = mfacc * c1o2 + mfabc * (uy + c1o2) + (mfaac + c1o6 * oMdrho) * (uy2 + uy) * c1o2;
						mfaac = m0;
						mfabc = m1;
						mfacc = m2;
						////////////////////////////////////////////////////////////////////////////////////
						////////////////////////////////////////////////////////////////////////////////////
						m0 = mfbca * c1o2 + mfbba * (uy - c1o2) + mfbaa * (uy2 - uy) * c1o2;
						m1 = -mfbca - 2. * mfbba * uy + mfbaa * (1. - uy2);
						m2 = mfbca * c1o2 + mfbba * (uy + c1o2) + mfbaa * (uy2 + uy) * c1o2;
						mfbaa = m0;
						mfbba = m1;
						mfbca = m2;
						/////////b//////////////////////////////////////////////////////////////////////////
						m0 = mfbcb * c1o2 + mfbbb * (uy - c1o2) + mfbab * (uy2 - uy) * c1o2;
						m1 = -mfbcb - 2. * mfbbb * uy + mfbab * (1. - uy2);
						m2 = mfbcb * c1o2 + mfbbb * (uy + c1o2) + mfbab * (uy2 + uy) * c1o2;
						mfbab = m0;
						mfbbb = m1;
						mfbcb = m2;
						/////////b//////////////////////////////////////////////////////////////////////////
						m0 = mfbcc * c1o2 + mfbbc * (uy - c1o2) + mfbac * (uy2 - uy) * c1o2;
						m1 = -mfbcc - 2. * mfbbc * uy + mfbac * (1. - uy2);
						m2 = mfbcc * c1o2 + mfbbc * (uy + c1o2) + mfbac * (uy2 + uy) * c1o2;
						mfbac = m0;
						mfbbc = m1;
						mfbcc = m2;
						////////////////////////////////////////////////////////////////////////////////////
						////////////////////////////////////////////////////////////////////////////////////
						m0 = mfcca * c1o2 + mfcba * (uy - c1o2) + (mfcaa + c1o18 * oMdrho) * (uy2 - uy) * c1o2;
						m1 = -mfcca - 2. * mfcba * uy + mfcaa * (1. - uy2) - c1o18 * oMdrho * uy2;
						m2 = mfcca * c1o2 + mfcba * (uy + c1o2) + (mfcaa + c1o18 * oMdrho) * (uy2 + uy) * c1o2;
						mfcaa = m0;
						mfcba = m1;
						mfcca = m2;
						/////////c//////////////////////////////////////////////////////////////////////////
						m0 = mfccb * c1o2 + mfcbb * (uy - c1o2) + (mfcab + c2o9 * oMdrho) * (uy2 - uy) * c1o2;
						m1 = -mfccb - 2. * mfcbb * uy + mfcab * (1. - uy2) - c2o9 * oMdrho * uy2;
						m2 = mfccb * c1o2 + mfcbb * (uy + c1o2) + (mfcab + c2o9 * oMdrho) * (uy2 + uy) * c1o2;
						mfcab = m0;
						mfcbb = m1;
						mfccb = m2;
						/////////c//////////////////////////////////////////////////////////////////////////
						m0 = mfccc * c1o2 + mfcbc * (uy - c1o2) + (mfcac + c1o18 * oMdrho) * (uy2 - uy) * c1o2;
						m1 = -mfccc - 2. * mfcbc * uy + mfcac * (1. - uy2) - c1o18 * oMdrho * uy2;
						m2 = mfccc * c1o2 + mfcbc * (uy + c1o2) + (mfcac + c1o18 * oMdrho) * (uy2 + uy) * c1o2;
						mfcac = m0;
						mfcbc = m1;
						mfccc = m2;
						////////////////////////////////////////////////////////////////////////////////////
						////////////////////////////////////////////////////////////////////////////////////
						// with 1/36, 1/9, 1/36, 1/9, 4/9, 1/9, 1/36, 1/9, 1/36 conditioning
						////////////////////////////////////////////////////////////////////////////////////
						// X - Dir
						m0 = mfcaa * c1o2 + mfbaa * (ux - c1o2) + (mfaaa + c1o36 * oMdrho) * (ux2 - ux) * c1o2;
						m1 = -mfcaa - 2. * mfbaa * ux + mfaaa * (1. - ux2) - c1o36 * oMdrho * ux2;
						m2 = mfcaa * c1o2 + mfbaa * (ux + c1o2) + (mfaaa + c1o36 * oMdrho) * (ux2 + ux) * c1o2;
						mfaaa = m0;
						mfbaa = m1;
						mfcaa = m2;
						////////////////////////////////////////////////////////////////////////////////////
						m0 = mfcba * c1o2 + mfbba * (ux - c1o2) + (mfaba + c1o9 * oMdrho) * (ux2 - ux) * c1o2;
						m1 = -mfcba - 2. * mfbba * ux + mfaba * (1. - ux2) - c1o9 * oMdrho * ux2;
						m2 = mfcba * c1o2 + mfbba * (ux + c1o2) + (mfaba + c1o9 * oMdrho) * (ux2 + ux) * c1o2;
						mfaba = m0;
						mfbba = m1;
						mfcba = m2;
						////////////////////////////////////////////////////////////////////////////////////
						m0 = mfcca * c1o2 + mfbca * (ux - c1o2) + (mfaca + c1o36 * oMdrho) * (ux2 - ux) * c1o2;
						m1 = -mfcca - 2. * mfbca * ux + mfaca * (1. - ux2) - c1o36 * oMdrho * ux2;
						m2 = mfcca * c1o2 + mfbca * (ux + c1o2) + (mfaca + c1o36 * oMdrho) * (ux2 + ux) * c1o2;
						mfaca = m0;
						mfbca = m1;
						mfcca = m2;
						////////////////////////////////////////////////////////////////////////////////////
						////////////////////////////////////////////////////////////////////////////////////
						m0 = mfcab * c1o2 + mfbab * (ux - c1o2) + (mfaab + c1o9 * oMdrho) * (ux2 - ux) * c1o2;
						m1 = -mfcab - 2. * mfbab * ux + mfaab * (1. - ux2) - c1o9 * oMdrho * ux2;
						m2 = mfcab * c1o2 + mfbab * (ux + c1o2) + (mfaab + c1o9 * oMdrho) * (ux2 + ux) * c1o2;
						mfaab = m0;
						mfbab = m1;
						mfcab = m2;
						///////////b////////////////////////////////////////////////////////////////////////
						m0 = mfcbb * c1o2 + mfbbb * (ux - c1o2) + (mfabb + c4o9 * oMdrho) * (ux2 - ux) * c1o2;
						m1 = -mfcbb - 2. * mfbbb * ux + mfabb * (1. - ux2) - c4o9 * oMdrho * ux2;
						m2 = mfcbb * c1o2 + mfbbb * (ux + c1o2) + (mfabb + c4o9 * oMdrho) * (ux2 + ux) * c1o2;
						mfabb = m0;
						mfbbb = m1;
						mfcbb = m2;
						///////////b////////////////////////////////////////////////////////////////////////
						m0 = mfccb * c1o2 + mfbcb * (ux - c1o2) + (mfacb + c1o9 * oMdrho) * (ux2 - ux) * c1o2;
						m1 = -mfccb - 2. * mfbcb * ux + mfacb * (1. - ux2) - c1o9 * oMdrho * ux2;
						m2 = mfccb * c1o2 + mfbcb * (ux + c1o2) + (mfacb + c1o9 * oMdrho) * (ux2 + ux) * c1o2;
						mfacb = m0;
						mfbcb = m1;
						mfccb = m2;
						////////////////////////////////////////////////////////////////////////////////////
						////////////////////////////////////////////////////////////////////////////////////
						m0 = mfcac * c1o2 + mfbac * (ux - c1o2) + (mfaac + c1o36 * oMdrho) * (ux2 - ux) * c1o2;
						m1 = -mfcac - 2. * mfbac * ux + mfaac * (1. - ux2) - c1o36 * oMdrho * ux2;
						m2 = mfcac * c1o2 + mfbac * (ux + c1o2) + (mfaac + c1o36 * oMdrho) * (ux2 + ux) * c1o2;
						mfaac = m0;
						mfbac = m1;
						mfcac = m2;
						///////////c////////////////////////////////////////////////////////////////////////
						m0 = mfcbc * c1o2 + mfbbc * (ux - c1o2) + (mfabc + c1o9 * oMdrho) * (ux2 - ux) * c1o2;
						m1 = -mfcbc - 2. * mfbbc * ux + mfabc * (1. - ux2) - c1o9 * oMdrho * ux2;
						m2 = mfcbc * c1o2 + mfbbc * (ux + c1o2) + (mfabc + c1o9 * oMdrho) * (ux2 + ux) * c1o2;
						mfabc = m0;
						mfbbc = m1;
						mfcbc = m2;
						///////////c////////////////////////////////////////////////////////////////////////
						m0 = mfccc * c1o2 + mfbcc * (ux - c1o2) + (mfacc + c1o36 * oMdrho) * (ux2 - ux) * c1o2;
						m1 = -mfccc - 2. * mfbcc * ux + mfacc * (1. - ux2) - c1o36 * oMdrho * ux2;
						m2 = mfccc * c1o2 + mfbcc * (ux + c1o2) + (mfacc + c1o36 * oMdrho) * (ux2 + ux) * c1o2;
						mfacc = m0;
						mfbcc = m1;
						mfccc = m2;

						mfbbb = rho * c1o3 * (mfbbb) + 0.5 * forcingTerm[0];
						mfcbb = rho * c1o3 * (mfcbb) + 0.5 * forcingTerm[1];
						mfbcb = rho * c1o3 * (mfbcb) + 0.5 * forcingTerm[3];
						mfbbc = rho * c1o3 * (mfbbc) + 0.5 * forcingTerm[5];
						mfccb = rho * c1o3 * (mfccb) + 0.5 * forcingTerm[7];
						mfacb = rho * c1o3 * (mfacb) + 0.5 * forcingTerm[8];
						mfcbc = rho * c1o3 * (mfcbc) + 0.5 * forcingTerm[11];
						mfabc = rho * c1o3 * (mfabc) + 0.5 * forcingTerm[12];
						mfbcc = rho * c1o3 * (mfbcc) + 0.5 * forcingTerm[15];
						mfbac = rho * c1o3 * (mfbac) + 0.5 * forcingTerm[16];
						mfccc = rho * c1o3 * (mfccc) + 0.5 * forcingTerm[19];
						mfacc = rho * c1o3 * (mfacc) + 0.5 * forcingTerm[20];
						mfcac = rho * c1o3 * (mfcac) + 0.5 * forcingTerm[21];
						mfaac = rho * c1o3 * (mfaac) + 0.5 * forcingTerm[22];
						mfabb = rho * c1o3 * (mfabb) + 0.5 * forcingTerm[2];
						mfbab = rho * c1o3 * (mfbab) + 0.5 * forcingTerm[4];
						mfbba = rho * c1o3 * (mfbba) + 0.5 * forcingTerm[6];
						mfaab = rho * c1o3 * (mfaab) + 0.5 * forcingTerm[10];
						mfcab = rho * c1o3 * (mfcab) + 0.5 * forcingTerm[9];
						mfaba = rho * c1o3 * (mfaba) + 0.5 * forcingTerm[14];
						mfcba = rho * c1o3 * (mfcba) + 0.5 * forcingTerm[13];
						mfbaa = rho * c1o3 * (mfbaa) + 0.5 * forcingTerm[18];
						mfbca = rho * c1o3 * (mfbca) + 0.5 * forcingTerm[17];
						mfaaa = rho * c1o3 * (mfaaa) + 0.5 * forcingTerm[26];
						mfcaa = rho * c1o3 * (mfcaa) + 0.5 * forcingTerm[25];
						mfaca = rho * c1o3 * (mfaca) + 0.5 * forcingTerm[24];
						mfcca = rho * c1o3 * (mfcca) + 0.5 * forcingTerm[23];

						pop[{X + c_alpha[0][0], Y + c_alpha[0][1], Z + c_alpha[0][2], 0}] = mfbbb;
						pop[{X + c_alpha[1][0], Y + c_alpha[1][1], Z + c_alpha[1][2], 1}] = mfcbb;
						pop[{X + c_alpha[2][0], Y + c_alpha[2][1], Z + c_alpha[2][2], 2}] = mfabb;
						pop[{X + c_alpha[3][0], Y + c_alpha[3][1], Z + c_alpha[3][2], 3}] = mfbcb;
						pop[{X + c_alpha[4][0], Y + c_alpha[4][1], Z + c_alpha[4][2], 4}] = mfbab;
						pop[{X + c_alpha[5][0], Y + c_alpha[5][1], Z + c_alpha[5][2], 5}] = mfbbc;
						pop[{X + c_alpha[6][0], Y + c_alpha[6][1], Z + c_alpha[6][2], 6}] = mfbba;
						pop[{X + c_alpha[7][0], Y + c_alpha[7][1], Z + c_alpha[7][2], 7}] = mfccb;
						pop[{X + c_alpha[8][0], Y + c_alpha[8][1], Z + c_alpha[8][2], 8}] = mfacb;
						pop[{X + c_alpha[9][0], Y + c_alpha[9][1], Z + c_alpha[9][2], 9}] = mfcab;
						pop[{X + c_alpha[10][0], Y + c_alpha[10][1], Z + c_alpha[10][2], 10}] = mfaab;
						pop[{X + c_alpha[11][0], Y + c_alpha[11][1], Z + c_alpha[11][2], 11}] = mfcbc;
						pop[{X + c_alpha[12][0], Y + c_alpha[12][1], Z + c_alpha[12][2], 12}] = mfabc;
						pop[{X + c_alpha[13][0], Y + c_alpha[13][1], Z + c_alpha[13][2], 13}] = mfcba;
						pop[{X + c_alpha[14][0], Y + c_alpha[14][1], Z + c_alpha[14][2], 14}] = mfaba;
						pop[{X + c_alpha[15][0], Y + c_alpha[15][1], Z + c_alpha[15][2], 15}] = mfbcc;
						pop[{X + c_alpha[16][0], Y + c_alpha[16][1], Z + c_alpha[16][2], 16}] = mfbac;
						pop[{X + c_alpha[17][0], Y + c_alpha[17][1], Z + c_alpha[17][2], 17}] = mfbca;
						pop[{X + c_alpha[18][0], Y + c_alpha[18][1], Z + c_alpha[18][2], 18}] = mfbaa;
						pop[{X + c_alpha[19][0], Y + c_alpha[19][1], Z + c_alpha[19][2], 19}] = mfccc;
						pop[{X + c_alpha[20][0], Y + c_alpha[20][1], Z + c_alpha[20][2], 20}] = mfacc;
						pop[{X + c_alpha[21][0], Y + c_alpha[21][1], Z + c_alpha[21][2], 21}] = mfcac;
						pop[{X + c_alpha[22][0], Y + c_alpha[22][1], Z + c_alpha[22][2], 22}] = mfaac;
						pop[{X + c_alpha[23][0], Y + c_alpha[23][1], Z + c_alpha[23][2], 23}] = mfcca;
						pop[{X + c_alpha[24][0], Y + c_alpha[24][1], Z + c_alpha[24][2], 24}] = mfaca;
						pop[{X + c_alpha[25][0], Y + c_alpha[25][1], Z + c_alpha[25][2], 25}] = mfcaa;
						pop[{X + c_alpha[26][0], Y + c_alpha[26][1], Z + c_alpha[26][2], 26}] = mfaaa;

					}  // EOB -> if (is_solid[{X,Y,Z}] == FALSE)
				}  // EOB -> for (Z =...
			}  // EOB -> for (y =...
		}  // EOB -> for (X =...
	}  // EOB -> if (MPI_parallel->processor_id != MASTER)
}
/// ***************************************************** ///
/// COLLISION AND STREAMING MRT MODEL FOR LOW MACH (TBC)  ///
/// ***************************************************** ///
void Flow_solver::LBM_MRT_LMNA(int time, Thermal_solver* Thermal, Species_solver* Species, Geometry* Geo, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		/// Swap populations
		swap(pop_old, pop);
		unsigned int X, Y, Z;
		double grad_rho[3];
		vector<double> omega_eff, Momeq, Mom, F;
		double ux, uy, rho, p_h;

		omega_eff.resize(Discrete_Velocity);
		Mom.resize(Discrete_Velocity);
		Momeq.resize(Discrete_Velocity);
		F.resize(Discrete_Velocity);
		for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (is_solid[{X, Y, Z}] == FALSE) {
						/// -----------> COMPUTE GRADIENT OF DENSITY FIELD (WITH FIRST ORDER UPWIND TO DISSIPATE OSCILLATIONS)
						grad_rho[0] = 0;
						grad_rho[1] = 0;
						grad_rho[2] = 0;
#if defined FD_CENTRAL || defined FD_UPWIND2 || defined FD_UPWIND
						grad_rho[0] = FD::CENTRALNONCONS(1.0, density[{X - 1, Y, Z}], density[{X + 1, Y, Z}]);
						grad_rho[1] = FD::CENTRALNONCONS(1.0, density[{X, Y - 1, Z}], density[{X, Y + 1, Z}]);
						grad_rho[2] = FD::CENTRALNONCONS(1.0, density[{X, Y, Z - 1}], density[{X, Y, Z + 1}]);
#endif  // defined
#if defined FD_CENTRAL4 || defined FD_WENO3
						grad_rho[0] = FD::CENTRAL4NONCONS(1.0, density[{X - 2, Y, Z}], density[{X - 1, Y, Z}], density[{X, Y, Z}],
						                                  density[{X + 1, Y, Z}], density[{X + 2, Y, Z}]);
						grad_rho[1] = FD::CENTRAL4NONCONS(1.0, density[{X, Y - 2, Z}], density[{X, Y - 1, Z}], density[{X, Y, Z}],
						                                  density[{X, Y + 1, Z}], density[{X, Y + 2, Z}]);
						grad_rho[2] = FD::CENTRAL4NONCONS(1.0, density[{X, Y, Z - 2}], density[{X, Y, Z - 1}], density[{X, Y, Z}],
						                                  density[{X, Y, Z + 1}], density[{X, Y, Z + 2}]);
						if (is_solid[{X + 1, Y, Z}] != -1 || is_solid[{X - 1, Y, Z}] != -1) {
							grad_rho[0] = FD::CENTRALNONCONS(1., density[{X - 1, Y, Z}], density[{X + 1, Y, Z}]);
						}
						if (is_solid[{X, Y + 1, Z}] != -1 || is_solid[{X, Y - 1, Z}] != -1) {
							grad_rho[1] = FD::CENTRALNONCONS(1., density[{X, Y - 1, Z}], density[{X, Y + 1, Z}]);
						}
						if (is_solid[{X, Y, Z + 1}] != -1 || is_solid[{X, Y, Z - 1}] != -1) {
							grad_rho[2] = FD::CENTRALNONCONS(1., density[{X, Y, Z - 1}], density[{X, Y, Z + 1}]);
						}
#endif  // defined
						if (Dimension < 2) grad_rho[1] = 0;
						if (Dimension < 3) grad_rho[2] = 0;

						if (Discrete_Velocity == 9) {
							omega_eff[0] = 1.;
							omega_eff[1] = 1.;
							omega_eff[2] = 1.;
							omega_eff[3] = 1. / (c_s2 * viscosity[{X, Y, Z}] + 0.5);
							omega_eff[4] = 1. / (c_s2 * viscosity[{X, Y, Z}] + 0.5);
							omega_eff[5] = 1. / (c_s2 * viscosity[{X, Y, Z}] + 0.5);
							omega_eff[6] = 1.;
							omega_eff[7] = 1.;
							omega_eff[8] = 1.;

							p_h = pressure[{X, Y, Z}];
							rho = density[{X, Y, Z}];
							ux = velocity[{X, Y, Z, 0}];
							uy = velocity[{X, Y, Z, 1}];
							/// *************************************************************************************************** ///
							///                                       EQUILIBRIUM CENTRAL MOMENTS                                   ///
							///                                             MOMENT SPACE :                                          ///
							///                                                  M_0,                                               ///
							///                                                H_x, H_y,                                            ///
							///                                             H_xx, H_yy, H_xy                                        ///
							///                                              H_xxy, H_xyy                                           ///
							///                                                  H_xxyy                                             ///
							/// *************************************************************************************************** ///
							Momeq[0] = p_h;
							Momeq[1] = rho * ux / c_s2;
							Momeq[2] = rho * uy / c_s2;
							Momeq[3] = rho * pow(ux, 2) / c_s2;
							Momeq[4] = rho * pow(uy, 2) / c_s2;
							Momeq[5] = rho * ux * uy / c_s2;
							Momeq[6] = rho * uy * pow(ux, 2) / c_s2;
							Momeq[7] = rho * ux * pow(uy, 2) / c_s2;
							Momeq[8] = rho * pow(ux, 2) * pow(uy, 2) / c_s2;

							F[0] = (ux * grad_rho[0] + uy * grad_rho[1] + rho * divU[{X, Y, Z}]) / c_s2;
							F[1] = 0;
							F[2] = 0;
							F[3] = (grad_rho[0] * ux * (2. / c_s2 - pow(ux, 2)) - grad_rho[1] * uy * pow(ux, 2)) / c_s2;
							F[4] = (grad_rho[1] * uy * (2. / c_s2 - pow(uy, 2)) - grad_rho[0] * ux * pow(uy, 2)) / c_s2;
							F[5] = ((grad_rho[1] * ux) * (1. / c_s2 - pow(uy, 2)) + (grad_rho[0] * uy) * (1. / c_s2 - pow(ux, 2))) / c_s2;
							F[6] = (grad_rho[1] * ux + 2. * grad_rho[0] * uy) / pow(c_s2, 2);
							F[7] = (grad_rho[0] * uy + 2. * grad_rho[1] * ux) / pow(c_s2, 2);
							F[8] = 0;

							Mom[0] = (1. - 0.5 * omega_eff[0]) * F[0] + omega_eff[0] * Momeq[0];  // + (1.-omega_eff[0])*(pop_old[{X, Y, Z, 0}] + pop_old[{X, Y, Z, 1}] + pop_old[{X, Y, Z, 2}] + pop_old[{X, Y, Z, 3}] + pop_old[{X, Y, Z, 4}] + pop_old[{X, Y, Z, 5}] + pop_old[{X, Y, Z, 6}] + pop_old[{X, Y, Z, 7}] + pop_old[{X, Y, Z, 8}]);
							Mom[1] = (1. - 0.5 * omega_eff[1]) * F[1] + omega_eff[1] * Momeq[1];  // + (1.-omega_eff[1])*(pop_old[{X, Y, Z, 1}] - pop_old[{X, Y, Z, 3}] + pop_old[{X, Y, Z, 5}] - pop_old[{X, Y, Z, 6}] - pop_old[{X, Y, Z, 7}] + pop_old[{X, Y, Z, 8}]);
							Mom[2] = (1. - 0.5 * omega_eff[2]) * F[2] + omega_eff[2] * Momeq[2];  // + (1.-omega_eff[2])*(pop_old[{X, Y, Z, 2}] - pop_old[{X, Y, Z, 4}] + pop_old[{X, Y, Z, 5}] + pop_old[{X, Y, Z, 6}] - pop_old[{X, Y, Z, 7}] - pop_old[{X, Y, Z, 8}]);
							Mom[3] = (1. - 0.5 * omega_eff[3]) * F[3] + omega_eff[3] * Momeq[3] + (1. - omega_eff[3]) * ((2 * pop_old[{X, Y, Z, 1}]) / (double)3 - pop_old[{X, Y, Z, 0}] / (double)3 - pop_old[{X, Y, Z, 2}] / (double)3 + (2 * pop_old[{X, Y, Z, 3}]) / (double)3 - pop_old[{X, Y, Z, 4}] / (double)3 + (2 * pop_old[{X, Y, Z, 5}]) / (double)3 + (2 * pop_old[{X, Y, Z, 6}]) / (double)3 + (2 * pop_old[{X, Y, Z, 7}]) / (double)3) + (2 * pop_old[{X, Y, Z, 8}]) / (double)3;
							Mom[4] = (1. - 0.5 * omega_eff[4]) * F[4] + omega_eff[4] * Momeq[4] + (1. - omega_eff[4]) * ((2 * pop_old[{X, Y, Z, 2}]) / (double)3 - pop_old[{X, Y, Z, 1}] / (double)3 - pop_old[{X, Y, Z, 0}] / (double)3 - (2 * pop_old[{X, Y, Z, 3}]) / (double)3 + pop_old[{X, Y, Z, 4}] / (double)3 + (2 * pop_old[{X, Y, Z, 5}]) / (double)3 + (2 * pop_old[{X, Y, Z, 6}]) / (double)3 + (2 * pop_old[{X, Y, Z, 7}]) / (double)3) + (2 * pop_old[{X, Y, Z, 8}]) / (double)3;
							Mom[5] = (1. - 0.5 * omega_eff[5]) * F[5] + omega_eff[5] * Momeq[5] + (1. - omega_eff[5]) * (pop_old[{X, Y, Z, 5}] - pop_old[{X, Y, Z, 6}] + pop_old[{X, Y, Z, 7}] - pop_old[{X, Y, Z, 8}]);
							Mom[6] = (1. - 0.5 * omega_eff[6]) * F[6] + omega_eff[6] * Momeq[6];  // + (1.-omega_eff[6])*(pop_old[{X, Y, Z, 4}]/(double)3 - pop_old[{X, Y, Z, 2}]/(double)3 + (2*pop_old[{X, Y, Z, 5}])/(double)3 + (2*pop_old[{X, Y, Z, 6}])/(double)3 - (2*pop_old[{X, Y, Z, 7}])/(double)3 - (2*pop_old[{X, Y, Z, 8}])/(double)3);
							Mom[7] = (1. - 0.5 * omega_eff[7]) * F[7] + omega_eff[7] * Momeq[7];  // + (1.-omega_eff[7])*(pop_old[{X, Y, Z, 3}]/(double)3 - pop_old[{X, Y, Z, 1}]/(double)3 + (2*pop_old[{X, Y, Z, 5}])/(double)3 - (2*pop_old[{X, Y, Z, 6}])/(double)3 - (2*pop_old[{X, Y, Z, 7}])/(double)3 + (2*pop_old[{X, Y, Z, 8}])/(double)3);
							Mom[8] = (1. - 0.5 * omega_eff[8]) * F[8] + omega_eff[8] * Momeq[8];  // + (1.-omega_eff[8])*(pop_old[{X, Y, Z, 0}]/(double)9 - (2*pop_old[{X, Y, Z, 1}])/(double)9 - (2*pop_old[{X, Y, Z, 2}])/(double)9 - (2*pop_old[{X, Y, Z, 3}])/(double)9 - (2*pop_old[{X, Y, Z, 4}])/(double)9 + (4*pop_old[{X, Y, Z, 5}])/(double)9 + (4*pop_old[{X, Y, Z, 6}])/(double)9 + (4*pop_old[{X, Y, Z, 7}])/(double)9 + (4*pop_old[{X, Y, Z, 8}])/(double)9);

							pop[{X + c_alpha[0][0], Y + c_alpha[0][1], Z + c_alpha[0][2], 0}] = (4 * Mom[0]) / (double)9 - (2 * Mom[3]) / (double)3 - (2 * Mom[4]) / (double)3 + Mom[8];
							pop[{X + c_alpha[1][0], Y + c_alpha[1][1], Z + c_alpha[1][2], 1}] = Mom[0] / (double)9 + Mom[1] / (double)3 + Mom[3] / (double)3 - Mom[4] / (double)6 - Mom[7] / (double)2 - Mom[8] / (double)2;
							pop[{X + c_alpha[2][0], Y + c_alpha[2][1], Z + c_alpha[2][2], 2}] = Mom[0] / (double)9 + Mom[2] / (double)3 - Mom[3] / (double)6 + Mom[4] / (double)3 - Mom[6] / (double)2 - Mom[8] / (double)2;
							pop[{X + c_alpha[3][0], Y + c_alpha[3][1], Z + c_alpha[3][2], 3}] = Mom[0] / (double)9 - Mom[1] / (double)3 + Mom[3] / (double)3 - Mom[4] / (double)6 + Mom[7] / (double)2 - Mom[8] / (double)2;
							pop[{X + c_alpha[4][0], Y + c_alpha[4][1], Z + c_alpha[4][2], 4}] = Mom[0] / (double)9 - Mom[2] / (double)3 - Mom[3] / (double)6 + Mom[4] / (double)3 + Mom[6] / (double)2 - Mom[8] / (double)2;
							pop[{X + c_alpha[5][0], Y + c_alpha[5][1], Z + c_alpha[5][2], 5}] = Mom[0] / (double)36 + Mom[1] / (double)12 + Mom[2] / (double)12 + Mom[3] / (double)12 + Mom[4] / (double)12 + Mom[5] / (double)4 + Mom[6] / (double)4 + Mom[7] / (double)4 + Mom[8] / (double)4;
							pop[{X + c_alpha[6][0], Y + c_alpha[6][1], Z + c_alpha[6][2], 6}] = Mom[0] / (double)36 - Mom[1] / (double)12 + Mom[2] / (double)12 + Mom[3] / (double)12 + Mom[4] / (double)12 - Mom[5] / (double)4 + Mom[6] / (double)4 - Mom[7] / (double)4 + Mom[8] / (double)4;
							pop[{X + c_alpha[7][0], Y + c_alpha[7][1], Z + c_alpha[7][2], 7}] = Mom[0] / (double)36 - Mom[1] / (double)12 - Mom[2] / (double)12 + Mom[3] / (double)12 + Mom[4] / (double)12 + Mom[5] / (double)4 - Mom[6] / (double)4 - Mom[7] / (double)4 + Mom[8] / (double)4;
							pop[{X + c_alpha[8][0], Y + c_alpha[8][1], Z + c_alpha[8][2], 8}] = Mom[0] / (double)36 + Mom[1] / (double)12 - Mom[2] / (double)12 + Mom[3] / (double)12 + Mom[4] / (double)12 - Mom[5] / (double)4 - Mom[6] / (double)4 + Mom[7] / (double)4 + Mom[8] / (double)4;
						}
					}
				}
			}
		}
		return;
	}
}
/// ***************************************************** ///
/// APPLICATION OF BOUNDARY CONITIONS                     ///
/// ***************************************************** ///
void Flow_solver::BC(int time, Thermal_solver* Thermal, Species_solver* Species, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->is_master()) {
		return;
	}

	auto compute_theta = [&](const Flow_fluid_boundary_node& node) {
#if defined compressible
		const double conv_factor = c_s2 * Thermal->T_0 / sqr(global_parameters.D_x / global_parameters.D_t);
#if defined Flow_With_Species
		// todo: molar_mass_av should be converted to a Tensor
		const double r = R_GAS / Species->molar_mass_av[{X + Boundaries[k].n[0], Y + Boundaries[k].n[1], Z + Boundaries[k].n[2]];
#else
		const double r = R_GAS / M_av;
#endif  // defined
			const double theta = r * node.img_stencil.interpolate(Thermal->temperature) * conv_factor;
#else
		constexpr double theta = 1. / 3.;
#endif
			return theta;
	};

	std::vector<double> pop_interpolated;
	pop_interpolated.resize(Discrete_Velocity);

	for (const Flow_boundary_data& boundary : boundaries) {
			switch (boundary.type) {
				/// half-way Bounce-Back wall
				case Fluid_BC_type::WALL_WEAK: {
					for (const Flow_solid_boundary_node& node : boundary.node_data) {
						for (int i = 0; i < node.directions.size(); ++i) {
							const int alpha = node.directions[i];
							const Flat_index target_idx = node.flat_idx - c_alpha_offsets[alpha];
							if (curved_bounce_back) {
								const double q = node.distance[i]; /* curved boundary treatment with 1st order interpolation  */
																   /* Bouzidi MH, Firdaouss M, Lallemand P. Momentum transfer */
																   /* of a Boltzmann-lattice fluid with boundaries. Physics of*/
																   /*fluids. 2001 Nov;13(11):3452-9.*/

								pop(target_idx, alpha_bar[alpha]) = q < 0.5 ? 2.0 * q * pop(node.flat_idx, alpha) + (1.0 - 2.0 * q) * pop(target_idx, alpha)
								                                            : (0.5 / q) * pop(node.flat_idx, alpha) + (1.0 - 0.5 / q) * pop(target_idx - c_alpha_offsets[alpha], alpha_bar[alpha]);
							} else {
								pop(target_idx, alpha_bar[alpha]) = pop(node.flat_idx, alpha);
							}
						}
					}
					break;
				}
				case Fluid_BC_type::VELOCITY_WEAK: {
					for (const Flow_solid_boundary_node& node : boundary.node_data) {
						const Vec3 v = node.v + boundary.get_turbulence();
						velocity(node.flat_idx, 0) = boundary.v[0];
						velocity(node.flat_idx, 1) = boundary.v[1];
						velocity(node.flat_idx, 2) = boundary.v[2];
						for (int alpha : node.directions) {
							const Flat_index target_idx = node.flat_idx - c_alpha_offsets[alpha];
							pop(target_idx, alpha_bar[alpha]) = pop(node.flat_idx, alpha)
							                                    - 2. * c_s2 * weight[alpha_bar[alpha]] * density[target_idx] * dot(c_alpha[alpha], v);
						}
					}
					break;
				}
				case Fluid_BC_type::PRESSURE_WEAK: {
					for (const Flow_solid_boundary_node& node : boundary.node_data) {
						for (int alpha : node.directions) {
							const Flat_index target_idx = node.flat_idx - c_alpha_offsets[alpha];
							pop(target_idx, alpha_bar[alpha]) = pop(node.flat_idx, alpha)
							                                    - 2. * c_s2 * weight[alpha_bar[alpha]] * dot(c_alpha[alpha], &velocity(target_idx, 0));
						}
					}
					break;
				}
				case Fluid_BC_type::ZERO_GRAD_1_WEAK: {
					for (const Flow_fluid_boundary_node& node : boundary.fluid_node_data) {
						for (int alpha : node.directions) {
							pop(node.flat_idx, alpha) = node.img_stencil.interpolate(pop, alpha);
						}
					}
					break;
				}
				case Fluid_BC_type::ZERO_GRAD_2_WEAK: {
					for (const Flow_fluid_boundary_node& node : boundary.fluid_node_data) {
						for (int alpha : node.directions) {
							pop(node.flat_idx, alpha) = 2 * node.img_stencil.interpolate(pop, alpha)
							                            - node.img_stencil_2.interpolate(pop, alpha);
						}
					}
					break;
				}
				case Fluid_BC_type::VELOCITY_NEQ: {
					for (const Flow_fluid_boundary_node& node : boundary.fluid_node_data) {
						double rho_f = 0.0;
						double rho_ff = 0.0;
						Vec3 u_f = {};
						/// Compute density and velocity at the image points
						for (int alpha = 0; alpha < Discrete_Velocity; alpha++) {
							const double pop_a = node.img_stencil.interpolate(pop, alpha);
							pop_interpolated[alpha] = pop_a;
							rho_f += pop_a;
							const Vec3 dir = Vec3{c_alpha[alpha][0], c_alpha[alpha][1], c_alpha[alpha][2]};
							u_f += pop_a * dir;

							const double pop_a2 = node.img_stencil_2.interpolate(pop, alpha);
							rho_ff += pop_a2;
						}
						u_f = u_f / rho_f;
						// the last fluid node is half way between the fist img-point and the wall
						const Vec3 u = node.v + boundary.get_turbulence();
						u_f = 0.5 * (u_f + u);
						// density is unknown and has to be extrapolated from the second img point
						const double rho_change = (rho_f - rho_ff) * node.distance;
						const double theta = compute_theta(node);
						/// Get equilibrium distribution function at last fluid node f^(eq)(rho_f, u_f)
						equilibrium(rho_f + rho_change, theta, u_f.data(), pop_eq);
						/// Get quilibrium distribution function at wall f^(eq)(rho_f, u_w)
						equilibrium(rho_f + 2 * rho_change, theta, u.data(), pop_w);
						for (int alpha : node.directions) {
							pop(node.flat_idx, alpha) = pop_w[alpha] + pop_interpolated[alpha] - pop_eq[alpha];
						}
					}
					break;
				}
				case Fluid_BC_type::PRESSURE_NEQ: {
					for (const Flow_fluid_boundary_node& node : boundary.fluid_node_data) {
						double rho_f = 0.0;
						double rho_ff = 0.0;
						Vec3 u_f = {};
						Vec3 u_ff = {};
						/// Compute density and velocity at the image points
						for (int alpha = 0; alpha < Discrete_Velocity; alpha++) {
							const double pop_a = node.img_stencil.interpolate(pop, alpha);
							pop_interpolated[alpha] = pop_a;
							rho_f += pop_a;
							const Vec3 dir = Vec3{c_alpha[alpha][0], c_alpha[alpha][1], c_alpha[alpha][2]};
							u_f += pop_a * dir;

							const double pop_a2 = node.img_stencil_2.interpolate(pop, alpha);
							u_ff += pop_a2 * dir;
							rho_ff += pop_a2;
						}
						u_f = u_f / rho_f;
						u_ff = u_ff / rho_ff;
						const double theta = compute_theta(node);
						// extrapolate velocity from img points
						const Vec3 u_change = (u_f - u_ff) * node.distance;
						u_f += u_change;
						rho_f = 0.5 * (rho_f + boundary.p);
						equilibrium(rho_f, theta, u_f.data(), pop_eq);
						/// Get quilibrium distribution function at wall f^(eq)(rho_w, u_f)
						u_f += u_change;
						equilibrium(boundary.p, theta, u_f.data(), pop_w);
						for (int alpha : node.directions) {
							pop(node.flat_idx, alpha) = pop_w[alpha] + pop_interpolated[alpha] - pop_eq[alpha];
						}
						break;
					}
				}
				case Fluid_BC_type::VELOCITY_EQ: {
					for (const Flow_fluid_boundary_node& node : boundary.fluid_node_data) {
						const Vec3 u = node.v + boundary.get_turbulence();
						double rho_f = 0.0;
						double rho_ff = 0.0;
						for (int alpha = 0; alpha < Discrete_Velocity; alpha++) {
							const double pop_a = node.img_stencil.interpolate(pop, alpha);
							rho_f += pop_a;
							const double pop_a2 = node.img_stencil_2.interpolate(pop, alpha);
							rho_ff += pop_a2;
						}
						//	const double rho_f = node.img_stencil.interpolate(density);
						//	const double rho_ff = node.img_stencil_2.interpolate(density);
						const double rho_change = (rho_f - rho_ff) * node.distance;
						equilibrium(rho_f + 2.0 * rho_change, compute_theta(node), u.data(), pop_eq);
						for (unsigned alpha = 0; alpha < Discrete_Velocity; alpha++) {
							pop(node.flat_idx, alpha) = pop_eq[alpha];
						}
					}
					break;
				}
				case Fluid_BC_type::PRESSURE_EQ: {
					for (const Flow_fluid_boundary_node& node : boundary.fluid_node_data) {
						Vec3 u_f = {};
						Vec3 u_ff = {};
						double rho_f = 0.0;
						double rho_ff = 0.0;
						for (int alpha = 0; alpha < Discrete_Velocity; alpha++) {
							const double pop_a = node.img_stencil.interpolate(pop, alpha);
							const Vec3 dir = Vec3{c_alpha[alpha][0], c_alpha[alpha][1], c_alpha[alpha][2]};
							u_f += pop_a * dir;
							rho_f += pop_a;
							const double pop_a2 = node.img_stencil_2.interpolate(pop, alpha);
							u_ff += pop_a2 * dir;
							rho_ff += pop_a2;
						}
						u_f = u_f / rho_f;
						u_ff = u_ff / rho_ff;
						/*	const Vec3 u_f = {node.img_stencil.interpolate(velocity, 0),
						                      node.img_stencil.interpolate(velocity, 1),
						                      node.img_stencil.interpolate(velocity, 2)};
						    const Vec3 u_ff = {node.img_stencil_2.interpolate(velocity, 0),
						                       node.img_stencil_2.interpolate(velocity, 1),
						                       node.img_stencil_2.interpolate(velocity, 2)};*/
						const Vec3 u_change = (u_f - u_ff) * node.distance;
						const Vec3 u = u_f + 2.0 * u_change;
						equilibrium(boundary.p, compute_theta(node), u.data(), pop_eq);
						for (int alpha = 0; alpha < Discrete_Velocity; alpha++) {
							pop(node.flat_idx, alpha) = pop_eq[alpha];
						}
					}
					break;
				}
				case Fluid_BC_type::VELOCITY_LMNA: {
					for (const Flow_solid_boundary_node& node : boundary.node_data) {
						Vec3 u = node.v + boundary.get_turbulence();
						for (int alpha : node.directions) {
							const Flat_index target_idx = node.flat_idx - c_alpha_offsets[alpha];
							constexpr double t_coeff = 1.0;
							pop(target_idx, alpha_bar[alpha]) = pop(node.flat_idx, alpha)
							                                    - 2.0 * weight[alpha_bar[alpha]] * density[node.flat_idx] * t_coeff * dot(c_alpha[alpha], u);
						}
					}
					break;
				}
				case Fluid_BC_type::PRESSURE_LMNA: {
					for (const Flow_solid_boundary_node& node : boundary.node_data) {
						for (int alpha : node.directions) {
							const Flat_index target_idx = node.flat_idx - c_alpha_offsets[alpha];
							constexpr double t_coeff = 1.0;
							pop(target_idx, alpha_bar[alpha]) = 0.0;  // pop(node.flat_idx, alpha)
							                                          //- 2.0 * weight[alpha_bar[alpha]] * density[target_idx] * boundary.p * t_coeff * dot(c_alpha[alpha], &velocity(target_idx, 0));
							if (std::isnan(pop(target_idx, alpha_bar[alpha]))) {
								std::cout << pop(node.flat_idx, alpha) << " - " << -2.0 * weight[alpha_bar[alpha]] * density[target_idx] * boundary.p * t_coeff * dot(c_alpha[alpha], &velocity(target_idx, 0)) << std::endl;
							}
						}
					}
					break;
				}
				case Fluid_BC_type::VELOCITY_NEQ_LMNA: {
					for (const Flow_fluid_boundary_node& node : boundary.fluid_node_data) {
						const int Xp = node.idx[0] + node.n[0];
						const int Yp = node.idx[1] + node.n[1];
						const int Zp = node.idx[2] + node.n[2];
						double P_f = 0;
						Vec3 u_f = {};
						const Vec3 grad_rho = {0.5 * (density[{Xp + 1, Yp, Zp}] - density[{Xp - 1, Yp, Zp}]),
						                       0.5 * (density[{Xp, Yp + 1, Zp}] - density[{Xp, Yp - 1, Zp}]),
						                       0.5 * (density[{Xp, Yp, Zp + 1}] - density[{Xp, Yp, Zp - 1}])};
						/// Compute density and velocity at last fluid node
						for (int alpha = 0; alpha < Discrete_Velocity; alpha++) {
							const double pop_a = pop[{Xp, Yp, Zp, alpha}];
							P_f += pop_a;
							u_f[0] += c_alpha[alpha][0] * pop_a;
							u_f[1] += c_alpha[alpha][1] * pop_a;
							u_f[2] += c_alpha[alpha][2] * pop_a;
						}
						const double dens_p = density[{Xp, Yp, Zp}];
						P_f += (0.5 / c_s2) * (dot(&velocity[{Xp, Yp, Zp, 0}], grad_rho) + previous_divU[{Xp, Yp, Zp}] * dens_p);
						u_f = u_f * (c_s2 / dens_p);
						/// Get equilibrium distribution function at last fluid node f^(eq)(rho_f, u_f)
						const double dens = density[node.flat_idx];
						equilibrium(dens, 1.0, boundary.v.data(), pop_w);
						equilibrium(dens_p, 1.0, u_f.data(), pop_eq);
						for (int alpha : node.directions) {
							pop_eq[alpha] = (1. / c_s2) * pop_eq[alpha] + weight[alpha] * (P_f - dens_p / c_s2);
							pop_w[alpha] = (1. / c_s2) * pop_w[alpha] + weight[alpha] * (P_f - dens / c_s2);
							pop(node.flat_idx, alpha) = pop_w[alpha] + pop[{Xp, Yp, Zp, alpha}] - pop_eq[alpha];
						}
					}
					break;
				}
				case Fluid_BC_type::PRESSURE_NEQ_LMNA: {
					for (const Flow_fluid_boundary_node& node : boundary.fluid_node_data) {
						const int Xp = node.idx[0] + node.n[0];
						const int Yp = node.idx[1] + node.n[1];
						const int Zp = node.idx[2] + node.n[2];
						double P_f = 0;
						Vec3 u_f = {};
						const Vec3 grad_rho = {0.5 * (density[{Xp + 1, Yp, Zp}] - density[{Xp - 1, Yp, Zp}]),
						                       0.5 * (density[{Xp, Yp + 1, Zp}] - density[{Xp, Yp - 1, Zp}]),
						                       0.5 * (density[{Xp, Yp, Zp + 1}] - density[{Xp, Yp, Zp - 1}])};
						/// Compute density and velocity at last fluid node
						for (int alpha = 0; alpha < Discrete_Velocity; alpha++) {
							const double pop_a = pop[{Xp, Yp, Zp, alpha}];
							P_f += pop_a;
							u_f[0] += c_alpha[alpha][0] * pop_a;
							u_f[1] += c_alpha[alpha][1] * pop_a;
							u_f[2] += c_alpha[alpha][2] * pop_a;
						}
						const double dens_p = density[{Xp, Yp, Zp}];
						P_f += (0.5 / c_s2) * (dot(&velocity[{Xp, Yp, Zp, 0}], grad_rho) + divU[{Xp, Yp, Zp}] * dens_p);
						u_f = u_f * (c_s2 / dens_p);
						/// Get equilibrium distribution function at last fluid node f^(eq)(rho_f, u_f)
						const double dens = density[node.flat_idx];
						equilibrium(dens, 1.0, u_f.data(), pop_w);
						equilibrium(dens_p, 1.0, u_f.data(), pop_eq);
						for (int alpha : node.directions) {
							pop_eq[alpha] = (1. / c_s2) * pop_eq[alpha] + weight[alpha] * (P_f - dens_p / c_s2);
							pop_w[alpha] = (1. / c_s2) * pop_w[alpha] + weight[alpha] * (boundary.p - dens / c_s2);
							pop(node.flat_idx, alpha) = pop_w[alpha] + pop[{Xp, Yp, Zp, alpha}] - pop_eq[alpha];
						}
					}
					break;
				}
				case Fluid_BC_type::CONVECTIVE_LMNA: {
					for (const Flow_fluid_boundary_node& node : boundary.fluid_node_data) {
						// population -> velocity involves scaling by density
						const double s = node.img_stencil.interpolate(density) / density[node.flat_idx];
						// overwride all directions to get equal velocity
						for (int alpha = 0; alpha < Discrete_Velocity; ++alpha) {
							pop(node.flat_idx, alpha) = s * node.img_stencil.interpolate(pop, alpha);
						}
					}
					break;
				}
				case Fluid_BC_type::SYMMETRICAL_LMNA: {
					for (const Flow_fluid_boundary_node& node : boundary.fluid_node_data) {
						for (int alpha : node.directions) {
							Index_vec3 c_alpha_sym = {node.n[0] ? -1 : 0, node.n[1] ? -1 : 0, node.n[2] ? -1 : 0};
							int alpha_p;
							for (alpha_p = 1; alpha_p < Discrete_Velocity; alpha_p++) {
								if (c_alpha[alpha_p][0] == c_alpha[alpha][0] * c_alpha_sym[0]
								    && c_alpha[alpha_p][1] == c_alpha[alpha][1] * c_alpha_sym[1]
								    && c_alpha[alpha_p][2] == c_alpha[alpha][2] * c_alpha_sym[2]) {
									break;
								}
							}
							pop(node.flat_idx, alpha) = pop(node.flat_idx - is_solid.flat_index(node.n), alpha_p);
						}
					}
					break;
				}
				default: {
					break;
				}
			};
	}
	}
	/// ***************************************************** ///
	/// APPLICATION OF FILTERE ON BOUNDARY CONITIONS          ///
	/// TO MINIMIZE REFLECTION EFFECTS                        ///
	/// ***************************************************** ///
	/* This function applies a low-pass filter to the populations of the boundary nodes to minimize reflection effects.
	  It is based on the work of Lallemand and Luo (2003) and is used to filter the populations of the boundary nodes by
	  applying a low-pass filter to the populations of the boundary nodes to minimize reflection effects.
	  It first computes the coefficients of the low-pass filter and then applies the filter to the populations of the boundary nodes.
	  The filtered populations are then stored in the boundary node data structure for the next time step.
	  The function is called in the main loop of the code after the populations have been updated and before the populations are streamed to the neighboring nodes.
	*/
	void Flow_solver::BC_filter(int time, Parallel_MPI* MPI_parallel) {
		if (MPI_parallel->processor_id != MASTER) {
			int alpha, Xp, Yp, Zp, X, Y, Z;
			double C_1, C_2, pop_temp;
			for (int k = 0; k < boundaries.size(); ++k) {
				X = boundaries[k].X;                                                                                     /// Get the x-coordinate of the boundary node
				Y = boundaries[k].Y;                                                                                     /// Get the y-coordinate of the boundary node
				Z = boundaries[k].Z;                                                                                     /// Get the z-coordinate of the boundary node
				for (alpha = 1; alpha < Discrete_Velocity; alpha++) {                                                    /// Loop over all velocities
					Xp = (X - c_alpha[alpha][0]);                                                                        /// Get the x-coordinate of the neighbor in direction alpha
					Yp = (Y - c_alpha[alpha][1]);                                                                        /// Get the y-coordinate of the neighbor in direction alpha
					Zp = (Z - c_alpha[alpha][2]);                                                                        /// Get the z-coordinate of the neighbor in direction alpha
					if (is_solid[{Xp, Yp, Zp}] != -1 && boundaries[k].filtered == 1) {                                   /// Make sure the neighbor is solid and that it is a filtered boundary
						C_2 = 2. / (double)boundaries[k].w_c;                                                            /// Compute the first coefficient of the low-pass filter
						C_1 = 1. / (double)(1 + C_2);                                                                    /// Compute the second coefficient of the low-pass filter
						pop_temp = pop[{X, Y, Z, alpha_bar[alpha]}];                                                     /// Store the unfiltered value of the population in temp
						pop[{X, Y, Z, alpha_bar[alpha]}] = C_1 * (pop_temp                                               /// compute the filtered population
						                                          + boundaries[k].pop_filtered[alpha_bar[alpha]]         /// using previous filtered population:f_old, current unfiltered:pop_temp
						                                          + (C_2 - 1.) * pop_old[{X, Y, Z, alpha_bar[alpha]}]);  /// and previous unfiltered: pop_filtered
						boundaries[k].pop_filtered[alpha_bar[alpha]] = pop_temp;                                         /// Put unfiltered population data from temp to pop_filtered for next t
					}
				}
			}
		}
		return;
	}
	/* Sponge zone
	rstart = can be any point towards the outlet. Lower bound of the sponge zone.
	rend = can be the end of the domain of the STL value. Upper bound of the sponge zone.
	direction = 0,1,2 based on which direction the outlet is. 0-x, 1-y, 2-z.
	max_viscosity = Maximum viscosity value to which the viscosity will be adjusted.

	The function, based on the chosen direction and the position of the point, adjusts the viscosity of the fluid in the sponge zone.
	If the cell is not blocked by a solid object and is within the sponge zone, it gradually changes the viscosity of the fluid in that cell.
	This adjustment ensures that the fluid behaves more predictably near the boundary, which can be helpful in simulations.
	In short:  the function helps control how "thick" or "sticky" the fluid is in a specific area near the boundary, making sure it behaves smoothly and predictably.
	*/
	void Flow_solver::Sponge_zone(double r_start, double r_end, int direction, stl_import* Geo_stl, Parallel_MPI* MPI_parallel) {
		int X, Y, Z;
		double max_viscosity = 1.5 * 0.5 / c_s2;
		if (MPI_parallel->processor_id != MASTER) {
			for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
				for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
					for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
						double xc, yc, zc, rr;
						MPI_parallel->get_coordinates(X, Y, Z, Geo_stl->x_center, Geo_stl->y_center, Geo_stl->z_center, xc, yc, zc);
						if (direction == 0) rr = xc;  // these xc, yc values are the global values in SI units which give the position of the point X,Y,Z
						if (direction == 1) rr = yc;
						if (direction == 2) rr = zc;
						if (is_solid[{X, Y, Z}] == FALSE && rr >= r_start && rr <= r_end) {
							viscosity[{X, Y, Z}] += ((max_viscosity - viscosity[{X, Y, Z}]) * .5 * (sin(M_PI * (rr - r_start) / (r_end - r_start) - M_PI / 2.) + 1.));
						}
					}
				}
			}
		}
	}
	/// ***************************************************** ///
	/// GET DISTRIBUTION FUNCTION MOMENTA                     ///
	/// ***************************************************** ///
	template <int D, int Q>
	void Flow_solver::momenta_impl(const Parallel_MPI& MPI_parallel) {
		swap(previous_velocity_magnitude, velocity_magnitude);
		swap(previous_pressure, pressure);
		swap(previous_density, density);
		swap(previous_velocity, velocity);
		swap(temp_force, force);

		// double theta;
		non_solid_lattice.update([this](Flat_index idx) {
			// theta = 1;
			// #if defined compressible
			// 						const double conv_factor = c_s2 * Thermal->T_0 / sqr(global_parameters.D_x / global_parameters.D_t);  /// theta = T / (kBT0/m0) = T / (dx2/dt2)/3
			// #if defined Flow_With_Species
			// 						const double r = R_GAS / Species->molar_mass_av[{X,Y,Z}];
			// #else
			// 						const double r = R_GAS / M_av;
			// #endif  // defined
			//          theta = r * Thermal->temperature[{X,Y,Z}] * conv_factor;
			// #endif

			double pressure_temp = 0.0;
			double density_temp = 0.0;
			double velocity_temp[3] = {};

			for (int alpha = 0; alpha < Q; alpha++) {
				const double population = pop(idx, alpha);
				density_temp += population;
				for (int d = 0; d < D; d++) {
					const double c_a = Stencil<D, Q>::c_alpha[alpha][d];
					velocity_temp[d] += population * c_a;
					pressure_temp += population * c_a * c_a;
				}
			}
			pressure_temp /= Dimension;
			pressure[idx] = pressure_temp;
			density[idx] = density_temp;
			double vel_mag = 0.0;
			for (int d = 0; d < D; d++) {
#if defined Guo || defined MED
				velocity_temp[d] += 0.5 * force[{X, Y, Z, d}];
#endif  // defined
				velocity_temp[d] /= density_temp;
				velocity(idx, d) = velocity_temp[d];
				force(idx, d) = gravity[d] * density_temp;
				vel_mag += sqr(velocity_temp[d]);
			}
			velocity_magnitude[idx] = sqrt(vel_mag);
		});

		for (const Flow_boundary_data& boundary : boundaries) {
			switch (boundary.type) {
				case Fluid_BC_type::VELOCITY_WEAK:
				case Fluid_BC_type::VELOCITY_NEQ:
					for (const Flow_fluid_boundary_node& node : boundary.fluid_node_data) {
						velocity(node.flat_idx, 0) = boundary.v[0];
						velocity(node.flat_idx, 1) = boundary.v[1];
						velocity(node.flat_idx, 2) = boundary.v[2];
					}
					break;
				case Fluid_BC_type::PRESSURE_WEAK:
				case Fluid_BC_type::PRESSURE_NEQ:
					for (const Flow_fluid_boundary_node& node : boundary.fluid_node_data) {
						density[node.flat_idx] = boundary.p;
					}
					break;
				default:
					break;
			}
		}
	}
	void Flow_solver::momenta(int time, Parallel_MPI* MPI_parallel, Thermal_solver* Thermal, Species_solver* Species) {
		if (MPI_parallel->processor_id != MASTER) {
			DISPATCH_BY_STENCIL(momenta_impl, *MPI_parallel);
		}
		simulation_divergence_monitor(MPI_parallel);
	}
	/// ***************************************************** ///
	/// GET DISTRIBUTION FUNCTION MOMENTA FOR LOW MACH        ///
	/// ***************************************************** ///
	/*
	This function calculates the velocity, pressure, and force distribution in a fluid domain, incorporating
	adjustments for low Mach number flows. It involves finite difference gradients, velocity magnitude computations, and pressure corrections based on discrete velocity populations.

	Compute Density Gradient: x-dir: (∂ρ/∂x = (ρ_i+1 - ρ_i-1)/2Δx).
	Adjust Velocity for Low Mach Number: v = c_s^2/ρ * v +1/2 * g.
	Compute Force: f=ρg.
	Pressure Correction: p=p+ (1/2*c_s^2) (∇⋅vρ+div Uρ).
	*/
	void Flow_solver::momenta_LMNA(int time, Parallel_MPI* MPI_parallel, Thermal_solver* Thermal, Species_solver* Species) {
		if (MPI_parallel->processor_id != MASTER) {
			unsigned int X, Y, Z, d, alpha;
			double grad_rho[3];
			for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
				for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
					for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
						if (is_solid[{X, Y, Z}] == FALSE) {
							grad_rho[0] = 0;
							grad_rho[1] = 0;
							grad_rho[2] = 0;
#if defined FD_CENTRAL || defined FD_UPWIND2 || defined FD_UPWIND
							grad_rho[0] = FD::CENTRALNONCONS(1.0, density[{X - 1, Y, Z}], density[{X + 1, Y, Z}]);
							grad_rho[1] = FD::CENTRALNONCONS(1.0, density[{X, Y - 1, Z}], density[{X, Y + 1, Z}]);
							grad_rho[2] = FD::CENTRALNONCONS(1.0, density[{X, Y, Z - 1}], density[{X, Y, Z + 1}]);
#endif  // defined
#if defined FD_CENTRAL4 || defined FD_WENO3
							grad_rho[0] = FD::CENTRAL4NONCONS(1.0, density[{X - 2, Y, Z}], density[{X - 1, Y, Z}], density[{X, Y, Z}],
							                                  density[{X + 1, Y, Z}], density[{X + 2, Y, Z}]);
							grad_rho[1] = FD::CENTRAL4NONCONS(1.0, density[{X, Y - 2, Z}], density[{X, Y - 1, Z}], density[{X, Y, Z}],
							                                  density[{X, Y + 1, Z}], density[{X, Y + 2, Z}]);
							grad_rho[2] = FD::CENTRAL4NONCONS(1.0, density[{X, Y, Z - 2}], density[{X, Y, Z - 1}], density[{X, Y, Z}],
							                                  density[{X, Y, Z + 1}], density[{X, Y, Z + 2}]);

							// If neighboring nodes in certain directions are solid, modify the density gradients accordingly.

							if (is_solid[{X + 1, Y, Z}] != -1 || is_solid[{X - 1, Y, Z}] != -1) {
								grad_rho[0] = FD::CENTRALNONCONS(1., density[{X - 1, Y, Z}], density[{X + 1, Y, Z}]);
							}
							if (is_solid[{X, Y + 1, Z}] != -1 || is_solid[{X, Y - 1, Z}] != -1) {
								grad_rho[1] = FD::CENTRALNONCONS(1., density[{X, Y - 1, Z}], density[{X, Y + 1, Z}]);
							}
							if (is_solid[{X, Y, Z + 1}] != -1 || is_solid[{X, Y, Z - 1}] != -1) {
								grad_rho[2] = FD::CENTRALNONCONS(1., density[{X, Y, Z - 1}], density[{X, Y, Z + 1}]);
							}
#endif  // defined
							if (Dimension < 2) grad_rho[1] = 0;
							if (Dimension < 3) grad_rho[2] = 0;
							previous_velocity_magnitude[{X, Y, Z}] = velocity_magnitude[{X, Y, Z}];
							velocity_magnitude[{X, Y, Z}] = 0;
							for (d = 0; d < 3; d++) {
								previous_velocity[{X, Y, Z, d}] = velocity[{X, Y, Z, d}];
								velocity[{X, Y, Z, d}] = 0;
							}
							pressure[{X, Y, Z}] = 0;
							// Loop over discrete velocity directions (alpha) and update the velocity and pressure based on the distribution function (pop).
							for (alpha = 0; alpha < Discrete_Velocity; alpha++) {
								pressure[{X, Y, Z}] += pop[{X, Y, Z, alpha}];
								for (d = 0; d < 3; d++) {
									velocity[{X, Y, Z, d}] += pop[{X, Y, Z, alpha}] * c_alpha[alpha][d];
								}
							}
							// Scale the velocity components based on the speed of sound and density. Add gravitational effects to v.
							// Compute the force at the node based on gravity and density.
							for (d = 0; d < 3; d++) {
								velocity[{X, Y, Z, d}] *= (c_s2 / density[{X, Y, Z}]);
								velocity[{X, Y, Z, d}] += 0.5 * gravity[d];
								velocity_magnitude[{X, Y, Z}] += sqr(velocity[{X, Y, Z, d}]);
								force[{X, Y, Z, d}] = gravity[d] * density[{X, Y, Z}];
							}
							if (Dimension < 2) force[{X, Y, Z, 1}] = 0;
							if (Dimension < 3) force[{X, Y, Z, 2}] = 0;
							// Pressure corrector: Update the pressure at the node by adding a correction term based on the density gradient and velocity divergence.
							// Recalculate the magnitude of the velocity at the node.
							pressure[{X, Y, Z}] += (0.5 / c_s2) * (dot(&velocity[{X, Y, Z, 0}], grad_rho) + divU[{X, Y, Z}] * density[{X, Y, Z}]);
							velocity_magnitude[{X, Y, Z}] = sqrt(velocity_magnitude[{X, Y, Z}]);
						}
					}
				}
			}
			/// -------------------> This part sets the pressure to the defined boundary pressure and recomputes velocity based on this new pressure
			for (const Flow_boundary_data& boundary : boundaries) {
				switch (boundary.type) {
					case Fluid_BC_type::PRESSURE_LMNA:
						for (const Flow_fluid_boundary_node& node : boundary.fluid_node_data) {
							pressure[node.flat_idx] = boundary.p;
						}
						break;
					default:
						break;
				}
			}
		}
	}
	/// ***************************************************** ///
	/// UPDATE THERMODYNAMIC PRESSURE FOR LOW MACH MODEL      ///
	/// ***************************************************** ///
	void Flow_solver::update_p_th_LMNA(int time, Parallel_MPI* MPI_parallel, Thermal_solver* Thermal, Species_solver* Species) {
		int X, Y, Z;
		double mass_temp = 0;
		double mass_tot = 0;
		if (MPI_parallel->processor_id != MASTER) {
			for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
				for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
					for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
						if (is_solid[{X, Y, Z}] == FALSE) {  /// if a node has at least one fluid neighbor
#if defined Flow_With_Species
							mass_temp += Species->molar_mass_av[{X, Y, Z}] / (Thermal->previous_temperature[{X, Y, Z}] * Thermal->T_0 * R_GAS);
#else
						mass_temp += M_av / (Thermal->previous_temperature[{X, Y, Z}] * Thermal->T_0 * R_GAS);
#endif
						}
					}
				}
			}
		}
		MPI_Allreduce(&mass_temp, &mass_tot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		/// mass_0 = sum_{x,y,z} 1/rT(t)
		p_th_previous = p_th;
		p_th = mass_0 * p_th_0 / mass_tot;
		if (MPI_parallel->processor_id == MASTER && time % t_vtk == 1) {
			stringstream output_filename;
			output_filename << "Alborz_Results/debug/thermodynamic_pressure_monitor.dat";
			ofstream output_file;
			/// Open file
			output_file.open(output_filename.str().c_str(), fstream::app);
			/// Write data
			output_file << time << " ";  // time step
			output_file << setprecision(30) << p_th << endl;
			/// Close file
			output_file.close();
		}

		if (MPI_parallel->processor_id != MASTER) {
			for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
				for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
					for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
						if (Thermal->solid_thermal_type[{X, Y, Z}] == -1) {
							Thermal->temperature[{X, Y, Z}] += (p_th - p_th_previous) / (Thermal->c_p[{X, Y, Z}] * density[{X, Y, Z}] * rho_0 * Thermal->T_0);

							divU[{X, Y, Z}] += (p_th - p_th_previous) * (1. / (Thermal->c_p[{X, Y, Z}] * density[{X, Y, Z}] * rho_0 * Thermal->previous_temperature[{X, Y, Z}] * Thermal->T_0) - 1. / p_th_previous);
						}
					}
				}
			}
		}
	}
	/// ***************************************************** ///
	/// UPDATE DENSITY FOR LOW MACH MODEL UISNG IDEAL GAS     ///
	/// ***************************************************** ///
	/*
	    The loop iterates over interior nodes (excluding boundaries) and updates the density based on the low Mach number assumption and ideal gas law.
	    If a node is not part of a solid region (fluid node), the previous density is stored, and the density is updated based on the low Mach number
	    assumption and ideal gas law.
	    Density is based on : p_th:Thermal pressure, M_av: average molarmass, R_GAS:Gas constant, T temperature,  T_0 and rho_0 reference values.
	*/
	void Flow_solver::update_density_LMNA(int time, Parallel_MPI* MPI_parallel, Thermal_solver* Thermal, Species_solver* Species) {
		if (MPI_parallel->is_master()) {
			return;
		}
		swap(previous_density, density);

		/// GET DENSITY ON INTERIOR NODES
		for (int X = 0; X < MPI_parallel->dev_end[0]; ++X) {
			for (int Y = 0; Y < MPI_parallel->dev_end[1]; ++Y) {
				for (int Z = 0; Z < MPI_parallel->dev_end[2]; ++Z) {
					if (is_solid[{X, Y, Z}] == FALSE) {  /// if a node has at least one fluid neighbor
#if defined Flow_With_Species
						density[{X, Y, Z}] = p_th * Species->molar_mass_av[{X, Y, Z}] / (R_GAS * Thermal->temperature[{X, Y, Z}] * Thermal->T_0 * rho_0);
#else
					density[{X, Y, Z}] = p_th * M_av / (R_GAS * Thermal->temperature[{X, Y, Z}] * Thermal->T_0 * rho_0);
#endif
					}
				}
			}
		}
		/// GET DENSITY ON BOUNDARY NODES
		for (auto& boundary : boundaries) {
			switch (boundary.type) {
				case Fluid_BC_type::ZERO_GRAD_1_WEAK:
				case Fluid_BC_type::ZERO_GRAD_2_WEAK:
				case Fluid_BC_type::CONVECTIVE_LMNA: {
					for (const Flow_solid_boundary_node& node : boundary.node_data) {
						const size_t flat_n = density.flat_index(node.n);
						//	std::cout << node.n << std::endl;
						density[node.flat_idx] = density[node.flat_idx - flat_n];
						/*
					const Flat_index pos = node.flat_idx;
					const Index_vec3 idx = node.idx;
						const double p1 = density[{idx[0] - 1, idx[1], idx[2]}];
						const double p2 = density[{idx[0] - 2, idx[1], idx[2]}];
						for(int X = idx[0]; X < MPI_parallel->dev_end[0]; ++X){
						    density[{X, idx[1], idx[2]}] = p2 + (2.0 + X-idx[0]) * (p1 - p2);
						}*/
						// density[{idx[0] + 1, idx[1], idx[2]}] = p2 + 3.0 * (p1 - p2);
						// density[pos] = p_th * Species->molar_mass_av[pos] / (R_GAS * Thermal->temperature[pos] * Thermal->T_0 * rho_0);
					}
					break;
				}
				default:
					break;
			};
			/*	for (const Flow_fluid_boundary_node& node : boundary.fluid_node_data) {
			        const Flat_index pos = node.flat_idx;
			        const Flat_index pos_p = pos - density.flat_index(node.n);
	#if defined Flow_With_Species
			        const Index_vec3 idx = node.idx;
			        const Index_vec3 idx_p = density.index(pos_p);
			        density[pos_p] = p_th * Species->molar_mass_av[{idx_p[0],idx_p[1],idx_p[2}]] / (R_GAS * Thermal->temperature[pos_p] * Thermal->T_0 * rho_0);
			        density[pos] = p_th * Species->molar_mass_av[{idx[0],idx[1],idx[2}]] / (R_GAS * Thermal->temperature[pos] * Thermal->T_0 * rho_0);
	#else
			        density[pos_p] = p_th * M_av / (R_GAS * Thermal->temperature[pos_p] * Thermal->T_0 * rho_0);
			        density[pos] = p_th * M_av / (R_GAS * Thermal->temperature[pos] * Thermal->T_0 * rho_0);
	#endif
			    }*/
		}
	}
	/// ***************************************************** ///
	/// SWAP NEW AND OLD DIVERGENCE FOR LOW MACH MODEL        ///
	/// ***************************************************** ///
	void Flow_solver::swap_divergence_LMNA(int time, Parallel_MPI* MPI_parallel, Thermal_solver* Thermal, Species_solver* Species) {
		if (MPI_parallel->processor_id != MASTER) {
			unsigned int X, Y, Z;
			for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
				for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
					for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
						previous_divU[{X, Y, Z}] = divU[{X, Y, Z}];
						divU[{X, Y, Z}] = 0;
					}
				}
			}
		}
	}
	/// ***************************************************** ///
	/// EXCHANGE MACROSCOPIC VARS BETWEEN PROCESSORS AT       ///
	/// INTERFACES                                            ///
	/// ***************************************************** ///
	void Flow_solver::Data_Exchange_Macroscopic(Parallel_MPI * MPI_parallel) {
		if (MPI_parallel->processor_id != MASTER) {
			macroscopic_group.exchange_data();
		}
	}
	/// ***************************************************** ///
	/// EXCHANGE POPULATIONS      BETWEEN PROCESSORS AT       ///
	/// INTERFACES                                            ///
	/// ***************************************************** ///
	void Flow_solver::Data_Exchange(Parallel_MPI * MPI_parallel) {
		/// ------------------------------------------------------------------------> FACES
		if (MPI_parallel->processor_id != MASTER) {
			pop_group.exchange_data();
		}
	}
	void Flow_solver::register_recovery(IO_interface & io, const Parallel_MPI& MPI_parallel) {
		// population has a different storage order but IO expects the first 3 dimensions to be spatial
		io.add_field(pop, "flow_solver_pop");
		io.add_field(pop_old, "flow_solver_old_pop");
		io.add_field(density, "flow_solver_density");
		io.add_field(pressure, "flow_solver_pressure");
		io.add_field(velocity, "flow_solver_velocity");
		io.add_field(force, "flow_solver_force");
		io.add_field(is_solid, "flow_solver_is_solid");
		io.add_field(viscosity, "flow_solver_viscosity");
		io.add_scalar(physical_time, "flow_solver_physical_time");
#if defined LMNA_solver
		io.add_field(divU, "flow_solver_divU");
		io.add_scalar(p_th, "flow_solver_p_th");
		io.add_scalar(p_th_0, "flow_solver_p_th_0");
#endif
		io.add_custom_read([this, &MPI_parallel](const std::string&, int) {
			non_solid_lattice.compute_intervals(is_solid, MPI_parallel);
		});
	}
	/// ***************************************************** ///
	/// WRITE CURVED BOUNDARY DISTANCES                       ///
	/// ***************************************************** ///
	void Flow_solver::write_curved_boundary_data(const std::string& base_name, const Parallel_MPI& MPI_parallel, const stl_import& geo_stl) {
		size_t num_dirs_total = 0;
		if (MPI_parallel.processor_id != MASTER) {
			int undefined_BC = 0;
			for (const Flow_boundary_data& boundary : boundaries) {
				for (const Flow_solid_boundary_node& node : boundary.node_data) {
					for (double d : node.distance) {
						if (d < 0.0) {
							++undefined_BC;
							break;
						}
					}
					num_dirs_total += node.distance.size();
				}
			}
			if (undefined_BC > 0) std::cout << "(WARNING: FLOW BC, PROC " << MPI_parallel.processor_id << "), UNDEFINED BCS: " << undefined_BC << "\n";
		}

		/* Slave processors will write *.vtp files */
		if (MPI_parallel.processor_id != MASTER) {
			stringstream DB_filename;
			DB_filename << base_name << "_" << MPI_parallel.processor_id << ".vtp";
			ofstream BCDEBUG(DB_filename.str(), fstream::trunc);
			/* Write VTK header */
			BCDEBUG << "<?xml version=\"1.0\"?>\n";
			BCDEBUG << "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
			BCDEBUG << "<PolyData>\n";
			BCDEBUG << "<Piece NumberOfPoints=\"" << num_dirs_total << "\" NumberOfVerts=\"0\" NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">\n";
			BCDEBUG << "<Points>\n";
			BCDEBUG << "<DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";

			/* Write point coordinates here */
			for (const Flow_boundary_data& boundary : boundaries) {
				for (const Flow_solid_boundary_node& node : boundary.node_data) {
					/* Get coordinates in stl space */
					double xc, yc, zc;
					MPI_parallel.get_coordinates(node.idx[0], node.idx[1], node.idx[2],
					                             geo_stl.x_center, geo_stl.y_center, geo_stl.z_center,
					                             xc, yc, zc);

					for (int alpha : node.directions) {
						BCDEBUG << setprecision(14) << xc - global_parameters.D_x * c_alpha[alpha][0] * 1e-5 << " ";
						BCDEBUG << setprecision(14) << yc - global_parameters.D_x * c_alpha[alpha][1] * 1e-5 << " ";
						BCDEBUG << setprecision(14) << zc - global_parameters.D_x * c_alpha[alpha][2] * 1e-5 << "\n";
					}
				}
			}

			BCDEBUG << "</DataArray>\n";
			BCDEBUG << "</Points>\n";
			/* List of parameters to be stored for each point */
			BCDEBUG << "<PointData Scalars=\"Type\" Vectors=\"Normal\">\n";
			/* Boundary type */
			BCDEBUG << "<DataArray Name=\"Type\" type=\"Float64\" format=\"ascii\">\n";

			for (const Flow_boundary_data& boundary : boundaries) {
				for (const Flow_solid_boundary_node& node : boundary.node_data) {
					for (int i = 0; i < node.directions.size(); ++i) {
						BCDEBUG << setprecision(14) << static_cast<unsigned>(boundary.type) << "\n";
					}
				}
			}

			BCDEBUG << "</DataArray>\n";

			/* Normal vector */
			BCDEBUG << "<DataArray Name=\"Normal\" type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";

			for (const Flow_boundary_data& boundary : boundaries) {
				for (const Flow_solid_boundary_node& node : boundary.node_data) {
					for (int i = 0; i < node.directions.size(); ++i) {
						const int alpha = node.directions[i];
						const double dist = node.distance[i];
						const double length_c = sqrt(sqr(c_alpha[alpha][0]) + sqr(c_alpha[alpha][1]) + sqr(c_alpha[alpha][2]));

						BCDEBUG << setprecision(14) << -1.0 * dist * global_parameters.D_x * c_alpha[alpha][0] / length_c << " ";
						BCDEBUG << setprecision(14) << -1.0 * dist * global_parameters.D_x * c_alpha[alpha][1] / length_c << " ";
						BCDEBUG << setprecision(14) << -1.0 * dist * global_parameters.D_x * c_alpha[alpha][2] / length_c << "\n";
					}
				}
			}

			BCDEBUG << "</DataArray>\n";
			BCDEBUG << "</PointData>\n";
			BCDEBUG << "</Piece>\n";
			BCDEBUG << "</PolyData>\n";
			BCDEBUG << "</VTKFile>\n";
			BCDEBUG.close();
		}

		/* Slave processors will write *.pvtp files */
		if (MPI_parallel.processor_id == MASTER) {
			ofstream BCDEBUG(base_name + ".pvtp");
			BCDEBUG << "<?xml version=\"1.0\"?>\n";
			BCDEBUG << "<VTKFile type=\"PPolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
			BCDEBUG << "<PPolyData GhostLevel=\"0\">\n";
			BCDEBUG << "<PPointData Scalars=\"Type\" Vectors=\"Normal\">\n";
			BCDEBUG << "<PDataArray type=\"Float64\" Name=\"Type\"/>\n";
			BCDEBUG << "<PDataArray type=\"Float64\" NumberOfComponents=\"3\" Name=\"Normal\"/>\n";
			BCDEBUG << "</PPointData>\n";
			BCDEBUG << "<PPoints>\n";
			BCDEBUG << "<PDataArray type=\"Float64\" NumberOfComponents=\"3\"/>\n";
			BCDEBUG << "</PPoints>\n";

			for (int i = 1; i < MPI_parallel.num_processors; i++) {
				BCDEBUG << "<Piece Source=\"Flow_Curved_Boundary_Conditions_" << i << ".vtp\"/>\n";
			}

			BCDEBUG << "</PPolyData>\n";
			BCDEBUG << "</VTKFile>\n";
			BCDEBUG.close();
		}
	}
	/// ***************************************************** ///
	/// COMPUTE STRESS TENSOR WITH LB APPROXIMATION           ///
	/// ***************************************************** ///
	void Flow_solver::Stress_tensor(unsigned X, unsigned int Y, unsigned int Z) {
		double Hxx, Hyy, Hzz, Hxy, Hxz, Hyz;
		/// Set stress variable to zero
		Stress[0][0] = 0.;
		Stress[1][1] = 0.;
		Stress[2][2] = 0.;
		Stress[0][1] = 0.;
		Stress[0][2] = 0.;
		Stress[1][2] = 0.;
		/// Compute a^(neq)_2 components
		for (unsigned alpha = 0; alpha < Discrete_Velocity; alpha++) {
			Hxx = c_alpha[alpha][0] * c_alpha[alpha][0] - (1. / c_s2);
			Stress[0][0] = Stress[0][0] + Hxx * (pop_old[{X, Y, Z, alpha}] - pop_eq[alpha]);

			Hyy = c_alpha[alpha][1] * c_alpha[alpha][1] - (1. / c_s2);
			Stress[1][1] = Stress[1][1] + Hyy * (pop_old[{X, Y, Z, alpha}] - pop_eq[alpha]);

			Hzz = c_alpha[alpha][2] * c_alpha[alpha][2] - (1. / c_s2);
			Stress[2][2] = Stress[2][2] + Hzz * (pop_old[{X, Y, Z, alpha}] - pop_eq[alpha]);

			Hxy = c_alpha[alpha][0] * c_alpha[alpha][1];
			Stress[0][1] = Stress[0][1] + Hxy * (pop_old[{X, Y, Z, alpha}] - pop_eq[alpha]);

			Hxz = c_alpha[alpha][0] * c_alpha[alpha][2];
			Stress[0][2] = Stress[0][2] + Hxz * (pop_old[{X, Y, Z, alpha}] - pop_eq[alpha]);

			Hyz = c_alpha[alpha][1] * c_alpha[alpha][2];
			Stress[1][2] = Stress[1][2] + Hyz * (pop_old[{X, Y, Z, alpha}] - pop_eq[alpha]);
		}
		Stress[1][0] = Stress[0][1];
		Stress[2][0] = Stress[0][2];
		Stress[2][1] = Stress[1][2];
	}
	/// ***************************************************** ///
	/// APPLY MODEL FOR COMPUTATION OF VISCOSITY              ///
	/// ***************************************************** ///
	void Flow_solver::Diffusion_Coefficient_computation(Thermal_solver * Thermal, Species_solver * Species, Parallel_MPI * MPI_parallel) {
		if (MPI_parallel->processor_id != MASTER) {
			constexpr double T_star = 273.;
			constexpr double S = 110.5;
#if defined Flow_With_Species
			const double rho_star = p_th_0 * Species->molar_mass_av[{0, 0, 0}] / (R_GAS * T_star);
#else
		double rho_star = p_th_0 * M_av / (R_GAS * T_star);
#endif
			for (unsigned X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
				for (unsigned Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
					for (unsigned Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
						if (is_solid[{X, Y, Z}] == FALSE) {
							viscosity[{X, Y, Z}] = Sutherland_viscosity(rho_star * fluid_constant_viscosity, T_star / Thermal->T_0, S / Thermal->T_0, Thermal->temperature[{X, Y, Z}]) / (rho_0 * density[{X, Y, Z}]);
						}
					}
				}
			}

			for (const Flow_boundary_data& boundary : boundaries) {
				for (const Flow_fluid_boundary_node& node : boundary.fluid_node_data) {
					const Flat_index pos = node.flat_idx;
					const Flat_index pos_p = pos - viscosity.flat_index(node.n);
					viscosity[pos] = Sutherland_viscosity(rho_star * fluid_constant_viscosity, T_star / Thermal->T_0, S / Thermal->T_0, Thermal->temperature[pos]) / (rho_0 * density[pos]);
					viscosity[pos_p] = Sutherland_viscosity(rho_star * fluid_constant_viscosity, T_star / Thermal->T_0, S / Thermal->T_0, Thermal->temperature[pos_p]) / (rho_0 * density[pos_p]);
				}
			}
		}
	}
	/// ***************************************************** ///
	/// APPLY MODEL FOR COMPUTATION OF VISCOSITY              ///
	/// ***************************************************** ///
	// calculates the diffusion coefficient (kinematic viscosity) for a fluid Flow simulation with constant viscosity.
	//  Additionally, it handles boundary nodes by mirroring the viscosity values across the boundaries,
	// ensuring uniform viscosity throughout the simulation domain.
	void Flow_solver::Diffusion_Coefficient_const_viscosity(Thermal_solver * Thermal, Species_solver * Species, Parallel_MPI * MPI_parallel) {
		if (MPI_parallel->processor_id != MASTER) {
			for (unsigned X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
				for (unsigned Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
					for (unsigned Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
						if (is_solid[{X, Y, Z}] == FALSE) {
							/// KINEMATIC VISCOSITY
							viscosity[{X, Y, Z}] = fluid_constant_viscosity;
						}
					}
				}
			}
			for (const Flow_boundary_data& boundary : boundaries) {
				for (const Flow_fluid_boundary_node& node : boundary.fluid_node_data) {
					const Flat_index pos = node.flat_idx;
					const Flat_index pos_p = pos - viscosity.flat_index(node.n);
					viscosity[pos] = fluid_constant_viscosity;
					viscosity[pos_p] = fluid_constant_viscosity;
				}
			}
		}
	}
	FLOWINI::FLOWINI(){};
	FLOWINI::~FLOWINI(){};
	/// ***************************************************** ///
	/// INIALIZE LIST OF INITIAL CONDITIONS                   ///
	/// ***************************************************** ///
	void FLOWINI::Initialize() {
		FLOWINIFunction["KIDA3D"] = &KIDA3D;
		FLOWINIFunction["TAYLORGREEN3D"] = &TAYLORGREEN3D;
		FLOWINIFunction["TAYLORGREEN2D"] = &TAYLORGREEN2D;
		FLOWINIFunction["CONVECTEDVORTEX"] = &CONVECTEDVORTEX;
		FLOWINIFunction["PERIODICSHEARLAYER"] = &PERIODICSHEARLAYER;
		FLOWINIFunction["ACOUSTICWAVES"] = &ACOUSTICWAVES;
		FLOWINIFunction["USERDEFINED"] = &USERDEFINED;
		FLOWINIFunction["USERDEFINEDFLUCT"] = &USERDEFINEDFLUCT;
	}
	/// ***************************************************** ///
	/// MONITORS BLOW UP BASED ON VELOCITY                    ///
	/// ***************************************************** ///
	void Flow_solver::simulation_divergence_monitor(Parallel_MPI * MPI_parallel) {
		int errors = 0;
		int errors_global = 0;
		if (MPI_parallel->processor_id != MASTER) {
			non_solid_lattice.update([this, &errors](Flat_index idx) {
				bool U = debug::undefined_number(velocity(idx, 0));
				bool V = debug::undefined_number(velocity(idx, 1));
				bool W = debug::undefined_number(velocity(idx, 2));
				bool R = debug::undefined_number(density[idx]);
				if (U || V || W || R) { errors++; }
			});
			if (errors != 0) std::cout << "(ERROR) : " << errors << " UNDEFINED POINTS IN PROCESSOR " << MPI_parallel->processor_id << "\n";
		}
		MPI_Allreduce(&errors, &errors_global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
		if (errors_global != 0 && MPI_parallel->processor_id == MASTER) std::cout << "(ERROR) : END OF CODE DUE TO DIVERGENCE IN FLOW SOLVER\n";
		if (errors_global != 0) exit(EXIT_FAILURE);
	}
	Vec3 generate_turbulence(double intensity) {
		std::uniform_real_distribution<double> dist(-0.5, 0.5);
		const double turbulence_magnitude = intensity * dist(g_random);
		const double turbulence_direction_phi = M_PI * 0.5 * dist(g_random);
		const double cos_phi = cos(turbulence_direction_phi);
		const double sin_phi = sin(turbulence_direction_phi);
		return {cos_phi * cos_phi * turbulence_magnitude,
		        cos_phi * sin_phi * turbulence_magnitude,
		        sin_phi * turbulence_magnitude};
	}
	Vec3 Flow_boundary_data::get_turbulence() const {
		if (turbulence_intensity) {
			return generate_turbulence(turbulence_intensity);
		}
		return Vec3{};
	}
	void Flow_boundary_data::remove_nodes(std::vector<size_t> fluid_idx) {
		std::sort(fluid_idx.begin(), fluid_idx.end());
		//	ASSERT(std::is_sorted(fluid_idx.begin(), fluid_idx.end()));

		std::set<size_t> solid_idx;

		// remove fluid nodes and collect solid nodes
		for (auto it = fluid_idx.rbegin(); it != fluid_idx.rend(); ++it) {
			const Flow_fluid_boundary_node& fluid_node = fluid_node_data[*it];
			solid_idx.insert(fluid_node.solid_boundary_idx.begin(), fluid_node.solid_boundary_idx.end());
			fluid_node_data.erase(fluid_node_data.begin() + *it);
		}

		for (auto it = solid_idx.rbegin(); it != solid_idx.rend(); ++it) {
			node_data.erase(node_data.begin() + *it);
		}
	}
	void Flow_boundary_data::move_nodes(const std::vector<size_t>& fluid_idx_slf, const std::vector<size_t>& fluid_idx_dest, Flow_boundary_data& dest) {
		ASSERT(fluid_idx_slf.size() == fluid_idx_dest.size());

		for (size_t k = 0; k < fluid_idx_slf.size(); ++k) {
			const auto& fluid_corner_slf = fluid_node_data[fluid_idx_slf[k]];
			auto& fluid_corner_dest = dest.fluid_node_data[fluid_idx_dest[k]];
			// add solid nodes
			for (size_t solid_node_idx : fluid_corner_slf.solid_boundary_idx) {
				fluid_corner_dest.solid_boundary_idx.push_back(dest.node_data.size());
				dest.node_data.push_back(node_data[solid_node_idx]);
			}
			// move over directions
			fluid_corner_dest.directions.insert(fluid_corner_dest.directions.end(), fluid_corner_slf.directions.begin(), fluid_corner_slf.directions.end());

			// ensure that everything is in order again
			std::sort(fluid_corner_dest.directions.begin(), fluid_corner_dest.directions.end());
			std::sort(fluid_corner_dest.solid_boundary_idx.begin(), fluid_corner_dest.solid_boundary_idx.end());
		}

		remove_nodes(fluid_idx_slf);
	}
	Flow_solver::~Flow_solver(){};

	/// ***************************************************** ///
	/// GET VISCOSITY USING SUTHERLAND MODEL                  ///
	/// ***************************************************** ///
	double Sutherland_viscosity(double mu_star, double T_star, double S, double T) {
		double mu = 0;
		//    double S = 110.5;                    /* [S] = K */
		//    double T_star = 298.;                /* [T_star] = K */
		//    double nu_star = nu_0;//1.782e-5;           /* [nu_star] = m/s^2 */

		mu = mu_star * sqrt(pow(T / (double)T_star, 3)) * (T_star + S) / (double)(T + S);
		/// nu = nu_star * pow(T/(double)T_star,.69);
		return mu;
	}
	/// ***************************************************** ///
	/// INITIAL CONDITIONS: 3-D KIDA VORTEX                   ///
	/// ***************************************************** ///
	void KIDA3D(Vector_field & u, Scalar_field & dens, Vector_field & force, Scalar_field & viscosity, Solid_field & solid, double N_x, double N_y, double N_z, double dx, double dt, double rho_0, double c_s2, const std::string& filename, int Zones, Parallel_MPI* MPI_parallel) {
		///----------------------------------------------------------------------///
		///                      TAYLOR-GREEN VORTEXE                            ///
		///----------------------------------------------------------------------///
		unsigned int xx, yy, zz, X, Y, Z;
		if (MPI_parallel->processor_id != MASTER) {
			double Ini_vel[3], Ini_density;
			int type, index;
			/// Open input file
			string input_filename(filename);
			input_filename += ".dat";
			ifstream input_file;  // File is open for READING
			input_file.open(input_filename.c_str(), ios::binary);
			string Line1;
			char comment_indicator = 'k';
			int beg_line;

			input_file.clear();
			input_file.seekg(0, ios::beg);
			do {
				std::getline(input_file, Line1);
			} while (Line1.find("c\tFlow Field Initial Conditions") == string::npos);
			std::getline(input_file, Line1);
			do {
				beg_line = input_file.tellg();
				input_file >> comment_indicator;
				if (comment_indicator == '#') {
					std::getline(input_file, Line1);
				}
			} while (comment_indicator == '#');
			input_file.seekg(beg_line);

			input_file >> index;
			input_file >> type;
			input_file >> Ini_density;
			Ini_density /= rho_0;
			for (int d = 0; d < 3; d++) {
				input_file >> Ini_vel[d];
				Ini_vel[d] = Ini_vel[d] * dt / (double)dx;
			}
			if (MPI_parallel->processor_id == (MASTER + 1)) {
				std::cout << "Kida Initial conditions \n";
				std::cout << "Zone : " << index << std::endl;
				std::cout << "Type : " << type << std::endl;
				std::cout << "Initial density : " << Ini_density << std::endl;
				std::cout << "Initial velocity : " << Ini_vel[0] << " " << Ini_vel[1] << " " << Ini_vel[2] << std::endl;
			}
			for (X = 0; X < MPI_parallel->dev_end[0]; X++) {
				xx = fmod(X - MPI_parallel->start_XYZ2[0] + MPI_parallel->start_XYZ[0] + N_x + 1, N_x);
				for (Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
					yy = fmod(Y - MPI_parallel->start_XYZ2[1] + MPI_parallel->start_XYZ[1] + N_y, N_y);
					for (Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
						zz = fmod(Z - MPI_parallel->start_XYZ2[2] + MPI_parallel->start_XYZ[2] + N_z, N_z);
						u[{X, Y, Z, 0}] = Ini_vel[0] * sin(2 * M_PI * (xx + 0.5) / (double)N_x)
						                  * (cos(3. * 2 * M_PI * (yy + 0.5) / (double)N_y) * cos(2 * M_PI * (zz + 0.5) / (double)N_z)
						                     - cos(2 * M_PI * (yy + 0.5) / (double)N_y) * cos(3. * 2 * M_PI * (zz + 0.5) / (double)N_z));
						u[{X, Y, Z, 1}] = Ini_vel[0] * sin(2 * M_PI * (yy + 0.5) / (double)N_y)
						                  * (cos(3. * 2 * M_PI * (zz + 0.5) / (double)N_z) * cos(2 * M_PI * (xx + 0.5) / (double)N_x)
						                     - cos(2 * M_PI * (zz + 0.5) / (double)N_z) * cos(3. * 2 * M_PI * (xx + 0.5) / (double)N_x));
						u[{X, Y, Z, 2}] = Ini_vel[0] * sin(2 * M_PI * (zz + 0.5) / (double)N_z)
						                  * (cos(3. * 2 * M_PI * (xx + 0.5) / (double)N_x) * cos(2 * M_PI * (yy + 0.5) / (double)N_y)
						                     - cos(2 * M_PI * (xx + 0.5) / (double)N_x) * cos(3. * 2 * M_PI * (yy + 0.5) / (double)N_y));
						force[{X, Y, Z, 0}] = 0;
						force[{X, Y, Z, 1}] = 0;
						force[{X, Y, Z, 2}] = 0;
						dens[{X, Y, Z}] = 1.;
						solid[{X, Y, Z}] = -1;
					}
				}
			}
		}
	}
	/// ***************************************************** ///
	/// INITIAL CONDITIONS: 3-D TAYLOR-GREEN VORTEX           ///
	/// ***************************************************** ///
	void TAYLORGREEN3D(Vector_field & u, Scalar_field & dens, Vector_field & force, Scalar_field & viscosity, Solid_field & solid, double N_x, double N_y, double N_z, double dx, double dt, double rho_0, double c_s2, const std::string& filename, int Zones, Parallel_MPI* MPI_parallel) {
		///----------------------------------------------------------------------///
		///                      TAYLOR-GREEN VORTEXE                            ///
		///----------------------------------------------------------------------///
		unsigned int xx, yy, zz, X, Y, Z;
		if (MPI_parallel->processor_id != MASTER) {
			double Ini_vel[3], Ini_density;
			int type, index;
			/// Open input file
			string input_filename(filename);
			input_filename += ".dat";
			ifstream input_file;  // File is open for READING
			input_file.open(input_filename.c_str(), ios::binary);
			string Line1;
			char comment_indicator = 'k';
			int beg_line;

			input_file.clear();
			input_file.seekg(0, ios::beg);
			do {
				std::getline(input_file, Line1);
			} while (Line1.find("c\tFlow Field Initial Conditions") == string::npos);
			std::getline(input_file, Line1);
			do {
				beg_line = input_file.tellg();
				input_file >> comment_indicator;
				if (comment_indicator == '#') {
					std::getline(input_file, Line1);
				}
			} while (comment_indicator == '#');
			input_file.seekg(beg_line);

			input_file >> index;
			input_file >> type;
			input_file >> Ini_density;
			Ini_density /= rho_0;
			for (int d = 0; d < 3; d++) {
				input_file >> Ini_vel[d];
				Ini_vel[d] = Ini_vel[d] * dt / (double)dx;
			}
			if (MPI_parallel->processor_id == (MASTER + 1)) {
				std::cout << "Taylor-Green Initial conditions \n";
				std::cout << "Zone : " << index << std::endl;
				std::cout << "Type : " << type << std::endl;
				std::cout << "Initial density : " << Ini_density << std::endl;
				std::cout << "Initial velocity : " << Ini_vel[0] << " " << Ini_vel[1] << " " << Ini_vel[2] << std::endl;
			}
			for (X = 0; X < MPI_parallel->dev_end[0]; X++) {
				xx = fmod(X - MPI_parallel->start_XYZ2[0] + MPI_parallel->start_XYZ[0] + N_x, N_x);
				for (Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
					yy = fmod(Y - MPI_parallel->start_XYZ2[1] + MPI_parallel->start_XYZ[1] + N_y, N_y);
					for (Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
						zz = fmod(Z - MPI_parallel->start_XYZ2[2] + MPI_parallel->start_XYZ[2] + N_z, N_z);
						// Use double instead of int to preserve the fractional part
						double adjusted_x = (xx) / (double) (N_x);
						double adjusted_y = (yy) / (double) (N_y);
						double adjusted_z = (zz) / (double) (N_z);
						
						u[{X, Y, Z, 0}] = Ini_vel[0] * sin(2. * M_PI * adjusted_x) * cos(2. * M_PI * adjusted_y) * cos(2. * M_PI * adjusted_z);
						u[{X, Y, Z, 1}] = -Ini_vel[1] * cos(2. * M_PI * adjusted_x) * sin(2. * M_PI * adjusted_y) * cos(2. * M_PI * adjusted_z);
						u[{X, Y, Z, 2}] = 0.;
						force[{X, Y, Z, 0}] = 0;
						force[{X, Y, Z, 1}] = 0;
						force[{X, Y, Z, 2}] = 0;
						dens[{X, Y, Z}] = Ini_density * (1. / c_s2) + Ini_density * (sqr(Ini_vel[0]) / 16.) * (cos(4 * M_PI * adjusted_y) + cos(4 * M_PI * adjusted_x)) * (cos(4 * M_PI * adjusted_z) + 2.);
						solid[{X, Y, Z}] = -1;
					}
				}
			}
		}
	}
	/// ***************************************************** ///
	/// INITIAL CONDITIONS: 2-D TAYLOR-GREEN VORTEX           ///
	/// ***************************************************** ///
	void TAYLORGREEN2D(Vector_field & u, Scalar_field & dens, Vector_field & force, Scalar_field & viscosity, Solid_field & solid, double N_x, double N_y, double N_z, double dx, double dt, double rho_0, double c_s2, const std::string& filename, int Zones, Parallel_MPI* MPI_parallel) {
		///----------------------------------------------------------------------///
		///                      TAYLOR-GREEN VORTEXE                            ///
		///----------------------------------------------------------------------///
		double xx, yy;
		int X, Y, Z;
		if (MPI_parallel->processor_id != MASTER) {
			double Ini_vel[3], Ini_density;
			int type, index;
			/// Open input file
			ifstream input_file(filename + ".dat", ios::binary);  // File is open for READING

			input_file.clear();
			input_file.seekg(0, ios::beg);
			find_line_after_header(input_file, "c\tFlow Field Initial Conditions");
			find_line_after_comment(input_file);
			/* Read Data for Taylor-Green 2-D */
			input_file >> index;
			input_file >> type;
			input_file >> Ini_density;
			Ini_density /= rho_0;
			for (int d = 0; d < 3; d++) {
				input_file >> Ini_vel[d];
				Ini_vel[d] = Ini_vel[d] * dt / (double)dx;
			}
			if (MPI_parallel->processor_id == (MASTER + 1)) {
				std::cout << "Taylor-Green 2-D Initial conditions \n";
				std::cout << "Zone : " << index << std::endl;
				std::cout << "Type : " << type << std::endl;
				std::cout << "Initial density : " << Ini_density << std::endl;
				std::cout << "Initial velocity : " << Ini_vel[0] << " " << Ini_vel[1] << " " << Ini_vel[2] << std::endl;
			}
			for (X = 0; X < MPI_parallel->dev_end[0]; X++) {
				xx = fmod(X - MPI_parallel->start_XYZ2[0] + MPI_parallel->start_XYZ[0] + N_x, N_x);
				// fmod is used to ensure that the velocity field is periodic in the x direction (i.e. the last point is the same as the first point)
				for (Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
					yy = fmod(Y - MPI_parallel->start_XYZ2[1] + MPI_parallel->start_XYZ[1] + N_y, N_y);
					for (Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
						// const double zz = fmod(Z - MPI_parallel->start_XYZ2[2] + MPI_parallel->start_XYZ[2] + N_z, N_z);
						// this 0.5 is dx/2 in lattice units
						u[{X, Y, Z, 0}] = Ini_vel[0] * sin(2 * M_PI * (xx + 0.5) / (double)N_x) * cos(2 * M_PI * (yy + 0.5) / (double)N_y);
						u[{X, Y, Z, 1}] = -Ini_vel[1] * cos(2 * M_PI * (xx + 0.5) / (double)N_x) * sin(2 * M_PI * (yy + 0.5) / (double)N_y);
						dens[{X, Y, Z}] = Ini_density - .125 * Ini_density * c_s2 * (Ini_vel[0] * Ini_vel[0] + Ini_vel[1] * Ini_vel[1]) * (cos(4 * M_PI * (xx + 0.5) / (double)N_x) + cos(4 * M_PI * (yy + 0.5) / (double)N_y));
						u[{X, Y, Z, 2}] = 0.;
						force[{X, Y, Z, 0}] = 0;
						force[{X, Y, Z, 1}] = 0;
						force[{X, Y, Z, 2}] = 0;
						solid[{X, Y, Z}] = -1;
					}
				}
			}
		}
	}
	/// ***************************************************** ///
	/// INITIAL CONDITIONS: 2-D CONVECTED VORTEX              ///
	/// ***************************************************** ///
	void CONVECTEDVORTEX(Vector_field & u, Scalar_field & dens, Vector_field & force, Scalar_field & viscosity, Solid_field & solid, double N_x, double N_y, double N_z, double dx, double dt, double rho_0, double c_s2, const std::string& filename, int Zones, Parallel_MPI* MPI_parallel) {
		///----------------------------------------------------------------------///
		///                      CONVECTED    VORTEXE                            ///
		///----------------------------------------------------------------------///
		unsigned int xx, yy, X, Y, Z;
		if (MPI_parallel->processor_id != MASTER) {
			double Ini_vel[3], Ini_density;
			int type, index;
			double beta, R, T;
			/// Open input file
			string input_filename(filename);
			input_filename += ".dat";
			ifstream input_file;  // File is open for READING
			input_file.open(input_filename.c_str(), ios::binary);
			string Line1;
			char comment_indicator = 'k';
			int beg_line;

			input_file.clear();
			input_file.seekg(0, ios::beg);
			do {
				std::getline(input_file, Line1);
			} while (Line1.find("c\tFlow Field Initial Conditions") == string::npos);
			std::getline(input_file, Line1);
			do {
				beg_line = input_file.tellg();
				input_file >> comment_indicator;
				if (comment_indicator == '#') {
					std::getline(input_file, Line1);
				}
			} while (comment_indicator == '#');
			input_file.seekg(beg_line);

			input_file >> index;
			input_file >> type;
			input_file >> Ini_density;
			Ini_density /= rho_0;
			for (int d = 0; d < 3; d++) {
				input_file >> Ini_vel[d];
				Ini_vel[d] = Ini_vel[d] * dt / (double)dx;
			}
			input_file >> beta >> R >> T;
			T = T * c_s2;
			if (MPI_parallel->processor_id == (MASTER + 1)) {
				std::cout << "Convected vortex Initial conditions \n";
				std::cout << "Zone : " << index << std::endl;
				std::cout << "Type : " << type << std::endl;
				std::cout << "Initial density : " << Ini_density << std::endl;
				std::cout << "Vortex intensity(beta) : " << beta << "\t Radius(points) : " << R << std::endl;
				std::cout << "Initial velocity : " << Ini_vel[0] << " " << Ini_vel[1] << " " << Ini_vel[2] << std::endl;
			}
			double xc, yc;
			double x_center, y_center;
			x_center = N_x / 2;
			y_center = N_y / 2;
			for (X = 0; X < MPI_parallel->dev_end[0]; X++) {
				xx = fmod(X - MPI_parallel->start_XYZ2[0] + MPI_parallel->start_XYZ[0] + N_x, N_x);
				for (Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
					yy = fmod(Y - MPI_parallel->start_XYZ2[1] + MPI_parallel->start_XYZ[1] + N_y, N_y);
					for (Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
						// const double zz = fmod(Z - MPI_parallel->start_XYZ2[2] + MPI_parallel->start_XYZ[2] + N_z, N_z);

						xc = (xx + 0.5 - x_center) / (double)R;
						yc = (yy + 0.5 - y_center) / (double)R;
						u[{X, Y, Z, 0}] = Ini_vel[0] - beta * yc * exp(-0.5 * (sqr(xc) + sqr(yc)));
						u[{X, Y, Z, 1}] = Ini_vel[1] + beta * xc * exp(-0.5 * (sqr(xc) + sqr(yc)));
						u[{X, Y, Z, 2}] = 0.;
						force[{X, Y, Z, 0}] = 0;
						force[{X, Y, Z, 1}] = 0;
						force[{X, Y, Z, 2}] = 0;
						dens[{X, Y, Z}] = 1. - 0.5 * sqr(beta) * (c_s2 / T) * exp(-1. * (sqr(xc) + sqr(yc)));
						solid[{X, Y, Z}] = -1;
					}
				}
			}
		}
	}
	/// ***************************************************** ///
	/// INITIAL CONDITIONS: 2-D PERDIOIC SHEAR WAVE           ///
	/// ***************************************************** ///
	void PERIODICSHEARLAYER(Vector_field & u, Scalar_field & dens, Vector_field & force, Scalar_field & viscosity, Solid_field & solid, double N_x, double N_y, double N_z, double dx, double dt, double rho_0, double c_s2, const std::string& filename, int Zones, Parallel_MPI* MPI_parallel) {
		///----------------------------------------------------------------------///
		///                        COMMAND LINE INPUT                            ///
		///----------------------------------------------------------------------///
		unsigned int xx, yy, X, Y, Z;
		if (MPI_parallel->processor_id != MASTER) {
			double Ini_vel[3], Ini_density;
			double alpha, delta;
			int type, index;
			/// Open input file
			string input_filename(filename);
			input_filename += ".dat";
			ifstream input_file;  // File is open for READING
			input_file.open(input_filename.c_str(), ios::binary);
			string Line1;
			char comment_indicator = 'k';
			int beg_line;

			input_file.clear();
			input_file.seekg(0, ios::beg);
			do {
				std::getline(input_file, Line1);
			} while (Line1.find("c\tFlow Field Initial Conditions") == string::npos);
			std::getline(input_file, Line1);
			do {
				beg_line = input_file.tellg();
				input_file >> comment_indicator;
				if (comment_indicator == '#') {
					std::getline(input_file, Line1);
				}
			} while (comment_indicator == '#');
			input_file.seekg(beg_line);

			input_file >> index;
			input_file >> type;
			input_file >> Ini_density;
			Ini_density /= rho_0;
			for (int d = 0; d < 3; d++) {
				input_file >> Ini_vel[d];
				Ini_vel[d] = Ini_vel[d] * dt / (double)dx;
			}
			input_file >> alpha >> delta;
			if (MPI_parallel->processor_id == (MASTER + 1)) {
				std::cout << "Double Periodic Shear Layer Initial conditions \n";
				std::cout << "Zone : " << index << std::endl;
				std::cout << "Type : " << type << std::endl;
				std::cout << "Initial density : " << Ini_density << std::endl;
				std::cout << "alpha : " << alpha << "\t delta : " << delta << std::endl;
				std::cout << "Initial velocity : " << Ini_vel[0] << " " << Ini_vel[1] << " " << Ini_vel[2] << std::endl;
			}
			for (X = 0; X < MPI_parallel->dev_end[0]; X++) {
				xx = fmod(X - MPI_parallel->start_XYZ2[0] + MPI_parallel->start_XYZ[0] + N_x, N_x);
				for (Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
					yy = fmod(Y - MPI_parallel->start_XYZ2[1] + MPI_parallel->start_XYZ[1] + N_y, N_y);
					for (Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
						dens[{X, Y, Z}] = 1.;
						// const double zz = fmod(Z - MPI_parallel->start_XYZ2[2] + MPI_parallel->start_XYZ[2] + N_z, N_z);
						u[{X, Y, Z, 0}] = Ini_vel[0] * tanh(alpha * (0.25 - fabs((yy + 0.5) / N_y - 0.5)));
						u[{X, Y, Z, 1}] = Ini_vel[1] * delta * sin(2. * M_PI * ((xx + 0.5) / N_x + 0.25));
						u[{X, Y, Z, 2}] = 0.;
						solid[{X, Y, Z}] = -1;
						force[{X, Y, Z, 0}] = 0;
						force[{X, Y, Z, 1}] = 0;
						force[{X, Y, Z, 2}] = 0;
						dens[{X, Y, Z}] = 1.;
					}
				}
			}
		}
	}
	/// ***************************************************** ///
	/// INITIAL CONDITIONS: 2-D PERDIOIC ACOUSTIC WAVE        ///
	/// ***************************************************** ///
	void ACOUSTICWAVES(Vector_field & u, Scalar_field & dens, Vector_field & force, Scalar_field & viscosity, Solid_field & solid, double N_x, double N_y, double N_z, double dx, double dt, double rho_0, double c_s2, const std::string& filename, int Zones, Parallel_MPI* MPI_parallel) {
		///----------------------------------------------------------------------///
		///                        COMMAND LINE INPUT                            ///
		///----------------------------------------------------------------------///
		unsigned int xx, X, Y, Z;
		if (MPI_parallel->processor_id != MASTER) {
			double Ini_vel[3], Ini_density;
			double del_rho;
			int type, index;
			/// Open input file
			ifstream input_file(filename + ".dat", ios::binary);
			find_line_after_header(input_file, "c\tFlow Field Initial Conditions");
			find_line_after_comment(input_file);
			input_file >> index;
			input_file >> type;
			input_file >> Ini_density;
			Ini_density /= rho_0;
			for (int d = 0; d < 3; d++) {
				input_file >> Ini_vel[d];
				Ini_vel[d] = Ini_vel[d] * dt / dx;
			}
			input_file >> del_rho;
			del_rho /= rho_0;
			if (MPI_parallel->processor_id == (MASTER + 1)) {
				std::cout << "1-D Periodic Linear Acoustic Waves Initial conditions \n";
				std::cout << "Zone : " << index << std::endl;
				std::cout << "Type : " << type << std::endl;
				std::cout << "Initial density : " << Ini_density << std::endl;
				std::cout << "Initial velocity : " << Ini_vel[0] << " " << Ini_vel[1] << " " << Ini_vel[2] << std::endl;
				std::cout << "perturbation : " << del_rho << std::endl;
			}
			for (X = 0; X < MPI_parallel->dev_end[0]; X++) {
				xx = fmod(X - MPI_parallel->start_XYZ2[0] + MPI_parallel->start_XYZ[0] + N_x, N_x);
				for (Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
					for (Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
						dens[{X, Y, Z}] = del_rho * sin(2. * M_PI * (xx + 0.5) / N_x) + Ini_density;
						u[{X, Y, Z, 0}] = Ini_vel[0];
						u[{X, Y, Z, 1}] = Ini_vel[1];
						u[{X, Y, Z, 2}] = Ini_vel[2];
						solid[{X, Y, Z}] = -1;
						force[{X, Y, Z, 0}] = 0;
						force[{X, Y, Z, 1}] = 0;
						force[{X, Y, Z, 2}] = 0;
					}
				}
			}
		}
	}
	/// ***************************************************** ///
	/// INITIAL CONDITIONS: 2-D PERDIOIC SHEAR WAVE           ///
	/// ***************************************************** ///
	void SHEARWAVEDECAY(Vector_field & u, Scalar_field & dens, Vector_field & force, Scalar_field & viscosity, Solid_field & solid, double N_x, double N_y, double N_z, double dx, double dt, double rho_0, double c_s2, const std::string& filename, int Zones, Parallel_MPI* MPI_parallel) {
		///----------------------------------------------------------------------///
		///                        COMMAND LINE INPUT                            ///
		///----------------------------------------------------------------------///
		unsigned int yy, X, Y, Z;
		if (MPI_parallel->processor_id != MASTER) {
			double Ini_vel[3], Ini_density;
			int type, index;
			/// Open input file
			string input_filename(filename);
			input_filename += ".dat";
			ifstream input_file;  // File is open for READING
			input_file.open(input_filename.c_str(), ios::binary);

			input_file.clear();
			input_file.seekg(0, ios::beg);
			find_line_after_header(input_file, "c\tFlow Field Initial Conditions");
			find_line_after_comment(input_file);

			input_file >> index;
			input_file >> type;
			input_file >> Ini_density;
			Ini_density /= rho_0;
			for (int d = 0; d < 3; d++) {
				input_file >> Ini_vel[d];
				Ini_vel[d] = Ini_vel[d] * dt / (double)dx;
			}
			if (MPI_parallel->processor_id == (MASTER + 1)) {
				std::cout << "2-D Decaying Shear Wave Initial conditions \n";
				std::cout << "Zone : " << index << std::endl;
				std::cout << "Type : " << type << std::endl;
				std::cout << "Initial density : " << Ini_density << std::endl;
				std::cout << "Initial velocity : " << Ini_vel[0] << " " << Ini_vel[1] << " " << Ini_vel[2] << std::endl;
			}
			for (X = 0; X < MPI_parallel->dev_end[0]; X++) {
				// const double xx = fmod(X - MPI_parallel->start_XYZ2[0] + MPI_parallel->start_XYZ[0] + N_x, N_x);
				for (Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
					yy = fmod(Y - MPI_parallel->start_XYZ2[1] + MPI_parallel->start_XYZ[1] + N_y, N_y);
					for (Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
						dens[{X, Y, Z}] = Ini_density;
						// const double zz = fmod(Z - MPI_parallel->start_XYZ2[2] + MPI_parallel->start_XYZ[2] + N_z, N_z);
						u[{X, Y, Z, 0}] = Ini_vel[0] * sin(2 * M_PI * (yy + 0.5) / N_y);
						u[{X, Y, Z, 1}] = Ini_vel[1];
						u[{X, Y, Z, 2}] = Ini_vel[2];
						force[{X, Y, Z, 0}] = 0;
						force[{X, Y, Z, 1}] = 0;
						force[{X, Y, Z, 2}] = 0;
						solid[{X, Y, Z}] = -1;
					}
				}
			}
		}
	}
	/// ***************************************************** ///
	/// INITIAL CONDITIONS: USER-DEFINED VALUES (FROM INPUT   ///
	/// FILE)                                                 ///
	/// ***************************************************** ///
	void USERDEFINED(Vector_field & u, Scalar_field & dens, Vector_field & force, Scalar_field & viscosity, Solid_field & solid, double N_x, double N_y, double N_z, double dx, double dt, double rho_0, double c_s2, const std::string& filename, int Zones, Parallel_MPI* MPI_parallel) {
		if (MPI_parallel->is_master()) {
			return;
		}
		const std::vector<double> initial_vec(3, 0.0);
		std::vector<std::vector<double>> ini_vel(Zones + 1, initial_vec);
		std::vector<double> ini_density(Zones + 1, 0.0);
		std::vector<int> type(Zones + 1, 0.0);
		std::vector<Initial_field_slice> special_volumes;
		type[0] = 1;  /// -----> Solids are set to +1
		ifstream input_file(filename + ".dat", ios::binary);
		find_line_after_header(input_file, "c\tFlow Field Initial Conditions");
		if (MPI_parallel->processor_id == (MASTER + 1)) {
			std::cout << "FLOW: User defined Initial conditions \n";
			std::cout << "================================ \n";
		}
		for (size_t i = 0; i < static_cast<size_t>(Zones); i++) {
			find_line_after_comment(input_file);
			int index;
			bool is_extra = false;
			input_file >> index;
			if (index < 0) {
				is_extra = true;
				index = ini_density.size();
				ini_density.push_back(0.0);
				ini_vel.push_back(initial_vec);
				type.push_back(0);
				--i;
			}
			input_file >> type[index];
			input_file >> ini_density[index];
			ini_density[index] /= rho_0;
			for (int d = 0; d < 3; d++) {
				input_file >> ini_vel[index][d];
				ini_vel[index][d] = ini_vel[index][d] * dt / (double)dx;
			}
			if (is_extra) {
				special_volumes.push_back({});
				Initial_field_slice& slice = special_volumes.back();
				slice.index = index;
				// input_file >> slice;
			}
			if (MPI_parallel->processor_id == (MASTER + 1)) {
				std::cout << "Zone : " << index << "\t Type : " << type[index] << std::endl;
				std::cout << "Initial density : " << ini_density[index] << std::endl;
				std::cout << "Initial velocity : " << ini_vel[index][0] << " " << ini_vel[index][1] << " " << ini_vel[index][2] << std::endl;
				std::cout << "-------------------------------- \n";
			}
		}
		input_file.close();
		for (unsigned X = 0; X < MPI_parallel->dev_end[0]; X++) {
			for (unsigned Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
				for (unsigned Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
					dens[{X, Y, Z}] = ini_density[solid[{X, Y, Z}]];
					// viscosity[{X,Y,Z}] = Ini_viscosity[solid[{X,Y,Z}]];
					for (int d = 0; d < 3; d++) {
						u[{X, Y, Z, d}] = ini_vel[solid[{X, Y, Z}]][d];
						force[{X, Y, Z, d}] = 0;
					}
					solid[{X, Y, Z}] = type[solid[{X, Y, Z}]];
				}
			}
		}
		// TANH METHOD
		/*	const Vec3 coords = MPI_parallel->get_coordinates(idx);
		    const double r = slice.get_scale(coords);
		    // dens[idx] = r * ini_density[slice.index] + (1.0 - r) * dens[idx];
		    for (int d = 0; d < 3; d++) {
		        const Index_vec4 idx4 = {X, Y, Z, d};
		        u[idx4] = r * ini_vel[slice.index][d] + (1.0 - r) * u[idx4];
		    }
		*/
	}
	/// ***************************************************** ///
	/// INITIAL CONDITIONS: USER-DEFINED VALUES (FROM INPUT   ///
	/// FILE) WITH WHITE NOISE                                ///
	/// ***************************************************** ///
	void USERDEFINEDFLUCT(Vector_field & u, Scalar_field & dens, Vector_field & force, Scalar_field & viscosity, Solid_field & solid, double N_x, double N_y, double N_z, double dx, double dt, double rho_0, double c_s2, const std::string& filename, int Zones, Parallel_MPI* MPI_parallel) {
		///----------------------------------------------------------------------///
		///                        COMMAND LINE INPUT                            ///
		///----------------------------------------------------------------------///
		if (MPI_parallel->processor_id != MASTER) {
			double **Ini_vel, *Ini_density;
			double** Ini_vel_fluc;
			int* type;
			int X, Y, Z, d;

			Ini_vel = new double*[Zones + 1];
			Ini_vel_fluc = new double*[Zones + 1];
			Ini_density = new double[Zones + 1];
			type = new int[Zones + 1];
			int index;
			for (int i = 0; i < Zones + 1; i++) {
				Ini_vel[i] = new double[3];
				Ini_vel_fluc[i] = new double[3];
				type[i] = 0;
				Ini_density[i] = 0;
				Ini_vel[i][0] = 0;
				Ini_vel[i][1] = 0;
				Ini_vel_fluc[i][2] = 0;
				Ini_vel_fluc[i][0] = 0;
				Ini_vel_fluc[i][1] = 0;
				Ini_vel[i][2] = 0;
			}
			type[0] = 1;  /// -----> Solids are set to +1
			/// Open input file
			string input_filename(filename);
			input_filename += ".dat";
			ifstream input_file;  // File is open for READING
			input_file.open(input_filename.c_str(), ios::binary);

			input_file.clear();
			input_file.seekg(0, ios::beg);
			find_line_after_header(input_file, "c\tFlow Field Initial Conditions");
			if (MPI_parallel->processor_id == (MASTER + 1)) {
				std::cout << "User defined Initial conditions with noise\n";
				std::cout << "================================ \n";
			}
			for (int i = 0; i < Zones; i++) {
				find_line_after_comment(input_file);
				input_file >> index;
				input_file >> type[index];
				input_file >> Ini_density[index];
				Ini_density[index] /= rho_0;
				for (d = 0; d < 3; d++) {
					input_file >> Ini_vel[index][d];
					Ini_vel[index][d] = Ini_vel[index][d] * dt / (double)dx;
				}
				for (d = 0; d < 3; d++) {
					input_file >> Ini_vel_fluc[index][d];
					Ini_vel_fluc[index][d] = Ini_vel_fluc[index][d] * dt / (double)dx;
				}
				if (MPI_parallel->processor_id == (MASTER + 1)) {
					std::cout << "Zone : " << index << std::endl;
					std::cout << "Type : " << type[index] << std::endl;
					std::cout << "Initial density : " << Ini_density[index] << std::endl;
					std::cout << "Initial velocity : " << Ini_vel[index][0] << " " << Ini_vel[index][1] << " " << Ini_vel[index][2] << std::endl;
					std::cout << "Initial velocity noise: " << Ini_vel_fluc[index][0] << " " << Ini_vel_fluc[index][1] << " " << Ini_vel_fluc[index][2] << std::endl;
					std::cout << "-------------------------------- \n";
				}
			}
			for (X = 0; X < MPI_parallel->dev_end[0]; X++) {
				for (Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
					for (Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
						// const double zz = fmod(Z - MPI_parallel->start_XYZ2[2] + MPI_parallel->start_XYZ[2] + N_z, N_z);
						dens[{X, Y, Z}] = Ini_density[solid[{X, Y, Z}]];
						// viscosity[{X,Y,Z}] = Ini_viscosity[solid[{X,Y,Z}]];
						for (d = 0; d < 3; d++) {
							u[{X, Y, Z, d}] = Ini_vel[solid[{X, Y, Z}]][d] + (rand() / (double)RAND_MAX - 0.5) * Ini_vel[solid[{X, Y, Z}]][d];
							force[{X, Y, Z, d}] = 0;
						}
						solid[{X, Y, Z}] = type[solid[{X, Y, Z}]];
					}
				}
			}
		}
		return;
	}
		double FROB(double** A, double** B) {
			double C;
			C = A[0][0] * B[0][0] + A[1][0] * B[1][0] + A[2][0] * B[2][0]
			    + A[0][1] * B[0][1] + A[1][1] * B[1][1] + A[2][1] * B[2][1]
			    + A[0][2] * B[0][2] + A[1][2] * B[1][2] + A[2][2] * B[2][2];
			return C;
		}
		constexpr std::array<const char*, static_cast<size_t>(Non_uniform_boundary::Type::COUNT)> Non_uniform_boundary::type_names;
		///--------------------- Space-dependent BC: Read data from file
		void Non_uniform_boundary::data_input(const std::string& filename, const Parallel_MPI& MPI_parallel) {
			ifstream input_file(filename + ".dat", ios::binary);
			find_line_after_header(input_file, "c\tFlow Field Space-Dependent Boundary Conditions");
			find_line_after_comment(input_file);
			input_file >> number_of_BC;  // First read how many space-dependent boundaries there are

			index_of_BC.resize(number_of_BC);  // Resize arrays holding space-dependent boundary conditions data
			Dimension.resize(number_of_BC);
			type.resize(number_of_BC);
			center.resize(number_of_BC);
			radii.resize(number_of_BC);
			thickness.resize(number_of_BC);
			max_velocity.resize(number_of_BC);

			for (unsigned i = 0; i < number_of_BC; i++) {
				find_line_after_comment(input_file);
				std::string type_str;
				input_file >> index_of_BC[i] >> Dimension[i] >> type_str >> center[i][0] >> center[i][1] >> center[i][2]
					>> radii[i] >> thickness[i] >> max_velocity[i][0] >> max_velocity[i][1] >> max_velocity[i][2];
				auto it = std::find_if(type_names.begin(), type_names.end(), [&](const char* name) {
					return type_str == name;
				});
				if (it != type_names.end()) {
					type[i] = static_cast<Type>(std::distance(type_names.begin(), it));
				} else {
					ERROR_ABORT("Unknown non uniform boundary type \"" << type_str << "\".");
				}

				max_velocity[i] *= (global_parameters.D_t / global_parameters.D_x);
			}
			input_file.close();

			if (MPI_parallel.processor_id == MASTER + 1) {
				std::cout << "Space-Dependent Flow Boundary Condition" << endl;
				for (unsigned i = 0; i < number_of_BC; i++) {
					std::cout << "BC index :" << index_of_BC[i] << "\n";

					std::cout << "\tDimension :\t";

					if (Dimension[i] == 2) std::cout << "2-D\n";
					if (Dimension[i] == 3) std::cout << "3-D\n";

					std::cout << "\tProfile :\t" << type_names[static_cast<int>(type[i])] << endl;
					std::cout << "\tCenter (x,y,z):\t(" << center[i][0] << ", " << center[i][1] << ", " << center[i][2] << ")\n";
					std::cout << "\tRadii :\t(" << radii[i] << ")\n";
					std::cout << "\tThickness :\t(" << thickness[i] << ")\n";
					std::cout << "\tMaximal_u (u,v,w):\t(" << max_velocity[i][0] << ", " << max_velocity[i][1] << ", " << max_velocity[i][2] << ")\n";
				}
			}
		}
		///--------------------- Update velocities at boundaries(USE RIGHT BEFORE BC FUNCTION
		void Non_uniform_boundary::set_values(Flow_solver & Flow, const stl_import& geo_stl, const Parallel_MPI& MPI_parallel) const {
			if (MPI_parallel.is_master()) {
				return;
			}
			for (size_t i = 0; i < index_of_BC.size(); ++i) {
				auto it = std::find_if(Flow.boundaries.begin(), Flow.boundaries.end(), [&](const Flow_boundary_data& boundary) {
					return boundary.index == index_of_BC[i];
				});
				if (it != Flow.boundaries.end()) {
					Flow_boundary_data& boundary = *it;
					for (Flow_fluid_boundary_node& node : boundary.fluid_node_data) {
						node.v = get_velocity_by_node(i, node.idx, geo_stl, MPI_parallel);
					}
					for (Flow_solid_boundary_node& node : boundary.node_data) {
						node.v = get_velocity_by_node(i, node.idx, geo_stl, MPI_parallel);
					}
				} else {
					ERROR_ABORT("Boundary with index " << index_of_BC[i] << ", requested for non uniform bc, does not exist.");
				}
			}
		}
		Vec3 Non_uniform_boundary::get_velocity_by_node(size_t j, const Index_vec3& idx, const stl_import& geo_stl, const Parallel_MPI& MPI_parallel) const {
			const Vec3 pos = MPI_parallel.get_coordinates(idx, Vec3{geo_stl.x_center, geo_stl.y_center, geo_stl.z_center});

			Vec3 diff = pos - center[j];
			if (Dimension[j] == 2) {
				diff[2] = 0.0;
			}

			const double distance = sqrt(dot(diff, diff));
			return get_velocity(j, distance);
		}
		Vec3 Non_uniform_boundary::get_velocity(int j, double distance) const {
			double s = 0.0;
			if (type[j] == Type::POISEULLE) {
				if (distance < radii[j]) {
					s = 1. - sqr(distance / radii[j]);
				}
			} else if (type[j] == Type::TANH) {
				s = 0.5 * (1. + tanh(2. * (radii[j] - distance) / thickness[j]));
			}

			return max_velocity[j] * s;
		}
		/// ***************************************************** ///
		void Flow_rate(double* m_dot, Flow_solver* Flow_Field, Parallel_MPI* MPI_parallel) {
			unsigned int counter = 0;
			unsigned int counter_global = 0;
			double m_dot_global[3];
			m_dot[0] = 0.;
			m_dot[1] = 0.;
			m_dot[2] = 0.;
			if (MPI_parallel->processor_id != MASTER) {
				int X, Y, Z;
				for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
					for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
						for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
							if (Flow_Field->is_solid[{X, Y, Z}] == FALSE) {
								m_dot[0] += Flow_Field->velocity[{X, Y, Z, 0}];
								m_dot[1] += Flow_Field->velocity[{X, Y, Z, 1}];
								m_dot[2] += Flow_Field->velocity[{X, Y, Z, 2}];
								counter++;
							}
						}
					}
				}
			}
			MPI_Reduce(&counter, &counter_global, 1, MPI_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);
			MPI_Reduce(&m_dot[0], &m_dot_global[0], 3, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
			if (MPI_parallel->processor_id == MASTER) {
				m_dot_global[0] = m_dot_global[0] / (double)counter_global;
				m_dot_global[1] = m_dot_global[1] / (double)counter_global;
				m_dot_global[2] = m_dot_global[2] / (double)counter_global;
			}
			MPI_Bcast(&counter_global, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
			MPI_Bcast(&m_dot_global[0], 3, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
			m_dot[0] = m_dot_global[0];
			m_dot[1] = m_dot_global[1];
			m_dot[2] = m_dot_global[2];
		}
		void Fix_flow_rate(double* m_dot, double* m_dot_in, double* n, Flow_solver* Flow_Field, Parallel_MPI* MPI_parallel) {
			if (MPI_parallel->processor_id != MASTER) {
				int X, Y, Z;
				for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
					for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
						for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
							if (Flow_Field->is_solid[{X, Y, Z}] == FALSE) {
								Flow_Field->force[{X, Y, Z, 0}] = (m_dot_in[0] - m_dot[0]) * n[0];
								Flow_Field->force[{X, Y, Z, 1}] = (m_dot_in[1] - m_dot[1]) * n[1];
								Flow_Field->force[{X, Y, Z, 2}] = (m_dot_in[2] - m_dot[2]) * n[2];
							}
						}
					}
				}
			}
		}
		/// ***************************************************** ///
		/// ADD VORTICES IN THE DOMAIN (PARAMETERS RED FROM INPUT  ///
		/// ***************************************************** ///
		void add_vortex(std::string filename, Flow_solver * Flow, Parallel_MPI * MPI_parallel, stl_import * Geo, unsigned int N_x, unsigned int N_y, unsigned int N_z) {
			/// READ DATA
			//	int column_width = 40;
			/// Open input file
			string input_filename(filename);
			input_filename += ".dat";
			ifstream input_file;  // File is open for READING
			input_file.open(input_filename.c_str(), ios::binary);

			input_file.clear();
			input_file.seekg(0, ios::beg);
			find_line_after_header(input_file, "c\tVortex");
			find_line_after_comment(input_file);
			unsigned int number_of_vortices;
			input_file >> number_of_vortices;
			vector<double> Psi;  /// vortex power in m2/s
			Psi.resize(number_of_vortices);
			vector<double> xc;  /// vortex center x-coordinate in m
			xc.resize(number_of_vortices);
			vector<double> yc;  /// vortex center y-coordinate in m
			yc.resize(number_of_vortices);
			vector<double> rc;  /// vortex radius in m
			rc.resize(number_of_vortices);
			vector<int> direction;  /// vortex rotation direction: +1->clock-wise, -1->counter clock-wise
			direction.resize(number_of_vortices);
			for (unsigned int i = 0; i < number_of_vortices; i++) {
				find_line_after_comment(input_file);
				input_file >> Psi[i] >> xc[i] >> yc[i] >> rc[i] >> direction[i];
				Psi[i] *= (global_parameters.D_t / pow(global_parameters.D_x, 1));
			}
			input_file.close();
			if (MPI_parallel->processor_id == MASTER + 1) {
				stringstream vortex_filename;
				vortex_filename << "Alborz_Results/debug/vortex.dat";
				ofstream vortex_out;
				vortex_out.open(vortex_filename.str().c_str(), fstream::trunc);
				vortex_out << "List of Vortices" << endl;
				vortex_out << "=====================" << endl;
				vortex_out << "Index\tPsi\tx\ty\tr\tdirection\n";
				for (int i = 0; i < xc.size(); i++) {
					vortex_out << i + 1 << "\t" << Psi[i] << "\t" << xc[i] << "\t" << yc[i] << "\t" << rc[i] << "\t" << direction[i] << "\n";
				}
				vortex_out.close();
			}
			unsigned int X, Y, Z, i;
			double xx, yy, x_dist, y_dist, r_dist, theta;
			double epsilon = 1e-20;
			if (MPI_parallel->processor_id != MASTER) {
				for (X = 0; X < MPI_parallel->dev_end[0]; X++) {
					xx = fmod(X - MPI_parallel->start_XYZ2[0] + MPI_parallel->start_XYZ[0] + N_x, N_x) * global_parameters.D_x + Geo->x_center;
					for (Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
						yy = fmod(Y - MPI_parallel->start_XYZ2[1] + MPI_parallel->start_XYZ[1] + N_y, N_y) * global_parameters.D_x + Geo->y_center;
						for (Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
							// const double zz = fmod(Z - MPI_parallel->start_XYZ2[2] + MPI_parallel->start_XYZ[2] + N_z, N_z) * global_parameters.D_x + Geo->z_center;
							for (i = 0; i < number_of_vortices; i++) {
								x_dist = xx - xc[i];
								y_dist = yy - yc[i];
								r_dist = sqrt(sqr(x_dist) + sqr(y_dist));
								// get angle
								theta = acos(x_dist / (r_dist + epsilon));
								if (y_dist < 0) theta = -theta;
								// if (r_dist <= rc[i]) {
								Flow->velocity[{X, Y, Z, 0}] += (Psi[i] / (2. * M_PI * (r_dist + epsilon))) * (1. - exp(-sqr(r_dist / rc[i]))) * direction[i] * sin(theta);
								Flow->velocity[{X, Y, Z, 1}] += (Psi[i] / (2. * M_PI * (r_dist + epsilon))) * (1. - exp(-sqr(r_dist / rc[i]))) * direction[i] * (-cos(theta));
								// }
							}
						}
					}
				}
			}
		}
		/// ***************************************************** ///
		/// PASSOT-POUQUET SPECTRUM                               ///
		/// ***************************************************** ///
		double passot_pouquet_spec(double k, double ke, double urms) {
			const double r1 = k / ke;
			const double tke = 16. * sqrt(2 / M_PI) * (urms * urms / ke) * r1 * r1 * r1 * r1 * exp(-2.0 * r1 * r1);
			return tke;
		}
		/// ***************************************************** ///
		/// INITIALIZE HOMOGENEOUS ISOTROPIC TURBULENCE GIVEN A   ///
		/// SPECTRUM                                              ///
		/// ***************************************************** ///
		void add_HIT(std::string filename, Flow_solver * Flow, Parallel_MPI * MPI_parallel, stl_import * Geo, unsigned int N_x, unsigned int N_y, unsigned int N_z) {
			///----------------------------------------------------------------------///
			///                      Homogeneous Isotropic Turbulence                ///
			///----------------------------------------------------------------------///
			unsigned int X, Y, Z;
			double xx, yy, zz;
			if (MPI_parallel->processor_id != MASTER) {
				/// READ DATA
				//	int column_width = 40;
				/// Open input file
				string input_filename(filename);
				input_filename += ".dat";
				ifstream input_file;  // File is open for READING
				input_file.open(input_filename.c_str(), ios::binary);

				input_file.clear();
				input_file.seekg(0, ios::beg);
				find_line_after_header(input_file, "c\tHIT");
				find_line_after_comment(input_file);

				double u_rms;
				double lx, ly, lz, le, x_start, y_start, z_start, x_end, y_end, z_end;
				int nModes;
				/// Open input file
				input_file >> nModes;
				input_file >> u_rms;
				input_file >> le;
				input_file >> x_start >> y_start >> z_start;
				input_file >> x_end >> y_end >> z_end;
				input_file >> lx >> ly >> lz;
				if (MPI_parallel->processor_id == (MASTER + 1)) {
					std::cout << "Homogeneous Isotropic Turbulence\n";
					std::cout << "u_rms : " << u_rms * global_parameters.D_t / global_parameters.D_x << std::endl;
					std::cout << "number of modes : " << nModes << std::endl;
					std::cout << "box size : " << lx << " x " << ly << " x " << lz << std::endl;
				}
				// specify which spectrum to use.
				double (*which_spec)(double, double, double);
				/// which_spec = &karman_spec;
				which_spec = &passot_pouquet_spec;

				double L = lx;
				const double km0 = 2.0 * M_PI / L;
				// set the x, y, and z resolutions. Currently support same res in all directions
				const int Nx = lx / global_parameters.D_x;
				const int Ny = ly / global_parameters.D_x;
				const int Nz = lz / global_parameters.D_x;

				// find dx and half dx (hdx) et al.
				const double dx = lx / Nx;
				const double hdx = dx / 2.0;
				const double dy = ly / Ny;
				const double hdy = dy / 2.0;
				const double dz = lz / Nz;
				const double hdz = dz / 2.0;

				//______________________________________________________________________
				// compute wave arrays using standard random number generation
				double* phi = (double*)malloc(sizeof(double) * (nModes + 1));
				double* nu = (double*)malloc(sizeof(double) * (nModes + 1));
				double* theta = (double*)malloc(sizeof(double) * (nModes + 1));
				double* psi = (double*)malloc(sizeof(double) * (nModes + 1));

				// seed the random number generator
				time_t t;
				srand((unsigned)time(&t));
				srand(0);
				for (int i = 0; i <= nModes; ++i) {
					phi[i] = 2.0 * M_PI * (double)rand() / RAND_MAX;
					nu[i] = (double)rand() / RAND_MAX;
					theta[i] = acos(2.0 * nu[i] - 1.0);
					psi[i] = M_PI * (double)rand() / RAND_MAX - M_PI / 2.0;
				}

				// maximum wave number supported on this grid
				const double kmmax = M_PI / dx;
				// find spacing in wave-space
				const double dk = (kmmax - km0) / nModes;
				// create an array of wave numbers
				double* km = (double*)malloc(sizeof(double) * (nModes + 1));
				for (int i = 0; i <= nModes; ++i) {
					km[i] = km0 + i * dk;
				}
				// create wave vector (kx, ky, kz)
				double* kx = (double*)malloc(sizeof(double) * (nModes + 1));
				double* ky = (double*)malloc(sizeof(double) * (nModes + 1));
				double* kz = (double*)malloc(sizeof(double) * (nModes + 1));
				for (int i = 0; i <= nModes; ++i) {
					kx[i] = sin(theta[i]) * cos(phi[i]) * km[i];
					ky[i] = sin(theta[i]) * sin(phi[i]) * km[i];
					kz[i] = cos(theta[i]) * km[i];
				}
				//________________________________________________________
				// ENFORCE MASS CONSERVATION
				double* ktx = (double*)malloc(sizeof(double) * (nModes + 1));
				double* kty = (double*)malloc(sizeof(double) * (nModes + 1));
				double* ktz = (double*)malloc(sizeof(double) * (nModes + 1));

				for (int i = 0; i <= nModes; ++i) {
					ktx[i] = sin(kx[i] * hdx) / dx;
					kty[i] = sin(ky[i] * hdy) / dy;
					ktz[i] = sin(kz[i] * hdz) / dz;
				}
				// find the direction vector sigma
				double* sxm = (double*)malloc(sizeof(double) * (nModes + 1));
				double* sym = (double*)malloc(sizeof(double) * (nModes + 1));
				double* szm = (double*)malloc(sizeof(double) * (nModes + 1));
				// CREATE VECTOR ZETA AND MAKE SIGMA = ZETA x K
				for (int i = 0; i <= nModes; ++i) {
					phi[i] = 2.0 * M_PI * (double)rand() / RAND_MAX;
					nu[i] = (double)rand() / RAND_MAX;
					theta[i] = acos(2.0 * nu[i] - 1.0);
				}
				for (int i = 0; i <= nModes; ++i) {
					double zetax = sin(theta[i]) * cos(phi[i]);
					double zetay = sin(theta[i]) * sin(phi[i]);
					double zetaz = cos(theta[i]);
					// take cross product of zeta with k_tilde
					sxm[i] = (zetay * ktz[i] - zetaz * kty[i]);
					sym[i] = -(zetax * ktz[i] - zetaz * ktx[i]);
					szm[i] = (zetax * kty[i] - zetay * ktx[i]);
					// now make sigma a unit vector
					double smag = 1.0 / sqrt(sxm[i] * sxm[i] + sym[i] * sym[i] + szm[i] * szm[i]);
					sxm[i] = sxm[i] * smag;
					sym[i] = sym[i] * smag;
					szm[i] = szm[i] * smag;
				}
				//________________________________________________________
				// update the sigma vector with the amplitude (reduces amount of computation)
				double espec;
				for (int i = 0; i <= nModes; ++i) {
					espec = 2.0 * sqrt(which_spec(km[i], M_PI / le, u_rms) * dk);
					sxm[i] *= espec;
					sym[i] *= espec;
					szm[i] *= espec;
				}
				//___________________________________________________________________________________________
				// compute the Fourier series at each point!
				for (X = 0; X < MPI_parallel->dev_end[0]; X++) {
					xx = fmod(X - MPI_parallel->start_XYZ2[0] + MPI_parallel->start_XYZ[0] + N_x, N_x) * global_parameters.D_x + Geo->x_center;
					for (Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
						yy = fmod(Y - MPI_parallel->start_XYZ2[1] + MPI_parallel->start_XYZ[1] + N_y, N_y) * global_parameters.D_x + Geo->y_center;
						for (Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
							zz = fmod(Z - MPI_parallel->start_XYZ2[2] + MPI_parallel->start_XYZ[2] + N_z, N_z) * global_parameters.D_x + Geo->z_center;
							if (xx >= x_start && yy >= y_start && zz >= z_start
							    && xx <= x_end && yy <= y_end && zz <= z_end) {
								for (int m = 0; m <= nModes; ++m) {
									const double arg = kx[m] * (xx - x_start) + ky[m] * (yy - y_start) + kz[m] * (zz - z_start) - psi[m];
									Flow->velocity[{X, Y, Z, 0}] += cos(arg - kx[m] * hdx) * sxm[m] * global_parameters.D_t / global_parameters.D_x;
									Flow->velocity[{X, Y, Z, 1}] += cos(arg - ky[m] * hdy) * sym[m] * global_parameters.D_t / global_parameters.D_x;
									Flow->velocity[{X, Y, Z, 2}] += cos(arg - kz[m] * hdz) * szm[m] * global_parameters.D_t / global_parameters.D_x;
								}
							}
						}
					}
				}
				free(phi);
				free(nu);
				free(theta);
				free(psi);
				free(km);
				free(kx);
				free(ky);
				free(kz);
				free(ktx);
				free(kty);
				free(ktz);
				free(sxm);
				free(sym);
				free(szm);
			}
		}
		/// ***************************************************** ///
		/// INITIAL CONDITIONS: 3-D PIPE (in x-direction)         ///
		/// ***************************************************** ///
		void PIPE3D(std::string filename, Flow_solver * Flow, Parallel_MPI * MPI_parallel, stl_import * Geo) {
			unsigned int X, Y, Z;
			double yy, zz;
			double radius, center_y, center_z, ustar, ystar;
			double deltaplus, rprime;
			if (MPI_parallel->processor_id != MASTER) {
				/// Open input file
				string input_filename(filename);
				input_filename += ".dat";
				ifstream input_file;  // File is open for READING
				input_file.open(input_filename.c_str(), ios::binary);
				string Line1;
				char comment_indicator = 'k';
				int beg_line;
				input_file.clear();
				input_file.seekg(0, ios::beg);
				do {
					std::getline(input_file, Line1);
				} while (Line1.find("c\tPipe Initialization") == string::npos);
				std::getline(input_file, Line1);
				do {
					beg_line = input_file.tellg();
					input_file >> comment_indicator;
					if (comment_indicator == '#') {
						std::getline(input_file, Line1);
					}
				} while (comment_indicator == '#');
				input_file.seekg(beg_line);
				/// READ PARAMETERS (in S.I. units)
				input_file >> center_y >> center_z >> radius >> ystar >> ustar;
				/// PRINT OUT PARAMETERS (in LB units)
				if (MPI_parallel->processor_id == (MASTER + 1)) {
					std::cout << "Turbulent Pipe Initial conditions \n";
					std::cout << "Ustar : " << ustar * global_parameters.D_t / global_parameters.D_x << std::endl;
					std::cout << "Ystar : " << ystar / global_parameters.D_x << std::endl;
					std::cout << "Radius : " << radius / global_parameters.D_x << std::endl;
					std::cout << "Center(y,z) : " << center_y / global_parameters.D_x << "\t" << center_z / global_parameters.D_x << std::endl;
				}
				for (X = 0; X < MPI_parallel->dev_end[0]; X++) {
					// const double xx = fmod(X - MPI_parallel->start_XYZ2[0] + MPI_parallel->start_XYZ[0] + global_parameters.Nx, global_parameters.Nx) * global_parameters.D_x + Geo->x_center;
					for (Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
						yy = fmod(Y - MPI_parallel->start_XYZ2[1] + MPI_parallel->start_XYZ[1] + global_parameters.Ny, global_parameters.Ny) * global_parameters.D_x + Geo->y_center;
						for (Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
							zz = fmod(Z - MPI_parallel->start_XYZ2[2] + MPI_parallel->start_XYZ[2] + global_parameters.Nz, global_parameters.Nz) * global_parameters.D_x + Geo->z_center;
							if (Flow->is_solid[{X, Y, Z}] == FALSE) {
								rprime = sqrt(sqr(yy - center_y) + sqr(zz - center_z));
								deltaplus = (radius - rprime) / ystar;
								if (deltaplus <= 10.8) {
									Flow->velocity[{X, Y, Z, 0}] = deltaplus * ustar * global_parameters.D_t / global_parameters.D_x;
									Flow->velocity[{X, Y, Z, 1}] = 0;
									Flow->velocity[{X, Y, Z, 2}] = 0;
								}
								if (deltaplus > 10.8) {
									Flow->velocity[{X, Y, Z, 0}] = ((1. / 0.4) * log(deltaplus) + 5.) * (ustar * global_parameters.D_t / global_parameters.D_x);
									Flow->velocity[{X, Y, Z, 1}] = 0;
									Flow->velocity[{X, Y, Z, 2}] = 0;
								}
							}
						}
					}
				}
			}
			return;
		}
		/// ***************************************************** ///
		/// INITIAL CONDITIONS: 3-D PIPE (in x-direction)         ///
		/// ***************************************************** ///
		void PIPE3DFORCE(Flow_solver * Flow, Parallel_MPI * MPI_parallel, stl_import * Geo) {
			unsigned int X, Y, Z;
			double xx, yy, zz;

			double rdistance, zdistance, theta;
			double Fr, Ftheta, Fz;
			/// THESE PARAMETERS (OF THE PIPE) ARE TO BE DEFINED DIRECTLY INSIDE THE FUNCTION BY THE USER IN S.I. UNITS
			double radius, center_y, center_z, length;
			radius = 0.1;
			center_y = 1.;
			center_z = 1.;
			length = 1.2;
			/// THESE PARAMETERS (OF THE NON-UNIFORM FORCE) ARE TO BE DEFINED DIRECTLY INSIDE THE FUNCTION BY THE USER IN S.I. UNITS
			double kappa, B0, kz, ktheta, l, l0, Period;
			kappa = 0.5;
			B0 = 50.;
			kz = 3.;
			ktheta = 2.;
			Period = 2000.;
			l0 = 0.2 * radius;
			l = 0.4 * radius;

			if (MPI_parallel->processor_id != MASTER) {
				for (X = 0; X < MPI_parallel->dev_end[0]; X++) {
					xx = fmod(X - MPI_parallel->start_XYZ2[0] + MPI_parallel->start_XYZ[0] + global_parameters.Nx, global_parameters.Nx) * global_parameters.D_x + Geo->x_center;
					for (Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
						yy = fmod(Y - MPI_parallel->start_XYZ2[1] + MPI_parallel->start_XYZ[1] + global_parameters.Ny, global_parameters.Ny) * global_parameters.D_x + Geo->y_center;
						for (Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
							zz = fmod(Z - MPI_parallel->start_XYZ2[2] + MPI_parallel->start_XYZ[2] + global_parameters.Nz, global_parameters.Nz) * global_parameters.D_x + Geo->z_center;
							if (Flow->is_solid[{X, Y, Z}] == FALSE) {
								rdistance = sqrt(sqr(yy - center_y) + sqr(zz - center_z));
								zdistance = xx - Geo->x_center;
								double ydistance, xdistance;
								ydistance = yy - center_y;
								xdistance = zz - center_z;
								/// GET THETA ANGLE
								if (ydistance >= 0) theta = acos(xdistance / rdistance);
								if (ydistance < 0) theta = -acos(xdistance / rdistance);

								if ((radius - rdistance) >= l0 && (radius - rdistance) <= (l0 + l)) {
									double g = Flow->gravity[0];
									Fr = -g * kappa * B0 * (kz * l / length) * (radius / rdistance) * sin(2. * M_PI * Flow->physical_time / Period) * (1. - cos(2. * M_PI * (radius - rdistance - l0) / l)) * cos(kz * 2. * M_PI * zdistance / length) * cos(ktheta * theta);
									Ftheta = g * (1. - kappa) * B0 * (kz / ktheta) * (2. * M_PI * radius / length) * sin(2. * M_PI * Flow->physical_time / Period) * sin(2. * M_PI * (radius - rdistance - l0) / l) * cos(kz * 2. * M_PI * zdistance / length) * sin(ktheta * theta);
									Fz = -g * B0 * (radius / rdistance) * sin(2. * M_PI * Flow->physical_time / Period) * sin(2. * M_PI * (radius - rdistance - l0) / l) * sin(kz * 2. * M_PI * zdistance / length) * cos(ktheta * theta);

									Flow->force[{X, Y, Z, 0}] += Fz;
									Flow->force[{X, Y, Z, 1}] += Fr * sin(theta) + Ftheta * sin(theta + M_PI / 2.);
									Flow->force[{X, Y, Z, 2}] += Fr * cos(theta) + Ftheta * cos(theta + M_PI / 2.);
								}
							}
						}
					}
				}
			}
			return;
		}
		/// ***************************************************** ///
		/// APPLY TIME-DEPENDENT BODY FORCE TO FLOW               ///
		/// ***************************************************** ///
		void time_dependent_force(Flow_solver * Flow, Parallel_MPI * MPI_parallel) {
			double body_force;
			double Period = 1.;  /// [1/s]
			int direction = 0;   /// 0->x, 1->y, 2->z
			double G = 0.008;    /// [m/s2]

			double time = Flow->physical_time / global_parameters.D_t;
			double omega = 2. * M_PI * global_parameters.D_t / Period;
			G *= sqr(global_parameters.D_t) / global_parameters.D_x;

			unsigned int X, Y, Z;
			if (MPI_parallel->processor_id != MASTER) {
				for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
					for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
						for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
							if (Flow->is_solid[{X, Y, Z}] == FALSE) {
								body_force = G * cos(omega * time);
								Flow->force[{X, Y, Z, direction}] = Flow->density[{X, Y, Z}] * body_force;
							}
						}
					}
				}
			}
		}
		/// ***************************************************** ///
		/// TIME-DEPENDENT BC: READ DATA FROM FILE                ///
		/// ***************************************************** ///
		void Time_dependent_boundary::data_input(const std::string& filename, const Parallel_MPI& MPI_parallel) {
			//	int column_width = 40;
			/// Open input file
			ifstream input_file(filename + ".dat", ios::binary);

			find_line_after_header(input_file, "c\tFlow Field Time-Dependent Boundary Conditions");
			find_line_after_comment(input_file);
			/// First read how many time-dependent boundaries there are
			input_file >> number_of_BC;
			/// Resize arrays holding time-dependent boundary conditions data
			index_of_BC.resize(number_of_BC);
			data_filename_of_BC.resize(number_of_BC);
			velocity = Tensor<Vec3, 2>({static_cast<Index>(number_of_BC), t_num});
			t = Tensor<double, 2>({static_cast<Index>(number_of_BC), t_num});
			for (int i = 0; i < number_of_BC; i++) {
				find_line_after_comment(input_file);
				input_file >> index_of_BC[i] >> data_filename_of_BC[i];
			}

			input_file.close();
			double t_temp[2];
			Vec3 u_temp0, u_temp1;
			for (int i = 0; i < number_of_BC; i++) {
				input_file.open(data_filename_of_BC[i].c_str(), ios::binary);
				input_file >> number_of_data_points;
				input_file >> t_temp[0] >> u_temp0[0] >> u_temp0[1] >> u_temp0[2];
				t_temp[0] /= global_parameters.D_t;
				u_temp0 *= global_parameters.D_t / global_parameters.D_x;
				velocity[{i, 0}] = u_temp0;
				int k = 1;
				for (int j = 1; j < number_of_data_points; j++) {
					input_file >> t_temp[1] >> u_temp1[0] >> u_temp1[1] >> u_temp1[2];
					t_temp[1] /= global_parameters.D_t;
					u_temp1 *= global_parameters.D_t / global_parameters.D_x;
					do {
						velocity[{i, k}] = u_temp0 + (k - t_temp[0]) * (u_temp1 - u_temp0) / (t_temp[1] - t_temp[0]);
						k++;
					} while (k > t_temp[0] && k <= t_temp[1] && k < t_num);
					k--;
					t_temp[0] = t_temp[1];
					u_temp0 = u_temp1;
				}
				input_file.close();
			}
			if (MPI_parallel.processor_id == MASTER + 1) {
				std::cout << "Time-Dependent Flow Boundary Condition" << endl;
				for (int i = 0; i < number_of_BC; i++) {
					std::cout << "BC index :" << index_of_BC[i] << "\t File name : " << data_filename_of_BC[i] << endl;
				}
			}
		}
		/// ***************************************************** ///
		/// UPDATE VELOCITIES AT BOUNDARIES (USE RIGHT BEFORE     ///
		/// BC FUNCTION                                           ///
		/// ***************************************************** ///
		void Time_dependent_boundary::set_values(unsigned int tm, Flow_solver& Flow, const Parallel_MPI& MPI_parallel) const {
			if (MPI_parallel.is_master()) {
				return;
			}

			for (size_t i = 0; i < index_of_BC.size(); ++i) {
				const unsigned bc_idx = index_of_BC[i];
				auto it = std::find_if(Flow.boundaries.begin(), Flow.boundaries.end(), [=](const Flow_boundary_data& boundary) {
					return boundary.index == bc_idx;
				});
				it->v = velocity[{static_cast<Index>(i), static_cast<Index>(tm)}];  //{ux[i][tm], uy[i][tm], uz[i][tm]};
			}
		}
