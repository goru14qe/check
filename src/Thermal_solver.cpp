// #include "stdafx.h"
#include <sstream>   // string streams
#include <iostream>  // for the use of 'cout'
#include <cmath>
#include <string.h>
#include <fstream>  // file stream
#include <filesystem>
#include "Fluid_read_write.h"

#include "Thermal_solver.h"
#include "ALBORZ_Macros.h"
#include "ALBORZ_SETTINGS.h"
#include "ALBORZ_GlobalVariables.h"
#include "Flow_solver.h"
#include "Species_solver.h"
#include "Geometry.h"
#include "io/IO_interface.h"
#include "REGATH_INTERFACE.h"
#include "CANTERA_INTERFACE.h"
using namespace std;

/* These headers contain cantera functions */
#define CT_SKIP_DEPRECATION_WARNINGS
#include "cantera/thermo.h"
#include "cantera/transport.h"
#include "cantera/kinetics/GasKinetics.h"
#include "cantera/thermo/Phase.h"
#include "cantera/base/Solution.h"
#include "utils/Config_utils.h"

using namespace Cantera;


std::vector<double> Ini_T;
std::vector<double> Ini_cp;
std::vector<double> Ini_lambda;

Thermal_solver::Thermal_solver() {
}
void NewtonRaphson(double& T, double H, double cp) {
	double x1, x2;
	x1 = T;
	x2 = T;
	do {
		x1 = x2;
		x2 = x1 - (H - EnthalpyTemp(x1, cp)) / (double)(-HeatCapacity(x1, cp));
	} while (fabs(H - EnthalpyTemp(x2, cp)) > 1e-5);
	T = x2;
}
double EnthalpyTemp(double T, double cp_in) {
	//	double T0 = 300.;
	const double H = cp_in * T;  // 0.5 * (cp_in/T0) * sqr(T) + 0.5 * cp_in*T0;
	return H;
}
double HeatCapacity(double T, double cp_in) {
	//	double T0 = 300.;
	const double cp = cp_in;  // * T/T0;
	return cp;
}
/// ***************************************************** ///
/// READ IN GENERAL GENERAL PARAMETERS                    ///
/// ***************************************************** ///
void Thermal_solver::General_data_input(std::string filename, Parallel_MPI* MPI_parallel) {
	int column_width = 40;
	/// Open input file
	string input_filename(filename);
	input_filename += ".dat";
	ifstream input_file;  // File is open for READING
	input_file.open(input_filename.c_str(), ios::binary);

	input_file.clear();
	input_file.seekg(0, ios::beg);
	find_line_after_header(input_file, "c\tTemperature Field Solver");
	find_line_after_comment(input_file);
	input_file >> Dimension >> Discrete_Velocity >> E_0 >> T_0;
	input_file >> gbeta;
	gbeta = gbeta * (global_parameters.D_t * global_parameters.D_t * E_0 / global_parameters.D_x);
	input_file >> T_infinity;
	T_infinity /= T_0;
	input_file >> VAR_switch >> Gamma;
	input_file.close();
	if (MPI_parallel->processor_id == MASTER + 1) {
		std::cout << "Energy field parameters" << endl;
		std::cout << "=====================" << endl;
		std::cout << "Stencil : " << left << "D" << Dimension << "Q" << Discrete_Velocity << endl;
		std::cout << setw(column_width) << left << "E_0 = " << E_0 << endl
				  << setw(column_width) << left << "T_0 = " << T_0 << endl;
		std::cout << setw(column_width) << left << "gbeta = " << gbeta << endl
				  << setw(column_width) << left << "T_inf = " << T_infinity << endl;
		std::cout << setw(column_width) << left << "E?T? = " << VAR_switch << endl
				  << setw(column_width) << left << "Gamma = " << Gamma << endl;
	}
}
/// ***************************************************** ///
/// INITIALIZE STENCIL, CREATE OUTPUT FOLDER AND ALLOCATE ///
/// MEMORY FOR POPULATIONS AND PARAMETERS                 ///
/// ***************************************************** ///
void Thermal_solver::Memory_allocation(Stencil_Definition Stencil_Def, Parallel_MPI* MPI_parallel) {
	///  Stencil parameters initialization
	Stencil_Def(Dimension, Discrete_Velocity, weight, c_alpha, alpha_bar, c_s2);

	weight_2.resize(Discrete_Velocity);
	weight_2[0] = 0;
	for (int alpha = 1; alpha < Discrete_Velocity; alpha++) {
		weight_2[alpha] = weight[alpha];
		weight_2[0] = weight_2[0] - weight_2[alpha];
	}

	// for collective IO additional dimensions need to be correct even for empty tensors
	const Index x_size = MPI_parallel->is_master() ? 0 : static_cast<Index>(MPI_parallel->dev_end[0]);
	const Index y_size = MPI_parallel->is_master() ? 0 : static_cast<Index>(MPI_parallel->dev_end[1]);
	const Index z_size = MPI_parallel->is_master() ? 0 : static_cast<Index>(MPI_parallel->dev_end[2]);

	const Scalar_field::Index_vec scalar_sizes{x_size,
	                                           y_size,
	                                           z_size};

	const Vector_field::Index_vec pop_sizes{x_size,
	                                        y_size,
	                                        z_size,
	                                        static_cast<Index>(Discrete_Velocity)};
	initial_CP = Scalar_field::zeros(scalar_sizes);
	solid_particle = Scalar_field::zeros(scalar_sizes);
	temperature = Scalar_field::zeros(scalar_sizes);
	energy = Scalar_field::zeros(scalar_sizes);
	energy_previous = Scalar_field::zeros(scalar_sizes);
	c_p = Scalar_field::zeros(scalar_sizes);
	force_thermal = Scalar_field::zeros(scalar_sizes);
	temp_force_thermal = Scalar_field::zeros(scalar_sizes);
	previous_temperature = Scalar_field::zeros(scalar_sizes);
	solid_thermal_type = Solid_field::zeros(scalar_sizes);
	thermal_diffusion_coefficient = Scalar_field::zeros(scalar_sizes);
	Production = Scalar_field::zeros(scalar_sizes);

	pop_t = Vector_field::zeros(pop_sizes);      // for temperature
	pop_old_t = Vector_field::zeros(pop_sizes);  // for temperature

	if (MPI_parallel->processor_id != MASTER) {
		pop_eq = new double[Discrete_Velocity];
	}

	if (!MPI_parallel->is_master()) {
		pop_group = Data_exchange_group(*MPI_parallel);
		pop_group.add_population(pop_t, c_alpha);
		macroscopic_group = Data_exchange_group(*MPI_parallel);
		macroscopic_group.add_field(energy);
		macroscopic_group.add_field(temperature);
	}
}
/// ***************************************************** ///
/// INITIALIZE STENCIL, CREATE OUTPUT FOLDER AND ALLOCATE ///
/// MEMORY FOR FD SOLVER                                  ///
/// ***************************************************** ///
void Thermal_solver::Memory_allocation_FD(Stencil_Definition Stencil_Def, Parallel_MPI* MPI_parallel) {
	///  Stencil parameters initialization
	Stencil_Def(Dimension, Discrete_Velocity, weight, c_alpha, alpha_bar, c_s2);

	weight_2.resize(Discrete_Velocity);
	weight_2[0] = 0;
	for (int alpha = 1; alpha < Discrete_Velocity; alpha++) {
		weight_2[alpha] = weight[alpha];
		weight_2[0] = weight_2[0] - weight_2[alpha];
	}

	// for collective IO additional dimensions need to be correct even for empty tensors
	const Index x_size = MPI_parallel->is_master() ? 0 : static_cast<Index>(MPI_parallel->dev_end[0]);
	const Index y_size = MPI_parallel->is_master() ? 0 : static_cast<Index>(MPI_parallel->dev_end[1]);
	const Index z_size = MPI_parallel->is_master() ? 0 : static_cast<Index>(MPI_parallel->dev_end[2]);

	const Scalar_field::Index_vec scalar_sizes{x_size,
	                                           y_size,
	                                           z_size};
	initial_CP = Scalar_field::zeros(scalar_sizes);
	solid_particle = Scalar_field::zeros(scalar_sizes);
	temperature = Scalar_field::zeros(scalar_sizes);
	previous_temperature = temperature;
	temp_temperature = Scalar_field::zeros(scalar_sizes);
	energy = Scalar_field::zeros(scalar_sizes);
	c_p = Scalar_field::zeros(scalar_sizes);
	solid_thermal_type = Solid_field::zeros(scalar_sizes);
	force_thermal = Scalar_field::zeros(scalar_sizes);
	temp_force_thermal = Scalar_field::zeros(scalar_sizes);
	thermal_diffusion_coefficient = Scalar_field::zeros(scalar_sizes);
	Production = Scalar_field::zeros(scalar_sizes);

	if (!MPI_parallel->is_master()) {
		macroscopic_group = Data_exchange_group(*MPI_parallel);
		macroscopic_group.add_field(energy);
		macroscopic_group.add_field(temperature);
	}
}
/// ***************************************************** ///
/// INITIALIZE FIELD VARIABLES, TEMPERATURE, ENERGY AND   ///
/// GEOMETRY                                              ///
/// ***************************************************** ///
void Thermal_solver::initialize_field(Geometry* Geo, stl_import* Geo_stl, Temperature_Ini Ini_Temp, Flow_solver* Flow, Parallel_MPI* MPI_parallel, std::string filename) {
    int X, Y, Z;
    if (MPI_parallel->processor_id != MASTER) {
        if (Geo->flag == TRUE) {
            for (X = 0; X < global_parameters.Nx; X++) {
                for (Y = 0; Y < global_parameters.Ny; Y++) {
                    for (Z = 0; Z < global_parameters.Nz; Z++) {
                        solid_thermal_type[{X, Y, Z}] = Geo->img[X][Y];
                    }
                }
            }
        }
        if (Geo_stl->flag == 1) {
            for (X = 0; X < MPI_parallel->dev_end[0]; X++) {
                for (Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
                    for (Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
                        solid_thermal_type[{X, Y, Z}] = Geo_stl->domain[{X, Y, Z}];
                    }
                }
            }
        }
		Ini_Temp(temperature, c_p, thermal_diffusion_coefficient, solid_thermal_type, global_parameters.Nx, global_parameters.Ny, global_parameters.Nz, c_s2 / Gamma, T_0, E_0, Flow->rho_0, filename, Geo_stl->Source_count, MPI_parallel);
        for (X = 0; X < MPI_parallel->dev_end[0]; ++X) {
            for (Y = 0; Y < MPI_parallel->dev_end[1]; ++Y) {
                for (Z = 0; Z < MPI_parallel->dev_end[2]; ++Z) {
                    previous_temperature[{X, Y, Z}] = temperature[{X, Y, Z}];
                    force_thermal[{X, Y, Z}] = 0;
                    temp_force_thermal[{X, Y, Z}] = 0;
                    Production[{X, Y, Z}] = 0;
                    solid_particle[{X, Y, Z}] = 0.0;
                    initial_CP[{X, Y, Z}] = c_p[{X, Y, Z}];
                }
            }
        }
    }
}
/// ***************************************************** ///
/// INITIALIZE FIELD VARIABLES, TEMPERATURE, ENERGY AND   ///
/// GEOMETRY FOR FD SOLVER                                ///
/// ***************************************************** ///
void Thermal_solver::initialize_field_FD(Geometry* Geo, stl_import* Geo_stl, Temperature_Ini Ini_Temp, Flow_solver* Flow, Parallel_MPI* MPI_parallel, std::string filename) {
	int X, Y, Z;
	if (MPI_parallel->processor_id != MASTER) {
		if (Geo_stl->flag == 1) {
			for (X = 0; X < MPI_parallel->dev_end[0]; X++) {
				for (Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
					for (Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
						solid_thermal_type[{X, Y, Z}] = Geo_stl->domain[{X, Y, Z}];
					}
				}
			}
		}
		Ini_Temp(temperature, c_p, thermal_diffusion_coefficient, solid_thermal_type, global_parameters.Nx, global_parameters.Ny, global_parameters.Nz, E_0 / T_0, T_0, E_0, Flow->rho_0, filename, Geo_stl->Source_count, MPI_parallel);
		// int xx, yy, zz;
		for (X = 0; X < MPI_parallel->dev_end[0]; X++) {
			// xx = fmod(X - MPI_parallel->start_XYZ2[0] + MPI_parallel->start_XYZ[0] + global_parameters.Nx + 1, global_parameters.Nx);
			for (Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
				// yy = fmod(Y - MPI_parallel->start_XYZ2[1] + MPI_parallel->start_XYZ[1] + global_parameters.Ny, global_parameters.Ny);
				for (Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
					// zz = fmod(Z - MPI_parallel->start_XYZ2[2] + MPI_parallel->start_XYZ[2] + global_parameters.Nz, global_parameters.Nz);
					// const double R = sqr(zz - 25) + sqr(yy - 25) + sqr(xx - 25);
					energy[{X, Y, Z}] = 0;
					previous_temperature[{X, Y, Z}] = temperature[{X, Y, Z}];
					Production[{X, Y, Z}] = 0;
					/// diffusion coefficient = initial lambda * (dt / dx^2) * (E_0 / T_0) * rho_0
					/// where lambda is the Thermal conductivity
					thermal_diffusion_coefficient[{X, Y, Z}] = thermal_diffusion_coefficient[{X, Y, Z}] * (E_0 * Flow->rho_0 / T_0);
					solid_particle[{X, Y, Z}] = 0.0;
					initial_CP[{X, Y, Z}] = c_p[{X, Y, Z}];
					force_thermal[{X, Y, Z}] = 0;
					temp_force_thermal[{X, Y, Z}] = 0;
				}
			}
		}
	}
	return;
}
/// ***************************************************** ///
/// INITIALIZE POPULATIONS USING ONLY EQUILIBRIUM PART    ///
/// ***************************************************** ///
void Thermal_solver::initialize_pop_eq(Flow_solver* Flow, Parallel_MPI* MPI_parallel, std::string filename) {
	int X, Y, Z, alpha;
	double p;
	double enthalpy;
	if (MPI_parallel->processor_id != MASTER) {
		// Use the equilibrium populations corresponding to the initialized fluid density and velocity.
		for (X = 0; X < MPI_parallel->dev_end[0]; ++X) {
			for (Y = 0; Y < MPI_parallel->dev_end[1]; ++Y) {
				for (Z = 0; Z < MPI_parallel->dev_end[2]; ++Z) {
					energy[{X, Y, Z}] = Flow->rho_0 * Flow->density[{X, Y, Z}] * (EnthalpyTemp(temperature[{X, Y, Z}] * T_0, c_p[{X, Y, Z}]))
					                    + Flow->rho_0 * Flow->density[{X, Y, Z}] * (+0.5 * sqr(global_parameters.D_x / global_parameters.D_t) * (sqr(Flow->velocity[{X, Y, Z, 0}]) + sqr(Flow->velocity[{X, Y, Z, 1}]) + sqr(Flow->velocity[{X, Y, Z, 2}])));
					energy[{X, Y, Z}] /= E_0;
					p = Flow->pressure[{X, Y, Z}] * (Flow->rho_0 * sqr(global_parameters.D_x / global_parameters.D_t)) / E_0;
					enthalpy = energy[{X, Y, Z}] + p;
					equilibrium(energy[{X, Y, Z}], enthalpy, temperature[{X, Y, Z}], &Flow->velocity[{X, Y, Z, 0}], pop_eq);
					for (alpha = 0; alpha < Discrete_Velocity; ++alpha) {
						pop_old_t[{X, Y, Z, alpha}] = pop_eq[alpha];
						pop_t[{X, Y, Z, alpha}] = pop_eq[alpha];
					}
				}
			}
		}
	}
}
void Thermal_solver::initialize_pop_eq_crystal(Flow_solver* Flow, Parallel_MPI* MPI_parallel, const std::string& filename) {
	int X, Y, Z;
	if (MPI_parallel->processor_id != MASTER) {
		for (X = 0; X < MPI_parallel->dev_end[0]; ++X) {
			for (Y = 0; Y < MPI_parallel->dev_end[1]; ++Y) {
				for (Z = 0; Z < MPI_parallel->dev_end[2]; ++Z) {
					equilibrium_crystal(temperature[{X, Y, Z}], &Flow->velocity[{X, Y, Z, 0}], pop_eq);
					for (int alpha = 0; alpha < Discrete_Velocity; ++alpha) {
						pop_old_t[{X, Y, Z, alpha}] = pop_eq[alpha];
						pop_t[{X, Y, Z, alpha}] = pop_eq[alpha];
					}
				}
			}
		}
	}
}
/// ***************************************************** ///
/// INITIALIZE BOUNDARY CONDITIONS                        ///
/// ***************************************************** ///
void Thermal_solver::initialize_BC(Geometry* Geo, stl_import* Geo_stl, Parallel_MPI* MPI_parallel, std::string filename) {
	if (MPI_parallel->processor_id != MASTER) {
		int X, Y, Z, i, alpha;
		int number_of_BC = 0;
		std::vector<thermal_boundary_data> temp;
		std::vector<int> index;
		int** intersection;
		/// Open input file
		string input_filename(filename);
		input_filename += ".dat";
		ifstream input_file;  // File is open for READING
		input_file.open(input_filename.c_str(), ios::binary);

		input_file.clear();
		input_file.seekg(0, ios::beg);
		find_line_after_header(input_file, "c\tTemperature Field Boundary Conditions");
		find_line_after_comment(input_file);
		input_file >> number_of_BC >> curved_boundaries;
		temp.resize(number_of_BC);
		index.resize(number_of_BC);
		intersection = new int*[number_of_BC];
		for (i = 0; i < number_of_BC; i++) {
			intersection[i] = new int[2];
			find_line_after_comment(input_file);
			input_file >> index[i] >> intersection[i][0] >> intersection[i][1] >> temp[i].type;
			switch (temp[i].type) {
				/// 1. Zero temperature BC: The temperature is set to zero. (Dirichlet boundary condition).
				case 1:  /// --> Zero temperature BC
				{
					temp[i].T = 0;
					break;
				}
				/// 2. Non-zero temperature BC: The temperature is set to a specific value. (Dirichlet boundary condition).
				case 2:  /// ---> Non-zero temperature BC
				{
					input_file >> temp[i].T;
					;
					temp[i].T = temp[i].T / (double)T_0;
					break;
				}
				/// 3. Zero-gradient (1st-order) BC: The temperature gradient is set to zero. (Neumann boundary condition).
				case 3:  /// --> Zero-gradient (1st-order) BC
				{
					temp[i].T = 0;
					break;
				}
				/// 4. Zero-flux BC: The heat flux is set to zero. (Neumann boundary condition).
				case 4:  /// --> No-flux
				{
					temp[i].T = 0;
					break;
				}
				/// 102. Non-zero temperature BC: The temperature is set to a specific value. (Dirichlet boundary condition).
				case 102:  /// ---> Non-zero temperature BC
				{
					input_file >> temp[i].T;
					temp[i].T = temp[i].T / (double)T_0;  // This operation scales the temperature value, normalizing it or converting it to a different unit.
					break;
				}
				/// 104. Zero-gradient (1st-order) BC: The temperature gradient is set to zero. (Neumann boundary condition).
				case 104:  /// --> Zero-gradient (1st-order) BC
				case 105:
				case 106:
				case 107: {
					temp[i].T = 0;
					break;
				}
				case 108: {
					temp[i].T = 0;
					break;
				}
				default:  /// --> Boundary not defined
				{
					std::cout << " Error : undefined Thermal boundary type \n";
					break;
				}
			}
		}
		if (MPI_parallel->processor_id == (MASTER + 1)) {
			stringstream output_filename;
			output_filename << "Alborz_Results/Data/Energy_Boundary_Conditions.dat";
			ofstream BC_output;
			BC_output.open(output_filename.str().c_str(), fstream::trunc);
			BC_output << "ENERGY: Boundary Conditions \n";
			if (curved_boundaries)
				BC_output << "Curved: "
						  << "\tON\t" << "\n";
			else
				BC_output << "Curved: "
						  << "\tOFF\n";
			BC_output << "T_0 : " << T_0 << "\n";
			BC_output << "================================ \n";
			for (i = 0; i < number_of_BC; i++) {
				BC_output << "BOUNDARY INDEX : " << i + 1 << "\t TYPE : " << temp[i].type << std::endl;
				BC_output << "TEMPERATURE : " << temp[i].T << std::endl;
				BC_output << "-------------------------------- \n";
			}
			if (number_of_BC == 0) BC_output << "NO THERMAL BOUNDARY CONDITIONS\n";
			BC_output.close();
		}
		input_file.close();
		///   ------------------->  Find and store Boundary nodes
		// This part of the code identifies lattice points that meet specific criteria and marks them as boundary nodes, storing relevant information about these boundaries in the Boundaries vector.
		bool BC_flag = 0;  // Boolean flag.
		int Xp, Yp, Zp;    // temporary variables to store neighboring lattice sites.
		double n_temp[3];  // array to store directional information.
		for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					for (i = 0; i < number_of_BC; i++) {
						if (Geo_stl->domain[{X, Y, Z}] == intersection[i][0] && solid_thermal_type[{X, Y, Z}] == -1) {
							n_temp[0] = 0;
							n_temp[1] = 0;
							n_temp[2] = 0;
							for (alpha = 1; alpha < Discrete_Velocity; alpha++) {
								Xp = X + c_alpha[alpha][0];
								Yp = Y + c_alpha[alpha][1];
								Zp = Z + c_alpha[alpha][2];

								if (Geo_stl->domain[{X + 1, Y, Z}] == intersection[i][1]) n_temp[0] = -1;
								if (Geo_stl->domain[{X - 1, Y, Z}] == intersection[i][1]) n_temp[0] = 1;
								if (Geo_stl->domain[{X, Y + 1, Z}] == intersection[i][1]) n_temp[1] = -1;
								if (Geo_stl->domain[{X, Y - 1, Z}] == intersection[i][1]) n_temp[1] = 1;
								if (Geo_stl->domain[{X, Y, Z + 1}] == intersection[i][1]) n_temp[2] = -1;
								if (Geo_stl->domain[{X, Y, Z - 1}] == intersection[i][1]) n_temp[2] = 1;

								if (Geo_stl->domain[{Xp, Yp, Zp}] == intersection[i][1]) {
									BC_flag = 1;
								}
							}
							if (BC_flag == 1) {
								temp[i].X = X;
								temp[i].Y = Y;
								temp[i].Z = Z;
								temp[i].directions.resize(Discrete_Velocity);
								if (n_temp[0] != 0) temp[i].n[0] = n_temp[0] / abs(n_temp[0]);
								if (n_temp[1] != 0) temp[i].n[1] = n_temp[1] / abs(n_temp[1]);
								if (n_temp[2] != 0) temp[i].n[2] = n_temp[2] / abs(n_temp[2]);

								if (n_temp[0] == 0) temp[i].n[0] = 0;
								if (n_temp[1] == 0) temp[i].n[1] = 0;
								if (n_temp[2] == 0) temp[i].n[2] = 0;  // Boundaries.n[] shows the direction fo this neighbour

								Boundaries.push_back(temp[i]);  // Boundaries define the fluid point that have a neighbour that is not fluid ( outside the domain)
								BC_flag = 0;
								/// ALLOCATE MEMORY TO SOLID NEIGHBOR LIST
								for (alpha = 0; alpha < Discrete_Velocity; alpha++) {
									Xp = X - c_alpha[alpha][0];
									Yp = Y - c_alpha[alpha][1];
									Zp = Z - c_alpha[alpha][2];
									thermal_diffusion_coefficient[{Xp, Yp, Zp}] = thermal_diffusion_coefficient[{X, Y, Z}];
									c_p[{Xp, Yp, Zp}] = c_p[{X, Y, Z}];
									Boundaries[Boundaries.size() - 1].directions[alpha] = -1;
									/// FOR NON-FD BOUNDARY CONDITIONS
									if (Geo_stl->domain[{Xp, Yp, Zp}] == intersection[i][1] && Boundaries[Boundaries.size() - 1].type < 100) { Boundaries[Boundaries.size() - 1].directions[alpha] = 1; }

									// directions for BC < 100 is -1 and BC > 1.
								}
							}
						}
					}
				}
			}
		}
		if (curved_boundaries) Initialize_curved_boundaries_FD(Geo_stl, MPI_parallel);
#if defined DEBUG_MODE
		stringstream DB_filename;
		DB_filename << "Alborz_Results/debug/Energy_Boundary_Conditions_DB_proc_" << MPI_parallel->processor_id << ".dat";
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
			for (alpha = 0; alpha < Discrete_Velocity; alpha++) {
				BCDEBUG << Boundaries[i].directions[alpha] << "\t";
			}

			BCDEBUG << "\n";
		}
		BCDEBUG.close();
#endif  // defined
	}
	if (MPI_parallel->processor_id != MASTER) {
		int undefined_n = 0;
		int oustide_pointing_normal = 0;
		for (int i = 0; i < Boundaries.size(); i++) {
			if (Boundaries[i].n[0] == 0 && Boundaries[i].n[1] == 0 && Boundaries[i].n[2] == 0) undefined_n++;
			int Xp = Boundaries[i].X + SGN(Boundaries[i].n[0]) * ceil(fabs(Boundaries[i].n[0]));
			int Yp = Boundaries[i].Y + SGN(Boundaries[i].n[1]) * ceil(fabs(Boundaries[i].n[1]));
			int Zp = Boundaries[i].Z + SGN(Boundaries[i].n[2]) * ceil(fabs(Boundaries[i].n[2]));
			if (solid_thermal_type[{Xp, Yp, Zp}] == 1) oustide_pointing_normal++;
		}
		if (undefined_n > 0) std::cout << "(WARNING ENERGY) : " << undefined_n << " BOUNDARY NODES WITH ZERO NORMAL, PROCESSOR " << MPI_parallel->processor_id << std::endl;
		if (oustide_pointing_normal > 0) std::cout << "(WARNING ENERGY) : " << oustide_pointing_normal << " BOUNDARY NODES WITH OUTWARD-POINTING NORMAL, PROCESSOR " << MPI_parallel->processor_id << std::endl;
	}
	MPI_parallel->Sync_Master();
	return;
}
/// ***************************************************** ///
/// GET DISTANCE FROM LAST FLUID NODE TO STL SURFACE      ///
/// ALONG STENCIL VECTORS FOR CURVED BOUNDARIES           ///
/// ***************************************************** ///
void Thermal_solver::Initialize_curved_boundaries_FD(stl_import* Geo_stl, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		std::vector<stl::triangle> triangles;
		std::vector<int> triangles_which_file;
		stl::point lower_limits, upper_limits;
		triangles.clear();
		for (int file_index = 0; file_index < Geo_stl->Source_count; file_index++) {
			/* read all triangles */
			stl::stl_data info = stl::parse_stl(Geo_stl->Geo_filename[file_index]);

			std::vector<int> temp_which_file(info.triangles.size(), file_index + 1);
			triangles.insert(triangles.end(), info.triangles.begin(), info.triangles.end());
			triangles_which_file.insert(triangles_which_file.end(), temp_which_file.begin(), temp_which_file.end());
		}
		/* go through boundary nodes and find normal distance */
		for (int i = 0; i < Boundaries.size(); i++) {
			stl::point O, intersection_coordinate, direction;
			/* get coordinates of boundary node in stl space */
			MPI_parallel->get_coordinates(Boundaries[i].X, Boundaries[i].Y, Boundaries[i].Z,
			                              Geo_stl->x_center, Geo_stl->y_center, Geo_stl->z_center,
			                              O.x, O.y, O.z);
			double minimum_distance = 1.0e3;
			/* Go through relevant stl files */
			for (int l = 0; l < triangles.size(); l++) {
				/* is the triangle related to the boundary node? */
				if (triangles_which_file[l] == Boundaries[i].V_out) {
					direction.x = triangles[l].normal.x;
					direction.y = triangles[l].normal.y;
					direction.z = triangles[l].normal.z;
					int intersection_indicator = Geo_stl->triangle_shortest_distance(triangles[l].v1, triangles[l].v2, triangles[l].v3, O, direction, &intersection_coordinate);
					double distance;
					if (intersection_indicator == 1) {
						distance = sqrt(pow(O.x - intersection_coordinate.x, 2) + pow(O.y - intersection_coordinate.y, 2) + pow(O.z - intersection_coordinate.z, 2));
						double normX = (intersection_coordinate.x - O.x) / global_parameters.D_x;
						double normY = (intersection_coordinate.y - O.y) / global_parameters.D_x;
						double normZ = (intersection_coordinate.z - O.z) / global_parameters.D_x;
						if (distance < minimum_distance && (distance / global_parameters.D_x) <= sqrt(Dimension)) {
							int Xp = Boundaries[i].X + SGN(normX) * ceil(fabs(normX));
							int Yp = Boundaries[i].Y + SGN(normY) * ceil(fabs(normY));
							int Zp = Boundaries[i].Z + SGN(normZ) * ceil(fabs(normZ));
							if (Geo_stl->domain[{Xp, Yp, Zp}] == Boundaries[i].V_in) {
								minimum_distance = distance;
								Boundaries[i].normal_distance = distance / global_parameters.D_x;
								Boundaries[i].n[0] = normX;
								Boundaries[i].n[1] = normY;
								Boundaries[i].n[2] = normZ;
							}
						}
					}
				}
			}
		}
		/* Initialize images point and neighbors */
		for (int i = 0; i < Boundaries.size(); i++) {
			/* put image coordinates in memory */
			if (Dimension < 3) Boundaries[i].n[2] = 0;
			Boundaries[i].get_image();
			for (int xx = floor(Boundaries[i].X_Image) - 1; xx <= ceil(Boundaries[i].X_Image) + 1; xx++) {
				for (int yy = floor(Boundaries[i].Y_Image) - 1; yy <= ceil(Boundaries[i].Y_Image) + 1; yy++) {
					int Z_Int = Boundaries[i].Z;
					if (Dimension > 2) {
						for (int zz = floor(Boundaries[i].Z_Image) - 1; zz <= ceil(Boundaries[i].Z_Image) + 1; zz++) {
							if (solid_thermal_type[{xx, yy, zz}] == FALSE) {
								Boundaries[i].X_Image_Int.push_back(xx);
								Boundaries[i].Y_Image_Int.push_back(yy);
								Boundaries[i].Z_Image_Int.push_back(zz);
							}
						}
					}
					if (Dimension < 3) {
						if (solid_thermal_type[{xx, yy, Z_Int}] == FALSE) {
							Boundaries[i].X_Image_Int.push_back(xx);
							Boundaries[i].Y_Image_Int.push_back(yy);
							Boundaries[i].Z_Image_Int.push_back(Z_Int);
						}
					}
				} /*End of scan in Y-dir*/
			} /*End of scan in X-dir*/
			Boundaries[i].get_image_int_weights();
			if (Boundaries[i].X_Image_Int.size() == 0) {
				Boundaries[i].X_Image_Int.push_back(Boundaries[i].X);
				Boundaries[i].Y_Image_Int.push_back(Boundaries[i].Y);
				Boundaries[i].Z_Image_Int.push_back(Boundaries[i].Z);
				Boundaries[i].W_Image_Int.push_back(1.);
			}
		} /*End of Image initialization*/
	}
}
/// ***************************************************** ///
/// GET EQUILIBRIUM POPULATIONS AND PUT THEM IN F_EQ      ///
/// ***************************************************** ///
void Thermal_solver::equilibrium(double& A, double& B, double& C, double* u, double* pop_eq) {
	double Hxx, Hyy, Hzz,
		Hx, Hy, Hz;

	for (int alpha = 0; alpha < Discrete_Velocity; alpha++) {
		Hx = c_alpha[alpha][0];
		Hy = c_alpha[alpha][1];
		Hz = c_alpha[alpha][2];
		Hxx = sqr(c_alpha[alpha][0]) - (1. / c_s2);
		Hyy = sqr(c_alpha[alpha][1]) - (1. / c_s2);
		Hzz = sqr(c_alpha[alpha][2]) - (1. / c_s2);
		pop_eq[alpha] = weight[alpha] * (A + c_s2 * B * (Hx * u[0] + Hy * u[1]) + 0.5 * c_s2 * (Hxx + Hyy) * (C - A));
		if (Dimension == 3) {
			pop_eq[alpha] += weight[alpha] * (c_s2 * B * Hz * u[2] + 0.5 * c_s2 * Hzz * (C - A));
		}
	}
}
void Thermal_solver::equilibrium_crystal(double& A, double* vel, double* pop_eq_t) {
	for (int alpha = 0; alpha < Discrete_Velocity; alpha++) {
		pop_eq_t[alpha] = weight[alpha] * A * (1.0 + c_s2 * DOT(c_alpha[alpha], vel));
	}
}
/// ***************************************************** ///
/// COLLISION AND STREAMING USING SRT MODEL               ///
/// ***************************************************** ///
void Thermal_solver::LBM_SRT(int time, Flow_solver* Flow, Parallel_MPI* MPI_parallel) {
	double enthalpy, enthalpy_previous, p, p_previous;
	int X, Y, Z, alpha;
	double omega_eff, tau_eff;
	double PHI;
	if (MPI_parallel->processor_id != MASTER) {
		swap(pop_old_t, pop_t);

		for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (solid_thermal_type[{X, Y, Z}] == -1) {
						tau_eff = (c_s2 * thermal_diffusion_coefficient[{X, Y, Z}] * (T_0 / E_0) + 0.5);
						omega_eff = 1. / tau_eff;
						p = Flow->pressure[{X, Y, Z}] * (Flow->rho_0 * sqr(global_parameters.D_x / global_parameters.D_t)) / E_0;
						p_previous = Flow->previous_pressure[{X, Y, Z}] * (Flow->rho_0 * sqr(global_parameters.D_x / global_parameters.D_t)) / E_0;
						enthalpy = energy[{X, Y, Z}] + p;
						enthalpy_previous = energy_previous[{X, Y, Z}] + p_previous;
						equilibrium(energy[{X, Y, Z}], enthalpy, temperature[{X, Y, Z}], &Flow->velocity[{X, Y, Z, 0}], pop_eq);
						for (alpha = 0; alpha < Discrete_Velocity; alpha++) {
							PHI = c_s2 * c_alpha[alpha][0] * (Flow->velocity[{X, Y, Z, 0}] * enthalpy - Flow->previous_velocity[{X, Y, Z, 0}] * enthalpy_previous)
							      + c_s2 * c_alpha[alpha][1] * (Flow->velocity[{X, Y, Z, 1}] * enthalpy - Flow->previous_velocity[{X, Y, Z, 1}] * enthalpy_previous)
							      + c_s2 * c_alpha[alpha][2] * (Flow->velocity[{X, Y, Z, 2}] * enthalpy - Flow->previous_velocity[{X, Y, Z, 2}] * enthalpy_previous);
							pop_t[{X + c_alpha[alpha][0], Y + c_alpha[alpha][1], Z + c_alpha[alpha][2], alpha}] = pop_old_t[{X, Y, Z, alpha}] * (1 - omega_eff)
							                                                                                      + pop_eq[alpha] * omega_eff + weight[alpha] * (1 - 0.5 * omega_eff) * (force_thermal[{X, Y, Z}] + Production[{X, Y, Z}] + PHI);
						}
					}
				}
			}
		}
	}
}
void Thermal_solver::LBM_SRT_crystal(int time, Flow_solver* Flow, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		swap(pop_old_t, pop_t);
		int X, Y, Z;
		double omega_eff;

		for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (solid_thermal_type[{X, Y, Z}] == -1) {
						omega_eff = 1. / (c_s2 * thermal_diffusion_coefficient[{X, Y, Z}] + 0.5);
						equilibrium_crystal(temperature[{X, Y, Z}], &Flow->velocity[{X, Y, Z, 0}], pop_eq);
						for (int alpha = 0; alpha < Discrete_Velocity; alpha++) {
							pop_t[{X + c_alpha[alpha][0], Y + c_alpha[alpha][1], Z + c_alpha[alpha][2], alpha}] = pop_old_t[{X, Y, Z, alpha}] * (1 - omega_eff)
							                                                                                      + pop_eq[alpha] * omega_eff + weight[alpha] * Production[{X, Y, Z}];
						}
					}
				}
			}
		}
	}
}
/// ***************************************************** ///
/// COLLISION AND STREAMING USING MRT MODEL               ///
/// ***************************************************** ///
void Thermal_solver::LBM_MRT(int time, Flow_solver* Flow, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		swap(pop_old_t, pop_t);
		int X, Y, Z;
		double tau_eff;

		vector<double> omega_eff;
		vector<double> Momeq;
		vector<double> Mom;
		double ux, uy, uz, T, p, E, Fx, Fy, E_previous, p_previous;

		if (Discrete_Velocity == 9) {
			omega_eff.resize(Discrete_Velocity);
			Mom.resize(Discrete_Velocity);
			Momeq.resize(Discrete_Velocity);
			for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
				for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
					for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
						if (solid_thermal_type[{X, Y, Z}] == -1) {
							tau_eff = c_s2 * thermal_diffusion_coefficient[{X, Y, Z}] * (T_0 / E_0) + 0.5;

							omega_eff[0] = 1.;  ///(c_s2 * thermal_diffusion_coefficient[{X,Y,Z}]*(T_0/E_0) + 0.5);
							omega_eff[1] = 1. / tau_eff;
							omega_eff[2] = 1. / tau_eff;
							omega_eff[3] = 1.;  ///(c_s2 * thermal_diffusion_coefficient[{X,Y,Z}]*(T_0/E_0) + 0.5);
							omega_eff[4] = 1.;  ///(c_s2 * thermal_diffusion_coefficient[{X,Y,Z}]*(T_0/E_0) + 0.5);
							omega_eff[5] = 1.;  ///(c_s2 * thermal_diffusion_coefficient[{X,Y,Z}]*(T_0/E_0) + 0.5);
							omega_eff[6] = 1.;  ///(c_s2 * thermal_diffusion_coefficient[{X,Y,Z}]*(T_0/E_0) + 0.5);
							omega_eff[7] = 1.;  ///(c_s2 * thermal_diffusion_coefficient[{X,Y,Z}]*(T_0/E_0) + 0.5);
							omega_eff[8] = 1.;  ///(c_s2 * thermal_diffusion_coefficient[{X,Y,Z}]*(T_0/E_0) + 0.5);

							// const double rho = Flow->density[{X,Y,Z}];
							ux = Flow->velocity[{X, Y, Z, 0}];
							uy = Flow->velocity[{X, Y, Z, 1}];
							T = temperature[{X, Y, Z}];
							E = energy[{X, Y, Z}];
							p = Flow->pressure[{X, Y, Z}] * (Flow->rho_0 * sqr(global_parameters.D_x / global_parameters.D_t)) / E_0;

							p_previous = Flow->previous_pressure[{X, Y, Z}] * (Flow->rho_0 * sqr(global_parameters.D_x / global_parameters.D_t)) / E_0;
							E_previous = energy_previous[{X, Y, Z}];

							Fx = (E + p) * Flow->velocity[{X, Y, Z, 0}] - (E_previous + p_previous) * Flow->previous_velocity[{X, Y, Z, 0}];
							Fy = (E + p) * Flow->velocity[{X, Y, Z, 1}] - (E_previous + p_previous) * Flow->previous_velocity[{X, Y, Z, 1}];
							/// *************************************************************************************************** ///
							///                                       EQUILIBRIUM CENTRAL MOMENTS                                   ///
							///                                             MOMENT SPACE :                                          ///
							///                                                  M_0,                                               ///
							///                                                M_x, M_y,                                            ///
							///                                       M_xx+M_yy, M_xx-M_yy, M_xy                                    ///
							///                                              M_xxy, M_xyy                                           ///
							///                                                  M_xxyy                                             ///
							/// *************************************************************************************************** ///
							Momeq[0] = E;
							Momeq[1] = (E + p) * ux;
							Momeq[2] = (E + p) * uy;
							Momeq[3] = 0;
							Momeq[4] = (T - E) / c_s2;
							Momeq[5] = (T - E) / c_s2;
							Momeq[6] = 0;
							Momeq[7] = 0;
							Momeq[8] = 0;

							Mom[0] = Momeq[0] * omega_eff[0] + (1. - omega_eff[0]) * (pop_old_t[{X, Y, Z, 0}] + pop_old_t[{X, Y, Z, 1}] + pop_old_t[{X, Y, Z, 2}] + pop_old_t[{X, Y, Z, 3}] + pop_old_t[{X, Y, Z, 4}] + pop_old_t[{X, Y, Z, 5}] + pop_old_t[{X, Y, Z, 6}] + pop_old_t[{X, Y, Z, 7}] + pop_old_t[{X, Y, Z, 8}]);
							Mom[1] = Momeq[1] * omega_eff[1] + (1. - omega_eff[1]) * (pop_old_t[{X, Y, Z, 1}] - pop_old_t[{X, Y, Z, 3}] + pop_old_t[{X, Y, Z, 5}] - pop_old_t[{X, Y, Z, 6}] - pop_old_t[{X, Y, Z, 7}] + pop_old_t[{X, Y, Z, 8}]);
							Mom[2] = Momeq[2] * omega_eff[2] + (1. - omega_eff[2]) * (pop_old_t[{X, Y, Z, 2}] - pop_old_t[{X, Y, Z, 4}] + pop_old_t[{X, Y, Z, 5}] + pop_old_t[{X, Y, Z, 6}] - pop_old_t[{X, Y, Z, 7}] - pop_old_t[{X, Y, Z, 8}]);
							Mom[3] = Momeq[3] * omega_eff[3] + (1. - omega_eff[3]) * (pop_old_t[{X, Y, Z, 5}] - pop_old_t[{X, Y, Z, 6}] + pop_old_t[{X, Y, Z, 7}] - pop_old_t[{X, Y, Z, 8}]);
							Mom[4] = Momeq[4] * omega_eff[4] + (1. - omega_eff[4]) * (-pop_old_t[{X, Y, Z, 0}] / (double)c_s2 - pop_old_t[{X, Y, Z, 2}] / (double)c_s2 - pop_old_t[{X, Y, Z, 4}] / (double)c_s2 - pop_old_t[{X, Y, Z, 1}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 3}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 5}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 6}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 7}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 8}] * (1 / (double)c_s2 - 1));
							Mom[5] = Momeq[5] * omega_eff[5] + (1. - omega_eff[5]) * (-pop_old_t[{X, Y, Z, 0}] / (double)c_s2 - pop_old_t[{X, Y, Z, 1}] / (double)c_s2 - pop_old_t[{X, Y, Z, 3}] / (double)c_s2 - pop_old_t[{X, Y, Z, 2}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 4}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 5}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 6}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 7}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 8}] * (1 / (double)c_s2 - 1));
							Mom[6] = Momeq[6] * omega_eff[6] + (1. - omega_eff[6]) * (pop_old_t[{X, Y, Z, 4}] / (double)c_s2 - pop_old_t[{X, Y, Z, 2}] / (double)c_s2 - pop_old_t[{X, Y, Z, 5}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 6}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 7}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 8}] * (1 / (double)c_s2 - 1));
							Mom[7] = Momeq[7] * omega_eff[7] + (1. - omega_eff[7]) * (pop_old_t[{X, Y, Z, 3}] / (double)c_s2 - pop_old_t[{X, Y, Z, 1}] / (double)c_s2 - pop_old_t[{X, Y, Z, 5}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 6}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 7}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 8}] * (1 / (double)c_s2 - 1));
							Mom[8] = Momeq[8] * omega_eff[8] + (1. - omega_eff[8]) * (pop_old_t[{X, Y, Z, 5}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 6}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 7}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 8}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 0}] / (double)sqr(c_s2) + (pop_old_t[{X, Y, Z, 1}] * (1 / (double)c_s2 - 1)) / (double)c_s2 + (pop_old_t[{X, Y, Z, 2}] * (1 / (double)c_s2 - 1)) / (double)c_s2 + (pop_old_t[{X, Y, Z, 3}] * (1 / (double)c_s2 - 1)) / (double)c_s2 + (pop_old_t[{X, Y, Z, 4}] * (1 / (double)c_s2 - 1)) / (double)c_s2);

							Mom[0] += (1. - 0.5 * omega_eff[0]) * Production[{X, Y, Z}];
							Mom[1] += (1. - 0.5 * omega_eff[1]) * Fx;
							Mom[2] += (1. - 0.5 * omega_eff[2]) * Fy;

							pop_t[{X + c_alpha[0][0], Y + c_alpha[0][1], Z + c_alpha[0][2], 0}] = Mom[8] - (Mom[4] * (c_s2 - 1)) / (double)c_s2 - (Mom[5] * (c_s2 - 1)) / (double)c_s2 + (Mom[0] * sqr(c_s2 - 1)) / (double)sqr(c_s2);
							pop_t[{X + c_alpha[1][0], Y + c_alpha[1][1], Z + c_alpha[1][2], 1}] = (Mom[0] * (c_s2 - 1)) / (double)(2 * sqr(c_s2)) - Mom[8] / (double)2 - Mom[5] / (double)(2 * c_s2) - Mom[7] / (double)2 + (Mom[1] * (c_s2 - 1)) / (double)(2 * c_s2) + (Mom[4] * (c_s2 - 1)) / (double)(2 * c_s2);
							pop_t[{X + c_alpha[2][0], Y + c_alpha[2][1], Z + c_alpha[2][2], 2}] = (Mom[0] * (c_s2 - 1)) / (double)(2 * sqr(c_s2)) - Mom[8] / (double)2 - Mom[4] / (double)(2 * c_s2) - Mom[6] / (double)2 + (Mom[2] * (c_s2 - 1)) / (double)(2 * c_s2) + (Mom[5] * (c_s2 - 1)) / (double)(2 * c_s2);
							pop_t[{X + c_alpha[3][0], Y + c_alpha[3][1], Z + c_alpha[3][2], 3}] = Mom[7] / (double)2 - Mom[8] / (double)2 - Mom[5] / (double)(2 * c_s2) + (Mom[0] * (c_s2 - 1)) / (double)(2 * sqr(c_s2)) - (Mom[1] * (c_s2 - 1)) / (double)(2 * c_s2) + (Mom[4] * (c_s2 - 1)) / (double)(2 * c_s2);
							pop_t[{X + c_alpha[4][0], Y + c_alpha[4][1], Z + c_alpha[4][2], 4}] = Mom[6] / (double)2 - Mom[8] / (double)2 - Mom[4] / (double)(2 * c_s2) + (Mom[0] * (c_s2 - 1)) / (double)(2 * sqr(c_s2)) - (Mom[2] * (c_s2 - 1)) / (double)(2 * c_s2) + (Mom[5] * (c_s2 - 1)) / (double)(2 * c_s2);
							pop_t[{X + c_alpha[5][0], Y + c_alpha[5][1], Z + c_alpha[5][2], 5}] = (Mom[0] + Mom[1] * c_s2 + Mom[2] * c_s2 + Mom[4] * c_s2 + Mom[5] * c_s2 + Mom[3] * sqr(c_s2) + Mom[6] * sqr(c_s2) + Mom[7] * sqr(c_s2) + Mom[8] * sqr(c_s2)) / (double)(4 * sqr(c_s2));
							pop_t[{X + c_alpha[6][0], Y + c_alpha[6][1], Z + c_alpha[6][2], 6}] = (Mom[0] - Mom[1] * c_s2 + Mom[2] * c_s2 + Mom[4] * c_s2 + Mom[5] * c_s2 - Mom[3] * sqr(c_s2) + Mom[6] * sqr(c_s2) - Mom[7] * sqr(c_s2) + Mom[8] * sqr(c_s2)) / (double)(4 * sqr(c_s2));
							pop_t[{X + c_alpha[7][0], Y + c_alpha[7][1], Z + c_alpha[7][2], 7}] = (Mom[0] - Mom[1] * c_s2 - Mom[2] * c_s2 + Mom[4] * c_s2 + Mom[5] * c_s2 + Mom[3] * sqr(c_s2) - Mom[6] * sqr(c_s2) - Mom[7] * sqr(c_s2) + Mom[8] * sqr(c_s2)) / (double)(4 * sqr(c_s2));
							pop_t[{X + c_alpha[8][0], Y + c_alpha[8][1], Z + c_alpha[8][2], 8}] = (Mom[0] + Mom[1] * c_s2 - Mom[2] * c_s2 + Mom[4] * c_s2 + Mom[5] * c_s2 - Mom[3] * sqr(c_s2) - Mom[6] * sqr(c_s2) + Mom[7] * sqr(c_s2) + Mom[8] * sqr(c_s2)) / (double)(4 * sqr(c_s2));
						}
					}
				}
			}
		}
		if (Discrete_Velocity == 7 && Dimension == 3) {
			omega_eff.resize(Discrete_Velocity);
			Mom.resize(Discrete_Velocity);
			Momeq.resize(Discrete_Velocity);
			for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
				for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
					for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
						if (solid_thermal_type[{X, Y, Z}] == -1) {
							tau_eff = c_s2 * thermal_diffusion_coefficient[{X, Y, Z}] * (T_0 / E_0) + 0.5;

							omega_eff[0] = 1.;  // /tau_eff;
							omega_eff[1] = 1. / tau_eff;
							omega_eff[2] = 1. / tau_eff;
							omega_eff[3] = 1. / tau_eff;
							omega_eff[4] = 1.;  // /tau_eff;
							omega_eff[5] = 1.;  // /tau_eff;
							omega_eff[6] = 1.;  // /tau_eff;

							// const double rho = Flow->density[{X,Y,Z}];
							ux = Flow->velocity[{X, Y, Z, 0}];
							uy = Flow->velocity[{X, Y, Z, 1}];
							uz = Flow->velocity[{X, Y, Z, 2}];
							T = temperature[{X, Y, Z}];
							E = energy[{X, Y, Z}];
							p = 0;  // Flow->pressure[{X,Y,Z}] * (Flow->rho_0 * sqr(global_parameters.D_x/global_parameters.D_t) )/E_0;

							///							Fx = (energy[{X,Y,Z}] + Flow->pressure[{X,Y,Z}])*Flow->velocity[{X,Y,Z,0}]
							///								- (energy_previous[{X,Y,Z}] + Flow->previous_pressure[{X,Y,Z}])*Flow->previous_velocity[{X,Y,Z,0}];
							///							Fx = (energy[{X,Y,Z}] + Flow->pressure[{X,Y,Z}])*Flow->velocity[{X,Y,Z,1}]
							///								- (energy_previous[{X,Y,Z}] + Flow->previous_pressure[{X,Y,Z}])*Flow->previous_velocity[{X,Y,Z,1}];
							/// *************************************************************************************************** ///
							///                                       EQUILIBRIUM CENTRAL MOMENTS                                   ///
							///                                             MOMENT SPACE :                                          ///
							///                                                  M_0,                                               ///
							///                                                M_x, M_y,                                            ///
							///                                       M_xx+M_yy, M_xx-M_yy, M_xy                                    ///
							///                                              M_xxy, M_xyy                                           ///
							///                                                  M_xxyy                                             ///
							/// *************************************************************************************************** ///
							Momeq[0] = E;
							Momeq[1] = (E + p) * ux;
							Momeq[2] = (E + p) * uy;
							Momeq[3] = (E + p) * uz;
							Momeq[4] = (T - E) / c_s2;
							Momeq[5] = (T - E) / c_s2;
							Momeq[6] = (T - E) / c_s2;

							Mom[0] = Momeq[0] * omega_eff[0] + (1. - omega_eff[0]) * (pop_old_t[{X, Y, Z, 0}] + pop_old_t[{X, Y, Z, 1}] + pop_old_t[{X, Y, Z, 2}] + pop_old_t[{X, Y, Z, 3}] + pop_old_t[{X, Y, Z, 4}] + pop_old_t[{X, Y, Z, 5}] + pop_old_t[{X, Y, Z, 6}]);
							Mom[1] = Momeq[1] * omega_eff[1] + (1. - omega_eff[1]) * (pop_old_t[{X, Y, Z, 1}] - pop_old_t[{X, Y, Z, 2}]);
							Mom[2] = Momeq[2] * omega_eff[2] + (1. - omega_eff[2]) * (pop_old_t[{X, Y, Z, 3}] - pop_old_t[{X, Y, Z, 4}]);
							Mom[3] = Momeq[3] * omega_eff[3] + (1. - omega_eff[3]) * (pop_old_t[{X, Y, Z, 5}] - pop_old_t[{X, Y, Z, 6}]);
							Mom[4] = Momeq[4] * omega_eff[4] + (1. - omega_eff[4]) * (-pop_old_t[{X, Y, Z, 0}] / (double)c_s2 - pop_old_t[{X, Y, Z, 3}] / (double)c_s2 - pop_old_t[{X, Y, Z, 4}] / (double)c_s2 - pop_old_t[{X, Y, Z, 5}] / (double)c_s2 - pop_old_t[{X, Y, Z, 6}] / (double)c_s2 - pop_old_t[{X, Y, Z, 1}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 2}] * (1 / (double)c_s2 - 1));
							Mom[5] = Momeq[5] * omega_eff[5] + (1. - omega_eff[5]) * (-pop_old_t[{X, Y, Z, 0}] / (double)c_s2 - pop_old_t[{X, Y, Z, 1}] / (double)c_s2 - pop_old_t[{X, Y, Z, 2}] / (double)c_s2 - pop_old_t[{X, Y, Z, 5}] / (double)c_s2 - pop_old_t[{X, Y, Z, 6}] / (double)c_s2 - pop_old_t[{X, Y, Z, 3}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 4}] * (1 / (double)c_s2 - 1));
							Mom[6] = Momeq[6] * omega_eff[6] + (1. - omega_eff[6]) * (-pop_old_t[{X, Y, Z, 0}] / (double)c_s2 - pop_old_t[{X, Y, Z, 1}] / (double)c_s2 - pop_old_t[{X, Y, Z, 2}] / (double)c_s2 - pop_old_t[{X, Y, Z, 3}] / (double)c_s2 - pop_old_t[{X, Y, Z, 4}] / (double)c_s2 - pop_old_t[{X, Y, Z, 5}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 6}] * (1 / (double)c_s2 - 1));
							// Mom[1] += (1. - 0.5*omega_eff[1]) * Fx;
							// Mom[2] += (1. - 0.5*omega_eff[2]) * Fy;

							pop_t[{X + c_alpha[0][0], Y + c_alpha[0][1], Z + c_alpha[0][2], 0}] = (Mom[0] * (c_s2 - 3)) / (double)c_s2 - Mom[5] - Mom[6] - Mom[4];
							pop_t[{X + c_alpha[1][0], Y + c_alpha[1][1], Z + c_alpha[1][2], 1}] = Mom[1] / (double)2 + Mom[4] / (double)2 + Mom[0] / (double)(2 * c_s2);
							pop_t[{X + c_alpha[2][0], Y + c_alpha[2][1], Z + c_alpha[2][2], 2}] = Mom[4] / (double)2 - Mom[1] / (double)2 + Mom[0] / (double)(2 * c_s2);
							pop_t[{X + c_alpha[3][0], Y + c_alpha[3][1], Z + c_alpha[3][2], 3}] = Mom[2] / (double)2 + Mom[5] / (double)2 + Mom[0] / (double)(2 * c_s2);
							pop_t[{X + c_alpha[4][0], Y + c_alpha[4][1], Z + c_alpha[4][2], 4}] = Mom[5] / (double)2 - Mom[2] / (double)2 + Mom[0] / (double)(2 * c_s2);
							pop_t[{X + c_alpha[5][0], Y + c_alpha[5][1], Z + c_alpha[5][2], 5}] = Mom[3] / (double)2 + Mom[6] / (double)2 + Mom[0] / (double)(2 * c_s2);
							pop_t[{X + c_alpha[6][0], Y + c_alpha[6][1], Z + c_alpha[6][2], 6}] = Mom[6] / (double)2 - Mom[3] / (double)2 + Mom[0] / (double)(2 * c_s2);
						}
					}
				}
			}
		}
		if (Discrete_Velocity == 27 && Dimension == 3) {
			omega_eff.resize(Discrete_Velocity);
			Mom.resize(Discrete_Velocity);
			Momeq.resize(Discrete_Velocity);
			for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
				for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
					for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
						if (solid_thermal_type[{X, Y, Z}] == -1) {
							tau_eff = c_s2 * thermal_diffusion_coefficient[{X, Y, Z}] * (T_0 / E_0) + 0.5;

							omega_eff[0] = 1.;  ///(c_s2 * thermal_diffusion_coefficient[{X,Y,Z}]*(T_0/E_0) + 0.5);
							omega_eff[1] = 1. / tau_eff;
							omega_eff[2] = 1. / tau_eff;
							omega_eff[3] = 1. / tau_eff;
							omega_eff[4] = 1.;  ///(c_s2 * thermal_diffusion_coefficient[{X,Y,Z}]*(T_0/E_0) + 0.5);
							omega_eff[5] = 1.;  ///(c_s2 * thermal_diffusion_coefficient[{X,Y,Z}]*(T_0/E_0) + 0.5);
							omega_eff[6] = 1.;  ///(c_s2 * thermal_diffusion_coefficient[{X,Y,Z}]*(T_0/E_0) + 0.5);
							omega_eff[7] = 1.;
							omega_eff[8] = 1.;
							omega_eff[9] = 1.;
							omega_eff[10] = 1.;
							omega_eff[11] = 1.;
							omega_eff[12] = 1.;
							omega_eff[13] = 1.;
							omega_eff[14] = 1.;
							omega_eff[15] = 1.;
							omega_eff[16] = 1.;
							omega_eff[17] = 1.;
							omega_eff[18] = 1.;
							omega_eff[19] = 1.;
							omega_eff[20] = 1.;
							omega_eff[21] = 1.;
							omega_eff[22] = 1.;
							omega_eff[23] = 1.;
							omega_eff[24] = 1.;
							omega_eff[25] = 1.;
							omega_eff[26] = 1.;

							// const double rho = Flow->density[{X,Y,Z}];
							ux = Flow->velocity[{X, Y, Z, 0}];
							uy = Flow->velocity[{X, Y, Z, 1}];
							uz = Flow->velocity[{X, Y, Z, 2}];
							T = temperature[{X, Y, Z}];
							E = energy[{X, Y, Z}];
							p = Flow->pressure[{X, Y, Z}] * (Flow->rho_0 * sqr(global_parameters.D_x / global_parameters.D_t)) / E_0;

							///							Fx = (energy[{X,Y,Z}] + Flow->pressure[{X,Y,Z}])*Flow->velocity[{X,Y,Z,0}]
							///								- (energy_previous[{X,Y,Z}] + Flow->previous_pressure[{X,Y,Z}])*Flow->previous_velocity[{X,Y,Z,0}];
							///							Fx = (energy[{X,Y,Z}] + Flow->pressure[{X,Y,Z}])*Flow->velocity[{X,Y,Z,1}]
							///								- (energy_previous[{X,Y,Z}] + Flow->previous_pressure[{X,Y,Z}])*Flow->previous_velocity[{X,Y,Z,1}];
							/// *************************************************************************************************** ///
							///                                       EQUILIBRIUM CENTRAL MOMENTS                                   ///
							///                                             MOMENT SPACE :                                          ///
							///                                                  M_0,                                               ///
							///                                                M_x, M_y,                                            ///
							///                                       M_xx+M_yy, M_xx-M_yy, M_xy                                    ///
							///                                              M_xxy, M_xyy                                           ///
							///                                                  M_xxyy                                             ///
							/// *************************************************************************************************** ///
							Momeq[0] = E;
							Momeq[1] = (E + p) * ux;
							Momeq[2] = (E + p) * uy;
							Momeq[3] = (E + p) * uz;
							Momeq[4] = 0;
							Momeq[5] = 0;
							Momeq[6] = 0;
							Momeq[7] = (T - E) / c_s2;
							Momeq[8] = (T - E) / c_s2;
							Momeq[9] = (T - E) / c_s2;
							Momeq[10] = 0;
							Momeq[11] = 0;
							Momeq[12] = 0;
							Momeq[13] = 0;
							Momeq[14] = 0;
							Momeq[15] = 0;
							Momeq[16] = 0;
							Momeq[17] = 0;
							Momeq[18] = 0;
							Momeq[19] = 0;
							Momeq[20] = 0;
							Momeq[21] = 0;
							Momeq[22] = 0;
							Momeq[23] = 0;
							Momeq[24] = 0;
							Momeq[25] = 0;
							Momeq[26] = 0;

							Mom[0] = Momeq[0] * omega_eff[0] + (1. - omega_eff[0]) * (pop_old_t[{X, Y, Z, 0}] + pop_old_t[{X, Y, Z, 1}] + pop_old_t[{X, Y, Z, 2}] + pop_old_t[{X, Y, Z, 3}] + pop_old_t[{X, Y, Z, 4}] + pop_old_t[{X, Y, Z, 5}] + pop_old_t[{X, Y, Z, 6}] + pop_old_t[{X, Y, Z, 7}] + pop_old_t[{X, Y, Z, 8}] + pop_old_t[{X, Y, Z, 9}] + pop_old_t[{X, Y, Z, 0}] + pop_old_t[{X, Y, Z, 1}] + pop_old_t[{X, Y, Z, 2}] + pop_old_t[{X, Y, Z, 3}] + pop_old_t[{X, Y, Z, 4}] + pop_old_t[{X, Y, Z, 5}] + pop_old_t[{X, Y, Z, 6}] + pop_old_t[{X, Y, Z, 7}] + pop_old_t[{X, Y, Z, 8}] + pop_old_t[{X, Y, Z, 9}] + pop_old_t[{X, Y, Z, 0}] + pop_old_t[{X, Y, Z, 1}] + pop_old_t[{X, Y, Z, 2}] + pop_old_t[{X, Y, Z, 3}] + pop_old_t[{X, Y, Z, 4}] + pop_old_t[{X, Y, Z, 5}] + pop_old_t[{X, Y, Z, 6}]);
							Mom[1] = Momeq[1] * omega_eff[1] + (1. - omega_eff[1]) * (pop_old_t[{X, Y, Z, 1}] - pop_old_t[{X, Y, Z, 2}] + pop_old_t[{X, Y, Z, 7}] - pop_old_t[{X, Y, Z, 8}] + pop_old_t[{X, Y, Z, 9}] - pop_old_t[{X, Y, Z, 0}] + pop_old_t[{X, Y, Z, 1}] - pop_old_t[{X, Y, Z, 2}] + pop_old_t[{X, Y, Z, 3}] - pop_old_t[{X, Y, Z, 4}] + pop_old_t[{X, Y, Z, 9}] - pop_old_t[{X, Y, Z, 0}] + pop_old_t[{X, Y, Z, 1}] - pop_old_t[{X, Y, Z, 2}] + pop_old_t[{X, Y, Z, 3}] - pop_old_t[{X, Y, Z, 4}] + pop_old_t[{X, Y, Z, 5}] - pop_old_t[{X, Y, Z, 6}]);
							Mom[2] = Momeq[2] * omega_eff[2] + (1. - omega_eff[2]) * (pop_old_t[{X, Y, Z, 3}] - pop_old_t[{X, Y, Z, 4}] + pop_old_t[{X, Y, Z, 7}] + pop_old_t[{X, Y, Z, 8}] - pop_old_t[{X, Y, Z, 9}] - pop_old_t[{X, Y, Z, 0}] + pop_old_t[{X, Y, Z, 5}] - pop_old_t[{X, Y, Z, 6}] + pop_old_t[{X, Y, Z, 7}] - pop_old_t[{X, Y, Z, 8}] + pop_old_t[{X, Y, Z, 9}] + pop_old_t[{X, Y, Z, 0}] - pop_old_t[{X, Y, Z, 1}] - pop_old_t[{X, Y, Z, 2}] + pop_old_t[{X, Y, Z, 3}] + pop_old_t[{X, Y, Z, 4}] - pop_old_t[{X, Y, Z, 5}] - pop_old_t[{X, Y, Z, 6}]);
							Mom[3] = Momeq[3] * omega_eff[3] + (1. - omega_eff[3]) * (pop_old_t[{X, Y, Z, 5}] - pop_old_t[{X, Y, Z, 6}] + pop_old_t[{X, Y, Z, 1}] + pop_old_t[{X, Y, Z, 2}] - pop_old_t[{X, Y, Z, 3}] - pop_old_t[{X, Y, Z, 4}] + pop_old_t[{X, Y, Z, 5}] + pop_old_t[{X, Y, Z, 6}] - pop_old_t[{X, Y, Z, 7}] - pop_old_t[{X, Y, Z, 8}] + pop_old_t[{X, Y, Z, 9}] + pop_old_t[{X, Y, Z, 0}] + pop_old_t[{X, Y, Z, 1}] + pop_old_t[{X, Y, Z, 2}] - pop_old_t[{X, Y, Z, 3}] - pop_old_t[{X, Y, Z, 4}] - pop_old_t[{X, Y, Z, 5}] - pop_old_t[{X, Y, Z, 6}]);
							Mom[4] = Momeq[4] * omega_eff[4] + (1. - omega_eff[4]) * (pop_old_t[{X, Y, Z, 7}] - pop_old_t[{X, Y, Z, 8}] - pop_old_t[{X, Y, Z, 9}] + pop_old_t[{X, Y, Z, 0}] + pop_old_t[{X, Y, Z, 9}] - pop_old_t[{X, Y, Z, 0}] - pop_old_t[{X, Y, Z, 1}] + pop_old_t[{X, Y, Z, 2}] + pop_old_t[{X, Y, Z, 3}] - pop_old_t[{X, Y, Z, 4}] - pop_old_t[{X, Y, Z, 5}] + pop_old_t[{X, Y, Z, 6}]);
							Mom[5] = Momeq[5] * omega_eff[5] + (1. - omega_eff[5]) * (pop_old_t[{X, Y, Z, 1}] - pop_old_t[{X, Y, Z, 2}] - pop_old_t[{X, Y, Z, 3}] + pop_old_t[{X, Y, Z, 4}] + pop_old_t[{X, Y, Z, 9}] - pop_old_t[{X, Y, Z, 0}] + pop_old_t[{X, Y, Z, 1}] - pop_old_t[{X, Y, Z, 2}] - pop_old_t[{X, Y, Z, 3}] + pop_old_t[{X, Y, Z, 4}] - pop_old_t[{X, Y, Z, 5}] + pop_old_t[{X, Y, Z, 6}]);
							Mom[6] = Momeq[6] * omega_eff[6] + (1. - omega_eff[6]) * (pop_old_t[{X, Y, Z, 5}] - pop_old_t[{X, Y, Z, 6}] - pop_old_t[{X, Y, Z, 7}] + pop_old_t[{X, Y, Z, 8}] + pop_old_t[{X, Y, Z, 9}] + pop_old_t[{X, Y, Z, 0}] - pop_old_t[{X, Y, Z, 1}] - pop_old_t[{X, Y, Z, 2}] - pop_old_t[{X, Y, Z, 3}] - pop_old_t[{X, Y, Z, 4}] + pop_old_t[{X, Y, Z, 5}] + pop_old_t[{X, Y, Z, 6}]);
							Mom[7] = Momeq[7] * omega_eff[7] + (1. - omega_eff[7]) * (-pop_old_t[{X, Y, Z, 0}] / (double)c_s2 - pop_old_t[{X, Y, Z, 3}] / (double)c_s2 - pop_old_t[{X, Y, Z, 4}] / (double)c_s2 - pop_old_t[{X, Y, Z, 5}] / (double)c_s2 - pop_old_t[{X, Y, Z, 6}] / (double)c_s2 - pop_old_t[{X, Y, Z, 5}] / (double)c_s2 - pop_old_t[{X, Y, Z, 6}] / (double)c_s2 - pop_old_t[{X, Y, Z, 7}] / (double)c_s2 - pop_old_t[{X, Y, Z, 8}] / (double)c_s2 - pop_old_t[{X, Y, Z, 1}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 2}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 7}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 8}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 9}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 0}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 1}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 2}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 3}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 4}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 9}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 0}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 1}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 2}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 3}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 4}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 5}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 6}] * (1 / (double)c_s2 - 1));
							Mom[8] = Momeq[8] * omega_eff[8] + (1. - omega_eff[8]) * (-pop_old_t[{X, Y, Z, 0}] / (double)c_s2 - pop_old_t[{X, Y, Z, 1}] / (double)c_s2 - pop_old_t[{X, Y, Z, 2}] / (double)c_s2 - pop_old_t[{X, Y, Z, 5}] / (double)c_s2 - pop_old_t[{X, Y, Z, 6}] / (double)c_s2 - pop_old_t[{X, Y, Z, 1}] / (double)c_s2 - pop_old_t[{X, Y, Z, 2}] / (double)c_s2 - pop_old_t[{X, Y, Z, 3}] / (double)c_s2 - pop_old_t[{X, Y, Z, 4}] / (double)c_s2 - pop_old_t[{X, Y, Z, 3}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 4}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 7}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 8}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 9}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 0}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 5}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 6}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 7}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 8}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 9}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 0}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 1}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 2}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 3}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 4}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 5}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 6}] * (1 / (double)c_s2 - 1));
							Mom[9] = Momeq[9] * omega_eff[9] + (1. - omega_eff[9]) * (-pop_old_t[{X, Y, Z, 0}] / (double)c_s2 - pop_old_t[{X, Y, Z, 1}] / (double)c_s2 - pop_old_t[{X, Y, Z, 2}] / (double)c_s2 - pop_old_t[{X, Y, Z, 3}] / (double)c_s2 - pop_old_t[{X, Y, Z, 4}] / (double)c_s2 - pop_old_t[{X, Y, Z, 7}] / (double)c_s2 - pop_old_t[{X, Y, Z, 8}] / (double)c_s2 - pop_old_t[{X, Y, Z, 9}] / (double)c_s2 - pop_old_t[{X, Y, Z, 0}] / (double)c_s2 - pop_old_t[{X, Y, Z, 5}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 6}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 1}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 2}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 3}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 4}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 5}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 6}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 7}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 8}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 9}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 0}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 1}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 2}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 3}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 4}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 5}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 6}] * (1 / (double)c_s2 - 1));
							Mom[10] = Momeq[10] * omega_eff[10] + (1. - omega_eff[10]) * (pop_old_t[{X, Y, Z, 4}] / (double)c_s2 - pop_old_t[{X, Y, Z, 3}] / (double)c_s2 - pop_old_t[{X, Y, Z, 5}] / (double)c_s2 + pop_old_t[{X, Y, Z, 6}] / (double)c_s2 - pop_old_t[{X, Y, Z, 7}] / (double)c_s2 + pop_old_t[{X, Y, Z, 8}] / (double)c_s2 - pop_old_t[{X, Y, Z, 7}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 8}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 9}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 0}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 9}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 0}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 1}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 2}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 3}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 4}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 5}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 6}] * (1 / (double)c_s2 - 1));
							Mom[11] = Momeq[11] * omega_eff[11] + (1. - omega_eff[11]) * (pop_old_t[{X, Y, Z, 6}] / (double)c_s2 - pop_old_t[{X, Y, Z, 5}] / (double)c_s2 - pop_old_t[{X, Y, Z, 5}] / (double)c_s2 - pop_old_t[{X, Y, Z, 6}] / (double)c_s2 + pop_old_t[{X, Y, Z, 7}] / (double)c_s2 + pop_old_t[{X, Y, Z, 8}] / (double)c_s2 - pop_old_t[{X, Y, Z, 1}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 2}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 3}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 4}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 9}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 0}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 1}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 2}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 3}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 4}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 5}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 6}] * (1 / (double)c_s2 - 1));
							Mom[12] = Momeq[12] * omega_eff[12] + (1. - omega_eff[12]) * (pop_old_t[{X, Y, Z, 2}] / (double)c_s2 - pop_old_t[{X, Y, Z, 1}] / (double)c_s2 - pop_old_t[{X, Y, Z, 1}] / (double)c_s2 + pop_old_t[{X, Y, Z, 2}] / (double)c_s2 - pop_old_t[{X, Y, Z, 3}] / (double)c_s2 + pop_old_t[{X, Y, Z, 4}] / (double)c_s2 - pop_old_t[{X, Y, Z, 7}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 8}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 9}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 0}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 9}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 0}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 1}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 2}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 3}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 4}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 5}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 6}] * (1 / (double)c_s2 - 1));
							Mom[13] = Momeq[13] * omega_eff[13] + (1. - omega_eff[13]) * (pop_old_t[{X, Y, Z, 2}] / (double)c_s2 - pop_old_t[{X, Y, Z, 1}] / (double)c_s2 - pop_old_t[{X, Y, Z, 7}] / (double)c_s2 + pop_old_t[{X, Y, Z, 8}] / (double)c_s2 - pop_old_t[{X, Y, Z, 9}] / (double)c_s2 + pop_old_t[{X, Y, Z, 0}] / (double)c_s2 - pop_old_t[{X, Y, Z, 1}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 2}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 3}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 4}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 9}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 0}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 1}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 2}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 3}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 4}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 5}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 6}] * (1 / (double)c_s2 - 1));
							Mom[14] = Momeq[14] * omega_eff[14] + (1. - omega_eff[14]) * (pop_old_t[{X, Y, Z, 4}] / (double)c_s2 - pop_old_t[{X, Y, Z, 3}] / (double)c_s2 - pop_old_t[{X, Y, Z, 7}] / (double)c_s2 - pop_old_t[{X, Y, Z, 8}] / (double)c_s2 + pop_old_t[{X, Y, Z, 9}] / (double)c_s2 + pop_old_t[{X, Y, Z, 0}] / (double)c_s2 - pop_old_t[{X, Y, Z, 5}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 6}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 7}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 8}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 9}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 0}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 1}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 2}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 3}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 4}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 5}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 6}] * (1 / (double)c_s2 - 1));
							Mom[15] = Momeq[15] * omega_eff[15] + (1. - omega_eff[15]) * (pop_old_t[{X, Y, Z, 6}] / (double)c_s2 - pop_old_t[{X, Y, Z, 5}] / (double)c_s2 - pop_old_t[{X, Y, Z, 1}] / (double)c_s2 - pop_old_t[{X, Y, Z, 2}] / (double)c_s2 + pop_old_t[{X, Y, Z, 3}] / (double)c_s2 + pop_old_t[{X, Y, Z, 4}] / (double)c_s2 - pop_old_t[{X, Y, Z, 5}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 6}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 7}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 8}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 9}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 0}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 1}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 2}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 3}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 4}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 5}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 6}] * (1 / (double)c_s2 - 1));
							Mom[16] = Momeq[16] * omega_eff[16] + (1. - omega_eff[16]) * (pop_old_t[{X, Y, Z, 9}] - pop_old_t[{X, Y, Z, 0}] - pop_old_t[{X, Y, Z, 1}] + pop_old_t[{X, Y, Z, 2}] - pop_old_t[{X, Y, Z, 3}] + pop_old_t[{X, Y, Z, 4}] + pop_old_t[{X, Y, Z, 5}] - pop_old_t[{X, Y, Z, 6}]);
							Mom[17] = Momeq[17] * omega_eff[17] + (1. - omega_eff[17]) * (pop_old_t[{X, Y, Z, 7}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 8}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 9}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 0}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 9}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 0}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 1}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 2}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 3}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 4}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 5}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 6}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 0}] / (double)sqr(c_s2) + pop_old_t[{X, Y, Z, 5}] / (double)sqr(c_s2) + pop_old_t[{X, Y, Z, 6}] / (double)sqr(c_s2) + (pop_old_t[{X, Y, Z, 1}] * (1 / (double)c_s2 - 1)) / (double)c_s2 + (pop_old_t[{X, Y, Z, 2}] * (1 / (double)c_s2 - 1)) / (double)c_s2 + (pop_old_t[{X, Y, Z, 3}] * (1 / (double)c_s2 - 1)) / (double)c_s2 + (pop_old_t[{X, Y, Z, 4}] * (1 / (double)c_s2 - 1)) / (double)c_s2 + (pop_old_t[{X, Y, Z, 1}] * (1 / (double)c_s2 - 1)) / (double)c_s2 + (pop_old_t[{X, Y, Z, 2}] * (1 / (double)c_s2 - 1)) / (double)c_s2 + (pop_old_t[{X, Y, Z, 3}] * (1 / (double)c_s2 - 1)) / (double)c_s2 + (pop_old_t[{X, Y, Z, 4}] * (1 / (double)c_s2 - 1)) / (double)c_s2 + (pop_old_t[{X, Y, Z, 5}] * (1 / (double)c_s2 - 1)) / (double)c_s2 + (pop_old_t[{X, Y, Z, 6}] * (1 / (double)c_s2 - 1)) / (double)c_s2 + (pop_old_t[{X, Y, Z, 7}] * (1 / (double)c_s2 - 1)) / (double)c_s2 + (pop_old_t[{X, Y, Z, 8}] * (1 / (double)c_s2 - 1)) / (double)c_s2);
							Mom[18] = Momeq[18] * omega_eff[18] + (1. - omega_eff[18]) * (pop_old_t[{X, Y, Z, 1}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 2}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 3}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 4}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 9}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 0}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 1}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 2}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 3}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 4}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 5}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 6}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 0}] / (double)sqr(c_s2) + pop_old_t[{X, Y, Z, 3}] / (double)sqr(c_s2) + pop_old_t[{X, Y, Z, 4}] / (double)sqr(c_s2) + (pop_old_t[{X, Y, Z, 1}] * (1 / (double)c_s2 - 1)) / (double)c_s2 + (pop_old_t[{X, Y, Z, 2}] * (1 / (double)c_s2 - 1)) / (double)c_s2 + (pop_old_t[{X, Y, Z, 5}] * (1 / (double)c_s2 - 1)) / (double)c_s2 + (pop_old_t[{X, Y, Z, 6}] * (1 / (double)c_s2 - 1)) / (double)c_s2 + (pop_old_t[{X, Y, Z, 7}] * (1 / (double)c_s2 - 1)) / (double)c_s2 + (pop_old_t[{X, Y, Z, 8}] * (1 / (double)c_s2 - 1)) / (double)c_s2 + (pop_old_t[{X, Y, Z, 9}] * (1 / (double)c_s2 - 1)) / (double)c_s2 + (pop_old_t[{X, Y, Z, 0}] * (1 / (double)c_s2 - 1)) / (double)c_s2 + (pop_old_t[{X, Y, Z, 5}] * (1 / (double)c_s2 - 1)) / (double)c_s2 + (pop_old_t[{X, Y, Z, 6}] * (1 / (double)c_s2 - 1)) / (double)c_s2 + (pop_old_t[{X, Y, Z, 7}] * (1 / (double)c_s2 - 1)) / (double)c_s2 + (pop_old_t[{X, Y, Z, 8}] * (1 / (double)c_s2 - 1)) / (double)c_s2);
							Mom[19] = Momeq[19] * omega_eff[19] + (1. - omega_eff[19]) * (pop_old_t[{X, Y, Z, 5}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 6}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 7}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 8}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 9}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 0}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 1}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 2}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 3}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 4}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 5}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 6}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 0}] / (double)sqr(c_s2) + pop_old_t[{X, Y, Z, 1}] / (double)sqr(c_s2) + pop_old_t[{X, Y, Z, 2}] / (double)sqr(c_s2) + (pop_old_t[{X, Y, Z, 3}] * (1 / (double)c_s2 - 1)) / (double)c_s2 + (pop_old_t[{X, Y, Z, 4}] * (1 / (double)c_s2 - 1)) / (double)c_s2 + (pop_old_t[{X, Y, Z, 5}] * (1 / (double)c_s2 - 1)) / (double)c_s2 + (pop_old_t[{X, Y, Z, 6}] * (1 / (double)c_s2 - 1)) / (double)c_s2 + (pop_old_t[{X, Y, Z, 7}] * (1 / (double)c_s2 - 1)) / (double)c_s2 + (pop_old_t[{X, Y, Z, 8}] * (1 / (double)c_s2 - 1)) / (double)c_s2 + (pop_old_t[{X, Y, Z, 9}] * (1 / (double)c_s2 - 1)) / (double)c_s2 + (pop_old_t[{X, Y, Z, 0}] * (1 / (double)c_s2 - 1)) / (double)c_s2 + (pop_old_t[{X, Y, Z, 1}] * (1 / (double)c_s2 - 1)) / (double)c_s2 + (pop_old_t[{X, Y, Z, 2}] * (1 / (double)c_s2 - 1)) / (double)c_s2 + (pop_old_t[{X, Y, Z, 3}] * (1 / (double)c_s2 - 1)) / (double)c_s2 + (pop_old_t[{X, Y, Z, 4}] * (1 / (double)c_s2 - 1)) / (double)c_s2);
							Mom[20] = Momeq[20] * omega_eff[20] + (1. - omega_eff[20]) * (pop_old_t[{X, Y, Z, 8}] / (double)c_s2 - pop_old_t[{X, Y, Z, 7}] / (double)c_s2 + pop_old_t[{X, Y, Z, 9}] / (double)c_s2 - pop_old_t[{X, Y, Z, 0}] / (double)c_s2 - pop_old_t[{X, Y, Z, 9}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 0}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 1}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 2}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 3}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 4}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 5}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 6}] * (1 / (double)c_s2 - 1));
							Mom[21] = Momeq[21] * omega_eff[21] + (1. - omega_eff[21]) * (pop_old_t[{X, Y, Z, 2}] / (double)c_s2 - pop_old_t[{X, Y, Z, 1}] / (double)c_s2 + pop_old_t[{X, Y, Z, 3}] / (double)c_s2 - pop_old_t[{X, Y, Z, 4}] / (double)c_s2 - pop_old_t[{X, Y, Z, 9}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 0}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 1}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 2}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 3}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 4}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 5}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 6}] * (1 / (double)c_s2 - 1));
							Mom[22] = Momeq[22] * omega_eff[22] + (1. - omega_eff[22]) * (pop_old_t[{X, Y, Z, 6}] / (double)c_s2 - pop_old_t[{X, Y, Z, 5}] / (double)c_s2 + pop_old_t[{X, Y, Z, 7}] / (double)c_s2 - pop_old_t[{X, Y, Z, 8}] / (double)c_s2 - pop_old_t[{X, Y, Z, 9}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 0}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 1}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 2}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 3}] * (1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 4}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 5}] * (1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 6}] * (1 / (double)c_s2 - 1));
							Mom[23] = Momeq[23] * omega_eff[23] + (1. - omega_eff[23]) * (pop_old_t[{X, Y, Z, 9}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 0}] * sqr(1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 1}] * sqr(1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 2}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 3}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 4}] * sqr(1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 5}] * sqr(1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 6}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 3}] / (double)sqr(c_s2) - pop_old_t[{X, Y, Z, 4}] / (double)sqr(c_s2) + (pop_old_t[{X, Y, Z, 7}] * (1 / (double)c_s2 - 1)) / (double)c_s2 + (pop_old_t[{X, Y, Z, 8}] * (1 / (double)c_s2 - 1)) / (double)c_s2 - (pop_old_t[{X, Y, Z, 9}] * (1 / (double)c_s2 - 1)) / (double)c_s2 - (pop_old_t[{X, Y, Z, 0}] * (1 / (double)c_s2 - 1)) / (double)c_s2 + (pop_old_t[{X, Y, Z, 5}] * (1 / (double)c_s2 - 1)) / (double)c_s2 - (pop_old_t[{X, Y, Z, 6}] * (1 / (double)c_s2 - 1)) / (double)c_s2 + (pop_old_t[{X, Y, Z, 7}] * (1 / (double)c_s2 - 1)) / (double)c_s2 - (pop_old_t[{X, Y, Z, 8}] * (1 / (double)c_s2 - 1)) / (double)c_s2);
							Mom[24] = Momeq[24] * omega_eff[24] + (1. - omega_eff[24]) * (pop_old_t[{X, Y, Z, 9}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 0}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 1}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 2}] * sqr(1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 3}] * sqr(1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 4}] * sqr(1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 5}] * sqr(1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 6}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 5}] / (double)sqr(c_s2) - pop_old_t[{X, Y, Z, 6}] / (double)sqr(c_s2) + (pop_old_t[{X, Y, Z, 1}] * (1 / (double)c_s2 - 1)) / (double)c_s2 + (pop_old_t[{X, Y, Z, 2}] * (1 / (double)c_s2 - 1)) / (double)c_s2 - (pop_old_t[{X, Y, Z, 3}] * (1 / (double)c_s2 - 1)) / (double)c_s2 - (pop_old_t[{X, Y, Z, 4}] * (1 / (double)c_s2 - 1)) / (double)c_s2 + (pop_old_t[{X, Y, Z, 5}] * (1 / (double)c_s2 - 1)) / (double)c_s2 + (pop_old_t[{X, Y, Z, 6}] * (1 / (double)c_s2 - 1)) / (double)c_s2 - (pop_old_t[{X, Y, Z, 7}] * (1 / (double)c_s2 - 1)) / (double)c_s2 - (pop_old_t[{X, Y, Z, 8}] * (1 / (double)c_s2 - 1)) / (double)c_s2);
							Mom[25] = Momeq[25] * omega_eff[25] + (1. - omega_eff[25]) * (pop_old_t[{X, Y, Z, 9}] * sqr(1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 0}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 1}] * sqr(1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 2}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 3}] * sqr(1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 4}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 5}] * sqr(1 / (double)c_s2 - 1) - pop_old_t[{X, Y, Z, 6}] * sqr(1 / (double)c_s2 - 1) + pop_old_t[{X, Y, Z, 1}] / (double)sqr(c_s2) - pop_old_t[{X, Y, Z, 2}] / (double)sqr(c_s2) + (pop_old_t[{X, Y, Z, 7}] * (1 / (double)c_s2 - 1)) / (double)c_s2 - (pop_old_t[{X, Y, Z, 8}] * (1 / (double)c_s2 - 1)) / (double)c_s2 + (pop_old_t[{X, Y, Z, 9}] * (1 / (double)c_s2 - 1)) / (double)c_s2 - (pop_old_t[{X, Y, Z, 0}] * (1 / (double)c_s2 - 1)) / (double)c_s2 + (pop_old_t[{X, Y, Z, 1}] * (1 / (double)c_s2 - 1)) / (double)c_s2 - (pop_old_t[{X, Y, Z, 2}] * (1 / (double)c_s2 - 1)) / (double)c_s2 + (pop_old_t[{X, Y, Z, 3}] * (1 / (double)c_s2 - 1)) / (double)c_s2 - (pop_old_t[{X, Y, Z, 4}] * (1 / (double)c_s2 - 1)) / (double)c_s2);
							Mom[26] = Momeq[26] * omega_eff[26] + (1. - omega_eff[26]) * (-pop_old_t[{X, Y, Z, 9}] * pow(1 / (double)c_s2 - 1, 3) - pop_old_t[{X, Y, Z, 0}] * pow(1 / (double)c_s2 - 1, 3) - pop_old_t[{X, Y, Z, 1}] * pow(1 / (double)c_s2 - 1, 3) - pop_old_t[{X, Y, Z, 2}] * pow(1 / (double)c_s2 - 1, 3) - pop_old_t[{X, Y, Z, 3}] * pow(1 / (double)c_s2 - 1, 3) - pop_old_t[{X, Y, Z, 4}] * pow(1 / (double)c_s2 - 1, 3) - pop_old_t[{X, Y, Z, 5}] * pow(1 / (double)c_s2 - 1, 3) - pop_old_t[{X, Y, Z, 6}] * pow(1 / (double)c_s2 - 1, 3) - pop_old_t[{X, Y, Z, 0}] / (double)pow(c_s2, 3) - (pop_old_t[{X, Y, Z, 7}] * sqr(1 / (double)c_s2 - 1)) / (double)c_s2 - (pop_old_t[{X, Y, Z, 8}] * sqr(1 / (double)c_s2 - 1)) / (double)c_s2 - (pop_old_t[{X, Y, Z, 9}] * sqr(1 / (double)c_s2 - 1)) / (double)c_s2 - (pop_old_t[{X, Y, Z, 0}] * sqr(1 / (double)c_s2 - 1)) / (double)c_s2 - (pop_old_t[{X, Y, Z, 1}] * sqr(1 / (double)c_s2 - 1)) / (double)c_s2 - (pop_old_t[{X, Y, Z, 2}] * sqr(1 / (double)c_s2 - 1)) / (double)c_s2 - (pop_old_t[{X, Y, Z, 3}] * sqr(1 / (double)c_s2 - 1)) / (double)c_s2 - (pop_old_t[{X, Y, Z, 4}] * sqr(1 / (double)c_s2 - 1)) / (double)c_s2 - (pop_old_t[{X, Y, Z, 5}] * sqr(1 / (double)c_s2 - 1)) / (double)c_s2 - (pop_old_t[{X, Y, Z, 6}] * sqr(1 / (double)c_s2 - 1)) / (double)c_s2 - (pop_old_t[{X, Y, Z, 7}] * sqr(1 / (double)c_s2 - 1)) / (double)c_s2 - (pop_old_t[{X, Y, Z, 8}] * sqr(1 / (double)c_s2 - 1)) / (double)c_s2 - (pop_old_t[{X, Y, Z, 1}] * (1 / (double)c_s2 - 1)) / (double)sqr(c_s2) - (pop_old_t[{X, Y, Z, 2}] * (1 / (double)c_s2 - 1)) / (double)sqr(c_s2) - (pop_old_t[{X, Y, Z, 3}] * (1 / (double)c_s2 - 1)) / (double)sqr(c_s2) - (pop_old_t[{X, Y, Z, 4}] * (1 / (double)c_s2 - 1)) / (double)sqr(c_s2) - (pop_old_t[{X, Y, Z, 5}] * (1 / (double)c_s2 - 1)) / (double)sqr(c_s2) - (pop_old_t[{X, Y, Z, 6}] * (1 / (double)c_s2 - 1)) / (double)sqr(c_s2));

							// Mom[1] += (1. - 0.5*omega_eff[1]) * Fx;
							// Mom[2] += (1. - 0.5*omega_eff[2]) * Fy;

							pop_t[{X + c_alpha[0][0], Y + c_alpha[0][1], Z + c_alpha[0][2], 0}] = (Mom[17] * (c_s2 - 1)) / (double)c_s2 - Mom[26] + (Mom[18] * (c_s2 - 1)) / (double)c_s2 + (Mom[19] * (c_s2 - 1)) / (double)c_s2 + (Mom[0] * pow(c_s2 - 1, 3)) / (double)pow(c_s2, 3) - (Mom[7] * sqr(c_s2 - 1)) / (double)sqr(c_s2) - (Mom[8] * sqr(c_s2 - 1)) / (double)sqr(c_s2) - (Mom[9] * sqr(c_s2 - 1)) / (double)sqr(c_s2);
							pop_t[{X + c_alpha[1][0], Y + c_alpha[1][1], Z + c_alpha[1][2], 1}] = Mom[25] / (double)2 + Mom[26] / (double)2 + Mom[19] / (double)(2 * c_s2) - (Mom[8] * (c_s2 - 1)) / (double)(2 * sqr(c_s2)) - (Mom[9] * (c_s2 - 1)) / (double)(2 * sqr(c_s2)) - (Mom[12] * (c_s2 - 1)) / (double)(2 * c_s2) - (Mom[13] * (c_s2 - 1)) / (double)(2 * c_s2) - (Mom[17] * (c_s2 - 1)) / (double)(2 * c_s2) - (Mom[18] * (c_s2 - 1)) / (double)(2 * c_s2) + (Mom[0] * sqr(c_s2 - 1)) / (double)(2 * pow(c_s2, 3)) + (Mom[1] * sqr(c_s2 - 1)) / (double)(2 * sqr(c_s2)) + (Mom[7] * sqr(c_s2 - 1)) / (double)(2 * sqr(c_s2));
							pop_t[{X + c_alpha[2][0], Y + c_alpha[2][1], Z + c_alpha[2][2], 2}] = Mom[26] / (double)2 - Mom[25] / (double)2 + Mom[19] / (double)(2 * c_s2) - (Mom[8] * (c_s2 - 1)) / (double)(2 * sqr(c_s2)) - (Mom[9] * (c_s2 - 1)) / (double)(2 * sqr(c_s2)) + (Mom[12] * (c_s2 - 1)) / (double)(2 * c_s2) + (Mom[13] * (c_s2 - 1)) / (double)(2 * c_s2) - (Mom[17] * (c_s2 - 1)) / (double)(2 * c_s2) - (Mom[18] * (c_s2 - 1)) / (double)(2 * c_s2) + (Mom[0] * sqr(c_s2 - 1)) / (double)(2 * pow(c_s2, 3)) - (Mom[1] * sqr(c_s2 - 1)) / (double)(2 * sqr(c_s2)) + (Mom[7] * sqr(c_s2 - 1)) / (double)(2 * sqr(c_s2));
							pop_t[{X + c_alpha[3][0], Y + c_alpha[3][1], Z + c_alpha[3][2], 3}] = Mom[23] / (double)2 + Mom[26] / (double)2 + Mom[18] / (double)(2 * c_s2) - (Mom[7] * (c_s2 - 1)) / (double)(2 * sqr(c_s2)) - (Mom[9] * (c_s2 - 1)) / (double)(2 * sqr(c_s2)) - (Mom[10] * (c_s2 - 1)) / (double)(2 * c_s2) - (Mom[14] * (c_s2 - 1)) / (double)(2 * c_s2) - (Mom[17] * (c_s2 - 1)) / (double)(2 * c_s2) - (Mom[19] * (c_s2 - 1)) / (double)(2 * c_s2) + (Mom[0] * sqr(c_s2 - 1)) / (double)(2 * pow(c_s2, 3)) + (Mom[2] * sqr(c_s2 - 1)) / (double)(2 * sqr(c_s2)) + (Mom[8] * sqr(c_s2 - 1)) / (double)(2 * sqr(c_s2));
							pop_t[{X + c_alpha[4][0], Y + c_alpha[4][1], Z + c_alpha[4][2], 4}] = Mom[26] / (double)2 - Mom[23] / (double)2 + Mom[18] / (double)(2 * c_s2) - (Mom[7] * (c_s2 - 1)) / (double)(2 * sqr(c_s2)) - (Mom[9] * (c_s2 - 1)) / (double)(2 * sqr(c_s2)) + (Mom[10] * (c_s2 - 1)) / (double)(2 * c_s2) + (Mom[14] * (c_s2 - 1)) / (double)(2 * c_s2) - (Mom[17] * (c_s2 - 1)) / (double)(2 * c_s2) - (Mom[19] * (c_s2 - 1)) / (double)(2 * c_s2) + (Mom[0] * sqr(c_s2 - 1)) / (double)(2 * pow(c_s2, 3)) - (Mom[2] * sqr(c_s2 - 1)) / (double)(2 * sqr(c_s2)) + (Mom[8] * sqr(c_s2 - 1)) / (double)(2 * sqr(c_s2));
							pop_t[{X + c_alpha[5][0], Y + c_alpha[5][1], Z + c_alpha[5][2], 5}] = Mom[24] / (double)2 + Mom[26] / (double)2 + Mom[17] / (double)(2 * c_s2) - (Mom[7] * (c_s2 - 1)) / (double)(2 * sqr(c_s2)) - (Mom[8] * (c_s2 - 1)) / (double)(2 * sqr(c_s2)) - (Mom[11] * (c_s2 - 1)) / (double)(2 * c_s2) - (Mom[15] * (c_s2 - 1)) / (double)(2 * c_s2) - (Mom[18] * (c_s2 - 1)) / (double)(2 * c_s2) - (Mom[19] * (c_s2 - 1)) / (double)(2 * c_s2) + (Mom[0] * sqr(c_s2 - 1)) / (double)(2 * pow(c_s2, 3)) + (Mom[3] * sqr(c_s2 - 1)) / (double)(2 * sqr(c_s2)) + (Mom[9] * sqr(c_s2 - 1)) / (double)(2 * sqr(c_s2));
							pop_t[{X + c_alpha[6][0], Y + c_alpha[6][1], Z + c_alpha[6][2], 6}] = Mom[26] / (double)2 - Mom[24] / (double)2 + Mom[17] / (double)(2 * c_s2) - (Mom[7] * (c_s2 - 1)) / (double)(2 * sqr(c_s2)) - (Mom[8] * (c_s2 - 1)) / (double)(2 * sqr(c_s2)) + (Mom[11] * (c_s2 - 1)) / (double)(2 * c_s2) + (Mom[15] * (c_s2 - 1)) / (double)(2 * c_s2) - (Mom[18] * (c_s2 - 1)) / (double)(2 * c_s2) - (Mom[19] * (c_s2 - 1)) / (double)(2 * c_s2) + (Mom[0] * sqr(c_s2 - 1)) / (double)(2 * pow(c_s2, 3)) - (Mom[3] * sqr(c_s2 - 1)) / (double)(2 * sqr(c_s2)) + (Mom[9] * sqr(c_s2 - 1)) / (double)(2 * sqr(c_s2));
							pop_t[{X + c_alpha[7][0], Y + c_alpha[7][1], Z + c_alpha[7][2], 7}] = (Mom[0] * (c_s2 - 1)) / (double)(4 * pow(c_s2, 3)) - Mom[23] / (double)4 - Mom[25] / (double)4 - Mom[26] / (double)4 - Mom[9] / (double)(4 * sqr(c_s2)) - Mom[13] / (double)(4 * c_s2) - Mom[14] / (double)(4 * c_s2) - Mom[18] / (double)(4 * c_s2) - Mom[19] / (double)(4 * c_s2) - Mom[20] / (double)4 + (Mom[1] * (c_s2 - 1)) / (double)(4 * sqr(c_s2)) + (Mom[2] * (c_s2 - 1)) / (double)(4 * sqr(c_s2)) + (Mom[4] * (c_s2 - 1)) / (double)(4 * c_s2) + (Mom[7] * (c_s2 - 1)) / (double)(4 * sqr(c_s2)) + (Mom[8] * (c_s2 - 1)) / (double)(4 * sqr(c_s2)) + (Mom[10] * (c_s2 - 1)) / (double)(4 * c_s2) + (Mom[12] * (c_s2 - 1)) / (double)(4 * c_s2) + (Mom[17] * (c_s2 - 1)) / (double)(4 * c_s2);
							pop_t[{X + c_alpha[8][0], Y + c_alpha[8][1], Z + c_alpha[8][2], 8}] = Mom[20] / (double)4 - Mom[23] / (double)4 + Mom[25] / (double)4 - Mom[26] / (double)4 - Mom[9] / (double)(4 * sqr(c_s2)) + Mom[13] / (double)(4 * c_s2) - Mom[14] / (double)(4 * c_s2) - Mom[18] / (double)(4 * c_s2) - Mom[19] / (double)(4 * c_s2) + (Mom[0] * (c_s2 - 1)) / (double)(4 * pow(c_s2, 3)) - (Mom[1] * (c_s2 - 1)) / (double)(4 * sqr(c_s2)) + (Mom[2] * (c_s2 - 1)) / (double)(4 * sqr(c_s2)) - (Mom[4] * (c_s2 - 1)) / (double)(4 * c_s2) + (Mom[7] * (c_s2 - 1)) / (double)(4 * sqr(c_s2)) + (Mom[8] * (c_s2 - 1)) / (double)(4 * sqr(c_s2)) + (Mom[10] * (c_s2 - 1)) / (double)(4 * c_s2) - (Mom[12] * (c_s2 - 1)) / (double)(4 * c_s2) + (Mom[17] * (c_s2 - 1)) / (double)(4 * c_s2);
							pop_t[{X + c_alpha[9][0], Y + c_alpha[9][1], Z + c_alpha[9][2], 9}] = Mom[20] / (double)4 + Mom[23] / (double)4 - Mom[25] / (double)4 - Mom[26] / (double)4 - Mom[9] / (double)(4 * sqr(c_s2)) - Mom[13] / (double)(4 * c_s2) + Mom[14] / (double)(4 * c_s2) - Mom[18] / (double)(4 * c_s2) - Mom[19] / (double)(4 * c_s2) + (Mom[0] * (c_s2 - 1)) / (double)(4 * pow(c_s2, 3)) + (Mom[1] * (c_s2 - 1)) / (double)(4 * sqr(c_s2)) - (Mom[2] * (c_s2 - 1)) / (double)(4 * sqr(c_s2)) - (Mom[4] * (c_s2 - 1)) / (double)(4 * c_s2) + (Mom[7] * (c_s2 - 1)) / (double)(4 * sqr(c_s2)) + (Mom[8] * (c_s2 - 1)) / (double)(4 * sqr(c_s2)) - (Mom[10] * (c_s2 - 1)) / (double)(4 * c_s2) + (Mom[12] * (c_s2 - 1)) / (double)(4 * c_s2) + (Mom[17] * (c_s2 - 1)) / (double)(4 * c_s2);
							pop_t[{X + c_alpha[10][0], Y + c_alpha[10][1], Z + c_alpha[10][2], 10}] = Mom[23] / (double)4 - Mom[20] / (double)4 + Mom[25] / (double)4 - Mom[26] / (double)4 - Mom[9] / (double)(4 * sqr(c_s2)) + Mom[13] / (double)(4 * c_s2) + Mom[14] / (double)(4 * c_s2) - Mom[18] / (double)(4 * c_s2) - Mom[19] / (double)(4 * c_s2) + (Mom[0] * (c_s2 - 1)) / (double)(4 * pow(c_s2, 3)) - (Mom[1] * (c_s2 - 1)) / (double)(4 * sqr(c_s2)) - (Mom[2] * (c_s2 - 1)) / (double)(4 * sqr(c_s2)) + (Mom[4] * (c_s2 - 1)) / (double)(4 * c_s2) + (Mom[7] * (c_s2 - 1)) / (double)(4 * sqr(c_s2)) + (Mom[8] * (c_s2 - 1)) / (double)(4 * sqr(c_s2)) - (Mom[10] * (c_s2 - 1)) / (double)(4 * c_s2) - (Mom[12] * (c_s2 - 1)) / (double)(4 * c_s2) + (Mom[17] * (c_s2 - 1)) / (double)(4 * c_s2);
							pop_t[{X + c_alpha[11][0], Y + c_alpha[11][1], Z + c_alpha[11][2], 11}] = (Mom[0] * (c_s2 - 1)) / (double)(4 * pow(c_s2, 3)) - Mom[24] / (double)4 - Mom[25] / (double)4 - Mom[26] / (double)4 - Mom[8] / (double)(4 * sqr(c_s2)) - Mom[12] / (double)(4 * c_s2) - Mom[15] / (double)(4 * c_s2) - Mom[17] / (double)(4 * c_s2) - Mom[19] / (double)(4 * c_s2) - Mom[21] / (double)4 + (Mom[1] * (c_s2 - 1)) / (double)(4 * sqr(c_s2)) + (Mom[3] * (c_s2 - 1)) / (double)(4 * sqr(c_s2)) + (Mom[5] * (c_s2 - 1)) / (double)(4 * c_s2) + (Mom[7] * (c_s2 - 1)) / (double)(4 * sqr(c_s2)) + (Mom[9] * (c_s2 - 1)) / (double)(4 * sqr(c_s2)) + (Mom[11] * (c_s2 - 1)) / (double)(4 * c_s2) + (Mom[13] * (c_s2 - 1)) / (double)(4 * c_s2) + (Mom[18] * (c_s2 - 1)) / (double)(4 * c_s2);
							pop_t[{X + c_alpha[12][0], Y + c_alpha[12][1], Z + c_alpha[12][2], 12}] = Mom[21] / (double)4 - Mom[24] / (double)4 + Mom[25] / (double)4 - Mom[26] / (double)4 - Mom[8] / (double)(4 * sqr(c_s2)) + Mom[12] / (double)(4 * c_s2) - Mom[15] / (double)(4 * c_s2) - Mom[17] / (double)(4 * c_s2) - Mom[19] / (double)(4 * c_s2) + (Mom[0] * (c_s2 - 1)) / (double)(4 * pow(c_s2, 3)) - (Mom[1] * (c_s2 - 1)) / (double)(4 * sqr(c_s2)) + (Mom[3] * (c_s2 - 1)) / (double)(4 * sqr(c_s2)) - (Mom[5] * (c_s2 - 1)) / (double)(4 * c_s2) + (Mom[7] * (c_s2 - 1)) / (double)(4 * sqr(c_s2)) + (Mom[9] * (c_s2 - 1)) / (double)(4 * sqr(c_s2)) + (Mom[11] * (c_s2 - 1)) / (double)(4 * c_s2) - (Mom[13] * (c_s2 - 1)) / (double)(4 * c_s2) + (Mom[18] * (c_s2 - 1)) / (double)(4 * c_s2);
							pop_t[{X + c_alpha[13][0], Y + c_alpha[13][1], Z + c_alpha[13][2], 13}] = Mom[21] / (double)4 + Mom[24] / (double)4 - Mom[25] / (double)4 - Mom[26] / (double)4 - Mom[8] / (double)(4 * sqr(c_s2)) - Mom[12] / (double)(4 * c_s2) + Mom[15] / (double)(4 * c_s2) - Mom[17] / (double)(4 * c_s2) - Mom[19] / (double)(4 * c_s2) + (Mom[0] * (c_s2 - 1)) / (double)(4 * pow(c_s2, 3)) + (Mom[1] * (c_s2 - 1)) / (double)(4 * sqr(c_s2)) - (Mom[3] * (c_s2 - 1)) / (double)(4 * sqr(c_s2)) - (Mom[5] * (c_s2 - 1)) / (double)(4 * c_s2) + (Mom[7] * (c_s2 - 1)) / (double)(4 * sqr(c_s2)) + (Mom[9] * (c_s2 - 1)) / (double)(4 * sqr(c_s2)) - (Mom[11] * (c_s2 - 1)) / (double)(4 * c_s2) + (Mom[13] * (c_s2 - 1)) / (double)(4 * c_s2) + (Mom[18] * (c_s2 - 1)) / (double)(4 * c_s2);
							pop_t[{X + c_alpha[14][0], Y + c_alpha[14][1], Z + c_alpha[14][2], 14}] = Mom[24] / (double)4 - Mom[21] / (double)4 + Mom[25] / (double)4 - Mom[26] / (double)4 - Mom[8] / (double)(4 * sqr(c_s2)) + Mom[12] / (double)(4 * c_s2) + Mom[15] / (double)(4 * c_s2) - Mom[17] / (double)(4 * c_s2) - Mom[19] / (double)(4 * c_s2) + (Mom[0] * (c_s2 - 1)) / (double)(4 * pow(c_s2, 3)) - (Mom[1] * (c_s2 - 1)) / (double)(4 * sqr(c_s2)) - (Mom[3] * (c_s2 - 1)) / (double)(4 * sqr(c_s2)) + (Mom[5] * (c_s2 - 1)) / (double)(4 * c_s2) + (Mom[7] * (c_s2 - 1)) / (double)(4 * sqr(c_s2)) + (Mom[9] * (c_s2 - 1)) / (double)(4 * sqr(c_s2)) - (Mom[11] * (c_s2 - 1)) / (double)(4 * c_s2) - (Mom[13] * (c_s2 - 1)) / (double)(4 * c_s2) + (Mom[18] * (c_s2 - 1)) / (double)(4 * c_s2);
							pop_t[{X + c_alpha[15][0], Y + c_alpha[15][1], Z + c_alpha[15][2], 15}] = (Mom[0] * (c_s2 - 1)) / (double)(4 * pow(c_s2, 3)) - Mom[23] / (double)4 - Mom[24] / (double)4 - Mom[26] / (double)4 - Mom[7] / (double)(4 * sqr(c_s2)) - Mom[10] / (double)(4 * c_s2) - Mom[11] / (double)(4 * c_s2) - Mom[17] / (double)(4 * c_s2) - Mom[18] / (double)(4 * c_s2) - Mom[22] / (double)4 + (Mom[2] * (c_s2 - 1)) / (double)(4 * sqr(c_s2)) + (Mom[3] * (c_s2 - 1)) / (double)(4 * sqr(c_s2)) + (Mom[6] * (c_s2 - 1)) / (double)(4 * c_s2) + (Mom[8] * (c_s2 - 1)) / (double)(4 * sqr(c_s2)) + (Mom[9] * (c_s2 - 1)) / (double)(4 * sqr(c_s2)) + (Mom[14] * (c_s2 - 1)) / (double)(4 * c_s2) + (Mom[15] * (c_s2 - 1)) / (double)(4 * c_s2) + (Mom[19] * (c_s2 - 1)) / (double)(4 * c_s2);
							pop_t[{X + c_alpha[16][0], Y + c_alpha[16][1], Z + c_alpha[16][2], 16}] = Mom[22] / (double)4 + Mom[23] / (double)4 - Mom[24] / (double)4 - Mom[26] / (double)4 - Mom[7] / (double)(4 * sqr(c_s2)) + Mom[10] / (double)(4 * c_s2) - Mom[11] / (double)(4 * c_s2) - Mom[17] / (double)(4 * c_s2) - Mom[18] / (double)(4 * c_s2) + (Mom[0] * (c_s2 - 1)) / (double)(4 * pow(c_s2, 3)) - (Mom[2] * (c_s2 - 1)) / (double)(4 * sqr(c_s2)) + (Mom[3] * (c_s2 - 1)) / (double)(4 * sqr(c_s2)) - (Mom[6] * (c_s2 - 1)) / (double)(4 * c_s2) + (Mom[8] * (c_s2 - 1)) / (double)(4 * sqr(c_s2)) + (Mom[9] * (c_s2 - 1)) / (double)(4 * sqr(c_s2)) - (Mom[14] * (c_s2 - 1)) / (double)(4 * c_s2) + (Mom[15] * (c_s2 - 1)) / (double)(4 * c_s2) + (Mom[19] * (c_s2 - 1)) / (double)(4 * c_s2);
							pop_t[{X + c_alpha[17][0], Y + c_alpha[17][1], Z + c_alpha[17][2], 17}] = Mom[22] / (double)4 - Mom[23] / (double)4 + Mom[24] / (double)4 - Mom[26] / (double)4 - Mom[7] / (double)(4 * sqr(c_s2)) - Mom[10] / (double)(4 * c_s2) + Mom[11] / (double)(4 * c_s2) - Mom[17] / (double)(4 * c_s2) - Mom[18] / (double)(4 * c_s2) + (Mom[0] * (c_s2 - 1)) / (double)(4 * pow(c_s2, 3)) + (Mom[2] * (c_s2 - 1)) / (double)(4 * sqr(c_s2)) - (Mom[3] * (c_s2 - 1)) / (double)(4 * sqr(c_s2)) - (Mom[6] * (c_s2 - 1)) / (double)(4 * c_s2) + (Mom[8] * (c_s2 - 1)) / (double)(4 * sqr(c_s2)) + (Mom[9] * (c_s2 - 1)) / (double)(4 * sqr(c_s2)) + (Mom[14] * (c_s2 - 1)) / (double)(4 * c_s2) - (Mom[15] * (c_s2 - 1)) / (double)(4 * c_s2) + (Mom[19] * (c_s2 - 1)) / (double)(4 * c_s2);
							pop_t[{X + c_alpha[18][0], Y + c_alpha[18][1], Z + c_alpha[18][2], 18}] = Mom[23] / (double)4 - Mom[22] / (double)4 + Mom[24] / (double)4 - Mom[26] / (double)4 - Mom[7] / (double)(4 * sqr(c_s2)) + Mom[10] / (double)(4 * c_s2) + Mom[11] / (double)(4 * c_s2) - Mom[17] / (double)(4 * c_s2) - Mom[18] / (double)(4 * c_s2) + (Mom[0] * (c_s2 - 1)) / (double)(4 * pow(c_s2, 3)) - (Mom[2] * (c_s2 - 1)) / (double)(4 * sqr(c_s2)) - (Mom[3] * (c_s2 - 1)) / (double)(4 * sqr(c_s2)) + (Mom[6] * (c_s2 - 1)) / (double)(4 * c_s2) + (Mom[8] * (c_s2 - 1)) / (double)(4 * sqr(c_s2)) + (Mom[9] * (c_s2 - 1)) / (double)(4 * sqr(c_s2)) - (Mom[14] * (c_s2 - 1)) / (double)(4 * c_s2) - (Mom[15] * (c_s2 - 1)) / (double)(4 * c_s2) + (Mom[19] * (c_s2 - 1)) / (double)(4 * c_s2);
							pop_t[{X + c_alpha[19][0], Y + c_alpha[19][1], Z + c_alpha[19][2], 19}] = (Mom[0] + Mom[1] * c_s2 + Mom[2] * c_s2 + Mom[3] * c_s2 + Mom[7] * c_s2 + Mom[8] * c_s2 + Mom[9] * c_s2 + Mom[4] * sqr(c_s2) + Mom[5] * sqr(c_s2) + Mom[6] * sqr(c_s2) + Mom[10] * sqr(c_s2) + Mom[11] * sqr(c_s2) + Mom[12] * sqr(c_s2) + Mom[13] * sqr(c_s2) + Mom[14] * sqr(c_s2) + Mom[15] * sqr(c_s2) + Mom[16] * pow(c_s2, 3) + Mom[17] * sqr(c_s2) + Mom[18] * sqr(c_s2) + Mom[19] * sqr(c_s2) + Mom[20] * pow(c_s2, 3) + Mom[21] * pow(c_s2, 3) + Mom[22] * pow(c_s2, 3) + Mom[23] * pow(c_s2, 3) + Mom[24] * pow(c_s2, 3) + Mom[25] * pow(c_s2, 3) + Mom[26] * pow(c_s2, 3)) / (double)(8 * pow(c_s2, 3));
							pop_t[{X + c_alpha[20][0], Y + c_alpha[20][1], Z + c_alpha[20][2], 20}] = (Mom[0] - Mom[1] * c_s2 + Mom[2] * c_s2 + Mom[3] * c_s2 + Mom[7] * c_s2 + Mom[8] * c_s2 + Mom[9] * c_s2 - Mom[4] * sqr(c_s2) - Mom[5] * sqr(c_s2) + Mom[6] * sqr(c_s2) + Mom[10] * sqr(c_s2) + Mom[11] * sqr(c_s2) - Mom[12] * sqr(c_s2) - Mom[13] * sqr(c_s2) + Mom[14] * sqr(c_s2) + Mom[15] * sqr(c_s2) - Mom[16] * pow(c_s2, 3) + Mom[17] * sqr(c_s2) + Mom[18] * sqr(c_s2) + Mom[19] * sqr(c_s2) - Mom[20] * pow(c_s2, 3) - Mom[21] * pow(c_s2, 3) + Mom[22] * pow(c_s2, 3) + Mom[23] * pow(c_s2, 3) + Mom[24] * pow(c_s2, 3) - Mom[25] * pow(c_s2, 3) + Mom[26] * pow(c_s2, 3)) / (double)(8 * pow(c_s2, 3));
							pop_t[{X + c_alpha[21][0], Y + c_alpha[21][1], Z + c_alpha[21][2], 21}] = (Mom[0] + Mom[1] * c_s2 - Mom[2] * c_s2 + Mom[3] * c_s2 + Mom[7] * c_s2 + Mom[8] * c_s2 + Mom[9] * c_s2 - Mom[4] * sqr(c_s2) + Mom[5] * sqr(c_s2) - Mom[6] * sqr(c_s2) - Mom[10] * sqr(c_s2) + Mom[11] * sqr(c_s2) + Mom[12] * sqr(c_s2) + Mom[13] * sqr(c_s2) - Mom[14] * sqr(c_s2) + Mom[15] * sqr(c_s2) - Mom[16] * pow(c_s2, 3) + Mom[17] * sqr(c_s2) + Mom[18] * sqr(c_s2) + Mom[19] * sqr(c_s2) - Mom[20] * pow(c_s2, 3) + Mom[21] * pow(c_s2, 3) - Mom[22] * pow(c_s2, 3) - Mom[23] * pow(c_s2, 3) + Mom[24] * pow(c_s2, 3) + Mom[25] * pow(c_s2, 3) + Mom[26] * pow(c_s2, 3)) / (double)(8 * pow(c_s2, 3));
							pop_t[{X + c_alpha[22][0], Y + c_alpha[22][1], Z + c_alpha[22][2], 22}] = (Mom[0] - Mom[1] * c_s2 - Mom[2] * c_s2 + Mom[3] * c_s2 + Mom[7] * c_s2 + Mom[8] * c_s2 + Mom[9] * c_s2 + Mom[4] * sqr(c_s2) - Mom[5] * sqr(c_s2) - Mom[6] * sqr(c_s2) - Mom[10] * sqr(c_s2) + Mom[11] * sqr(c_s2) - Mom[12] * sqr(c_s2) - Mom[13] * sqr(c_s2) - Mom[14] * sqr(c_s2) + Mom[15] * sqr(c_s2) + Mom[16] * pow(c_s2, 3) + Mom[17] * sqr(c_s2) + Mom[18] * sqr(c_s2) + Mom[19] * sqr(c_s2) + Mom[20] * pow(c_s2, 3) - Mom[21] * pow(c_s2, 3) - Mom[22] * pow(c_s2, 3) - Mom[23] * pow(c_s2, 3) + Mom[24] * pow(c_s2, 3) - Mom[25] * pow(c_s2, 3) + Mom[26] * pow(c_s2, 3)) / (double)(8 * pow(c_s2, 3));
							pop_t[{X + c_alpha[23][0], Y + c_alpha[23][1], Z + c_alpha[23][2], 23}] = (Mom[0] + Mom[1] * c_s2 + Mom[2] * c_s2 - Mom[3] * c_s2 + Mom[7] * c_s2 + Mom[8] * c_s2 + Mom[9] * c_s2 + Mom[4] * sqr(c_s2) - Mom[5] * sqr(c_s2) - Mom[6] * sqr(c_s2) + Mom[10] * sqr(c_s2) - Mom[11] * sqr(c_s2) + Mom[12] * sqr(c_s2) + Mom[13] * sqr(c_s2) + Mom[14] * sqr(c_s2) - Mom[15] * sqr(c_s2) - Mom[16] * pow(c_s2, 3) + Mom[17] * sqr(c_s2) + Mom[18] * sqr(c_s2) + Mom[19] * sqr(c_s2) + Mom[20] * pow(c_s2, 3) - Mom[21] * pow(c_s2, 3) - Mom[22] * pow(c_s2, 3) + Mom[23] * pow(c_s2, 3) - Mom[24] * pow(c_s2, 3) + Mom[25] * pow(c_s2, 3) + Mom[26] * pow(c_s2, 3)) / (double)(8 * pow(c_s2, 3));
							pop_t[{X + c_alpha[24][0], Y + c_alpha[24][1], Z + c_alpha[24][2], 24}] = (Mom[0] - Mom[1] * c_s2 + Mom[2] * c_s2 - Mom[3] * c_s2 + Mom[7] * c_s2 + Mom[8] * c_s2 + Mom[9] * c_s2 - Mom[4] * sqr(c_s2) + Mom[5] * sqr(c_s2) - Mom[6] * sqr(c_s2) + Mom[10] * sqr(c_s2) - Mom[11] * sqr(c_s2) - Mom[12] * sqr(c_s2) - Mom[13] * sqr(c_s2) + Mom[14] * sqr(c_s2) - Mom[15] * sqr(c_s2) + Mom[16] * pow(c_s2, 3) + Mom[17] * sqr(c_s2) + Mom[18] * sqr(c_s2) + Mom[19] * sqr(c_s2) - Mom[20] * pow(c_s2, 3) + Mom[21] * pow(c_s2, 3) - Mom[22] * pow(c_s2, 3) + Mom[23] * pow(c_s2, 3) - Mom[24] * pow(c_s2, 3) - Mom[25] * pow(c_s2, 3) + Mom[26] * pow(c_s2, 3)) / (double)(8 * pow(c_s2, 3));
							pop_t[{X + c_alpha[25][0], Y + c_alpha[25][1], Z + c_alpha[25][2], 25}] = (Mom[0] + Mom[1] * c_s2 - Mom[2] * c_s2 - Mom[3] * c_s2 + Mom[7] * c_s2 + Mom[8] * c_s2 + Mom[9] * c_s2 - Mom[4] * sqr(c_s2) - Mom[5] * sqr(c_s2) + Mom[6] * sqr(c_s2) - Mom[10] * sqr(c_s2) - Mom[11] * sqr(c_s2) + Mom[12] * sqr(c_s2) + Mom[13] * sqr(c_s2) - Mom[14] * sqr(c_s2) - Mom[15] * sqr(c_s2) + Mom[16] * pow(c_s2, 3) + Mom[17] * sqr(c_s2) + Mom[18] * sqr(c_s2) + Mom[19] * sqr(c_s2) - Mom[20] * pow(c_s2, 3) - Mom[21] * pow(c_s2, 3) + Mom[22] * pow(c_s2, 3) - Mom[23] * pow(c_s2, 3) - Mom[24] * pow(c_s2, 3) + Mom[25] * pow(c_s2, 3) + Mom[26] * pow(c_s2, 3)) / (double)(8 * pow(c_s2, 3));
							pop_t[{X + c_alpha[26][0], Y + c_alpha[26][1], Z + c_alpha[26][2], 26}] = (Mom[0] - Mom[1] * c_s2 - Mom[2] * c_s2 - Mom[3] * c_s2 + Mom[7] * c_s2 + Mom[8] * c_s2 + Mom[9] * c_s2 + Mom[4] * sqr(c_s2) + Mom[5] * sqr(c_s2) + Mom[6] * sqr(c_s2) - Mom[10] * sqr(c_s2) - Mom[11] * sqr(c_s2) - Mom[12] * sqr(c_s2) - Mom[13] * sqr(c_s2) - Mom[14] * sqr(c_s2) - Mom[15] * sqr(c_s2) - Mom[16] * pow(c_s2, 3) + Mom[17] * sqr(c_s2) + Mom[18] * sqr(c_s2) + Mom[19] * sqr(c_s2) + Mom[20] * pow(c_s2, 3) + Mom[21] * pow(c_s2, 3) + Mom[22] * pow(c_s2, 3) - Mom[23] * pow(c_s2, 3) - Mom[24] * pow(c_s2, 3) - Mom[25] * pow(c_s2, 3) + Mom[26] * pow(c_s2, 3)) / (double)(8 * pow(c_s2, 3));
						}
					}
				}
			}
		}
		return;
	}
}
/// ***************************************************** ///
/// UPDATE TEMPERATURE (WITH ADVECTION TERM) USING FD     ///
/// ***************************************************** ///
/* ***************ENERGY BALANCE EQUATION*********
Convective-diffusive energy balance equation for a multi-Species reactive Flow.
    ρ * Cp * (∂T/∂t + u·∂T/∂xi) = ∂·(λ.∂T/∂xi)/∂xi + ρ Σ_k=1^N_sp (cpk * Yk * Vk · ∂T/∂xi) + ω˙T
    λ = thermal_diffusion_coefficient D
    Source: TNC book page 24. eq:1.75
solved here is: u·∂T/∂xi]
    [∂T/∂t + u·∂T/∂xi] = [∂·(D.∂T/∂xi)/∂xi/(ρ * Cp) + ω˙T/(ρ * Cp)]+ [+ Σ_k=1^N_sp (cpk * Yk * Vk · ∂T/∂xi)/(Cp)]
    [FD_Euler]		   = [FD_Euler_diffusion] + [FD_Euler_species_diffusion_enthalpy]

where,
-N_sp is the number of Species.
-Y_k is the mass fraction of Species k.
-V_k is the velocity of Species k.
-ω˙T represents any volumetric heat sources or sinks due to chemical reactions.
-ρ is the density.
-Cp is the specific heat capacity.
-λ is the Thermal conductivity. In the book it is mentioned as "heat diffusion coefficient". It is same as Thermal conductivity.
    It is denoted by thermal_diffusion_coefficient in the code.
-u is the velocity vector of the fluid.
-∂T/∂xi is the gradient of temperature.
-∂T/∂t is the time rate of change of temperature.
-∂·(λ.∂T/∂xi)/∂xi is the diffusive term representing the spreading or dissipation of temperature due to molecular diffusion (Thermal conductivity).
-ρ Σ_k=1^N_sp (cpk * Yk * Vk · ∂T/∂xi) represents the convective term due to the transport of heat by the Species.
The convective term accounts for the movement of heat with the Species velocity.

The terms on the left side of the equation represent the convective and diffusive heat transfer
The term on the right side of the equation represents the heat sources or sinks due to chemical reactions.
The equation describes the energy balance in a reactive Flow considering convective heat transfer, diffusive heat transfer, and heat sources or sinks due to chemical reactions.
The convective term represents the transport of heat by the fluid Flow, and the diffusive term represents the spreading or dissipation of heat due to molecular diffusion (Thermal conductivity).
The production term is scaled by the temperature T0 to account for the temperature dependence of the reaction rates.
The equation is solved using finite-difference schemes to calculate the advection and diffusion terms, and the production term is calculated using the reaction rates and Species mass fractions.
************************************************************************************************************************************************************************************
*/
/* 1: "FD_Euler" solves (∂T/∂t + u·∂T/∂xi)
    From the equation:	ρ * Cp * (∂T/∂t + u·∂T/∂xi) = ∂·(λ.∂T/∂xi)/∂xi + ρ Σ_k=1^N_sp (cpk * Yk * Vk · ∂T/∂xi) + ω˙T
    Modified :			[∂T/∂t + u·∂T/∂xi]	= [∂·(D.∂T/∂xi)/∂xi/(ρ * Cp) + ω˙T/(ρ * Cp)] + [Σ_k=1^N_sp (cpk * Yk * Vk · ∂T/∂xi)/(Cp)]

1.1:	swap(previous_temperature, temperature) = ∂T/∂t

1.2:	dT_x,y,z = WENO3NONCONS(ui,P_T(X-2),P_T(X-1),P_T(X),P_T(X+1),P_T(X+2)) for bulk points
        dT_x,y,z = CENTRALNONCONS(ui,P_T(X-1),P_T(X),P_T(X+1)) for boundary points

1.3: T = P_T - (dT_x + dT_y + dT_z)	=>	∂T/∂t = -u·∂T/∂xi
The advection term represents the transport of temperature by fluid Flow. It accounts for the movement of temperature with the fluid velocity.
------------------------------------------------------------------------------------------------------------
*/
void Thermal_solver::FD_Euler(int time, Flow_solver* Flow, Species_solver* Species, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		swap(previous_temperature, temperature);  // Swap pointers for time-stepping
		// Loop over the computational domain
		for (int X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (int Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (int Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					// Check if the point is not in a solid region
					if (solid_thermal_type[{X, Y, Z}] == -1) {
						double dT_x, dT_y, dT_z;                     // Variables for temperature gradients in x, y, and z directions
						double T = previous_temperature[{X, Y, Z}];  // Temperature at the current point
						double T_xp = previous_temperature[{X + 1, Y, Z}];
						double T_xpp = previous_temperature[{X + 2, Y, Z}];
						double T_xn = previous_temperature[{X - 1, Y, Z}];
						double T_xnn = previous_temperature[{X - 2, Y, Z}];
						double T_yp = previous_temperature[{X, Y + 1, Z}];
						double T_ypp = previous_temperature[{X, Y + 2, Z}];
						double T_yn = previous_temperature[{X, Y - 1, Z}];
						double T_ynn = previous_temperature[{X, Y - 2, Z}];
						double T_zp = previous_temperature[{X, Y, Z + 1}];
						double T_zpp = previous_temperature[{X, Y, Z + 2}];
						double T_zn = previous_temperature[{X, Y, Z - 1}];
						double T_znn = previous_temperature[{X, Y, Z - 2}];

						double vel_X = Flow->velocity[{X, Y, Z, 0}];
						double vel_Y = Flow->velocity[{X, Y, Z, 1}];
						double vel_Z = Flow->velocity[{X, Y, Z, 2}];
// Advection term calculation using finite-difference schemes
#if defined FD_UPWIND
						dT_x = FD::UPWIND1NONCONS(vel_X, T_xn, T, T_xp);
						dT_y = FD::UPWIND1NONCONS(vel_Y, T_yn, T, T_yp);
						dT_z = FD::UPWIND1NONCONS(vel_Z, T_zn, T, T_zp);
#endif  // defined
#if defined FD_UPWIND2
						dT_x = FD::UPWIND2NONCONS(vel_X, T_xnn, T_xn, T, T_xp, T_xpp);
						dT_y = FD::UPWIND2NONCONS(vel_Y, T_ynn, T_yn, T, T_yp, T_ypp);
						dT_z = FD::UPWIND2NONCONS(vel_Z, T_znn, T_zn, T, T_zp, T_zpp);
#endif  // defined
#if defined FD_CENTRAL
						dT_x = FD::CENTRALNONCONS(vel_X, T_xn, T_xp);
						dT_y = FD::CENTRALNONCONS(vel_Y, T_yn, T_yp);
						dT_z = FD::CENTRALNONCONS(vel_Z, T_zn, T_zp);
#endif  // defined
#if defined FD_CENTRAL4
						dT_x = FD::CENTRAL4NONCONS(vel_X, T_xnn, T_xn, T, T_xp, T_xpp);
						dT_y = FD::CENTRAL4NONCONS(vel_Y, T_ynn, T_yn, T, T_yp, T_ypp);
						dT_z = FD::CENTRAL4NONCONS(vel_Z, T_znn, T_zn, T, T_zp, T_zpp);
#endif  // defined
#if defined FD_WENO3
						dT_x = FD::WENO3NONCONS(vel_X, T_xnn, T_xn, T, T_xp, T_xpp);
						dT_y = FD::WENO3NONCONS(vel_Y, T_ynn, T_yn, T, T_yp, T_ypp);
						dT_z = FD::WENO3NONCONS(vel_Z, T_znn, T_zn, T, T_zp, T_zpp);
#endif  // defined
#if defined FD_WENO5
						double T_xnnn = previous_temperature[{X - 3, Y, Z}];
						double T_xppp = previous_temperature[{X + 3, Y, Z}];
						double T_ynnn = previous_temperature[{X, Y - 3, Z}];
						double T_yppp = previous_temperature[{X, Y + 3, Z}];
						double T_znnn = previous_temperature[{X, Y, Z - 3}];
						double T_zppp = previous_temperature[{X, Y, Z + 3}];
						dT_x = FD::WENO5NONCONS(vel_X, T_xnnn, T_xnn, T_xn, T, T_xp, T_xpp, T_xppp);
						dT_y = FD::WENO5NONCONS(vel_Y, T_ynnn, T_ynn, T_yn, T, T_yp, T_ypp, T_yppp);
						dT_z = FD::WENO5NONCONS(vel_Z, T_znnn, T_znn, T_zn, T, T_zp, T_zpp, T_zppp);
						if (solid_thermal_type[{X + 2, Y, Z}] != -1 || solid_thermal_type[{X - 2, Y, Z}] != -1) dT_x = FD::CENTRALNONCONS(vel_X, T_xn, T_xp);
						if (solid_thermal_type[{X, Y + 2, Z}] != -1 || solid_thermal_type[{X, Y - 2, Z}] != -1) dT_y = FD::CENTRALNONCONS(vel_Y, T_yn, T_yp);
						if (solid_thermal_type[{X, Y, Z + 2}] != -1 || solid_thermal_type[{, Y, Z - 2}] != -1) dT_z = FD::CENTRALNONCONS(vel_Z, T_zn, T_zp);
#endif
						// The code includes checks for solid boundaries (solid_thermal_type values equal to -1) to handle the advection term near solid objects. Central differencing is used when neighboring cells are solid.
						if (solid_thermal_type[{X + 1, Y, Z}] != -1 || solid_thermal_type[{X - 1, Y, Z}] != -1) dT_x = FD::CENTRALNONCONS(vel_X, T_xn, T_xp);
						if (solid_thermal_type[{X, Y + 1, Z}] != -1 || solid_thermal_type[{X, Y + 1, Z}] != -1) dT_y = FD::CENTRALNONCONS(vel_Y, T_yn, T_yp);
						if (solid_thermal_type[{X, Y, Z + 1}] != -1 || solid_thermal_type[{X, Y, Z + 1}] != -1) dT_z = FD::CENTRALNONCONS(vel_Z, T_zn, T_zp);
						if (Dimension < 2) dT_y = 0;
						if (Dimension < 3) dT_z = 0;
						//// if(solid_particle[{X, Y, Z}]!=1.0){
						// T = (∂T/∂t - u·∂T/∂xi)
						temperature[{X, Y, Z}] = previous_temperature[{X, Y, Z}] - (dT_x + dT_y + dT_z);
						//	}
					}
				}
			}
		}
	}
	return;
}
/// ***************************************************** ///
/// ADD TEMPERATURE FIELD DIFFUSION TO FD SOLVER          ///
/// ***************************************************** ///
/* 2: "FD_Euler_diffusion" solves (∂·(D.∂T/∂xi)/∂xi + ω˙T)/ (ρ*Cp)
                                  [(∂.D.∂T/∂xi)/∂xi + ω˙T]/(ρ*Cp)

    From the equation:	ρ * Cp * (∂T/∂t + u·∂T/∂xi) = ∂·(λ.∂T/∂xi)/∂xi + ρ Σ_k=1^N_sp (cpk * Yk * Vk · ∂T/∂xi) + ω˙T
    Modified :			[∂T/∂t + u·∂T/∂xi]	= [[∂·(D.∂T/∂xi)/∂xi + ω˙T]/(ρ * Cp)] + [Σ_k=1^N_sp (cpk * Yk * Vk · ∂T/∂xi)/(Cp)]

    It solves: ([∂·(D.∂T/∂xi)/∂xi]/cp*ρ) by the term [grad2_T/cp*ρ] : is the heat flux per unit volume due to molecular diffusion, and it represents the spreading or dissipation of heat due to molecular diffusion.
    and (ω˙T/(ρ*Cp)) by the term [Production/ T_0 / (c_p* Flow->density * Flow->rho_0)] :  represents the heat sources or sinks due to chemical reactions, and it is scaled by the density, specific heat capacity, and temperature to account for the temperature dependence of the reaction rates.

    which makes the equation: (FD_Euler) = (FD_Euler_diffusion) + (Σ_k=1^N_sp (cpk * Yk * Vk · ∇T)/Cp)

2.1: ∂.Dk.(∂T/∂xi)/∂xi => ∂^2(Dk.T)/∂xi^2 => grad2_T
    for bulk points
    => central4flux(P_T(X-1),P_T(X),P_T(X+1),P_T(X+2),D(X-1),D(X),D(X+1),D(X+2)) - central4flux(P_T(X-2),P_T(X-1),P_T(X),P_T(X+1),D(X-2),D(X-1),D(X),D(X+1))
    => central4flux => f = D1 * (f1 / 8. - f2 / 6. + f3 / 24.) + D2 * (-f1 / 6. - 3. * f2 / 8. + 2. * f3 / 3. - f4 / 8.) + D3 * (f1 / 8. - 2. * f2 / 3. + 3. * f3 / 8. + f4 / 6.) + D4 * (-1. * f2 / 24. + f3 / 6. - f4 / 8.);
    => central4flux => {T= D(X-1) * (P_T(X-1) / 8. - P_T(X) / 6. + P_T(X+1) / 24.)
                        + D(X) * (-P_T(X-1) / 6. - 3. * P_T(X) / 8. + 2. * P_T(X+1) / 3. - P_T(X+2) / 8.)
                        + D(X+1) * (P_T(X-1) / 8. - 2. * P_T(X) / 3. + 3. * P_T(X+1) / 8. + P_T(X+2) / 6.)
                        + D(X+2) * (-1. * P_T(X) / 24. + P_T(X+1) / 6. - P_T(X+2) / 8.)
                        - D(X-2) * (P_T(X-2) / 8. - P_T(X-1) / 6. + P_T(X) / 24.)
                        - D(X-1) * (-P_T(X-2) / 6. - 3. * P_T(X-1) / 8. + 2. * P_T(X) / 3. - P_T(X+1) / 8.)
                        - D(X) * (P_T(X-2) / 8. - 2. * P_T(X-1) / 3. + 3. * P_T(X) / 8. + P_T(X+1) / 6.)
                        - D(X+1) * (-1. * P_T(X-1) / 24. + P_T(X) / 6. - P_T(X+1) / 8.}
    for boundary points
    => central2flux(P_T(X),P_T(X+1),D(X),D(X+1) - central2flux(P_T(X-1),P_T(X),D(X-1),D(X)
    => central2flux => f = 0.5 * (D2 + D1) * (f2 - f1);
    => central2flux => T = 0.5 * (D_{X+1} + D_{X}) * (T_{X+1} - T_{X}) - 0.5 * (D_{X} + D_{X-1}) * (T_{X} - T_{X-1})

2.2: T = (Force + grad2_T + (Production / T0)) / (cp * ρ * ρ0)
    where, force_thermal is the external Thermal force which is ZERO in the current set up, can be adjusted to get from Input file,
    grad2_T is the diffusive term, and Production is the production term.

The diffusion term represents the spreading or dissipation of heat due to molecular diffusion (Thermal conductivity).
The term is given by λ∇²T, where λ is the Thermal conductivity and ∇²T is the Laplacian of temperature.
The Laplacian of temperature is the divergence of the gradient of temperature.
*/
void Thermal_solver::FD_Euler_diffusion(int time, Flow_solver* Flow, Species_solver* Species, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		int X, Y, Z;
		double grad2_T;
		double Hp, Hn;
		for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					// If it is a solid region, the diffusion process is not applied.
					if (solid_thermal_type[{X, Y, Z}] == -1) {
						double T = previous_temperature[{X, Y, Z}];
						double T_xp = previous_temperature[{X + 1, Y, Z}];
						double T_xpp = previous_temperature[{X + 2, Y, Z}];
						double T_xn = previous_temperature[{X - 1, Y, Z}];
						double T_xnn = previous_temperature[{X - 2, Y, Z}];
						double T_yp = previous_temperature[{X, Y + 1, Z}];
						double T_ypp = previous_temperature[{X, Y + 2, Z}];
						double T_yn = previous_temperature[{X, Y - 1, Z}];
						double T_ynn = previous_temperature[{X, Y - 2, Z}];
						double T_zp = previous_temperature[{X, Y, Z + 1}];
						double T_zpp = previous_temperature[{X, Y, Z + 2}];
						double T_zn = previous_temperature[{X, Y, Z - 1}];
						double T_znn = previous_temperature[{X, Y, Z - 2}];

						double D = thermal_diffusion_coefficient[{X, Y, Z}];
						double D_xp = thermal_diffusion_coefficient[{X + 1, Y, Z}];
						double D_xpp = thermal_diffusion_coefficient[{X + 2, Y, Z}];
						double D_xn = thermal_diffusion_coefficient[{X - 1, Y, Z}];
						double D_xnn = thermal_diffusion_coefficient[{X - 2, Y, Z}];
						double D_yp = thermal_diffusion_coefficient[{X, Y + 1, Z}];
						double D_ypp = thermal_diffusion_coefficient[{X, Y + 2, Z}];
						double D_yn = thermal_diffusion_coefficient[{X, Y - 1, Z}];
						double D_ynn = thermal_diffusion_coefficient[{X, Y - 2, Z}];
						double D_zp = thermal_diffusion_coefficient[{X, Y, Z + 1}];
						double D_zpp = thermal_diffusion_coefficient[{X, Y, Z + 2}];
						double D_zn = thermal_diffusion_coefficient[{X, Y, Z - 1}];
						double D_znn = thermal_diffusion_coefficient[{X, Y, Z - 2}];
#if defined FD_UPWIND || defined FD_UPWIND2 || defined FD_CENTRAL
						grad2_T = FD::CENTRAL2FLUX(T, T_xp, D, D_xp) - FD::CENTRAL2FLUX(T_xn, T, D_xn, D);
						if (Dimension > 1) { grad2_T += (FD::CENTRAL2FLUX(T, T_yp, D, D_yp) - FD::CENTRAL2FLUX(T_yn, T, D_yn, D)); }
						if (Dimension > 2) { grad2_T += (FD::CENTRAL2FLUX(T, T_zp, D, D_zp) - FD::CENTRAL2FLUX(T_zn, T, D_zn, D)); }
#endif
#if defined FD_CENTRAL4 || defined FD_WENO3 || defined FD_WENO5
						// if both neighboring grid points in the positive and negative x-directions are not solid.
						if (solid_thermal_type[{X + 1, Y, Z}] == -1 && solid_thermal_type[{X - 1, Y, Z}] == -1) {
							Hp = FD::CENTRAL4FLUX(T_xn, T, T_xp, T_xpp, D_xn, D, D_xp, D_xpp);
							Hn = FD::CENTRAL4FLUX(T_xnn, T_xn, T, T_xp, D_xnn, D_xn, D, D_xp);
							grad2_T = (Hp - Hn);
						}
						// if at least one of the neighboring grid points is solid: a central second-order flux function is used to compute the flux contribution, and it is added to the grad2_T.
						if (solid_thermal_type[{X + 1, Y, Z}] != -1 || solid_thermal_type[{X - 1, Y, Z}] != -1) {
							grad2_T = FD::CENTRAL2FLUX(T, T_xp, D, D_xp) - FD::CENTRAL2FLUX(T_xn, T, D_xn, D);
						}
						///  y-direction
						if (Dimension > 1) {
							if (solid_thermal_type[{X, Y + 1, Z}] == -1 && solid_thermal_type[{X, Y - 1, Z}] == -1) {
								Hp = FD::CENTRAL4FLUX(T_yn, T, T_yp, T_ypp, D_yn, D, D_yp, D_ypp);
								Hn = FD::CENTRAL4FLUX(T_ynn, T_yn, T, T_yp, D_ynn, D_yn, D, D_yp);
								grad2_T += (Hp - Hn);
							}
							if (solid_thermal_type[{X, Y + 1, Z}] != -1 || solid_thermal_type[{X, Y - 1, Z}] != -1) {
								grad2_T = FD::CENTRAL2FLUX(T, T_yp, D, D_yp) - FD::CENTRAL2FLUX(T_yn, T, D_yn, D);
							}
						}
						///  z-direction
						if (Dimension > 2) {
							if (solid_thermal_type[{X, Y, Z + 1}] == -1 && solid_thermal_type[{X, Y, Z - 1}] == -1) {
								Hp = FD::CENTRAL4FLUX(T_zn, T, T_zp, T_zpp, D_zn, D, D_zp, D_zpp);
								Hn = FD::CENTRAL4FLUX(T_znn, T_zn, T, T_zp, D_znn, D_zn, D, D_zp);
								grad2_T += (Hp - Hn);
							}
							if (solid_thermal_type[{X, Y, Z + 1}] != -1 || solid_thermal_type[{X, Y, Z - 1}] != -1) {
								grad2_T = FD::CENTRAL2FLUX(T, T_zp, D, D_zp) - FD::CENTRAL2FLUX(T_zn, T, D_zn, D);
							}
						}
#endif
						// explicit time-stepping scheme for updating the temperature field based on the convection-diffusion equation.
						double Force_Term = force_thermal[{X, Y, Z}] / (c_p[{X, Y, Z}] * Flow->density[{X, Y, Z}] * Flow->rho_0);
						double Production_Term = (Production[{X, Y, Z}] / T_0) / (c_p[{X, Y, Z}] * Flow->density[{X, Y, Z}] * Flow->rho_0);
						double gradient_Term = grad2_T / (c_p[{X, Y, Z}] * Flow->density[{X, Y, Z}] * Flow->rho_0);

						temperature[{X, Y, Z}] += Force_Term + gradient_Term + Production_Term;
						// std::cout << "thermal_diffusion_coefficient: " << thermal_diffusion_coefficient[{X, Y, Z}] << " Production_Term: " << Production_Term << " gradient_Term: " << gradient_Term << std::endl;
#if defined LMNA_solver
						Flow->divU[{X, Y, Z}] += (gradient_Term + Production_Term) / T;
#endif
					}
				}
			}
		}
	}
	return;
}
/*
FOR LMNA ABOVE.
The Low Mach Number Approximation (LMNA) is an assumption applied when the Flow velocities are much smaller
than the speed of sound. In such cases, the compressibility effects related to changes in density are negligible,
and the governing equations can be simplified.
{Flow->divU}[X, Y, Z] += {{{grad2_T} + {{Production}[X, Y, Z]}/{{T_0}}}}/{{previous_temperature}[X, Y, Z] dot {Flow->density}[X, Y, Z] dot {c_p}[X, Y, Z] dot {Flow->rho_0}}}
In the LMNA, the assumption is that the density variations are small, and therefore, changes in density can be neglected.
This often leads to the elimination of terms related to density variations from the governing equations.
In the expression above, the density appears in the denominator, this density is often treated as a constant. Additionally,
the velocity field is assumed to be divergence-free (∇⋅T=0) in incompressible flows, and the term ∇⋅u is related to the changes in density and temperature.
In the LMNA, this divergence term is often simplified.
*/
/// ***************************************************** ///
/// ADD ENTHALPY TRANSPORT BY SPECIES DIFFUSION FD SOLVER ///
/// ***************************************************** ///
/* 3: "FD_Euler_species_diffusion_enthalpy" solves ρ Σ_k=1^N_sp (cpk * Yk * Vk · ∂T/∂xi)
                                                => Σ_k=1^N_sp (cpk * Yk * Vk · ∇T)/Cp

    From the equation:	ρ * Cp * (∂T/∂t + u·∂T/∂xi) = ∂·(λ.∂T/∂xi)/∂xi + ρ Σ_k=1^N_sp (cpk * Yk * Vk · ∂T/∂xi) + ω˙T
    Modified :			[∂T/∂t + u·∂T/∂xi]	= [∂·(λ/(ρ * Cp).∂T/∂xi)/∂xi + ω˙T/(ρ * Cp)] + [Σ_k=1^N_sp (cpk * Yk * Vk · ∂T/∂xi)/(Cp)]

    The enthalpy transport due to Species diffusion is calculated using the following steps:
    1. The temperature gradient is calculated using the central difference scheme.
    2. The velocity of Species k is calculated using the central difference scheme.
    3. The enthalpy transport due to Species diffusion is calculated using the central difference scheme.
    4. The enthalpy transport due to Species diffusion is added to the temperature field.

    The dot product signifies the directional derivative of temperature in the direction of the molar production rate vector for Species k.
    This term captures the heat transfer associated with the chemical reactions involving Species k in the reacting Flow.
    In summary, this term accounts for how the transport of different Species affects the temperature field in the reacting Flow by considering the specific heat and molar production rate of each Species.

3.1: dT[0] = FD::CENTRALNONCONS(1., previous_temperature[{X - 1, Y, Z}], previous_temperature[{X + 1, Y, Z}]);
    where Centeralnoncons: f = 0.5 * u * (f2 - f1);
    so	dT[0] = 0.5 * u * (T_{X-1,Y,Z} - T_{X+1,Y,Z})
        dT[1] = 0.5 * v * (T_{X,Y-1,Z} - T_{X,Y+1,Z})
        dT[2] = 0.5 * w * (T_{X,Y,Z-1} - T_{X,Y,Z+1})

3.2: Vk = (-Dk * ∇Xk)/Xk
    In code: Vk[0] = (Species->thermal_diffusion_coefficient[{X, Y, Z, k}] / Species->molar_mass_av[{X, Y, Z}])
                * FD::CENTRALNONCONS(1., Species->molar_mass_av[{X - 1, Y, Z}] * Species->previous_mass_fraction[{X - 1, Y, Z, k}],
                    Species->molar_mass_av[{X + 1, Y, Z}] * Species->previous_mass_fraction[{X + 1, Y, Z, k}]);
    that means:	Vk[0] = (D_k / X_k) * 0.5 * (X_{X-1,Y,Z} - X_{X+1,Y,Z})
                Vk = - Dk * ∇Xk / Wk
                Where,
                Dk is the diffusion coefficient of Species k,
                Xk is the molar mass of Species k,
                ∇Xk is the difference in molar mass of Species k between the two adjacent cells.

3.3: flux -= Species->cp_k[k] * (Vk[0] * dT[0] + Vk[1] * dT[1] + Vk[2] * dT[2]) , the minus comes from Vk.
    Where,
        flux -= cpk * Yk * Vk · ∇T
    Where,
        cpk is the specific heat capacity of Species k,
        Yk is the mass fraction of Species k,
        Vk is the velocity of Species k,
        ∇T is the gradient of temperature.

3.4: temperature[{X, Y, Z}] -= (flux / c_p[{X, Y, Z}]);
    Where,
        temperature[{X, Y, Z}] -= Σ_k=1^N_sp (cpk * Yk * Vk · ∇T)/Cp
*/
void Thermal_solver::FD_Euler_species_diffusion_enthalpy(int time, Flow_solver* Flow, Species_solver* Species, Parallel_MPI* MPI_parallel) {
#if defined Flow_With_Species
	if (MPI_parallel->processor_id != MASTER) {
		unsigned int X = 0, Y = 0, Z = 0, k;
		double dT_x, dT_y, dT_z;
		double flux;
#if defined REGATH_LIB
		int np = 1;
		int Nb_spec = Species->Nb_spec;
#endif
		for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (solid_thermal_type[{X, Y, Z}] == -1) {
						double T = previous_temperature[{X, Y, Z}];
						double T_xp = previous_temperature[{X + 1, Y, Z}];
						double T_xpp = previous_temperature[{X + 2, Y, Z}];
						double T_xn = previous_temperature[{X - 1, Y, Z}];
						double T_xnn = previous_temperature[{X - 2, Y, Z}];
						double T_yp = previous_temperature[{X, Y + 1, Z}];
						double T_ypp = previous_temperature[{X, Y + 2, Z}];
						double T_yn = previous_temperature[{X, Y - 1, Z}];
						double T_ynn = previous_temperature[{X, Y - 2, Z}];
						double T_zp = previous_temperature[{X, Y, Z + 1}];
						double T_zpp = previous_temperature[{X, Y, Z + 2}];
						double T_zn = previous_temperature[{X, Y, Z - 1}];
						double T_znn = previous_temperature[{X, Y, Z - 2}];

						// ∂T/∂xi           the temperature field is updated based on the enthalpy transport due to Species diffusion.
#if defined FD_UPWIND || defined FD_UPWIND2 || defined FD_CENTRAL
						dT_x = FD::CENTRALNONCONS(1., T_xn, T_xp);
						dT_y = FD::CENTRALNONCONS(1., T_yn, T_yp);
						dT_z = FD::CENTRALNONCONS(1., T_zn, T_zp);
#endif  // defined
#if defined FD_CENTRAL4 || defined FD_WENO3
						dT_x = FD::CENTRAL4NONCONS(1., T_xnn, T_xn, T, T_xp, T_xpp);
						dT_y = FD::CENTRAL4NONCONS(1., T_ynn, T_yn, T, T_yp, T_ypp);
						dT_z = FD::CENTRAL4NONCONS(1., T_znn, T_zn, T, T_zp, T_zpp);
#endif  // defined
						if (solid_thermal_type[{X + 1, Y, Z}] != -1 || solid_thermal_type[{X - 1, Y, Z}] != -1) dT_x = FD::CENTRALNONCONS(1., T_xn, T_xp);
						if (solid_thermal_type[{X, Y + 1, Z}] != -1 || solid_thermal_type[{X, Y + 1, Z}] != -1) dT_y = FD::CENTRALNONCONS(1., T_yn, T_yp);
						if (solid_thermal_type[{X, Y, Z + 1}] != -1 || solid_thermal_type[{X, Y, Z + 1}] != -1) dT_z = FD::CENTRALNONCONS(1., T_zn, T_zp);
						if (Dimension < 2) dT_y = 0;
						if (Dimension < 3) dT_z = 0;
#if defined REGATH_LIB
						__mod_regath_interface_MOD_regath_cp_spec_mass_cpp(&Nb_spec, &np, &(previous_temperature[{X, Y, Z}]), Species->cp_k);
#endif  // defined
        // The enthalpy flux due to Species diffusion is calculated for each cell, considering the mass fraction and diffusion coefficients of individual Species.
        // Σ_k=1^N_sp (cpk * Yk * (Vk · ∂T/∂xi)) / cp

						// double* cp_k_calculated = new double[Species->Nb_spec];
						// thermo_chemistry->compute_cp_k(&Species->Nb_spec, &T, &(Flow->p_th_0), &Species->previous_mass_fraction[{X, Y, Z}], Species->cp_k);

						double Vk_x, Vk_y, Vk_z;
						flux = 0;
						for (k = 0; k < Species->Nb_spec; k++) {
							double Dk_sp = Species->diffusion_coefficient[{X, Y, Z, k}];
							// double Yk = Species->previous_mass_fraction[{X, Y, Z, k}];
							double Yk_xp = Species->previous_mass_fraction[{X + 1, Y, Z, k}];
							double Yk_xn = Species->previous_mass_fraction[{X - 1, Y, Z, k}];
							double Yk_yp = Species->previous_mass_fraction[{X, Y + 1, Z, k}];
							double Yk_yn = Species->previous_mass_fraction[{X, Y - 1, Z, k}];
							double Yk_zp = Species->previous_mass_fraction[{X, Y, Z + 1, k}];
							double Y_zn = Species->previous_mass_fraction[{X, Y, Z - 1, k}];

							double W = Species->molar_mass_av[{X, Y, Z}];
							double W_xp = Species->molar_mass_av[{X + 1, Y, Z}];
							double W_xn = Species->molar_mass_av[{X - 1, Y, Z}];
							double W_yp = Species->molar_mass_av[{X, Y + 1, Z}];
							double W_yn = Species->molar_mass_av[{X, Y - 1, Z}];
							double W_zp = Species->molar_mass_av[{X, Y, Z + 1}];
							double W_zn = Species->molar_mass_av[{X, Y, Z - 1}];

							Vk_x = (Dk_sp / W) * FD::CENTRALNONCONS(1., W_xn * Yk_xn, W_xp * Yk_xp);
							Vk_y = (Dk_sp / W) * FD::CENTRALNONCONS(1., W_yn * Yk_yn, W_yp * Yk_yp);
							Vk_z = (Dk_sp / W) * FD::CENTRALNONCONS(1., W_zn * Y_zn, W_zp * Yk_zp);
							// The enthalpy flux due to Species diffusion is calculated for each cell, considering the mass fraction and diffusion coefficients of individual Species.
							// Σ_k=1^N_sp (cpk * Yk * (Vk · ∂T/∂xi)) / cp
							flux -= Species->cp_k[k] * (Vk_x * dT_x + Vk_y * dT_y + Vk_z * dT_z);
						}
						temperature[{X, Y, Z}] -= (flux / c_p[{X, Y, Z}]);
#if defined LMNA_solver
						Flow->divU[{X, Y, Z}] -= (flux / (T * c_p[{X, Y, Z}]));
#endif
					}
				}
			}
		}
	}
	return;
#endif  // defined
}
/// ***************************************************** ///
/// ADD PRESSURE * DIV U TERM FOR FD (COMPRESSIBLE)       ///
/// ***************************************************** ///
void Thermal_solver::FD_Euler_pressure(int time, Flow_solver* Flow, Species_solver* Species, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		int X, Y, Z;
		double du_x, du_y, du_z;

		for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (solid_thermal_type[{X, Y, Z}] == -1) {
#if defined(FD_CENTRAL) || defined(FD_UPWIND) || defined(FD_UPWIND2) || defined(FD_WENO3)
						du_x = FD::CENTRALNONCONS(1., Flow->velocity[{X - 1, Y, Z, 0}], Flow->velocity[{X + 1, Y, Z, 0}]);
						du_y = FD::CENTRALNONCONS(1., Flow->velocity[{X, Y - 1, Z, 1}], Flow->velocity[{X, Y + 1, Z, 1}]);
						du_z = FD::CENTRALNONCONS(1., Flow->velocity[{X, Y, Z - 1, 2}], Flow->velocity[{X, Y, Z + 1, 2}]);
#endif  // defined
#if defined FD_CENTRAL4
						du_x = FD::CENTRAL4NONCONS(1., Flow->velocity[{X - 2, Y, Z, 0}], Flow->velocity[{X - 1, Y, Z, 0}], Flow->velocity[{X, Y, Z, 0}],
						                           Flow->velocity[{X + 1, Y, Z, 0}], Flow->velocity[{X + 2, Y, Z, 0}]);
						du_y = FD::CENTRAL4NONCONS(1., Flow->velocity[{X, Y - 2, Z, 1}], Flow->velocity[{X, Y - 1, Z, 1}], Flow->velocity[{X, Y, Z, 1}],
						                           Flow->velocity[{X, Y + 1, Z, 1}], Flow->velocity[{X, Y + 2, Z, 1}]);
						du_z = FD::CENTRAL4NONCONS(1., Flow->velocity[{X, Y, Z - 2, 2}], Flow->velocity[{X, Y, Z - 1, 2}], Flow->velocity[{X, Y, Z, 2}],
						                           Flow->velocity[{X, Y, Z + 1, 2}], Flow->velocity[{X, Y, Z + 2, 2}]);
						if (solid_thermal_type[{X + 1, Y, Z}] != -1)
							du_x = FD::CENTRAL4NONCONS(1., Flow->velocity[{X - 2, Y, Z, 0}], Flow->velocity[{X - 1, Y, Z, 0}], Flow->velocity[{X, Y, Z, 0}],
							                           Flow->velocity[{X + 1, Y, Z, 0}], Flow->velocity[{X + 1, Y, Z, 0}]);
						if (solid_thermal_type[{X - 1, Y, Z}] != -1)
							du_x = FD::CENTRAL4NONCONS(1., Flow->velocity[{X - 1, Y, Z, 0}], Flow->velocity[{X - 1, Y, Z, 0}], Flow->velocity[{X, Y, Z, 0}],
							                           Flow->velocity[{X + 1, Y, Z, 0}], Flow->velocity[{X + 2, Y, Z, 0}]);

						if (solid_thermal_type[{X, Y + 1, Z}] != -1)
							du_y = FD::CENTRAL4NONCONS(1., Flow->velocity[{X, Y - 2, Z, 1}], Flow->velocity[{X, Y - 1, Z, 1}], Flow->velocity[{X, Y, Z, 1}],
							                           Flow->velocity[{X, Y + 1, Z, 1}], Flow->velocity[{X, Y + 1, Z, 1}]);
						if (solid_thermal_type[{X, Y - 1, Z}] != -1)
							du_x = FD::CENTRAL4NONCONS(1., Flow->velocity[{X, Y - 1, Z, 1}], Flow->velocity[{X, Y - 1, Z, 1}], Flow->velocity[{X, Y, Z, 1}],
							                           Flow->velocity[{X, Y + 1, Z, 1}], Flow->velocity[{X, Y + 2, Z, 1}]);

						if (solid_thermal_type[{X, Y, Z + 1}] != -1)
							du_y = FD::CENTRAL4NONCONS(1., Flow->velocity[{X, Y, Z - 2, 2}], Flow->velocity[{X, Y, Z - 1, 2}], Flow->velocity[{X, Y, Z, 2}],
							                           Flow->velocity[{X, Y, Z + 1, 2}], Flow->velocity[{X, Y, Z + 1, 2}]);
						if (solid_thermal_type[{X, Y, Z - 1}] != -1)
							du_x = FD::CENTRAL4NONCONS(1., Flow->velocity[{X, Y, Z - 1, 2}], Flow->velocity[{X, Y, Z - 1, 2}], Flow->velocity[{X, Y, Z, 2}],
							                           Flow->velocity[{X, Y, Z + 1, 2}], Flow->velocity[{X, Y, Z + 2, 2}]);
#endif  // defined

#if defined Flow_With_Species
							// const double r = R_GAS / Species->molar_mass_av[{X, Y, Z}];
#else
						const double r = R_GAS / Flow->M_av;
#endif  // defined
						temperature[{X, Y, Z}] -= (1.4 - 1) * ((previous_temperature[{X, Y, Z}]) * (du_x + du_y + du_z));
					}
				}
			}
		}
	}
	return;
}
void Thermal_solver::FD_ugradT(int time, Flow_solver* Flow, Species_solver* Species, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		int X, Y, Z;
		double dP_x, dP_y, dP_z;

		for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (solid_thermal_type[{X, Y, Z}] == -1) {
#if defined FD_UPWIND
						dP_x = FD::UPWIND1NONCONS(Flow->velocity[{X, Y, Z, 0}], Flow->pressure[{X - 1, Y, Z}], Flow->pressure[{X, Y, Z}], Flow->pressure[{X + 1, Y, Z}]);
						dP_y = FD::UPWIND1NONCONS(Flow->velocity[{X, Y, Z, 1}], Flow->pressure[{X, Y - 1, Z}], Flow->pressure[{X, Y, Z}], Flow->pressure[{X, Y + 1, Z}]);
						dP_z = FD::UPWIND1NONCONS(Flow->velocity[{X, Y, Z, 2}], Flow->pressure[{X, Y, Z - 1}], Flow->pressure[{X, Y, Z}], Flow->pressure[{X, Y, Z + 1}]);
#endif  // defined
#if defined FD_UPWIND2
						dP_x = FD::UPWIND2NONCONS(Flow->velocity[{X, Y, Z, 0}], Flow->pressure[{X - 2, Y, Z}], Flow->pressure[{X - 1, Y, Z}], Flow->pressure[{X, Y, Z}], Flow->pressure[{X + 1, Y, Z}], Flow->pressure[{X + 2, Y, Z}]);
						dP_y = FD::UPWIND2NONCONS(Flow->velocity[{X, Y, Z, 1}], Flow->pressure[{X, Y - 2, Z}], Flow->pressure[{X, Y - 1, Z}], Flow->pressure[{X, Y, Z}], Flow->pressure[{X, Y + 1, Z}], Flow->pressure[{X, Y + 2, Z}]);
						dP_z = FD::UPWIND2NONCONS(Flow->velocity[{X, Y, Z, 2}], Flow->pressure[{X, Y, Z - 2}], Flow->pressure[{X, Y, Z - 1}], Flow->pressure[{X, Y, Z}], Flow->pressure[{X, Y, Z + 1}], Flow->pressure[{X, Y, Z + 2}]);
#endif  // defined
#if defined FD_CENTRAL
						dP_x = FD::CENTRALNONCONS(Flow->velocity[{X, Y, Z, 0}], Flow->pressure[{X - 1, Y, Z}], Flow->pressure[{X + 1, Y, Z}]);
						dP_y = FD::CENTRALNONCONS(Flow->velocity[{X, Y, Z, 1}], Flow->pressure[{X, Y - 1, Z}], Flow->pressure[{X, Y + 1, Z}]);
						dP_z = FD::CENTRALNONCONS(Flow->velocity[{X, Y, Z, 2}], Flow->pressure[{X, Y, Z - 1}], Flow->pressure[{X, Y, Z + 1}]);
#endif  // defined
#if defined FD_CENTRAL4
						dP_x = FD::CENTRAL4NONCONS(Flow->velocity[{X, Y, Z, 0}], Flow->pressure[{X - 2, Y, Z}], Flow->pressure[{X - 1, Y, Z}], Flow->pressure[{X, Y, Z}],
						                           previous_temperature[{X + 1, Y, Z}], previous_temperature[{X + 2, Y, Z}]);
						dP_y = FD::CENTRAL4NONCONS(Flow->velocity[{X, Y, Z, 1}], Flow->pressure[{X, Y - 2, Z}], Flow->pressure[{X, Y - 1, Z}], Flow->pressure[{X, Y, Z}],
						                           Flow->pressure[{X, Y + 1, Z}], Flow->pressure[{X, Y + 2, Z}]);
						dP_z = FD::CENTRAL4NONCONS(Flow->velocity[{X, Y, Z, 2}], Flow->pressure[{X, Y, Z - 2}], Flow->pressure[{X, Y, Z - 1}], Flow->pressure[{X, Y, Z}],
						                           Flow->pressure[{X, Y, Z + 1}], Flow->pressure[{X, Y, Z + 2}]);
#endif  // defined
#if defined FD_WENO3
						dP_x = FD::WENO3NONCONS(Flow->velocity[{X, Y, Z, 0}], Flow->pressure[{X - 2, Y, Z}], Flow->pressure[{X - 1, Y, Z}],
						                        Flow->pressure[{X, Y, Z}], Flow->pressure[{X + 1, Y, Z}], Flow->pressure[{X + 2, Y, Z}]);
						dP_y = FD::WENO3NONCONS(Flow->velocity[{X, Y, Z, 1}], Flow->pressure[{X, Y - 2, Z}], Flow->pressure[{X, Y - 1, Z}],
						                        Flow->pressure[{X, Y, Z}], Flow->pressure[{X, Y + 1, Z}], Flow->pressure[{X, Y + 2, Z}]);
						dP_z = FD::WENO3NONCONS(Flow->velocity[{X, Y, Z, 2}], Flow->pressure[{X, Y, Z - 2}], Flow->pressure[{X, Y, Z - 1}],
						                        Flow->pressure[{X, Y, Z}], Flow->pressure[{X, Y, Z + 1}], Flow->pressure[{X, Y, Z + 2}]);
#endif  // defined
						if (solid_thermal_type[{X + 1, Y, Z}] != -1 || solid_thermal_type[{X - 1, Y, Z}] != -1)
							dP_x = FD::CENTRALNONCONS(Flow->velocity[{X, Y, Z, 0}], Flow->pressure[{X - 1, Y, Z}], Flow->pressure[{X + 1, Y, Z}]);
						if (solid_thermal_type[{X, Y + 1, Z}] != -1 || solid_thermal_type[{X, Y - 1, Z}] != -1)
							dP_y = FD::CENTRALNONCONS(Flow->velocity[{X, Y, Z, 1}], Flow->pressure[{X, Y - 1, Z}], Flow->pressure[{X, Y + 1, Z}]);
						if (solid_thermal_type[{X, Y, Z + 1}] != -1 || solid_thermal_type[{X, Y, Z - 1}] != -1)
							dP_z = FD::CENTRALNONCONS(Flow->velocity[{X, Y, Z, 2}], Flow->pressure[{X, Y, Z - 1}], Flow->pressure[{X, Y, Z + 1}]);

						temperature[{X, Y, Z}] += ((sqr(global_parameters.D_x / global_parameters.D_t)) * (dP_x + dP_y + dP_z)) / (c_p[{X, Y, Z}] * T_0 * Flow->density[{X, Y, Z}]);
					}
				}
			}
		}
	}
	return;
}
/// ***************************************************** ///
/// APPLICATION OF BOUNDARY CONITIONS                     ///
/// ***************************************************** ///
void Thermal_solver::BC(int time, Flow_solver* Flow, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		int bou;
		int X, Y, Z, alpha, Xp, Yp, Zp;
		for (int k = 0; k < Boundaries.size(); ++k) {
			X = Boundaries[k].X;  // position of the boundary node inside the Flow
			Y = Boundaries[k].Y;
			Z = Boundaries[k].Z;
			bou = Boundaries[k].type;
			switch (bou) {
				/// 1: Zero temperature BC (adiabatic wall): ∂T/∂n = 0
				///  Temp neighbor = - Temp boundary, so that the temperature gradient is zero. It is similar to the zero-flux BC.
				case 1: {
					for (alpha = 1; alpha < Discrete_Velocity; alpha++) {
						Xp = (X - c_alpha[alpha][0]);
						Yp = (Y - c_alpha[alpha][1]);
						Zp = (Z - c_alpha[alpha][2]);
						if (solid_thermal_type[{Xp, Yp, Zp}] != -1) {                              // checks if the neighboring point is not part of a solid boundary (solid_thermal_type -1 indicates a solid).
							if (DOT(c_alpha[alpha], Boundaries[k].n) >= 0) {                       // This condition essentially checks if the velocity vector is pointing into the boundary.
								pop_t[{X, Y, Z, alpha}] = -pop_t[{Xp, Yp, Zp, alpha_bar[alpha]}];  // Zero temperature
								                                                                   // it updates the Thermal distribution function pop_t at the current lattice point ({X, Y, Z}) and velocity direction (alpha).
								                                                                   // It sets it to the negative of the Thermal distribution function at the opposite lattice point ({Xp, Yp, Zp}) and opposite direction (alpha_bar[alpha]).
							}
						}
					}
					break;
				}
				/// 2: Non-zero temperature BC: Temp_boundary = T
				///  Temp neighbor = - Temp boundary, so that the temperature gradient is zero. It is similar to the zero-temperature BC.
				case 2: {
					for (alpha = 1; alpha < Discrete_Velocity; alpha++) {
						Xp = (X - c_alpha[alpha][0]);
						Yp = (Y - c_alpha[alpha][1]);
						Zp = (Z - c_alpha[alpha][2]);
						if (solid_thermal_type[{Xp, Yp, Zp}] != -1) {
							if (DOT(c_alpha[alpha], Boundaries[k].n) >= 0) {
								pop_t[{X, Y, Z, alpha}] = (weight[alpha] + weight[alpha_bar[alpha]]) * (Boundaries[k].T)
								                          - pop_t[{Xp, Yp, Zp, alpha_bar[alpha]}];  /// Non-zero temperature on walls
							}
						}
					}
					break;
				}
				/// 3: Zero-gradient BC: ∂T/∂n = 0
				///  Boundaries[i].n = normal vector to the boundary. It is added to the current lattice point to get the neighboring point.
				///  Temp neighbor = Temp boundary, so that the temperature gradient is zero. It is similar to the zero-flux BC.
				case 3: {
					for (alpha = 1; alpha < Discrete_Velocity; alpha++) {
						Xp = (X - c_alpha[alpha][0]);
						Yp = (Y - c_alpha[alpha][1]);
						Zp = (Z - c_alpha[alpha][2]);
						if (solid_thermal_type[{Xp, Yp, Zp}] != -1) {
							if (DOT(c_alpha[alpha], Boundaries[k].n) > 0) {
								pop_t[{X, Y, Z, alpha}] = pop_t[{X + Boundaries[k].n[0], Y + Boundaries[k].n[1], Z + Boundaries[k].n[2], alpha}];
							}
						}
					}
					break;
				}
				/// 4: Zero-flux BC: ∂T/∂n = 0
				///  Temp neighbor = Temp boundary, so that the temperature gradient is zero. It is similar to the zero-gradient BC.
				case 4: {
					for (alpha = 1; alpha < Discrete_Velocity; alpha++) {
						Xp = (X - c_alpha[alpha][0]);
						Yp = (Y - c_alpha[alpha][1]);
						Zp = (Z - c_alpha[alpha][2]);
						if (solid_thermal_type[{Xp, Yp, Zp}] != -1) {
							if (DOT(c_alpha[alpha], Boundaries[k].n) >= 0) {
								pop_t[{X, Y, Z, alpha}] = pop_t[{Xp, Yp, Zp, alpha_bar[alpha]}];
							}
						}
					}
					break;
				}
				/// 102: Non-zero temperature BC: Temp_boundary = T
				///  Temp neighbor = 2 * Temp boundary - Temp neighbor, so that the temperature gradient is zero. It is similar to the zero-temperature BC.
				// X + B means Xp lies in the direction of the boundary normal. That means Xp is outwards from the boundary.
				case 102: {  /// ---> Non-zero temperature BC
					if (!curved_boundaries) {
						Xp = (X - Boundaries[k].n[0]);  // Xp , Yp, Zp :outside points
						Yp = (Y - Boundaries[k].n[1]);
						Zp = (Z - Boundaries[k].n[2]);
						if (Boundaries[k].n[0] != 0) {
							temperature[{Xp, Y, Z}] = 2. * Boundaries[k].T - temperature[{X, Y, Z}];
							previous_temperature[{Xp, Y, Z}] = 2. * Boundaries[k].T - temperature[{X, Y, Z}];
							temp_temperature[{Xp, Y, Z}] = 2. * Boundaries[k].T - temperature[{X, Y, Z}];
						}
						if (Boundaries[k].n[1] != 0) {
							temperature[{X, Yp, Z}] = 2. * Boundaries[k].T - temperature[{X, Y, Z}];
							previous_temperature[{X, Yp, Z}] = 2. * Boundaries[k].T - temperature[{X, Y, Z}];
							temp_temperature[{X, Yp, Z}] = 2. * Boundaries[k].T - temperature[{X, Y, Z}];
						}
						if (Boundaries[k].n[2] != 0) {
							temperature[{X, Y, Zp}] = 2. * Boundaries[k].T - temperature[{X, Y, Z}];
							previous_temperature[{X, Y, Zp}] = 2. * Boundaries[k].T - temperature[{X, Y, Z}];
							temp_temperature[{X, Y, Zp}] = 2. * Boundaries[k].T - temperature[{X, Y, Z}];
						}
					}
					if (curved_boundaries) {
						/* first get temperature at image point */
						double T_image = 0;
						double previous_T_image = 0;
						for (int i = 0; i < Boundaries[k].X_Image_Int.size(); i++) {
							Xp = Boundaries[k].X_Image_Int[i];
							Yp = Boundaries[k].Y_Image_Int[i];
							Zp = Boundaries[k].Z_Image_Int[i];
							T_image += Boundaries[k].W_Image_Int[i] * temperature[{Xp, Yp, Zp}];
							previous_T_image += Boundaries[k].W_Image_Int[i] * temperature[{Xp, Yp, Zp}];
						}
						if (Boundaries[k].n[0] != 0) {
							temperature[{X, Y, Z}] = 2.0 * Boundaries[k].T - T_image;
							previous_temperature[{X, Y, Z}] = 2.0 * Boundaries[k].T - previous_T_image;
						}
						if (Boundaries[k].n[1] != 0) {
							temperature[{X, Y, Z}] = 2.0 * Boundaries[k].T - T_image;
							previous_temperature[{X, Y, Z}] = 2.0 * Boundaries[k].T - previous_T_image;
						}
						if (Boundaries[k].n[2] != 0) {
							temperature[{X, Y, Z}] = 2.0 * Boundaries[k].T - T_image;
							previous_temperature[{X, Y, Z}] = 2.0 * Boundaries[k].T - previous_T_image;
						}
					}
					break;
				}
				/// 104: Zero-gradient (0th-order) - Neumann BC : T[i] = T[i-1]
				///  Temp neighbor = Temp boundary, so that the temperature gradient is zero. It is similar to the zero-flux BC.
				case 104: {  // --> Zero-gradient (1st-order) BC
					if (!curved_boundaries) {
						Xp = (X - Boundaries[k].n[0]);
						Yp = (Y - Boundaries[k].n[1]);
						Zp = (Z - Boundaries[k].n[2]);
						if (Boundaries[k].n[0] != 0) {
							temperature[{Xp, Y, Z}] = temperature[{X, Y, Z}];
						}
						if (Boundaries[k].n[1] != 0) {
							temperature[{X, Yp, Z}] = temperature[{X, Y, Z}];
						}
						if (Boundaries[k].n[2] != 0) {
							temperature[{X, Y, Zp}] = temperature[{X, Y, Z}];
						}
					}
					if (curved_boundaries) {
						/* first get temperature at image point */
						double T_image = 0;
						for (int i = 0; i < Boundaries[k].X_Image_Int.size(); i++) {
							Xp = Boundaries[k].X_Image_Int[i];
							Yp = Boundaries[k].Y_Image_Int[i];
							Zp = Boundaries[k].Z_Image_Int[i];
							T_image += Boundaries[k].W_Image_Int[i] * temperature[{Xp, Yp, Zp}];
						}
						temperature[{X, Y, Z}] = T_image;
					}
					break;
				}
				/// 105: Zero-gradient (1st-order) : T[i] = 2*T[i-1] - T[i-2]
				///  Temp neighbor = 2 * Temp boundary - Temp neighbor, so that the temperature gradient is zero. It is similar to the zero-temperature BC.
				case 105: {
					int Xp = (X - Boundaries[k].n[0]);
					int Yp = (Y - Boundaries[k].n[1]);
					int Zp = (Z - Boundaries[k].n[2]);

					int Xp1 = (X - 2 * Boundaries[k].n[0]);
					int Yp1 = (Y - 2 * Boundaries[k].n[1]);
					int Zp1 = (Z - 2 * Boundaries[k].n[2]);
					if (!curved_boundaries) {
						if (Boundaries[k].n[0] != 0) {
							temperature[{X, Y, Z}] = 2. * temperature[{Xp, Y, Z}] - temperature[{Xp1, Y, Z}];
						}
						if (Boundaries[k].n[1] != 0) {
							temperature[{X, Y, Z}] = 2. * temperature[{X, Yp, Z}] - temperature[{X, Yp1, Z}];
						}
						if (Boundaries[k].n[2] != 0) {
							temperature[{X, Y, Z}] = 2. * temperature[{X, Y, Zp}] - temperature[{X, Y, Zp1}];
						}
					}
					if (curved_boundaries) {
						/* first get temperature at image point */
						double T_image = 0;
						for (int i = 0; i < Boundaries[k].X_Image_Int.size(); i++) {
							Xp = Boundaries[k].X_Image_Int[i];
							Yp = Boundaries[k].Y_Image_Int[i];
							Zp = Boundaries[k].Z_Image_Int[i];
							T_image += Boundaries[k].W_Image_Int[i] * temperature[{Xp, Yp, Zp}];
						}
						if (Boundaries[k].n[0] != 0) {
							temperature[{X, Y, Z}] = 2. * T_image - temperature[{Xp1, Y, Z}];
						}
						if (Boundaries[k].n[1] != 0) {
							temperature[{X, Y, Z}] = 2. * T_image - temperature[{X, Yp1, Z}];
						}
						if (Boundaries[k].n[2] != 0) {
							temperature[{X, Y, Z}] = 2. * T_image - temperature[{X, Y, Zp1}];
						}
					}
					break;
				}
				/// 106: Zero-gradient (2nd-order) - Richardson extrapolation : T[i] = (4/3)*T[i-1] - (1/3)*T[i-2]
				case 106: {
					int Xp = X + Boundaries[k].n[0];
					int Yp = Y + Boundaries[k].n[1];
					int Zp = Z + Boundaries[k].n[2];

					int Xp1 = X + 2 * Boundaries[k].n[0];
					int Yp1 = Y + 2 * Boundaries[k].n[1];
					int Zp1 = Z + 2 * Boundaries[k].n[2];

					if (!curved_boundaries) {
						if (Boundaries[k].n[0] != 0) {
							temperature[{X, Y, Z}] = (4.0 / 3.0) * temperature[{Xp, Y, Z}] - (1.0 / 3.0) * temperature[{Xp1, Y, Z}];
						}
						if (Boundaries[k].n[1] != 0) {
							temperature[{X, Y, Z}] = (4.0 / 3.0) * temperature[{X, Yp, Z}] - (1.0 / 3.0) * temperature[{X, Yp1, Z}];
						}
						if (Boundaries[k].n[2] != 0) {
							temperature[{X, Y, Z}] = (4.0 / 3.0) * temperature[{X, Y, Zp}] - (1.0 / 3.0) * temperature[{X, Y, Zp1}];
						}
					}
					if (curved_boundaries) {
						// First get temperature at image point
						double T_image = 0;
						for (int i = 0; i < Boundaries[k].X_Image_Int.size(); i++) {
							int Xpi = Boundaries[k].X_Image_Int[i];
							int Ypi = Boundaries[k].Y_Image_Int[i];
							int Zpi = Boundaries[k].Z_Image_Int[i];
							T_image += Boundaries[k].W_Image_Int[i] * temperature[{Xpi, Ypi, Zpi}];
						}
						// Next calculate the temperature using the third-order zero gradient boundary condition
						if (Boundaries[k].n[0] != 0) {
							temperature[{X, Y, Z}] = (4.0 / 3.0) * T_image - (1.0 / 3.0) * temperature[{Xp1, Y, Z}];
						}
						if (Boundaries[k].n[1] != 0) {
							temperature[{X, Y, Z}] = (4.0 / 3.0) * T_image - (1.0 / 3.0) * temperature[{X, Yp1, Z}];
						}
						if (Boundaries[k].n[2] != 0) {
							temperature[{X, Y, Z}] = (4.0 / 3.0) * T_image - (1.0 / 3.0) * temperature[{X, Y, Zp1}];
						}
					}
				}
				/// 107: Zero-gradient (3rd-order) - Richardson extrapolation : T[i] = (18/11)*T[i-1] - (9/11)*T[i-2] + (2/11)*T[i-3]
				case 107: {
					int Xp = X - Boundaries[k].n[0];
					int Yp = Y - Boundaries[k].n[1];
					int Zp = Z - Boundaries[k].n[2];

					int Xp1 = X - 2 * Boundaries[k].n[0];
					int Yp1 = Y - 2 * Boundaries[k].n[1];
					int Zp1 = Z - 2 * Boundaries[k].n[2];

					int Xp2 = X - 3 * Boundaries[k].n[0];
					int Yp2 = Y - 3 * Boundaries[k].n[1];
					int Zp2 = Z - 3 * Boundaries[k].n[2];

					if (!curved_boundaries) {
						if (Boundaries[k].n[0] != 0) {
							temperature[{X, Y, Z}] = (18.0 / 11.0) * temperature[{Xp, Y, Z}] - (9.0 / 11.0) * temperature[{Xp1, Y, Z}] + (2.0 / 11.0) * temperature[{Xp2, Y, Z}];
						}
						if (Boundaries[k].n[1] != 0) {
							temperature[{X, Y, Z}] = (18.0 / 11.0) * temperature[{X, Yp, Z}] - (9.0 / 11.0) * temperature[{X, Yp1, Z}] + (2.0 / 11.0) * temperature[{X, Yp2, Z}];
						}
						if (Boundaries[k].n[2] != 0) {
							temperature[{X, Y, Z}] = (18.0 / 11.0) * temperature[{X, Y, Zp}] - (9.0 / 11.0) * temperature[{X, Y, Zp1}] + (2.0 / 11.0) * temperature[{X, Y, Zp2}];
						}
					}
					if (curved_boundaries) {
						double T_image = 0;
						for (int i = 0; i < Boundaries[k].X_Image_Int.size(); i++) {
							int Xpi = Boundaries[k].X_Image_Int[i];
							int Ypi = Boundaries[k].Y_Image_Int[i];
							int Zpi = Boundaries[k].Z_Image_Int[i];
							T_image += Boundaries[k].W_Image_Int[i] * temperature[{Xpi, Ypi, Zpi}];
						}
						if (Boundaries[k].n[0] != 0) {
							temperature[{X, Y, Z}] = (18.0 / 11.0) * T_image - (9.0 / 11.0) * temperature[{Xp1, Y, Z}] + (2.0 / 11.0) * temperature[{Xp2, Y, Z}];
						}
						if (Boundaries[k].n[1] != 0) {
							temperature[{X, Y, Z}] = (18.0 / 11.0) * T_image - (9.0 / 11.0) * temperature[{X, Yp1, Z}] + (2.0 / 11.0) * temperature[{X, Yp2, Z}];
						}
						if (Boundaries[k].n[2] != 0) {
							temperature[{X, Y, Z}] = (18.0 / 11.0) * T_image - (9.0 / 11.0) * temperature[{X, Y, Zp1}] + (2.0 / 11.0) * temperature[{X, Y, Zp2}];
						}
					}
					break;
				}
				/// 108: Temperature Relaxation or Temperature Damping BC : T[i] = (T[i] + λ*T[i-1])/(1 + λ).
				/// λ = -min(∇T.u, 0), this gives the temperature relaxation or damping at the boundary.
				/// If λ < 0, indicating the Flow is against the boundary, the temperature is relaxed or damped.
				case 108: {
					Xp = (X + Boundaries[k].n[0]);
					Yp = (Y + Boundaries[k].n[1]);
					Zp = (Z + Boundaries[k].n[2]);
					double normal_velocity = Boundaries[k].n[0] * Flow->velocity[0]
					                         + Boundaries[k].n[1] * Flow->velocity[1]
					                         + Boundaries[k].n[2] * Flow->velocity[2];
					double lambda = -std::min(normal_velocity, 0.0);
					double previous_temp = previous_temperature[{X, Y, Z}];
					double neighbor_temp = temperature[{Xp, Yp, Zp}];
					temperature[{X, Y, Z}] = (previous_temp + lambda * neighbor_temp) / (1.0 + lambda);
					break;
				}
			}
		}
		return;
	}
}
/// ***************************************************** ///
/// GET DISTRIBUTION FUNCTION MOMENTA                     ///
/// ***************************************************** ///
void Thermal_solver::momenta(Flow_solver* Flow, Parallel_MPI* MPI_parallel) {
	double energy_temp;
	if (MPI_parallel->processor_id != MASTER) {
		int X, Y, Z, alpha;
		for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (solid_thermal_type[{X, Y, Z}] == -1) {
						previous_temperature[{X, Y, Z}] = temperature[{X, Y, Z}];
						energy_previous[{X, Y, Z}] = energy[{X, Y, Z}];
						energy[{X, Y, Z}] = 0;
						for (alpha = 0; alpha < Discrete_Velocity; alpha++) {
							energy[{X, Y, Z}] += pop_t[{X, Y, Z, alpha}];
						}
						energy[{X, Y, Z}] += 0.5 * (force_thermal[{X, Y, Z}] + Production[{X, Y, Z}]);
						force_thermal[{X, Y, Z}] = 0.0;
						temp_force_thermal[{X, Y, Z}] = 0.0;
						Production[{X, Y, Z}] = 0.0;
						energy_temp = E_0 * energy[{X, Y, Z}] / (Flow->density[{X, Y, Z}] * Flow->rho_0)
						              - 0.5 * sqr(global_parameters.D_x / global_parameters.D_t) * (sqr(Flow->velocity[{X, Y, Z, 0}]) + sqr(Flow->velocity[{X, Y, Z, 1}]) + sqr(Flow->velocity[{X, Y, Z, 2}]));
						NewtonRaphson(temperature[{X, Y, Z}], energy_temp, c_p[{X, Y, Z}]);
						temperature[{X, Y, Z}] /= T_0;
					}
				}
			}
		}
	}
	return;
}
void Thermal_solver::momenta_FD(Flow_solver* Flow, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		int X, Y, Z;
		for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (solid_thermal_type[{X, Y, Z}] == -1) {
						// previous_temperature[{X, Y, Z}] = temperature[{X, Y, Z}];
						// energy_previous[{X, Y, Z}] = energy[{X, Y, Z}];
						force_thermal[{X, Y, Z}] = 0.0;
						temp_force_thermal[{X, Y, Z}] = 0.0;
					}
				}
			}
		}
	}
	return;
}
void Thermal_solver::momenta_crystal(Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		std::swap(previous_temperature, temperature);
		int X, Y, Z, alpha;
		for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (solid_thermal_type[{X, Y, Z}] == -1) {
						temperature[{X, Y, Z}] = 0;
						for (alpha = 0; alpha < Discrete_Velocity; alpha++) {
							temperature[{X, Y, Z}] += pop_t[{X, Y, Z, alpha}];
						}
						energy[{X, Y, Z}] = temperature[{X, Y, Z}];
					}
				}
			}
		}
	}
}
/// ***************************************************** ///
/// GET VISCOUS HEATING (ONLY DDF LB)                     ///
/// ***************************************************** ///
void Thermal_solver::viscous_heating(Flow_solver* Flow, Species_solver* Species, Parallel_MPI* MPI_parallel) {
	double tau_F, tau_T, theta, fneq;
	if (MPI_parallel->processor_id != MASTER) {
		int X, Y, Z, alpha;

#if defined compressible
		const double conv_factor = c_s2 * T_0 / sqr(global_parameters.D_x / global_parameters.D_t);
#endif
		for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (solid_thermal_type[{X, Y, Z}] == -1) {
						tau_T = c_s2 * thermal_diffusion_coefficient[{X, Y, Z}] * (T_0 / E_0) + 0.5;
						theta = 1;
#if defined compressible
#if defined Flow_With_Species
						const double r = R_GAS / Species->molar_mass_av[{X, Y, Z}];
#else
						const double r = R_GAS / Flow->M_av;
#endif  // defined
						theta = r * temperature[{X, Y, Z}] * conv_factor;
#endif
						tau_F = c_s2 * (Flow->viscosity[{X, Y, Z}] / theta) + 0.5;
						for (alpha = 0; alpha < Discrete_Velocity; alpha++) {
							fneq = Flow->pop[{X + c_alpha[alpha][0], Y + c_alpha[alpha][1], Z + c_alpha[alpha][2], alpha}] - Flow->pop_old[{X, Y, Z, alpha}];
							pop_t[{X + c_alpha[alpha][0], Y + c_alpha[alpha][1], Z + c_alpha[alpha][2], alpha}] +=
								((1. - 2. * tau_F) / (2. * tau_T))
								* (Flow->velocity[{X, Y, Z, 0}] * c_alpha[alpha][0] + Flow->velocity[{X, Y, Z, 1}] * c_alpha[alpha][1] + Flow->velocity[{X, Y, Z, 2}] * c_alpha[alpha][2])
								* fneq * (sqr(global_parameters.D_x / global_parameters.D_t) * Flow->rho_0 / E_0);
						}
					}
				}
			}
		}
	}
	return;
}
/// ***************************************************** ///
/// EXCHANGE POPULATIONS      BETWEEN PROCESSORS AT       ///
/// INTERFACES                                            ///
/// ***************************************************** ///
void Thermal_solver::Data_Exchange(Parallel_MPI* MPI_parallel) {
	if (!MPI_parallel->is_master()) {
		pop_group.exchange_data();
	}
}
/// ***************************************************** ///
/// EXCHANGE MACROSCOPIC VARS BETWEEN PROCESSORS AT       ///
/// INTERFACES                                            ///
/// ***************************************************** ///
void Thermal_solver::Data_Exchange_Macroscopic(Parallel_MPI* MPI_parallel) {
	if (!MPI_parallel->is_master()) {
		macroscopic_group.exchange_data();
	}
}
/// ***************************************************** ///
/// WRITE RECOVERY FILE                                   ///
/// ***************************************************** ///
void Thermal_solver::Recovery_write(Parallel_MPI* MPI_parallel, int& t) {
	if (MPI_parallel->processor_id != MASTER) {
		std::stringstream str_line1;
		str_line1 << "Alborz_Results/recovery/recover_Temperature_" << t << "_" << MPI_parallel->processor_id << ".dat";
		std::string strstr_line1 = str_line1.str();
		ofstream output_recovery;
		output_recovery.open(strstr_line1.c_str(), std::ios::out | std::ios::binary);
		//	    float mm[1];
		unsigned int X, Y, Z;

		for (X = 0; X < MPI_parallel->dev_end[0]; X++) {
			for (Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
				for (Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
					//			        for (unsigned alpha = 0; alpha < Discrete_Velocity; ++alpha) {
					//				        mm[0] = pop_t[{alpha,X,Y,Z}];
					//                        output_recovery.write((char*)mm, sizeof(float)* 1);
					//                        }
					output_recovery.write((char*)&(temperature[{X, Y, Z}]), sizeof(double) * 1);
					output_recovery.write((char*)&(Production[{X, Y, Z}]), sizeof(double) * 1);
					output_recovery.write((char*)&(thermal_diffusion_coefficient[{X, Y, Z}]), sizeof(double) * 1);
					output_recovery.write((char*)&(solid_thermal_type[{X, Y, Z}]), sizeof(int) * 1);
					//			        mm[0] = temperature[{X,Y,Z}];
					//			        output_recovery.write((char*)mm, sizeof(float)* 1);
					//			        mm[0] = energy[{X,Y,Z}];
					//			        output_recovery.write((char*)mm, sizeof(float)* 1);
					//			        mm[0] = Production[{X,Y,Z}];
					//			        output_recovery.write((char*)mm, sizeof(float)* 1);
					//			        mm[0] = force_thermal[{X,Y,Z}];
					//			        output_recovery.write((char*)mm, sizeof(float)* 1);
					//			        mm[0] = solid_thermal_type[{X,Y,Z}];
					//			        output_recovery.write((char*)mm, sizeof(float)* 1);
					//			        mm[0] = thermal_diffusion_coefficient[{X,Y,Z}];
					//			        output_recovery.write((char*)mm, sizeof(float)* 1);
				}
			}
		}
		output_recovery.close();
	}
	return;
}
/// ***************************************************** ///
/// READ  RECOVERY FILE                                   ///
/// ***************************************************** ///
void Thermal_solver::Recovery_read(Parallel_MPI* MPI_parallel, int& t) {
	if (MPI_parallel->processor_id != MASTER) {
		std::stringstream str_line1;
		str_line1 << "Alborz_Results/recovery/recover_Temperature_" << t << "_" << MPI_parallel->processor_id << ".dat";
		std::string strstr_line1 = str_line1.str();
		std::ifstream intput_recovery(strstr_line1, std::ios::in | std::ios::binary);
		//	    float tt[1];
		unsigned int X, Y, Z;

		for (X = 0; X < MPI_parallel->dev_end[0]; X++) {
			for (Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
				for (Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
					//			        for (unsigned alpha = 0; alpha < Discrete_Velocity; ++alpha) {
					//                        intput_recovery.read((char*)tt, sizeof(float));
					//                        pop_t[{alpha,X,Y,Z}] = tt[0];
					//                        }
					//                    intput_recovery.read((char*)tt, sizeof(float));
					//			        temperature[{X,Y,Z}] = tt[0];
					//			        intput_recovery.read((char*)tt, sizeof(float));
					//			        energy[{X,Y,Z}] = tt[0];
					//			        intput_recovery.read((char*)tt, sizeof(float));
					//			        Production[{X,Y,Z}] = tt[0];
					//			        intput_recovery.read((char*)tt, sizeof(float));
					//			        force_thermal[{X,Y,Z}] = tt[0];
					//			        intput_recovery.read((char*)tt, sizeof(float));
					//			        solid_thermal_type[{X,Y,Z}] = tt[0];
					//			        intput_recovery.read((char*)tt, sizeof(float));
					//			        thermal_diffusion_coefficient[{X,Y,Z}] = tt[0];
					intput_recovery.read((char*)&(temperature[{X, Y, Z}]), sizeof(double));
					previous_temperature[{X, Y, Z}] = temperature[{X, Y, Z}];
					intput_recovery.read((char*)&(Production[{X, Y, Z}]), sizeof(double));
					intput_recovery.read((char*)&(thermal_diffusion_coefficient[{X, Y, Z}]), sizeof(double));
					intput_recovery.read((char*)&(solid_thermal_type[{X, Y, Z}]), sizeof(int));
				}
			}
		}
		intput_recovery.close();
	}
}
void Thermal_solver::register_recovery(IO_interface& io) {
	io.add_field(temperature, "thermal_solver_temperature");
	io.add_field(Production, "thermal_solver_Production");
	io.add_field(thermal_diffusion_coefficient, "thermal_solver_diffusion_coefficient");
	io.add_field(solid_thermal_type, "thermal_solver_solid_thermal_type");
}
/// ***************************************************** ///
/// SPONGE ZONE                                           ///
/// ***************************************************** ///
/*
Sponge function is used to damp the Flow field near the boundaries to avoid the reflection of the waves.
Just like viscosity controls how fluid flows, the thermal_diffusion_coefficient controls how heat spreads within the fluid.
A higher diffusion coefficient means heat spreads more quickly through the fluid, while a lower coefficient means it spreads more slowly.
In regions with a higher diffusion coefficient, temperature changes will propagate faster, leading to a more uniform temperature distribution over time.
Conversely, in regions with a lower diffusion coefficient, temperature changes will propagate more slowly, leading to more localized temperature variations.
*/
void Thermal_solver::Sponge_zone(double r_start, double r_end, int direction, Flow_solver* Flow, stl_import* Geo_stl, Parallel_MPI* MPI_parallel) {
	int X, Y, Z;
	double max_alpha = 0.5 / c_s2;
	if (MPI_parallel->processor_id != MASTER) {
		for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					double xc, yc, zc, rr;
					MPI_parallel->get_coordinates(X, Y, Z, Geo_stl->x_center, Geo_stl->y_center, Geo_stl->z_center, xc, yc, zc);
					if (direction == 0) rr = xc;
					if (direction == 1) rr = yc;
					if (direction == 2) rr = zc;
					if (solid_thermal_type[{X, Y, Z}] == FALSE && rr >= r_start && rr <= r_end) {
						double max_lambda = max_alpha * c_p[{X, Y, Z}] * Flow->density[{X, Y, Z}] * Flow->rho_0;
						thermal_diffusion_coefficient[{X, Y, Z}] += ((max_lambda - thermal_diffusion_coefficient[{X, Y, Z}]) * .5 * (sin(M_PI * (rr - r_start) / (r_end - r_start) - M_PI / 2.) + 1.));
					}
				}
			}
		}
	}
}
/// ***************************************************** ///
/// APPLY MODEL FOR COMPUTATION OF DIFFUSION COEFF        ///
/// ***************************************************** ///
void Thermal_solver::Diffusion_Coefficient_computation(Flow_solver* Flow, Species_solver* Species, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		unsigned int X = 0, Y = 0, Z = 0, Xp, Yp, Zp, k;
		double rho_star;
		double T_star = 273.;
		double S = 110.5;
#if defined Flow_With_Species
		rho_star = Flow->p_th_0 * Species->molar_mass_av[{X, Y, Z}] / (R_GAS * T_star);
#endif
#if !defined Flow_With_Species
		rho_star = Flow->p_th_0 * Flow->M_av / (R_GAS * T_star);
#endif
		for (X = MPI_parallel->start_XYZ2[0] - 1; X <= MPI_parallel->end_XYZ2[0] + 1; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1] - 1; Y <= MPI_parallel->end_XYZ2[1] + 1; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2] - 1; Z <= MPI_parallel->end_XYZ2[2] + 1; ++Z) {
					if (solid_thermal_type[{X, Y, Z}] == -1) {
						thermal_diffusion_coefficient[{X, Y, Z}] = c_p[{X, Y, Z}] * Sutherland_thermal_diffusion(rho_star * Flow->fluid_constant_viscosity, T_star / T_0, S / T_0, temperature[{X, Y, Z}], 0.71);
					}
				}
			}
		}

		for (k = 0; k < Boundaries.size(); ++k) {
			X = Boundaries[k].X;
			Y = Boundaries[k].Y;
			Z = Boundaries[k].Z;
			Xp = (X - Boundaries[k].n[0]);
			Yp = (Y - Boundaries[k].n[1]);
			Zp = (Z - Boundaries[k].n[2]);
			thermal_diffusion_coefficient[{X, Y, Z}] = c_p[{X, Y, Z}] * Sutherland_thermal_diffusion(rho_star * Flow->fluid_constant_viscosity, T_star / T_0, S / T_0, temperature[{X, Y, Z}], 0.71);
			thermal_diffusion_coefficient[{Xp, Yp, Zp}] = c_p[{Xp, Yp, Zp}] * Sutherland_thermal_diffusion(rho_star * Flow->fluid_constant_viscosity, T_star / T_0, S / T_0, temperature[{Xp, Yp, Zp}], 0.71);
		}
	}
}
void Thermal_solver::check_residual(Parallel_MPI* MPI_parallel, int t) {  // Based on paper Applied Mathematical Modelling 39 (2015) 2436�2451
	                                                                      //	bool st_cr = false;
	double thermal_criteria = 0;
	double thermal_criteria_2 = 0;
	double thermal_global, thermal_global_2;
	if (MPI_parallel->processor_id != MASTER) {
		for (int X = MPI_parallel->start_XYZ2[0]; X < MPI_parallel->end_XYZ2[0]; X++) {
			for (int Y = MPI_parallel->start_XYZ2[1]; Y < MPI_parallel->end_XYZ2[1]; Y++) {
				for (int Z = MPI_parallel->start_XYZ2[2]; Z < MPI_parallel->end_XYZ2[2]; Z++) {
					if (solid_thermal_type[{X, Y, Z}] == -1) {
						thermal_criteria += abs(temperature[{X, Y, Z}] - previous_temperature[{X, Y, Z}]);
						thermal_criteria_2 += abs(temperature[{X, Y, Z}]);
					}
				}
			}
		}
	}
	MPI_Allreduce(&thermal_criteria, &thermal_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&thermal_criteria_2, &thermal_global_2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	if (MPI_parallel->processor_id == MASTER) {
		/// Create filename
		stringstream output_filename;
		output_filename << "Alborz_Results/debug/convergence_T.dat";
		ofstream output_file;
		/// Open file
		output_file.open(output_filename.str().c_str(), fstream::app);
		/// Write data
		output_file << t << " ";  // time step
		output_file << setprecision(30) << (thermal_global / thermal_global_2) << endl;
		/// Close file
		output_file.close();
	}
}
/// ***************************************************** ///
/// FUNCTIONS NEEDED FOR BINARY VTK WRITER                ///
/// ***************************************************** ///
void Thermal_solver::swap_endian(char* buffer, size_t size) {
	for (size_t b = 0; b < size / 2; b++) {
		char temp = buffer[b];
		buffer[b] = buffer[size - 1 - b];
		buffer[size - 1 - b] = temp;
	}
}
void Thermal_solver::swap_endian(char* buffer, char* buffer2, size_t size) {
	for (size_t b = 0; b < size; b++) {
		buffer2[b] = buffer[size - 1 - b];
	}
}
void Thermal_solver::write_bigendian(std::ofstream& file, char* buffer, size_t count, size_t size) {
	char* _buf = new char[size];
	for (size_t i = 0; i < count; i++) {
		swap_endian(buffer + (size * i), _buf, size);
		file.write(_buf, size);
	}
	delete[] _buf;
}
template <typename T>
void Thermal_solver::write_bigendian(std::ofstream& file, T* buffer, size_t count) {
	char* _buf = new char[sizeof(T)];
	for (size_t i = 0; i < count; i++) {
		swap_endian((char*)(buffer + i), _buf, sizeof(T));
		file.write(_buf, sizeof(T));
	}
	delete[] _buf;
}
/// ***************************************************** ///
/// WRITE VTK FILE                                        ///
/// ***************************************************** ///
void Thermal_solver::write_vtk(int time, int t_vtk, stl_import* Geo, Parallel_MPI* MPI_parallel) {
	if (time % t_vtk == 0) {
		/// Create filename
		int X, Y, Z;
		double mm[1];
		vector<unsigned int> begin_vtk;
		vector<unsigned int> end_vtk;
		vector<unsigned int> begin_vtk_gen_co;
		vector<unsigned int> end_vtk_gen_co;
		vector<unsigned int> length_vtk;
		begin_vtk.resize(3);
		end_vtk.resize(3);
		begin_vtk_gen_co.resize(3);
		end_vtk_gen_co.resize(3);
		length_vtk.resize(3);
		/// Create filename
		/// Create output data file, filename format: fluid_t%time-step%_%MPI_parallel->processor_id%.vti
		stringstream output_filename;
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
			output_filename << "Alborz_Results/vtk_fluid/temperature_t" << time << "_" << MPI_parallel->processor_id << ".vti";
			ofstream output_file;
			/// Open file
#if defined VTK_ASCII
			output_file.open(output_filename.str().c_str(), ios::out);
			string DataType = "ascii";
#endif  // defined
#if defined VTK_BINARY
			output_file.open(output_filename.str().c_str(), std::ios::binary | ios::out);
			string DataType = "appended";
#endif  // defined
        /// Write VTI XML header
			output_file << "<?xml version=\"1.0\"?>" << endl;
			output_file << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"BigEndian\" header_type=\"UInt64\">" << endl;
			output_file << "<ImageData WholeExtent=\"";
			output_file << std::setprecision(14) << begin_vtk_gen_co[0] << " " << end_vtk_gen_co[0] << " ";
			output_file << std::setprecision(14) << begin_vtk_gen_co[1] << " " << end_vtk_gen_co[1] << " ";
			output_file << std::setprecision(14) << begin_vtk_gen_co[2] << " " << end_vtk_gen_co[2] << "\" ";
			output_file << std::setprecision(14) << "Origin=\" " << Geo->x_center << " " << Geo->y_center << " " << Geo->z_center << "\" Spacing=\"" << global_parameters.D_x << " " << global_parameters.D_x << " " << global_parameters.D_x << "\">" << endl;
			output_file << std::setprecision(14) << "<Piece Extent=\"" << begin_vtk_gen_co[0] << " " << end_vtk_gen_co[0] << " ";
			output_file << std::setprecision(14) << begin_vtk_gen_co[1] << " " << end_vtk_gen_co[1] << " ";
			output_file << std::setprecision(14) << begin_vtk_gen_co[2] << " " << end_vtk_gen_co[2] << "\">" << endl;
			output_file << "<PointData Scalars=\"Fluid\">" << endl;
			unsigned int pointer_index = 0;
			output_file << "<DataArray type=\"Float64\" Name=\"T"
						<< "\" format=\"" << DataType;
#if defined VTK_BINARY
			output_file << "\" offset=\"" << pointer_index * length_vtk[0] * length_vtk[1] * length_vtk[2] * sizeof(double) + pointer_index * sizeof(int64_t);
			pointer_index++;
#endif  // defined
			output_file << "\">" << endl;
#if defined VTK_ASCII
			for (Z = begin_vtk[2]; Z <= end_vtk[2]; ++Z) {
				for (Y = begin_vtk[1]; Y <= end_vtk[1]; ++Y) {
					for (X = begin_vtk[0]; X <= end_vtk[0]; ++X) {
						output_file << std::setprecision(8) << T_0 * temperature[{X, Y, Z}] << endl;
					}
				}
			}
#endif  // defined
			output_file << "</DataArray>" << endl;
			output_file << "<DataArray type=\"Float64\" Name=\"HR_rate\" format=\"" << DataType;
#if defined VTK_BINARY
			output_file << "\" offset=\"" << pointer_index * length_vtk[0] * length_vtk[1] * length_vtk[2] * sizeof(double) + pointer_index * sizeof(int64_t);
			;
			pointer_index++;
#endif  // defined
			output_file << "\">" << endl;
#if defined VTK_ASCII
			for (Z = begin_vtk[2]; Z <= end_vtk[2]; ++Z) {
				for (Y = begin_vtk[1]; Y <= end_vtk[1]; ++Y) {
					for (X = begin_vtk[0]; X <= end_vtk[0]; ++X) {
						output_file << std::setprecision(8) << Production[{X, Y, Z}] / global_parameters.D_t << endl;  /// [J/m3s]
					}
				}
			}
#endif  // defined
			output_file << "</DataArray>" << endl;
			output_file << "</PointData>" << endl;
			output_file << "</Piece>" << endl;
			output_file << "</ImageData>" << endl;
#if defined VTK_BINARY
			int64_t nn[1];
			output_file << "<AppendedData encoding=\"raw\">" << endl;
			output_file << "_";
			nn[0] = sizeof(double) * length_vtk[0] * length_vtk[1] * length_vtk[2];
			write_bigendian(output_file, nn, 1);
			for (Z = begin_vtk[2]; Z <= end_vtk[2]; ++Z) {
				for (Y = begin_vtk[1]; Y <= end_vtk[1]; ++Y) {
					for (X = begin_vtk[0]; X <= end_vtk[0]; ++X) {
						mm[0] = T_0 * temperature[{X, Y, Z}];
#if !defined DEBUG_MODE
						if (solid_thermal_type[{X, Y, Z}] != -1) mm[0] = -1;
#endif
						write_bigendian(output_file, mm, 1);
					}
				}
			}
			output_file.flush();
			nn[0] = sizeof(double) * length_vtk[0] * length_vtk[1] * length_vtk[2];
			write_bigendian(output_file, nn, 1);
			for (Z = begin_vtk[2]; Z <= end_vtk[2]; ++Z) {
				for (Y = begin_vtk[1]; Y <= end_vtk[1]; ++Y) {
					for (X = begin_vtk[0]; X <= end_vtk[0]; ++X) {
						mm[0] = Production[{X, Y, Z}] / global_parameters.D_t;
#if !defined DEBUG_MODE
						if (solid_thermal_type[{X, Y, Z}] != -1) mm[0] = -1;
#endif
						write_bigendian(output_file, mm, 1);
					}
				}
			}
			output_file.flush();
			output_file << "</AppendedData>" << endl;
#endif  // defined
			output_file << "</VTKFile>" << endl;
			/// Close file
			output_file.close();
		}
		MPI_parallel->Sync_Master();
		if (MPI_parallel->processor_id == MASTER) {
			output_filename << "Alborz_Results/vtk_fluid/temperature_t" << time << ".pvti";
			ofstream output_file;
			/// Open file
			output_file.open(output_filename.str().c_str());
#if defined VTK_ASCII
			string DataType = "ascii";
#endif  // defined
#if defined VTK_BINARY
			string DataType = "appended";
#endif  // defined
        /// Write VTI XML header
			output_file << "<?xml version=\"1.0\"?>" << endl;
			output_file << "<VTKFile type=\"PImageData\" version=\"0.1\">" << endl;
			output_file << "<PImageData WholeExtent=\"";
			output_file << std::setprecision(14) << MPI_parallel->start_XYZ[0] << " " << global_parameters.Nx - 1 << " ";
			output_file << std::setprecision(14) << MPI_parallel->start_XYZ[1] << " " << global_parameters.Ny - 1 << " ";
			output_file << std::setprecision(14) << MPI_parallel->start_XYZ[2] << " " << global_parameters.Nz - 1 << " "
						<< "\" ";
			output_file << std::setprecision(14) << "Origin=\" " << Geo->x_center << " " << Geo->y_center << " " << Geo->z_center << "\" Spacing=\"" << global_parameters.D_x << " " << global_parameters.D_x << " " << global_parameters.D_x << "\" GhostLevel=\"0\">" << endl;
			output_file << "<PPointData Scalars=\"Fluid\">" << endl;

			output_file << "<PDataArray type=\"Float64\" Name=\"T"
						<< "\" format=\"" << DataType << "\">" << endl;
			output_file << "</PDataArray>" << endl;
			output_file << "<PDataArray type=\"Float64\" Name=\"HR_rate\" format=\"" << DataType << "\">" << endl;
			output_file << "</PDataArray>" << endl;
			output_file << "</PPointData>" << endl;
			for (int i = 1; i < MPI_parallel->num_processors; i++) {
				output_file << "<Piece Extent=\"";
				begin_vtk[0] = MPI_parallel->start_XYZ[0 + 3 * i];
				begin_vtk[1] = MPI_parallel->start_XYZ[1 + 3 * i];
				begin_vtk[2] = MPI_parallel->start_XYZ[2 + 3 * i];
				end_vtk[0] = MPI_parallel->end_XYZ[0 + 3 * i] + 1;
				end_vtk[1] = MPI_parallel->end_XYZ[1 + 3 * i] + 1;
				end_vtk[2] = MPI_parallel->end_XYZ[2 + 3 * i] + 1;
				if (MPI_parallel->end_XYZ[0 + 3 * i] == global_parameters.Nx - 1) end_vtk[0] -= 1;
				if (MPI_parallel->end_XYZ[1 + 3 * i] == global_parameters.Ny - 1) end_vtk[1] -= 1;
				if (MPI_parallel->end_XYZ[2 + 3 * i] == global_parameters.Nz - 1) end_vtk[2] -= 1;

				output_file << std::setprecision(14) << begin_vtk[0] << " " << end_vtk[0] << " ";
				output_file << std::setprecision(14) << begin_vtk[1] << " " << end_vtk[1] << " ";
				output_file << std::setprecision(14) << begin_vtk[2] << " " << end_vtk[2] << "\" ";
				output_file << "Source=\"temperature_t" << time << "_" << i << ".vti\"/>" << endl;
			}
			output_file << "</PImageData>" << endl;
			output_file << "</VTKFile>" << endl;
			/// Close file
			output_file.close();
		}
	}
	return;
}
/// ***************************************************** ///
/// COMPUTE MAXIMUM FO IN DOMAIN                          ///
/// ***************************************************** ///
double Thermal_solver::Fo_monitor(Flow_solver* Flow, Parallel_MPI* MPI_parallel, int time, unsigned int output) {
	double alpha_max = -1;
	double alpha_min = 1e16;
	double alpha_max_global, alpha_min_global;
	if (MPI_parallel->processor_id != MASTER) {
		for (int X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (int Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (int Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (solid_thermal_type[{X, Y, Z}] == -1) {
						alpha_max = std::max(alpha_max, thermal_diffusion_coefficient[{X, Y, Z}] / (c_p[{X, Y, Z}] * Flow->density[{X, Y, Z}] * Flow->rho_0));
						alpha_min = std::min(alpha_min, thermal_diffusion_coefficient[{X, Y, Z}] / (c_p[{X, Y, Z}] * Flow->density[{X, Y, Z}] * Flow->rho_0));
					}
				}
			}
		}
	}
	MPI_Allreduce(&alpha_max, &alpha_max_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);  // MPI processors perform a reduction operation to find the global maximum (alpha_max_global) and minimum (alpha_min_global) Fourier numbers across all processors.
	MPI_Allreduce(&alpha_min, &alpha_min_global, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
	if (MPI_parallel->processor_id == MASTER && output == 1) {
		/// Create filename
		stringstream output_filename;
		output_filename << "Alborz_Results/debug/ThermalDiffusivityMinMax.dat";
		ofstream output_file;
		/// Open file
		output_file.open(output_filename.str().c_str(), fstream::app);
		/// Write data
		output_file << time << " ";  // time step
		output_file << setprecision(30) << alpha_min_global << " " << alpha_max_global << endl;
		/// Close file
		output_file.close();
	}
	return alpha_max_global;
}
double Thermal_solver::average_temp(const Parallel_MPI& MPI_parallel) const {
	double temp = 0.0;
	size_t nodes = 0;
	if (!MPI_parallel.is_master()) {
		for (int X = MPI_parallel.start_XYZ2[0]; X <= MPI_parallel.end_XYZ2[0]; ++X) {
			for (int Y = MPI_parallel.start_XYZ2[1]; Y <= MPI_parallel.end_XYZ2[1]; ++Y) {
				for (int Z = MPI_parallel.start_XYZ2[2]; Z <= MPI_parallel.end_XYZ2[2]; ++Z) {
					if (solid_thermal_type[{X, Y, Z}] == FALSE) {
						temp += temperature[{X, Y, Z}];
						++nodes;
					}
				}
			}
		}
	}
	double temp_total = temp;
	size_t nodes_total = nodes;
	MPI_Reduce(&temp, &temp_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&nodes, &nodes_total, 1, to_MPI_type<size_t>::value(), MPI_SUM, 0, MPI_COMM_WORLD);

	return temp_total / nodes_total;
}
void Thermal_solver::temp_monitor(unsigned int time_step, Flow_solver* Flow, const Parallel_MPI& MPI_parallel) {
	double t_physical_time = 0.0;
	t_physical_time += global_parameters.D_t;
	double max_temp_local = 0.0;
	int max_position_local[3] = {0, 0, 0};
	if (!MPI_parallel.is_master()) {
		for (int X = MPI_parallel.start_XYZ2[0]; X <= MPI_parallel.end_XYZ2[0]; ++X) {
			for (int Y = MPI_parallel.start_XYZ2[1]; Y <= MPI_parallel.end_XYZ2[1]; ++Y) {
				for (int Z = MPI_parallel.start_XYZ2[2]; Z <= MPI_parallel.end_XYZ2[2]; ++Z) {
					if (solid_thermal_type[{X, Y, Z}] == FALSE) {
						if (temperature[{X, Y, Z}] > max_temp_local) {
							max_temp_local = temperature[{X, Y, Z}] * T_0;
							max_position_local[0] = X;
							// max_position_local[1] = Y * global_parameters.D_x;
							// max_position_local[2] = Z * global_parameters.D_x;
						}
					}
				}
			}
		}
	}

	struct {
		double temp;
		int position[3];
	} local_data, global_data;
	local_data.temp = max_temp_local;
	local_data.position[0] = max_position_local[0];
	// local_data.position[1] = max_position_local[1];
	// local_data.position[2] = max_position_local[2];
	//  Reduce data globally
	MPI_Reduce(&local_data, &global_data, 1, MPI_DOUBLE_INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);
	if (MPI_parallel.is_master()) {
		ofstream output_file("Alborz_Results/debug/temp_monitor.dat", fstream::app);
		if (output_file.tellp() == 0) {
			output_file << "Time_Step\tPhysical_Time\tMax_Temp" << std::endl;
		}
		output_file << time_step << "\t" << t_physical_time << "\t";  // time step
		output_file << "\t" << setprecision(15) << global_data.temp << std::endl;
		output_file.close();
	}
}
Thermal_solver::~Thermal_solver() {
}
/// ***************************************************** ///
/// GAUSSIAN HEAT SOURCE: READ INPUT FROM FILE            ///
/// ***************************************************** ///
// A Gaussian heat source is a type of heat source that has a spatial distribution described by a Gaussian function.
// In the context of heat transfer or Thermal simulations, a Gaussian heat source is often used to model localized
// heating or energy deposition.
// The function here takes info about each heat source,
// including position (Xs, Ys, Zs), radius (Rs), start and end times (Tstart and Tend),
// standard deviations (Rsigma and Tsigma), and energy (Es).
void Heat_source::read_input(std::string filename, double Dt, Parallel_MPI* MPI_parallel) {
	//	int column_width = 40;
	/// Open input file
	string input_filename(filename);
	input_filename += ".dat";
	ifstream input_file;  // File is open for READING
	input_file.open(input_filename.c_str(), ios::binary);

	input_file.clear();
	input_file.seekg(0, ios::beg);
	find_line_after_header(input_file, "c\tHeat Sources");
	find_line_after_comment(input_file);
	int N_HS;
	input_file >> N_HS;
	find_line_after_comment(input_file);
	Xs.resize(N_HS);
	Ys.resize(N_HS);
	Zs.resize(N_HS);
	Rs.resize(N_HS);
	Tstart.resize(N_HS);
	Tend.resize(N_HS);
	Rsigma.resize(N_HS);
	Tsigma.resize(N_HS);
	Es.resize(N_HS);
	for (int i = 0; i < Xs.size(); i++) {
		input_file >> Es[i] >> Xs[i] >> Ys[i] >> Zs[i] >> Rs[i] >> Tstart[i] >> Tend[i] >> Rsigma[i] >> Tsigma[i];
		Es[i] *= Dt;
	}
	input_file.close();
	if (MPI_parallel->processor_id == MASTER + 1) {
		stringstream HS_filename;
		HS_filename << "Alborz_Results/debug/HeatSources.dat";
		ofstream HSout;
		HSout.open(HS_filename.str().c_str(), fstream::trunc);
		HSout << "List of Heat sources" << endl;
		HSout << "=====================" << endl;
		HSout << "Index\tE\tX\tY\tZ\tR\tt_start\tt_end\tr_sigma\tt_sigma\n";
		for (int i = 0; i < Xs.size(); i++) {
			HSout << i + 1 << "\t" << Es[i] << "\t" << Xs[i] << "\t" << Ys[i] << "\t" << Zs[i] << "\t" << Rs[i] << "\t" << Tstart[i] << "\t" << Tend[i] << "\t" << Rsigma[i] << "\t" << Tsigma[i] << "\n";
		}
		HSout.close();
	}
}
/// ***************************************************** ///
/// GAUSSIAN HEAT SOURCE: ADD HEAT                        ///
/// ***************************************************** ///
void Heat_source::Add_heat(Thermal_solver* Thermal, Parallel_MPI* MPI_parallel, int tm, int N_x, int N_y, int N_z) {
	int X, Y, Z, i, xx, yy, zz, check;
	if (MPI_parallel->processor_id != MASTER) {
		check = -1;
		for (i = 0; i < Xs.size(); i++) {
			if (tm < (Tend[i] + 2 * Tsigma[i]) && tm > (Tstart[i] - 2 * Tsigma[i])) check = 1;
		}
		if (check == 1) {
			for (X = 0; X < MPI_parallel->dev_end[0]; X++) {
				xx = fmod(X - MPI_parallel->start_XYZ2[0] + MPI_parallel->start_XYZ[0] + N_x, N_x);
				for (Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
					yy = fmod(Y - MPI_parallel->start_XYZ2[1] + MPI_parallel->start_XYZ[1] + N_y, N_y);
					for (Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
						zz = fmod(Z - MPI_parallel->start_XYZ2[2] + MPI_parallel->start_XYZ[2] + N_z, N_z);
						for (i = 0; i < Xs.size(); i++) {
							if (Thermal->temperature[{X, Y, Z}] < 1100) {
								Thermal->Production[{X, Y, Z}] += Es[i] * 0.5 * (tanh((tm - Tstart[i]) / Tsigma[i]) - tanh((tm - Tend[i]) / Tsigma[i]))
								                                  * 0.5 * (1. - tanh((sqrt(sqr(xx - Xs[i]) + sqr(yy - Ys[i]) + sqr(zz - Zs[i])) - Rs[i]) / Rsigma[i]));
							}
						}
					}
				}
			}
		}
	}
}
/// ***************************************************** ///
/// INITIAL CONDITIONS: USER-DEFINED VALUES (FROM INPUT   ///
/// FILE)                                                 ///
/// ***************************************************** ///
void Inline_User_Defined(Scalar_field& T, Scalar_field& c_p, Scalar_field& thermal_diffusion_coefficient, Solid_field& solid, double N_x, double N_y, double N_z, double cs_2, double T_0, double E_0, double rho_0, const std::string& filename, int Zones, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->is_master()) {
		return;
	}
	Ini_T.resize(Zones + 1, 0.0);
	Ini_cp.resize(Zones + 1, 0.0);
	Ini_lambda.resize(Zones + 1, 0.0);
	std::vector<int> type(Zones + 1, 0.0);
	type[0] = 1;  /// -----> Solids are set to +1
	std::vector<Initial_field_slice> special_volumes;
	/// Open input file
	ifstream input_file(filename + ".dat", ios::binary);
	find_line_after_header(input_file, "c\tTemperature Field Initial Conditions");
	if (MPI_parallel->processor_id == (MASTER + 1)) {
		std::cout << "ENERGY: User defined Initial conditions \n";
		std::cout << "================================ \n";
	}
	for (int i = 0; i < Zones; i++) {
		find_line_after_comment(input_file);
		int index = 0;
		input_file >> index;
		const bool is_extra = index < 0;
		// special area with geometry defined by the config
		if (is_extra) {
			if (!special_volumes.empty() && MPI_parallel->processor_id == (MASTER + 1)) {
				std::cerr << "[Warning] Currently only one special volume is fully supported for initial conditions.\n";
			}
			index = Ini_T.size();
			type.push_back(0);
			Ini_T.push_back(0.0);
			Ini_cp.push_back(0.0);
			Ini_lambda.push_back(0.0);
			--i;
		}
		input_file >> type[index];
		input_file >> Ini_T[index];
		input_file >> Ini_cp[index];
		input_file >> Ini_lambda[index];
		Ini_T[index] /= T_0;
		Ini_lambda[index] *= (global_parameters.D_t / sqr(global_parameters.D_x));

		if (is_extra) {
			special_volumes.push_back({});
			Initial_field_slice& slice = special_volumes.back();
			slice.index = index;
		}
		if (MPI_parallel->processor_id == (MASTER + 1)) {
			std::cout << "Zone : " << index << "\t Type : " << type[index] << "\n";
			std::cout << "Initial temperature : " << Ini_T[index] << "\t lambda : " << Ini_lambda[index] * (T_0 / E_0) << "\n";
			std::cout << "Initial cp: " << Ini_cp[index] << "\n";
		}
	}
	input_file.close();
	for (unsigned X = 0; X < MPI_parallel->dev_end[0]; X++) {
		for (unsigned Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
			for (unsigned Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
				const int zone = solid[{X, Y, Z}];
				T[{X, Y, Z}] = Ini_T[zone];
				c_p[{X, Y, Z}] = Ini_cp[zone];
				thermal_diffusion_coefficient[{X, Y, Z}] = Ini_lambda[zone];
				solid[{X, Y, Z}] = type[zone];
			}
		}
	}
}
/// ***************************************************** ///
/// INITIAL CONDITIONS: 3-D COMPRESSIBLE TAYLOR-GREEN     ///
/// (INPUT FATA FILE NEEDED) VORTEX                       ///
/// ***************************************************** ///
/* The Taylor-Green vortex is a solution to the Navier-Stokes equations that describes the Flow field of a fluid in a periodic box.
The Flow field is characterized by a pair of counter-rotating vortices that are aligned with the x and y axes.
The equations for the Taylor-Green vortex are given by:
u(x, y, z, t) = sin(x) * cos(y) * cos(z) * exp(-2 * nu * t)
v(x, y, z, t) = -cos(x) * sin(y) * cos(z) * exp(-2 * nu * t)
w(x, y, z, t) = 0
p(x, y, z, t) = -0.5 * rho * (cos(2 * x) + cos(2 * y)) * exp(-4 * nu * t)
where u, v, and w are the velocity components in the x, y, and z directions, respectively, p is the pressure, rho is the density of the fluid, nu is the kinematic viscosity of the fluid, and t is the time.

The steps used in Thermal initialization are:
1. Read the initial temperature field from a file.
2. Determine the temperature field at each grid point based on the initial temperature field.
3. Set the heat capacity at each grid point to a constant value.
4. Set the diffusion coefficient at each grid point to a constant value.
5. Set the solid field at each grid point to a constant value.
6. Set the temperature field at each grid point to the initial temperature field.
*/
void TGV3Dcold_thermal(Scalar_field& T, Scalar_field& c_p, Scalar_field& thermal_diffusion_coefficient, Solid_field& solid,
                       double N_x, double N_y, double N_z, double cs_2, double T_0, double E_0, double rho_0,
                       const std::string& filename, int Zones, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		unsigned int X, Y, Z, i, N_grid;
		double xx;
		std::string file_address;
		std::vector<double> T_initial;
		std::vector<double> x_initial;
		std::ifstream input_file(filename + ".dat", std::ios::binary);
		find_line_after_header(input_file, "c\tThermal Initial Profile File");
		// keep numbers in scientific form
		input_file >> N_grid >> file_address;
		input_file.close();
		std::string data_filename = file_address;
		std::ifstream data_file(data_filename, std::ios::in | std::ios::binary);
		T_initial.resize(N_grid);
		x_initial.resize(N_grid);
		for (X = 0; X < N_grid; ++X) {
			data_file >> x_initial[X] >> T_initial[X];
			x_initial[X] *= 0.01;
		}
		data_file.close();
		for (X = 0; X < MPI_parallel->dev_end[0]; ++X) {
			xx = fmod(X - MPI_parallel->start_XYZ2[0] + MPI_parallel->start_XYZ[0] + N_x, N_x) * global_parameters.D_x;
			unsigned index = 0;
			for (i = 0; i < x_initial.size() - 1; ++i) {
				if (xx > x_initial[i] && xx <= x_initial[i + 1]) {
					index = i;
					break;
				}
			}
			for (Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
				for (Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
					T[{X, Y, Z}] = T_initial[index] + (xx - x_initial[index]) * (T_initial[index + 1] - T_initial[index]) / (x_initial[index + 1] - x_initial[index]);
					T[{X, Y, Z}] = T[{X, Y, Z}] / T_0;
					c_p[{X, Y, Z}] = 1.;
					thermal_diffusion_coefficient[{X, Y, Z}] = 0.005;
					solid[{X, Y, Z}] = -1;
				}
			}
		}
		if (MPI_parallel->processor_id == (MASTER + 1)) {
			std::cout << "ENERGY: non-reacting TGV-3D initial conditions \n";
			std::cout << "================================ \n";
			std::cout << "Thermal file address: " << file_address << std::endl;
			std::cout << "N_grid: " << N_grid << std::endl;
			std::cout << "================================ \n";
		}
	}
}
/// ***************************************************** ///
/// INITIAL CONDITIONS: 3-D REACTING TAYLOR-GREEN         ///
/// (INPUT FATA FILE NEEDED) VORTEX                       ///
/// ***************************************************** ///
/* The Taylor-Green vortex is a solution to the Navier-Stokes equations that describes the Flow field of a fluid in a periodic box.
The Flow field is characterized by a pair of counter-rotating vortices that are aligned with the x and y axes.
The equations for the Taylor-Green vortex are given by:
u(x, y, z, t) = sin(x) * cos(y) * cos(z) * exp(-2 * nu * t)
v(x, y, z, t) = -cos(x) * sin(y) * cos(z) * exp(-2 * nu * t)
w(x, y, z, t) = 0
p(x, y, z, t) = -0.5 * rho * (cos(2 * x) + cos(2 * y)) * exp(-4 * nu * t)
where u, v, and w are the velocity components in the x, y, and z directions, respectively, p is the pressure, rho is the density of the fluid, nu is the kinematic viscosity of the fluid, and t is the time.

The steps used in Thermal initialization are:
1. Read the initial temperature field from a file.
2. Determine the temperature field at each grid point based on the initial temperature field.
3. Set the heat capacity at each grid point to a constant value.
4. Set the diffusion coefficient at each grid point to a constant value.
5. Set the solid field at each grid point to a constant value.
6. Set the temperature field at each grid point to the initial temperature field.
*/
void TGV3Dreacting_thermal(Scalar_field& T, Scalar_field& c_p, Scalar_field& thermal_diffusion_coefficient, Solid_field& solid,
                           double N_x, double N_y, double N_z, double cs_2, double T_0, double E_0, double rho_0,
                           const std::string& filename, int Zones, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		unsigned int X, Y, Z, i, N_grid;
		double xx;
		std::string file_address;
		std::vector<double> T_initial;
		std::vector<double> x_initial;

		// Read file address and initial profiles from the header file
		std::ifstream input_file(filename + ".dat", std::ios::binary);
		find_line_after_header(input_file, "c\tThermal Initial Profile File");
		input_file >> N_grid >> file_address;
		input_file.close();

		// Read data from the extracted filename
		std::string data_filename = file_address;
		std::ifstream data_file(data_filename, std::ios::in | std::ios::binary);
		T_initial.resize(N_grid);
		x_initial.resize(N_grid);

		for (X = 0; X < N_grid; ++X) {
			data_file >> x_initial[X] >> T_initial[X];
			x_initial[X] *= 0.01;
		}
		data_file.close();

		// Process the data
		for (X = 0; X < MPI_parallel->dev_end[0]; ++X) {
			xx = fmod(X - MPI_parallel->start_XYZ2[0] + MPI_parallel->start_XYZ[0] + N_x, N_x) * global_parameters.D_x;

			unsigned index = 0;
			for (i = 0; i < x_initial.size() - 1; ++i) {
				if (xx > x_initial[i] && xx <= x_initial[i + 1]) {
					index = i;
					break;
				}
			}

			for (Y = 0; Y < MPI_parallel->dev_end[1]; ++Y) {
				for (Z = 0; Z < MPI_parallel->dev_end[2]; ++Z) {
					double interpolated_T = T_initial[index] + (xx - x_initial[index]) * (T_initial[index + 1] - T_initial[index]) / (x_initial[index + 1] - x_initial[index]);
					T[{X, Y, Z}] = interpolated_T / T_0;
					c_p[{X, Y, Z}] = 1.0;
					thermal_diffusion_coefficient[{X, Y, Z}] = 0.005;
					solid[{X, Y, Z}] = -1;
				}
			}
		}
		if (MPI_parallel->processor_id == (MASTER + 1)) {
			std::cout << "ENERGY: reacting TGV-3D initial conditions \n";
			std::cout << "================================ \n";
			std::cout << "Thermal file address: " << file_address << std::endl;
			std::cout << "N_grid: " << N_grid << std::endl;
			std::cout << "================================ \n";
		}
	}
}
/// ***************************************************** ///
/// INITIAL CONDITIONS: GAUSSIAN HILL                     ///
/// ***************************************************** ///
void Gaussian_thermal(Scalar_field& T, Scalar_field& c_p, Scalar_field& thermal_diffusion_coefficient, Solid_field& solid, double N_x, double N_y, double N_z, double cs_2, double T_0, double E_0, double rho_0, const std::string& filename, int Zones, Parallel_MPI* MPI_parallel) {
	///----------------------------------------------------------------------///
	///                        COMMAND LINE INPUT                            ///
	///----------------------------------------------------------------------///
	if (MPI_parallel->processor_id != MASTER) {
		if (MPI_parallel->processor_id == (MASTER + 1)) {
			std::cout << "ENERGY: Gaussian Hill conditions \n";
			std::cout << "================================ \n";
		}
		/// Open input file
		string input_filename(filename);
		input_filename += ".dat";
		ifstream input_file;  // File is open for READING
		input_file.open(input_filename.c_str(), ios::binary);
		input_file.clear();
		input_file.seekg(0, ios::beg);
		find_line_after_header(input_file, "c\tTemperature Field Gaussian Initial Conditions");
		double dim, sigma0, x_0, y_0, z_0, T_max, T_min, cp, lambda;
		find_line_after_comment(input_file);
		input_file >> dim >> sigma0 >> x_0 >> y_0 >> z_0 >> T_min >> T_max >> cp >> lambda;
		lambda = lambda * (global_parameters.D_t / sqr(global_parameters.D_x));
		T_min = T_min / T_0;
		T_max = T_max / T_0;

		input_file.close();
		for (int X = 0; X < MPI_parallel->dev_end[0]; X++) {
			double xx = fmod(X - MPI_parallel->start_XYZ2[0] + MPI_parallel->start_XYZ[0] + N_x, N_x) * global_parameters.D_x;
			for (int Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
				double yy = fmod(Y - MPI_parallel->start_XYZ2[1] + MPI_parallel->start_XYZ[1] + N_y, N_y) * global_parameters.D_x;
				for (int Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
					//	double zz = fmod(Z - MPI_parallel->start_XYZ2[2] + MPI_parallel->start_XYZ[2] + N_z, N_z) * global_parameters.D_x;
					c_p[{X, Y, Z}] = cp;
					thermal_diffusion_coefficient[{X, Y, Z}] = lambda;
					double distance = sqr(xx - x_0) + sqr(yy - y_0);
					if (dim > 2) distance += sqr(yy - z_0);
					T[{X, Y, Z}] = T_min + (T_max - T_min) * exp(-0.5 * distance / sqr(sigma0));
					solid[{X, Y, Z}] = -1;
				}
			}
		}
	}
}
/// ***************************************************** ///
/// GET THERMAL DIFFUSION USING SUTHERLAND MODEL          ///
/// ***************************************************** ///
double Sutherland_thermal_diffusion(double mu_star, double T_star, double S, double T, double Pr) {
	double lambda = 0;
	//    double S = 110.5;                    /* [S] = K */
	//    double T_star = 298.;                /* [T_star] = K */
	//    double lambda_star = 1.782e-5;       /* [lambda_star] = kg.m/s^3.K */
	//    double Pr = 0.682;

	double lambda_star = mu_star / Pr;
	lambda = lambda_star * sqrt(pow(T / (double)T_star, 3)) * (T_star + S) / (double)(T + S);
	/// lambda = lambda_star * pow(T/(double)T_star,0.69);
	return lambda;
}
void Thermal_solver::Boussinesq_force(int direction, Flow_solver* Flow, Thermal_solver* Thermal, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		for (int X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (int Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (int Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (Flow->is_solid[{X, Y, Z}] == FALSE) {
						if (direction == 0) { Flow->force[{X, Y, Z, 0}] += (1.0 - solid_particle[{X, Y, Z}]) * Thermal->gbeta * (Thermal->temperature[{X, Y, Z}] - Thermal->T_infinity); }
						if (direction == 1) { Flow->force[{X, Y, Z, 1}] += (1.0 - solid_particle[{X, Y, Z}]) * Thermal->gbeta * (Thermal->temperature[{X, Y, Z}] - Thermal->T_infinity); }
						if (direction == 2) { Flow->force[{X, Y, Z, 2}] += (1.0 - solid_particle[{X, Y, Z}]) * Thermal->gbeta * (Thermal->temperature[{X, Y, Z}] - Thermal->T_infinity); }
					}
				}
			}
		}
	}
}
double Thermal_solver::calculateThermalDiffusivity(int time, Thermal_solver* Thermal, Flow_solver* Flow, Parallel_MPI* MPI_parallel) {
	double tdiff = 0.0;
	double thermal_diffusivity = 0.0;
	if (MPI_parallel->processor_id != MASTER) {
		for (int X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (int Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (int Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (Flow->is_solid[{X, Y, Z}] == FALSE) {
						tdiff = thermal_diffusion_coefficient[{X, Y, Z}] / (c_p[{X, Y, Z}] * Flow->rho_0);
						// thermal_diffusivity += tdiff;
						std::cout << time << " Thermal Diffusivity: " << tdiff << std::endl;
						std::cout << "Diffusion Coefficient: " << thermal_diffusion_coefficient[{X, Y, Z}] << std::endl;
						std::cout << "Specific Heat Capacity: " << c_p[{X, Y, Z}] << std::endl;
						std::cout << "Density: " << Flow->rho_0 << std::endl;
					}
				}
			}
		}
	}
	return thermal_diffusivity;
}

void Thermal_solver::initialize_field_FD_TGV_temp_reactive(Scalar_field& T, Scalar_field& c_p, Scalar_field& thermal_diffusion_coefficient, Solid_field& solid, Geometry* Geo, stl_import* Geo_stl, Flow_solver* Flow, Species_solver* Species, Parallel_MPI* MPI_parallel, std::string filename, Thermo_chemistry_cantera* thermo_chemistry) {
	int X, Y, Z;
	if (MPI_parallel->processor_id != MASTER) {
		if (Geo_stl->flag == 1) {
			// Initialize the solid_thermal_type based on geometry
			for (X = 0; X < MPI_parallel->dev_end[0]; ++X) {
				for (Y = 0; Y < MPI_parallel->dev_end[1]; ++Y) {
					for (Z = 0; Z < MPI_parallel->dev_end[2]; ++Z) {
						solid_thermal_type[{X, Y, Z}] = Geo_stl->domain[{X, Y, Z}];
					}
				}
			}
		}
		for (X = 0; X < MPI_parallel->dev_end[0]; ++X) {
			for (Y = 0; Y < MPI_parallel->dev_end[1]; ++Y) {
				for (Z = 0; Z < MPI_parallel->dev_end[2]; ++Z) {
					T[{X, Y, Z}] = temperature[{X, Y, Z}] / T_0;
                    c_p[{X, Y, Z}] = c_p[{X, Y, Z}];
                    thermal_diffusion_coefficient[{X, Y, Z}] = thermal_diffusion_coefficient[{X, Y, Z}];
					solid[{X, Y, Z}] = -1;
				}
			}
		}
		if (MPI_parallel->processor_id == (MASTER + 1)) {
		/*	std::cout << "ENERGY: reacting TGV-3D initial conditions \n";
			std::cout << "================================ \n";
			std::cout << "Thermal initialization based on Cantera equilibrium \n";
			std::cout << "================================ \n";
			*/
		}
		// Initialize energy and other related fields
		for (X = 0; X < MPI_parallel->dev_end[0]; ++X) {
			for (Y = 0; Y < MPI_parallel->dev_end[1]; ++Y) {
				for (Z = 0; Z < MPI_parallel->dev_end[2]; ++Z) {
					energy[{X, Y, Z}] = 0;
					previous_temperature[{X, Y, Z}] = T[{X, Y, Z}];
					Production[{X, Y, Z}] = 0;
					thermal_diffusion_coefficient[{X, Y, Z}] = thermal_diffusion_coefficient[{X, Y, Z}] * (E_0 * Flow->rho_0 / T_0);
					solid_particle[{X, Y, Z}] = 0.0;
					initial_CP[{X, Y, Z}] = c_p[{X, Y, Z}];
					force_thermal[{X, Y, Z}] = 0;
					temp_force_thermal[{X, Y, Z}] = 0;
				}
			}
		}
	}
}