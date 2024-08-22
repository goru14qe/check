// #include "stdafx.h"
#include "Fluid_read_write.h"
#include "ALBORZ_Macros.h"
#include "ALBORZ_SETTINGS.h"
#include "ALBORZ_GlobalVariables.h"
#include <iostream>  // for the use of 'cout'
#include <cmath>
#include <fstream>    // file stream
#include <sstream>    // string streams
#include <cstdlib>    // standard library
#include <iomanip>    // For set precision. From ver 28
#include <algorithm>  // min,max

#include <cstring>
#include <stdint.h>  //to use int32_t

#include "Geometry.h"
#include "Flow_solver.h"
#include "Thermal_solver.h"
#include "Species_solver.h"
#include "Phase_Field.h"
#include "Particle_sim.h"

using namespace std;  // permanently use the standard namespace
double ini_velocity;
Fluid_read_write::Fluid_read_write() {
}
/// ***************************************************** ///
/// GET AVERAGE ENTROPY FROM FLOW SOLVER                  ///
/// ***************************************************** ///
/* This function computes the average entropy of the Flow field.
Entropy is a measure of the disorder or randomness of a system. In fluid dynamics, entropy is a measure of the amount of energy in a system that is not available to do work. Entropy is a scalar quantity that is defined as the logarithm of the number of microstates that correspond to a macrostate. In other words, entropy is a measure of the number of ways a system can be arranged.

The equation for entropy is given by:
Entropy = -Σ p_i * log(p_i)
where:
- p_i is the probability of the i-th microstate.
- log is the natural logarithm function.

The steps to calculate the entropy are as follows:
1. Calculate the entropy of the Flow field by taking the logarithm of the number of microstates that correspond to a macrostate.
2. Sum the entropy of the Flow field to get the total entropy.
3. Calculate the average entropy by dividing the total entropy by the number of cells.
4. Write the average entropy to a file.
*/
void Fluid_read_write::AverageEntropy(Flow_solver* Flow, Parallel_MPI* MPI_parallel, int time) {
	int counter = 0;
	int counter_global = 0;
	/// second-order moment

	HE = 0;
	HE_global = 0;
	unsigned int X, Y, Z, alpha;
	if (MPI_parallel->processor_id != MASTER) {
		for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					for (alpha = 0; alpha < Flow->Discrete_Velocity; alpha++) {
						HE += Flow->pop[{X, Y, Z, alpha}] * log(Flow->pop[{X, Y, Z, alpha}] / Flow->weight[alpha]);
					}
					counter++;
				}
			}
		}
	}
	MPI_Allreduce(&HE, &HE_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&counter, &counter_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	if (MPI_parallel->processor_id == MASTER) {
		/// Create filename
		stringstream output_filename;
		output_filename << "Alborz_Results/debug/AverageEntropy.dat";
		ofstream output_file;
		/// Open file
		output_file.open(output_filename.str().c_str(), fstream::app);
		/// Write data
		KE_global = HE_global / (double)counter_global;
		output_file << time << "/t";  // time step
		output_file << setprecision(30) << HE_global << endl;
		/// Close file
		output_file.close();
	}
	return;
}
/// ***************************************************** ///
/// GET AVERAGE KINETIC ENERGY FROM FLOW SOLVER           ///
/// ***************************************************** ///
/* This function computes the average kinetic energy of the Flow field.

The equation for kinetic energy is given by:
Kinetic Energy = 0.5 * u^2
where:
- u is the velocity field.

The steps to calculate the kinetic energy are as follows:
1. Calculate the square of the velocity field.
2. Sum the square of the velocity field to get the kinetic energy.
3. Calculate the average kinetic energy by dividing the kinetic energy by the number of cells.
4. Calculate the square of the kinetic energy by taking the square of the average kinetic energy.
5. Calculate the standard deviation of the kinetic energy by taking the square root of the difference between the square of the kinetic energy and the average kinetic energy.
6. Write the average kinetic energy and standard deviation to a file.
*/
void Fluid_read_write::AverageKineticEnergy(Flow_solver* Flow, Parallel_MPI* MPI_parallel, int time) {
	int counter = 0;
	int counter_global = 0;
	KE2 = 0;
	KE2_global = 0;
	KE = 0;
	KE_global = 0;
	if (MPI_parallel->processor_id != MASTER) {
		for (int X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (int Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (int Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					counter++;
					KE += 0.5 * Flow->velocity_magnitude[{X, Y, Z}] * Flow->velocity_magnitude[{X, Y, Z}];           // 0.5 * u^2
					KE2 += pow(0.5 * Flow->velocity_magnitude[{X, Y, Z}] * Flow->velocity_magnitude[{X, Y, Z}], 2);  // 0.5 * u^2
				}
			}
		}
	}
	MPI_Allreduce(&KE, &KE_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&KE2, &KE2_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&counter, &counter_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	if (MPI_parallel->processor_id == MASTER) {
		constexpr const char* output_filename = "Alborz_Results/debug/AverageKineticEnergy.dat";
		/// First check if file already exists
		const bool write_header = !std::ifstream(output_filename).is_open();
		/// Open file
		ofstream output_file(output_filename, fstream::app);
		/// Write data
		KE_global = KE_global / (double)counter_global;
		KE2_global = KE2_global / (double)counter_global;
		if (write_header) {
			output_file << setw(32) << "time"
						<< "\t" << setw(32) << "Kinetic_Energy"
						<< "\t" << setw(32) << "Standard_deviation" << endl;
		}  // time step
		output_file << setprecision(30) << fixed << Flow->physical_time << "\t";  // time step
		output_file << setprecision(30) << fixed << KE_global * global_parameters.D_x * global_parameters.D_x / (global_parameters.D_t * global_parameters.D_t) << "\t" << sqrt(KE2_global - KE_global * KE_global) * global_parameters.D_x * global_parameters.D_x / (global_parameters.D_t * global_parameters.D_t) << endl;

		output_file.close();
	}
	return;
}
/// ***************************************************** ///
/// GET AVERAGE ENSTROPHY FROM FLOW SOLVER                ///
/// ***************************************************** ///
/* This function computes the average enstrophy of the Flow field.
Enstrophy is a measure of the rotational energy (vorticity) within a fluid Flow, providing insights into turbulence and other complex Flow phenomena. Enstrophy is defined as the square of the vorticity vector, which is the curl of the velocity field.
Enstrophy: A scalar quantity analogous to kinetic energy associated with vorticity.  Defined as half the squared magnitude of the vorticity vector (ω):
This function also computes the average dissipation of the Flow field.
Disipation: The rate at which mechanical energy is converted into heat due to the viscosity of the fluid. It is a measure of the energy lost in the fluid Flow due to the internal friction of the fluid.

The equation for enstrophy is given by:
Enstrophy = 0.5 * (du/dy - dv/dx)^2 + 0.5 * (du/dz - dw/dx)^2 + 0.5 * (dv/dz - dw/dy)^2
where:
- du/dx, du/dy, du/dz, dv/dx, dv/dy, dv/dz, dw/dx, dw/dy, and dw/dz are the derivatives of the velocity field in the x, y, and z directions.
- The terms in the equation represent the different components of the enstrophy, including the rate of rotation, the rate of deformation, and the rate of strain of the fluid Flow.

The equation for dissipation is given by:
Dissipation = ν * ((du/dx)^2 + (dv/dy)^2 + (dw/dz)^2 + 2 * (du/dx + dv/dy) + 2 * (dv/dy + dw/dz) + 2 * (du/dx + dw/dz))
where:
- ν is the kinematic viscosity of the fluid.
- The terms in the equation represent the different components of the dissipation, including the rate of rotation, the rate of deformation, and the rate of strain of the fluid Flow.

The steps to calculate the enstrophy are as follows:
1. Calculate the velocity gradient tensor by taking the central difference of the velocity field.
2. Calculate the components of the velocity gradient tensor by taking the central difference of the velocity field.
3. Calculate the square of the enstrophy by taking the square of the components of the velocity gradient tensor.
4. Sum the square of the components of the velocity gradient tensor to get the enstrophy.
5. Calculate the average enstrophy by dividing the enstrophy by the number of cells.
6. Calculate the square of the enstrophy by taking the square of the average enstrophy.
7. Calculate the standard deviation of the enstrophy by taking the square root of the difference between the square of the enstrophy and the average enstrophy.
8. Write the average enstrophy and standard deviation to a file.
*/
void Fluid_read_write::AverageEnstrophy(Flow_solver* Flow, Parallel_MPI* MPI_parallel, int time) {
	int counter = 0;
	double Dissipation_temp = 0;
	double Dissipation = 0;  // temporary variable to store the dissipation of a single cell
	double Dissipation2 = 0;
	double frob_xx, frob_yy, frob_zz, frob_xy, frob_yx, frob_xz, frob_zx, frob_yz, frob_zy;  // Frobenius norm components

	if (MPI_parallel->processor_id != MASTER) {
		for (int X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (int Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (int Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					counter++;
					// Calculate Frobenius norm components of the velocity gradient tensor
					frob_xx = FD::CENTRALNONCONS(1., Flow->velocity[{X - 1, Y, Z, 0}], Flow->velocity[{X + 1, Y, Z, 0}]);
					frob_yy = FD::CENTRALNONCONS(1., Flow->velocity[{X, Y - 1, Z, 1}], Flow->velocity[{X, Y + 1, Z, 1}]);
					frob_zz = FD::CENTRALNONCONS(1., Flow->velocity[{X, Y, Z - 1, 2}], Flow->velocity[{X, Y, Z + 1, 2}]);

					frob_xy = FD::CENTRALNONCONS(1., Flow->velocity[{X - 1, Y, Z, 1}], Flow->velocity[{X + 1, Y, Z, 1}]);
					frob_yx = FD::CENTRALNONCONS(1., Flow->velocity[{X, Y - 1, Z, 0}], Flow->velocity[{X, Y + 1, Z, 0}]);
					frob_xz = FD::CENTRALNONCONS(1., Flow->velocity[{X - 1, Y, Z, 2}], Flow->velocity[{X + 1, Y, Z, 2}]);
					frob_zx = FD::CENTRALNONCONS(1., Flow->velocity[{X, Y, Z - 1, 0}], Flow->velocity[{X, Y, Z + 1, 0}]);
					frob_yz = FD::CENTRALNONCONS(1., Flow->velocity[{X, Y - 1, Z, 2}], Flow->velocity[{X, Y + 1, Z, 2}]);
					frob_zy = FD::CENTRALNONCONS(1., Flow->velocity[{X, Y, Z - 1, 1}], Flow->velocity[{X, Y, Z + 1, 1}]);

					// Compute dissipation rate using Frobenius norm components
					Dissipation_temp = sqr(frob_xx) + sqr(frob_yy) + sqr(frob_zz)
					                   + sqr(frob_xy) + sqr(frob_xz) + sqr(frob_yz)
					                   + sqr(frob_yx) + sqr(frob_zx) + sqr(frob_zy);
					Dissipation += Dissipation_temp;
					Dissipation2 += sqr(Dissipation_temp);
				}
			}
		}
	}
	// MPI reduction for global dissipation calculation
	double Dissipation_global = 0;
	double Dissipation2_global = 0;
	int counter_global = 0;
	MPI_Allreduce(&Dissipation, &Dissipation_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&Dissipation2, &Dissipation2_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&counter, &counter_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

	// Process results on MASTER processor
	if (MPI_parallel->processor_id == MASTER) {
		constexpr const char* output_filename = "Alborz_Results/debug/AverageEnstrophy.dat";
		// First check if file already exists
		const bool write_header = !std::ifstream(output_filename).is_open();
		ofstream output_file(output_filename, fstream::app);

		Dissipation_global = Dissipation_global / (double)counter_global;
		Dissipation2_global = Dissipation2_global / (double)counter_global;

		// Write data to file
		if (write_header) {
			output_file << setw(32) << "time"
						<< "\t" << setw(32) << "Dissipation"
						<< "\t" << setw(32) << "Standard_deviation" << endl;
		}
		output_file << setprecision(30) << fixed << Flow->physical_time << "\t";
		output_file << setprecision(30) << fixed << Dissipation_global / (global_parameters.D_t * global_parameters.D_t * 1000) << "\t";
		output_file << sqrt(Dissipation2_global - (Dissipation_global * Dissipation_global)) / (global_parameters.D_t * global_parameters.D_t) << endl;
		output_file.close();
	}
}
/// ***************************************************** ///
/// GET GLOBAL SIMULATION DATA FROM INPUT                 ///
/// ***************************************************** ///
/* This function reads the simulation data from the input file and stores it in the global parameters structure.
The steps to read the simulation data are as follows:
1. Open the input file for reading.
2. Read the geometry data from the input file.
3. Read the general data from the input file.
4. Read the input-output data from the input file.
5. Read the residual data from the input file.
6. Close the input file.

The simulation data includes the following parameters:
1. Geometry: The geometry of the simulation domain, which can be a bitmap image or an STL file.
2. General: The general parameters of the simulation, including the grid size and time step.
3. Input-Output Data: The input-output parameters of the simulation, including the maximum number of time steps, the frequency of data saving, and the frequency of file generation.
4. Residual Data: The residual parameters of the simulation, including the frequency of residual reporting and the residual values for Flow, Thermal, and Species.

The simulation data is stored in the global parameters structure for use in the simulation.
*/
void Fluid_read_write::get_sim_data(Geometry* Geo, stl_import* Geo_stl, std::string filename, Parallel_MPI* MPI_parallel) {
	int column_width = 40;
	string input_filename(filename);
	input_filename += ".dat";
	ifstream input_file;  // File is open for READINGMPI_parallel->num_processors
	                      /// Open file
	input_file.open(input_filename.c_str(), ios::binary);
	string Line1;
	///	char comment_indicator = 'k';
	string Geo_Source;

	find_line_after_header(input_file, "c\tGeometry");
	find_line_after_comment(input_file);
	input_file >> Geo_Source;
	if (Geo_Source.compare("bmp") == 0) {
		std::cout << Geo_Source << endl;
		Geo->flag = 1;
		Geo_stl->flag = 0;
		//		    Geo->Get_matrix_from_bmp(Geo_filename);
	}
	if (Geo_Source.compare("None") == 0) {
		Geo->flag = 0;
		Geo_stl->flag = 0;
	}
	if (Geo_Source.compare("stl") == 0) {
		Geo_stl->flag = 1;
		Geo->flag = 0;
		input_file >> Geo_stl->Source_count;
		input_file >> Geo_stl->units;
		input_file >> Geo_stl->x_center >> Geo_stl->y_center >> Geo_stl->z_center;
		find_line_after_comment(input_file);
		std::string temp;
		for (int i = 0; i < Geo_stl->Source_count; i++) {
			input_file >> temp;
			Geo_stl->Geo_filename.push_back(temp);
		}
	}
	find_line_after_header(input_file, "c\tGeneral");
	find_line_after_comment(input_file);
	double CFL_number;
	input_file >> global_parameters.D_x >> CFL_number >> ini_velocity >> global_parameters.Nx >> global_parameters.Ny >> global_parameters.Nz;
	global_parameters.D_t = (CFL_number * global_parameters.D_x) / ini_velocity;
	if (Geo->flag == 1) {
		global_parameters.Nx = Geo->w;
		global_parameters.Ny = Geo->h;
	}
	find_line_after_header(input_file, "c\tInput-Output Data");
	find_line_after_comment(input_file);
	bool use_physical_time;
	input_file >> use_physical_time;
	if (use_physical_time == 1) {
		input_file >> physical_time_cal >> t_data >> t_vtk >> t_info >> t_time >> t_recovery >> recovery_step;
		t_num = static_cast<int>(physical_time_cal / global_parameters.D_t)+10;
	} else {
		input_file >> t_num >> t_data >> t_vtk >> t_info >> t_time >> t_recovery >> recovery_step;
		physical_time_cal = t_num * global_parameters.D_t;
	}

	find_line_after_header(input_file, "c\tResidual Data");
	find_line_after_comment(input_file);
	input_file >> t_residual >> residual_flow >> residual_thermal >> residual_species;

	if (MPI_parallel->processor_id == MASTER + 1) {
		std::cout << "Simulation Parameters" << endl;
		std::cout << "=====================" << endl;
		std::cout << "Input filename :" << left << filename << endl;
		std::cout << setw(column_width) << left << "Dx = " << global_parameters.D_x << endl
				  << setw(column_width) << left << "Dt = " << global_parameters.D_t << endl
				  << setw(column_width) << left << "CFL number = " << CFL_number << endl
				  << setw(column_width) << left << "Velocity = " << ini_velocity << endl;
		std::cout << setw(column_width) << left << "Nx = " << global_parameters.Nx << endl
				  << setw(column_width) << left << "Ny = " << global_parameters.Ny << endl
				  << setw(column_width) << left << "Nz = " << global_parameters.Nz << endl
				  << setw(column_width) << left << "=====================" << endl;
		std::cout << "\nInput and Output parameters" << endl;
		std::cout << "=====================" << endl;
		std::cout << setw(column_width) << left << "Max. number of time steps = " << t_num << endl
				  << setw(column_width) << left << "Physical time = " << physical_time_cal << endl
				  << setw(column_width) << left << "Flow data save frequency = " << t_data << endl;
		std::cout << setw(column_width) << left << ".vtk file creation frequency = " << t_vtk << endl
				  << setw(column_width) << left << "On-screen data print frequency = " << t_info << endl;
		std::cout << setw(column_width) << left << "On-screen run-time print frequency = " << t_time << endl
				  << setw(column_width) << left << "Recovery file generation frequency = " << t_recovery << endl;
		std::cout << setw(column_width) << left << "start time-step = " << recovery_step << endl;
		std::cout << setw(column_width) << left << "Residual report frequency = " << t_residual << endl;
		std::cout << setw(column_width) << left << "Flow residual = " << residual_flow << endl;
		std::cout << setw(column_width) << left << "Thermal residual = " << residual_thermal << endl;
		std::cout << setw(column_width) << left << "Species residual = " << residual_species << endl;
		std::cout << "=====================" << endl;
	}
	input_file.close();
}  //=============================================================================================================================
/// ***************************************************** ///
/// AUTOMATIC INPUT GENERATION (FOR SHORT QUEUES)         ///
/// ***************************************************** ///
/* This function automatically generates the input file for the simulation.
The steps to generate the input file are as follows:
1. Open the input file for writing.
2. Write the geometry data to the input file.
3. Write the general data to the input file.
4. Write the input-output data to the input file.
5. Write the residual data to the input file.
6. Close the input file.
*/
void Fluid_read_write::automatic_input_file_update_for_Neumann(std::string filename, Parallel_MPI* MPI_parallel) {
	MPI_Barrier(MPI_COMM_WORLD);
	if (MPI_parallel->processor_id == MASTER) {
		string old_filename(filename);
		string new_filename(filename);
		old_filename += ".dat";
		new_filename += "_temp.dat";
		ifstream old_file;  // File is open for READING
		ofstream new_file;  // File is open for READING
		old_file.open(old_filename.c_str(), ios::binary);
		new_file.open(new_filename.c_str(), ios::binary);
		string Line;
		int indicator = -1;
		string header = "c\tInput-Output Data";
		while (std::getline(old_file, Line)) {
			if (indicator == 1 && Line.at(0) != '#') {
				indicator = -1;
				new_file << t_num + t_recovery << "\t"
						 << t_data << "\t"
						 << t_vtk << "\t"
						 << t_info << "\t"
						 << t_time << "\t"
						 << t_recovery << "\t"
						 << t_num - 2 << "\n";
				Line = "#" + Line;
			}
			if (Line.find(header) != std::string::npos) {
				indicator = 1;
			}
			new_file << Line << std::endl;
		}
		old_file.close();
		new_file.close();
		if (remove(old_filename.c_str()) != 0) perror("Input file does not exist!");
		rename(new_filename.c_str(), old_filename.c_str());
	}
	MPI_Barrier(MPI_COMM_WORLD);
}

Fluid_read_write::~Fluid_read_write() {
}
/// ***************************************************** ///
/// COMPUTE MAXIMUM CFL IN DOMAIN                         ///
/// ***************************************************** ///
/* This function computes the maximum Courant-Friedrichs-Lewy (CFL) number in the domain.
The CFL number is a dimensionless number that is used in fluid dynamics to quantify the stability of a numerical solution to a partial differential equation. It is defined as the ratio of the speed of a wave through a medium to the speed of the medium itself. The CFL number is used to determine the time step size in a numerical simulation, with a smaller CFL number indicating a more stable solution.
The steps to calculate the maximum CFL number are as follows:
1. Calculate the maximum velocity in the domain by taking the maximum of the velocity field.
2. Calculate the maximum speed of sound in the domain by taking the maximum of the speed of sound field.
3. Calculate the maximum CFL number by taking the maximum of the ratio of the maximum velocity to the maximum speed of sound.
4. Write the maximum CFL number to a file.
*/
void CFL_monitor(Flow_solver* Flow, Parallel_MPI* MPI_parallel, unsigned int time) {
	double u_max = -1;
	double u_min = 1e16;
	double u_max_global, u_min_global;
	double u_mag;
	if (MPI_parallel->processor_id != MASTER) {
		for (int X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (int Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (int Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (Flow->is_solid[{X, Y, Z}] == FALSE) {
						u_mag = sqrt(pow(Flow->velocity[{X, Y, Z, 0}], 2) + pow(Flow->velocity[{X, Y, Z, 1}], 2) + pow(Flow->velocity[{X, Y, Z, 2}], 2));
#if defined Shan_Chen || defined Kupershtokh
						u_mag = sqrt(pow(Flow->velocity[{X, Y, Z, 0}] + .5 * Flow->temp_force[{X, Y, Z, 0}] / Flow->density[{X, Y, Z}], 2)
						             + pow(Flow->velocity[{X, Y, Z, 1}] + .5 * Flow->temp_force[{X, Y, Z, 1}] / Flow->density[{X, Y, Z}], 2)
						             + pow(Flow->velocity[{X, Y, Z, 2}] + .5 * Flow->temp_force[{X, Y, Z, 2}] / Flow->density[{X, Y, Z}], 2));
#endif
						u_max = MAX(u_max, u_mag);
						u_min = MIN(u_min, u_mag);
					}
				}
			}
		}
	}
	MPI_Allreduce(&u_max, &u_max_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(&u_min, &u_min_global, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
	if (MPI_parallel->processor_id == MASTER) {
		stringstream output_filename;
		output_filename << "Alborz_Results/debug/CFLMinMax.dat";
		ofstream output_file;
		output_file.open(output_filename.str().c_str(), fstream::app);
		output_file << time << "\t";  // time step
		output_file << setprecision(30) << u_min_global << "\t" << u_max_global << endl;
		output_file.close();
	}
	return;
}
/// ***************************************************** ///
/// GET MIN/MAX NON_DIMENSIONAL VISCOSITY                 ///
/// ***************************************************** ///
/* This function computes the minimum and maximum non-dimensional viscosity in the domain.
Viscosity is a measure of the resistance of a fluid to Flow. It is a property of the fluid that determines how easily it can be deformed or moved. Viscosity is a scalar quantity that is defined as the ratio of the shear stress to the rate of deformation in a fluid. In fluid dynamics, viscosity is an important parameter that affects the Flow behavior of a fluid.
The steps to calculate the minimum and maximum non-dimensional viscosity are as follows:
1. Calculate the non-dimensional viscosity of the Flow field by taking the product of the square of the speed of sound and the viscosity field.
2. Calculate the minimum and maximum non-dimensional viscosity by taking the minimum and maximum of the non-dimensional viscosity field.
3. Write the minimum and maximum non-dimensional viscosity to a file.
*/
void Viscosity_monitor(Flow_solver* Flow, Thermal_solver* Thermal, Species_solver* Species, Parallel_MPI* MPI_parallel, int time) {
	double nu_max = -1;
	double nu_min = 1e16;
	double nu_max_global, nu_min_global;

	if (MPI_parallel->processor_id != MASTER) {
		// double conv_factor = Flow->c_s2 * Thermal->T_0 / sqr(global_parameters.D_x / global_parameters.D_t);
		for (int X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (int Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (int Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (Flow->is_solid[{X, Y, Z}] == FALSE) {
						// r = R_GAS / Flow->M_av;
#if defined Flow_With_Species
						// r = R_GAS / Species->molar_mass_av[{X, Y, Z}];
#endif  // defined
        // theta = r * Thermal->temperature[{X, Y, Z}] * conv_factor;
						nu_max = std::max(nu_max, Flow->c_s2 * Flow->viscosity[{X, Y, Z}]);
						nu_min = std::min(nu_min, Flow->c_s2 * Flow->viscosity[{X, Y, Z}]);
					}
				}
			}
		}
	}
	MPI_Allreduce(&nu_max, &nu_max_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(&nu_min, &nu_min_global, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
	if (MPI_parallel->processor_id == MASTER) {
		/// Create filename
		stringstream output_filename;
		output_filename << "Alborz_Results/debug/ViscosityMinMax.dat";
		ofstream output_file;
		/// Open file
		output_file.open(output_filename.str().c_str(), fstream::app);
		/// Write data
		output_file << time << "\t";  // time step
		output_file << setprecision(30) << nu_min_global << "\t" << nu_max_global << endl;
		/// Close file
		output_file.close();
	}
	return;
}
/// ***************************************************** ///
/// GET MIN/MAX NON_DIMENSIONAL THERMAL DIFFUSIVITY       ///
/// ***************************************************** ///
/* This function computes the minimum and maximum non-dimensional Thermal diffusivity in the domain.
Thermal diffusivity is a measure of the rate at which heat is transferred through a material. It is a property of the material that determines how quickly heat can move through it. Thermal diffusivity is a scalar quantity that is defined as the ratio of the Thermal conductivity to the product of the density and specific heat capacity of the material. In fluid dynamics, Thermal diffusivity is an important parameter that affects the heat transfer behavior of a fluid.
The steps to calculate the minimum and maximum non-dimensional Thermal diffusivity are as follows:
1. Calculate the non-dimensional Thermal diffusivity of the Flow field by taking the product of the square of the speed of sound and the Thermal diffusivity field.
2. Calculate the minimum and maximum non-dimensional Thermal diffusivity by taking the minimum and maximum of the non-dimensional Thermal diffusivity field.
3. Write the minimum and maximum non-dimensional Thermal diffusivity to a file.
*/
void Thermal_diffusion_monitor(Flow_solver* Flow, Thermal_solver* Thermal, Parallel_MPI* MPI_parallel, int time) {
	double alpha_max = -1;
	double alpha_min = 1e16;
	double alpha_max_global, alpha_min_global;
	if (MPI_parallel->processor_id != MASTER) {
		for (int X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (int Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (int Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (Flow->is_solid[{X, Y, Z}] == FALSE) {
						alpha_max = std::max(alpha_max, Thermal->thermal_diffusion_coefficient[{X, Y, Z}] / (Thermal->c_p[{X, Y, Z}] * Flow->density[{X, Y, Z}] * Flow->rho_0));
						alpha_min = std::min(alpha_min, Thermal->thermal_diffusion_coefficient[{X, Y, Z}] / (Thermal->c_p[{X, Y, Z}] * Flow->density[{X, Y, Z}] * Flow->rho_0));
					}
				}
			}
		}
	}
	MPI_Allreduce(&alpha_max, &alpha_max_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(&alpha_min, &alpha_min_global, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
	if (MPI_parallel->processor_id == MASTER) {
		/// Create filename
		stringstream output_filename;
		output_filename << "Alborz_Results/debug/ThermalDiffusivityMinMax.dat";
		ofstream output_file;
		/// Open file
		output_file.open(output_filename.str().c_str(), fstream::app);
		/// Write data
		output_file << time << "\t";  // time step
		output_file << setprecision(30) << alpha_min_global << "\t" << alpha_max_global << endl;
		/// Close file
		output_file.close();
	}
	return;
}
/// ***************************************************** ///
/// GET MIN/MAX NON_DIMENSIONAL SPECIES DIFFUSION         ///
/// ***************************************************** ///
/* This function computes the minimum and maximum non-dimensional Species diffusion in the domain.
Species diffusion is a measure of the rate at which Species are transferred through a material. It is a property of the material that determines how quickly Species can move through it. Species diffusion is a scalar quantity that is defined as the ratio of the Species diffusion coefficient to the product of the density and specific heat capacity of the material. In fluid dynamics, Species diffusion is an important parameter that affects the Species transfer behavior of a fluid.
The steps to calculate the minimum and maximum non-dimensional Species diffusion are as follows:
1. Calculate the non-dimensional Species diffusion of the Flow field by taking the product of the square of the speed of sound and the Species diffusion field.
2. Calculate the minimum and maximum non-dimensional Species diffusion by taking the minimum and maximum of the non-dimensional Species diffusion field.
3. Write the minimum and maximum non-dimensional Species diffusion to a file.
*/
void Diffusion_species_monitor(Species_solver* Species, Flow_solver* Flow, Parallel_MPI* MPI_parallel, int time) {
	double D_max = -1;
	double D_min = 1e16;
	double D_max_global, D_min_global;
	if (MPI_parallel->processor_id != MASTER) {
		int X, Y, Z, k;
		for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (Flow->is_solid[{X, Y, Z}] == FALSE) {
						for (k = 0; k < Species->Nb_spec; k++) {
							D_max = std::max(D_max, Species->diffusion_coefficient[{X, Y, Z, k}] / Flow->density[{X, Y, Z}]);
							D_min = std::min(D_min, Species->diffusion_coefficient[{X, Y, Z, k}] / Flow->density[{X, Y, Z}]);
						}
					}
				}
			}
		}
	}
	MPI_Allreduce(&D_max, &D_max_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(&D_min, &D_min_global, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
	if (MPI_parallel->processor_id == MASTER) {
		/// Create filename
		stringstream output_filename;
		output_filename << "Alborz_Results/debug/SpeciesDiffusionMinMax.dat";
		ofstream output_file;
		/// Open file
		output_file.open(output_filename.str().c_str(), fstream::app);
		/// Write data
		output_file << time << "\t";  // time step
		output_file << setprecision(30) << D_min_global << "\t" << D_max_global << endl;
		/// Close file
		output_file.close();
	}
	return;
}

void Check_Mass_Fraction_Conservation(int time, Species_solver* Species, Parallel_MPI* MPI_parallel) {
	double total_mass_fraction_sum = 0.0;
    double total_mass_fraction_sum_global;
    int local_cell_count = 0;
    int global_cell_count;
    
    if (MPI_parallel->processor_id != MASTER) {
        int X, Y, Z, k;
        for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
            for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
                for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
                    if (!Species->solid_species[{X, Y, Z}]) {
                        double cell_sum = 0.0;
                        for (k = 0; k < Species->Nb_spec; ++k) {
                            cell_sum += Species->mass_fraction[{X, Y, Z, k}];
                        }
                        total_mass_fraction_sum += cell_sum;
                        local_cell_count++;
                    }
                }
            }
        }
    }

    // Reduce the total mass fraction sum and count across all processors
    MPI_Allreduce(&total_mass_fraction_sum, &total_mass_fraction_sum_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_cell_count, &global_cell_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (MPI_parallel->processor_id == MASTER) {
        double average_mass_fraction = (global_cell_count > 0) ? (total_mass_fraction_sum_global / global_cell_count) : 0.0;
        std::stringstream output_filename;
        output_filename << "Alborz_Results/debug/mass_fraction_conservation.dat";
        std::ofstream output_file;
        output_file.open(output_filename.str().c_str(), std::fstream::app);
        output_file << time << "\t";  // Time step
        output_file << std::setprecision(30) << average_mass_fraction << std::endl;
        output_file.close();
    }
}
/// ***************************************************** ///
/// GET MIN/MAX NON_DIMENSIONAL TEMPERATURE               ///
/// ***************************************************** ///
/* This function computes the minimum and maximum non-dimensional temperature in the domain.
Here theta is the non-dimensional temperature, T is the temperature field, and T_0 is the reference temperature.
M_min and M_max are the minimum and maximum molar masses of the Species in the domain, respectively.
The steps to calculate the minimum and maximum non-dimensional temperature are as follows:
1. Calculate the non-dimensional temperature of the Flow field by taking the product of the temperature field and the temperature scale.
2. Calculate the minimum and maximum non-dimensional temperature by taking the minimum and maximum of the non-dimensional temperature field.
3. Write the minimum and maximum non-dimensional temperature to a file.
*/
void temperature_monitor(Flow_solver* Flow, Thermal_solver* Thermal, Species_solver* Species, Parallel_MPI* MPI_parallel, int time) {
	double T_max = -1e16;
	double T_min = 1e16;
	double theta_max = -1e16;
	double theta_min = 1e16;
	double M_max = -1e16;
	double M_min = 1e16;
	double M_av;
	double T_max_global, T_min_global, theta_max_global, theta_min_global, M_max_global, M_min_global;
	double conv_factor, theta, r;

	if (MPI_parallel->processor_id != MASTER) {
		conv_factor = Flow->c_s2 * Thermal->T_0 / sqr(global_parameters.D_x / global_parameters.D_t);
		for (int X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (int Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (int Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (Flow->is_solid[{X, Y, Z}] == FALSE) {
						r = R_GAS / Flow->M_av;
						M_av = Flow->M_av;
#if defined Flow_With_Species
						r = R_GAS / Species->molar_mass_av[{X, Y, Z}];
						M_av = Species->molar_mass_av[{X, Y, Z}];
#endif  // defined
						theta = r * Thermal->temperature[{X, Y, Z}] * conv_factor;
						theta_max = std::max(theta_max, theta);
						theta_min = std::min(theta_min, theta);
						T_max = std::max(T_max, Thermal->temperature[{X, Y, Z}] * Thermal->T_0);
						T_min = std::min(T_min, Thermal->temperature[{X, Y, Z}] * Thermal->T_0);
						M_max = std::max(M_max, M_av);
						M_min = std::min(M_min, M_av);
					}
				}
			}
		}
	}
	MPI_Allreduce(&theta_max, &theta_max_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(&theta_min, &theta_min_global, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
	MPI_Allreduce(&T_max, &T_max_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(&T_min, &T_min_global, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
	MPI_Allreduce(&M_max, &M_max_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(&M_min, &M_min_global, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
	if (MPI_parallel->processor_id == MASTER) {
		/// Create filename
		stringstream output_filename;
		output_filename << "Alborz_Results/debug/TMinMax.dat";
		ofstream output_file;
		/// Open file
		output_file.open(output_filename.str().c_str(), fstream::app);
		if (output_file.tellp() == 0) {
			// output_file << "Time_Step\tPhysical_Time\tAvg_Pressure\tAvg_Density\tAvg_Velocity" << std::endl;
			output_file << "time_step\tTheta_min\tTheta_max\tT_min\tT_max\tM_min\tM_max" << std::endl;
		}
		/// Write data
		output_file << time << "\t";  // time step
		output_file << setprecision(30) << theta_min_global << "\t" << theta_max_global << "\t"
					<< T_min_global << "\t" << T_max_global << "\t"
					<< M_min_global << "\t" << M_max_global << endl;
		// here theta_min_global and theta_max_global are the non-dimensional temperature
		//  and T_min_global and T_max_global are the dimensional temperature
		//  and M_min_global and M_max_global are the molar masses of the Species
		/// Close file
		output_file.close();
	}
	return;
}
/// ***************************************************** ///
/// GET MIN/MAX TOTAL ENERGY                              ///
/// ***************************************************** ///
/* This function computes the minimum and maximum total energy in the domain.
Total energy is a measure of the sum of the internal energy and the kinetic energy of a fluid. It is a property of the fluid that determines the amount of energy that is present in the fluid. Total energy is a scalar quantity that is defined as the sum of the internal energy and the kinetic energy of the fluid. In fluid dynamics, total energy is an important parameter that affects the Flow behavior of a fluid.
The steps to calculate the minimum and maximum total energy are as follows:
1. Calculate the total energy of the Flow field by taking the sum of the internal energy and the kinetic energy.
2. Calculate the minimum and maximum total energy by taking the minimum and maximum of the total energy field.
3. Write the minimum and maximum total energy to a file.
*/
void Energy_monitor(Flow_solver* Flow, Thermal_solver* Thermal, Species_solver* Species, Parallel_MPI* MPI_parallel, int time) {
	double E = 0;
	double E_global = 0;
	double conv_factor, theta, r;
	if (MPI_parallel->processor_id != MASTER) {
		conv_factor = Flow->c_s2 * Thermal->T_0 / sqr(global_parameters.D_x / global_parameters.D_t);
		for (int X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (int Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (int Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (Flow->is_solid[{X, Y, Z}] == FALSE) {
						r = R_GAS / Flow->M_av;
#if defined Flow_With_Species
						r = R_GAS / Species->molar_mass_av[{X, Y, Z}];
#endif  // defined
						theta = r * Thermal->temperature[{X, Y, Z}] * conv_factor;
						E += (Flow->velocity_magnitude[{X, Y, Z}] * Flow->velocity_magnitude[{X, Y, Z}] + sqr(Flow->density[{X, Y, Z}] - 1.) * theta / Flow->c_s2);
					}
				}
			}
		}
	}
	MPI_Allreduce(&E, &E_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	if (MPI_parallel->processor_id == MASTER) {
		/// Create filename
		stringstream output_filename;
		output_filename << "Alborz_Results/debug/Energy.dat";
		ofstream output_file;
		/// Open file
		output_file.open(output_filename.str().c_str(), fstream::app);
		/// Write data
		output_file << time << "\t";  // time step
		output_file << setprecision(30) << E_global * global_parameters.D_x * global_parameters.D_x / (global_parameters.D_t * global_parameters.D_t) << endl;
		/// Close file
		output_file.close();
	}
	return;
}
/// ***************************************************** ///
/// CHECK CONVERGENCE OF L2 RESIDUAL                      ///
/// ***************************************************** ///
/* This function checks the convergence of the L2 residual in the domain.
The L2 residual is a measure of the difference between the current solution and the previous solution in the domain. It is a scalar quantity that is defined as the sum of the square of the difference between the current solution and the previous solution. The L2 residual is used to determine the convergence of a numerical solution to a partial differential equation. If the L2 residual is below a certain threshold, the solution is considered to be converged.
The steps to check the convergence of the L2 residual are as follows:
1. Calculate the L2 residual of the all the functions by taking the sum of the square of the difference between the current solution and the previous solution.
2. Calculate the criteria of the L2 residual by taking the sum of the L2 residual divided by the sum of the current solution.
3. Write the criteria of the L2 residual to a file.
4. Check if the criteria of the L2 residual is below a certain threshold.
5. Return true if the criteria of the L2 residual is below the threshold, otherwise return false.
*/
bool check_L2_residual(const Scalar_field& field, const Scalar_field& field_old, const Solid_field& solid, double threshold, const Parallel_MPI& MPI_parallel, int time, const std::string& field_name) {
	double st_cr;
	double criteria = 0;
	double criteria_2 = 0;
	double global, global_2;

	if (MPI_parallel.processor_id != MASTER) {
		for (int X = MPI_parallel.start_XYZ2[0]; X <= MPI_parallel.end_XYZ2[0]; ++X) {
			for (int Y = MPI_parallel.start_XYZ2[1]; Y <= MPI_parallel.end_XYZ2[1]; ++Y) {
				for (int Z = MPI_parallel.start_XYZ2[2]; Z <= MPI_parallel.end_XYZ2[2]; ++Z) {
					if (solid[{X, Y, Z}] == FALSE) {
						criteria += abs(field[{X, Y, Z}] - field_old[{X, Y, Z}]);
						criteria_2 += field[{X, Y, Z}];
					}
				}
			}
		}
	}
	MPI_Allreduce(&criteria, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&criteria_2, &global_2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	st_cr = global / (global_2 + 1e-10);
	if (MPI_parallel.processor_id == MASTER) {
		std::stringstream output_filename;
		output_filename << "Alborz_Results/debug/convergence_L2_" << field_name << ".dat";
		std::ofstream output_file;
		output_file.open(output_filename.str().c_str(), std::fstream::app);
		output_file << std::setprecision(30) << time << "\t";  // time step
		output_file << std::setprecision(30) << st_cr << std::endl;
		output_file.close();
	}
	double residual = st_cr;
	if (residual <= threshold && time > 10) {
		return true;
	}
	if (residual > threshold || time < 11) {
		return false;
	}
	return residual <= threshold && time > 10;
}
/* This function integrates the domain of a scalar field over time.
The integral is calculated by summing the values of the scalar field over the domain.
The steps to integrate the domain are as follows:
1. Calculate the integral of the scalar field by summing the values of the scalar field over the domain.
2. Write the integral of the scalar field to a file.
*/
void integrate_domain(const Scalar_field& field, const Solid_field& solid, const Parallel_MPI& MPI_parallel, int time, std::string keyword) {
	double integral = 0;
	double global_integral = 0;
	if (MPI_parallel.processor_id != MASTER) {
		for (int X = MPI_parallel.start_XYZ2[0]; X <= MPI_parallel.end_XYZ2[0]; ++X) {
			for (int Y = MPI_parallel.start_XYZ2[1]; Y <= MPI_parallel.end_XYZ2[1]; ++Y) {
				for (int Z = MPI_parallel.start_XYZ2[2]; Z <= MPI_parallel.end_XYZ2[2]; ++Z) {
					if (solid[{X, Y, Z}] == FALSE) {
						integral += field[{X, Y, Z}];
					}
				}
			}
		}
	}
	MPI_Allreduce(&integral, &global_integral, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	if (MPI_parallel.processor_id == MASTER) {
		stringstream output_filename;
		output_filename << "Alborz_Results/debug/integral" << keyword << ".dat";
		ofstream output_file;
		output_file.open(output_filename.str().c_str(), fstream::app);
		output_file << setprecision(30) << time << "\t";  // time step
		output_file << setprecision(30) << global_integral << endl;
		output_file.close();
	}
}

/* This function reads the velocity at the monitoring points and writes them to a file.
The monitoring points are specified by the user in the input file.
The steps to read the velocity at the monitoring points are as follows:
1. Read the velocity at the monitoring points from the velocity field.
2. Write the velocity at the monitoring points to a file.
*/
void DataSampling::Save_VelPoint(std::string filename, stl_import* Geo_stl, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		string input_filename(filename);
		input_filename += ".dat";
		ifstream input_file;
		input_file.open(input_filename.c_str(), ios::binary);
		find_line_after_header(input_file, "c\tMonitoring Points");
		find_line_after_comment(input_file);
		/// First read how many points
		double frequency_temp;
		input_file >> Point_number >> frequency_temp;
		frequency = int(frequency_temp / global_parameters.D_t);
		/* resize arrays holding coordinate data */
		std::vector<double> P_x(Point_number, 0);
		std::vector<double> P_y(Point_number, 0);
		std::vector<double> P_z(Point_number, 0);
		/* read list of coordinates from files */
		for (int i = 0; i < Point_number; i++) {
			input_file >> P_x[i] >> P_y[i] >> P_z[i];
		}
		/* close text file */
		input_file.close();
		for (int X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (int Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (int Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					for (int i = 0; i < Point_number; i++) {
						double xc, yc, zc;
						input_file >> Geo_stl->x_center >> Geo_stl->y_center >> Geo_stl->z_center;
						MPI_parallel->get_coordinates(X, Y, Z, Geo_stl->x_center, Geo_stl->y_center, Geo_stl->z_center, xc, yc, zc);
						int xx = xc;
						int yy = yc;
						int zz = zc;
						if (xx == int(P_x[i]) && yy == int(P_y[i]) && zz == int(P_z[i])) {  // is_solid[X][Y][Z]==FALSE &&
							                                                                // std::cout << "scanned point: " << xx << "\t" << yy << "\t" << zz << " procs: " << MPI_parallel->processor_id << "\n";
							X_monitor.push_back(X);
							Y_monitor.push_back(Y);
							Z_monitor.push_back(Z);
							index_monitor.push_back(i + 1);
							std::cout << MPI_parallel->processor_id << "\t" << X << "\t" << Y << "\t" << Z << "\n";
							// cout << X_0[i] << "\t"<<  Y_0[i] << "\t" <<  Z_0[i] <<endl;
						} /* end of if is_solid */
					} /* end of loop over points*/
				} /* end of loop over Z */
			} /* end of loop over Y */
		} /* end of loop over X */
		// std::cout << index_monitor.size() << ", processor id = " << processor_id << "\n";
	} /* end of processor_id if */
} /* end of function */

/* This function reads the velocity at the monitoring points and writes them to a file.
The monitoring points are specified by the user in the input file.
The steps to read the velocity at the monitoring points are as follows:
1. Read the velocity at the monitoring points from the velocity field.
2. Write the velocity at the monitoring points to a file.
*/
void DataSampling::Out_VelPoint(const Scalar_field& field, stl_import* Geo_stl, Parallel_MPI* MPI_parallel, int& t) {
	if (MPI_parallel->processor_id != MASTER) {  //&& t%frequency==0 ) {
		for (int i = 0; i < index_monitor.size(); i++) {
			int Xmonitor = X_monitor[i];
			int Ymonitor = Y_monitor[i];
			int Zmonitor = Z_monitor[i];
			stringstream output_filename;
			output_filename << "Alborz_Results//debug//Point"
							<< "_" << index_monitor[i] << ".dat";
			ofstream output_file;
			output_file.open(output_filename.str().c_str(), fstream::app);
			output_file << std::setprecision(10) << t * global_parameters.D_t << "\t";
			output_file << std::setprecision(10) << field[{Xmonitor, Ymonitor, Zmonitor}] << "\n";  // Corrected line
			output_file.close();
		} /* end of loop over points*/
	}
}