#include <sstream>   // string streams
#include <iostream>  // for the use of 'cout'
#include <fstream>   // file stream
#include <iomanip>
#include <ctype.h>
#include <cmath>
#include <string.h>
#include <vector>
#include <stdio.h>
#include "Fluid_read_write.h"

#include "Species_solver.h"
#include "SectionalSolver.h"
#include "ALBORZ_Macros.h"
#include "ALBORZ_SETTINGS.h"
#include "ALBORZ_GlobalVariables.h"

#include "Flow_solver.h"
#include "Thermal_solver.h"
#include "Geometry.h"
using namespace std;

Sectional_solver::Sectional_solver() {
}
void Sectional_solver::General_data_input(std::string filename, Parallel_MPI* MPI_parallel) {
	int column_width = 40;
	/// Open input file
	string input_filename(filename);
	input_filename += ".dat";
	ifstream input_file;  // File is open for READING
	input_file.open(input_filename.c_str(), ios::binary);

	input_file.clear();
	input_file.seekg(0, ios::beg);
	find_line_after_header(input_file, "c\tSectional Field Solver");
	find_line_after_comment(input_file);
	input_file >> Dimension >> Discrete_Velocity >> Nb_sections;
	Min_radius.resize(Nb_sections);
	Max_radius.resize(Nb_sections);
	for (unsigned int k = 0; k < Nb_sections; ++k) {
		find_line_after_comment(input_file);
		input_file >> Min_radius[k];
		input_file >> Max_radius[k];
	}
	input_file.close();
	if (MPI_parallel->processor_id == MASTER + 1) {
		std::cout << "Sectional field parameters" << endl;
		std::cout << "=====================" << endl;
		std::cout << "Stencil : " << left << "D" << Dimension << "Q" << Discrete_Velocity << endl;
		std::cout << setw(column_width) << left << "Number of Sections : " << Nb_sections << endl;
		for (unsigned int k = 0; k < Nb_sections; ++k) {
			std::cout << Min_radius[k] << "\t" << Max_radius[k] << "\n";
		}
		std::cout << endl;
	}
}
void Sectional_solver::Memory_allocation_FD(Stencil_Definition Stencil_Def, Parallel_MPI* MPI_parallel) {
	unsigned int X, Y, Z;
	/// -------------------------------------------------------------------------------------------
	Stencil_Def(Dimension, Discrete_Velocity, weight, c_alpha, alpha_bar, c_s2);
	weight_2.resize(Discrete_Velocity);
	weight_2[0] = 0;
	for (unsigned int alpha = 1; alpha < Discrete_Velocity; alpha++) {
		weight_2[alpha] = weight[alpha];
		weight_2[0] = weight_2[0] - weight_2[alpha];
	}
	if (MPI_parallel->processor_id != MASTER) {
		Y_section = new double***[MPI_parallel->dev_end[0]];
		previous_Y_section = new double***[MPI_parallel->dev_end[0]];
		Production = new double***[MPI_parallel->dev_end[0]];
		solid_sectional = new int**[MPI_parallel->dev_end[0]];
		for (X = 0; X < MPI_parallel->dev_end[0]; ++X) {
			Y_section[X] = new double**[MPI_parallel->dev_end[1]];
			previous_Y_section[X] = new double**[MPI_parallel->dev_end[1]];
			Production[X] = new double**[MPI_parallel->dev_end[1]];
			solid_sectional[X] = new int*[MPI_parallel->dev_end[1]];
			for (Y = 0; Y < MPI_parallel->dev_end[1]; ++Y) {
				Y_section[X][Y] = new double*[MPI_parallel->dev_end[2]];
				previous_Y_section[X][Y] = new double*[MPI_parallel->dev_end[2]];
				Production[X][Y] = new double*[MPI_parallel->dev_end[2]];
				solid_sectional[X][Y] = new int[MPI_parallel->dev_end[2]];
				for (Z = 0; Z < MPI_parallel->dev_end[2]; ++Z) {
					Y_section[X][Y][Z] = new double[Nb_sections];
					previous_Y_section[X][Y][Z] = new double[Nb_sections];
					Production[X][Y][Z] = new double[Nb_sections];
				}
			}
		}
	}
}
void Sectional_solver::initialize_field_FD(Geometry* Geo, stl_import* Geo_stl, Sectional_Ini Ini_Sections, Flow_solver* Flow, Thermal_solver* Thermal, Parallel_MPI* MPI_parallel, std::string filename) {
	unsigned int X, Y, Z, k;
	if (MPI_parallel->processor_id != MASTER) {
		if (Geo_stl->flag == 1) {
			for (X = 0; X < MPI_parallel->dev_end[0]; X++) {
				for (Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
					for (Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
						solid_sectional[X][Y][Z] = Geo_stl->domain[{X,Y,Z}];
					}
				}
			}
		}
		Ini_Sections(Y_section, solid_sectional, Min_radius,
		             Max_radius, MPI_parallel->dev_end[0], MPI_parallel->dev_end[1], MPI_parallel->dev_end[2], Nb_sections,
		             filename, Geo_stl->Source_count, MPI_parallel);
		/// -------------------------------------------------------------------------------------------
		// unsigned int xx, yy, zz;
		for (X = 0; X < MPI_parallel->dev_end[0]; X++) {
			// xx = fmod(X - MPI_parallel->start_XYZ2[0] + MPI_parallel->start_XYZ[0] + global_parameters.Nx + 1, global_parameters.Nx);
			for (Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
				// yy = fmod(Y - MPI_parallel->start_XYZ2[1] + MPI_parallel->start_XYZ[1] + global_parameters.Ny, global_parameters.Ny);
				for (Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
					// zz = fmod(Z - MPI_parallel->start_XYZ2[2] + MPI_parallel->start_XYZ[2] + global_parameters.Nz, global_parameters.Nz);
					for (k = 0; k < Nb_sections; ++k) {
						Production[X][Y][Z][k] = 0.;
						previous_Y_section[X][Y][Z][k] = Y_section[X][Y][Z][k];
					}
				}
			}
		}
	}
	return;
}

void Sectional_solver::initialize_BC(Geometry* Geo, stl_import* Geo_stl, Parallel_MPI* MPI_parallel, std::string filename) {
	if (MPI_parallel->processor_id != MASTER) {
		unsigned int X, Y, Z, i, k, j, alpha;
		int number_of_BC = 0;
		std::vector<sectional_boundary_data> temp;
		std::vector<int> index;
		int** intersection;
		/// Open input file
		string input_filename(filename);
		input_filename += ".dat";
		ifstream input_file;  // File is open for READING
		input_file.open(input_filename.c_str(), ios::binary);

		input_file.clear();
		input_file.seekg(0, ios::beg);
		find_line_after_header(input_file, "c\tSectional Field Boundary Conditions");
		find_line_after_comment(input_file);
		input_file >> number_of_BC;
		temp.resize(number_of_BC);
		index.resize(number_of_BC);
		intersection = new int*[number_of_BC];

		for (i = 0; i < number_of_BC; i++) {
			intersection[i] = new int[2];
			find_line_after_comment(input_file);
			input_file >> index[i] >> intersection[i][0] >> intersection[i][1] >> temp[i].type;
			temp[i].Y_section.resize(Nb_sections);
			switch (temp[i].type) {
				case 102:  /// ---> Non-zero fraction BC
				{
					for (k = 0; k < Nb_sections; k++) {
						input_file >> temp[i].Y_section[k];
					}
					break;
				}
				case 104:  /// --> zero flux (1st-order) BC
				{
					break;
				}
				default:  /// --> Boundary not defined
				{
					std::cout << " Error : undefined Species boundary type \n";
					break;
				}
			}
		}
		if (MPI_parallel->processor_id == (MASTER + 1)) {
			stringstream output_filename;
			output_filename << "Sectional_Boundary_Conditions.dat";
			ofstream BC_output;
			BC_output.open(output_filename.str().c_str(), fstream::trunc);
			BC_output << "SECTIONS: Boundary Conditions \n";
			BC_output << "================================ \n";
			for (i = 0; i < number_of_BC; i++) {
				BC_output << "BC : " << i << "\t Type : " << temp[i].type << std::endl;
				BC_output << "Fraction : " << std::endl;
				for (k = 0; k < Nb_sections; ++k) {
					BC_output << Min_radius[k] << "-" << Max_radius[k] << " :\t" << temp[i].Y_section[k] << "\n";
				}
				BC_output << endl;
				BC_output << "-------------------------------- \n";
			}
			if (number_of_BC == 0)
				BC_output << "NO SECTIONAL BOUNDARY CONDITIONS\n";
			BC_output.close();
		}
		input_file.close();
		///   ------------------->  Find and store Boundary nodes
		bool BC_flag = 0;
		int Xp, Yp, Zp;
		double n_temp[3];
		for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					for (i = 0; i < number_of_BC; i++) {
						if (Geo_stl->domain[{X,Y,Z}] == intersection[i][0] && solid_sectional[X][Y][Z] == FALSE) {
							n_temp[0] = 0;
							n_temp[1] = 0;
							n_temp[2] = 0;
							for (alpha = 1; alpha < Discrete_Velocity; alpha++) {
								Xp = X + c_alpha[alpha][0];
								Yp = Y + c_alpha[alpha][1];
								Zp = Z + c_alpha[alpha][2];

								if (Geo_stl->domain[{X + 1,Y,Z}] == intersection[i][1])
									n_temp[0] = -1;
								if (Geo_stl->domain[{X - 1,Y,Z}] == intersection[i][1])
									n_temp[0] = 1;
								if (Geo_stl->domain[{X,Y + 1,Z}] == intersection[i][1])
									n_temp[1] = -1;
								if (Geo_stl->domain[{X,Y - 1,Z}] == intersection[i][1])
									n_temp[1] = 1;
								if (Geo_stl->domain[{X,Y,Z + 1}] == intersection[i][1])
									n_temp[2] = -1;
								if (Geo_stl->domain[{X,Y,Z - 1}] == intersection[i][1])
									n_temp[2] = 1;

								if (Geo_stl->domain[{Xp,Yp,Zp}] == intersection[i][1]) {
									BC_flag = 1;
								}
							}
							if (BC_flag == 1) {
								temp[i].X = X;
								temp[i].Y = Y;
								temp[i].Z = Z;
								temp[i].directions.resize(Discrete_Velocity);
								if (n_temp[0] != 0)
									temp[i].n[0] = n_temp[0] / abs(n_temp[0]);
								if (n_temp[1] != 0)
									temp[i].n[1] = n_temp[1] / abs(n_temp[1]);
								if (n_temp[2] != 0)
									temp[i].n[2] = n_temp[2] / abs(n_temp[2]);

								if (n_temp[0] == 0)
									temp[i].n[0] = 0;
								if (n_temp[1] == 0)
									temp[i].n[1] = 0;
								if (n_temp[2] == 0)
									temp[i].n[2] = 0;
								Boundaries.push_back(temp[i]);
								BC_flag = 0;
								/// ALLOCATE MEMORY TO SOLID NEIGHBOR LIST
								for (alpha = 0; alpha < Discrete_Velocity; alpha++) {
									Xp = X - c_alpha[alpha][0];
									Yp = Y - c_alpha[alpha][1];
									Zp = Z - c_alpha[alpha][2];
									Boundaries[Boundaries.size() - 1].directions[alpha] = -1;
									if (Geo_stl->domain[{Xp,Yp,Zp}] == intersection[i][1] && Boundaries[Boundaries.size() - 1].type < 100) {
										Boundaries[Boundaries.size() - 1].directions[alpha] = 1;
									}
								}
							}
						}
					}
				}
			}
		}
		unsigned int boundary_number, boundary_number_2;
		boundary_number = Boundaries.size();
		boundary_number_2 = Boundaries.size();
		/// FOR THE FD BOUNDARIES : IT IS GOING TO DETECTE OPEN CORNERS
		int Xp1, Yp1, Zp1, Xp2, Yp2, Zp2;
		for (i = 0; i < boundary_number; i++) {
			Xp1 = Boundaries[i].X - Boundaries[i].n[0];
			Yp1 = Boundaries[i].Y - Boundaries[i].n[1];
			Zp1 = Boundaries[i].Z - Boundaries[i].n[2];
			for (j = i + 1; j < boundary_number_2; j++) {
				Xp2 = Boundaries[j].X - Boundaries[j].n[0];
				Yp2 = Boundaries[j].Y - Boundaries[j].n[1];
				Zp2 = Boundaries[j].Z - Boundaries[j].n[2];
				if (Xp1 == Xp2 && Yp1 == Yp2 && Zp1 == Zp2 && fabs(Boundaries[i].n[0]) + fabs(Boundaries[i].n[1]) + fabs(Boundaries[i].n[2]) != 0 && fabs(Boundaries[j].n[0]) + fabs(Boundaries[j].n[1]) + fabs(Boundaries[j].n[2]) != 0 && fabs(Boundaries[j].n[0] - Boundaries[i].n[0]) + fabs(Boundaries[j].n[1] - Boundaries[i].n[1]) + fabs(Boundaries[j].n[2] - Boundaries[i].n[2]) != 0) {
					/// GET NORMAL TO NEW BOUNDARY
					n_temp[0] = Boundaries[i].n[0] + Boundaries[j].n[0];
					n_temp[1] = Boundaries[i].n[1] + Boundaries[j].n[1];
					n_temp[2] = Boundaries[i].n[2] + Boundaries[j].n[2];
					/// DETERMINE POSITION OF NEW BOUNDARY (CORNER)
					temp[0].X = abs(Boundaries[i].n[0]) * Boundaries[i].X + abs(Boundaries[j].n[0]) * Boundaries[j].X;
					temp[0].Y = abs(Boundaries[i].n[1]) * Boundaries[i].Y + abs(Boundaries[j].n[1]) * Boundaries[j].Y;
					temp[0].Z = abs(Boundaries[i].n[2]) * Boundaries[i].Z + abs(Boundaries[j].n[2]) * Boundaries[j].Z;
					if (Boundaries[i].X == Boundaries[j].X)
						temp[0].X = Boundaries[i].X;
					if (Boundaries[i].Y == Boundaries[j].Y)
						temp[0].Y = Boundaries[i].Y;
					if (Boundaries[i].Z == Boundaries[j].Z)
						temp[0].Z = Boundaries[i].Z;

					temp[0].n[0] = n_temp[0];
					temp[0].n[1] = n_temp[1];
					temp[0].n[2] = n_temp[2];

					/// INTERSECTION (CORNER NODE) BETWEEN TWO SYMMETRICAL BOUNDARIES
					if (Boundaries[i].type == 104 && Boundaries[j].type == 104) {
						temp[0].type = 104;
						/// SET NORMAL VECTORS OF PREVIOUS BOUNDARIES TO ZERO, EFFECTIVELY TAKING THEM OUT
						Boundaries[i].n[0] = 0;
						Boundaries[i].n[1] = 0;
						Boundaries[i].n[2] = 0;
						Boundaries[j].n[0] = 0;
						Boundaries[j].n[1] = 0;
						Boundaries[j].n[2] = 0;
						temp[0].directions.resize(Discrete_Velocity);
						for (alpha = 0; alpha < Discrete_Velocity; alpha++) {
							Boundaries[i].directions[alpha] = -1;
							Boundaries[j].directions[alpha] = -1;
							temp[0].directions[alpha] = -1;
						}
						Boundaries.push_back(temp[0]);
					}
				}

				boundary_number_2 = Boundaries.size();
			}
		}
#if defined DEBUG_MODE
		stringstream DB_filename;
		DB_filename << "Sections_Boundary_Conditions_DB_proc_" << MPI_parallel->processor_id << ".dat";
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
	return;
}

void Sectional_solver::FD(int time, Flow_solver* Flow, Thermal_solver* Thermal, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		double**** swap_temp = previous_Y_section;
		previous_Y_section = Y_section;
		Y_section = swap_temp;
		int k, X, Y, Z;
		double dY_x, dY_y, dY_z;

		for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (solid_sectional[X][Y][Z] == FALSE) {
						for (k = 0; k < Nb_sections; ++k) {
#if defined FD_UPWIND
							dY_x = FD::UPWIND1NONCONS(Flow->velocity[{X, Y, Z, 0}], previous_Y_section[X - 1][Y][Z][k], previous_Y_section[X][Y][Z][k], previous_Y_section[X + 1][Y][Z][k]);
							dY_y = FD::UPWIND1NONCONS(Flow->velocity[{X, Y, Z, 1}], previous_Y_section[X][Y - 1][Z][k], previous_Y_section[X][Y][Z][k], previous_Y_section[X][Y + 1][Z][k]);
							dY_z = FD::UPWIND1NONCONS(Flow->velocity[{X, Y, Z, 2}], previous_Y_section[X][Y][Z - 1][k], previous_Y_section[X][Y][Z][k], previous_Y_section[X][Y][Z + 1][k]);
#endif  // defined
#if defined FD_UPWIND2
							dY_x = FD::UPWIND2NONCONS(Flow->velocity[{X, Y, Z, 0}], previous_Y_section[X - 2][Y][Z][k], previous_Y_section[X - 1][Y][Z][k], previous_Y_section[X][Y][Z][k], previous_Y_section[X + 1][Y][Z][k], previous_Y_section[X + 2][Y][Z][k]);
							dY_y = FD::UPWIND2NONCONS(Flow->velocity[{X, Y, Z, 1}], previous_Y_section[X][Y - 2][Z][k], previous_Y_section[X][Y - 1][Z][k], previous_Y_section[X][Y][Z][k], previous_Y_section[X][Y + 1][Z][k], previous_Y_section[X][Y + 2][Z][k]);
							dY_z = FD::UPWIND2NONCONS(Flow->velocity[{X, Y, Z, 2}], previous_Y_section[X][Y][Z - 2][k], previous_Y_section[X][Y][Z - 1][k], previous_Y_section[X][Y][Z][k], previous_Y_section[X][Y][Z + 1][k], previous_Y_section[X][Y][Z + 2][k]);
#endif  // defined
#if defined FD_CENTRAL
							dY_x = FD::CENTRALNONCONS(Flow->velocity[{X, Y, Z, 0}], previous_Y_section[X - 1][Y][Z][k], previous_Y_section[X + 1][Y][Z][k]);
							dY_y = FD::CENTRALNONCONS(Flow->velocity[{X, Y, Z, 1}], previous_Y_section[X][Y - 1][Z][k], previous_Y_section[X][Y + 1][Z][k]);
							dY_z = FD::CENTRALNONCONS(Flow->velocity[{X, Y, Z, 2}], previous_Y_section[X][Y][Z - 1][k], previous_Y_section[X][Y][Z + 1][k]);
#endif  // defined
#if defined FD_CENTRAL4
							dY_x = FD::CENTRAL4NONCONS(Flow->velocity[{X, Y, Z, 0}], previous_Y_section[X - 2][Y][Z][k], previous_Y_section[X - 1][Y][Z][k], previous_Y_section[X][Y][Z][k],
							                           previous_Y_section[X + 1][Y][Z][k], previous_Y_section[X + 2][Y][Z][k]);
							dY_y = FD::CENTRAL4NONCONS(Flow->velocity[{X, Y, Z, 1}], previous_Y_section[X][Y - 2][Z][k], previous_Y_section[X][Y - 1][Z][k], previous_Y_section[X][Y][Z][k],
							                           previous_Y_section[X][Y + 1][Z][k], previous_Y_section[X][Y + 2][Z][k]);
							dY_z = FD::CENTRAL4NONCONS(Flow->velocity[{X, Y, Z, 2}], previous_Y_section[X][Y][Z - 2][k], previous_Y_section[X][Y][Z - 1][k], previous_Y_section[X][Y][Z][k],
							                           previous_Y_section[X][Y][Z + 1][k], previous_Y_section[X][Y][Z + 2][k]);
							if (solid_sectional[X + 1][Y][Z] != -1 || solid_sectional[X - 1][Y][Z] != -1) {
								dY_x = FD::CENTRALNONCONS(Flow->velocity[{X, Y, Z, 0}], previous_Y_section[X - 1][Y][Z][k], previous_Y_section[X + 1][Y][Z][k]);
							}
							if (solid_sectional[X][Y + 1][Z] != -1 || solid_sectional[X][Y - 1][Z] != -1) {
								dY_y = FD::CENTRALNONCONS(Flow->velocity[{X, Y, Z, 1}], previous_Y_section[X][Y - 1][Z][k], previous_Y_section[X][Y + 1][Z][k]);
							}
							if (solid_sectional[X][Y][Z + 1] != -1 || solid_sectional[X][Y][Z - 1] != -1) {
								dY_z = FD::CENTRALNONCONS(Flow->velocity[{X, Y, Z, 2}], previous_Y_section[X][Y][Z - 1][k], previous_Y_section[X][Y][Z + 1][k]);
							}
#endif  // defined
#if defined FD_WENO3
							dY_x = FD::WENO3NONCONS(Flow->velocity[{X, Y, Z, 0}], previous_Y_section[X - 2][Y][Z][k], previous_Y_section[X - 1][Y][Z][k],
							                        previous_Y_section[X][Y][Z][k], previous_Y_section[X + 1][Y][Z][k], previous_Y_section[X + 2][Y][Z][k]);
							dY_y = FD::WENO3NONCONS(Flow->velocity[{X, Y, Z, 1}], previous_Y_section[X][Y - 2][Z][k], previous_Y_section[X][Y - 1][Z][k],
							                        previous_Y_section[X][Y][Z][k], previous_Y_section[X][Y + 1][Z][k], previous_Y_section[X][Y + 2][Z][k]);
							dY_z = FD::WENO3NONCONS(Flow->velocity[{X, Y, Z, 2}], previous_Y_section[X][Y][Z - 2][k], previous_Y_section[X][Y][Z - 1][k],
							                        previous_Y_section[X][Y][Z][k], previous_Y_section[X][Y][Z + 1][k], previous_Y_section[X][Y][Z + 2][k]);
#endif  // defined
							Y_section[X][Y][Z][k] = previous_Y_section[X][Y][Z][k] - (dY_x + dY_y + dY_z)
							                        + Production[X][Y][Z][k] / (Flow->rho_0 * Flow->density[{X, Y, Z}]);
						}
					}
				}
			}
		}
		return;
	}
}

void Sectional_solver::BC(int time, Flow_solver* Flow, Thermal_solver* Thermal, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		int bou;
		int X, Y, Z, k, kk, Xp, Yp, Zp;

		for (k = 0; k < Boundaries.size(); ++k) {
			X = Boundaries[k].X;
			Y = Boundaries[k].Y;
			Z = Boundaries[k].Z;

			bou = Boundaries[k].type;
			switch (bou) {
				case 102: {
					for (kk = 0; kk < Nb_sections; ++kk) {
						Xp = (X - Boundaries[k].n[0]);
						Yp = (Y - Boundaries[k].n[1]);
						Zp = (Z - Boundaries[k].n[2]);
						Y_section[Xp][Yp][Zp][kk] = Boundaries[k].Y_section[kk];
						previous_Y_section[Xp][Yp][Zp][kk] = Boundaries[k].Y_section[kk];
					}
					break;
				}
				case 104: {
					for (kk = 0; kk < Nb_sections; ++kk) {
						Xp = (X - Boundaries[k].n[0]);
						Yp = (Y - Boundaries[k].n[1]);
						Zp = (Z - Boundaries[k].n[2]);
						Y_section[Xp][Yp][Zp][kk] = Y_section[X][Y][Z][kk];
						previous_Y_section[Xp][Yp][Zp][kk] = previous_Y_section[X][Y][Z][kk];
					}
					break;
				}
			}
		}
	}
	return;
}
void Sectional_solver::Data_Exchange_Macroscopic(Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		int Nparams = Nb_sections * 2;
		///--------------------------------------------------------->  In Y-direction
		static double* buf_totop_macro_y = new double[MPI_parallel->dev_end[0] * MPI_parallel->dev_end[2] * Nparams];
		static double* buf_tobottom_macro_y = new double[MPI_parallel->dev_end[0] * MPI_parallel->dev_end[2] * Nparams];
		static double* buf_fromtop_macro_y = new double[MPI_parallel->dev_end[0] * MPI_parallel->dev_end[2] * Nparams];
		static double* buf_frombottom_macro_y = new double[MPI_parallel->dev_end[0] * MPI_parallel->dev_end[2] * Nparams];
		// Prepare messages to be sent
		for (int X = 0; X < MPI_parallel->dev_end[0]; X++) {
			for (int Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
				memcpy(buf_totop_macro_y + X * MPI_parallel->dev_end[2] * Nparams + Z * Nparams + 0,
				       &Y_section[X][MPI_parallel->end_XYZ2[1]][Z][0], Nb_sections * sizeof(double));
				memcpy(buf_totop_macro_y + X * MPI_parallel->dev_end[2] * Nparams + Z * Nparams + Nb_sections,
				       &Y_section[X][MPI_parallel->end_XYZ2[1] - 1][Z][0], Nb_sections * sizeof(double));

				memcpy(buf_tobottom_macro_y + X * MPI_parallel->dev_end[2] * Nparams + Z * Nparams + 0,
				       &Y_section[X][2][Z][0], Nb_sections * sizeof(double));
				memcpy(buf_tobottom_macro_y + X * MPI_parallel->dev_end[2] * Nparams + Z * Nparams + Nb_sections,
				       &Y_section[X][3][Z][0], Nb_sections * sizeof(double));
			}
		}
		int Bottom_neighbour, Top_neighbour;
		Top_neighbour = MPI_parallel->proc_arrangement[MPI_parallel->proc_position[0]][(MPI_parallel->proc_position[1] + 1) % MPI_parallel->Np_Y][MPI_parallel->proc_position[2]];
		Bottom_neighbour = MPI_parallel->proc_arrangement[MPI_parallel->proc_position[0]][(MPI_parallel->proc_position[1] + MPI_parallel->Np_Y - 1) % MPI_parallel->Np_Y][MPI_parallel->proc_position[2]];
		////Send-recv all totop+fromt
		MPI_Sendrecv(buf_totop_macro_y, Nparams * MPI_parallel->dev_end[0] * MPI_parallel->dev_end[2], MPI_DOUBLE, Top_neighbour, LTAG,
		             buf_frombottom_macro_y, Nparams * MPI_parallel->dev_end[0] * MPI_parallel->dev_end[2], MPI_DOUBLE, Bottom_neighbour, LTAG,
		             MPI_COMM_WORLD, &status);
		// Send-recv all toleft+fromleft
		MPI_Sendrecv(buf_tobottom_macro_y, Nparams * MPI_parallel->dev_end[0] * MPI_parallel->dev_end[2], MPI_DOUBLE, Bottom_neighbour, RTAG,
		             buf_fromtop_macro_y, Nparams * MPI_parallel->dev_end[0] * MPI_parallel->dev_end[2], MPI_DOUBLE, Top_neighbour, RTAG,
		             MPI_COMM_WORLD, &status);

		for (int X = 0; X < MPI_parallel->dev_end[0]; X++) {
			for (int Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
				memcpy(&Y_section[X][1][Z][0],
				       buf_frombottom_macro_y + X * MPI_parallel->dev_end[2] * Nparams + Z * Nparams + 0,
				       Nb_sections * sizeof(double));
				memcpy(&Y_section[X][0][Z][0],
				       buf_frombottom_macro_y + X * MPI_parallel->dev_end[2] * Nparams + Z * Nparams + Nb_sections,
				       Nb_sections * sizeof(double));

				memcpy(&Y_section[X][MPI_parallel->actual_rows_XYZ[1] + 2][Z][0],
				       buf_fromtop_macro_y + X * MPI_parallel->dev_end[2] * Nparams + Z * Nparams + 0,
				       Nb_sections * sizeof(double));
				memcpy(&Y_section[X][MPI_parallel->actual_rows_XYZ[1] + 3][Z][0],
				       buf_fromtop_macro_y + X * MPI_parallel->dev_end[2] * Nparams + Z * Nparams + Nb_sections,
				       Nb_sections * sizeof(double));
			}
		}
		///////// Exchange data with neighbors /////////
		static double* buf_toleft_macro_y = new double[MPI_parallel->dev_end[1] * MPI_parallel->dev_end[2] * Nparams];
		static double* buf_toright_macro_y = new double[MPI_parallel->dev_end[1] * MPI_parallel->dev_end[2] * Nparams];
		static double* buf_fromleft_macro_y = new double[MPI_parallel->dev_end[1] * MPI_parallel->dev_end[2] * Nparams];
		static double* buf_fromright_macro_y = new double[MPI_parallel->dev_end[1] * MPI_parallel->dev_end[2] * Nparams];
		///--------------------------------------------------------->  In X-direction
		// Prepare messages to be sent
		for (int Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
			for (int Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
				memcpy(buf_toright_macro_y + Y * MPI_parallel->dev_end[2] * Nparams + Z * Nparams + 0,
				       &Y_section[MPI_parallel->end_XYZ2[0]][Y][Z][0], Nb_sections * sizeof(double));
				memcpy(buf_toright_macro_y + Y * MPI_parallel->dev_end[2] * Nparams + Z * Nparams + Nb_sections,
				       &Y_section[MPI_parallel->end_XYZ2[0] - 1][Y][Z][0], Nb_sections * sizeof(double));

				memcpy(buf_toleft_macro_y + Y * MPI_parallel->dev_end[2] * Nparams + Z * Nparams + 0,
				       &Y_section[2][Y][Z][0], Nb_sections * sizeof(double));
				memcpy(buf_toleft_macro_y + Y * MPI_parallel->dev_end[2] * Nparams + Z * Nparams + Nb_sections,
				       &Y_section[3][Y][Z][0], Nb_sections * sizeof(double));
			}
		}
		int Left_neighbour, Right_neighbour;
		Right_neighbour = MPI_parallel->proc_arrangement[(MPI_parallel->proc_position[0] + 1) % MPI_parallel->Np_X][MPI_parallel->proc_position[1]][MPI_parallel->proc_position[2]];
		Left_neighbour = MPI_parallel->proc_arrangement[(MPI_parallel->proc_position[0] + MPI_parallel->Np_X - 1) % MPI_parallel->Np_X][MPI_parallel->proc_position[1]][MPI_parallel->proc_position[2]];
		////Send-recv all toright+fromleft
		MPI_Sendrecv(buf_toright_macro_y, Nparams * MPI_parallel->dev_end[1] * MPI_parallel->dev_end[2], MPI_DOUBLE, Right_neighbour, LTAG,
		             buf_fromleft_macro_y, Nparams * MPI_parallel->dev_end[1] * MPI_parallel->dev_end[2], MPI_DOUBLE, Left_neighbour, LTAG,
		             MPI_COMM_WORLD, &status);
		// Send-recv all toleft+fromright
		MPI_Sendrecv(buf_toleft_macro_y, Nparams * MPI_parallel->dev_end[1] * MPI_parallel->dev_end[2], MPI_DOUBLE, Left_neighbour, RTAG,
		             buf_fromright_macro_y, Nparams * MPI_parallel->dev_end[1] * MPI_parallel->dev_end[2], MPI_DOUBLE, Right_neighbour, RTAG,
		             MPI_COMM_WORLD, &status);
		// Postprocess messages
		for (int Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
			for (int Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
				memcpy(&Y_section[1][Y][Z][0], buf_fromleft_macro_y + Y * MPI_parallel->dev_end[2] * Nparams + Z * Nparams + 0, Nb_sections * sizeof(double));
				memcpy(&Y_section[0][Y][Z][0], buf_fromleft_macro_y + Y * MPI_parallel->dev_end[2] * Nparams + Z * Nparams + Nb_sections, Nb_sections * sizeof(double));

				memcpy(&Y_section[MPI_parallel->actual_rows_XYZ[0] + 2][Y][Z][0],
				       buf_fromright_macro_y + Y * MPI_parallel->dev_end[2] * Nparams + Z * Nparams + 0, Nb_sections * sizeof(double));
				memcpy(&Y_section[MPI_parallel->actual_rows_XYZ[0] + 3][Y][Z][0],
				       buf_fromright_macro_y + Y * MPI_parallel->dev_end[2] * Nparams + Z * Nparams + Nb_sections, Nb_sections * sizeof(double));
			}
		}
		///--------------------------------------------------------->  In Z-direction
		static double* buf_tofront_macro_y = new double[MPI_parallel->dev_end[0] * MPI_parallel->dev_end[1] * Nparams];
		static double* buf_torear_macro_y = new double[MPI_parallel->dev_end[0] * MPI_parallel->dev_end[1] * Nparams];
		static double* buf_fromfront_macro_y = new double[MPI_parallel->dev_end[0] * MPI_parallel->dev_end[1] * Nparams];
		static double* buf_fromrear_macro_y = new double[MPI_parallel->dev_end[0] * MPI_parallel->dev_end[1] * Nparams];
		// Prepare messages to be sent
		for (int X = 0; X < MPI_parallel->dev_end[0]; X++) {
			for (int Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
				memcpy(buf_tofront_macro_y + X * MPI_parallel->dev_end[1] * Nparams + Y * Nparams + 0,
				       &Y_section[X][Y][MPI_parallel->end_XYZ2[2]][0], Nb_sections * sizeof(double));
				memcpy(buf_tofront_macro_y + X * MPI_parallel->dev_end[1] * Nparams + Y * Nparams + Nb_sections,
				       &Y_section[X][Y][MPI_parallel->end_XYZ2[2] - 1][0], Nb_sections * sizeof(double));

				memcpy(buf_torear_macro_y + X * MPI_parallel->dev_end[1] * Nparams + Y * Nparams + 0,
				       &Y_section[X][Y][2][0], Nb_sections * sizeof(double));
				memcpy(buf_torear_macro_y + X * MPI_parallel->dev_end[1] * Nparams + Y * Nparams + Nb_sections,
				       &Y_section[X][Y][3][0], Nb_sections * sizeof(double));
			}
		}
		int Rear_neighbour, Front_neighbour;
		Front_neighbour = MPI_parallel->proc_arrangement[MPI_parallel->proc_position[0]][MPI_parallel->proc_position[1]][(MPI_parallel->proc_position[2] + 1) % MPI_parallel->Np_Z];
		Rear_neighbour = MPI_parallel->proc_arrangement[MPI_parallel->proc_position[0]][MPI_parallel->proc_position[1]][(MPI_parallel->proc_position[2] + MPI_parallel->Np_Z - 1) % MPI_parallel->Np_Z];
		////Send-recv all totop+fromt
		MPI_Sendrecv(buf_tofront_macro_y, Nparams * MPI_parallel->dev_end[0] * MPI_parallel->dev_end[1], MPI_DOUBLE, Front_neighbour, LTAG,
		             buf_fromrear_macro_y, Nparams * MPI_parallel->dev_end[0] * MPI_parallel->dev_end[1], MPI_DOUBLE, Rear_neighbour, LTAG,
		             MPI_COMM_WORLD, &status);
		// Send-recv all toleft+fromleft
		MPI_Sendrecv(buf_torear_macro_y, Nparams * MPI_parallel->dev_end[0] * MPI_parallel->dev_end[1], MPI_DOUBLE, Rear_neighbour, RTAG,
		             buf_fromfront_macro_y, Nparams * MPI_parallel->dev_end[0] * MPI_parallel->dev_end[1], MPI_DOUBLE, Front_neighbour, RTAG,
		             MPI_COMM_WORLD, &status);
		for (int X = 0; X < MPI_parallel->dev_end[0]; X++) {
			for (int Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
				memcpy(&Y_section[X][Y][1][0], buf_fromrear_macro_y + X * MPI_parallel->dev_end[1] * Nparams + Y * Nparams + 0, Nb_sections * sizeof(double));
				memcpy(&Y_section[X][Y][0][0], buf_fromrear_macro_y + X * MPI_parallel->dev_end[1] * Nparams + Y * Nparams + Nb_sections, Nb_sections * sizeof(double));

				memcpy(&Y_section[X][Y][MPI_parallel->actual_rows_XYZ[2] + 2][0],
				       buf_fromfront_macro_y + X * MPI_parallel->dev_end[1] * Nparams + Y * Nparams + 0, Nb_sections * sizeof(double));
				memcpy(&Y_section[X][Y][MPI_parallel->actual_rows_XYZ[2] + 3][0],
				       buf_fromfront_macro_y + X * MPI_parallel->dev_end[1] * Nparams + Y * Nparams + Nb_sections, Nb_sections * sizeof(double));
			}
		}
	}
}
void Sectional_solver::swap_endian(char* buffer, size_t size) {
	for (size_t b = 0; b < size / 2; b++) {
		char temp = buffer[b];
		buffer[b] = buffer[size - 1 - b];
		buffer[size - 1 - b] = temp;
	}
}
void Sectional_solver::swap_endian(char* buffer, char* buffer2, size_t size) {
	for (size_t b = 0; b < size; b++) {
		buffer2[b] = buffer[size - 1 - b];
	}
}
void Sectional_solver::write_bigendian(std::ofstream& file, char* buffer, size_t count, size_t size) {
	char* _buf = new char[size];
	for (size_t i = 0; i < count; i++) {
		swap_endian(buffer + (size * i), _buf, size);
		file.write(_buf, size);
	}
	delete[] _buf;
}
template <typename T>
void Sectional_solver::write_bigendian(std::ofstream& file, T* buffer, size_t count) {
	char* _buf = new char[sizeof(T)];
	for (size_t i = 0; i < count; i++) {
		swap_endian((char*)(buffer + i), _buf, sizeof(T));
		file.write(_buf, sizeof(T));
	}
	delete[] _buf;
}
void Sectional_solver::write_vtk(int time, int t_vtk, Geometry* Geo, Parallel_MPI* MPI_parallel) {
	if (time % t_vtk == 0) {
		/// Create filename
		int X, Y, Z, k = 0;
		double mm[1];
		/// Create filename
		/// Create output data file, filename format: fluid_t%time-step%_%processor_id%.vti
		stringstream output_filename;
		if (MPI_parallel->processor_id != MASTER) {
			output_filename << "Alborz_Results/vtk_fluid/sectional_t" << time << "_" << MPI_parallel->processor_id << ".vti";
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
			output_file << MPI_parallel->start_XYZ[0] << " " << MPI_parallel->end_XYZ[0] + 1 << " ";
			output_file << MPI_parallel->start_XYZ[1] << " " << MPI_parallel->end_XYZ[1] + 1 << " ";
			output_file << MPI_parallel->start_XYZ[2] << " " << MPI_parallel->end_XYZ[2] + 1 << " "
						<< "\" ";
			output_file << "Origin=\" 0.000000 0.000000 0.000000\" Spacing=\"" << 1.0 << " " << 1.0 << " " << 1.0 << "\">" << endl;
			output_file << "<Piece Extent=\"" << MPI_parallel->start_XYZ[0] << " " << MPI_parallel->end_XYZ[0] + 1 << " ";
			output_file << MPI_parallel->start_XYZ[1] << " " << MPI_parallel->end_XYZ[1] + 1 << " ";
			output_file << MPI_parallel->start_XYZ[2] << " " << MPI_parallel->end_XYZ[2] + 1 << "\">" << endl;
			output_file << "<PointData Scalars=\"Fluid\">" << endl;
			unsigned int pointer_index = 0;
			for (k = 0; k < Nb_sections; ++k) {
				output_file << "<DataArray type=\"Float64\" Name=\"Y_" << k << "\" format=\"" << DataType;
#if defined VTK_BINARY
				output_file << "\" offset=\"" << pointer_index * (MPI_parallel->actual_rows_XYZ[2] + 1) * (MPI_parallel->actual_rows_XYZ[1] + 1) * (MPI_parallel->actual_rows_XYZ[0] + 1) * sizeof(double) + pointer_index * sizeof(int64_t);
				pointer_index++;
#endif  // defined
				output_file << "\">" << endl;
#if defined VTK_ASCII
				for (Z = 2; Z <= MPI_parallel->actual_rows_XYZ[2] + 2; ++Z) {
					for (Y = 2; Y <= MPI_parallel->actual_rows_XYZ[1] + 2; ++Y) {
						for (X = 2; X <= MPI_parallel->actual_rows_XYZ[0] + 2; ++X) {
							output_file << std::setprecision(8) << Y_section[X][Y][Z][k] << endl;
						}
					}
				}
#endif  // defined
				output_file << "</DataArray>" << endl;
			}
			output_file << "</PointData>" << endl;
			output_file << "</Piece>" << endl;
			output_file << "</ImageData>" << endl;
#if defined VTK_BINARY
			int64_t nn[1];
			output_file << "<AppendedData encoding=\"raw\">" << endl;
			output_file << "_";
			nn[0] = sizeof(double) * (MPI_parallel->actual_rows_XYZ[2] + 1) * (MPI_parallel->actual_rows_XYZ[1] + 1) * (MPI_parallel->actual_rows_XYZ[0] + 1);
			for (k = 0; k < Nb_sections; ++k) {
				write_bigendian(output_file, nn, 1);
				for (Z = 2; Z <= MPI_parallel->actual_rows_XYZ[2] + 2; ++Z) {
					for (Y = 2; Y <= MPI_parallel->actual_rows_XYZ[1] + 2; ++Y) {
						for (X = 2; X <= MPI_parallel->actual_rows_XYZ[0] + 2; ++X) {
							mm[0] = Y_section[X][Y][Z][k];
							write_bigendian(output_file, mm, 1);
						}
					}
				}
			}
			output_file << "</AppendedData>" << endl;
#endif  // defined
			output_file << "</VTKFile>" << endl;

			/// Close file
			output_file.close();
		}
		MPI_parallel->Sync_Master();
		if (MPI_parallel->processor_id == MASTER) {
			output_filename << "Alborz_Results/vtk_fluid/sectional_t" << time << ".pvti";
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
			output_file << MPI_parallel->start_XYZ[0] << " " << global_parameters.Nx << " ";
			output_file << MPI_parallel->start_XYZ[1] << " " << global_parameters.Ny << " ";
			output_file << MPI_parallel->start_XYZ[2] << " " << global_parameters.Nz << " "
						<< "\" ";
			output_file << "Origin=\" 0.000000 0.000000 0.000000\" Spacing=\"" << 1.0 << " " << 1.0 << " " << 1.0 << "\" GhostLevel=\"0\">" << endl;
			output_file << "<PPointData Scalars=\"Fluid\">" << endl;

			for (k = 0; k < Nb_sections; ++k) {
				output_file << "<PDataArray type=\"Float64\" Name=\"Y_" << k << "\" format=\"" << DataType << "\">" << endl;
				output_file << "</PDataArray>" << endl;
			}
			output_file << "</PPointData>" << endl;
			for (int i = 1; i < MPI_parallel->num_processors; i++) {
				output_file << "<Piece Extent=\"";
				output_file << MPI_parallel->start_XYZ[0 + 3 * i] << " " << MPI_parallel->end_XYZ[0 + 3 * i] + 1 << " ";
				output_file << MPI_parallel->start_XYZ[1 + 3 * i] << " " << MPI_parallel->end_XYZ[1 + 3 * i] + 1 << " ";
				output_file << MPI_parallel->start_XYZ[2 + 3 * i] << " " << MPI_parallel->end_XYZ[2 + 3 * i] + 1 << "\" ";
				output_file << "Source=\"sectional_t" << time << "_" << i << ".vti\"/>" << endl;
			}
			output_file << "</PImageData>" << endl;
			output_file << "</VTKFile>" << endl;
			/// Close file
			output_file.close();
		}
	}
	return;
}

void Inline_User_Defined(double**** Y_section, int*** solid, std::vector<double> Min_radius,
                         std::vector<double> Max_radius, double N_x, double N_y, double N_z, unsigned int Nb_sections,
                         std::string filename, int Zones, Parallel_MPI* MPI_parallel) {
	///----------------------------------------------------------------------///
	///                        COMMAND LINE INPUT                            ///
	///----------------------------------------------------------------------///
	if (MPI_parallel->processor_id != MASTER) {
		double** Ini_Y_sectional;
		int* type;
		unsigned int X, Y, Z, k, i, index;
		Ini_Y_sectional = new double*[Zones + 1];
		type = new int[Zones + 1];
		for (i = 0; i < Zones + 1; i++) {
			type[i] = 0;
			Ini_Y_sectional[i] = new double[Nb_sections];
			for (k = 0; k < Nb_sections; k++) {
				Ini_Y_sectional[i][k] = 0;
			}
		}
		type[0] = 1;  /// -----> Solids are set to +1
		/// Open input file
		string input_filename(filename);
		input_filename += ".dat";
		ifstream input_file_s;  // File is open for READING
		input_file_s.open(input_filename.c_str(), ios::binary);
		input_file_s.clear();
		input_file_s.seekg(0, ios::beg);
		find_line_after_header(input_file_s, "c\tSectional Field Initial Conditions");
		if (MPI_parallel->processor_id == (MASTER + 1)) {
			std::cout << "SECTIONAL: User defined Initial conditions \n";
			std::cout << "================================ \n";
		}
		for (i = 0; i < Zones; i++) {
			find_line_after_comment(input_file_s);
			input_file_s >> index;
			input_file_s >> type[index];
			find_line_after_comment(input_file_s);
			for (k = 0; k < Nb_sections; k++) {
				input_file_s >> Ini_Y_sectional[index][k];
			}
			if (MPI_parallel->processor_id == (MASTER + 1)) {
				std::cout << "Zone : " << index << "\t Type : " << type[index] << std::endl;
				std::cout << "Initial Sectional Fractions : " << std::endl;
				for (int k = 0; k < Nb_sections; ++k) {
					std::cout << Min_radius[k] << "-" << Max_radius[k] << " :"
							  << "\t" << Ini_Y_sectional[index][k] << "\n";
				}
				std::cout << endl;
				std::cout << "-------------------------------- \n";
			}
		}
		input_file_s.close();
		for (X = 0; X < N_x; X++) {
			for (Y = 0; Y < N_y; Y++) {
				for (Z = 0; Z < N_z; Z++) {
					for (k = 0; k < Nb_sections; k++) {
						Y_section[X][Y][Z][k] = Ini_Y_sectional[solid[X][Y][Z]][k];
					}
					solid[X][Y][Z] = type[solid[X][Y][Z]];
				}
			}
		}
	}
	return;
}
Sectional_solver::~Sectional_solver() {
}
