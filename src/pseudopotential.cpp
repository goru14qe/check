#include "pseudopotential.h"
#include "Flow_solver.h"
#include "ALBORZ_Macros.h"
#include "ALBORZ_SETTINGS.h"
#include "ALBORZ_GlobalVariables.h"
#include "Geometry.h"
pseudopotential::pseudopotential() {
	// ctor
}
void pseudopotential::General_data_input(std::string filename, Parallel_MPI* MPI_parallel) {
}
void pseudopotential::initialize_field(Geometry* Geo, stl_import* Geo_stl, Parallel_MPI* MPI_parallel, int& tot_sol, std::string filename) {
	/// Open input file
	string input_filename(filename);
	input_filename += ".dat";
	ifstream input_file;  // File is open for READING
	input_file.open(input_filename.c_str(), ios::binary);

	input_file.clear();
	input_file.seekg(0, ios::beg);
	find_line_after_header(input_file, "c\tPseudo-Potential");
	find_line_after_comment(input_file);
	input_file >> G >> G1 >> G2;

	input_file.close();
	return;
}
void pseudopotential::Memory_allocation(Parallel_MPI* MPI_parallel) {
	unsigned int X, Y;
	if (MPI_parallel->processor_id != MASTER) {
		Psi = new double**[MPI_parallel->dev_end[0]];
		for (X = 0; X < MPI_parallel->dev_end[0]; ++X) {
			Psi[X] = new double*[MPI_parallel->dev_end[1]];
			for (Y = 0; Y < MPI_parallel->dev_end[1]; ++Y) {
				Psi[X][Y] = new double[MPI_parallel->dev_end[2]];
			}
		}
	}
}
void pseudopotential::Get_potential(PseudoPotentialPressure get_pressure, Flow_solver* Flow, Thermal_solver* Thermal, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		unsigned int X, Y, Z;
		double cs2, dens, temp;
		cs2 = 1. / Flow->c_s2;
		for (X = 0; X < MPI_parallel->dev_end[0]; ++X) {
			for (Y = 0; Y < MPI_parallel->dev_end[1]; ++Y) {
				for (Z = 0; Z < MPI_parallel->dev_end[2]; ++Z) {
					/* Get pressure from EoS */
					dens = Flow->density[{X, Y, Z}];
					temp = Thermal->temperature[{X, Y, Z}];
					get_pressure(dens, temp, Flow->pressure[{X, Y, Z}], cs2);
					/// Flow->pressure[{X,Y,Z}] /= Flow->rho_0;
					/* Determine value of G to guarantee positivity of P - rho c_s^2 */
					Psi[X][Y][Z] = sqrt(2. * (Flow->pressure[{X, Y, Z}] - Flow->density[{X, Y, Z}] * cs2) / (G * cs2));
					/// std::cout << Psi[X][Y][Z] << " ";
				}
			}
		}
	}
}
void pseudopotential::Store_pressure(PseudoPotentialPressure get_pressure, Flow_solver* Flow, Thermal_solver* Thermal, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		unsigned int X, Y, Z;
		double cs2, dens, temp;
		cs2 = 1. / Flow->c_s2;
		for (X = 0; X < MPI_parallel->dev_end[0]; ++X) {
			for (Y = 0; Y < MPI_parallel->dev_end[1]; ++Y) {
				for (Z = 0; Z < MPI_parallel->dev_end[2]; ++Z) {
					/* Get pressure from EoS */
					dens = Flow->density[{X, Y, Z}];
					temp = Thermal->temperature[{X, Y, Z}];
					get_pressure(dens, temp, Flow->pressure[{X, Y, Z}], cs2);
					Psi[X][Y][Z] = sqrt(2. * (Flow->pressure[{X, Y, Z}] - Flow->density[{X, Y, Z}] * cs2) / (G * cs2));
					Flow->pressure[{X, Y, Z}] = Flow->density[{X, Y, Z}] * cs2 + .5 * G * sqr(Psi[X][Y][Z]) * cs2;
				}
			}
		}
	}
}
/// ***************************************************** ///
/// GET PSEUDO-POTENTIAL FORCE (FIRST-ORDER)              ///
/// ***************************************************** ///
void pseudopotential::Get_force_1storder(Flow_solver* Flow, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		unsigned int X, Y, Z, alpha;
		double Fxtemp, Fytemp, Fztemp;
		//	double cs2 = 1. / Flow->c_s2;
		double s_function;
		for (X = MPI_parallel->start_XYZ2[0] - 1; X <= MPI_parallel->end_XYZ2[0] + 1; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1] - 1; Y <= MPI_parallel->end_XYZ2[1] + 1; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2] - 1; Z <= MPI_parallel->end_XYZ2[2] + 1; ++Z) {
					if (Flow->is_solid[{X, Y, Z}] == -1) {
						Fxtemp = 0;
						Fytemp = 0;
						Fztemp = 0;
						for (alpha = 1; alpha < Flow->Discrete_Velocity; alpha++) {
							s_function = 1;  //.5 * (1 - Flow->is_solid[X + Flow->c_alpha[alpha][0]][Y + Flow->c_alpha[alpha][1]][Z + Flow->c_alpha[alpha][2]]);
							Fxtemp += (Flow->weight[alpha] * Flow->c_alpha[alpha][0] * Psi[X + Flow->c_alpha[alpha][0]][Y + Flow->c_alpha[alpha][1]][Z + Flow->c_alpha[alpha][2]]) * s_function;
							Fytemp += (Flow->weight[alpha] * Flow->c_alpha[alpha][1] * Psi[X + Flow->c_alpha[alpha][0]][Y + Flow->c_alpha[alpha][1]][Z + Flow->c_alpha[alpha][2]]) * s_function;
							Fztemp += (Flow->weight[alpha] * Flow->c_alpha[alpha][2] * Psi[X + Flow->c_alpha[alpha][0]][Y + Flow->c_alpha[alpha][1]][Z + Flow->c_alpha[alpha][2]]) * s_function;
						}
						Flow->force[{X, Y, Z, 0}] = Flow->force[{X, Y, Z, 0}] - (G1 * Psi[X][Y][Z] * Fxtemp);
						if (Flow->Dimension > 1)
							Flow->force[{X, Y, Z, 1}] = Flow->force[{X, Y, Z, 1}] - (G * Psi[X][Y][Z] * Fytemp);
						if (Flow->Dimension > 2)
							Flow->force[{X, Y, Z, 2}] = Flow->force[{X, Y, Z, 2}] - (G * Psi[X][Y][Z] * Fztemp);
					}
				}
			}
		}
	}
}
/// ***************************************************** ///
/// GET PSEUDO-POTENTIAL FORCE (FIRST-ORDER)              ///
/// ***************************************************** ///
void pseudopotential::Get_force_1storder_improved(Flow_solver* Flow, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		unsigned int X, Y, Z, alpha;
		double Fxtemp, Fytemp, Fztemp;
		double Fxtemp2, Fytemp2, Fztemp2;
		//	double cs2 = 1. / Flow->c_s2;
		double s_function;
		double beta = 1.16;
		for (X = MPI_parallel->start_XYZ2[0] - 1; X <= MPI_parallel->end_XYZ2[0] + 1; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1] - 1; Y <= MPI_parallel->end_XYZ2[1] + 1; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2] - 1; Z <= MPI_parallel->end_XYZ2[2] + 1; ++Z) {
					if (Flow->is_solid[{X, Y, Z}] == -1) {
						Fxtemp = 0;
						Fytemp = 0;
						Fztemp = 0;
						Fxtemp2 = 0;
						Fytemp2 = 0;
						Fztemp2 = 0;
						for (alpha = 1; alpha < Flow->Discrete_Velocity; alpha++) {
							s_function = 1;  //.5 * (1 - Flow->is_solid[X + Flow->c_alpha[alpha][0]][Y + Flow->c_alpha[alpha][1]][Z + Flow->c_alpha[alpha][2]]);
							Fxtemp += (Flow->weight[alpha] * Flow->c_alpha[alpha][0] * Psi[X + Flow->c_alpha[alpha][0]][Y + Flow->c_alpha[alpha][1]][Z + Flow->c_alpha[alpha][2]]) * s_function;
							Fytemp += (Flow->weight[alpha] * Flow->c_alpha[alpha][1] * Psi[X + Flow->c_alpha[alpha][0]][Y + Flow->c_alpha[alpha][1]][Z + Flow->c_alpha[alpha][2]]) * s_function;
							Fztemp += (Flow->weight[alpha] * Flow->c_alpha[alpha][2] * Psi[X + Flow->c_alpha[alpha][0]][Y + Flow->c_alpha[alpha][1]][Z + Flow->c_alpha[alpha][2]]) * s_function;

							Fxtemp2 += (Flow->weight[alpha] * Flow->c_alpha[alpha][0] * sqr(Psi[X + Flow->c_alpha[alpha][0]][Y + Flow->c_alpha[alpha][1]][Z + Flow->c_alpha[alpha][2]])) * s_function;
							Fytemp2 += (Flow->weight[alpha] * Flow->c_alpha[alpha][1] * sqr(Psi[X + Flow->c_alpha[alpha][0]][Y + Flow->c_alpha[alpha][1]][Z + Flow->c_alpha[alpha][2]])) * s_function;
							Fztemp2 += (Flow->weight[alpha] * Flow->c_alpha[alpha][2] * sqr(Psi[X + Flow->c_alpha[alpha][0]][Y + Flow->c_alpha[alpha][1]][Z + Flow->c_alpha[alpha][2]])) * s_function;
						}
						Flow->force[{X, Y, Z, 0}] = Flow->force[{X, Y, Z, 0}] - (beta * G * Psi[X][Y][Z] * Fxtemp + .5 * (1. - beta) * G * Fxtemp2);
						if (Flow->Dimension > 1)
							Flow->force[{X, Y, Z, 1}] = Flow->force[{X, Y, Z, 1}] - (beta * G1 * Psi[X][Y][Z] * Fytemp + .5 * (1. - beta) * G1 * Fytemp2);
						if (Flow->Dimension > 2)
							Flow->force[{X, Y, Z, 2}] = Flow->force[{X, Y, Z, 2}] - (beta * G1 * Psi[X][Y][Z] * Fztemp + .5 * (1. - beta) * G1 * Fztemp2);
					}
				}
			}
		}
	}
}
/// ***************************************************** ///
/// GET PSEUDO-POTENTIAL FORCE (MULTI-RANGE)              ///
/// ***************************************************** ///
void pseudopotential::Get_force_2ndorder(Flow_solver* Flow, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		unsigned int X, Y, Z, alpha;
		double Fx1, Fy1, Fz1;
		double Fx2, Fy2, Fz2;
		//	double cs2 = 1. / Flow->c_s2;
		for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (Flow->is_solid[{X, Y, Z}] == FALSE) {
						Fx1 = 0;
						Fy1 = 0;
						Fz1 = 0;
						Fx2 = 0;
						Fy2 = 0;
						Fz2 = 0;
						/// Short range force
						for (alpha = 1; alpha < Flow->Discrete_Velocity; alpha++) {
							Fx1 += (Flow->weight[alpha] * Flow->c_alpha[alpha][0] * Psi[X + Flow->c_alpha[alpha][0]][Y + Flow->c_alpha[alpha][1]][Z + Flow->c_alpha[alpha][2]]);
							Fy1 += (Flow->weight[alpha] * Flow->c_alpha[alpha][1] * Psi[X + Flow->c_alpha[alpha][0]][Y + Flow->c_alpha[alpha][1]][Z + Flow->c_alpha[alpha][2]]);
							Fz1 += (Flow->weight[alpha] * Flow->c_alpha[alpha][2] * Psi[X + Flow->c_alpha[alpha][0]][Y + Flow->c_alpha[alpha][1]][Z + Flow->c_alpha[alpha][2]]);

							Fx2 += (Flow->weight[alpha] * Flow->c_alpha[alpha][0] * Psi[X + 2 * Flow->c_alpha[alpha][0]][Y + 2 * Flow->c_alpha[alpha][1]][Z + 2 * Flow->c_alpha[alpha][2]]);
							Fy2 += (Flow->weight[alpha] * Flow->c_alpha[alpha][1] * Psi[X + 2 * Flow->c_alpha[alpha][0]][Y + 2 * Flow->c_alpha[alpha][1]][Z + 2 * Flow->c_alpha[alpha][2]]);
							Fz2 += (Flow->weight[alpha] * Flow->c_alpha[alpha][2] * Psi[X + 2 * Flow->c_alpha[alpha][0]][Y + 2 * Flow->c_alpha[alpha][1]][Z + 2 * Flow->c_alpha[alpha][2]]);
						}
						Flow->force[{X, Y, Z, 0}] = Flow->force[{X, Y, Z, 0}] - Psi[X][Y][Z] * (G1 * Fx1 + G2 * Fx2);
						if (Flow->Dimension > 1)
							Flow->force[{X, Y, Z, 1}] = Flow->force[{X, Y, Z, 1}] - Psi[X][Y][Z] * (G1 * Fy1 + G2 * Fy2);
						if (Flow->Dimension > 2)
							Flow->force[{X, Y, Z, 2}] = Flow->force[{X, Y, Z, 2}] - Psi[X][Y][Z] * (G1 * Fz1 + G2 * Fz2);
					}
				}
			}
		}
	}
}
/// ***************************************************** ///
/// GET PSEUDO-POTENTIAL FORCE (MULTI-RANGE)              ///
/// ***************************************************** ///
void pseudopotential::Get_wall_force_2ndorder(Flow_solver* Flow, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		unsigned int X, Y, Z, alpha;
		double Fx1, Fy1, Fz1;
		double Fx2, Fy2, Fz2;
		//	double cs2 = 1. / Flow->c_s2;
		double s_function;
		for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (Flow->is_solid[{X, Y, Z}] == -1) {
						Fx1 = 0;
						Fy1 = 0;
						Fz1 = 0;
						Fx2 = 0;
						Fy2 = 0;
						Fz2 = 0;
						/// Short range force
						for (alpha = 1; alpha < Flow->Discrete_Velocity; alpha++) {
							s_function = .5 * (Flow->is_solid[{X + Flow->c_alpha[alpha][0], Y + Flow->c_alpha[alpha][1], Z + Flow->c_alpha[alpha][2]}] + 1);
							Fx1 += (Flow->weight[alpha] * Flow->c_alpha[alpha][0] * Psi[X][Y][Z]) * s_function;
							Fy1 += (Flow->weight[alpha] * Flow->c_alpha[alpha][1] * Psi[X][Y][Z]) * s_function;
							Fz1 += (Flow->weight[alpha] * Flow->c_alpha[alpha][2] * Psi[X][Y][Z]) * s_function;

							Fx2 += (Flow->weight[alpha] * Flow->c_alpha[alpha][0] * Psi[X][Y][Z]) * .5 * (Flow->is_solid[{X + 2 * Flow->c_alpha[alpha][0], Y + 2 * Flow->c_alpha[alpha][1], Z + 2 * Flow->c_alpha[alpha][2]}] + 1);
							Fy2 += (Flow->weight[alpha] * Flow->c_alpha[alpha][1] * Psi[X][Y][Z]) * .5 * (Flow->is_solid[{X + 2 * Flow->c_alpha[alpha][0], Y + 2 * Flow->c_alpha[alpha][1], Z + 2 * Flow->c_alpha[alpha][2]}] + 1);
							Fz2 += (Flow->weight[alpha] * Flow->c_alpha[alpha][2] * Psi[X][Y][Z]) * .5 * (Flow->is_solid[{X + 2 * Flow->c_alpha[alpha][0], Y + 2 * Flow->c_alpha[alpha][1], Z + 2 * Flow->c_alpha[alpha][2]}] + 1);
						}
						Flow->force[{X, Y, Z, 0}] = Flow->force[{X, Y, Z, 0}] - Psi[X][Y][Z] * (Gw * Fx1 + 0 * Fx2);
						if (Flow->Dimension > 1)
							Flow->force[{X, Y, Z, 1}] = Flow->force[{X, Y, Z, 1}] - Psi[X][Y][Z] * (Gw * Fy1 + 0 * Fy2);
						if (Flow->Dimension > 2)
							Flow->force[{X, Y, Z, 2}] = Flow->force[{X, Y, Z, 2}] - Psi[X][Y][Z] * (Gw * Fz1 + 0 * Fz2);
					}
				}
			}
		}
	}
}
pseudopotential::~pseudopotential() {}

free_energy::free_energy() {}
void free_energy::Memory_allocation(Parallel_MPI* MPI_parallel) {
	unsigned int X, Y;
	if (MPI_parallel->processor_id != MASTER) {
		kappa_s = new double**[MPI_parallel->dev_end[0]];
		L_rho = new double**[MPI_parallel->dev_end[0]];
		Psi = new double**[MPI_parallel->dev_end[0]];
		for (X = 0; X < MPI_parallel->dev_end[0]; ++X) {
			kappa_s[X] = new double*[MPI_parallel->dev_end[1]];
			L_rho[X] = new double*[MPI_parallel->dev_end[1]];
			Psi[X] = new double*[MPI_parallel->dev_end[1]];
			for (Y = 0; Y < MPI_parallel->dev_end[1]; ++Y) {
				kappa_s[X][Y] = new double[MPI_parallel->dev_end[2]];
				L_rho[X][Y] = new double[MPI_parallel->dev_end[2]];
				Psi[X][Y] = new double[MPI_parallel->dev_end[2]];
			}
		}
	}
}
void free_energy::initialize_field(Geometry* Geo, stl_import* Geo_stl, FreeEnergy_Ini Ini_Field, Parallel_MPI* MPI_parallel, int& tot_sol, std::string filename) {
	/// Open input file
	string input_filename(filename);
	input_filename += ".dat";
	ifstream input_file;  // File is open for READING
	input_file.open(input_filename.c_str(), ios::binary);

	input_file.clear();
	input_file.seekg(0, ios::beg);

	find_line_after_header(input_file, "c\tFree Energy");
	find_line_after_comment(input_file);
	input_file >> kappa;
	input_file.close();
	if (MPI_parallel->processor_id != MASTER) {
		tot_sol = 0;
		Ini_Field(kappa_s, Geo_stl->domain, global_parameters.Nx, global_parameters.Ny, global_parameters.Nz, global_parameters.D_x, global_parameters.D_t, filename, Geo_stl->Source_count, MPI_parallel);
	}
	return;
}
/// ***************************************************** ///
/// COMPUTE KORTWEG STRESS TENSOR                         ///
/// ***************************************************** ///
void free_energy::Kortweg_stress(PseudoPotentialPressure get_pressure, Flow_solver* Flow, Thermal_solver* Thermal, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		unsigned int X, Y, Z, alpha;
		double Fxtemp, Fytemp, Fztemp;
		double cs2 = 1. / Flow->c_s2;
		double dens, temp;
		/// I - Compute potential function PHI (everywhere)
		for (X = 0; X < MPI_parallel->dev_end[0]; ++X) {
			for (Y = 0; Y < MPI_parallel->dev_end[1]; ++Y) {
				for (Z = 0; Z < MPI_parallel->dev_end[2]; ++Z) {
					if (Flow->is_solid[{X, Y, Z}] == FALSE) {
						dens = Flow->density[{X, Y, Z}];
						temp = Thermal->temperature[{X, Y, Z}];
						get_pressure(dens, temp, Flow->pressure[{X, Y, Z}], cs2);
						Psi[X][Y][Z] = sqrt(Flow->density[{X, Y, Z}] * cs2 - Flow->pressure[{X, Y, Z}]);
					}
				}
			}
		}
		/// II - Compute Laplacian of density field (everywhere)
		for (X = MPI_parallel->start_XYZ2[0] - 1; X <= MPI_parallel->end_XYZ2[0] + 1; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1] - 1; Y <= MPI_parallel->end_XYZ2[1] + 1; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2] - 1; Z <= MPI_parallel->end_XYZ2[2] + 1; ++Z) {
					if (Flow->is_solid[{X, Y, Z}] == FALSE) {
						L_rho[X][Y][Z] = 0;
						for (alpha = 0; alpha < Flow->Discrete_Velocity; alpha++) {
							L_rho[X][Y][Z] += Flow->weight[alpha] * Flow->density[{X + Flow->c_alpha[alpha][0], Y + Flow->c_alpha[alpha][1], Z + Flow->c_alpha[alpha][2]}];
						}
						L_rho[X][Y][Z] -= Flow->density[{X, Y, Z}];
						L_rho[X][Y][Z] *= (2. / cs2);
					}
				}
			}
		}
		/// III - Compute Kortweg stress tensor
		for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (Flow->is_solid[{X, Y, Z}] == FALSE) {
						Fxtemp = 0;
						Fytemp = 0;
						Fztemp = 0;
						for (alpha = 1; alpha < Flow->Discrete_Velocity; alpha++) {
							Fxtemp += (2. * Psi[X][Y][Z] * (Flow->weight[alpha] * Flow->c_alpha[alpha][0] * Psi[X + Flow->c_alpha[alpha][0]][Y + Flow->c_alpha[alpha][1]][Z + Flow->c_alpha[alpha][2]])
							           - kappa * Flow->density[{X, Y, Z}] * (Flow->weight[alpha] * Flow->c_alpha[alpha][0] * L_rho[X + Flow->c_alpha[alpha][0]][Y + Flow->c_alpha[alpha][1]][Z + Flow->c_alpha[alpha][2]]));
							Fytemp += (2. * Psi[X][Y][Z] * (Flow->weight[alpha] * Flow->c_alpha[alpha][1] * Psi[X + Flow->c_alpha[alpha][0]][Y + Flow->c_alpha[alpha][1]][Z + Flow->c_alpha[alpha][2]])
							           - kappa * Flow->density[{X, Y, Z}] * (Flow->weight[alpha] * Flow->c_alpha[alpha][1] * L_rho[X + Flow->c_alpha[alpha][0]][Y + Flow->c_alpha[alpha][1]][Z + Flow->c_alpha[alpha][2]]));
							Fztemp += (2. * Psi[X][Y][Z] * (Flow->weight[alpha] * Flow->c_alpha[alpha][2] * Psi[X + Flow->c_alpha[alpha][0]][Y + Flow->c_alpha[alpha][1]][Z + Flow->c_alpha[alpha][2]])
							           - kappa * Flow->density[{X, Y, Z}] * (Flow->weight[alpha] * Flow->c_alpha[alpha][2] * L_rho[X + Flow->c_alpha[alpha][0]][Y + Flow->c_alpha[alpha][1]][Z + Flow->c_alpha[alpha][2]]));
						}
						Flow->force[{X, Y, Z, 0}] = Flow->force[{X, Y, Z, 0}] + Fxtemp * (1. / cs2);
						if (Flow->Dimension > 1)
							Flow->force[{X, Y, Z, 1}] = Flow->force[{X, Y, Z, 1}] + Fytemp * (1. / cs2);
						if (Flow->Dimension > 2)
							Flow->force[{X, Y, Z, 2}] = Flow->force[{X, Y, Z, 2}] + Fztemp * (1. / cs2);
					}
				}
			}
		}
	}
}
void free_energy::wall_interaction(Flow_solver* Flow, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		unsigned int X, Y, Z, alpha;
		double gs_x, gs_y, gs_z, kappa_s, psi;
		kappa_s = 0;
		for (X = MPI_parallel->start_XYZ2[0] - 1; X <= MPI_parallel->end_XYZ2[0] + 1; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1] - 1; Y <= MPI_parallel->end_XYZ2[1] + 1; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2] - 1; Z <= MPI_parallel->end_XYZ2[2] + 1; ++Z) {
					if (Flow->is_solid[{X, Y, Z}] == FALSE) {
						gs_x = 0;
						gs_y = 0;
						gs_z = 0;
						for (alpha = 0; alpha < Flow->Discrete_Velocity; alpha++) {
							psi = 0;
							if (Flow->is_solid[{X + Flow->c_alpha[alpha][0], Y + Flow->c_alpha[alpha][1], Z + Flow->c_alpha[alpha][2]}] == TRUE)
								psi = 1.;
							gs_x += Flow->weight[alpha] * psi * Flow->c_alpha[alpha][0];
							gs_y += Flow->weight[alpha] * psi * Flow->c_alpha[alpha][1];
							gs_z += Flow->weight[alpha] * psi * Flow->c_alpha[alpha][2];
						}
						Flow->force[{X, Y, Z, 0}] += kappa_s * Flow->density[{X, Y, Z}] * gs_x;
						Flow->force[{X, Y, Z, 1}] += kappa_s * Flow->density[{X, Y, Z}] * gs_y;
						Flow->force[{X, Y, Z, 2}] += kappa_s * Flow->density[{X, Y, Z}] * gs_z;
					}
				}
			}
		}
	}
}
void free_energy::Store_pressure(PseudoPotentialPressure get_pressure, Flow_solver* Flow, Thermal_solver* Thermal, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		unsigned int X, Y, Z;
		double cs2, dens, temp;
		cs2 = 1. / Flow->c_s2;
		for (X = 0; X < MPI_parallel->dev_end[0]; ++X) {
			for (Y = 0; Y < MPI_parallel->dev_end[1]; ++Y) {
				for (Z = 0; Z < MPI_parallel->dev_end[2]; ++Z) {
					/* Get pressure from EoS */
					dens = Flow->density[{X, Y, Z}];
					temp = Thermal->temperature[{X, Y, Z}];
					get_pressure(dens, temp, Flow->pressure[{X, Y, Z}], cs2);
					Psi[X][Y][Z] = sqrt(Flow->density[{X, Y, Z}] * cs2 - Flow->pressure[{X, Y, Z}]);
					Flow->pressure[{X, Y, Z}] = Flow->density[{X, Y, Z}] * cs2 - sqr(Psi[X][Y][Z]);
				}
			}
		}
	}
}

free_energy::~free_energy() {}

void SC_pressure(double& rho, double& reduced_temperature, double& pressure, double& cs2) /* SHAN CHEN EoS         */ {
	pressure = 0.5 * (-1. / reduced_temperature) * cs2 * sqr(1. - exp(-rho)) + rho * cs2;
}
void CS_pressure(double& rho, double& reduced_temperature, double& pressure, double& cs2) /* CARNAHAN STARLING EoS */ {
	double a_CS = 1.00, b_CS = 4.00, R = 1.00;
	double critical_temperature = 0.3773 * a_CS / (b_CS * R);
	//	double critical_pressure = 0.18727 * 0.3773 * a_CS / (b_CS * b_CS);
	//	double critical_rho = 0.115;
	double temperature = reduced_temperature * critical_temperature;
	double fac = b_CS * rho / 4.;
	pressure = rho * R * temperature * (1. + fac + fac * fac - fac * fac * fac) / pow(1. - fac, 3) - a_CS * rho * rho;
}
void RK_pressure(double& rho, double& reduced_temperature, double& pressure, double& cs2) /* REDLICH KWONG EoS     */ {
	double a = 0.0408, b = 0.0952, R = 1.00;  // a, b, R -> EOS parameters (non-dimensional)
	double critical_temperature = 0.1961;     //  t_c->critical temperature (tu), p -> pressure (mu*mu/ts^2)
	double temperature = reduced_temperature * critical_temperature;
	double den = sqrt(temperature) * (1. + b * rho);
	pressure = rho * R * temperature / (1. - b * rho) - a * rho * rho / den;  // den: denominator
}
void PR_pressure(double& rho, double& reduced_temperature, double& pressure, double& cs2) /* PENG ROBINSON EoS     */ {
	double a_PR = 1. / 98., b_PR = 2. / 21., R = 1.00, acc = 0.344;
	double critical_temperature = 0.172338378094655 * a_PR / (b_PR * R);
	//	double pressure_critical = 0.013237774472925 * (a_PR / sqr(b_PR));
	//	double critical_rho = 2.541858478656377;

	double temperature = reduced_temperature * critical_temperature;  // t:temperature (tu)
	double alpha = 1. + (0.37464 + 1.54226 * acc - 0.26992 * acc * acc) * (1. - sqrt(reduced_temperature));
	double fac = alpha * alpha / (1. + 2. * b_PR * rho - b_PR * b_PR * rho * rho);
	pressure = rho * R * temperature / (1. - b_PR * rho) - a_PR * rho * rho * fac;
}
void VdW_pressure(double& rho, double& reduced_temperature, double& pressure, double& cs2) /* VANDERWAALS EoS       */ {
	double a_vdW = 2. / 49., b_vdW = 2. / 21. /*, R = 1.00*/;
	//	double critical_rho = 7. / 2.;
	double critical_temperature = (8.0 * a_vdW) / (27.0 * b_vdW);
	// double critical_pressure = 3./4.;
	double temperature = reduced_temperature * critical_temperature;
	// pressure = density * R * temperature / (1. - b * density) - a * density * density;

	pressure = rho * temperature / (1.0 - b_vdW * rho) - a_vdW * rho * rho;
	// pressure = (8./3.) * temperature * rho / (1. - rho/3.) - 3. * rho * rho;
	// pressure = pressure/critical_pressure;
}

void USERDEFINED_FREE_ENERGY(double*** kappa_s, const Solid_field& solid, double N_x, double N_y, double N_z, double dx, double dt, std::string filename, int Zones, Parallel_MPI* MPI_parallel) {
	///----------------------------------------------------------------------///
	///                        COMMAND LINE INPUT                            ///
	///----------------------------------------------------------------------///
	if (MPI_parallel->processor_id != MASTER) {
		double* kappa_s_in;
		int X, Y, Z;

		kappa_s_in = new double[Zones + 1];
		int index;
		for (int i = 0; i < Zones + 1; i++) {
			kappa_s_in[i] = 0;
		}
		/// Open input file
		string input_filename(filename);
		input_filename += ".dat";
		ifstream input_file;  // File is open for READING
		input_file.open(input_filename.c_str(), ios::binary);

		input_file.clear();
		input_file.seekg(0, ios::beg);
		find_line_after_header(input_file, "c\tFree Energy Initial Conditions");
		if (MPI_parallel->processor_id == (MASTER + 1)) {
			std::cout << "FREE ENERGY: User defined Initial conditions \n";
			std::cout << "================================ \n";
		}
		for (int i = 0; i < Zones; i++) {
			find_line_after_comment(input_file);
			input_file >> index;
			input_file >> kappa_s_in[index];
			if (MPI_parallel->processor_id == (MASTER + 1)) {
				std::cout << "Zone : " << index << std::endl;
				std::cout << "Kappa_s: " << kappa_s_in[index] << std::endl;
				std::cout << "-------------------------------- \n";
			}
		}
		input_file.close();
		for (X = 0; X < MPI_parallel->dev_end[0]; X++) {
			for (Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
				for (Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
					// const double zz = fmod(Z - MPI_parallel->start_XYZ2[2] + MPI_parallel->start_XYZ[2] + N_z, N_z);
					kappa_s[X][Y][Z] = kappa_s_in[solid[{X,Y,Z}]];
				}
			}
		}
	}
	return;
}
/// ***************************************************** ///
/// SMOOTHEN DENSITY FIELDS           				      ///
/// ***************************************************** ///
void smoothen_density_multiphase(Flow_solver* Flow, Parallel_MPI* MPI_parallel, unsigned int Nt, double D) {
	unsigned int X, Y, Z;
	double s_x, s_y, s_z;
	if (MPI_parallel->processor_id != MASTER) {
		for (unsigned int t = 0; t < Nt; t++) {
			for (X = 0; X < MPI_parallel->dev_end[0]; ++X) {
				for (Y = 0; Y < MPI_parallel->dev_end[1]; ++Y) {
					for (Z = 0; Z < MPI_parallel->dev_end[2]; ++Z) {
						Flow->pop_old[{X, Y, Z, 0}] = Flow->density[{X, Y, Z}];
						Flow->pop_old[{X, Y, Z, 1}] = Flow->velocity[{X, Y, Z, 0}];
						Flow->pop_old[{X, Y, Z, 2}] = Flow->velocity[{X, Y, Z, 1}];
						Flow->pop_old[{X, Y, Z, 3}] = Flow->velocity[{X, Y, Z, 2}];
					}
				}
			}
			for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
				for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
					for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
						if (Flow->is_solid[{X, Y, Z}] == FALSE) {
							s_x = 1;
							s_y = 1;
							s_z = 1;
							Flow->density[{X, Y, Z}] = D * s_x * (Flow->pop_old[{X + 1, Y, Z, 0}] + Flow->pop_old[{X - 1, Y, Z, 0}] - 2. * Flow->pop_old[{X, Y, Z, 0}])
							                           + D * s_y * (Flow->pop_old[{X, Y + 1, Z, 0}] + Flow->pop_old[{X, Y - 1, Z, 0}] - 2. * Flow->pop_old[{X, Y, Z, 0}])
							                           + D * s_z * (Flow->pop_old[{X, Y, Z + 1, 0}] + Flow->pop_old[{X, Y, Z - 1, 0}] - 2. * Flow->pop_old[{X, Y, Z, 0}])
							                           + Flow->pop_old[{X, Y, Z, 0}];
							Flow->velocity[{X, Y, Z, 0}] = D * s_x * (Flow->pop_old[{X + 1, Y, Z, 1}] + Flow->pop_old[{X - 1, Y, Z, 1}] - 2. * Flow->pop_old[{X, Y, Z, 1}])
							                               + D * s_y * (Flow->pop_old[{X, Y + 1, Z, 1}] + Flow->pop_old[{X, Y - 1, Z, 1}] - 2. * Flow->pop_old[{X, Y, Z, 1}])
							                               + D * s_z * (Flow->pop_old[{X, Y, Z + 1, 1}] + Flow->pop_old[{X, Y, Z - 1, 1}] - 2. * Flow->pop_old[{X, Y, Z, 1}])
							                               + Flow->pop_old[{X, Y, Z, 1}];
							Flow->velocity[{X, Y, Z, 1}] = D * s_x * (Flow->pop_old[{X + 1, Y, Z, 2}] + Flow->pop_old[{X - 1, Y, Z, 2}] - 2. * Flow->pop_old[{X, Y, Z, 2}])
							                               + D * s_y * (Flow->pop_old[{X, Y + 1, Z, 2}] + Flow->pop_old[{X, Y - 1, Z, 2}] - 2. * Flow->pop_old[{X, Y, Z, 2}])
							                               + D * s_z * (Flow->pop_old[{X, Y, Z + 1, 2}] + Flow->pop_old[{X, Y, Z - 1, 2}] - 2. * Flow->pop_old[{X, Y, Z, 2}])
							                               + Flow->pop_old[{X, Y, Z, 2}];
							Flow->velocity[{X, Y, Z, 2}] = D * s_x * (Flow->pop_old[{X + 1, Y, Z, 3}] + Flow->pop_old[{X - 1, Y, Z, 3}] - 2. * Flow->pop_old[{X, Y, Z, 3}])
							                               + D * s_y * (Flow->pop_old[{X, Y + 1, Z, 3}] + Flow->pop_old[{X, Y - 1, Z, 3}] - 2. * Flow->pop_old[{X, Y, Z, 3}])
							                               + D * s_z * (Flow->pop_old[{X, Y, Z + 1, 3}] + Flow->pop_old[{X, Y, Z - 1, 3}] - 2. * Flow->pop_old[{X, Y, Z, 3}])
							                               + Flow->pop_old[{X, Y, Z, 3}];
						}
					}
				}
			}
			Flow->Data_Exchange_Macroscopic(MPI_parallel);
		}
	}
}

/// ***************************************************** ///
/// APPLY MODEL FOR COMPUTATION OF VISCOSITY              ///
/// ***************************************************** ///
void Diffusion_Coefficient_density_dependent(int time, double rho_L, double rho_G, double nu_L, double nu_G, Flow_solver* Flow, Parallel_MPI* MPI_parallel) {
	int t_stop = 300;
	if (MPI_parallel->processor_id != MASTER) {
		for (unsigned X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (unsigned Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (unsigned Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (Flow->is_solid[{X, Y, Z}] == FALSE) {
						/// KINEMATIC VISCOSITY
						if (time < t_stop)
							Flow->viscosity[{X, Y, Z}] = 0.167;
						if (time > t_stop - 1)
							Flow->viscosity[{X, Y, Z}] = nu_L + (nu_G - nu_L) * (rho_L - Flow->density[{X, Y, Z}]) / (rho_L - rho_G);
					}
				}
			}
		}
		for(const Flow_boundary_data& boundary : Flow->boundaries){
		for (const Flow_fluid_boundary_node& node : boundary.fluid_node_data) {
			if (time < t_stop)
				Flow->viscosity[node.flat_idx] = 0.167;
			if (time > t_stop - 1)
				Flow->viscosity[node.flat_idx] = nu_L + (nu_G - nu_L) * (rho_L - Flow->density[node.flat_idx]) / (rho_L - rho_G);
			
			const Flat_index pos_p = node.flat_idx - Flow->viscosity.flat_index(node.n);
			if (time < t_stop)
				Flow->viscosity[pos_p] = 0.167;
			if (time > t_stop - 1)
				Flow->viscosity[pos_p] = nu_L + (nu_G - nu_L) * (rho_L - Flow->density[node.flat_idx]) / (rho_L - rho_G);
		}
		}
	}
}

/// ***************************************************** ///
/// CONSISTENT FIELD INITIALIZATION PSEUDO-POTENTIAL      ///
/// ***************************************************** ///
void initialize_multiphase_pseudopotential(Flow_solver* Flow, pseudopotential* MultiPhase, Thermal_solver* Thermal, Species_solver* Species, Geometry* Geo, Parallel_MPI* MPI_parallel, unsigned int Nt) {
	if (MPI_parallel->processor_id == MASTER + 1) {
		cout << endl;
		cout << "\t|-------------------------------------|" << endl;
		cout << "\t|      ... INITIALIZING FIELD ...     |" << endl;
		cout << "\t|-------------------------------------|" << endl;
		cout << endl;
		srand(1);
	}
	unsigned int X, Y, Z;
	/// ----------------------------------------------------------------------------
	/// allocate memory for temporary holder, for initial velocity distributions
	/// ----------------------------------------------------------------------------
	double**** u_temp = nullptr;
	u_temp = new double***[MPI_parallel->dev_end[0]];
	for (X = 0; X < MPI_parallel->dev_end[0]; ++X) {
		u_temp[X] = new double**[MPI_parallel->dev_end[1]];
		for (Y = 0; Y < MPI_parallel->dev_end[1]; ++Y) {
			u_temp[X][Y] = new double*[MPI_parallel->dev_end[2]];
			for (Z = 0; Z < MPI_parallel->dev_end[2]; ++Z) {
				u_temp[X][Y][Z] = new double[3];
			}
		}
	}
	/// ----------------------------------------------------------------------------
	/// Transfer initial velocity to temporary array
	/// ----------------------------------------------------------------------------
	if (MPI_parallel->processor_id != MASTER) {
		for (X = 0; X < MPI_parallel->dev_end[0]; ++X) {
			for (Y = 0; Y < MPI_parallel->dev_end[1]; ++Y) {
				for (Z = 0; Z < MPI_parallel->dev_end[2]; ++Z) {
					u_temp[X][Y][Z][0] = Flow->velocity[{X, Y, Z, 0}];
					u_temp[X][Y][Z][1] = Flow->velocity[{X, Y, Z, 1}];
					u_temp[X][Y][Z][2] = Flow->velocity[{X, Y, Z, 2}];

					Flow->velocity[{X, Y, Z, 0}] = 0;
					Flow->velocity[{X, Y, Z, 1}] = 0;
					Flow->velocity[{X, Y, Z, 2}] = 0;
				}
			}
		}
	}
	/// ----------------------------------------------------------------------------
	/// Get minimum and maximum density in domain (under the isothermal assumption
	/// would be the gas and liquid densities
	/// ----------------------------------------------------------------------------
	double R_L_temp = -1e16;
	double R_G_temp = 1e16;
	double rho, R_L, R_G;
	if (MPI_parallel->processor_id != MASTER) {
		for (int X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (int Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (int Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (Flow->is_solid[{X, Y, Z}] == FALSE) {
						rho = Flow->density[{X, Y, Z}];
						R_L_temp = MAX(R_L_temp, rho);
						R_G_temp = MIN(R_G_temp, rho);
					}
				}
			}
		}
	}
	MPI_Allreduce(&R_L_temp, &R_L, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(&R_G_temp, &R_G, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
	if (MPI_parallel->processor_id == MASTER) {
		std::cout << setprecision(10) << "rho_L : " << R_L << " rho_G : " << R_G << endl;
	}
	/// ----------------------------------------------------------------------------
	/// main loop simulation
	/// ----------------------------------------------------------------------------
	for (unsigned int tm = 0; tm <= Nt; tm++) {
		if (MPI_parallel->processor_id == MASTER + 1)
			std::cout << "\t step : " << tm << "\n";
		Flow->Data_Exchange_Macroscopic(MPI_parallel);
		/// ******************************
		///  LIMITE DENSITIES TO FLUID/LIQUID
		/// ******************************
		if (MPI_parallel->processor_id != MASTER && tm < 300) {
			for (X = 0; X < MPI_parallel->dev_end[0]; ++X) {
				for (Y = 0; Y < MPI_parallel->dev_end[1]; ++Y) {
					for (Z = 0; Z < MPI_parallel->dev_end[2]; ++Z) {
						if (Flow->density[{X, Y, Z}] > 1.1 * R_L)
							Flow->density[{X, Y, Z}] = 1.1 * R_L;
						if (Flow->density[{X, Y, Z}] < .9 * R_G)
							Flow->density[{X, Y, Z}] = .9 * R_G;
					}
				}
			}
		}
		MultiPhase->Store_pressure((PseudoPotentialPressure)PR_pressure, Flow, Thermal, MPI_parallel);
		Diffusion_Coefficient_density_dependent(0, 100, 1e-5, 0.167, 0.167, Flow, MPI_parallel);
		/// ******************************
		///   PSEUDO-POTENTIAL FORCE
		/// ******************************
		MultiPhase->Get_potential((PseudoPotentialPressure)PR_pressure, Flow, Thermal, MPI_parallel);
		MultiPhase->Get_force_1storder_improved(Flow, MPI_parallel);
		/// MultiPhase->Get_force_2ndorder(Flow, MPI_parallel);
		/// ******************************
		///  LBM ALGORITHM
		/// ******************************
		Flow->LBM_CM_MRT(tm, Thermal, Species, Geo, MPI_parallel);
		/// ******************************
		///  DATA EXCHANGE BETWEEN CORES
		/// ******************************
		Flow->Data_Exchange(MPI_parallel);
		/// ******************************
		///  BOUNDARY CONDITIONS
		/// ******************************
		Flow->BC(tm, Thermal, Species, MPI_parallel);
		/// ******************************
		///  MOMENTS EVALUATION
		/// ******************************
		Flow->momenta(tm, MPI_parallel, Thermal, Species);
	}
	if (MPI_parallel->processor_id != MASTER) {
		for (X = 0; X < MPI_parallel->dev_end[0]; ++X) {
			for (Y = 0; Y < MPI_parallel->dev_end[1]; ++Y) {
				for (Z = 0; Z < MPI_parallel->dev_end[2]; ++Z) {
					if (Flow->density[{X, Y, Z}] > 0.9 * R_L) {
						Flow->velocity[{X, Y, Z, 0}] += u_temp[X][Y][Z][0];
						Flow->velocity[{X, Y, Z, 1}] += u_temp[X][Y][Z][1];
						Flow->velocity[{X, Y, Z, 2}] += u_temp[X][Y][Z][2];
					}
				}
			}
		}
	}

	for (X = 0; X < MPI_parallel->dev_end[0]; ++X) {
		for (Y = 0; Y < MPI_parallel->dev_end[1]; ++Y) {
			for (Z = 0; Z < MPI_parallel->dev_end[2]; ++Z) {
				delete[] u_temp[X][Y][Z];
			}
			delete[] u_temp[X][Y];
		}
		delete[] u_temp[X];
	}
	delete[] u_temp;

	if (MPI_parallel->processor_id == MASTER + 1) {
		cout << endl;
		cout << "\t|----------------------------------------|" << endl;
		cout << "\t|      ... END OF INITIALIZATION ...     |" << endl;
		cout << "\t|----------------------------------------|" << endl;
		cout << endl;
		srand(1);
	}
}
