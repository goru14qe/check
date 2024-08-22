#include <sstream>   // string streams
#include <iostream>  // for the use of 'cout'
#include <cmath>
#include <string.h>
#include <fstream>  // file stream

#include "Phase_Field.h"
#include "ALBORZ_Macros.h"
#include "ALBORZ_SETTINGS.h"
#include "ALBORZ_GlobalVariables.h"
#include "Flow_solver.h"
#include "Thermal_solver.h"
#include "Species_solver.h"
#include "Geometry.h"
#include "io/IO_interface.h"

using namespace std;

Phase_Field::Phase_Field() {
}
void Phase_Field::General_data_input(std::string filename, Parallel_MPI* MPI_parallel) {
	int column_width = 40;
	/// Open input file
	string input_filename(filename);
	input_filename += ".dat";
	ifstream input_file;  // File is open for READING
	input_file.open(input_filename.c_str(), ios::binary);

	input_file.clear();
	input_file.seekg(0, ios::beg);
	find_line_after_header(input_file, "c\tPhase Field Solver");
	find_line_after_comment(input_file);

	//  input_file >> Dimension >> Discrete_Velocity >> M_phase >> Xi_phase;
	//	Xi_phase = Xi_phase/global_parameters.D_x;
	//	M_phase = M_phase * global_parameters.D_t / pow(global_parameters.D_x,2);
	input_file >> Dimension >> Discrete_Velocity >> W_zero >> tau_zero >> lambda;  /// new phase-field added
	input_file >> tet >> MC_inf >> K_sup_sat >> D_sup_sat >> Radi >> L_sat;        /// new phase-field added
	Radi = Radi / global_parameters.D_x;
	input_file >> epsilon[0] >> epsilon[1] >> epsilon[2];
	input_file >> Gamma_phase[0] >> Gamma_phase[1] >> Gamma_phase[2] >> lambdal >> lambdas >> D_S;
	W_zero = W_zero / global_parameters.D_x;      /// new phase-field added
	tau_zero = tau_zero / global_parameters.D_t;  /// new phase-field added
	M_phase = SQ(W_zero) / tau_zero;              /// new phase-field added
	D_sup_sat = D_sup_sat * global_parameters.D_t / SQ(global_parameters.D_x);
	lambdal = lambdal * global_parameters.D_t / SQ(global_parameters.D_x);
	lambdas = lambdas * global_parameters.D_t / SQ(global_parameters.D_x);
	D_S = D_S * global_parameters.D_t / SQ(global_parameters.D_x);
	input_file >> center_x >> center_y >> center_z;
	input_file.close();
	if (MPI_parallel->processor_id == MASTER + 1) {
		std::cout << "Phase field parameters" << endl;
		std::cout << "=====================" << endl;
		std::cout << "Stencil : " << left << "D" << Dimension << "Q" << Discrete_Velocity << endl
				  << endl;
		std::cout << setw(column_width) << left << "W_zero = " << W_zero << endl
				  << setw(column_width) << left << "tau_zero = " << tau_zero << endl;  /// new phase-field added
		std::cout << setw(column_width) << left << "M_phase = " << M_phase << endl;    /// new phase-field added
		std::cout << setw(column_width) << left << "radius = " << Radi << endl;        /// new phase-field added
		std::cout << setw(column_width) << left << "L_sat = " << L_sat << endl
				  << setw(column_width) << left << "lambda = " << lambda << endl;
		std::cout << setw(column_width) << left << "D_sup_sat = " << D_sup_sat << endl
				  << setw(column_width) << left << "MC_inf = " << MC_inf << endl;  /// new phase-field added
		std::cout << setw(column_width) << left << "partition coefficient = " << K_sup_sat << endl
				  << setw(column_width) << endl;  /// new phase-field added
		std::cout << setw(column_width) << left << "Gamma_phase_x = " << Gamma_phase[0] << endl
				  << setw(column_width) << left << "Gamma_phase_y = " << Gamma_phase[1] << endl;  /// new phase-field added
		std::cout << setw(column_width) << left << "Gamma_phase_z = " << Gamma_phase[2] << endl
				  << setw(column_width) << left << "Ds = " << D_S << endl;  /// new phase-field added
		std::cout << setw(column_width) << left << "epsilon_x = " << epsilon[0] << endl
				  << setw(column_width) << left << "epsilon_y = " << epsilon[1] << endl;  /// new phase-field added
		std::cout << setw(column_width) << left << "kappal = " << lambdal << endl
				  << setw(column_width) << left << "kappas = " << lambdas << endl;  /// new phase-field added
		std::cout << setw(column_width) << left << "epsilon_z = " << epsilon[2] << endl
				  << endl;
		std::cout << setw(column_width) << left << "center_x = " << center_x << endl
				  << setw(column_width) << left << "center_y = " << center_y << endl;  /// new phase-field added
		std::cout << setw(column_width) << left << "center_z = " << center_z << endl
				  << endl;
	}
}
void Phase_Field::Memory_allocation(Stencil_Definition Stencil_Def, Parallel_MPI* MPI_parallel) {
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
	phase = Scalar_field::zeros(scalar_sizes);
	previous_phase = phase;
	Production = Scalar_field::zeros(scalar_sizes);
	solid_phase_type = Solid_field::zeros(scalar_sizes);
	omega_p = Scalar_field::zeros(scalar_sizes);

	const Vector_field::Index_vec pop_sizes{x_size,
	                                        y_size,
	                                        z_size,
	                                        Discrete_Velocity};
	pop_p = Vector_field::zeros(pop_sizes);
	pop_old_p = pop_p;
	pop_eq_p = new double[Discrete_Velocity];

	if(!MPI_parallel->is_master()){
		pop_group = Data_exchange_group(*MPI_parallel);
		pop_group.add_population(pop_p, c_alpha);

		macroscopic_group = Data_exchange_group(*MPI_parallel);
		macroscopic_group.add_field(phase);
	}
}

void Phase_Field::initialize_p(Geometry* Geo, stl_import* Geo_stl, Phase_Ini Ini_Phase, Parallel_MPI* MPI_parallel, std::string filename) {
	if (MPI_parallel->processor_id != MASTER) {
		if (Geo->flag == TRUE) {
			for (int X = 0; X < global_parameters.Nx; X++) {
				for (int Y = 0; Y < global_parameters.Ny; Y++) {
					for (int Z = 0; Z < global_parameters.Nz; Z++) {
						solid_phase_type[{X, Y, Z}] = Geo->img[X][Y];
					}
				}
			}
		}
		///***********************************   INITIAL  GEOMETRY  ****************************************
		if (Geo_stl->flag == 1) {
			for (int X = 0; X < MPI_parallel->dev_end[0]; X++) {
				for (int Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
					for (int Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
						solid_phase_type[{X, Y, Z}] = Geo_stl->domain[{X, Y, Z}];
					}
				}
			}
		}
		Ini_Phase(phase, solid_phase_type, global_parameters.Nx, global_parameters.Ny, global_parameters.Nz, filename, Geo_stl->Source_count, MPI_parallel);  // added W_zero in phase-field new

		///------------------------------------------------------------------------------------------///
		if (Dimension == 2) {
			for (int X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; X++) {
				int xx = fmod(X - MPI_parallel->start_XYZ2[0] + MPI_parallel->start_XYZ[0] + global_parameters.Nx, global_parameters.Nx);
				for (int Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; Y++) {
					int yy = fmod(Y - MPI_parallel->start_XYZ2[1] + MPI_parallel->start_XYZ[1] + global_parameters.Ny, global_parameters.Ny);
					for (int Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; Z++) {
						//	int zz = fmod(Z - MPI_parallel->start_XYZ2[2] + MPI_parallel->start_XYZ[2] + global_parameters.Nz, global_parameters.Nz);
						const int cenx = floor((center_x - Geo_stl->x_center) / global_parameters.D_x);
						const int ceny = floor((center_y - Geo_stl->y_center) / global_parameters.D_x);
						//                cenz = floor((center_z - Geo_stl->z_center) / global_parameters.D_x);
						/// 2D
						//                if (zz >= (cenz - 1) && zz <= (cenz + 1)){
						phase[{X, Y, Z}] = tanh((Radi - sqrt(SQ(xx - cenx) + SQ(yy - ceny))) / (sqrt(2.0) * W_zero));
						//            }
					}
				}
			}
		}
		///------------------------------------------------------------------------------------------///
		// if (Dimension == 3){
		// double radz = 2.0;
		// for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; X++) {
		//         xx = fmod(X-MPI_parallel->start_XYZ2[0]+MPI_parallel->start_XYZ[0]+global_parameters.Nx,global_parameters.Nx);
		//         for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; Y++) {
		//             yy = fmod(Y-MPI_parallel->start_XYZ2[1]+MPI_parallel->start_XYZ[1]+Ny,Ny);
		//             for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; Z++) {
		//                 zz = fmod(Z-MPI_parallel->start_XYZ2[2]+MPI_parallel->start_XYZ[2]+Nz,Nz);
		//                 cenx = floor((center_x - Geo_stl->x_center) / global_parameters.D_x);
		//                 ceny = floor((center_y - Geo_stl->y_center) / global_parameters.D_x);
		//                 cenz = floor((center_z - Geo_stl->z_center) / global_parameters.D_x);
		//                 if (zz >= (cenz - radz) && zz <= (cenz + radz)){
		////                         phase[{X,Y,Z}] = 0.5*(tanh((Radi - sqrt(SQ(xx -  cenx) + SQ(yy - ceny)))/(sqrt(2.0) * W_zero) + tanh((radz - fabs(zz - cenz))/sqrt(2.0 * W_zero)));
		//                phase[{X,Y,Z}] = tanh((Radi - sqrt(SQ(xx -  cenx) + SQ(yy - ceny)))/(sqrt(2.0) * W_zero *global_parameters.D_x));
		//                }
		//            }
		//        }
		//    }
		//}
		///********************************* END OF GEOMETRY INITIAL ************************************************

		///----------------------------------------------------------------------------------------------------------
		for (int X = 0; X < MPI_parallel->dev_end[0]; ++X) {
			for (int Y = 0; Y < MPI_parallel->dev_end[1]; ++Y) {
				for (int Z = 0; Z < MPI_parallel->dev_end[2]; ++Z) {
					previous_phase[{X, Y, Z}] = phase[{X, Y, Z}];
					Production[{X, Y, Z}] = 0;
					omega_p[{X, Y, Z}] = 1.0;
					//				    omega_p[{X,Y,Z}] = 1./( c_s2 * M_phase + 0.5);
				}
			}
		}
	}
}
void Phase_Field::initialize_pop_eq_p(Parallel_MPI* MPI_parallel, std::string filename) {
	int X, Y, Z;
	if (MPI_parallel->processor_id != MASTER) {
		/// Initialize the populations
		// Use the equilibrium populations corresponding to the initialized phase
		for (X = 0; X < MPI_parallel->dev_end[0]; ++X) {
			for (Y = 0; Y < MPI_parallel->dev_end[1]; ++Y) {
				for (Z = 0; Z < MPI_parallel->dev_end[2]; ++Z) {
					//			    n_p[0] = 0.5*(phase[{(X+1)%MPI_parallel->dev_end[0],Y,Z}] - phase[(X-1+MPI_parallel->dev_end[0])%MPI_parallel->dev_end[0]][Y][Z]);// new phase-field added
					//			    n_p[1] = 0.5*(phase[{X,(Y+1)%MPI_parallel->dev_end[1],Z}] - phase[X][(Y-1+MPI_parallel->dev_end[1])%MPI_parallel->dev_end[1]][Z]);// new phase-field added
					//			    n_p[2] = 0.5*(phase[{X,Y,(Z+1)%MPI_parallel->dev_end[2}]] - phase[{X,Y,(Z-1+MPI_parallel->dev_end[2}])%MPI_parallel->dev_end[2]]);// new phase-field added
					//
					//			    n_p[0] /= -(sqrt(pow(n[0],2) + pow(n[1],2) + pow(n[2],2)) + 1e-20);
					//			    n_p[1] /= -(sqrt(pow(n[0],2) + pow(n[1],2) + pow(n[2],2)) + 1e-20);
					//			    n_p[2] /= -(sqrt(pow(n[0],2) + pow(n[1],2) + pow(n[2],2)) + 1e-20);

					equilibrium_p(phase[{X, Y, Z}], 0, n_p);
					for (int c_i = 0; c_i < Discrete_Velocity; ++c_i) {
						pop_old_p[{X, Y, Z, c_i}] = pop_eq_p[c_i];
						pop_p[{X, Y, Z, c_i}] = pop_eq_p[c_i];
					}
				}
			}
		}
	}
}
void Phase_Field::initialize_BC_p(Geometry* Geo, stl_import* Geo_stl, Parallel_MPI* MPI_parallel, std::string filename) {
	if (MPI_parallel->processor_id != MASTER) {
		int X, Y, Z, i, alpha;
		int number_of_BC = 0;
		std::vector<phase_boundary_data> temp;
		std::vector<int> index;
		int** intersection;
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
		} while (Line1.find("c\tPhase Field Boundary Conditions") == string::npos);
		do {
			beg_line = input_file.tellg();
			input_file >> comment_indicator;
			if (comment_indicator == '#') {
				std::getline(input_file, Line1);
			}
		} while (comment_indicator == '#');
		input_file.seekg(beg_line);
		input_file >> number_of_BC;
		temp.resize(number_of_BC);
		index.resize(number_of_BC);
		intersection = new int*[number_of_BC];
		for (i = 0; i < number_of_BC; i++) {
			intersection[i] = new int[2];
			std::getline(input_file, Line1);
			do {
				beg_line = input_file.tellg();
				input_file >> comment_indicator;
				if (comment_indicator == '#') {
					std::getline(input_file, Line1);
				}
			} while (comment_indicator == '#');
			input_file.seekg(beg_line);
			input_file >> index[i] >> intersection[i][0] >> intersection[i][1] >> temp[i].type;
			input_file >> temp[i].n[0] >> temp[i].n[1] >> temp[i].n[2];
			switch (temp[i].type) {
				case 1:  /// --> Zero temperature BC
				{
					temp[i].Phase = 0;
					break;
				}
				case 2:  /// ---> Non-zero temperature BC
				{
					input_file >> temp[i].Phase;
					;
					break;
				}
				case 3:  /// --> Zero-gradient (1st-order) BC
				{
					temp[i].Phase = 0;
					break;
				}
				case 4:  /// --> Constant flux (1st-order) BC
				{
					temp[i].Phase = 0;
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
			output_filename << "Phase_Boundary_Conditions.dat";
			ofstream BC_output;
			BC_output.open(output_filename.str().c_str(), fstream::trunc);
			BC_output << "PHASE: Boundary Conditions \n";
			BC_output << "================================ \n";
			for (i = 0; i < number_of_BC; i++) {
				BC_output << "BOUNDARY INDEX : " << i + 1 << "\t TYPE : " << temp[i].type << std::endl;
				BC_output << "PHASE : " << temp[i].Phase << std::endl;
				BC_output << "-------------------------------- \n";
			}
			if (number_of_BC == 0) BC_output << "NO PHASE BOUNDARY CONDITIONS\n";
			BC_output.close();
		}
		input_file.close();
		///   ------------------->  Find and store Boundary nodes
		bool BC_flag = 0;
		for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					for (i = 0; i < number_of_BC; i++) {
						if (Geo_stl->domain[{X, Y, Z}] == intersection[i][0]) {
							for (alpha = 1; alpha < Discrete_Velocity; alpha++) {
								const int Xp = X - c_alpha[alpha][0];
								const int Yp = Y - c_alpha[alpha][1];
								const int Zp = Z - c_alpha[alpha][2];
								if (Geo_stl->domain[{Xp, Yp, Zp}] == intersection[i][1]) {
									BC_flag = 1;
								}
							}
							if (BC_flag == 1) {
								temp[i].X = X;
								temp[i].Y = Y;
								temp[i].Z = Z;
								Boundaries.push_back(temp[i]);
								BC_flag = 0;
							}
						}
					}
				}
			}
		}
	}
}

void Phase_Field::equilibrium_p(double phase, double coeff, double* n) {
	double Hxx;
	double Hyy;
	double Hzz;
	for (int alpha = 0; alpha < Discrete_Velocity; alpha++) {
		Hxx = c_alpha[alpha][0] * c_alpha[alpha][0] - 1. / c_s2;
		Hyy = c_alpha[alpha][1] * c_alpha[alpha][1] - 1. / c_s2;
		Hzz = c_alpha[alpha][2] * c_alpha[alpha][2] - 1. / c_s2;
		pop_eq_p[alpha] = weight[alpha] * (phase + coeff * c_s2 * DOT(c_alpha[alpha], n) + 0.5 * phase * c_s2 * ((SQ(Gamma_phase[0]) - 1) * Hxx + (SQ(Gamma_phase[1]) - 1) * Hyy + (SQ(Gamma_phase[2]) - 1) * Hzz));
	}
}
void Phase_Field::Crystal_DW(int time, Thermal_solver* Thermal, Species_solver* Species, Parallel_MPI* MPI_parallel) {  /// for Temperature and Species
	if (MPI_parallel->processor_id != MASTER) {
		for (int X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (int Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (int Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (solid_phase_type[{X, Y, Z}] == -1) {
						Production[{X, Y, Z}] = 0.0;
						Production[{X, Y, Z}] = (phase[{X, Y, Z}] + lambda * (MC_inf * Species->mass_fraction[{X, Y, Z, 0}] - Thermal->temperature[{X, Y, Z}]) * (1. - phase[{X, Y, Z}] * phase[{X, Y, Z}])) * (1. - phase[{X, Y, Z}] * phase[{X, Y, Z}]) / tau_zero;
					}
				}
			}
		}
	}
}
void Phase_Field::Crystal_DW_T(int time, Thermal_solver* Thermal, Species_solver* Species, Parallel_MPI* MPI_parallel) {  /// for Temperature
	if (MPI_parallel->processor_id != MASTER) {
		for (int X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (int Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (int Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (solid_phase_type[{X, Y, Z}] == -1) {
						Production[{X, Y, Z}] = 0.0;
						Production[{X, Y, Z}] = (phase[{X, Y, Z}] - lambda * Thermal->temperature[{X, Y, Z}] * (1. - phase[{X, Y, Z}] * phase[{X, Y, Z}])) * (1. - phase[{X, Y, Z}] * phase[{X, Y, Z}]) / tau_zero;
					}
				}
			}
		}
	}
}
void Phase_Field::Crystal_DW_S(int time, Thermal_solver* Thermal, Species_solver* Species, Parallel_MPI* MPI_parallel) {  /// for Species
	if (MPI_parallel->processor_id != MASTER) {
		for (int X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (int Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (int Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (solid_phase_type[{X, Y, Z}] == -1) {
						Production[{X, Y, Z}] = 0.0;
						Production[{X, Y, Z}] = (phase[{X, Y, Z}] + lambda * Species->mass_fraction[{X, Y, Z, 0}] * (1. - phase[{X, Y, Z}] * phase[{X, Y, Z}])) * (1. - phase[{X, Y, Z}] * phase[{X, Y, Z}]) / tau_zero;
					}
				}
			}
		}
	}
}
void Phase_Field::Crystal_DW_snow(int time, Species_solver* Species, Parallel_MPI* MPI_parallel) {  /// added time
	if (MPI_parallel->processor_id != MASTER) {
		double B;
		double mag_n;
		for (int X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (int Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (int Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (solid_phase_type[{X, Y, Z}] == -1) {
						Production[{X, Y, Z}] = 0.0;
						///-----------------------------  Qianyan Tan 3D  for n_x^2 + n_y^2 + n_z^2 = 1 ------------------------------///
						//                        n_p[0] = 1.0/6 * (phase[{X + 1,Y,Z}] - phase[{X - 1,Y,Z}]) + 1.0/12 * (phase[{X + 1,Y + 1,Z}] - phase[{X - 1,Y + 1,Z}] + phase[{X + 1,Y - 1,Z}] - phase[{X - 1,Y - 1,Z}] + phase[{X + 1,Y,Z + 1}] - phase[{X - 1,Y,Z + 1}] + phase[{X + 1,Y,Z - 1}] - phase[{X - 1,Y,Z - 1}]);
						//                        n_p[1] = 1.0/6 * (phase[{X,Y + 1,Z}] - phase[{X,Y - 1,Z}]) + 1.0/12 * (phase[{X + 1,Y + 1,Z}] - phase[{X + 1,Y - 1,Z}] + phase[{X - 1,Y + 1,Z}] - phase[{X - 1,Y - 1,Z}] + phase[{X,Y + 1,Z + 1}] - phase[{X,Y - 1,Z + 1}] + phase[{X,Y + 1,Z - 1}] - phase[{X,Y - 1,Z - 1}]);
						//                        n_p[2] = 1.0/6 * (phase[{X,Y,Z + 1}] - phase[{X,Y,Z - 1}]) + 1.0/12 * (phase[{X + 1,Y,Z + 1}] - phase[{X + 1,Y,Z - 1}] + phase[{X - 1,Y,Z + 1}] - phase[{X - 1,Y,Z - 1}] + phase[{X,Y + 1,Z + 1}] - phase[{X,Y + 1,Z - 1}] + phase[{X,Y - 1,Z + 1}] - phase[{X,Y - 1,Z - 1}]);
						//                        mag_n = sqrt(SQ(n_p[0]) + SQ(n_p[1]) + SQ(n_p[2]));
						//                        n_p[0] = -n_p[0]/(mag_n + 1e-20);
						//                        n_p[1] = -n_p[1]/(mag_n + 1e-20);
						//                        n_p[2] = -n_p[2]/(mag_n + 1e-20);
						//                        B = sqrt(1.0 + (SQ(Gamma_phase[2]) - 1.0) * SQ(n_p[2]));
						///------------------------------  End of 3D n_x^2 + n_y^2 + n_z^2 = 1  ---------------------------------------///

						///-----------------------------  Qianyan Tan 2D/3D  for n_x^2 + n_y^2 = 1 ------------------------------///
						n_p[0] = 1.0 / 6 * (phase[{X + 1, Y, Z}] - phase[{X - 1, Y, Z}]) + 1.0 / 12 * (phase[{X + 1, Y + 1, Z}] - phase[{X - 1, Y + 1, Z}] + phase[{X + 1, Y - 1, Z}] - phase[{X - 1, Y - 1, Z}] + phase[{X + 1, Y, Z + 1}] - phase[{X - 1, Y, Z + 1}] + phase[{X + 1, Y, Z - 1}] - phase[{X - 1, Y, Z - 1}]);
						n_p[1] = 1.0 / 6 * (phase[{X, Y + 1, Z}] - phase[{X, Y - 1, Z}]) + 1.0 / 12 * (phase[{X + 1, Y + 1, Z}] - phase[{X + 1, Y - 1, Z}] + phase[{X - 1, Y + 1, Z}] - phase[{X - 1, Y - 1, Z}] + phase[{X, Y + 1, Z + 1}] - phase[{X, Y - 1, Z + 1}] + phase[{X, Y + 1, Z - 1}] - phase[{X, Y - 1, Z - 1}]);
						n_p[2] = 1.0 / 6 * (phase[{X, Y, Z + 1}] - phase[{X, Y, Z - 1}]) + 1.0 / 12 * (phase[{X + 1, Y, Z + 1}] - phase[{X + 1, Y, Z - 1}] + phase[{X - 1, Y, Z + 1}] - phase[{X - 1, Y, Z - 1}] + phase[{X, Y + 1, Z + 1}] - phase[{X, Y + 1, Z - 1}] + phase[{X, Y - 1, Z + 1}] - phase[{X, Y - 1, Z - 1}]);
						mag_n = sqrt(SQ(n_p[0]) + SQ(n_p[1]) + SQ(n_p[2]));
						n_p[0] = -n_p[0] / (mag_n + 1e-20);
						n_p[1] = -n_p[1] / (mag_n + 1e-20);
						n_p[2] = -n_p[2] / (mag_n + 1e-20);
						B = sqrt(SQ(Gamma_phase[0] * n_p[0]) + SQ(Gamma_phase[1] * n_p[1]) + SQ(Gamma_phase[2] * n_p[2]));

						///------------------------------  End of 3D n_x^2 + n_y^2 = 1  ---------------------------------------///

						///------------  3D as = 1 + sinnphi * eps_xy * cos(6theta) + eps_z * coss(2*phi) ----------///
						/// tan  3D gamma = 1 then scale equal to original equation
						//                        n_p[0] = 1.0/6 * (phase[{X + 1,Y,Z}] - phase[{X - 1,Y,Z}]) + 1.0/12 * (phase[{X + 1,Y + 1,Z}] - phase[{X - 1,Y + 1,Z}] + phase[{X + 1,Y - 1,Z}] - phase[{X - 1,Y - 1,Z}] + phase[{X + 1,Y,Z + 1}] - phase[{X - 1,Y,Z + 1}] + phase[{X + 1,Y,Z - 1}] - phase[{X - 1,Y,Z - 1}]);
						//                        n_p[1] = 1.0/6 * (phase[{X,Y + 1,Z}] - phase[{X,Y - 1,Z}]) + 1.0/12 * (phase[{X + 1,Y + 1,Z}] - phase[{X + 1,Y - 1,Z}] + phase[{X - 1,Y + 1,Z}] - phase[{X - 1,Y - 1,Z}] + phase[{X,Y + 1,Z + 1}] - phase[{X,Y - 1,Z + 1}] + phase[{X,Y + 1,Z - 1}] - phase[{X,Y - 1,Z - 1}]);
						//                        n_p[2] = 1.0/6 * (phase[{X,Y,Z + 1}] - phase[{X,Y,Z - 1}]) + 1.0/12 * (phase[{X + 1,Y,Z + 1}] - phase[{X + 1,Y,Z - 1}] + phase[{X - 1,Y,Z + 1}] - phase[{X - 1,Y,Z - 1}] + phase[{X,Y + 1,Z + 1}] - phase[{X,Y + 1,Z - 1}] + phase[{X,Y - 1,Z + 1}] - phase[{X,Y - 1,Z - 1}]);
						//                        mag_n = sqrt(SQ(n_p[0]) + SQ(n_p[1]) + (1.0 / SQ(Gamma_z)) * SQ(n_p[2]));
						//                        n_p[2] = -((1.0/Gamma_z) * n_p[2])/(mag_n + 1e-20);
						//                        B = sqrt(1.0 + (SQ(Gamma_z) - 1.0) * SQ(n_p[2]));
						///------------------------------   END OF 3D --------------------------------------------

						///-------------------------------  2D n_z control -----------------------------///
						//                        n_p[0] = 1.0/6 * (phase[{X + 1,Y,Z}] - phase[{X - 1,Y,Z}]) + 1.0/12 * (phase[{X + 1,Y + 1,Z}] - phase[{X - 1,Y + 1,Z}] + phase[{X + 1,Y - 1,Z}] - phase[{X - 1,Y - 1,Z}] + phase[{X + 1,Y,Z + 1}] - phase[{X - 1,Y,Z + 1}] + phase[{X + 1,Y,Z - 1}] - phase[{X - 1,Y,Z - 1}]);
						//                        n_p[1] = 1.0/6 * (phase[{X,Y + 1,Z}] - phase[{X,Y - 1,Z}]) + 1.0/12 * (phase[{X + 1,Y + 1,Z}] - phase[{X + 1,Y - 1,Z}] + phase[{X - 1,Y + 1,Z}] - phase[{X - 1,Y - 1,Z}] + phase[{X,Y + 1,Z + 1}] - phase[{X,Y - 1,Z + 1}] + phase[{X,Y + 1,Z - 1}] - phase[{X,Y - 1,Z - 1}]);
						//                        const double mag_n_xy = sqrt(SQ(n_p[0]) + SQ(n_p[1]));
						//                        mag_n = sqrt(SQ(n_p[0]) + SQ(n_p[1]) + (SQ(n_p[0]) + SQ(n_p[1]))/SQ(tan(tet)));
						//                        n_p[0] = -n_p[0]/(mag_n_xy + 1e-20);
						//                        n_p[1] = -n_p[1]/(mag_n_xy + 1e-20);
						//                        n_p[2] = cos(tet);
						//                        B = sqrt(1.0 + (SQ(Gamma_phase[2]) - 1.0) * SQ(n_p[2]));
						///------------------------------ end of 2D n_z control  -------------------------///
						Production[{X, Y, Z}] = (phase[{X, Y, Z}] + lambda * B * Species->mass_fraction[{X, Y, Z, 0}] * (1. - phase[{X, Y, Z}] * phase[{X, Y, Z}])) * (1. - phase[{X, Y, Z}] * phase[{X, Y, Z}]) / tau_zero;
					}
				}
			}
		}
	}
}

void Phase_Field::Crystal_Heat(int time, Thermal_solver* Thermal, Parallel_MPI* MPI_parallel) {  /// added time
	if (MPI_parallel->processor_id != MASTER) {
		for (int X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (int Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (int Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (solid_phase_type[{X, Y, Z}] == -1) {
						Thermal->Production[{X, Y, Z}] = 0.0;
						Thermal->Production[{X, Y, Z}] += 0.5 * (phase[{X, Y, Z}] - previous_phase[{X, Y, Z}]);
					}
				}
			}
		}
	}
}
void Phase_Field::Thermal_Heat(int time, Thermal_solver* Thermal, Parallel_MPI* MPI_parallel) {  /// added time
	if (MPI_parallel->processor_id != MASTER) {
		// const double	H_cryst = 25000;
		//	const double cpL = 75;
		//	const double cpS = 340.0;
		//	const double c_average = 0.5 * (1.0 + phase[{X,Y,Z}]) * cpS + 0.5 * (1.0 - phase[{X,Y,Z}]) * cpL;
		//  const double T_average = 300;
		//	const double L = H_cryst / (T_average * T_average);
		for (int X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (int Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (int Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (solid_phase_type[{X, Y, Z}] == -1) {
						Thermal->Production[{X, Y, Z}] = 0.0;
						Thermal->Production[{X, Y, Z}] += 0.5 * (phase[{X, Y, Z}] - previous_phase[{X, Y, Z}]);
					}
				}
			}
		}
	}
}
void Phase_Field::Crystal_Specie_snow(int time, Species_solver* Species, Parallel_MPI* MPI_parallel) {  /// added time
	if (MPI_parallel->processor_id != MASTER) {
		double B;
		double mag_n;

		for (int X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (int Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (int Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (solid_phase_type[{X, Y, Z}] == -1) {
						Species->Production[{X, Y, Z, 0}] = 0.0;

						///-----------------------------  Qianyan Tan 3D  for n_x^2 + n_y^2 + n_z^2 = 1 ------------------------------///
						//                        n_p[0] = 1.0/6 * (phase[{X + 1,Y,Z}] - phase[{X - 1,Y,Z}]) + 1.0/12 * (phase[{X + 1,Y + 1,Z}] - phase[{X - 1,Y + 1,Z}] + phase[{X + 1,Y - 1,Z}] - phase[{X - 1,Y - 1,Z}] + phase[{X + 1,Y,Z + 1}] - phase[{X - 1,Y,Z + 1}] + phase[{X + 1,Y,Z - 1}] - phase[{X - 1,Y,Z - 1}]);
						//                        n_p[1] = 1.0/6 * (phase[{X,Y + 1,Z}] - phase[{X,Y - 1,Z}]) + 1.0/12 * (phase[{X + 1,Y + 1,Z}] - phase[{X + 1,Y - 1,Z}] + phase[{X - 1,Y + 1,Z}] - phase[{X - 1,Y - 1,Z}] + phase[{X,Y + 1,Z + 1}] - phase[{X,Y - 1,Z + 1}] + phase[{X,Y + 1,Z - 1}] - phase[{X,Y - 1,Z - 1}]);
						//                        n_p[2] = 1.0/6 * (phase[{X,Y,Z + 1}] - phase[{X,Y,Z - 1}]) + 1.0/12 * (phase[{X + 1,Y,Z + 1}] - phase[{X + 1,Y,Z - 1}] + phase[{X - 1,Y,Z + 1}] - phase[{X - 1,Y,Z - 1}] + phase[{X,Y + 1,Z + 1}] - phase[{X,Y + 1,Z - 1}] + phase[{X,Y - 1,Z + 1}] - phase[{X,Y - 1,Z - 1}]);
						//                        mag_n = sqrt(SQ(n_p[0]) + SQ(n_p[1]) + SQ(n_p[2]));
						//                        n_p[0] = -n_p[0]/(mag_n + 1e-20);
						//                        n_p[1] = -n_p[1]/(mag_n + 1e-20);
						//                        n_p[2] = -n_p[2]/(mag_n + 1e-20);

						///------------------------------  End of 3D n_x^2 + n_y^2 + n_z^2 = 1  ---------------------------------------///

						///-----------------------------  Qianyan Tan 2D/3D  for n_x^2 + n_y^2 = 1 ------------------------------///
						n_p[0] = 1.0 / 6 * (phase[{X + 1, Y, Z}] - phase[{X - 1, Y, Z}]) + 1.0 / 12 * (phase[{X + 1, Y + 1, Z}] - phase[{X - 1, Y + 1, Z}] + phase[{X + 1, Y - 1, Z}] - phase[{X - 1, Y - 1, Z}] + phase[{X + 1, Y, Z + 1}] - phase[{X - 1, Y, Z + 1}] + phase[{X + 1, Y, Z - 1}] - phase[{X - 1, Y, Z - 1}]);
						n_p[1] = 1.0 / 6 * (phase[{X, Y + 1, Z}] - phase[{X, Y - 1, Z}]) + 1.0 / 12 * (phase[{X + 1, Y + 1, Z}] - phase[{X + 1, Y - 1, Z}] + phase[{X - 1, Y + 1, Z}] - phase[{X - 1, Y - 1, Z}] + phase[{X, Y + 1, Z + 1}] - phase[{X, Y - 1, Z + 1}] + phase[{X, Y + 1, Z - 1}] - phase[{X, Y - 1, Z - 1}]);
						n_p[2] = 1.0 / 6 * (phase[{X, Y, Z + 1}] - phase[{X, Y, Z - 1}]) + 1.0 / 12 * (phase[{X + 1, Y, Z + 1}] - phase[{X + 1, Y, Z - 1}] + phase[{X - 1, Y, Z + 1}] - phase[{X - 1, Y, Z - 1}] + phase[{X, Y + 1, Z + 1}] - phase[{X, Y + 1, Z - 1}] + phase[{X, Y - 1, Z + 1}] - phase[{X, Y - 1, Z - 1}]);
						mag_n = sqrt(SQ(n_p[0]) + SQ(n_p[1]) + SQ(n_p[2]));
						n_p[0] = -n_p[0] / (mag_n + 1e-20);
						n_p[1] = -n_p[1] / (mag_n + 1e-20);
						n_p[2] = -n_p[2] / (mag_n + 1e-20);
						B = sqrt(SQ(Gamma_phase[0] * n_p[0]) + SQ(Gamma_phase[1] * n_p[1]) + SQ(Gamma_phase[2] * n_p[2]));

						///------------------------------  End of 3D n_x^2 + n_y^2 = 1  ---------------------------------------///

						///------------  3D as = 1 + sinnphi * eps_xy * cos(6theta) + eps_z * coss(2*phi) ----------///
						/// tan  3D gamma = 1 then scale equal to original equation

						//                        n_p[0] = 1.0/6 * (phase[{X + 1,Y,Z}] - phase[{X - 1,Y,Z}]) + 1.0/12 * (phase[{X + 1,Y + 1,Z}] - phase[{X - 1,Y + 1,Z}] + phase[{X + 1,Y - 1,Z}] - phase[{X - 1,Y - 1,Z}] + phase[{X + 1,Y,Z + 1}] - phase[{X - 1,Y,Z + 1}] + phase[{X + 1,Y,Z - 1}] - phase[{X - 1,Y,Z - 1}]);
						//                        n_p[1] = 1.0/6 * (phase[{X,Y + 1,Z}] - phase[{X,Y - 1,Z}]) + 1.0/12 * (phase[{X + 1,Y + 1,Z}] - phase[{X + 1,Y - 1,Z}] + phase[{X - 1,Y + 1,Z}] - phase[{X - 1,Y - 1,Z}] + phase[{X,Y + 1,Z + 1}] - phase[{X,Y - 1,Z + 1}] + phase[{X,Y + 1,Z - 1}] - phase[{X,Y - 1,Z - 1}]);
						//                        n_p[2] = 1.0/6 * (phase[{X,Y,Z + 1}] - phase[{X,Y,Z - 1}]) + 1.0/12 * (phase[{X + 1,Y,Z + 1}] - phase[{X + 1,Y,Z - 1}] + phase[{X - 1,Y,Z + 1}] - phase[{X - 1,Y,Z - 1}] + phase[{X,Y + 1,Z + 1}] - phase[{X,Y + 1,Z - 1}] + phase[{X,Y - 1,Z + 1}] - phase[{X,Y - 1,Z - 1}]);
						//                        mag_n = sqrt(SQ(n_p[0]) + SQ(n_p[1]) + (1.0 / SQ(Gamma_z)) * SQ(n_p[2]));
						//                        n_p[2] = -((1.0/Gamma_z) * n_p[2])/(mag_n + 1e-20);
						//                        B = sqrt(1.0 + (SQ(Gamma_z) - 1.0) * SQ(n_p[2]));
						///------------------------------   END OF 3D --------------------------------------------

						///-------------------------------------  2D n_z control ----------------------------------------------------///
						//                        n_p[0] = 1.0/6 * (phase[{X + 1,Y,Z}] - phase[{X - 1,Y,Z}]) + 1.0/12 * (phase[{X + 1,Y + 1,Z}] - phase[{X - 1,Y + 1,Z}] + phase[{X + 1,Y - 1,Z}] - phase[{X - 1,Y - 1,Z}] + phase[{X + 1,Y,Z + 1}] - phase[{X - 1,Y,Z + 1}] + phase[{X + 1,Y,Z - 1}] - phase[{X - 1,Y,Z - 1}]);
						//                        n_p[1] = 1.0/6 * (phase[{X,Y + 1,Z}] - phase[{X,Y - 1,Z}]) + 1.0/12 * (phase[{X + 1,Y + 1,Z}] - phase[{X + 1,Y - 1,Z}] + phase[{X - 1,Y + 1,Z}] - phase[{X - 1,Y - 1,Z}] + phase[{X,Y + 1,Z + 1}] - phase[{X,Y - 1,Z + 1}] + phase[{X,Y + 1,Z - 1}] - phase[{X,Y - 1,Z - 1}]);
						//                        const double mag_n_xy = sqrt(SQ(n_p[0]) + SQ(n_p[1]));
						//                        mag_n = sqrt(SQ(n_p[0]) + SQ(n_p[1]) + (SQ(n_p[0]) + SQ(n_p[1]))/SQ(tan(tet)));
						//                        n_p[0] = -n_p[0]/(mag_n_xy + 1e-20);
						//                        n_p[1] = -n_p[1]/(mag_n_xy + 1e-20);
						//                        n_p[2] = cos(tet);
						//                        B = sqrt(1.0 + (SQ(Gamma_phase[2]) - 1.0) * SQ(n_p[2]));
						///------------------------------------ end of 2D n_z control  --------------------------------------///
						Species->Production[{X, Y, Z, 0}] += -0.5 * B * L_sat * (phase[{X, Y, Z}] - previous_phase[{X, Y, Z}]);
					}
				}
			}
		}
	}
}
void Phase_Field::Crystal_Species(int time, Species_solver* Species, Parallel_MPI* MPI_parallel) {  /// added time
	if (MPI_parallel->processor_id != MASTER) {
		int X, Y, Z;
		for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (solid_phase_type[{X, Y, Z}] == -1) {
						Species->Production[{X, Y, Z, 0}] = 0.0;
						Species->Production[{X, Y, Z, 0}] += -0.5 * (phase[{X, Y, Z}] - previous_phase[{X, Y, Z}]);
					}
				}
			}
		}
	}
}
void Phase_Field::Diffusion_Coefficient_computation_T(int time, Thermal_solver* Thermal, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		int X, Y, Z;
		for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (solid_phase_type[{X, Y, Z}] == -1) {
						Thermal->thermal_diffusion_coefficient[{X, Y, Z}] = 0.0;
						Thermal->thermal_diffusion_coefficient[{X, Y, Z}] = 0.5 * lambdal * (1.0 - phase[{X, Y, Z}]) + 0.5 * lambdas * (1.0 + phase[{X, Y, Z}]);
					}
				}
			}
		}
	}
}
void Phase_Field::Diffusion_Coefficient_computation_S(int time, Species_solver* Species, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		int X, Y, Z;
		for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (solid_phase_type[{X, Y, Z}] == -1) {
						Species->diffusion_coefficient[{X, Y, Z, 0}] = 0.0;
						Species->diffusion_coefficient[{X, Y, Z, 0}] = D_S * (1.0 - phase[{X, Y, Z}]);
					}
				}
			}
		}
	}
}
void Phase_Field::Force_on_Fluid(int time, Flow_solver* Flow, Parallel_MPI* MPI_parallel) {  /// new phase_field added
	if (MPI_parallel->processor_id != MASTER) {
		constexpr double h = 2.757;
		double factor, Psi;
		for (int X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (int Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (int Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (solid_phase_type[{X, Y, Z}] == -1) {
						Psi = 0.5 * (1. + phase[{X, Y, Z}]);
						factor = -2. * h * Flow->density[{X, Y, Z}] * Flow->viscosity[{X, Y, Z}] * pow(Psi / W_zero, 2) * (1. - Psi);
						Flow->force[{X, Y, Z, 0}] += factor * Flow->velocity[{X, Y, Z, 0}];
						Flow->force[{X, Y, Z, 1}] += factor * Flow->velocity[{X, Y, Z, 1}];
						Flow->force[{X, Y, Z, 2}] += factor * Flow->velocity[{X, Y, Z, 2}];
					}
				}
			}
		}
	}
}
void Phase_Field::Reset_Velocity(int time, Flow_solver* Flow, Parallel_MPI* MPI_parallel) {  /// new phase_field added
	if (MPI_parallel->processor_id != MASTER) {
		int X, Y, Z;
		double Psi, factor;
		for (X = 2; X <= MPI_parallel->actual_rows_XYZ[0] + 1; ++X) {
			for (Y = 2; Y <= MPI_parallel->actual_rows_XYZ[1] + 1; ++Y) {
				for (Z = 2; Z <= MPI_parallel->end_XYZ2[2] + 1; ++Z) {
					if (solid_phase_type[{X, Y, Z}] == -1) {
						Psi = 0.5 * (1. + phase[{X, Y, Z}]);
						factor = 1. - Psi;
						Flow->velocity[{X, Y, Z, 0}] *= factor;
						Flow->velocity[{X, Y, Z, 1}] *= factor;
						Flow->velocity[{X, Y, Z, 2}] *= factor;
					}
				}
			}
		}
	}
}
void Phase_Field::LBMNONCONS_p(int time, Crystal_Anisotropy Ani_Funct, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		swap(pop_old_p, pop_p);
		int X, Y, Z, alpha;
		double NN[3], a_s, B, mag_n, mag_gamma;  /// new added

		for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (solid_phase_type[{X, Y, Z}] == -1) {
						///****************************************************************************
						///----------------------------------------------------------------------------
						///               START OF  2-order   directional derivatives
						///----------------------------------------------------------------------------
						///****************************************************************************

						///.............  Start of second order central finite difference  ..................
						//                        n_p[0] = 0.5*(phase[{X + 1,Y,Z}] - phase[{X - 1,Y,Z}]);
						//                        n_p[1] = 0.5*(phase[{X,Y + 1,Z}] - phase[{X,Y - 1,Z}]);
						//                        n_p[2] = 0.5*(phase[{X,Y,Z + 1}] - phase[{X,Y,Z - 1}]);

						///.............  End of second order central finite difference  .....................

						///.........................  START OF E8 in 2D  ................................
						//                        n_p[0] = 4.0/21 * (phase[{X + 1,Y,Z}]       -    phase[{X - 1,Y,Z}])
						//                               + 4.0/45 * (phase[{X + 1,Y + 1,Z}]   -    phase[{X - 1,Y + 1,Z}]
						//                                         + phase[{X + 1,Y - 1,Z}]   -    phase[{X - 1,Y - 1,Z}])
						//                               + 1.0/30 * (phase[{X + 2,Y,Z}]       -    phase[{X - 2,Y,Z}])
						//                               + 4.0/315 *(phase[{X + 2,Y + 1,Z}]   -    phase[{X - 2,Y + 1,Z}]
						//                                         + phase[{X + 2,Y - 1,Z}]   -    phase[{X - 2,Y - 1,Z}])
						//                               + 2.0/315 *(phase[{X + 1,Y + 2,Z}]   -    phase[{X - 1,Y + 2,Z}]
						//                                         + phase[{X + 1,Y - 2,Z}]   -    phase[{X - 1,Y - 2,Z}])
						//                               + 1.0/2520*(phase[{X + 2,Y + 2,Z}]   -    phase[{X - 2,Y + 2,Z}]
						//                                         + phase[{X + 2,Y - 2,Z}]   -    phase[{X - 2,Y - 2,Z}]);
						//
						//                        n_p[1] = 4.0/21 * (phase[{X,Y + 1,Z}]       -    phase[{X,Y - 1,Z}])
						//                               + 4.0/45 * (phase[{X + 1,Y + 1,Z}]   -    phase[{X + 1,Y - 1,Z}]
						//                                         + phase[{X - 1,Y + 1,Z}]   -    phase[{X - 1,Y - 1,Z}])
						//                               + 1.0/30 * (phase[{X,Y + 2,Z}]       -    phase[{X,Y - 2,Z}])
						//                               + 4.0/315 *(phase[{X + 1,Y + 2,Z}]   -    phase[{X + 1,Y - 2,Z}]
						//                                         + phase[{X - 1,Y + 2,Z}]   -    phase[{X - 1,Y - 2,Z}])
						//                               + 2.0/315 *(phase[{X + 2,Y + 1,Z}]   -    phase[{X + 2,Y - 1,Z}]
						//                                         + phase[{X - 2,Y + 1,Z}]   -    phase[{X - 2,Y - 1,Z}])
						//                               + 1.0/2520*(phase[{X + 2,Y + 2,Z}]   -    phase[{X + 2,Y - 2,Z}]
						//                                         + phase[{X - 2,Y + 2,Z}]   -    phase[{X - 2,Y - 2,Z}]);
						//                        n_p[2] = 0.0;
						///.........................  END OF E8 in 2D  ................................

						///.........................  START OF E10 in 2D  .............................

						//                 n_p[0] = 262.0/1785 * (phase[{X + 1,Y,Z}]     - phase[{X - 1,Y,Z}])
						//                        + 93.0/1190 *  (phase[{X + 1,Y + 1,Z}] - phase[{X - 1,Y + 1,Z}]
						//                                      + phase[{X + 1,Y - 1,Z}] - phase[{X - 1,Y - 1,Z}])
						//          				+ 7.0/170  *   (phase[{X + 2,Y,Z}]     - phase[{X - 2,Y,Z}])
						//                        + 6.0/595  *   (phase[{X + 1,Y + 2,Z}] - phase[{X - 1,Y + 2,Z}]
						//					                  + phase[{X + 1,Y - 2,Z}] - phase[{X - 1,Y - 2,Z}])
						//                        + 12.0/595  *  (phase[{X + 2,Y + 1,Z}] - phase[{X - 2,Y + 1,Z}]
						//					                  + phase[{X + 2,Y - 1,Z}] - phase[{X - 2,Y - 1,Z}])
						//                        + 9.0/4760  *  (phase[{X + 2,Y + 2,Z}] - phase[{X - 2,Y + 2,Z}]
						//					                  + phase[{X + 2,Y - 2,Z}] - phase[{X - 2,Y - 2,Z}])
						//                        + 2.0/1785  *  (phase[{(X + 3)%MPI_parallel->dev_end[0],Y,Z}]     - phase[(X - 3 + MPI_parallel->dev_end[0])%MPI_parallel->dev_end[0]][Y][Z])
						//					    + 1.0/7140  *  (phase[{X + 1,(Y + 3)%MPI_parallel->dev_end[1],Z}] - phase[{X - 1,(Y + 3)%MPI_parallel->dev_end[1],Z}]
						//					                  + phase[X + 1][(Y - 3 + MPI_parallel->dev_end[0])%MPI_parallel->dev_end[1]][Z] - phase[X - 1][(Y - 3 + MPI_parallel->dev_end[0])%MPI_parallel->dev_end[1]][Z])
						//                        + 1.0/2380  *  (phase[{(X + 3)%MPI_parallel->dev_end[0],Y + 1,Z}] - phase[(X - 3 + MPI_parallel->dev_end[0])%MPI_parallel->dev_end[0]][Y + 1][Z]
						//					                  + phase[{(X + 3)%MPI_parallel->dev_end[0],Y - 1,Z}] - phase[(X - 3 + MPI_parallel->dev_end[0])%MPI_parallel->dev_end[0]][Y - 1][Z]);
						//
						//
						//                  n_p[1] = 262.0/1785 *(phase[{X,Y + 1,Z}]     - phase[{X,Y - 1,Z}])
						//					     + 93.0/1190 * (phase[{X + 1,Y + 1,Z}] - phase[{X + 1,Y - 1,Z}]
						//					                  + phase[{X - 1,Y + 1,Z}] - phase[{X - 1,Y - 1,Z}])
						//                         + 7.0/170  *  (phase[{X,Y + 2,Z}]     - phase[{X,Y - 2,Z}])
						//					     + 6.0/595  *  (phase[{X + 2,Y + 1,Z}] - phase[{X + 2,Y - 1,Z}]
						//					                  + phase[{X - 2,Y + 1,Z}] - phase[{X - 2,Y - 1,Z}])
						//           				 + 12.0/595 *  (phase[{X + 1,Y + 2,Z}] - phase[{X + 1,Y - 2,Z}]
						//					                  + phase[{X - 1,Y + 2,Z}] - phase[{X - 1,Y - 2,Z}])
						//                         + 9.0/4760 *  (phase[{X + 2,Y + 2,Z}] - phase[{X + 2,Y - 2,Z}]
						//					                  + phase[{X - 2,Y + 2,Z}] - phase[{X - 2,Y - 2,Z}])
						//                         + 2.0/1785 *  (phase[{X,(Y + 3)%MPI_parallel->dev_end[1],Z}]     - phase[X][(Y - 3 + MPI_parallel->dev_end[0])%MPI_parallel->dev_end[1]][Z])
						//					     + 1.0/7140 *  (phase[{(X + 3)%MPI_parallel->dev_end[0],Y + 1,Z}] - phase[{(X + 3)%MPI_parallel->dev_end[0],Y - 1,Z}]
						//					                  + phase[(X - 3 + MPI_parallel->dev_end[0])%MPI_parallel->dev_end[0]][Y + 1][Z] - phase[(X - 3 + MPI_parallel->dev_end[0])%MPI_parallel->dev_end[0]][Y - 1][Z])
						//           			  	 + 1.0/2380 *  (phase[{X + 1,(Y + 3)%MPI_parallel->dev_end[1],Z}] - phase[X + 1][(Y - 3 + MPI_parallel->dev_end[0])%MPI_parallel->dev_end[1]][Z]
						//					                  + phase[{X - 1,(Y + 3)%MPI_parallel->dev_end[1],Z}] - phase[X - 1][(Y - 3 + MPI_parallel->dev_end[0])%MPI_parallel->dev_end[1]][Z]);
						//
						//                  n_p[2] = 0.0;
						///.........................  END OF E10 in 2D  ...............................

						///.........................  START OF E12 in 2D  .............................

						//                 n_p[0] = 68.0/585  *  (phase[{X + 1,Y,Z}]     - phase[{X - 1,Y,Z}])
						//                        + 68.0/1001 *  (phase[{X + 1,Y + 1,Z}] - phase[{X - 1,Y + 1,Z}]
						//                                      + phase[{X + 1,Y - 1,Z}] - phase[{X - 1,Y - 1,Z}])
						//          				+ 2.0/45   *   (phase[{X + 2,Y,Z}]     - phase[{X - 2,Y,Z}])
						//                        + 62.0/5005  * (phase[{X + 1,Y + 2,Z}] - phase[{X - 1,Y + 2,Z}]
						//					                  + phase[{X + 1,Y - 2,Z}] - phase[{X - 1,Y - 2,Z}])
						//                        + 124.0/5005 * (phase[{X + 2,Y + 1,Z}] - phase[{X - 2,Y + 1,Z}]
						//					                  + phase[{X + 2,Y - 1,Z}] - phase[{X - 2,Y - 1,Z}])
						//                        + 1.0/260   *  (phase[{X + 2,Y + 2,Z}] - phase[{X - 2,Y + 2,Z}]
						//					                  + phase[{X + 2,Y - 2,Z}] - phase[{X - 2,Y - 2,Z}])
						//                        + 4.0/1365  *  (phase[{(X + 3)%MPI_parallel->dev_end[0],Y,Z}]     - phase[(X - 3 + MPI_parallel->dev_end[0])%MPI_parallel->dev_end[0]][Y][Z])
						//					    + 2.0/4095  *  (phase[{X + 1,(Y + 3)%MPI_parallel->dev_end[1],Z}] - phase[{X - 1,(Y + 3)%MPI_parallel->dev_end[1],Z}]
						//					                  + phase[X + 1][(Y - 3 + MPI_parallel->dev_end[1])%MPI_parallel->dev_end[1]][Z] - phase[X - 1][(Y - 3 + MPI_parallel->dev_end[1])%MPI_parallel->dev_end[1]][Z])
						//                        + 2.0/1365  *  (phase[{(X + 3)%MPI_parallel->dev_end[0],Y + 1,Z}] - phase[(X - 3 + MPI_parallel->dev_end[0])%MPI_parallel->dev_end[0]][Y + 1][Z]
						//					                  + phase[{(X + 3)%MPI_parallel->dev_end[0],Y - 1,Z}] - phase[(X - 3 + MPI_parallel->dev_end[0])%MPI_parallel->dev_end[0]][Y - 1][Z])
						//                        + 4.0/45045 *  (phase[{X + 2,(Y + 3)%MPI_parallel->dev_end[1],Z}] - phase[{X - 2,(Y + 3)%MPI_parallel->dev_end[1],Z}]
						//                                      + phase[X + 2][(Y - 3 + MPI_parallel->dev_end[1])%MPI_parallel->dev_end[1]][Z] - phase[X - 2][(Y - 3 + MPI_parallel->dev_end[1])%MPI_parallel->dev_end[1]][Z])
						//                        + 2.0/15015  * (phase[{(X + 3)%MPI_parallel->dev_end[0],Y + 2,Z}] - phase[(X - 3 + MPI_parallel->dev_end[0])%MPI_parallel->dev_end[0]][Y + 2][Z]
						//                                      + phase[{(X + 3)%MPI_parallel->dev_end[0],Y - 2,Z}] - phase[(X - 3 + MPI_parallel->dev_end[0])%MPI_parallel->dev_end[0]][Y - 2][Z])
						//                        + 1.0/120120 * (phase[{(X + 4)%MPI_parallel->dev_end[0],Y,Z}]     - phase[(X - 4 + MPI_parallel->dev_end[0])%MPI_parallel->dev_end[0]][Y][Z]);
						//
						//                n_p[1] = 68.0/585  *(phase[{X,Y + 1,Z}]         - phase[{X,Y - 1,Z}])
						//					     + 68.0/1001 *(phase[{X + 1,Y + 1,Z}]   - phase[{X + 1,Y - 1,Z}]
						//                                     + phase[{X - 1,Y + 1,Z}]   - phase[{X - 1,Y - 1,Z}])
						//                         + 2.0/45  *  (phase[{X,Y + 2,Z}]       - phase[{X,Y - 2,Z}])
						//					     + 62.0/5005 *(phase[{X + 2,Y + 1,Z}]   - phase[{X + 2,Y - 1,Z}]
						//                                     + phase[{X - 2,Y + 1,Z}]   - phase[{X - 2,Y - 1,Z}])
						//           				 + 124.0/5005*(phase[{X + 1,Y + 2,Z}]   - phase[{X + 1,Y - 2,Z}]
						//                                     + phase[{X - 1,Y + 2,Z}]   - phase[{X - 1,Y - 2,Z}])
						//                         +  1.0/260 * (phase[{X + 2,Y + 2,Z}]   - phase[{X + 2,Y - 2,Z}]
						//                                     + phase[{X - 2,Y + 2,Z}]   - phase[{X - 2,Y - 2,Z}])
						//                         + 4.0/1365 * (phase[{X,(Y + 3)%MPI_parallel->dev_end[1],Z}]     - phase[X][(Y - 3 + MPI_parallel->dev_end[1])%MPI_parallel->dev_end[1]][Z])
						//					     + 2.0/4095 * (phase[{(X + 3)%MPI_parallel->dev_end[0],Y + 1,Z}] - phase[{(X + 3)%MPI_parallel->dev_end[0],Y - 1,Z}]
						//					                  + phase[(X - 3 + MPI_parallel->dev_end[0])%MPI_parallel->dev_end[0]][Y + 1][Z] - phase[(X - 3 + MPI_parallel->dev_end[0])%MPI_parallel->dev_end[0]][Y - 1][Z])
						//           			  	 + 2.0/1365 *  (phase[{X + 1,(Y + 3)%MPI_parallel->dev_end[1],Z}] - phase[X + 1][(Y - 3 + MPI_parallel->dev_end[1])%MPI_parallel->dev_end[1]][Z]
						//					                  + phase[{X - 1,(Y + 3)%MPI_parallel->dev_end[1],Z}] - phase[X - 1][(Y - 3 + MPI_parallel->dev_end[1])%MPI_parallel->dev_end[1]][Z])
						//                         + 4.0/45045 * (phase[{(X + 3)%MPI_parallel->dev_end[0],Y + 2,Z}] - phase[{(X + 3)%MPI_parallel->dev_end[0],Y - 2,Z}]
						//                                      + phase[(X - 3 + MPI_parallel->dev_end[0])%MPI_parallel->dev_end[0]][Y + 2][Z] - phase[(X - 3 + MPI_parallel->dev_end[0])%MPI_parallel->dev_end[0]][Y - 2][Z])
						//                         + 2.0/15015  *(phase[{X + 2,(Y + 3)%MPI_parallel->dev_end[1],Z}] - phase[X + 2][(Y - 3 + MPI_parallel->dev_end[1])%MPI_parallel->dev_end[1]][Z]
						//                                      + phase[{X - 2,(Y + 3)%MPI_parallel->dev_end[1],Z}] - phase[X - 2][(Y - 3 + MPI_parallel->dev_end[1])%MPI_parallel->dev_end[1]][Z])
						//                         + 1.0/120120 *(phase[{X,(Y + 4)%MPI_parallel->dev_end[1],Z}]     - phase[X][(Y - 4 + MPI_parallel->dev_end[1])%MPI_parallel->dev_end[1]][Z]);
						//
						//                  n_p[2] = 0.0;
						///.........................  END OF E12 in 2D  ...............................
						///****************************************************************************
						///----------------------------------------------------------------------------
						///               END OF  2D   directional derivatives
						///----------------------------------------------------------------------------
						///****************************************************************************

						///*********************************************************************************
						///---------------------------------------------------------------------------------
						///                    3D   directional derivatives
						///---------------------------------------------------------------------------------
						///*********************************************************************************

						///.........................  START OF E4 in 3D  .............................
						//                        n_p[0] = 1.0/6 * (phase[{X + 1,Y,Z}] - phase[{X - 1,Y,Z}]) + 1.0/12 * (phase[{X + 1,Y + 1,Z}] - phase[{X - 1,Y + 1,Z}] + phase[{X + 1,Y - 1,Z}] - phase[{X - 1,Y - 1,Z}] + phase[{X + 1,Y,Z + 1}] - phase[{X - 1,Y,Z + 1}] + phase[{X + 1,Y,Z - 1}] - phase[{X - 1,Y,Z - 1}]);
						//                        n_p[1] = 1.0/6 * (phase[{X,Y + 1,Z}] - phase[{X,Y - 1,Z}]) + 1.0/12 * (phase[{X + 1,Y + 1,Z}] - phase[{X + 1,Y - 1,Z}] + phase[{X - 1,Y + 1,Z}] - phase[{X - 1,Y - 1,Z}] + phase[{X,Y + 1,Z + 1}] - phase[{X,Y - 1,Z + 1}] + phase[{X,Y + 1,Z - 1}] - phase[{X,Y - 1,Z - 1}]);
						//                        n_p[2] = 1.0/6 * (phase[{X,Y,Z + 1}] - phase[{X,Y,Z - 1}]) + 1.0/12 * (phase[{X + 1,Y,Z + 1}] - phase[{X + 1,Y,Z - 1}] + phase[{X - 1,Y,Z + 1}] - phase[{X - 1,Y,Z - 1}] + phase[{X,Y + 1,Z + 1}] - phase[{X,Y + 1,Z - 1}] + phase[{X,Y - 1,Z + 1}] - phase[{X,Y - 1,Z - 1}]);
						///.........................  END OF E4 in 3D  ................................

						///.........................  START OF E6 in 3D  .............................
						////
						//                        n_p[0] = 2.0/15 * (phase[{X + 1,Y,Z}] - phase[{X - 1,Y,Z}]) + 1.0/15 * (phase[{X + 1,Y + 1,Z}] - phase[{X - 1,Y + 1,Z}] + phase[{X + 1,Y - 1,Z}] - phase[{X - 1,Y - 1,Z}] + phase[{X + 1,Y,Z + 1}] - phase[{X - 1,Y,Z + 1}] + phase[{X + 1,Y,Z - 1}] - phase[{X - 1,Y,Z - 1}])
						//                                 + 1.0/60 * (phase[{X + 1,Y + 1,Z + 1}] - phase[{X - 1,Y + 1,Z + 1}] + phase[{X + 1,Y - 1,Z + 1}] - phase[{X - 1,Y - 1,Z + 1}] + phase[{X + 1,Y + 1,Z - 1}] - phase[{X - 1,Y + 1,Z - 1}] + phase[{X + 1,Y - 1,Z - 1}] - phase[{X - 1,Y - 1,Z - 1}] + phase[{X + 2,Y,Z}] - phase[{X - 2,Y,Z}]);
						//                        n_p[1] = 2.0/15 * (phase[{X,Y + 1,Z}] - phase[{X,Y - 1,Z}]) + 1.0/15 * (phase[{X + 1,Y + 1,Z}] - phase[{X + 1,Y - 1,Z}] + phase[{X - 1,Y + 1,Z}] - phase[{X - 1,Y - 1,Z}] + phase[{X,Y + 1,Z + 1}] - phase[{X,Y - 1,Z + 1}] + phase[{X,Y + 1,Z - 1}] - phase[{X,Y - 1,Z - 1}])
						//                                 + 1.0/60 * (phase[{X + 1,Y + 1,Z + 1}] - phase[{X + 1,Y - 1,Z + 1}] + phase[{X - 1,Y + 1,Z + 1}] - phase[{X - 1,Y - 1,Z + 1}] + phase[{X + 1,Y + 1,Z - 1}] - phase[{X + 1,Y - 1,Z - 1}] + phase[{X - 1,Y + 1,Z - 1}] - phase[{X - 1,Y - 1,Z - 1}] + phase[{X,Y + 2,Z}] - phase[{X,Y - 2,Z}]);
						//                        n_p[2] = 2.0/15 * (phase[{X,Y,Z + 1}] - phase[{X,Y,Z - 1}]) + 1.0/15 * (phase[{X + 1,Y,Z + 1}] - phase[{X + 1,Y,Z - 1}] + phase[{X - 1,Y,Z + 1}] - phase[{X - 1,Y,Z - 1}] + phase[{X,Y + 1,Z + 1}] - phase[{X,Y + 1,Z - 1}] + phase[{X,Y - 1,Z + 1}] - phase[{X,Y - 1,Z - 1}])
						//                                 + 1.0/60 * (phase[{X + 1,Y + 1,Z + 1}] - phase[{X + 1,Y + 1,Z - 1}] + phase[{X - 1,Y + 1,Z + 1}] - phase[{X - 1,Y + 1,Z - 1}] + phase[{X + 1,Y - 1,Z + 1}] - phase[{X + 1,Y - 1,Z - 1}] + phase[{X - 1,Y - 1,Z + 1}] - phase[{X - 1,Y - 1,Z - 1}] + phase[{X,Y,Z + 2}] - phase[{X,Y,Z - 2}]);

						///.........................  END OF E6 in 3D  ................................

						///.........................  START OF E8 in 3D  .............................

						//                        n_p[0] = 4.0/45 * (phase[{X + 1,Y,Z}] - phase[{X - 1,Y,Z}]) + 1.0/21 * (phase[{X + 1,Y + 1,Z}] - phase[{X - 1,Y + 1,Z}] + phase[{X + 1,Y - 1,Z}] - phase[{X - 1,Y - 1,Z}] + phase[{X + 1,Y,Z + 1}] - phase[{X - 1,Y,Z + 1}] + phase[{X + 1,Y,Z - 1}] - phase[{X - 1,Y,Z - 1}])
						//                               + 2.0/105 * (phase[{X + 1,Y + 1,Z + 1}] - phase[{X - 1,Y + 1,Z + 1}] + phase[{X + 1,Y - 1,Z + 1}] - phase[{X - 1,Y - 1,Z + 1}] + phase[{X + 1,Y + 1,Z - 1}] - phase[{X - 1,Y + 1,Z - 1}] + phase[{X + 1,Y - 1,Z - 1}] - phase[{X - 1,Y - 1,Z - 1}])
						//                               + 5.0/252 * (phase[{X + 2,Y,Z}] - phase[{X - 2,Y,Z}]) + 1.0/315 * (phase[{X + 1,Y,Z + 2}] - phase[{X - 1,Y,Z + 2}] + phase[{X + 1,Y,Z - 2}] - phase[{X - 1,Y,Z - 2}] + phase[{X + 1,Y + 2,Z}] - phase[{X - 1,Y + 2,Z}] + phase[{X + 1,Y - 2,Z}] - phase[{X - 1,Y - 2,Z}])
						//                               + 2.0/315 * (phase[{X + 2,Y,Z + 1}] - phase[{X - 2,Y,Z + 1}] + phase[{X + 2,Y,Z - 1}] - phase[{X - 2,Y,Z - 1}] + phase[{X + 2,Y + 1,Z}] - phase[{X - 2,Y + 1,Z}] + phase[{X + 2,Y - 1,Z}] - phase[{X - 2,Y - 1,Z}])
						//                               + 1.0/630 * (phase[{X + 1,Y + 1,Z + 2}] - phase[{X - 1,Y + 1,Z + 2}] + phase[{X + 1,Y + 1,Z - 2}] - phase[{X - 1,Y + 1,Z - 2}] + phase[{X + 1,Y + 2,Z + 1}] - phase[{X - 1,Y + 2,Z + 1}] + phase[{X + 1,Y + 2,Z - 1}] - phase[{X - 1,Y + 2,Z - 1}]
						//                                             + phase[{X + 1,Y - 1,Z + 2}] - phase[{X - 1,Y - 1,Z + 2}] + phase[{X + 1,Y - 1,Z - 2}] - phase[{X - 1,Y - 1,Z - 2}] + phase[{X + 1,Y - 2,Z + 1}] - phase[{X - 1,Y - 2,Z + 1}] + phase[{X + 1,Y - 2,Z - 1}] - phase[{X - 1,Y - 2,Z - 1}])
						//                               + 1.0/315 * (phase[{X + 2,Y + 1,Z + 1}] - phase[{X - 2,Y + 1,Z + 1}] + phase[{X + 2,Y + 1,Z - 1}] - phase[{X - 2,Y + 1,Z - 1}] + phase[{X + 2,Y - 1,Z + 1}] - phase[{X - 2,Y - 1,Z + 1}] + phase[{X + 2,Y - 1,Z - 1}] - phase[{X - 2,Y - 1,Z - 1}])
						//                               + 1.0/2520 * (phase[{X + 2,Y,Z + 2}] - phase[{X - 2,Y,Z + 2}] + phase[{X + 2,Y,Z - 2}] - phase[{X -2,Y,Z - 2}] + phase[{X + 2,Y + 2,Z}] - phase[{X - 2,Y + 2,Z}] + phase[{X + 2,Y - 2,Z}] - phase[{X - 2,Y - 2,Z}]);
						//
						//                        n_p[1] = 4.0/45 * (phase[{X,Y + 1,Z}] - phase[{X,Y - 1,Z}]) + 1.0/21 * (phase[{X + 1,Y + 1,Z}] - phase[{X + 1,Y - 1,Z}] + phase[{X - 1,Y + 1,Z}] - phase[{X - 1,Y - 1,Z}] + phase[{X,Y + 1,Z + 1}] - phase[{X,Y - 1,Z + 1}] + phase[{X,Y + 1,Z - 1}] - phase[{X,Y - 1,Z - 1}])
						//                               + 2.0/105 * (phase[{X + 1,Y + 1,Z + 1}] - phase[{X + 1,Y - 1,Z + 1}] + phase[{X - 1,Y + 1,Z + 1}] - phase[{X - 1,Y - 1,Z + 1}] + phase[{X + 1,Y + 1,Z - 1}] - phase[{X + 1,Y - 1,Z - 1}] + phase[{X - 1,Y + 1,Z - 1}] - phase[{X - 1,Y - 1,Z - 1}])
						//                               + 5.0/252 * (phase[{X,Y + 2,Z}] - phase[{X,Y - 2,Z}]) + 1.0/315 * (phase[{X,Y + 1,Z + 2}] - phase[{X,Y - 1,Z + 2}] + phase[{X,Y + 1,Z - 2}] - phase[{X,Y - 1,Z - 2}] + phase[{X - 2,Y + 1,Z}] - phase[{X - 2,Y - 1,Z}] + phase[{X + 2,Y + 1,Z}] - phase[{X + 2,Y - 1,Z}])
						//                               + 2.0/315 * (phase[{X,Y + 2,Z + 1}] - phase[{X,Y - 2,Z + 1}] + phase[{X,Y + 2,Z - 1}] - phase[{X,Y - 2,Z - 1}] + phase[{X - 1,Y + 2,Z}] - phase[{X - 1,Y - 2,Z}] + phase[{X + 1,Y + 2,Z}] - phase[{X + 1,Y - 2,Z}])
						//                               + 1.0/630 * (phase[{X - 1,Y + 1,Z + 2}] - phase[{X - 1,Y - 1,Z + 2}] + phase[{X + 1,Y + 1,Z + 2}] - phase[{X + 1,Y - 1,Z + 2}] + phase[{X - 2,Y + 1,Z + 1}] - phase[{X - 2,Y - 1,Z + 1}] + phase[{X + 2,Y + 1,Z + 1}] - phase[{X + 2,Y - 1,Z + 1}]
						//                                             + phase[{X - 2,Y + 1,Z - 1}] - phase[{X - 2,Y - 1,Z - 1}] + phase[{X + 2,Y + 1,Z - 1}] - phase[{X + 2,Y - 1,Z - 1}] + phase[{X - 1,Y + 1,Z - 2}] - phase[{X - 1,Y - 1,Z - 2}] + phase[{X + 1,Y + 1,Z - 2}] - phase[{X + 1,Y - 1,Z - 2}])
						//                               + 1.0/315 * (phase[{X - 1,Y + 2,Z + 1}] - phase[{X - 1,Y - 2,Z + 1}] + phase[{X + 1,Y + 2,Z + 1}] - phase[{X + 1,Y - 2,Z + 1}] + phase[{X - 1,Y + 2,Z - 1}] - phase[{X - 1,Y - 2,Z - 1}] + phase[{X + 1,Y + 2,Z - 1}] - phase[{X + 1,Y - 2,Z - 1}])
						//                               + 1.0/2520 * (phase[{X,Y + 2,Z + 2}] - phase[{X,Y - 2,Z + 2}] + phase[{X,Y + 2,Z - 2}] - phase[{X,Y - 2,Z - 2}] + phase[{X - 2,Y + 2,Z}] - phase[{X - 2,Y - 2,Z}] + phase[{X + 2,Y + 2,Z}] - phase[{X + 2,Y - 2,Z}]);
						//
						//
						//                        n_p[2] = 4.0/45 * (phase[{X,Y,Z + 1}] - phase[{X,Y,Z - 1}]) + 1.0/21 * (phase[{X + 1,Y,Z + 1}] - phase[{X + 1,Y,Z - 1}] + phase[{X - 1,Y,Z + 1}] - phase[{X - 1,Y,Z - 1}] + phase[{X,Y + 1,Z + 1}] - phase[{X,Y + 1,Z - 1}] + phase[{X,Y - 1,Z + 1}] - phase[{X,Y - 1,Z - 1}])
						//                               + 2.0/105 * (phase[{X + 1,Y + 1,Z + 1}] - phase[{X + 1,Y + 1,Z - 1}] + phase[{X - 1,Y + 1,Z + 1}] - phase[{X - 1,Y + 1,Z - 1}] + phase[{X + 1,Y - 1,Z + 1}] - phase[{X + 1,Y - 1,Z - 1}] + phase[{X - 1,Y - 1,Z + 1}] - phase[{X - 1,Y - 1,Z - 1}])
						//                               + 5.0/252 * (phase[{X,Y,Z + 2}] - phase[{X,Y,Z - 2}]) + 1.0/315 * (phase[{X - 2,Y,Z + 1}] - phase[{X - 2,Y,Z - 1}] + phase[{X + 2,Y,Z + 1}] - phase[{X + 2,Y,Z - 1}] + phase[{X,Y + 2,Z + 1}] - phase[{X,Y + 2,Z - 1}] + phase[{X,Y - 2,Z + 1}] - phase[{X,Y - 2,Z - 1}])
						//                               + 2.0/315 * (phase[{X - 1,Y,Z + 2}] - phase[{X - 1,Y,Z - 2}] + phase[{X + 1,Y,Z + 2}] - phase[{X + 1,Y,Z - 2}] + phase[{X,Y + 1,Z + 2}] - phase[{X,Y + 1,Z - 2}] + phase[{X,Y - 1,Z + 2}] - phase[{X,Y - 1,Z - 2}])
						//                               + 1.0/630 * (phase[{X - 2,Y + 1,Z + 1}] - phase[{X - 2,Y + 1,Z - 1}] + phase[{X + 2,Y + 1,Z + 1}] - phase[{X + 2,Y + 1,Z - 1}] + phase[{X - 1,Y + 2,Z + 1}] - phase[{X - 1,Y + 2,Z - 1}] + phase[{X + 1,Y + 2,Z + 1}] - phase[{X + 1,Y + 2,Z - 1}]
						//                                            + phase[{X - 2,Y - 1,Z + 1}] - phase[{X - 2,Y - 1,Z - 1}] + phase[{X + 2,Y - 1,Z + 1}] - phase[{X + 2,Y - 1,Z - 1}] + phase[{X - 1,Y - 2,Z + 1}] - phase[{X - 1,Y - 2,Z - 1}] + phase[{X + 1,Y - 2,Z + 1}] - phase[{X + 1,Y - 2,Z - 1}])
						//                               + 1.0/315 * (phase[{X - 1,Y + 1,Z + 2}] - phase[{X - 1,Y + 1,Z - 2}] + phase[{X + 1,Y + 1,Z + 2}] - phase[{X + 1,Y + 1,Z - 2}] + phase[{X - 1,Y - 1,Z + 2}] - phase[{X - 1,Y - 1,Z - 2}] + phase[{X + 1,Y - 1,Z + 2}] - phase[{X + 1,Y - 1,Z - 2}])
						//                               + 1.0/2520 * (phase[{X - 2,Y,Z + 2}] - phase[{X - 2,Y,Z - 2}] + phase[{X + 2,Y,Z + 2}] - phase[{X + 2,Y,Z - 2}] + phase[{X,Y - 2,Z + 2}] - phase[{X,Y - 2,Z - 2}] + phase[{X,Y + 2,Z + 2}] - phase[{X,Y + 2,Z - 2}]);

						///.........................  END OF E8 in 3D  ................................

						///.........................  START OF E10 in 3D  .............................
						//                        n_p[0] = 352.0/5355 *(phase[{X + 1,Y,Z}] - phase[{X - 1,Y,Z}]) + 38.0/1071 * (phase[{X + 1,Y + 1,Z}] - phase[{X - 1,Y + 1,Z}] + phase[{X + 1,Y - 1,Z}] - phase[{X - 1,Y - 1,Z}] + phase[{X + 1,Y,Z + 1}] - phase[{X - 1,Y,Z + 1}] + phase[{X + 1,Y,Z - 1}] - phase[{X - 1,Y,Z - 1}])
						//                               + 271.0/14280 * (phase[{X + 1,Y + 1,Z + 1}] - phase[{X - 1,Y + 1,Z + 1}] + phase[{X + 1,Y - 1,Z + 1}] - phase[{X - 1,Y - 1,Z + 1}] + phase[{X + 1,Y + 1,Z - 1}] - phase[{X - 1,Y + 1,Z - 1}] + phase[{X + 1,Y - 1,Z - 1}] - phase[{X - 1,Y - 1,Z - 1}])
						//                               + 139.0/7140 * (phase[{X + 2,Y,Z}] - phase[{X - 2,Y,Z}]) + 53.0/10710 * (phase[{X + 1,Y,Z + 2}] - phase[{X - 1,Y,Z + 2}] + phase[{X + 1,Y,Z - 2}] - phase[{X - 1,Y,Z - 2}] + phase[{X + 1,Y + 2,Z}] - phase[{X - 1,Y + 2,Z}] + phase[{X + 1,Y - 2,Z}] - phase[{X - 1,Y - 2,Z}])
						//                               + 53.0/5355 * (phase[{X + 2,Y,Z + 1}] - phase[{X - 2,Y,Z + 1}] + phase[{X + 2,Y,Z - 1}] - phase[{X - 2,Y,Z - 1}] + phase[{X + 2,Y + 1,Z}] - phase[{X - 2,Y + 1,Z}] + phase[{X + 2,Y - 1,Z}] - phase[{X - 2,Y - 1,Z}])
						//                               + 5.0/2142 * (phase[{X + 1,Y + 1,Z + 2}] - phase[{X - 1,Y + 1,Z + 2}] + phase[{X + 1,Y + 1,Z - 2}] - phase[{X - 1,Y + 1,Z - 2}] + phase[{X + 1,Y + 2,Z + 1}] - phase[{X - 1,Y + 2,Z + 1}] + phase[{X + 1,Y + 2,Z - 1}] - phase[{X - 1,Y + 2,Z - 1}]
						//                                             + phase[{X + 1,Y - 1,Z + 2}] - phase[{X - 1,Y - 1,Z + 2}] + phase[{X + 1,Y - 1,Z - 2}] - phase[{X - 1,Y - 1,Z - 2}] + phase[{X + 1,Y - 2,Z + 1}] - phase[{X - 1,Y - 2,Z + 1}] + phase[{X + 1,Y - 2,Z - 1}] - phase[{X - 1,Y - 2,Z - 1}])
						//                               + 5.0/1071 * (phase[{X + 2,Y + 1,Z + 1}] - phase[{X -2,Y + 1,Z + 1}] + phase[{X + 2,Y + 1,Z - 1}] - phase[{X - 2,Y + 1,Z - 1}] + phase[{X + 2,Y - 1,Z + 1}] - phase[{X -2,Y - 1,Z + 1}] + phase[{X + 2,Y - 1,Z - 1}] - phase[{X - 2,Y - 1,Z - 1}])
						//                               + 41.0/42840 * (phase[{X + 2,Y,Z + 2}] - phase[{X - 2,Y,Z + 2}] + phase[{X + 2,Y,Z - 2}] - phase[{X -2,Y,Z - 2}] + phase[{X + 2,Y + 2,Z}] - phase[{X - 2,Y + 2,Z}] + phase[{X + 2,Y - 2,Z}] - phase[{X - 2,Y - 2,Z}])
						//                               + 1.0/4284 * (phase[{X + 1,Y + 2,Z + 2}] - phase[{X - 1,Y + 2,Z + 2}] + phase[{X + 1,Y + 2,Z - 2}] - phase[{X - 1,Y + 2,Z - 2}] + phase[{X + 1,Y - 2,Z + 2}] - phase[{X - 1,Y - 2,Z + 2}] + phase[{X + 1,Y - 2,Z - 2}] - phase[{X - 1,Y - 2,Z - 2}])
						//                               + 1.0/2142 * (phase[{X + 2,Y + 1,Z + 2}] - phase[{X - 2,Y + 1,Z + 2}] + phase[{X + 2,Y + 1,Z - 2}] - phase[{X - 2,Y + 1,Z - 2}] + phase[{X + 2,Y - 1,Z + 2}] - phase[{X - 2,Y - 1,Z + 2}] + phase[{X + 2,Y - 1,Z - 2}] - phase[{X - 2,Y - 1,Z - 2}]
						//                                             + phase[{X + 2,Y + 2,Z + 1}] - phase[{X - 2,Y + 2,Z + 1}] + phase[{X + 2,Y + 2,Z - 1}] - phase[{X - 2,Y + 2,Z - 1}] + phase[{X + 2,Y - 2,Z + 1}] - phase[{X - 2,Y - 2,Z + 1}] + phase[{X + 2,Y - 2,Z - 1}] - phase[{X - 2,Y - 2,Z - 1}])
						//                               + 1.0/1785 * (phase[{X + 3,Y,Z}] - phase[{X - 3,Y,Z}]) + 1.0/10710 * (phase[{X + 1,Y + 3,Z}] - phase[{X - 1,Y + 3,Z}] + phase[{X + 1,Y - 3,Z}] - phase[{X - 1,Y - 3,Z}] + phase[{X + 1,Y,Z + 3}] - phase[{X - 1,Y,Z + 3}] + phase[{X + 1,Y,Z - 3}] - phase[{X - 1,Y,Z - 3}])
						//                               + 1.0/3570 * (phase[{X + 3,Y + 1,Z}] - phase[{X - 3,Y + 1,Z}] + phase[{X + 3,Y - 1,Z}] - phase[{X - 3,Y - 1,Z}] + phase[{X + 3,Y,Z + 1}] - phase[{X - 3,Y,Z + 1}] + phase[{X + 3,Y,Z - 1}] - phase[{X - 3,Y,Z - 1}])
						//                               + 1.0/42840 * (phase[{X + 1,Y + 1,Z + 3}] - phase[{X - 1,Y + 1,Z + 3}] + phase[{X + 1,Y + 1,Z - 3}] - phase[{X - 1,Y + 1,Z - 3}]+ phase[{X + 1,Y - 1,Z + 3}] - phase[{X - 1,Y - 1,Z + 3}] + phase[{X + 1,Y - 1,Z - 3}] - phase[{X - 1,Y - 1,Z - 3}]
						//                                              + phase[{X + 1,Y + 3,Z + 1}] - phase[{X - 1,Y + 3,Z + 1}] + phase[{X + 1,Y + 3,Z - 1}] - phase[{X - 1,Y + 3,Z - 1}] + phase[{X + 1,Y - 3,Z + 1}] - phase[{X - 1,Y - 3,Z + 1}] + phase[{X + 1,Y - 3,Z - 1}] - phase[{X - 1,Y - 3,Z - 1}])
						//                               + 1.0/14280 * (phase[{X + 3,Y + 1,Z + 1}] - phase[{X - 3,Y + 1,Z + 1}] + phase[{X + 3,Y + 1,Z - 1}] - phase[{X - 3,Y + 1,Z - 1}] + phase[{X + 3,Y - 1,Z + 1}] - phase[{X - 3,Y - 1,Z + 1}] + phase[{X + 3,Y - 1,Z - 1}] - phase[{X - 3,Y - 1,Z - 1}]);
						//
						//                        n_p[1] = 352/5355 *(phase[{X,Y + 1,Z}] - phase[{X,Y - 1,Z}]) + 38.0/1071 * (phase[{X + 1,Y + 1,Z}] - phase[{X + 1,Y - 1,Z}] + phase[{X - 1,Y + 1,Z}] - phase[{X - 1,Y - 1,Z}] + phase[{X,Y + 1,Z + 1}] - phase[{X,Y - 1,Z + 1}] + phase[{X,Y + 1,Z - 1}] - phase[{X,Y - 1,Z - 1}])
						//                               + 271.0/14280 * (phase[{X + 1,Y + 1,Z + 1}] - phase[{X + 1,Y - 1,Z + 1}] + phase[{X - 1,Y + 1,Z + 1}] - phase[{X - 1,Y - 1,Z + 1}] + phase[{X + 1,Y + 1,Z - 1}] - phase[{X + 1,Y - 1,Z - 1}] + phase[{X - 1,Y + 1,Z - 1}] - phase[{X - 1,Y - 1,Z - 1}])
						//                               + 139.0/7140 * (phase[{X,Y + 2,Z}] - phase[{X,Y - 2,Z}]) + 53.0/10710 * (phase[{X,Y + 1,Z + 2}] - phase[{X,Y - 1,Z + 2}] + phase[{X,Y + 1,Z - 2}] - phase[{X,Y - 1,Z - 2}] + phase[{X - 2,Y + 1,Z}] - phase[{X - 2,Y - 1,Z}] + phase[{X + 2,Y + 1,Z}] - phase[{X + 2,Y - 1,Z}])
						//                               + 53.0/5355  * (phase[{X,Y + 2,Z + 1}] - phase[{X,Y - 2,Z + 1}] + phase[{X,Y + 2,Z - 1}] - phase[{X,Y - 2,Z - 1}] + phase[{X - 1,Y + 2,Z}] - phase[{X - 1,Y - 2,Z}] + phase[{X + 1,Y + 2,Z}] - phase[{X + 1,Y - 2,Z}])
						//                               + 5.0/2142 * (phase[{X - 1,Y + 1,Z + 2}] - phase[{X - 1,Y - 1,Z + 2}] + phase[{X + 1,Y + 1,Z + 2}] - phase[{X + 1,Y - 1,Z + 2}] + phase[{X - 2,Y + 1,Z + 1}] - phase[{X - 2,Y - 1,Z + 1}] + phase[{X + 2,Y + 1,Z + 1}] - phase[{X + 2,Y - 1,Z + 1}]
						//                                             + phase[{X - 2,Y + 1,Z - 1}] - phase[{X - 2,Y - 1,Z - 1}] + phase[{X + 2,Y + 1,Z - 1}] - phase[{X + 2,Y - 1,Z - 1}] + phase[{X - 1,Y + 1,Z - 2}] - phase[{X - 1,Y - 1,Z - 2}] + phase[{X + 1,Y + 1,Z - 2}] - phase[{X + 1,Y - 1,Z - 2}])
						//                               + 5.0/1071 * (phase[{X - 1,Y + 2,Z + 1}] - phase[{X - 1,Y - 2,Z + 1}] + phase[{X + 1,Y + 2,Z + 1}] - phase[{X + 1,Y - 2,Z + 1}] + phase[{X - 1,Y + 2,Z - 1}] - phase[{X - 1,Y - 2,Z - 1}] + phase[{X + 1,Y + 2,Z - 1}] - phase[{X + 1,Y - 2,Z - 1}])
						//                               + 41.0/42840 * (phase[{X,Y + 2,Z + 2}] - phase[{X,Y - 2,Z + 2}] + phase[{X,Y + 2,Z - 2}] - phase[{X,Y - 2,Z - 2}] + phase[{X - 2,Y + 2,Z}] - phase[{X - 2,Y - 2,Z}] + phase[{X + 2,Y + 2,Z}] - phase[{X + 2,Y - 2,Z}])
						//                               + 1.0/4284 * (phase[{X + 2,Y + 1,Z + 2}] - phase[{X + 2,Y - 1,Z + 2}] + phase[{X + 2,Y + 1,Z - 2}] - phase[{X + 2,Y - 1,Z - 2}] + phase[{X - 2,Y + 1,Z + 2}] - phase[{X - 2,Y - 1,Z + 2}] + phase[{X - 2,Y + 1,Z - 2}] - phase[{X - 2,Y - 1,Z - 2}])
						//                               + 1.0/2142 * (phase[{X + 1,Y + 2,Z + 2}] - phase[{X + 1,Y - 2,Z + 2}] + phase[{X + 1,Y + 2,Z - 2}] - phase[{X + 1,Y - 2,Z - 2}] + phase[{X - 1,Y + 2,Z + 2}] - phase[{X - 1,Y - 2,Z + 2}] + phase[{X - 1,Y + 2,Z - 2}] - phase[{X - 1,Y - 2,Z - 2}]
						//                                             + phase[{X + 2,Y + 2,Z + 1}] - phase[{X + 2,Y - 2,Z + 1}] + phase[{X + 2,Y + 2,Z - 1}] - phase[{X + 2,Y - 2,Z - 1}] + phase[{X - 2,Y + 2,Z + 1}] - phase[{X - 2,Y - 2,Z + 1}] + phase[{X - 2,Y + 2,Z - 1}] - phase[{X - 2,Y - 2,Z - 1}])
						//                               + 1.0/1785 * (phase[{X,Y + 3,Z}] - phase[{X,Y - 3,Z}]) + 1.0/10710 * (phase[{X + 3,Y + 1,Z}] - phase[{X + 3,Y - 1,Z}] + phase[{X - 3,Y + 1,Z}] - phase[{X - 3,Y - 1,Z}] + phase[{X,Y + 1,Z + 3}] - phase[{X,Y - 1,Z + 3}] + phase[{X,Y + 1,Z - 3}] - phase[{X,Y - 1,Z - 3}])
						//                               + 1.0/3570 * (phase[{X,Y + 3,Z + 1}] - phase[{X,Y - 3,Z + 1}] + phase[{X,Y + 3,Z - 1}] - phase[{X,Y - 3,Z - 1}] + phase[{X + 1,Y + 3,Z}] - phase[{X + 1,Y - 3,Z}] + phase[{X - 1,Y + 3,Z}] - phase[{X - 1,Y - 3,Z}])
						//                               + 1.0/42840 * (phase[{X + 1,Y + 1,Z + 3}] - phase[{X + 1,Y - 1,Z + 3}] + phase[{X + 1,Y + 1,Z - 3}] - phase[{X + 1,Y - 1,Z - 3}] + phase[{X - 1,Y + 1,Z + 3}] - phase[{X - 1,Y - 1,Z + 3}] + phase[{X - 1,Y + 1,Z - 3}] - phase[{X - 1,Y - 1,Z - 3}]
						//                                              + phase[{X + 3,Y + 1,Z + 1}] - phase[{X + 3,Y - 1,Z + 1}] + phase[{X + 3,Y + 1,Z - 1}] - phase[{X + 3,Y - 1,Z - 1}] + phase[{X - 3,Y + 1,Z + 1}] - phase[{X - 3,Y - 1,Z + 1}] + phase[{X - 3,Y + 1,Z - 1}] - phase[{X - 3,Y - 1,Z - 1}])
						//                               + 1.0/14280 * (phase[{X + 1,Y + 3,Z + 1}] - phase[{X + 1,Y - 3,Z + 1}] + phase[{X + 1,Y + 3,Z - 1}] - phase[{X + 1,Y - 3,Z - 1}] + phase[{X - 1,Y + 3,Z + 1}] - phase[{X - 1,Y - 3,Z + 1}] + phase[{X - 1,Y + 3,Z - 1}] - phase[{X - 1,Y - 3,Z - 1}]);
						//
						//                        n_p[2] = 352/5355 *(phase[{X,Y,Z + 1}] - phase[{X,Y,Z - 1}]) + 38.0/1071 * (phase[{X + 1,Y,Z + 1}] - phase[{X + 1,Y,Z - 1}] + phase[{X - 1,Y,Z + 1}] - phase[{X - 1,Y,Z - 1}] + phase[{X,Y + 1,Z + 1}] - phase[{X,Y + 1,Z - 1}] + phase[{X,Y - 1,Z + 1}] - phase[{X,Y - 1,Z - 1}])
						//                               + 271.0/14280 * (phase[{X + 1,Y + 1,Z + 1}] - phase[{X + 1,Y + 1,Z - 1}] + phase[{X - 1,Y + 1,Z + 1}] - phase[{X - 1,Y + 1,Z - 1}] + phase[{X + 1,Y - 1,Z + 1}] - phase[{X + 1,Y - 1,Z - 1}] + phase[{X - 1,Y - 1,Z + 1}] - phase[{X - 1,Y - 1,Z - 1}])
						//                               + 139.0/7140 * (phase[{X,Y,Z + 2}] - phase[{X,Y,Z - 2}]) + 53.0/10710 * (phase[{X - 2,Y,Z + 1}] - phase[{X - 2,Y,Z - 1}] + phase[{X + 2,Y,Z + 1}] - phase[{X + 2,Y,Z - 1}] + phase[{X,Y + 2,Z + 1}] - phase[{X,Y + 2,Z - 1}] + phase[{X,Y - 2,Z + 1}] - phase[{X,Y - 2,Z - 1}])
						//                               + 53.0/5355 * (phase[{X - 1,Y,Z + 2}] - phase[{X - 1,Y,Z - 2}] + phase[{X + 1,Y,Z + 2}] - phase[{X + 1,Y,Z - 2}] + phase[{X,Y + 1,Z + 2}] - phase[{X,Y + 1,Z - 2}] + phase[{X,Y - 1,Z + 2}] - phase[{X,Y - 1,Z - 2}])
						//                               + 5.0/2142 * (phase[{X - 2,Y + 1,Z + 1}] - phase[{X - 2,Y + 1,Z - 1}] + phase[{X + 2,Y + 1,Z + 1}] - phase[{X + 2,Y + 1,Z - 1}] + phase[{X - 1,Y + 2,Z + 1}] - phase[{X - 1,Y + 2,Z - 1}] + phase[{X + 1,Y + 2,Z + 1}] - phase[{X + 1,Y + 2,Z - 1}]
						//                                            + phase[{X - 2,Y - 1,Z + 1}] - phase[{X - 2,Y - 1,Z - 1}] + phase[{X + 2,Y - 1,Z + 1}] - phase[{X + 2,Y - 1,Z - 1}] + phase[{X - 1,Y - 2,Z + 1}] - phase[{X - 1,Y - 2,Z - 1}] + phase[{X + 1,Y - 2,Z + 1}] - phase[{X + 1,Y - 2,Z - 1}])
						//                               + 5.0/1071 * (phase[{X - 1,Y + 1,Z + 2}] - phase[{X - 1,Y + 1,Z - 2}] + phase[{X + 1,Y + 1,Z + 2}] - phase[{X + 1,Y + 1,Z - 2}] + phase[{X - 1,Y - 1,Z + 2}] - phase[{X - 1,Y - 1,Z - 2}] + phase[{X + 1,Y - 1,Z + 2}] - phase[{X + 1,Y - 1,Z - 2}])
						//                               + 41.0/42840 * (phase[{X - 2,Y,Z + 2}] - phase[{X - 2,Y,Z - 2}] + phase[{X + 2,Y,Z + 2}] - phase[{X + 2,Y,Z - 2}] + phase[{X,Y - 2,Z + 2}] - phase[{X,Y - 2,Z - 2}] + phase[{X,Y + 2,Z + 2}] - phase[{X,Y + 2,Z - 2}])
						//                               + 1.0/4284 * (phase[{X + 2,Y + 2,Z + 1}] - phase[{X + 2,Y + 2,Z - 1}] + phase[{X + 2,Y - 2,Z + 1}] - phase[{X + 2,Y - 2,Z - 1}] + phase[{X - 2,Y + 2,Z + 1}] - phase[{X - 2,Y + 2,Z - 1}] + phase[{X - 2,Y - 2,Z + 1}] - phase[{X - 2,Y - 2,Z - 1}])
						//                               + 1.0/2142 * (phase[{X + 1,Y + 2,Z + 2}] - phase[{X + 1,Y + 2,Z - 2}] + phase[{X + 1,Y - 2,Z + 2}] - phase[{X + 1,Y - 2,Z - 2}] + phase[{X - 1,Y + 2,Z + 2}] - phase[{X - 1,Y + 2,Z - 2}] + phase[{X - 1,Y - 2,Z + 2}] - phase[{X - 1,Y - 2,Z - 2}]
						//                                             + phase[{X + 2,Y + 1,Z + 2}] - phase[{X + 2,Y + 1,Z - 2}] + phase[{X + 2,Y - 1,Z + 2}] - phase[{X + 2,Y - 1,Z - 2}] + phase[{X - 2,Y + 1,Z + 2}] - phase[{X - 2,Y + 1,Z - 2}] + phase[{X - 2,Y - 1,Z + 2}] - phase[{X - 2,Y - 1,Z - 2}])
						//                               + 1.0/1785 * (phase[{X,Y,Z + 3}] - phase[{X,Y,Z - 3}]) + 1.0/10710 * (phase[{X + 3,Y,Z + 1}] - phase[{X + 3,Y,Z - 1}] + phase[{X - 3,Y,Z + 1}] - phase[{X - 3,Y,Z - 1}] + phase[{X,Y + 3,Z + 1}] - phase[{X,Y + 3,Z - 1}] + phase[{X,Y - 3,Z + 1}] - phase[{X,Y - 3,Z - 1}])
						//                               + 1.0/3570 * (phase[{X,Y + 1,Z + 3}] - phase[{X,Y + 1,Z - 3}] + phase[{X,Y - 1,Z + 3}] - phase[{X,Y - 1,Z - 3}] + phase[{X + 1,Y,Z + 3}] - phase[{X + 1,Y,Z - 3}] + phase[{X - 1,Y,Z + 3}] - phase[{X - 1,Y,Z - 3}])
						//                               + 1.0/42840 * (phase[{X + 1,Y + 3,Z + 1}] - phase[{X + 1,Y + 3,Z - 1}] + phase[{X + 1,Y - 3,Z + 1}] - phase[{X + 1,Y - 3,Z - 1}] + phase[{X - 1,Y + 3,Z + 1}] - phase[{X - 1,Y + 3,Z - 1}] + phase[{X - 1,Y - 3,Z + 1}] - phase[{X - 1,Y - 3,Z - 1}]
						//                                              + phase[{X + 3,Y + 1,Z + 1}] - phase[{X + 3,Y + 1,Z - 1}] + phase[{X + 3,Y - 1,Z + 1}] - phase[{X + 3,Y - 1,Z - 1}] + phase[{X - 3,Y + 1,Z + 1}] - phase[{X - 3,Y + 1,Z - 1}] + phase[{X - 3,Y - 1,Z + 1}] - phase[{X - 3,Y - 1,Z - 1}])
						//                               + 1.0/14280 * (phase[{X + 1,Y + 1,Z + 3}] - phase[{X + 1,Y + 1,Z - 3}] + phase[{X + 1,Y - 1,Z + 3}] - phase[{X + 1,Y - 1,Z - 3}] + phase[{X - 1,Y + 1,Z + 3}] - phase[{X - 1,Y + 1,Z - 3}] + phase[{X - 1,Y - 1,Z + 3}] - phase[{X - 1,Y - 1,Z - 3}]);

						///.........................  END OF E10 in 3D  .............................

						///****************************************************************************
						///----------------------------------------------------------------------------
						///                END OF 2-order directional derivatives
						///----------------------------------------------------------------------------
						///****************************************************************************

						///****************************************************************************
						///----------------------------------------------------------------------------
						///               START OF  4-order   directional derivatives
						///----------------------------------------------------------------------------
						///****************************************************************************
						// n_p[0] = 4.0/9 * (phase[{X + 1,Y,Z}] - phase[{X - 1,Y,Z}]) + 1.0/9 * (phase[{X + 1,Y + 1,Z}] - phase[{X - 1,Y + 1,Z}] + phase[{X + 1,Y - 1,Z}] - phase[{X - 1,Y - 1,Z}]) - 1.0/18 *(phase[{X + 2,Y,Z}] - phase[{X - 2,Y,Z}]) -1.0/72 *(phase[{X + 2,Y + 2,Z}] - phase[{X - 2,Y + 2,Z}] + phase[{X + 2,Y - 2,Z}] - phase[{X - 2,Y - 2,Z}]);
						// n_p[1] = 4.0/9 * (phase[{X,Y + 1,Z}] - phase[{X,Y - 1,Z}]) + 1.0/9 * (phase[{X + 1,Y + 1,Z}] - phase[{X + 1,Y - 1,Z}] + phase[{X - 1,Y + 1,Z}] - phase[{X - 1,Y - 1,Z}]) - 1.0/18 *(phase[{X,Y + 2,Z}] - phase[{X,Y - 2,Z}]) - 1.0/72 *(phase[{X + 2,Y + 2,Z}] - phase[{X + 2,Y - 2,Z}] + phase[{X - 2,Y + 2,Z}] - phase[{X - 2,Y - 2,Z}]);
						// n_p[2] = 0.0;
						///****************************************************************************
						///----------------------------------------------------------------------------
						///               END OF  $-order   directional derivatives
						///----------------------------------------------------------------------------
						///****************************************************************************

						///**********************************************************************************************
						///*********************************   SNOWFLAKES   GILLE  *************************************
						///**********************************************************************************************

						///-----------------------------  Qianyan Tan snowflakes in 2D for 3D contour------------------------------///
						//                        n_p[0] = 1.0/6 * (phase[{X + 1,Y,Z}] - phase[{X - 1,Y,Z}]) + 1.0/12 * (phase[{X + 1,Y + 1,Z}] - phase[{X - 1,Y + 1,Z}] + phase[{X + 1,Y - 1,Z}] - phase[{X - 1,Y - 1,Z}] + phase[{X + 1,Y,Z + 1}] - phase[{X - 1,Y,Z + 1}] + phase[{X + 1,Y,Z - 1}] - phase[{X - 1,Y,Z - 1}]);
						//                        n_p[1] = 1.0/6 * (phase[{X,Y + 1,Z}] - phase[{X,Y - 1,Z}]) + 1.0/12 * (phase[{X + 1,Y + 1,Z}] - phase[{X + 1,Y - 1,Z}] + phase[{X - 1,Y + 1,Z}] - phase[{X - 1,Y - 1,Z}] + phase[{X,Y + 1,Z + 1}] - phase[{X,Y - 1,Z + 1}] + phase[{X,Y + 1,Z - 1}] - phase[{X,Y - 1,Z - 1}]);
						//
						//                        mag_n = sqrt(SQ(n_p[0]) + SQ(n_p[1]));
						//
						//                        n_p[0] = -n_p[0]/(mag_n + 1e-20);
						//                        n_p[1] = -n_p[1]/(mag_n + 1e-20);
						//                        n_p[2] = 0.0;
						//
						//                        Ani_Funct(n_p, NN, a_s, epsilon);
						//
						//                        NN[0] = NN[0] *(mag_n * M_phase * a_s);
						//                        NN[1] = NN[1] *(mag_n * M_phase * a_s);
						//                        NN[2] = NN[2] *(mag_n * M_phase * a_s);

						///-----------------------------  Qianyan Tan 3D  for n_x^2 + n_y^2 = 1 ------------------------------///
						//                        n_p[0] = 1.0/6 * (phase[{X + 1,Y,Z}] - phase[{X - 1,Y,Z}]) + 1.0/12 * (phase[{X + 1,Y + 1,Z}] - phase[{X - 1,Y + 1,Z}] + phase[{X + 1,Y - 1,Z}] - phase[{X - 1,Y - 1,Z}] + phase[{X + 1,Y,Z + 1}] - phase[{X - 1,Y,Z + 1}] + phase[{X + 1,Y,Z - 1}] - phase[{X - 1,Y,Z - 1}]);
						//                        n_p[1] = 1.0/6 * (phase[{X,Y + 1,Z}] - phase[{X,Y - 1,Z}]) + 1.0/12 * (phase[{X + 1,Y + 1,Z}] - phase[{X + 1,Y - 1,Z}] + phase[{X - 1,Y + 1,Z}] - phase[{X - 1,Y - 1,Z}] + phase[{X,Y + 1,Z + 1}] - phase[{X,Y - 1,Z + 1}] + phase[{X,Y + 1,Z - 1}] - phase[{X,Y - 1,Z - 1}]);
						//                        n_p[2] = 1.0/6 * (phase[{X,Y,Z + 1}] - phase[{X,Y,Z - 1}]) + 1.0/12 * (phase[{X + 1,Y,Z + 1}] - phase[{X + 1,Y,Z - 1}] + phase[{X - 1,Y,Z + 1}] - phase[{X - 1,Y,Z - 1}] + phase[{X,Y + 1,Z + 1}] - phase[{X,Y + 1,Z - 1}] + phase[{X,Y - 1,Z + 1}] - phase[{X,Y - 1,Z - 1}]);
						//
						//                        const double mag_n_xy = sqrt(SQ(n_p[0]) + SQ(n_p[1]));
						//                        mag_n = sqrt(SQ(n_p[0]) + SQ(n_p[1]) + SQ(n_p[2]));
						//                        mag_gamma = sqrt(SQ(Gamma_phase[0] * n_p[0]) + SQ(Gamma_phase[1] * n_p[1]) + SQ(Gamma_phase[2] * n_p[2]));
						//
						//                        n_p[0] = -n_p[0]/(mag_n_xy + 1e-20);
						//                        n_p[1] = -n_p[1]/(mag_n_xy + 1e-20);
						//                        n_p[2] = -n_p[2]/(mag_n + 1e-20);
						//
						//                        Ani_Funct(n_p, NN, a_s, epsilon);
						//
						//                        NN[0] = NN[0] *(Gamma_phase[0] * mag_gamma * M_phase * a_s);
						//                        NN[1] = NN[1] *(Gamma_phase[1] * mag_gamma * M_phase * a_s);
						//                        NN[2] = NN[2] *(Gamma_phase[2] * mag_gamma * M_phase * a_s);

						///-----------------------------  Qianyan Tan 3D  for n_x^2 + n_y^2 + n_z^2 = 1  -------------------------///
						n_p[0] = 1.0 / 6 * (phase[{X + 1, Y, Z}] - phase[{X - 1, Y, Z}]) + 1.0 / 12 * (phase[{X + 1, Y + 1, Z}] - phase[{X - 1, Y + 1, Z}] + phase[{X + 1, Y - 1, Z}] - phase[{X - 1, Y - 1, Z}] + phase[{X + 1, Y, Z + 1}] - phase[{X - 1, Y, Z + 1}] + phase[{X + 1, Y, Z - 1}] - phase[{X - 1, Y, Z - 1}]);
						n_p[1] = 1.0 / 6 * (phase[{X, Y + 1, Z}] - phase[{X, Y - 1, Z}]) + 1.0 / 12 * (phase[{X + 1, Y + 1, Z}] - phase[{X + 1, Y - 1, Z}] + phase[{X - 1, Y + 1, Z}] - phase[{X - 1, Y - 1, Z}] + phase[{X, Y + 1, Z + 1}] - phase[{X, Y - 1, Z + 1}] + phase[{X, Y + 1, Z - 1}] - phase[{X, Y - 1, Z - 1}]);
						n_p[2] = 1.0 / 6 * (phase[{X, Y, Z + 1}] - phase[{X, Y, Z - 1}]) + 1.0 / 12 * (phase[{X + 1, Y, Z + 1}] - phase[{X + 1, Y, Z - 1}] + phase[{X - 1, Y, Z + 1}] - phase[{X - 1, Y, Z - 1}] + phase[{X, Y + 1, Z + 1}] - phase[{X, Y + 1, Z - 1}] + phase[{X, Y - 1, Z + 1}] - phase[{X, Y - 1, Z - 1}]);
						mag_n = sqrt(SQ(n_p[0]) + SQ(n_p[1]) + SQ(n_p[2]));
						n_p[0] = -n_p[0] / (mag_n + 1e-20);
						n_p[1] = -n_p[1] / (mag_n + 1e-20);
						n_p[2] = -n_p[2] / (mag_n + 1e-20);
						B = sqrt(1.0 + (SQ(Gamma_phase[2]) - 1.0) * SQ(n_p[2]));
						mag_gamma = SQ(B) * mag_n;
						Ani_Funct(n_p, NN, a_s, epsilon);
						NN[0] = NN[0] * (Gamma_phase[0] * mag_gamma * M_phase * a_s);
						NN[1] = NN[1] * (Gamma_phase[1] * mag_gamma * M_phase * a_s);
						NN[2] = NN[2] * (Gamma_phase[2] * mag_gamma * M_phase * a_s);
						///----------------------------------- End of 3D n_x^2 + n_y^2 + n_z^2 = 1  ------------------------------///
						//                        n_p[0] = 1.0/6 * (phase[{X + 1,Y,Z}] - phase[{X - 1,Y,Z}]) + 1.0/12 * (phase[{X + 1,Y + 1,Z}] - phase[{X - 1,Y + 1,Z}] + phase[{X + 1,Y - 1,Z}] - phase[{X - 1,Y - 1,Z}] + phase[{X + 1,Y,Z + 1}] - phase[{X - 1,Y,Z + 1}] + phase[{X + 1,Y,Z - 1}] - phase[{X - 1,Y,Z - 1}]);
						//                        n_p[1] = 1.0/6 * (phase[{X,Y + 1,Z}] - phase[{X,Y - 1,Z}]) + 1.0/12 * (phase[{X + 1,Y + 1,Z}] - phase[{X + 1,Y - 1,Z}] + phase[{X - 1,Y + 1,Z}] - phase[{X - 1,Y - 1,Z}] + phase[{X,Y + 1,Z + 1}] - phase[{X,Y - 1,Z + 1}] + phase[{X,Y + 1,Z - 1}] - phase[{X,Y - 1,Z - 1}]);
						//                        n_p[2] = 1.0/6 * (phase[{X,Y,Z + 1}] - phase[{X,Y,Z - 1}]) + 1.0/12 * (phase[{X + 1,Y,Z + 1}] - phase[{X + 1,Y,Z - 1}] + phase[{X - 1,Y,Z + 1}] - phase[{X - 1,Y,Z - 1}] + phase[{X,Y + 1,Z + 1}] - phase[{X,Y + 1,Z - 1}] + phase[{X,Y - 1,Z + 1}] - phase[{X,Y - 1,Z - 1}]);
						//
						//                        mag_n_xy = sqrt(SQ(n_p[0]) + SQ(n_p[1]));
						//                        mag_n = sqrt(SQ(n_p[0]) + SQ(n_p[1]) + SQ(n_p[2]));
						//                        mag_gamma = sqrt(SQ(Gamma_phase[0] * n_p[0]) + SQ(Gamma_phase[1] * n_p[1]) + SQ(Gamma_phase[2] * n_p[2]));
						//
						//                        n_p[0] = -n_p[0]/(mag_n_xy + 1e-20);
						//                        n_p[1] = -n_p[1]/(mag_n_xy + 1e-20);
						//                        n_p[2] = -n_p[2]/(mag_n + 1e-20);
						//
						//                        Ani_Funct(n_p, NN, a_s, epsilon,mag_n_xy, );
						//
						//                        NN[0] = NN[0] *(Gamma_phase[0] * mag_n_xy * M_phase * a_s);
						//                        NN[1] = NN[1] *(Gamma_phase[1] * mag_n_xy * M_phase * a_s);
						//                        NN[2] = NN[2] *(Gamma_phase[2] * mag_n_xy * M_phase * a_s);

						///--------------------------------- Start for xy-plane assumption test for 3D ----------------------------------------------------------///
						//                        n_p[0] = 1.0/6 * (phase[{X + 1,Y,Z}] - phase[{X - 1,Y,Z}]) + 1.0/12 * (phase[{X + 1,Y + 1,Z}] - phase[{X - 1,Y + 1,Z}] + phase[{X + 1,Y - 1,Z}] - phase[{X - 1,Y - 1,Z}] + phase[{X + 1,Y,Z + 1}] - phase[{X - 1,Y,Z + 1}] + phase[{X + 1,Y,Z - 1}] - phase[{X - 1,Y,Z - 1}]);
						//                        n_p[1] = 1.0/6 * (phase[{X,Y + 1,Z}] - phase[{X,Y - 1,Z}]) + 1.0/12 * (phase[{X + 1,Y + 1,Z}] - phase[{X + 1,Y - 1,Z}] + phase[{X - 1,Y + 1,Z}] - phase[{X - 1,Y - 1,Z}] + phase[{X,Y + 1,Z + 1}] - phase[{X,Y - 1,Z + 1}] + phase[{X,Y + 1,Z - 1}] - phase[{X,Y - 1,Z - 1}]);
						//                        n_p[2] = 1.0;

						//                        n_p[0] = 0.0;
						//                        n_p[1] = 0.0;
						//                        n_p[2] = 1.0;
						//
						//                        mag_n_xy = sqrt(SQ(n_p[0]) + SQ(n_p[1]));
						//                        mag_n = sqrt(SQ(n_p[0]) + SQ(n_p[1]))/(1.0 - SQ(n_p[2]));
						//                        n_p[0] = -n_p[0]/(mag_n_xy + 1e-20);
						//                        n_p[1] = -n_p[1]/(mag_n_xy + 1e-20);
						//
						//                        Ani_Funct(n_p, NN, a_s, epsilon);
						//                        NN[0] = NN[0] *(Gamma_phase[0] * mag_n * M_phase * a_s);
						//                        NN[1] = NN[1] *(Gamma_phase[1] * mag_n * M_phase * a_s);
						//                        NN[2] = NN[2] *(Gamma_phase[2] * mag_n * M_phase * a_s);
						///--------------------------------- End of xy-plane test -------------------------------------------------------------------------------///

						///------------------------------------------------ Start of Mandelic Acid ---------------------------------------------------------------///
						//                        n_p[0] = 1.0/6 * (phase[{X + 1,Y,Z}] - phase[{X - 1,Y,Z}]) + 1.0/12 * (phase[{X + 1,Y + 1,Z}] - phase[{X - 1,Y + 1,Z}] + phase[{X + 1,Y - 1,Z}] - phase[{X - 1,Y - 1,Z}] + phase[{X + 1,Y,Z + 1}] - phase[{X - 1,Y,Z + 1}] + phase[{X + 1,Y,Z - 1}] - phase[{X - 1,Y,Z - 1}]);
						//                        n_p[1] = 1.0/6 * (phase[{X,Y + 1,Z}] - phase[{X,Y - 1,Z}]) + 1.0/12 * (phase[{X + 1,Y + 1,Z}] - phase[{X + 1,Y - 1,Z}] + phase[{X - 1,Y + 1,Z}] - phase[{X - 1,Y - 1,Z}] + phase[{X,Y + 1,Z + 1}] - phase[{X,Y - 1,Z + 1}] + phase[{X,Y + 1,Z - 1}] - phase[{X,Y - 1,Z - 1}]);
						//                        mag_n = sqrt(SQ(n_p[0]) + SQ(n_p[1]));
						//
						//                        n_p[0] = -n_p[0]/(mag_n + 1e-20);
						//                        n_p[1] = -n_p[1]/(mag_n + 1e-20);
						//                        n_p[2] = 0.0;
						//
						//                        Ani_Funct(n_p, NN, a_s, epsilon);
						//
						//                        NN[0] = NN[0] *(mag_n * M_phase * a_s);
						//                        NN[1] = NN[1] *(mag_n * M_phase * a_s);
						//                        NN[2] = NN[2] *(mag_n * M_phase * a_s);
						///------------------------------------------------- End of Mandelic Acid ---------------------------------------------------------------///
						omega_p[{X, Y, Z}] = 1. / (c_s2 * a_s * a_s * M_phase + 0.5);
						equilibrium_p(phase[{X, Y, Z}], 1., NN);
						///******************************    END OF SNOW ***********************************************
						for (alpha = 0; alpha < Discrete_Velocity; alpha++) {
							pop_p[{X + c_alpha[alpha][0], Y + c_alpha[alpha][1], Z + c_alpha[alpha][2], alpha}] =
								(pop_old_p[{X, Y, Z, alpha}] * (1 - omega_p[{X, Y, Z}]) - (1. - a_s * a_s) * pop_old_p[{X + c_alpha[alpha][0], Y + c_alpha[alpha][1], Z + c_alpha[alpha][2], alpha}]
							     + pop_eq_p[alpha] * omega_p[{X, Y, Z}] + weight[alpha] * Production[{X, Y, Z}])
								/ (a_s * a_s);
						}
					}
				}
			}
		}
		return;
	}
}
void Phase_Field::BC_p(int time, Flow_solver* Flow, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		int bou;
		int X, Y, Z, alpha, Xp, Yp, Zp;
		for (int k = 0; k < Boundaries.size(); ++k) {
			X = Boundaries[k].X;
			Y = Boundaries[k].Y;
			Z = Boundaries[k].Z;
			for (alpha = 1; alpha < Discrete_Velocity; alpha++) {
				Xp = (X - c_alpha[alpha][0]);
				Yp = (Y - c_alpha[alpha][1]);
				Zp = (Z - c_alpha[alpha][2]);
				if (solid_phase_type[{Xp, Yp, Zp}] != -1) {
					bou = Boundaries[k].type;
					switch (bou) {
						case 1: {
							if (DOT(c_alpha[alpha], Boundaries[k].n) >= 0) {
								pop_p[{X, Y, Z, alpha}] = -pop_p[{Xp, Yp, Zp, alpha_bar[alpha]}];  /// Zero phase field
							}
							break;
						}
						case 2: {
							if (DOT(c_alpha[alpha], Boundaries[k].n) >= 0) {
								pop_p[{X, Y, Z, alpha}] = (weight[alpha] + weight[alpha_bar[alpha]]) * (Gamma * Boundaries[k].Phase)
								                          - pop_p[{Xp, Yp, Zp, alpha_bar[alpha]}];  /// Non-zero temperature on walls
							}
							break;
						}
						case 3: {
							if (DOT(c_alpha[alpha], Boundaries[k].n) > 0) {
								pop_p[{X, Y, Z, alpha}] = pop_p[{X + Boundaries[k].n[0], Y + Boundaries[k].n[1], Z + Boundaries[k].n[2], alpha}];  /// Zero-gradient
							}
							break;
						}
					}
				}
			}
		}
		return;
	}
}
void Phase_Field::momenta_p(Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		int X, Y, Z, alpha;
		for (X = 2; X <= MPI_parallel->actual_rows_XYZ[0] + 1; ++X) {
			for (Y = 2; Y <= MPI_parallel->actual_rows_XYZ[1] + 1; ++Y) {
				for (Z = 2; Z <= MPI_parallel->actual_rows_XYZ[2] + 1; ++Z) {
					if (solid_phase_type[{X, Y, Z}] == -1) {
						previous_phase[{X, Y, Z}] = phase[{X, Y, Z}];
						phase[{X, Y, Z}] = 0;
						for (alpha = 0; alpha < Discrete_Velocity; alpha++) {
							phase[{X, Y, Z}] += pop_p[{X, Y, Z, alpha}];
						}
					}
				}
			}
		}
	}
	return;
}
void Phase_Field::Data_Exchange(Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		pop_group.exchange_data();
	}
}

void Phase_Field::Data_Exchange_Macroscopic(Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		macroscopic_group.exchange_data();
	}
}

bool Phase_Field::check_residual(double res_thermal, Parallel_MPI* MPI_parallel, int t) {  // Based on paper Applied Mathematical Modelling 39 (2015) 2436�2451
	bool st_cr = false;
	double phase_criteria = 0;
	double phase_criteria_2 = 0;
	double phase_global, phase_global_2;
	if (MPI_parallel->processor_id != MASTER) {
		for (int X = MPI_parallel->start_XYZ2[0]; X < MPI_parallel->end_XYZ2[0]; X++) {
			for (int Y = MPI_parallel->start_XYZ2[1]; Y < MPI_parallel->end_XYZ2[1]; Y++) {
				for (int Z = MPI_parallel->start_XYZ2[2]; Z < MPI_parallel->end_XYZ2[2]; Z++) {
					//  if (is_solid[X][Y] == FALSE)			{
					phase_criteria += abs(phase[{X, Y, Z}] - previous_phase[{X, Y, Z}]);
					phase_criteria_2 += abs(phase[{X, Y, Z}]);
					// }
				}
			}
		}
	}
	MPI_Allreduce(&phase_criteria, &phase_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&phase_criteria_2, &phase_global_2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	if ((phase_global / phase_global_2) < res_thermal) {
		st_cr = true;
		return st_cr;
	}
	if (t % t_residual == 0 && MPI_parallel->processor_id == MASTER) {
		std::cout << t << " Phase Residual = " << (phase_global / phase_global_2) << endl;
	}
	return st_cr;
}

void Phase_Field::register_recovery(IO_interface& io) {
	io.add_field(pop_p, "phase_field_pop_p");
	// todo: just set to pop_p with custom task
	io.add_field(pop_old_p, "phase_field_pop_old_p");
	io.add_field(phase, "phase_field_phase");
	io.add_field(previous_phase, "phase_field_previous_phase");
	io.add_field(solid_phase_type, "phase_field_solid_phase_type");
}

Phase_Field::~Phase_Field() {
}
void Inline_User_Defined_p(Scalar_field& Phase, Solid_field& solid, double N_x, double N_y, double N_z, const std::string& filename, int Zones, Parallel_MPI* MPI_parallel) {
	///----------------------------------------------------------------------///
	///                        COMMAND LINE INPUT                            ///
	///----------------------------------------------------------------------///
	if (MPI_parallel->processor_id != MASTER) {
		double* Ini_P;
		int* type;
		int X, Y, Z;

		Ini_P = new double[Zones + 1];
		type = new int[Zones + 1];
		int index;
		for (int i = 0; i < Zones + 1; i++) {
			type[i] = 0;
			Ini_P[i] = 0;
		}
		type[0] = 1;  /// -----> Solids are set to +1
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
		} while (Line1.find("c\tPhase Field Initial Conditions") == string::npos);
		if (MPI_parallel->processor_id == (MASTER + 1)) {
			std::cout << "FLOW: User defined Initial conditions \n";
			std::cout << "================================ \n";
		}
		for (int i = 0; i < Zones; i++) {
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
			input_file >> type[index];
			input_file >> Ini_P[index];
			if (MPI_parallel->processor_id == (MASTER + 1)) {
				std::cout << "Zone : " << index << std::endl;
				std::cout << "Type : " << type[index] << std::endl;
				std::cout << "Initial Phase : " << Ini_P[index] << std::endl;
			}
		}
		// double xx, yy, zz;
		for (X = 0; X < MPI_parallel->dev_end[0]; X++) {
			// xx = fmod(X - MPI_parallel->start_XYZ2[0] + MPI_parallel->start_XYZ[0] + N_x + 1, N_x);
			for (Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
				// yy = fmod(Y - MPI_parallel->start_XYZ2[1] + MPI_parallel->start_XYZ[1] + N_y, N_y);
				for (Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
					// zz = fmod(Z - MPI_parallel->start_XYZ2[2] + MPI_parallel->start_XYZ[2] + N_z, N_z);
					//                Phase[X][Y][Z] = -1.;
					//                solid[{X,Y,Z}] = -1.;
					//                if ( sqrt( pow(xx-50,2)+pow(yy-50,2)+pow(zz-50,2) )<3  ) {
					//                    Phase[X][Y][Z] = 1.;
					//                }
					Phase[{X,Y,Z}] = Ini_P[solid[{X,Y,Z}]];
					solid[{X,Y,Z}] = type[solid[{X,Y,Z}]];
				}
			}
		}
	}
}
/// void Anisotropy_2D(double *n, double *NN, double &a_s, double &epsilon){
void Anisotropy_multi(double* n_p, double* NN, double& a_s, double* epsilon) {  /// new phase-field added
	/// Obviously setting epsilon to 0, results in Isotropic growth
	a_s = 1. + epsilon[0] * (pow(n_p[0], 4) + pow(n_p[1], 4) + pow(n_p[2], 4) - 3. / 5.) - 0.02 * (3 * (pow(n_p[0], 4) + pow(n_p[1], 4) + pow(n_p[2], 4)) + 66. * n_p[0] * n_p[0] * n_p[1] * n_p[1] * n_p[1] * n_p[2] * n_p[2] - 17. / 7.);
	NN[0] = a_s * 4. * epsilon[0] * pow(n_p[0], 3) + 0.02 * (3. * 4. * pow(n_p[0], 3) + 66. * 2. * n_p[0] * n_p[1] * n_p[1] * n_p[2] * n_p[2]);
	NN[1] = a_s * 4. * epsilon[0] * pow(n_p[1], 3) + 0.02 * (3. * 4. * pow(n_p[1], 3) + 66. * 2. * n_p[0] * n_p[0] * n_p[1] * n_p[2] * n_p[2]);
	NN[2] = a_s * 4. * epsilon[0] * pow(n_p[2], 3) + 0.02 * (3. * 4. * pow(n_p[2], 3) + 66. * 2. * n_p[0] * n_p[0] * n_p[1] * n_p[1] * n_p[2]);
}

void Tetrahedral_2D(double* n_p, double* NN, double& a_s, double* epsilon) {  /// new or six dendrites along coordinates in X, Y, Z directions in 3D
	a_s = 1. - 3. * epsilon[0] + 4. * epsilon[0] * (pow(n_p[0], 4) + pow(n_p[1], 4) + pow(n_p[2], 4));

	NN[0] = -16. * epsilon[0] * n_p[0] * (pow(n_p[1], 4) - pow(n_p[0], 2) * pow(n_p[1], 2) - pow(n_p[0], 2) * pow(n_p[2], 2) + pow(n_p[2], 4));
	NN[1] = -16. * epsilon[0] * n_p[1] * (pow(n_p[0], 4) - pow(n_p[1], 2) * pow(n_p[0], 2) - pow(n_p[1], 2) * pow(n_p[2], 2) + pow(n_p[2], 4));
	NN[2] = -16. * epsilon[0] * n_p[2] * (pow(n_p[0], 4) - pow(n_p[2], 2) * pow(n_p[0], 2) - pow(n_p[2], 2) * pow(n_p[1], 2) + pow(n_p[1], 4));
}

void Hexahedral_2D(double* n_p, double* NN, double& a_s, double* epsilon) {  /// new phase-field added
																			 //	const double axy = pow(n_p[0], 6) - 15 * pow(n_p[0], 4) * pow(n_p[1], 2) + 15 * pow(n_p[0], 2) * pow(n_p[1], 4) - pow(n_p[1], 6);
	///******  normal 2D 6 tips function  ************///
	a_s = 1. + epsilon[0] * (pow(n_p[0], 6) - 15 * pow(n_p[0], 4) * pow(n_p[1], 2) + 15 * pow(n_p[0], 2) * pow(n_p[1], 4) - pow(n_p[1], 6));
	NN[0] = 12. * epsilon[0] * n_p[0] * (3. * pow(n_p[0], 4) * pow(n_p[1], 2) - 10. * pow(n_p[0], 2) * pow(n_p[1], 4) + 3. * pow(n_p[1], 6));
	NN[1] = -12. * epsilon[0] * n_p[1] * (3. * pow(n_p[0], 2) * pow(n_p[1], 4) - 10. * pow(n_p[0], 4) * pow(n_p[1], 2) + 3. * pow(n_p[0], 6));
	NN[2] = 0.;
	///-----------------------------------------------------------------------------------
	/// initial angle rotation 30 degree
	//        a_s = 1. - epsilon[0] * ( pow(n_p[0],6) - 15 * pow(n_p[0],4) * pow(n_p[1],2) + 15 * pow(n_p[0],2) * pow(n_p[1],4) - pow(n_p[1],6) );
	//        NN[0] = -12. * epsilon[0] * n_p[0] * (3. * pow(n_p[0],4) * pow(n_p[1],2) - 10. * pow(n_p[0],2) * pow(n_p[1],4) + 3.* pow(n_p[1],6));
	//        NN[1] =  12. * epsilon[0] * n_p[1] * (3. * pow(n_p[0],2) * pow(n_p[1],4) - 10. * pow(n_p[0],4) * pow(n_p[1],2) + 3.* pow(n_p[0],6));
	//        NN[2] =  0.;

	///*****   initial angle rotation ******************////
	//        double theta_0 = M_PI/6;///clock-direction rotation
	//        a_s = 1. + epsilon[0] * (cos(6.0*theta_0)*( pow(n_p[0],6) - 15 * pow(n_p[0],4) * pow(n_p[1],2) + 15 * pow(n_p[0],2) * pow(n_p[1],4) - pow(n_p[1],6) ) + 2.0 * sin(6.0*theta_0)*(3.*pow(n_p[0],5) * n_p[1]-10.*pow(n_p[0],3) * pow(n_p[1],3)+3.*pow(n_p[1],5) * n_p[0]));
	//        NN[0] =  12. * epsilon[0] * n_p[0] *cos(6.0*theta_0)*(3. * pow(n_p[0],4) * pow(n_p[1],2) - 10. * pow(n_p[0],2) * pow(n_p[1],4) + 3.* pow(n_p[1],6))-6.*sin(6.*theta_0)*n_p[1]*epsilon[0]*axy;
	//        NN[1] = -12. * epsilon[0] * n_p[1] *cos(6.0*theta_0)*(3. * pow(n_p[0],2) * pow(n_p[1],4) - 10. * pow(n_p[0],4) * pow(n_p[1],2) + 3.* pow(n_p[0],6))+6.*sin(6.*theta_0)*n_p[0]*epsilon[0]*axy;
	//        NN[2] = 0.;
	///-----------------------------------------------------------------------------------

	///**************   regularization ***********************************///
	//        const double aa = 1.0 - 1./epsilon[0];
	//        const double nnx = sqrt(21./(400.* pow((sqrt(SQ(aa/320. - 81./8000.) - 9261./64000000.)- aa/320. + 81./8000), 1./3)) + pow((sqrt(SQ(aa/320. - 81./8000) - 9261./64000000.) - aa/320. + 81./8000.),1.0/3) + 3./10);
	//        ///threshold for xy-axis angle   *****cos(theta_m)****
	//
	//        const double nny = sqrt(1. - SQ(nnx));  ///threshold for xy-axis angle   *****sin(theta_m)****
	//        const double cos_6theta_m = pow(nnx,6) - 15 * pow(nnx,4) * pow(nny,2) + 15 * pow(nnx,2) * pow(nny,4) - pow(nny,6); ///cos(6*theta_m)
	//        const double B1 = (1.0 + epsilon[0] * cos_6theta_m)/nnx;  ///B_1    6*epsilon_xy*sin(6*theta_m)/sin(theta_m)
	//
	//        if ((n_p[0]>=nnx))
	//        {
	//        a_s = B1 * n_p[0];
	//        NN[0] =  B1 * SQ(n_p[1]);
	//        NN[1] = -B1 * n_p[0] * n_p[1];
	//        NN[2] = 0.0;
	//        }
	//        else if ((n_p[0]>= (0.5 * nnx - sqrt(3.)/2. * nny)) && (n_p[0]<= (0.5 * nnx + sqrt(3.)/2. * nny)) && (n_p[1]> 0))
	//        {
	//        a_s = B1 * (0.5 * n_p[0] + sqrt(3.)/2. * n_p[1]);
	//        NN[0] = B1 * ( 0.5 * SQ(n_p[1]) - sqrt(3.)/2. * n_p[0] * n_p[1]);
	//        NN[1] = B1 * (-0.5 * n_p[0] * n_p[1] + sqrt(3.)/2 * SQ(n_p[0]));
	//        NN[2] = 0.0;
	//        }
	//        else if((n_p[0]>= (-0.5 * nnx - sqrt(3.)/2. * nny)) && (n_p[0]<= (-0.5 * nnx + sqrt(3.)/2. * nny))&& (n_p[1]>0))
	//        {
	//        a_s = B1 * (-0.5 * n_p[0] + sqrt(3.)/2. * n_p[1]);
	//        NN[0] = B1 * (-0.5 * SQ(n_p[1]) - sqrt(3.)/2. * n_p[0] * n_p[1]);
	//        NN[1] = B1 * ( 0.5 * n_p[0] * n_p[1] + sqrt(3.)/2 * SQ(n_p[0]));
	//        NN[2] = 0.0;
	//        }
	//        else if((n_p[0]<= -nnx))
	//        {
	//        a_s = B1 * (-n_p[0]);
	//        NN[0] = -B1 * SQ(n_p[1]);
	//        NN[1] =  B1 * n_p[0] * n_p[1];
	//        NN[2] =  0.0;
	//        }
	//        else if((n_p[0]>= (-0.5 * nnx - sqrt(3.)/2. * nny)) && (n_p[0]<= (-0.5 * nnx + sqrt(3.)/2. * nny))&& (n_p[1]<0))
	//        {
	//        a_s = B1 * (-0.5 * n_p[0] - sqrt(3.)/2. * n_p[1]);
	//        NN[0] = B1 * (-0.5 * SQ(n_p[1]) + sqrt(3.)/2. * n_p[0] * n_p[1]);
	//        NN[1] = B1 * ( 0.5 * n_p[0] * n_p[1] - sqrt(3.)/2 * SQ(n_p[0]));
	//        NN[2] = 0.0;
	//        }
	//        else if((n_p[0]>= (0.5 * nnx - sqrt(3.)/2. * nny)) && (n_p[0]<= (0.5 * nnx + sqrt(3.)/2. * nny))&& (n_p[1]<0))
	//        {
	//        a_s = B1 * (0.5 * n_p[0] - sqrt(3.)/2. * n_p[1]);
	//        NN[0] = B1 * (0.5 * SQ(n_p[1]) + sqrt(3.)/2. * n_p[0] * n_p[1]);
	//        NN[1] = B1 * (-0.5 * n_p[0] * n_p[1] - sqrt(3.)/2 * SQ(n_p[0]));
	//        NN[2] = 0.0;
	//        }
	//        else{
	//        a_s = 1. + epsilon[0] * ( pow(n_p[0],6) - 15 * pow(n_p[0],4) * pow(n_p[1],2) + 15 * pow(n_p[0],2) * pow(n_p[1],4) - pow(n_p[1],6) );
	//        NN[0] =  12. * epsilon[0] * n_p[0] * (3. * pow(n_p[0],4) * pow(n_p[1],2) - 10. * pow(n_p[0],2) * pow(n_p[1],4) + 3.* pow(n_p[1],6));
	//        NN[1] = -12. * epsilon[0] * n_p[1] * (3. * pow(n_p[0],2) * pow(n_p[1],4) - 10. * pow(n_p[0],4) * pow(n_p[1],2) + 3.* pow(n_p[0],6));
	//        NN[2] = 0.;
	//        }
	///******************************    END OF regularization ***********************///
}
void snow_vapor(double* n_p, double* NN, double& a_s, double* epsilon) {                                                               /// new phase-field added
	const double axy = pow(n_p[0], 6) - 15 * pow(n_p[0], 4) * pow(n_p[1], 2) + 15 * pow(n_p[0], 2) * pow(n_p[1], 4) - pow(n_p[1], 6);  /// cos6theta
																																	   //	const double sin6theta = 6.0 * n_p[1] * pow(n_p[0], 5) + 6.0 * n_p[0] * pow(n_p[1], 5) - 20.0 * pow(n_p[0], 3) * pow(n_p[1], 3);
	                                                                                                                                   //	const double sq_p1 = SQ(n_p[1]) + SQ(n_p[2]);
	                                                                                                                                   //	const double sq_p2 = SQ(n_p[0]) + SQ(n_p[1]);
	                                                                                                                                   //	const double sq_p3 = SQ(n_p[0]) + SQ(n_p[2]);
	                                                                                                                                   //	const double sq_p4 = SQ(n_p[0]) + SQ(n_p[1]) + SQ(n_p[2]);

	///************************************************************
	///  Case 1 (1)
	///  As = 1 + ep_xy * sin^6(phi) * cos(6 * theta) + ep_z * cos(2 * phi)
	///  for n_x^2 + n_y^2 + n_z^2 = 1
	///**************************************************************
	//    a_s = 1. + epsilon[0] * axy + epsilon[2] * (2.0*SQ(n_p[2])-1.0);
	//   	NN[0] =  6 * epsilon[0] * n_p[0] * (6 * pow(n_p[1], 6) + 6 * pow(n_p[0], 4) * pow(n_p[1], 2) + pow(n_p[0], 4) * pow(n_p[2], 2) + 5 * pow(n_p[1], 4)* pow(n_p[2], 2) - 20 * pow(n_p[0], 2) * pow(n_p[1], 4) - 10 * pow(n_p[0], 2) * pow(n_p[1], 2)* pow(n_p[2], 2)) - 4 * epsilon[2] *n_p[0]* SQ(n_p[2]);
	//	NN[1] = -6 * epsilon[0] * n_p[1] * (6 * pow(n_p[0], 6) + 6 * pow(n_p[1], 4) * pow(n_p[0], 2) + pow(n_p[1], 4) * pow(n_p[2], 2) + 5 * pow(n_p[0], 4)* pow(n_p[2], 2) - 20 * pow(n_p[1], 2) * pow(n_p[0], 4) - 10 * pow(n_p[0], 2) * pow(n_p[1], 2)* pow(n_p[2], 2)) - 4 * epsilon[2] *n_p[1]* SQ(n_p[2]);
	//  	NN[2] = -6 * epsilon[0] * n_p[2] * axy + 4 * epsilon[2] * n_p[2] * (SQ(n_p[0]) + SQ(n_p[1]));
	///---------------------------------------------------------------------------------------------------

	///************************************************************
	///  Case 1 (2)
	///  As = 1 + ep_xy * sin^6(phi) * cos(6 * theta) + ep_z * cos(2 * phi)
	///  for n_x^2 + n_y^2 = 1
	///**************************************************************
	//    a_s = 1.0 + epsilon[0] * axy * pow((1.0 - SQ(n_p[2])), 3) + epsilon[2] * (2.0*SQ(n_p[2])-1.0);
	//   	NN[0] =  6 * epsilon[0] * n_p[0] * pow((1.0 - SQ(n_p[2])), 2.5) * (6 * pow(n_p[1], 6) + 6 * pow(n_p[0], 4) * pow(n_p[1], 2) - 20 * pow(n_p[0], 2) * pow(n_p[1], 4) + axy * SQ(n_p[2])) - 4.0 * epsilon[2] * sqrt(1.0 - SQ(n_p[2])) * SQ(n_p[2]) * n_p[0];
	//	NN[1] =  -6 * epsilon[0] * n_p[1] * pow((1.0 - SQ(n_p[2])), 2.5) * (6 * pow(n_p[0], 6) + 6 * pow(n_p[1], 4) * pow(n_p[0], 2) - 20 * pow(n_p[1], 2) * pow(n_p[0], 4) - axy * SQ(n_p[2])) - 4.0 * epsilon[2] * sqrt(1.0 - SQ(n_p[2])) * SQ(n_p[2]) * n_p[1];
	//  	NN[2] =  -6 * epsilon[0] * n_p[2] * axy * pow((1.0 - SQ(n_p[2])), 3) + 4 * epsilon[2] * n_p[2] * (1.0 - SQ(n_p[2]));
	///---------------------------------------------------------------------------------------------------

	///******************************************************************************************************************************
	///  Case 1(3)  Regularize
	///  As = 1 + ep_xy * sin^6(phi) * cos(6 * theta) + ep_z * cos(2 * phi)
	///  for n_x^2 + n_y^2 = 1
	///***************************************************************************************************************************
	//    const double aa = 1.0 - (1 + epsilon[2]*(2.*SQ(n_p[2]) -1.))/(epsilon[0]*pow((1.0 - SQ(n_p[2])), 3));
	//    const double nnx = sqrt(21./(400.* pow((sqrt(SQ(aa/320. - 81./8000.) - 9261./64000000.) - aa/320. + 81./8000), 1./3)) + pow((sqrt(SQ(aa/320. - 81./8000) - 9261./64000000.) - aa/320. + 81./8000.),1.0/3) + 3./10);
	//	///threshold for xy-axis angle   *****cos(theta_m)****
	//	const double nny = sqrt(1. - SQ(nnx));  ///threshold for xy-axis angle   *****sin(theta_m)****
	//	const double cos_6theta_m = pow(nnx,6) - 15 * pow(nnx,4) * pow(nny,2) + 15 * pow(nnx,2) * pow(nny,4) - pow(nny,6); ///cos(6*theta_m)
	//	const double B1 = 12. * epsilon[0] *pow((1.0 - SQ(n_p[2])),3)* (3. * pow(nnx, 5) + 3. * nnx * pow(nny,4)  - 10. * pow(nnx,3) * pow(nny,2));  ///B_1    6*epsilon_xy*sin(6*theta_m)/sin(theta_m)
	//    const double A1 = 1.0 + epsilon[0] * pow((1.0 - SQ(n_p[2])),3) * cos_6theta_m + epsilon[2] * (2. * SQ(n_p[2]) - 1.0) - B1 * nnx;
	//    const double grad_B1 = 72. * epsilon[0] *pow((1.0 - SQ(n_p[2])),2.5)*n_p[2]* (3. * pow(nnx, 5) + 3. * nnx * pow(nny,4)  - 10. * pow(nnx,3) * pow(nny,2));
	//    const double grad_A1 = 6.0 * epsilon[0] * pow((1.0 - SQ(n_p[2])),2.5) * n_p[2]*cos_6theta_m - 4.0* epsilon[2] * sqrt(1.0 - SQ(n_p[2])) * n_p[2] - grad_B1 * nnx;
	//        if ((n_p[0]>=nnx))
	//        {
	//        a_s = A1 + B1 * n_p[0];
	//        NN[0] =  B1 * n_p[1] * n_p[1]/sqrt(1.0 - SQ(n_p[2])) + (grad_A1 + n_p[0] * grad_B1) * n_p[2] * n_p[0];
	//        NN[1] = -B1 * n_p[1] * n_p[0]/sqrt(1.0 - SQ(n_p[2])) + (grad_A1 + n_p[0] * grad_B1) * n_p[2] * n_p[1];
	//        NN[2] = -(grad_A1 + n_p[0] * grad_B1) * sqrt(1.0 - SQ(n_p[2]));
	//        }
	//        else if ((n_p[0]>= (0.5 * nnx - sqrt(3.)/2. * nny)) && (n_p[0]<= (0.5 * nnx + sqrt(3.)/2. * nny)) && (n_p[1]> 0))
	//        {
	//        a_s = A1 + B1 * (0.5 * n_p[0] + sqrt(3.)/2. * n_p[1]);
	//        NN[0] =  B1 * (0.5 * n_p[1] - sqrt(3.)/2. * n_p[0]) * n_p[1]/sqrt(1.0 - SQ(n_p[2])) + (grad_A1 + (0.5 * n_p[0] + sqrt(3.)/2. * n_p[1]) * grad_B1) * n_p[2] * n_p[0];
	//        NN[1] = -B1 * (0.5 * n_p[1] - sqrt(3.)/2. * n_p[0]) * n_p[0]/sqrt(1.0 - SQ(n_p[2])) + (grad_A1 + (0.5 * n_p[0] + sqrt(3.)/2. * n_p[1]) * grad_B1) * n_p[2] * n_p[1];
	//        NN[2] = -(grad_A1 + (0.5 * n_p[0] + sqrt(3.)/2. * n_p[1]) * grad_B1) * sqrt(1.0 - SQ(n_p[2]));
	//        }
	//        else if((n_p[0]>= (-0.5 * nnx - sqrt(3.)/2. * nny)) && (n_p[0]<= (-0.5 * nnx + sqrt(3.)/2. * nny))&& (n_p[1]>0))
	//        {
	//        a_s = A1 + B1 * (-0.5 * n_p[0] + sqrt(3.)/2. * n_p[1]);
	//        NN[0] = -B1 * (0.5 * n_p[1] + sqrt(3.)/2. * n_p[0]) * n_p[1]/sqrt(1.0 - SQ(n_p[2])) + (grad_A1 + (-0.5 * n_p[0] + sqrt(3.)/2. * n_p[1]) * grad_B1) * n_p[2] * n_p[0];
	//        NN[1] =  B1 * (0.5 * n_p[1] + sqrt(3.)/2. * n_p[0]) * n_p[0]/sqrt(1.0 - SQ(n_p[2])) + (grad_A1 + (-0.5 * n_p[0] + sqrt(3.)/2. * n_p[1]) * grad_B1) * n_p[2] * n_p[1];
	//        NN[2] = -(grad_A1 + (-0.5 * n_p[0] + sqrt(3.)/2. * n_p[1]) * grad_B1) * sqrt(1.0 - SQ(n_p[2]));
	//        }
	//        else if((n_p[0]<= -nnx))
	//        {
	//        a_s = A1 + B1 * (-n_p[0]);
	//        NN[0] = -B1 * n_p[1] * n_p[1]/sqrt(1.0 - SQ(n_p[2])) + (grad_A1 + (-n_p[0]) * grad_B1) * n_p[2] * n_p[0];
	//        NN[1] =  B1 * n_p[1] * n_p[0]/sqrt(1.0 - SQ(n_p[2])) + (grad_A1 + (-n_p[0]) * grad_B1) * n_p[2] * n_p[1];
	//        NN[2] = -(grad_A1 + (-n_p[0]) * grad_B1) * sqrt(1.0 - SQ(n_p[2]));
	//        }
	//        else if((n_p[0]>= (-0.5 * nnx - sqrt(3.)/2. * nny)) && (n_p[0]<= (-0.5 * nnx + sqrt(3.)/2. * nny))&& (n_p[1]<0))
	//        {
	//        a_s = A1 + B1 * (-0.5 * n_p[0] - sqrt(3.)/2. * n_p[1]);
	//        NN[0] = -B1 * (0.5 * n_p[1] - sqrt(3.)/2. * n_p[0]) * n_p[1]/sqrt(1.0 - SQ(n_p[2])) + (grad_A1 + (-0.5 * n_p[0] - sqrt(3.)/2. * n_p[1]) * grad_B1) * n_p[2] * n_p[0];
	//        NN[1] =  B1 * (0.5 * n_p[1] - sqrt(3.)/2. * n_p[0]) * n_p[0]/sqrt(1.0 - SQ(n_p[2])) + (grad_A1 + (-0.5 * n_p[0] - sqrt(3.)/2. * n_p[1]) * grad_B1) * n_p[2] * n_p[1];
	//        NN[2] = -(grad_A1 + (-0.5 * n_p[0] - sqrt(3.)/2. * n_p[1]) * grad_B1) * sqrt(1.0 - SQ(n_p[2]));
	//        }
	//        else if((n_p[0]>= (0.5 * nnx - sqrt(3.)/2. * nny)) && (n_p[0]<= (0.5 * nnx + sqrt(3.)/2. * nny))&& (n_p[1]<0))
	//        {
	//        a_s = A1 + B1 * (0.5 * n_p[0] - sqrt(3.)/2. * n_p[1]);
	//        NN[0] =  B1 * (0.5 * n_p[1] + sqrt(3.)/2. * n_p[0]) * n_p[1]/sqrt(1.0 - SQ(n_p[2])) + (grad_A1 + (0.5 * n_p[0] - sqrt(3.)/2. * n_p[1]) * grad_B1) * n_p[2] * n_p[0];
	//        NN[1] = -B1 * (0.5 * n_p[1] + sqrt(3.)/2. * n_p[0]) * n_p[0]/sqrt(1.0 - SQ(n_p[2])) + (grad_A1 + (0.5 * n_p[0] - sqrt(3.)/2. * n_p[1]) * grad_B1) * n_p[2] * n_p[1];
	//        NN[2] = -(grad_A1 + (0.5 * n_p[0] - sqrt(3.)/2. * n_p[1]) * grad_B1) * sqrt(1.0 - SQ(n_p[2]));
	//       }
	//        else{
	//        a_s = 1. + epsilon[0] * axy * pow((1.0 - SQ(n_p[2])), 3) + epsilon[2] * (2.0*SQ(n_p[2])-1.0);
	//        NN[0] =  6 * epsilon[0] * n_p[0] * pow((1.0 - SQ(n_p[2])), 2.5) * (6 * pow(n_p[1], 6) + 6 * pow(n_p[0], 4) * pow(n_p[1], 2) - 20 * pow(n_p[0], 2) * pow(n_p[1], 4) + axy * SQ(n_p[2])) - 4.0 * epsilon[2] * sqrt(1.0 - SQ(n_p[2])) * SQ(n_p[2]) * n_p[0];
	//        NN[1] = -6 * epsilon[0] * n_p[1] * pow((1.0 - SQ(n_p[2])), 2.5) * (6 * pow(n_p[0], 6) + 6 * pow(n_p[1], 4) * pow(n_p[0], 2) - 20 * pow(n_p[1], 2) * pow(n_p[0], 4) - axy * SQ(n_p[2])) - 4.0 * epsilon[2] * sqrt(1.0 - SQ(n_p[2])) * SQ(n_p[2]) * n_p[1];
	//        NN[2] = -6 * epsilon[0] * n_p[2] * axy * pow((1.0 - SQ(n_p[2])), 3) + 4 * epsilon[2] * n_p[2] * (1.0 - SQ(n_p[2]));
	//
	//        }
	///------------------------------- End of regular mit sin^6  -------------------------------------///
	///  Case 2 (1)
	///  As = 1 + ep_xy * cos(6 * theta) + ep_z * cos(2 * phi)
	///  for n_x^2 + n_y^2 + n_z^2 = 1
	///*******************************************************
	//    a_s = 1. + epsilon[0] * axy/(pow(sq_p2,3) + 1e-20) + epsilon[2] * (2.0*SQ(n_p[2])-1.0);
	//    NN[0] =  6 * epsilon[0] * n_p[0] * (6 * pow(n_p[1], 6) + 6 * pow(n_p[0], 4) * pow(n_p[1], 2) - 20 * pow(n_p[0], 2) * pow(n_p[1], 4))/(pow((1.0 - SQ(n_p[2])),4) + 1e-20) - 4 * epsilon[2] *n_p[0]* SQ(n_p[2]);
	//	NN[1] = -6 * epsilon[0] * n_p[1] * (6 * pow(n_p[0], 6) + 6 * pow(n_p[1], 4) * pow(n_p[0], 2) - 20 * pow(n_p[1], 2) * pow(n_p[0], 4))/(pow((1.0 - SQ(n_p[2])),4) + 1e-20) - 4 * epsilon[2] *n_p[1]* SQ(n_p[2]);
	//  	NN[2] =  4 * epsilon[2] * n_p[2] * (1.0 - SQ(n_p[2]));
	///---------------------------------------------------------------------------------------------------
	///********************************************************
	///  Case 2 (2)
	///  As = 1 + ep_xy * cos(6 * theta) + ep_z * cos(2 * phi)
	///  for n_x^2 + n_y^2 = 1
	///*******************************************************
	//    const double sinphi = (sqrt(2.)/2.0) * (sqrt(1.0 - SQ(n_p[2])) + fabs(n_p[2]));(sinphi + 1e-20)

	//    a_s = 1. + epsilon[0] * axy + epsilon[2] * (2.0*SQ(n_p[2])-1.0);
	//  	NN[0] =  6 * epsilon[0] * n_p[0] * (6 * pow(n_p[1], 6) + 6 * pow(n_p[0], 4) * pow(n_p[1], 2) - 20 * pow(n_p[0], 2) * pow(n_p[1], 4))/(sqrt(1.0 - SQ(n_p[2])) + 1e-20) - 4 * epsilon[2] *n_p[0]* SQ(n_p[2])*sqrt(1.0 - SQ(n_p[2]));
	//	NN[1] = -6 * epsilon[0] * n_p[1] * (6 * pow(n_p[0], 6) + 6 * pow(n_p[1], 4) * pow(n_p[0], 2) - 20 * pow(n_p[1], 2) * pow(n_p[0], 4))/(sqrt(1.0 - SQ(n_p[2])) + 1e-20) - 4 * epsilon[2] *n_p[1]* SQ(n_p[2])*sqrt(1.0 - SQ(n_p[2]));
	//  	NN[2] =  4 * epsilon[2] * n_p[2] * (1.0 - SQ(n_p[2]));

	//    a_s = 1. + epsilon[0] * axy + epsilon[2] * (2.0*SQ(n_p[2])-1.0);
	//  	NN[0] =  6 * epsilon[0] * (6 * pow(n_p[1], 5) * pow(n_p[0], 1) + 6 * pow(n_p[0], 5) * pow(n_p[1], 1) - 20 * pow(n_p[0], 3) * pow(n_p[1], 3)) - 4 * epsilon[2] *n_p[0]* SQ(n_p[2])*sqrt(1.0 - SQ(n_p[2]));
	//	NN[1] = -6 * epsilon[0] * (6 * pow(n_p[0], 6) + 6 * pow(n_p[1], 4) * pow(n_p[0], 2) - 20 * pow(n_p[1], 2) * pow(n_p[0], 4)) - 4 * epsilon[2] *n_p[1]* SQ(n_p[2])*sqrt(1.0 - SQ(n_p[2]));
	//  	NN[2] =  4 * epsilon[2] * n_p[2] * (1.0 - SQ(n_p[2]));
	///---------------------------------------------------------------------------------------------------

	///******************************************************************************************************************************
	///  Case 2(3) Regularize
	///  As = 1 + ep_xy * cos(6 * theta) + ep_z * cos(2 * phi)
	///  for n_x^2 + n_y^2 = 1
	///***************************************************************************************************************************
	//    aa = 1.0 - (1 + epsilon[2]*(2.*SQ(n_p[2]) -1.))/epsilon[0];
	//    nnx = sqrt(21./(400.* pow((sqrt(SQ(aa/320. - 81./8000.) - 9261./64000000.) - aa/320. + 81./8000), 1./3)) + pow((sqrt(SQ(aa/320. - 81./8000) - 9261./64000000.) - aa/320. + 81./8000.),1.0/3) + 3./10);
	//	///threshold for xy-axis angle   *****cos(theta_m)****
	//	nny = sqrt(1. - SQ(nnx));  ///threshold for xy-axis angle   *****sin(theta_m)****
	//	cos_6theta_m = pow(nnx,6) - 15 * pow(nnx,4) * pow(nny,2) + 15 * pow(nnx,2) * pow(nny,4) - pow(nny,6); ///cos(6*theta_m)
	//	B1 = 12. * epsilon[0] * (3. * pow(nnx, 5) + 3. * nnx * pow(nny,4)  - 10. * pow(nnx,3) * pow(nny,2));  ///B_1    6*epsilon_xy*sin(6*theta_m)/sin(theta_m)
	//    A1 = 1.0 + epsilon[0] * cos_6theta_m + epsilon[2] * (2. * SQ(n_p[2]) - 1.0) - B1 * nnx;
	//    grad_A1 = - 4.0* epsilon[2] * sqrt(1.0 - SQ(n_p[2])) * n_p[2];
	//        if ((n_p[0]>=nnx))
	//        {
	//        a_s = A1 + B1 * n_p[0];
	//        NN[0] =  B1 * n_p[1] * n_p[1]/sqrt(1.0 - SQ(n_p[2])) + grad_A1 * n_p[2] * n_p[0];
	//        NN[1] = -B1 * n_p[1] * n_p[0]/sqrt(1.0 - SQ(n_p[2])) + grad_A1 * n_p[2] * n_p[1];
	//        NN[2] = -grad_A1 * sqrt(1.0 - SQ(n_p[2]));
	//        }
	//        else if ((n_p[0]>= (0.5 * nnx - sqrt(3.)/2. * nny)) && (n_p[0]<= (0.5 * nnx + sqrt(3.)/2. * nny)) && (n_p[1]> 0))
	//        {
	//        a_s = A1 + B1 * (0.5 * n_p[0] + sqrt(3.)/2. * n_p[1]);
	//        NN[0] =  B1 * (0.5 * n_p[1] - sqrt(3.)/2. * n_p[0]) * n_p[1]/sqrt(1.0 - SQ(n_p[2])) + grad_A1 * n_p[2] * n_p[0];
	//        NN[1] = -B1 * (0.5 * n_p[1] - sqrt(3.)/2. * n_p[0]) * n_p[0]/sqrt(1.0 - SQ(n_p[2])) + grad_A1 * n_p[2] * n_p[1];
	//        NN[2] = -grad_A1 * sqrt(1.0 - SQ(n_p[2]));
	//        }
	//        else if((n_p[0]>= (-0.5 * nnx - sqrt(3.)/2. * nny)) && (n_p[0]<= (-0.5 * nnx + sqrt(3.)/2. * nny))&& (n_p[1]>0))
	//        {
	//        a_s = A1 + B1 * (-0.5 * n_p[0] + sqrt(3.)/2. * n_p[1]);
	//        NN[0] = -B1 * (0.5 * n_p[1] + sqrt(3.)/2. * n_p[0]) * n_p[1]/sqrt(1.0 - SQ(n_p[2])) + grad_A1 * n_p[2] * n_p[0];
	//        NN[1] =  B1 * (0.5 * n_p[1] + sqrt(3.)/2. * n_p[0]) * n_p[0]/sqrt(1.0 - SQ(n_p[2])) + grad_A1 * n_p[2] * n_p[1];
	//        NN[2] = -grad_A1 * sqrt(1.0 - SQ(n_p[2]));
	//        }
	//        else if((n_p[0]<= -nnx))
	//        {
	//        a_s = A1 + B1 * (-n_p[0]);
	//        NN[0] = -B1 * n_p[1] * n_p[1]/sqrt(1.0 - SQ(n_p[2])) + grad_A1 * n_p[2] * n_p[0];
	//        NN[1] =  B1 * n_p[1] * n_p[0]/sqrt(1.0 - SQ(n_p[2])) + grad_A1 * n_p[2] * n_p[1];
	//        NN[2] = -grad_A1 * sqrt(1.0 - SQ(n_p[2]));
	//        }
	//        else if((n_p[0]>= (-0.5 * nnx - sqrt(3.)/2. * nny)) && (n_p[0]<= (-0.5 * nnx + sqrt(3.)/2. * nny))&& (n_p[1]<0))
	//        {
	//        a_s = A1 + B1 * (-0.5 * n_p[0] - sqrt(3.)/2. * n_p[1]);
	//        NN[0] = -B1 * (0.5 * n_p[1] - sqrt(3.)/2. * n_p[0]) * n_p[1]/sqrt(1.0 - SQ(n_p[2])) + grad_A1 * n_p[2] * n_p[0];
	//        NN[1] =  B1 * (0.5 * n_p[1] - sqrt(3.)/2. * n_p[0]) * n_p[0]/sqrt(1.0 - SQ(n_p[2])) + grad_A1 * n_p[2] * n_p[1];
	//        NN[2] = -grad_A1 * sqrt(1.0 - SQ(n_p[2]));
	//        }
	//        else if((n_p[0]>= (0.5 * nnx - sqrt(3.)/2. * nny)) && (n_p[0]<= (0.5 * nnx + sqrt(3.)/2. * nny))&& (n_p[1]<0))
	//        {
	//        a_s = A1 + B1 * (0.5 * n_p[0] - sqrt(3.)/2. * n_p[1]);
	//        NN[0] =  B1 * (0.5 * n_p[1] + sqrt(3.)/2. * n_p[0]) * n_p[1]/sqrt(1.0 - SQ(n_p[2])) + grad_A1 * n_p[2] * n_p[0];
	//        NN[1] = -B1 * (0.5 * n_p[1] + sqrt(3.)/2. * n_p[0]) * n_p[0]/sqrt(1.0 - SQ(n_p[2])) + grad_A1 * n_p[2] * n_p[1];
	//        NN[2] = -grad_A1 * sqrt(1.0 - SQ(n_p[2]));
	//       }
	//        else{
	//        a_s = 1. + epsilon[0] * axy + epsilon[2] * (2.0*SQ(n_p[2])-1.0);
	//        NN[0] =  6 * epsilon[0] * n_p[0] * (6 * pow(n_p[1], 6) + 6 * pow(n_p[0], 4) * pow(n_p[1], 2) - 20 * pow(n_p[0], 2) * pow(n_p[1], 4))/(sqrt(1.0 - SQ(n_p[2])) + 1e-20) - 4 * epsilon[2] *n_p[0]* SQ(n_p[2])*sqrt(1.0 - SQ(n_p[2]));
	//        NN[1] = -6 * epsilon[0] * n_p[1] * (6 * pow(n_p[0], 6) + 6 * pow(n_p[1], 4) * pow(n_p[0], 2) - 20 * pow(n_p[1], 2) * pow(n_p[0], 4))/(sqrt(1.0 - SQ(n_p[2])) + 1e-20) - 4 * epsilon[2] *n_p[1]* SQ(n_p[2])*sqrt(1.0 - SQ(n_p[2]));
	//        NN[2] =  4 * epsilon[2] * n_p[2] * (1.0 - SQ(n_p[2]));
	//     }
	///------------------------------- End of regular mit no sin  -------------------------------------///
	///********************************************************
	///  Case 3(1)
	///  As = 1 + ep_xy * sin(phi) * cos(6 * theta) + ep_z * cos(2 * phi)
	///  for n_x^2 + n_y^2 = 1
	/////*******************************************************
	a_s = 1. + epsilon[0] * axy * sqrt(1.0 - SQ(n_p[2])) + epsilon[2] * (2.0 * SQ(n_p[2]) - 1.0);
	NN[0] = epsilon[0] * n_p[0] * (36. * pow(n_p[1], 6) + 36. * pow(n_p[0], 4) * pow(n_p[1], 2) - 120. * pow(n_p[0], 2) * pow(n_p[1], 4) + axy * SQ(n_p[2])) - 4.0 * epsilon[2] * sqrt(1.0 - SQ(n_p[2])) * n_p[0] * SQ(n_p[2]);
	NN[1] = -epsilon[0] * n_p[1] * (36. * pow(n_p[0], 6) + 36. * pow(n_p[1], 4) * pow(n_p[0], 2) - 120. * pow(n_p[1], 2) * pow(n_p[0], 4) - axy * SQ(n_p[2])) - 4.0 * epsilon[2] * sqrt(1.0 - SQ(n_p[2])) * n_p[1] * SQ(n_p[2]);
	NN[2] = 4 * epsilon[2] * n_p[2] * (1.0 - SQ(n_p[2])) - epsilon[0] * n_p[2] * axy * sqrt(1.0 - SQ(n_p[2]));
	///---------------------------------------------------------------------------------------------------

	///******************************************************************************************************************************
	/// Case 3(2) Regularize
	///  As = 1 + ep_xy * sin(phi) * cos(6 * theta) + ep_z * cos(2 * phi)
	///  for n_x^2 + n_y^2 = 1
	///***************************************************************************************************************************
	//    aa = 1.0 - (1 + epsilon[2]*(2.*SQ(n_p[2]) -1.))/(epsilon[0]*sqrt(1.0 - SQ(n_p[2])));
	//    nnx = sqrt(21./(400.* pow((sqrt(SQ(aa/320. - 81./8000.) - 9261./64000000.) - aa/320. + 81./8000), 1./3)) + pow((sqrt(SQ(aa/320. - 81./8000) - 9261./64000000.) - aa/320. + 81./8000.),1.0/3) + 3./10);
	//	///threshold for xy-axis angle   *****cos(theta_m)****
	//	nny = sqrt(1. - SQ(nnx));  ///threshold for xy-axis angle   *****sin(theta_m)****
	//	cos_6theta_m = pow(nnx,6) - 15 * pow(nnx,4) * pow(nny,2) + 15 * pow(nnx,2) * pow(nny,4) - pow(nny,6); ///cos(6*theta_m)
	//	B1 = 12. * epsilon[0] * sqrt(1.0 - SQ(n_p[2])) * (3. * pow(nnx, 5) + 3. * nnx * pow(nny,4)  - 10. * pow(nnx,3) * pow(nny,2));  ///B_1    6*epsilon_xy*sin(6*theta_m)/sin(theta_m)
	//    A1 = 1.0 + epsilon[0] * sqrt(1.0 - SQ(n_p[2])) * cos_6theta_m + epsilon[2] * (2. * SQ(n_p[2]) - 1.0) - B1 * nnx;
	//    grad_B1 = 12. * epsilon[0] * n_p[2]* (3. * pow(nnx, 5) + 3. * nnx * pow(nny,4)  - 10. * pow(nnx,3) * pow(nny,2));
	//    grad_A1 = epsilon[0] * n_p[2]* cos_6theta_m - 4.0* epsilon[2] * sqrt(1.0 - SQ(n_p[2])) * n_p[2] - grad_B1 * nnx;
	//        if ((n_p[0]>=nnx))
	//        {
	//        a_s = A1 + B1 * n_p[0];
	//        NN[0] =  B1 * n_p[1] * n_p[1]/sqrt(1.0 - SQ(n_p[2])) + (grad_A1 + n_p[0] * grad_B1) * n_p[2] * n_p[0];
	//        NN[1] = -B1 * n_p[1] * n_p[0]/sqrt(1.0 - SQ(n_p[2])) + (grad_A1 + n_p[0] * grad_B1) * n_p[2] * n_p[1];
	//        NN[2] = -(grad_A1 + n_p[0] * grad_B1) * sqrt(1.0 - SQ(n_p[2]));
	//        }
	//        else if ((n_p[0]>= (0.5 * nnx - sqrt(3.)/2. * nny)) && (n_p[0]<= (0.5 * nnx + sqrt(3.)/2. * nny)) && (n_p[1]> 0))
	//        {
	//        a_s = A1 + B1 * (0.5 * n_p[0] + sqrt(3.)/2. * n_p[1]);
	//        NN[0] =  B1 * (0.5 * n_p[1] - sqrt(3.)/2. * n_p[0]) * n_p[1]/sqrt(1.0 - SQ(n_p[2])) + (grad_A1 + (0.5 * n_p[0] + sqrt(3.)/2. * n_p[1]) * grad_B1) * n_p[2] * n_p[0];
	//        NN[1] = -B1 * (0.5 * n_p[1] - sqrt(3.)/2. * n_p[0]) * n_p[0]/sqrt(1.0 - SQ(n_p[2])) + (grad_A1 + (0.5 * n_p[0] + sqrt(3.)/2. * n_p[1]) * grad_B1) * n_p[2] * n_p[1];
	//        NN[2] = -(grad_A1 + (0.5 * n_p[0] + sqrt(3.)/2. * n_p[1]) * grad_B1) * sqrt(1.0 - SQ(n_p[2]));
	//        }
	//        else if((n_p[0]>= (-0.5 * nnx - sqrt(3.)/2. * nny)) && (n_p[0]<= (-0.5 * nnx + sqrt(3.)/2. * nny))&& (n_p[1]>0))
	//        {
	//        a_s = A1 + B1 * (-0.5 * n_p[0] + sqrt(3.)/2. * n_p[1]);
	//        NN[0] = -B1 * (0.5 * n_p[1] + sqrt(3.)/2. * n_p[0]) * n_p[1]/sqrt(1.0 - SQ(n_p[2])) + (grad_A1 + (-0.5 * n_p[0] + sqrt(3.)/2. * n_p[1]) * grad_B1) * n_p[2] * n_p[0];
	//        NN[1] =  B1 * (0.5 * n_p[1] + sqrt(3.)/2. * n_p[0]) * n_p[0]/sqrt(1.0 - SQ(n_p[2])) + (grad_A1 + (-0.5 * n_p[0] + sqrt(3.)/2. * n_p[1]) * grad_B1) * n_p[2] * n_p[1];
	//        NN[2] = -(grad_A1 + (-0.5 * n_p[0] + sqrt(3.)/2. * n_p[1]) * grad_B1) * sqrt(1.0 - SQ(n_p[2]));
	//        }
	//        else if((n_p[0]<= -nnx))
	//        {
	//        a_s = A1 + B1 * (-n_p[0]);
	//        NN[0] = -B1 * n_p[1] * n_p[1]/sqrt(1.0 - SQ(n_p[2])) + (grad_A1 + (-n_p[0]) * grad_B1) * n_p[2] * n_p[0];
	//        NN[1] =  B1 * n_p[1] * n_p[0]/sqrt(1.0 - SQ(n_p[2])) + (grad_A1 + (-n_p[0]) * grad_B1) * n_p[2] * n_p[1];
	//        NN[2] = -(grad_A1 + (-n_p[0]) * grad_B1) * sqrt(1.0 - SQ(n_p[2]));
	//        }
	//        else if((n_p[0]>= (-0.5 * nnx - sqrt(3.)/2. * nny)) && (n_p[0]<= (-0.5 * nnx + sqrt(3.)/2. * nny))&& (n_p[1]<0))
	//        {
	//        a_s = A1 + B1 * (-0.5 * n_p[0] - sqrt(3.)/2. * n_p[1]);
	//        NN[0] = -B1 * (0.5 * n_p[1] - sqrt(3.)/2. * n_p[0]) * n_p[1]/sqrt(1.0 - SQ(n_p[2])) + (grad_A1 + (-0.5 * n_p[0] - sqrt(3.)/2. * n_p[1]) * grad_B1) * n_p[2] * n_p[0];
	//        NN[1] =  B1 * (0.5 * n_p[1] - sqrt(3.)/2. * n_p[0]) * n_p[0]/sqrt(1.0 - SQ(n_p[2])) + (grad_A1 + (-0.5 * n_p[0] - sqrt(3.)/2. * n_p[1]) * grad_B1) * n_p[2] * n_p[1];
	//        NN[2] = -(grad_A1 + (-0.5 * n_p[0] - sqrt(3.)/2. * n_p[1]) * grad_B1) * sqrt(1.0 - SQ(n_p[2]));
	//        }
	//        else if((n_p[0]>= (0.5 * nnx - sqrt(3.)/2. * nny)) && (n_p[0]<= (0.5 * nnx + sqrt(3.)/2. * nny))&& (n_p[1]<0))
	//        {
	//        a_s = A1 + B1 * (0.5 * n_p[0] - sqrt(3.)/2. * n_p[1]);
	//        NN[0] =  B1 * (0.5 * n_p[1] + sqrt(3.)/2. * n_p[0]) * n_p[1]/sqrt(1.0 - SQ(n_p[2])) + (grad_A1 + (0.5 * n_p[0] - sqrt(3.)/2. * n_p[1]) * grad_B1) * n_p[2] * n_p[0];
	//        NN[1] = -B1 * (0.5 * n_p[1] + sqrt(3.)/2. * n_p[0]) * n_p[0]/sqrt(1.0 - SQ(n_p[2])) + (grad_A1 + (0.5 * n_p[0] - sqrt(3.)/2. * n_p[1]) * grad_B1) * n_p[2] * n_p[1];
	//        NN[2] = -(grad_A1 + (0.5 * n_p[0] - sqrt(3.)/2. * n_p[1]) * grad_B1) * sqrt(1.0 - SQ(n_p[2]));
	//       }
	//        else{
	//        a_s = 1. + epsilon[0] * axy * sqrt(1.0 - SQ(n_p[2])) + epsilon[2] * (2.0*SQ(n_p[2])-1.0);
	//        NN[0] =  epsilon[0] * n_p[0] * (36. * pow(n_p[1], 6) + 36. * pow(n_p[0], 4) * pow(n_p[1], 2) - 120. * pow(n_p[0], 2) * pow(n_p[1], 4) + axy * SQ(n_p[2])) - 4.0 * epsilon[2] * sqrt(1.0 - SQ(n_p[2])) * n_p[0]* SQ(n_p[2]);
	//        NN[1] = -epsilon[0] * n_p[1] * (36. * pow(n_p[0], 6) + 36. * pow(n_p[1], 4) * pow(n_p[0], 2) - 120. * pow(n_p[1], 2) * pow(n_p[0], 4) - axy * SQ(n_p[2])) - 4.0 * epsilon[2] * sqrt(1.0 - SQ(n_p[2])) * n_p[1]* SQ(n_p[2]);
	//        NN[2] =  4 * epsilon[2] * n_p[2] * (1.0 - SQ(n_p[2])) - epsilon[0] * n_p[2] * axy * sqrt(1.0 - SQ(n_p[2]));
	//        }
	///------------------------------- End of regular mit one sin  -------------------------------------///
	///********************************************************
	/// Case 4(1)
	/// As = 1 + ep_xy * cos(6 * theta)
	/// For n_x^2 + n_y^2 = 1 not equal to original equation
	///*******************************************************
	//    a_s = 1. + epsilon[0] * axy;
	//  	NN[0] =  epsilon[0] * n_p[0] * (36. * pow(n_p[1], 6) + 36. * pow(n_p[0], 4) * pow(n_p[1], 2) - 120. * pow(n_p[0], 2) * pow(n_p[1], 4));
	//	NN[1] = -epsilon[0] * n_p[1] * (36. * pow(n_p[0], 6) + 36. * pow(n_p[1], 4) * pow(n_p[0], 2) - 120. * pow(n_p[1], 2) * pow(n_p[0], 4));
	//  	NN[2] =  0.0;
	//    a_s = 1. + epsilon[0] * axy + epsilon[2] * (2.0*SQ(n_p[2])-1.0);
	//  	NN[0] =  6.0 * epsilon[0] * n_p[0] * (pow(n_p[0], 4) - 10 * SQ(n_p[0]) * SQ(n_p[1]) + 5 * pow(n_p[0], 4));
	//	NN[1] = -6.0 * epsilon[0] * n_p[1] * (pow(n_p[0], 4) - 10 * SQ(n_p[0]) * SQ(n_p[1]) + 5 * pow(n_p[0], 4));
	//  	NN[2] =  0.0;
	///---------------------------------------------------------------------------------------------------
	///  Case 5 (1)
	///  As = 1 + ep_xy * cos(6 * theta) + ep_z * cos(2 * phi)
	///  For n_x^2 + n_y^2 = 1 not equal to original equation
	///*******************************************************
	//    a_s = 1. + epsilon[0] * axy + epsilon[2] * (2.0*SQ(n_p[2])-1.0);
	//    NN[0] =  6 * epsilon[0] * n_p[1] * sin6theta;
	//	NN[1] = -6 * epsilon[0] * n_p[0] * sin6theta;
	//  	NN[2] =  4 * epsilon[2] * n_p[2] * (1.0 - SQ(n_p[2]));
	///---------------------------------------------------------------------------------------------------
	///  Case 6 (1)
	///  As = 1 + sin(phi) * ep_xy * cos(6 * theta) + ep_z * cos(2 * phi)
	///  For n_x^2 + n_y^2 = 1 gamma = 1 then scale equal to original
	///*******************************************************
	//    a_s = 1. + epsilon[0] * axy * sqrt(1.0 - SQ(n_p[2])) + epsilon[2] * (2.0*SQ(n_p[2])-1.0);
	//  	NN[0] =  epsilon[0] * n_p[0] * (36. * pow(n_p[1], 6) + 36. * pow(n_p[0], 4) * pow(n_p[1], 2) - 120. * pow(n_p[0], 2) * pow(n_p[1], 4) + axy * SQ(n_p[2])) - 4.0 * epsilon[2] * sqrt(1.0 - SQ(n_p[2])) * n_p[0]* SQ(n_p[2]);
	//	NN[1] = -epsilon[0] * n_p[1] * (36. * pow(n_p[0], 6) + 36. * pow(n_p[1], 4) * pow(n_p[0], 2) - 120. * pow(n_p[1], 2) * pow(n_p[0], 4) - axy * SQ(n_p[2])) - 4.0 * epsilon[2] * sqrt(1.0 - SQ(n_p[2])) * n_p[1]* SQ(n_p[2]);
	//  	NN[2] =  4 * epsilon[2] * n_p[2] * (1.0 - SQ(n_p[2])) - epsilon[0] * n_p[2] * axy * sqrt(1.0 - SQ(n_p[2]));
	///  Case 7 (1)
	///  As = 1 + sin(phi) * ep_xy * cos(6 * theta) + ep_z * cos(2 * phi)
	///  For n_x^2 + n_y^2 = 1
	///*******************************************************
	//    a_s = 1. + epsilon[0] * axy * sqrt(1.0 - SQ(n_p[2])) + epsilon[2] * (2.0*SQ(n_p[2])-1.0);
	//  	NN[0] =  6.0 * epsilon[0] * sin6theta * n_p[1] + epsilon[0] * axy * SQ(n_p[2]) * n_p[0] - 4.0 * epsilon[2] * sqrt(1.0 - SQ(n_p[2])) * SQ(n_p[2]) * n_p[0];
	//	NN[1] = -6.0 * epsilon[0] * sin6theta * n_p[0] + epsilon[0] * axy * SQ(n_p[2]) * n_p[1] - 4.0 * epsilon[2] * sqrt(1.0 - SQ(n_p[2])) * SQ(n_p[2]) * n_p[1];
	//  	NN[2] = -epsilon[0] * n_p[2] * axy * sqrt(1.0 - SQ(n_p[2])) + 4 * epsilon[2] * n_p[2] * (1.0 - SQ(n_p[2])) ;
}
void twodendrite(double* n_p, double* NN, double& a_s, double* epsilon) {  /// new phase-field added
	/// Obviously setting epsilon to 0, results in Isotropic growth
	a_s = 1. + epsilon[0] * (SQ(n_p[0]) - SQ(n_p[1]));
	NN[0] = 4. * epsilon[0] * n_p[0] * SQ(n_p[1]);
	NN[1] = -4. * epsilon[0] * n_p[1] * SQ(n_p[0]);
	NN[2] = 0.;
}

void Isotropic(double* n, double* NN, double& a_s, double& epsilon) {  /// new phase-field added
	/// Obviously setting epsilon to 0, results in Isotropic growth
	a_s = 1.;
	NN[0] = 0.;
	NN[1] = 0.;
	NN[2] = 0.;
}
