// #include "stdafx.h"
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
#include "ALBORZ_Macros.h"
#include "ALBORZ_SETTINGS.h"
#include "ALBORZ_GlobalVariables.h"

#include "Flow_solver.h"
#include "Thermal_solver.h"
#include "Geometry.h"
#include "Phase_Field.h"
#include "CANTERA_INTERFACE.h"

/* These headers contain cantera functions */
#define CT_SKIP_DEPRECATION_WARNINGS
#include "cantera/thermo.h"
#include "cantera/transport.h"
#include "cantera/kinetics/GasKinetics.h"
#include "cantera/thermo/Phase.h"
#include "cantera/base/Solution.h"
#include "utils/Config_utils.h"

using namespace Cantera;

#if defined IMPLEX
#include "Radau/radau.h"
#endif
#include "io/IO_interface.h"
using namespace std;

std::vector<std::vector<double>> Ini_s;
std::vector<std::vector<double>> Ini_D;

#if defined IMPLEX && !defined REGATH_LIB
temp_Data_1 ImSolver1;
///_________________________________________________________________________________________________///
///                                  RADAU IMPLICIT SOLVER INITIALIZATION                           ///
///_________________________________________________________________________________________________///
double Imsolver_t, Imsolver_t_end, Imsolver_h;
int npp;  // This is dimension of the implicit problem = (Nb_spec+1)
double *y, *f, *work;
int* iwork;
int ipar = 0;
int lwork, liwork, idid;

// output routine is not used during integration*/
int iout = 0;
double rpar = 0;
//-------------------------------------------------------------------------------
// Tolerances
//-------------------------------------------------------------------------------
int itoler = 0;          // rtol and atol are scalars
double rtoler = 1.0e-6;  // relative tolerance
double atoler = 1.0e-6;  // absolute tolerance
//-------------------------------------------------------------------------------
// Jacobian matrix
//-------------------------------------------------------------------------------
int ijac = 1;  // analytical Jacobian function provided
int mljac;     // number of non-zero rows below main diagonal of Jacobian
int mujac;     // number of non-zero rows above main diagonal of Jacobian
//-------------------------------------------------------------------------------
// Mass matrix
//-------------------------------------------------------------------------------
int imas = 0;   // Mass matrix routine is identity
int mlmas = 0;  // number of non-zero rows below main diagonal of mass matrix
int mumas = 0;  // number of non-zero rows above main diagonal of mass matrix
//-------------------------------------------------------
//-------------------------------------------------------
#endif
Species_solver::Species_solver() {
}
/// ***************************************************** ///
/// READ IN GENERAL FLOW PARAMETERS, I.E. STENCIL AND     ///
/// AND SPECIES, ETC                                      ///
/// ***************************************************** ///
void Species_solver::General_data_input(const std::string& filename, Parallel_MPI* MPI_parallel) {
	int column_width = 40;
	/// Open input file
	string input_filename(filename);
	input_filename += ".dat";
	ifstream input_file;  // File is open for READING
	input_file.open(input_filename.c_str(), ios::binary);

	input_file.clear();
	input_file.seekg(0, ios::beg);
	find_line_after_header(input_file, "c\tSpecies Field Solver");
	find_line_after_comment(input_file);
	input_file >> Dimension >> Discrete_Velocity >> Nb_spec >> molar_mass_av_0;
	species_name_RG.resize(Nb_spec);
	Molar_mass.resize(Nb_spec);
	Le.resize(Nb_spec);
	D_k.resize(Nb_spec);
	for (unsigned int k = 0; k < Nb_spec; ++k) {
		find_line_after_comment(input_file);
		input_file >> species_name_RG[k];
		input_file >> Molar_mass[k];
#if defined ConstLewis_Species_diffusion
		input_file >> Le[k];
		// input_file >> D_k[k];
#endif
		/*#if defined Diffusion_model
		    // here we read the diffusion model for each Species, if it is not defined, we assume it is constant_Dk
		    // the different models are constant_Dk = constant diffusion coefficient, constant_Le = constant Lewis number, constant_Sc = constant Schmidt number
		    find_line_after_header(input_file, "c\tSpecies Diffusion Model");
		    input_file >> diffusion_model;
		    std::cout << "Diffusion Model: " << diffusion_model << std::endl;
		    for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
		        for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
		            for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
		                if (solid_species[{X, Y, Z}] == FALSE) {
		                    for (k = 0; k < Nb_spec; ++k) {
		                        if (diffusion_model == "constant_Dk") {
		                            D_k[k] = D_k[k];
		                            // poin2diff = &const_D;
		                            for (unsigned int k = 0; k < Nb_spec; ++k) {
		                                D_k[k] *= global_parameters.D_t / pow(global_parameters.D_x, 2);
		                            }
		                        }
		                        if (diffusion_model == "constant_Le")
		                            D_k[k] = Thermal->diffusion_coefficient[{X, Y, Z}] / (Thermal->c_p[{X, Y, Z}] * Le[k]);
		                        // poin2diff = &const_Le;
		                        if (diffusion_model == "constant_Sc")
		                            D_k[k] = Thermal->diffusion_coefficient[{X, Y, Z}] / (Thermal->c_p[{X, Y, Z}] * Sc[k]);
		                        // poin2diff = &const_Sc;
		                    }
		                }
		            }
		        }
		    }
		#endif
		*/
	}
	input_file.close();

#if defined IMPLEX && !defined REGATH_LIB
	ImSolver1.Nb_spec = Nb_spec;
	ImSolver1.Y_k = new double[ImSolver1.Nb_spec];
	ImSolver1.M = new double[ImSolver1.Nb_spec];
	ImSolver1.C = new double[ImSolver1.Nb_spec];
	ImSolver1.w_k = new double[ImSolver1.Nb_spec];
	for (unsigned int k = 0; k < ImSolver1.Nb_spec; ++k) {
		ImSolver1.M[k] = Molar_mass[k];
	}
	npp = Nb_spec + 1;
	y = new double[npp];
	f = new double[npp];
	mljac = npp;
	mujac = npp;
	lwork = (1 + 8) * npp * npp + 3 * (7 + 1) * npp + 20;
	liwork = (2 + (7 - 1) / 2) * npp + 20;
	work = new double[lwork];
	iwork = new int[liwork];
	for (int i = 0; i < lwork; i++) {
		work[i] = 0.0;
	}
	for (int i = 0; i < liwork; i++) {
		iwork[i] = 0;
	}
#endif  // defined
	if (MPI_parallel->processor_id == MASTER + 1) {
		std::cout << "Species field parameters" << endl;
		std::cout << "=====================" << endl;
		std::cout << "Stencil : " << left << "D" << Dimension << "Q" << Discrete_Velocity << endl;
		std::cout << setw(column_width) << left << "Number of Species : " << Nb_spec << endl;
		std::cout << setw(column_width) << left << "Reference molar mass : " << molar_mass_av_0 << endl;
		for (unsigned int k = 0; k < Nb_spec; ++k) {
			std::cout << species_name_RG[k] << "\t";
			std::cout << Molar_mass[k] << "\t";
#if defined ConstLewis_Species_diffusion
			std::cout << Le[k] << "\n";
#endif
			if ((k + 1) % 6 == 0) std::cout << "\n";
		}
		std::cout << endl;
	}
}
/// ***************************************************** ///
/// READ REACTIONS FROM INPUT FILE AND INITIALIZE         ///
/// ***************************************************** ///
void Species_solver::initialize_reactions(const std::string& filename, Parallel_MPI* MPI_parallel) {
	/// Open input file
	string input_filename(filename);
	input_filename += ".dat";
	ifstream input_file;  // File is open for READING
	input_file.open(input_filename.c_str(), ios::binary);

	input_file.clear();
	input_file.seekg(0, ios::beg);
	find_line_after_header(input_file, "c\tReactions");
	find_line_after_comment(input_file);
	input_file >> Nb_reac;

	Stoechio_coeff_fi.resize(Nb_reac);
	Stoechio_coeff_ri.resize(Nb_reac);
	Reac_order_fi.resize(Nb_reac);
	Reac_order_ri.resize(Nb_reac);
	Reac_coeff.resize(Nb_reac);
	Reac_type.resize(Nb_reac);
	for (unsigned int i = 0; i < Nb_reac; ++i) {
		Stoechio_coeff_fi[i].resize(Nb_spec);
		Stoechio_coeff_ri[i].resize(Nb_spec);

		Reac_order_fi[i].resize(Nb_spec);
		Reac_order_ri[i].resize(Nb_spec);
		Reac_coeff[i].resize(5);
		find_line_after_comment(input_file);
		input_file >> Reac_type[i];
		find_line_after_comment(input_file);
		for (unsigned int k = 0; k < Nb_spec; ++k) {
			input_file >> Stoechio_coeff_fi[i][k];
		}
		find_line_after_comment(input_file);
		for (unsigned int k = 0; k < Nb_spec; ++k) {
			input_file >> Stoechio_coeff_ri[i][k];
		}
		find_line_after_comment(input_file);
		for (unsigned int k = 0; k < Nb_spec; ++k) {
			input_file >> Reac_order_fi[i][k];
		}
		find_line_after_comment(input_file);
		for (unsigned int k = 0; k < Nb_spec; ++k) {
			input_file >> Reac_order_ri[i][k];
		}
		find_line_after_comment(input_file);
		for (int k = 0; k < 5; ++k) {
			input_file >> Reac_coeff[i][k];
		}
	}
#if defined IMPLEX && !defined REGATH_LIB
	ImSolver1.Nb_reac = Nb_reac;
	ImSolver1.Nb_spec = Nb_spec;
	ImSolver1.Stoechio_coeff.resize(ImSolver1.Nb_reac);
	ImSolver1.Reac_order.resize(ImSolver1.Nb_reac);
	ImSolver1.Reac_coeff.resize(ImSolver1.Nb_reac);
	ImSolver1.Reac_type.resize(ImSolver1.Nb_reac);
	for (unsigned int i = 0; i < ImSolver1.Nb_reac; ++i) {
		ImSolver1.Stoechio_coeff[i].resize(ImSolver1.Nb_spec);
		ImSolver1.Reac_order[i].resize(ImSolver1.Nb_spec);
		ImSolver1.Reac_coeff[i].resize(5);
		ImSolver1.Reac_type[i] = Reac_type[i];
		for (unsigned int k = 0; k < ImSolver1.Nb_spec; ++k) {
			ImSolver1.Stoechio_coeff[i][k] = Stoechio_coeff[i][k];
		}
		for (unsigned int k = 0; k < ImSolver1.Nb_spec; ++k) {
			ImSolver1.Reac_order[i][k] = Reac_order[i][k];
		}
		for (int k = 0; k < 5; ++k) {
			ImSolver1.Reac_coeff[i][k] = Reac_coeff[i][k];
		}
	}
#endif  // defined
	/// ===================================================
	///       Output Reactions
	/// ===================================================
	if (MPI_parallel->processor_id == MASTER + 1) {
		stringstream reaction_output_filename;
		reaction_output_filename << "Chemical_Scheme.dat";
		ofstream reaction_output;
		reaction_output.open(reaction_output_filename.str().c_str(), fstream::trunc);
		reaction_output << "CHEMICAL SCHEME PARAMETERS" << endl;
		reaction_output << " SOLVER TYPE : ";
#if !defined IMPLEX
		reaction_output << "EULER 1ST ORDER" << endl;
#endif
#if defined IMPLEX
		reaction_output << "RADAU SOLVER" << endl;
#endif  // defined
		reaction_output << "=========================================" << endl;
		reaction_output << "NUMBER OF REACTIONS :\t" << Nb_reac << "\n";
		reaction_output << "=========================================" << endl;
		int pluscounter;
		for (int i = 0; i < Nb_reac; i++) {
			reaction_output << "REACTION INDEX :\t" << i + 1 << "\t(out of " << Nb_reac << ")\n";
			reaction_output << "REACTION TYPE :\t" << Reac_type[i] << "\n";
			/// WRITE OUT REACTION EQUATION
			reaction_output << "REACTION :\n";
			pluscounter = 0;
			for (unsigned int k = 0; k < Nb_spec; k++) {
				if (Stoechio_coeff_fi[i][k] > 0) {
					if (pluscounter > 0) {
						reaction_output << " + ";
					}
					if (Stoechio_coeff_fi[i][k] > 0) {
						reaction_output << Stoechio_coeff_fi[i][k];
					}
					reaction_output << species_name_RG[k];
					pluscounter = pluscounter + 1;
				}
				//                    /// THIS IS FOR THE ENZYME IN MICHAELIS-MENTEN AND ARRHENIUS TYPE REACTIONS
				//                    if (Reac_order[i][k]<0 && Stoechio_coeff_fi[i][k] == Stoechio_coeff_ri[i][k]){
				//                        if (pluscounter>0){
				//                            reaction_output << " + ";
				//                            }
				//                        reaction_output << "(" << species_name_RG[k] << ")";
				//                        pluscounter = pluscounter+1;
				//                        }
			}
			reaction_output << " -----------> ";
			pluscounter = 0;
			for (unsigned int k = 0; k < Nb_spec; k++) {
				if (Stoechio_coeff_ri[i][k] > 0) {
					if (pluscounter > 0) {
						reaction_output << " + ";
					}
					if (Stoechio_coeff_ri[i][k] > 0) {
						reaction_output << Stoechio_coeff_ri[i][k];
					}
					reaction_output << species_name_RG[k];
					pluscounter = pluscounter + 1;
				}
				//                    /// THIS IS FOR THE ENZYME IN MICHAELIS-MENTEN AND ARRHENIUS TYPE REACTIONS
				//                    if (Reac_order[i][k]<0 && Stoechio_coeff_fi[i][k] == Stoechio_coeff_ri[i][k]){
				//                        if (pluscounter>0){
				//                            reaction_output << " + ";
				//                            }
				//                        reaction_output << "(" << species_name_RG[k] << ")";
				//                        pluscounter = pluscounter+1;
				//                        }
			}
			reaction_output << "\n";

			/// WRITE OUT REACTION RATE LAW
			reaction_output << "PRODCTION RATE :\n";
			/// WRITE OUT REACTION RATE LAW : ARRHENIUS-TYPE LAWS
			if (Reac_type[i] == "Arrhenius") {
				reaction_output << "d/dt = " << Reac_coeff[i][0] << " ";
				for (unsigned int k = 0; k < Nb_spec; k++) {
					if (Reac_order_fi[i][k] != 0 && Stoechio_coeff_fi[i][k] > 0) {
						reaction_output << "[" << species_name_RG[k] << "]^" << Reac_order_fi[i][k] << " ";
					}
				}
				reaction_output << "T^" << Reac_coeff[i][1] << " exp(-" << Reac_coeff[i][2] << "/RT) - ";
				reaction_output << Reac_coeff[i][4] << " ";
				for (unsigned int k = 0; k < Nb_spec; k++) {
					if (Reac_order_ri[i][k] != 0 && Stoechio_coeff_ri[i][k] > 0) {
						reaction_output << "[" << species_name_RG[k] << "]^" << fabs(Reac_order_ri[i][k]) << " ";
					}
				}
			}
			//                /// WRITE OUT REACTION RATE LAW : MICHAELIS-TYPE LAWS
			//                if (Reac_type[i] == "Michaelis") {
			//                    reaction_output << "d/dt = " << Reac_coeff[i][0] << " ";
			//                    for (unsigned int k=0; k < Nb_spec; k++){
			//                        if (Reac_order[i][k] == -1 && Stoechio_coeff_fi[i][k] == Stoechio_coeff_ri[i][k]) {
			//                            reaction_output << "[" << species_name_RG[k] << "]_0 ";
			//                            }
			//                        }
			//                    for (unsigned int k=0; k < Nb_spec; k++){
			//                        if (Reac_order[i][k] == 1) {
			//                            reaction_output << "[" << species_name_RG[k] << "] ";
			//                            }
			//                        }
			//                    reaction_output << "/(" << Reac_coeff[i][1] << " + ";
			//                    for (unsigned int k=0; k < Nb_spec; k++){
			//                        if (Reac_order[i][k] == 1) {
			//                            reaction_output << "[" << species_name_RG[k] << "] ";
			//                            }
			//                        }
			//                    reaction_output << " )";
			//                    }
			reaction_output << "\n----------------------------------------\n";
		}
		reaction_output.close();
	}
}
/// ***************************************************** ///
/// INITIALIZE STENCIL, CREATE OUTPUT FOLDER AND ALLOCATE ///
/// MEMORY FOR POPULATIONS AND PARAMETERS                 ///
/// ***************************************************** ///
void Species_solver::Memory_allocation(Stencil_Definition Stencil_Def, Parallel_MPI* MPI_parallel) {
	unsigned int X, Y, Z, alpha, k;
	/// -------------------------------------------------------------------------------------------
	Stencil_Def(Dimension, Discrete_Velocity, weight, c_alpha, alpha_bar, c_s2);
	weight_2.resize(Discrete_Velocity);
	weight_2[0] = 0;
	cp_k = new double[Nb_spec];
	for (alpha = 1; alpha < Discrete_Velocity; alpha++) {
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

	const Vector_field::Index_vec vec_sizes{x_size,
	                                        y_size,
	                                        z_size,
	                                        Nb_spec};

	molar_mass_av = Scalar_field::zeros(scalar_sizes);
	mass_fraction = Vector_field::zeros(vec_sizes);
	
	temp_T_eq = Scalar_field::zeros(scalar_sizes);
	temp_cp_eq = Scalar_field::zeros(scalar_sizes);
	temp_diffusion_coefficient_eq = Scalar_field::zeros(scalar_sizes);

	previous_mass_fraction = mass_fraction;
	V_c = Vector_field::zeros({x_size, y_size, z_size, 3});
	diffusion_coefficient = Vector_field::zeros(vec_sizes);
	Production = Vector_field::zeros(vec_sizes);
	solid_species = Solid_field::zeros(scalar_sizes);

	if (MPI_parallel->processor_id != MASTER) {
		Flux = new double****[MPI_parallel->dev_end[0]];
		for (X = 0; X < MPI_parallel->dev_end[0]; ++X) {
			Flux[X] = new double***[MPI_parallel->dev_end[1]];
			for (Y = 0; Y < MPI_parallel->dev_end[1]; ++Y) {
				Flux[X][Y] = new double**[MPI_parallel->dev_end[2]];
				for (Z = 0; Z < MPI_parallel->dev_end[2]; ++Z) {
					Flux[X][Y][Z] = new double*[Nb_spec];
					for (k = 0; k < Nb_spec; k++) {
						Flux[X][Y][Z][k] = new double[3];
					}
				}
			}
		}
		pop_s = new double****[Discrete_Velocity];
		pop_old_s = new double****[Discrete_Velocity];
		Buffer = new double****[Discrete_Velocity];
		pop_eq_s = new double[Discrete_Velocity];

		for (unsigned int c_i = 0; c_i < Discrete_Velocity; ++c_i) {
			pop_s[c_i] = new double***[MPI_parallel->dev_end[0]];
			pop_old_s[c_i] = new double***[MPI_parallel->dev_end[0]];
			Buffer[c_i] = new double***[MPI_parallel->dev_end[0]];
			for (X = 0; X < MPI_parallel->dev_end[0]; ++X) {
				pop_s[c_i][X] = new double**[MPI_parallel->dev_end[1]];
				pop_old_s[c_i][X] = new double**[MPI_parallel->dev_end[1]];
				Buffer[c_i][X] = new double**[MPI_parallel->dev_end[1]];
				for (Y = 0; Y < MPI_parallel->dev_end[1]; ++Y) {
					pop_s[c_i][X][Y] = new double*[MPI_parallel->dev_end[2]];
					pop_old_s[c_i][X][Y] = new double*[MPI_parallel->dev_end[2]];
					Buffer[c_i][X][Y] = new double*[MPI_parallel->dev_end[2]];
					for (Z = 0; Z < MPI_parallel->dev_end[2]; ++Z) {
						pop_s[c_i][X][Y][Z] = new double[Nb_spec];
						pop_old_s[c_i][X][Y][Z] = new double[Nb_spec];
						Buffer[c_i][X][Y][Z] = new double[Nb_spec];
						for (k = 0; k < Nb_spec; ++k) {
							pop_s[c_i][X][Y][Z][k] = 0;
							pop_old_s[c_i][X][Y][Z][k] = 0;
						}
					}
				}
			}
			// see Data_Exchange_Macroscopic
			//	macroscopic_group = Data_exchange_group(*MPI_parallel);
			//	macroscopic_group.add_field(mass_fraction);
		}
	}
}
/// ***************************************************** ///
/// INITIALIZE STENCIL, CREATE OUTPUT FOLDER AND ALLOCATE ///
/// MEMORY FOR FD SOLVER                                  ///
/// ***************************************************** ///
void Species_solver::Memory_allocation_FD(Stencil_Definition Stencil_Def, Parallel_MPI* MPI_parallel) {
	unsigned int X, Y, Z, k;
	/// -------------------------------------------------------------------------------------------
	Stencil_Def(Dimension, Discrete_Velocity, weight, c_alpha, alpha_bar, c_s2);
	weight_2.resize(Discrete_Velocity);
	weight_2[0] = 0;
	for (unsigned int alpha = 1; alpha < Discrete_Velocity; alpha++) {
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

	const Vector_field::Index_vec vec_sizes{x_size,
	                                        y_size,
	                                        z_size,
	                                        Nb_spec};

	molar_mass_av = Scalar_field::zeros(scalar_sizes);
	mass_fraction = Vector_field::zeros(vec_sizes);
	previous_mass_fraction = mass_fraction;
	temp_mass_fraction = mass_fraction;
	V_c = Vector_field::zeros({x_size, y_size, z_size, 3});
	diffusion_coefficient = Vector_field::zeros(vec_sizes);
	Production = Vector_field::zeros(vec_sizes);
	solid_species = Solid_field::zeros(scalar_sizes);

	cp_k = new double[Nb_spec];
	if (MPI_parallel->processor_id != MASTER) {
		Flux = new double****[MPI_parallel->dev_end[0]];
		for (X = 0; X < MPI_parallel->dev_end[0]; ++X) {
			Flux[X] = new double***[MPI_parallel->dev_end[1]];
			for (Y = 0; Y < MPI_parallel->dev_end[1]; ++Y) {
				Flux[X][Y] = new double**[MPI_parallel->dev_end[2]];
				for (Z = 0; Z < MPI_parallel->dev_end[2]; ++Z) {
					Flux[X][Y][Z] = new double*[Nb_spec];
					for (k = 0; k < Nb_spec; k++) {
						Flux[X][Y][Z][k] = new double[3];
					}
				}
			}
		}
	}
}
/// ***************************************************** ///
/// INITIALIZE FIELD VARIABLES, MASS FRACTION AND         ///
/// GEOMETRY                                              ///
/// ***************************************************** ///
void Species_solver::initialize_field(Geometry* Geo, stl_import* Geo_stl, Species_Ini Ini_Spec, Flow_solver* Flow, Parallel_MPI* MPI_parallel, const std::string& filename) {
    unsigned int X, Y, Z, k;
    COMP = 0;
    if (MPI_parallel->processor_id != MASTER) {
        if (Geo_stl->flag == 1) {
            for (X = 0; X < MPI_parallel->dev_end[0]; X++) {
                for (Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
                    for (Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
                        solid_species[{X, Y, Z}] = Geo_stl->domain[{X, Y, Z}];
                    }
                }
            }
        }

		Ini_Spec(mass_fraction, diffusion_coefficient, solid_species, species_name_RG, global_parameters.Nx,
		         global_parameters.Ny, global_parameters.Nz, Nb_spec, c_s2,
		         filename, Geo_stl->Source_count, MPI_parallel);
        /// -------------------------------------------------------------------------------------------
        for (X = 0; X < MPI_parallel->dev_end[0]; ++X) {
            for (Y = 0; Y < MPI_parallel->dev_end[1]; ++Y) {
                for (Z = 0; Z < MPI_parallel->dev_end[2]; ++Z) {
                    V_c[{X, Y, Z, 0}] = 0;
                    V_c[{X, Y, Z, 1}] = 0;
                    V_c[{X, Y, Z, 2}] = 0;
                    molar_mass_av[{X, Y, Z}] = 0;
                    for (k = 0; k < Nb_spec; ++k) {
                        molar_mass_av[{X, Y, Z}] += mass_fraction[{X, Y, Z, k}] / Molar_mass[k];
                        Production[{X, Y, Z, k}] = 0.;
                        Flux[X][Y][Z][k][0] = 0.;
                        Flux[X][Y][Z][k][1] = 0.;
                        Flux[X][Y][Z][k][2] = 0.;
                        previous_mass_fraction[{X, Y, Z, k}] = mass_fraction[{X, Y, Z, k}];
                    }
                    molar_mass_av[{X, Y, Z}] = 1. / molar_mass_av[{X, Y, Z}];
                }
            }
        }
    }
    return;
}
/// ***************************************************** ///
/// INITIALIZE FIELD VARIABLES, MASS FRACTION AND         ///
/// GEOMETRY FOR FD SOLVER                                ///
/// ***************************************************** ///
void Species_solver::initialize_field_FD(Geometry* Geo, stl_import* Geo_stl, Species_Ini Ini_Spec, Flow_solver* Flow, Thermal_solver* Thermal, Parallel_MPI* MPI_parallel, const std::string& filename) {
	unsigned int X, Y, Z, k;
	COMP = 0;
	if (MPI_parallel->processor_id != MASTER) {
		if (Geo_stl->flag == 1) {
			for (X = 0; X < MPI_parallel->dev_end[0]; X++) {
				for (Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
					for (Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
						solid_species[{X, Y, Z}] = Geo_stl->domain[{X, Y, Z}];
					}
				}
			}
		}
		Ini_Spec(mass_fraction, diffusion_coefficient, solid_species, species_name_RG, global_parameters.Nx,
		         global_parameters.Ny, global_parameters.Nz, Nb_spec, c_s2,
		         filename, Geo_stl->Source_count, MPI_parallel);
		/// -------------------------------------------------------------------------------------------
		for (X = 0; X < MPI_parallel->dev_end[0]; X++) {
			//	const unsigned xx = fmod(X - MPI_parallel->start_XYZ2[0] + MPI_parallel->start_XYZ[0] + global_parameters.Nx + 1, global_parameters.Nx);
			for (Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
				//	const unsigned yy = fmod(Y - MPI_parallel->start_XYZ2[1] + MPI_parallel->start_XYZ[1] + global_parameters.Ny, global_parameters.Ny);
				for (Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
					//	const unsigned zz = fmod(Z - MPI_parallel->start_XYZ2[2] + MPI_parallel->start_XYZ[2] + global_parameters.Nz, global_parameters.Nz);
					V_c[{X, Y, Z, 0}] = 0;
					V_c[{X, Y, Z, 1}] = 0;
					V_c[{X, Y, Z, 2}] = 0;
					molar_mass_av[{X, Y, Z}] = 0;
					for (k = 0; k < Nb_spec; ++k) {
						molar_mass_av[{X, Y, Z}] += mass_fraction[{X, Y, Z, k}] / Molar_mass[k];
						Production[{X, Y, Z, k}] = 0.;
						Flux[X][Y][Z][k][0] = 0.;
						Flux[X][Y][Z][k][1] = 0.;
						Flux[X][Y][Z][k][2] = 0.;
						previous_mass_fraction[{X, Y, Z, k}] = mass_fraction[{X, Y, Z, k}];
					}
					molar_mass_av[{X, Y, Z}] = 1. / molar_mass_av[{X, Y, Z}];
				}
			}
		}
	}
	return;
}
/// ***************************************************** ///
/// INITIALIZE POPULATIONS USING ONLY EQUILIBRIUM PART    ///
/// ***************************************************** ///
void Species_solver::initialize_pop_eq(Flow_solver* Flow, Parallel_MPI* MPI_parallel, const std::string& filename) {
	unsigned int X, Y, Z, c_i, k;
	double X_k, gamma_0;
	if (MPI_parallel->processor_id != MASTER) {
		/// Allocate memory for the populations
		for (X = 0; X < MPI_parallel->dev_end[0]; ++X) {
			for (Y = 0; Y < MPI_parallel->dev_end[1]; ++Y) {
				for (Z = 0; Z < MPI_parallel->dev_end[2]; ++Z) {
					for (k = 0; k < Nb_spec; ++k) {
						gamma_0 = molar_mass_av_0 / Molar_mass[k];
						X_k = mass_fraction[{X, Y, Z, k}] * (molar_mass_av[{X, Y, Z}] / Molar_mass[k]) / (Flow->density[{X, Y, Z}] * gamma_0);
						equilibrium(mass_fraction[{X, Y, Z, k}], mass_fraction[{X, Y, Z, k}], X_k, &Flow->velocity[{X, Y, Z, 0}], pop_eq_s);
						for (c_i = 0; c_i < Discrete_Velocity; ++c_i) {
							pop_old_s[c_i][X][Y][Z][k] = pop_eq_s[c_i];
							pop_s[c_i][X][Y][Z][k] = pop_eq_s[c_i];
						}
					}
				}
			}
		}
	}
	return;
}
void Species_solver::initialize_pop_eq_snow(Flow_solver* Flow, Phase_Field* Phase, Parallel_MPI* MPI_parallel, const std::string& filename) {
	unsigned int X, Y, Z, c_i, k;
	if (MPI_parallel->processor_id != MASTER) {
		/// Allocate memory for the populations
		for (X = 0; X < MPI_parallel->dev_end[0]; ++X) {
			for (Y = 0; Y < MPI_parallel->dev_end[1]; ++Y) {
				for (Z = 0; Z < MPI_parallel->dev_end[2]; ++Z) {
					for (k = 0; k < Nb_spec; ++k) {
						equilibrium_snow(mass_fraction[{X, Y, Z, k}], &Flow->velocity[{X, Y, Z, 0}], 1.0, 1.0, 1.0);
						for (c_i = 0; c_i < Discrete_Velocity; ++c_i) {
							pop_old_s[c_i][X][Y][Z][k] = pop_eq_s[c_i];
							pop_s[c_i][X][Y][Z][k] = pop_eq_s[c_i];
						}
					}
				}
			}
		}
	}
}
/// ***************************************************** ///
/// INITIALIZE BOUNDARY CONDITIONS                        ///
/// ***************************************************** ///
void Species_solver::initialize_BC(Geometry* Geo, stl_import* Geo_stl, Parallel_MPI* MPI_parallel, const std::string& filename) {
	if (MPI_parallel->processor_id != MASTER) {
		unsigned int X, Y, Z, i, k, j, alpha;
		int number_of_BC = 0;
		double M_av_temp, gamma_0;
		std::vector<species_boundary_data> temp;
		std::vector<int> index;
		int** intersection;
		/// Open input file
		string input_filename(filename);
		input_filename += ".dat";
		ifstream input_file;  // File is open for READING
		input_file.open(input_filename.c_str(), ios::binary);

		input_file.clear();
		input_file.seekg(0, ios::beg);
		find_line_after_header(input_file, "c\tSpecies Field Boundary Conditions");
		find_line_after_comment(input_file);
		input_file >> number_of_BC >> curved_boundaries;
		temp.resize(number_of_BC);
		index.resize(number_of_BC);
		intersection = new int*[number_of_BC];
		int counter_nonzero = 0;
		std::string species_name;

		for (i = 0; i < number_of_BC; i++) {
			intersection[i] = new int[2];
			find_line_after_comment(input_file);
			input_file >> index[i] >> intersection[i][0] >> intersection[i][1] >> temp[i].type;
			temp[i].Y_k.resize(Nb_spec);
			temp[i].k.resize(Nb_spec);
			temp[i].kp.resize(Nb_spec);
			switch (temp[i].type) {
				/// 1: Zero mass fraction BC used when the boundary is a wall. It is used to set the mass fraction to
				/// zero at the boundary. Eq: Yk = 0 at the boundary. It is aka Dirichlet BC of 0th order.
				case 1:  /// --> Zero mass fraction BC
				{
					for (k = 0; k < Nb_spec; k++) {
						temp[i].Y_k[k] = 0.;
						temp[i].k[k] = -1;
					}
					break;
				}
				/// 2: Non-zero mass fraction BC used when the boundary is a wall. It is used to set the mass fraction to
				/// a non-zero value at the boundary. Which means that the mass fraction is set to a non-zero value at the
				/// boundary. Eq: Yk = Yk at the boundary. It re-scales the mass fraction to get the re-scaled equivalent
				/// mole fraction. Eq: Xk = Yk*W/Wk at the boundary. It is aka Dirichlet BC of 1st order.
				case 2:  /// ---> Non-zero mass fraction BC
				{
					counter_nonzero = 0;
					// This index reads the number of Species with nonzero mass fraction at the boundary
					input_file >> counter_nonzero;
					for (k = 0; k < Nb_spec; k++) {
						temp[i].Y_k[k] = 0;
						temp[i].k[k] = -1;
					}
					for (j = 0; j < counter_nonzero; j++) {
						input_file >> species_name;
						for (k = 0; k < Nb_spec; k++) {
							if (!species_name.compare(species_name_RG[k])) {
								input_file >> temp[i].Y_k[k];
								temp[i].k[k] = 1;
							}
						}
					}
					/// re-scale to get re-scaled equivalent mole fraction
					M_av_temp = 0;
					for (k = 0; k < Nb_spec; k++) {
						M_av_temp += (temp[i].Y_k[k] / Molar_mass[k]);
					}
					M_av_temp = 1. / M_av_temp;
					for (k = 0; k < Nb_spec; k++) {
						gamma_0 = molar_mass_av_0 / Molar_mass[k];
						temp[i].Y_k[k] = temp[i].Y_k[k] * (M_av_temp / Molar_mass[k]) / gamma_0;
					}
					break;
				}
				/// 3: Zero-gradient (1st-order) BC used when the boundary is an inlet or outlet. It is used to set the
				/// mass fraction gradient to zero at the boundary. Eq: dYk/dn = 0 at the boundary. It is used to set the
				/// mass fraction gradient to zero at the boundary. It is aka Neumann BC of 1st order (zero-gradient).
				case 3:  /// --> Zero-gradient (1st-order) BC
				{
					counter_nonzero = 0;
					// This index reads the number of Species with nonzero mass fraction at the boundary
					input_file >> counter_nonzero;
					for (k = 0; k < Nb_spec; k++) {
						temp[i].Y_k[k] = 0;
						temp[i].k[k] = -1;
					}
					for (j = 0; j < counter_nonzero; j++) {
						input_file >> species_name;
						for (k = 0; k < Nb_spec; k++) {
							if (!species_name.compare(species_name_RG[k])) {
								temp[i].k[k] = 1;
							}
						}
					}
					break;
				}
				/// 4: Constant flux (1st-order) BC used when the boundary is an inlet or outlet. It is used to set the
				/// mass fraction flux to a constant value at the boundary. Eq: dYk/dn = C at the boundary. It is used to
				/// set the mass fraction flux to a constant value at the boundary. It is aka Neumann BC of 1st order (constant flux).
				case 4:  /// --> Constant flux (1st-order) BC
				{
					counter_nonzero = 0;
					// This index reads the number of Species with nonzero mass fraction at the boundary
					input_file >> counter_nonzero;
					for (k = 0; k < Nb_spec; k++) {
						temp[i].Y_k[k] = 0;
						temp[i].k[k] = -1;
					}
					for (j = 0; j < counter_nonzero; j++) {
						input_file >> species_name;
						for (k = 0; k < Nb_spec; k++) {
							if (!species_name.compare(species_name_RG[k])) {
								input_file >> temp[i].Y_k[k];
								temp[i].k[k] = 1;
							}
						}
					}
					break;
				}
				/// 102: Zero-gradient (2nd-order) BC used when the boundary is an inlet or outlet. It is used to set the
				/// mass fraction gradient to zero at the boundary. Eq: d2Yk/dn2 = 0 at the boundary. It is used to set the
				/// mass fraction gradient to zero at the boundary. It is aka Neumann BC of 2nd order.
				case 102:  /// ---> Non-zero mass fraction BC
				{
					// This index reads the number of Species with nonzero mass fraction at the boundary
					int counter_nonzero = 0;
					std::string species_name;
					input_file >> counter_nonzero;
					for (k = 0; k < Nb_spec; k++) {
						temp[i].Y_k[k] = 0;
						temp[i].k[k] = -1;
					}
					for (j = 0; j < counter_nonzero; j++) {
						input_file >> species_name;
						for (k = 0; k < Nb_spec; k++) {
							if (!species_name.compare(species_name_RG[k])) {
								input_file >> temp[i].Y_k[k];
								temp[i].k[k] = 1;
							}
						}
					}
					break;
				}
				/// 104: Constant flux (2nd-order) BC used when the boundary is an inlet or outlet. It is used to set the
				/// mass fraction flux to a constant value at the boundary. Eq: d2Yk/dn2 = C at the boundary. It is used to
				/// set the mass fraction flux to a constant value at the boundary. It is aka Neumann BC of 2nd order.
				case 104:
				case 105:
				case 106:
				case 107: {
					counter_nonzero = 0;
					// This index reads the number of Species with nonzero mass fraction at the boundary
					input_file >> counter_nonzero;
					for (k = 0; k < Nb_spec; k++) {
						temp[i].Y_k[k] = 0;
						temp[i].k[k] = -1;
					}
					for (j = 0; j < counter_nonzero; j++) {
						input_file >> species_name;
						for (k = 0; k < Nb_spec; k++) {
							if (!species_name.compare(species_name_RG[k])) {
								temp[i].k[k] = 1;  // Mark this Species as having a boundary condition
							}
						}
					}
					break;
				}
				/// 108: Zero flux (1st-order) BC used when the boundary is an inlet or outlet. It is used to set the
				/// mass fraction flux to a constant value at the boundary. Eq: dYk/dn = 0 at the boundary. It is used to
				/// set the mass fraction flux to a constant value at the boundary. It is aka Neumann BC of 1st order (zero flux).
				case 108:  /// --> zero flux (1st-order) BC
				{
					int counter_nonzero = 0;
					input_file >> counter_nonzero;
					for (int k = 0; k < Nb_spec; k++) {
						temp[i].Y_k[k] = 0;
						temp[i].k[k] = -1;
					}
					for (int j = 0; j < counter_nonzero; j++) {
						std::string species_name;
						input_file >> species_name;
						for (int k = 0; k < Nb_spec; k++) {
							if (!species_name.compare(species_name_RG[k])) {
								temp[i].k[k] = 1;
							}
						}
					}
					break;
				}
				/// 109: Non-zero flux fraction BC used when the boundary is an inlet or outlet. It is used to set the
				/// mass fraction flux to a constant value at the boundary. Eq: dYk/dn = C at the boundary. It is used to
				/// set the mass fraction flux to a constant value at the boundary. It is aka Neumann BC of 1st order (constant flux).
				case 109:  /// ---> Non-zero flux fraction BC
				{
					// This index reads the number of Species with nonzero mass fraction at the boundary
					int counter_nonzero = 0;
					std::string species_name;
					std::string species_name_2;
					input_file >> counter_nonzero;
					for (k = 0; k < Nb_spec; k++) {
						temp[i].Y_k[k] = 0;
						temp[i].k[k] = -1;
					}
					for (j = 0; j < counter_nonzero; j++) {
						input_file >> species_name;
						for (k = 0; k < Nb_spec; k++) {
							if (!species_name.compare(species_name_RG[k])) {
								input_file >> temp[i].Y_k[k];
								temp[i].Y_k[k] = temp[i].Y_k[k] * global_parameters.D_t / global_parameters.D_x;
								temp[i].k[k] = 1;
								input_file >> species_name_2;
								for (unsigned int kk = 0; kk < Nb_spec; kk++) {
									if (!species_name_2.compare(species_name_RG[kk])) {
										temp[i].kp[k] = kk;
									}
								}
							}
						}
					}
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
			ofstream BC_output("Alborz_Results/Data/Species_Boundary_Conditions.dat", fstream::app);
			BC_output << "SPECIES: Boundary Conditions \n";
			BC_output << "================================ \n";
			if (curved_boundaries)
				BC_output << "Curved: "
						  << "\tON\t" << "\n";
			else
				BC_output << "Curved: "
						  << "\tOFF\n";
			BC_output << "================================ \n";

			for (i = 0; i < number_of_BC; i++) {
				BC_output << "BC : " << i << "\t Type : " << temp[i].type << std::endl;
				BC_output << "Mass fraction : " << std::endl;
				for (k = 0; k < Nb_spec; ++k) {
					BC_output << species_name_RG[k] << " :\t" << temp[i].Y_k[k] << "\t, Affected :\t";
					if (temp[i].k[k] == 1) { BC_output << "YES\n"; }
					if (temp[i].k[k] == -1) { BC_output << "NO\n"; }
				}
				BC_output << endl;
				BC_output << "-------------------------------- \n";
			}
			if (number_of_BC == 0) BC_output << "NO SPECIES BOUNDARY CONDITIONS\n";
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
						if (Geo_stl->domain[{X, Y, Z}] == intersection[i][0] && solid_species[{X, Y, Z}] == FALSE) {
							n_temp[0] = 0;
							n_temp[1] = 0;
							n_temp[2] = 0;
							/* c_alpha: Stencil vectors or discrete velocity vectors
							They are used to find the neighboring nodes by adding or subtracting them from the current node.
							c_alpha are the stencil vectors that are used to find the neighboring nodes of a node.
							For example, c_alpha[0] is the vector that points to the node in the positive x-direction.
							For a 3D lattice, there are 13 vectors in total. They are : the node itself and the 12 neighboring nodes, which are the 6 face-centered nodes, the 4 edge-centered nodes, and the 2 corner-centered nodes.
							For a 2D lattice, there are 5 vectors in total. They are : the node itself and the four neighboring nodes, which are the 4 edge-centered nodes.
							For a 1D lattice, there are 3 vectors in total. This is the node itself and the two neighboring nodes, which are the 2 edge-centered nodes.
							For a 0D lattice, there is only 1 vector. This is the node itself, and there are no neighboring nodes.

							These are used to determine the neighboring nodes of a node.
							When Xp = X  + c_alph, then the neighboring node is in the positive direction, which means that the neighboring node lies outside the domain.
							When Xp = X  - c_alph, then the neighboring node is in the negative direction, which means that the neighboring node lies inside the domain.
							*/
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
								if (n_temp[2] == 0) temp[i].n[2] = 0;
								Boundaries.push_back(temp[i]);
								BC_flag = 0;
								/// ALLOCATE MEMORY TO SOLID NEIGHBOR LIST
								for (alpha = 0; alpha < Discrete_Velocity; alpha++) {
									Xp = X - c_alpha[alpha][0];
									Yp = Y - c_alpha[alpha][1];
									Zp = Z - c_alpha[alpha][2];
									Boundaries[Boundaries.size() - 1].directions[alpha] = -1;
									if (Geo_stl->domain[{Xp, Yp, Zp}] == intersection[i][1] && Boundaries[Boundaries.size() - 1].type < 100) { Boundaries[Boundaries.size() - 1].directions[alpha] = 1; }
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
		DB_filename << "Alborz_Results/debug/Species_Boundary_Conditions_DB_proc_" << MPI_parallel->processor_id << ".dat";
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
/// ***************************************************** ///
/// GET DISTANCE FROM LAST FLUID NODE TO STL SURFACE      ///
/// ALONG STENCIL VECTORS FOR CURVED BOUNDARIES           ///
/// ***************************************************** ///
void Species_solver::Initialize_curved_boundaries_FD(stl_import* Geo_stl, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		std::vector<stl::triangle> triangles;
		std::vector<int> triangles_which_file;
		stl::point lower_limits, upper_limits;
		triangles.clear();
		for (int file_index = 0; file_index < Geo_stl->Source_count; file_index++) {
			stl::stl_data info = stl::parse_stl(Geo_stl->Geo_filename[file_index]);
			std::vector<int> temp_which_file(info.triangles.size(), file_index + 1);
			triangles.insert(triangles.end(), info.triangles.begin(), info.triangles.end());
			triangles_which_file.insert(triangles_which_file.end(), temp_which_file.begin(), temp_which_file.end());
		}
		for (auto& boundary : Boundaries) {
			stl::point O, intersection_coordinate, direction;
			MPI_parallel->get_coordinates(boundary.X, boundary.Y, boundary.Z, Geo_stl->x_center, Geo_stl->y_center, Geo_stl->z_center, O.x, O.y, O.z);
			double minimum_distance = 1.0e3;
			for (size_t l = 0; l < triangles.size(); l++) {
				if (triangles_which_file[l] == boundary.V_out) {
					direction = triangles[l].normal;
					double distance;
					int intersection_indicator = Geo_stl->triangle_shortest_distance(triangles[l].v1, triangles[l].v2, triangles[l].v3, O, direction, &intersection_coordinate);
					if (intersection_indicator == 1) {
						distance = sqrt(pow(O.x - intersection_coordinate.x, 2) + pow(O.y - intersection_coordinate.y, 2) + pow(O.z - intersection_coordinate.z, 2));
						double normX = (intersection_coordinate.x - O.x) / global_parameters.D_x;
						double normY = (intersection_coordinate.y - O.y) / global_parameters.D_x;
						double normZ = (intersection_coordinate.z - O.z) / global_parameters.D_x;
						if (distance < minimum_distance && (distance / global_parameters.D_x) <= sqrt(Dimension)) {
							int Xp = boundary.X + SGN(normX) * ceil(fabs(normX));
							int Yp = boundary.Y + SGN(normY) * ceil(fabs(normY));
							int Zp = boundary.Z + SGN(normZ) * ceil(fabs(normZ));
							// Xp is outside the domain
							if (Geo_stl->domain[{Xp, Yp, Zp}] == boundary.V_in) {
								minimum_distance = distance;
								boundary.normal_distance = distance / global_parameters.D_x;
								boundary.n[0] = normX;
								boundary.n[1] = normY;
								boundary.n[2] = normZ;
							}
						}
					}
				}
			}
		}
		// Initialize image points and their neighbors
		for (auto& boundary : Boundaries) {
			int D = Dimension;
			if (D < 3) boundary.n[2] = 0;
			boundary.get_image();
			for (int xx = floor(boundary.X_Image) - 1; xx <= ceil(boundary.X_Image) + 1; ++xx) {
				for (int yy = floor(boundary.Y_Image) - 1; yy <= ceil(boundary.Y_Image) + 1; ++yy) {
					if (D > 2) {
						for (int zz = floor(boundary.Z_Image) - 1; zz <= ceil(boundary.Z_Image) + 1; ++zz) {
							if (solid_species[{xx, yy, zz}] == FALSE) {
								boundary.X_Image_Int.push_back(xx);
								boundary.Y_Image_Int.push_back(yy);
								boundary.Z_Image_Int.push_back(zz);
							}
						}
					}
					if (D < 3) {
						if (solid_species[{xx, yy, boundary.Z}] == FALSE) {
							boundary.X_Image_Int.push_back(xx);
							boundary.Y_Image_Int.push_back(yy);
							boundary.Z_Image_Int.push_back(boundary.Z);
						}
					}
				} /*End of scan in Y-dir*/
			} /*End of scan in X-dir*/
			boundary.get_image_int_weights();
			if (boundary.X_Image_Int.empty()) {
				boundary.X_Image_Int.push_back(boundary.X);
				boundary.Y_Image_Int.push_back(boundary.Y);
				boundary.Z_Image_Int.push_back(boundary.Z);
				boundary.W_Image_Int.push_back(1.0);
			}
		} /*End of Image initialization*/
	}
}
/// ***************************************************** ///
/// GET EQUILIBRIUM POPULATIONS AND PUT THEM IN F_EQ      ///
/// ***************************************************** ///
void Species_solver::equilibrium(double& A, double& B, double& C, double* u, double* pop_eq) {
	double H0, Hx, Hy, Hz, Hxx, Hyy, Hzz;
	double a0, ax, ay, az, axx, ayy, azz;
	a0 = A;
	ax = B * u[0];
	ay = B * u[1];
	az = B * u[2];
	axx = (C - A);
	ayy = (C - A);
	azz = (C - A);
	for (int alpha = 0; alpha < Discrete_Velocity; alpha++) {
		H0 = 1.;
		Hx = c_alpha[alpha][0];
		Hy = c_alpha[alpha][1];
		Hz = c_alpha[alpha][2];
		Hxx = c_alpha[alpha][0] * c_alpha[alpha][0] - (1. / c_s2);
		Hyy = c_alpha[alpha][1] * c_alpha[alpha][1] - (1. / c_s2);

		pop_eq[alpha] = weight[alpha] * (a0 * H0 + c_s2 * (ax * Hx + ay * Hy) + 0.5 * c_s2 * (axx * Hxx + ayy * Hyy));
		if (Dimension == 3) {
			Hzz = c_alpha[alpha][2] * c_alpha[alpha][2] - (1. / c_s2);
			pop_eq[alpha] += weight[alpha] * (c_s2 * az * Hz + 0.5 * c_s2 * (azz * Hzz));
		}
	}
}
void Species_solver::equilibrium_snow(double& A, double* vel, double Gamma_x, double Gamma_y, double Gamma_z) {
	double Hxx;
	double Hyy;
	double Hzz;
	/// no Flow velocity
	for (int alpha = 0; alpha < Discrete_Velocity; alpha++) {
		Hxx = c_alpha[alpha][0] * c_alpha[alpha][0] - 1. / c_s2;
		Hyy = c_alpha[alpha][1] * c_alpha[alpha][1] - 1. / c_s2;
		Hzz = c_alpha[alpha][2] * c_alpha[alpha][2] - 1. / c_s2;
		pop_eq_s[alpha] = weight[alpha] * A * (1. + 0.5 * c_s2 * ((SQ(Gamma_x) - 1) * Hxx + (SQ(Gamma_y) - 1) * Hyy + (SQ(Gamma_z) - 1) * Hzz) + c_s2 * DOT(c_alpha[alpha], vel));
	}
}
/// ***************************************************** ///
/// COLLISION AND STREAMING USING SRT MODEL               ///
/// ***************************************************** ///
void Species_solver::LBM_SRT(int time, Flow_solver* Flow, Thermal_solver* Thermal, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		double***** swap_temp = pop_old_s;
		pop_old_s = pop_s;
		pop_s = swap_temp;
		int k, alpha, X, Y, Z;
		double omega_eff, X_k, gamma_0, PHI, rhok, rhok_previous;
		for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (solid_species[{X, Y, Z}] == FALSE) {
						for (k = 0; k < Nb_spec; ++k) {
							gamma_0 = molar_mass_av_0 / Molar_mass[k];
							X_k = mass_fraction[{X, Y, Z, k}] * (molar_mass_av[{X, Y, Z}] / Molar_mass[k]) / (Flow->density[{X, Y, Z}] * gamma_0);
							equilibrium(mass_fraction[{X, Y, Z, k}], mass_fraction[{X, Y, Z, k}], X_k, &Flow->velocity[{X, Y, Z, 0}], pop_eq_s);
							omega_eff = 1. / (c_s2 * Flow->density[{X, Y, Z}] * gamma_0 * diffusion_coefficient[{X, Y, Z, k}] / (molar_mass_av[{X, Y, Z}] / Molar_mass[k]) + 0.5);
							rhok = mass_fraction[{X, Y, Z, k}];
							rhok_previous = previous_mass_fraction[{X, Y, Z, k}];
							for (alpha = 0; alpha < Discrete_Velocity; alpha++) {
								PHI = c_s2 * c_alpha[alpha][0] * (Flow->velocity[{X, Y, Z, 0}] * rhok - Flow->previous_velocity[{X, Y, Z, 0}] * rhok_previous)
								      + c_s2 * c_alpha[alpha][1] * (Flow->velocity[{X, Y, Z, 1}] * rhok - Flow->previous_velocity[{X, Y, Z, 1}] * rhok_previous)
								      + c_s2 * c_alpha[alpha][2] * (Flow->velocity[{X, Y, Z, 2}] * rhok - Flow->previous_velocity[{X, Y, Z, 2}] * rhok_previous);
								pop_s[alpha][X + c_alpha[alpha][0]][Y + c_alpha[alpha][1]][Z + c_alpha[alpha][2]][k] =
									pop_old_s[alpha][X][Y][Z][k] * (1. - omega_eff) + pop_eq_s[alpha] * omega_eff
									+ weight[alpha] * (1. - 0.5 * omega_eff) * (Production[{X, Y, Z, k}] + PHI);
							}
						}
					}
				}
			}
		}
		return;
	}
}
void Species_solver::LBM_SRT_snow(int time, Flow_solver* Flow, Thermal_solver* Thermal, Phase_Field* Phase, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		double***** swap_temp = pop_old_s;
		pop_old_s = pop_s;
		pop_s = swap_temp;
		int k, alpha, X, Y, Z;
		double Gamma_x, Gamma_y, Gamma_z;
		double omega_eff;

		for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (solid_species[{X, Y, Z}] == -1) {
						for (k = 0; k < Nb_spec; ++k) {
							Gamma_x = Phase->Gamma_phase[0];
							Gamma_y = Phase->Gamma_phase[1];
							Gamma_z = Phase->Gamma_phase[2];

							omega_eff = 1. / (c_s2 * diffusion_coefficient[{X, Y, Z, k}] + 0.5);

							equilibrium_snow(mass_fraction[{X, Y, Z, k}], &Flow->velocity[{X, Y, Z, 0}], Gamma_x, Gamma_y, Gamma_z);
							for (alpha = 0; alpha < Discrete_Velocity; alpha++) {
								pop_s[alpha][X + c_alpha[alpha][0]][Y + c_alpha[alpha][1]][Z + c_alpha[alpha][2]][k] =
									pop_old_s[alpha][X][Y][Z][k] * (1.0 - omega_eff)
									+ pop_eq_s[alpha] * omega_eff
									+ weight[alpha] * Production[{X, Y, Z, k}];
							}
						}
					}
				}
			}
		}
		return;
	}
}
/// ***************************************************** ///
/// COLLISION AND STREAMING USING MRT MODEL               ///
/// ***************************************************** ///
void Species_solver::LBM_MRT(int time, Flow_solver* Flow, Thermal_solver* Thermal, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		double***** swap_temp_s = pop_old_s;
		pop_old_s = pop_s;
		pop_s = swap_temp_s;
		int X, Y, Z, k;
		double tau_eff;

		vector<double> omega_eff;
		vector<double> Momeq;
		vector<double> Mom;
		double ux, uy, Xk, rhok, rhok_previous, Fx, Fy, gamma_0;

		if (Discrete_Velocity == 9) {
			omega_eff.resize(Discrete_Velocity);
			Mom.resize(Discrete_Velocity);
			Momeq.resize(Discrete_Velocity);
			for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
				for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
					for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
						if (solid_species[{X, Y, Z}] == FALSE) {
							for (k = 0; k < Nb_spec; ++k) {
								gamma_0 = molar_mass_av_0 / Molar_mass[k];
								tau_eff = c_s2 * Flow->density[{X, Y, Z}] * gamma_0 * diffusion_coefficient[{X, Y, Z, k}] / (molar_mass_av[{X, Y, Z}] / Molar_mass[k]) + 0.5;

								omega_eff[0] = 1.;
								omega_eff[1] = 1. / tau_eff;
								omega_eff[2] = 1. / tau_eff;
								omega_eff[3] = 1.;
								omega_eff[4] = 1.;
								omega_eff[5] = 1.;
								omega_eff[6] = 1.;
								omega_eff[7] = 1.;
								omega_eff[8] = 1.;

								//	const double rho = Flow->density[{X,Y,Z}];
								ux = Flow->velocity[{X, Y, Z, 0}];
								uy = Flow->velocity[{X, Y, Z, 1}];
								rhok = mass_fraction[{X, Y, Z, k}];
								Xk = mass_fraction[{X, Y, Z, k}] * (molar_mass_av[{X, Y, Z}] / Molar_mass[k]) / (Flow->density[{X, Y, Z}] * gamma_0);

								rhok_previous = previous_mass_fraction[{X, Y, Z, k}];

								Fx = rhok * Flow->velocity[{X, Y, Z, 0}] - rhok_previous * Flow->previous_velocity[{X, Y, Z, 0}];
								Fy = rhok * Flow->velocity[{X, Y, Z, 1}] - rhok_previous * Flow->previous_velocity[{X, Y, Z, 1}];
								/// *************************************************************************************************** ///
								///                                       EQUILIBRIUM CENTRAL MOMENTS                                   ///
								///                                             MOMENT SPACE :                                          ///
								///                                                  M_0,                                               ///
								///                                                M_x, M_y,                                            ///
								///                                       M_xx+M_yy, M_xx-M_yy, M_xy                                    ///
								///                                              M_xxy, M_xyy                                           ///
								///                                                  M_xxyy                                             ///
								/// *************************************************************************************************** ///
								Momeq[0] = rhok;
								Momeq[1] = rhok * ux;
								Momeq[2] = rhok * uy;
								Momeq[3] = 0;
								Momeq[4] = (Xk - rhok) / c_s2;
								Momeq[5] = (Xk - rhok) / c_s2;
								Momeq[6] = 0;
								Momeq[7] = 0;
								Momeq[8] = 0;

								Mom[0] = Momeq[0] * omega_eff[0] + (1. - omega_eff[0]) * (pop_old_s[0][X][Y][Z][k] + pop_old_s[1][X][Y][Z][k] + pop_old_s[2][X][Y][Z][k] + pop_old_s[3][X][Y][Z][k] + pop_old_s[4][X][Y][Z][k] + pop_old_s[5][X][Y][Z][k] + pop_old_s[6][X][Y][Z][k] + pop_old_s[7][X][Y][Z][k] + pop_old_s[8][X][Y][Z][k]);
								Mom[1] = Momeq[1] * omega_eff[1] + (1. - omega_eff[1]) * (pop_old_s[1][X][Y][Z][k] - pop_old_s[3][X][Y][Z][k] + pop_old_s[5][X][Y][Z][k] - pop_old_s[6][X][Y][Z][k] - pop_old_s[7][X][Y][Z][k] + pop_old_s[8][X][Y][Z][k]);
								Mom[2] = Momeq[2] * omega_eff[2] + (1. - omega_eff[2]) * (pop_old_s[2][X][Y][Z][k] - pop_old_s[4][X][Y][Z][k] + pop_old_s[5][X][Y][Z][k] + pop_old_s[6][X][Y][Z][k] - pop_old_s[7][X][Y][Z][k] - pop_old_s[8][X][Y][Z][k]);
								Mom[3] = Momeq[3] * omega_eff[3] + (1. - omega_eff[3]) * (pop_old_s[5][X][Y][Z][k] - pop_old_s[6][X][Y][Z][k] + pop_old_s[7][X][Y][Z][k] - pop_old_s[8][X][Y][Z][k]);
								Mom[4] = Momeq[4] * omega_eff[4] + (1. - omega_eff[4]) * (-pop_old_s[0][X][Y][Z][k] / (double)c_s2 - pop_old_s[2][X][Y][Z][k] / (double)c_s2 - pop_old_s[4][X][Y][Z][k] / (double)c_s2 - pop_old_s[1][X][Y][Z][k] * (1 / (double)c_s2 - 1) - pop_old_s[3][X][Y][Z][k] * (1 / (double)c_s2 - 1) - pop_old_s[5][X][Y][Z][k] * (1 / (double)c_s2 - 1) - pop_old_s[6][X][Y][Z][k] * (1 / (double)c_s2 - 1) - pop_old_s[7][X][Y][Z][k] * (1 / (double)c_s2 - 1) - pop_old_s[8][X][Y][Z][k] * (1 / (double)c_s2 - 1));
								Mom[5] = Momeq[5] * omega_eff[5] + (1. - omega_eff[5]) * (-pop_old_s[0][X][Y][Z][k] / (double)c_s2 - pop_old_s[1][X][Y][Z][k] / (double)c_s2 - pop_old_s[3][X][Y][Z][k] / (double)c_s2 - pop_old_s[2][X][Y][Z][k] * (1 / (double)c_s2 - 1) - pop_old_s[4][X][Y][Z][k] * (1 / (double)c_s2 - 1) - pop_old_s[5][X][Y][Z][k] * (1 / (double)c_s2 - 1) - pop_old_s[6][X][Y][Z][k] * (1 / (double)c_s2 - 1) - pop_old_s[7][X][Y][Z][k] * (1 / (double)c_s2 - 1) - pop_old_s[8][X][Y][Z][k] * (1 / (double)c_s2 - 1));
								Mom[6] = Momeq[6] * omega_eff[6] + (1. - omega_eff[6]) * (pop_old_s[4][X][Y][Z][k] / (double)c_s2 - pop_old_s[2][X][Y][Z][k] / (double)c_s2 - pop_old_s[5][X][Y][Z][k] * (1 / (double)c_s2 - 1) - pop_old_s[6][X][Y][Z][k] * (1 / (double)c_s2 - 1) + pop_old_s[7][X][Y][Z][k] * (1 / (double)c_s2 - 1) + pop_old_s[8][X][Y][Z][k] * (1 / (double)c_s2 - 1));
								Mom[7] = Momeq[7] * omega_eff[7] + (1. - omega_eff[7]) * (pop_old_s[3][X][Y][Z][k] / (double)c_s2 - pop_old_s[1][X][Y][Z][k] / (double)c_s2 - pop_old_s[5][X][Y][Z][k] * (1 / (double)c_s2 - 1) + pop_old_s[6][X][Y][Z][k] * (1 / (double)c_s2 - 1) + pop_old_s[7][X][Y][Z][k] * (1 / (double)c_s2 - 1) - pop_old_s[8][X][Y][Z][k] * (1 / (double)c_s2 - 1));
								Mom[8] = Momeq[8] * omega_eff[8] + (1. - omega_eff[8]) * (pop_old_s[5][X][Y][Z][k] * sqr(1 / (double)c_s2 - 1) + pop_old_s[6][X][Y][Z][k] * sqr(1 / (double)c_s2 - 1) + pop_old_s[7][X][Y][Z][k] * sqr(1 / (double)c_s2 - 1) + pop_old_s[8][X][Y][Z][k] * sqr(1 / (double)c_s2 - 1) + pop_old_s[0][X][Y][Z][k] / (double)sqr(c_s2) + (pop_old_s[1][X][Y][Z][k] * (1 / (double)c_s2 - 1)) / (double)c_s2 + (pop_old_s[2][X][Y][Z][k] * (1 / (double)c_s2 - 1)) / (double)c_s2 + (pop_old_s[3][X][Y][Z][k] * (1 / (double)c_s2 - 1)) / (double)c_s2 + (pop_old_s[4][X][Y][Z][k] * (1 / (double)c_s2 - 1)) / (double)c_s2);

								Mom[1] += (1. - 0.5 * omega_eff[1]) * Fx;
								Mom[2] += (1. - 0.5 * omega_eff[2]) * Fy;

								pop_s[0][X + c_alpha[0][0]][Y + c_alpha[0][1]][Z + c_alpha[0][2]][k] = Mom[8] - (Mom[4] * (c_s2 - 1)) / (double)c_s2 - (Mom[5] * (c_s2 - 1)) / (double)c_s2 + (Mom[0] * sqr(c_s2 - 1)) / (double)sqr(c_s2);
								pop_s[1][X + c_alpha[1][0]][Y + c_alpha[1][1]][Z + c_alpha[1][2]][k] = (Mom[0] * (c_s2 - 1)) / (double)(2 * sqr(c_s2)) - Mom[8] / (double)2 - Mom[5] / (double)(2 * c_s2) - Mom[7] / (double)2 + (Mom[1] * (c_s2 - 1)) / (double)(2 * c_s2) + (Mom[4] * (c_s2 - 1)) / (double)(2 * c_s2);
								pop_s[2][X + c_alpha[2][0]][Y + c_alpha[2][1]][Z + c_alpha[2][2]][k] = (Mom[0] * (c_s2 - 1)) / (double)(2 * sqr(c_s2)) - Mom[8] / (double)2 - Mom[4] / (double)(2 * c_s2) - Mom[6] / (double)2 + (Mom[2] * (c_s2 - 1)) / (double)(2 * c_s2) + (Mom[5] * (c_s2 - 1)) / (double)(2 * c_s2);
								pop_s[3][X + c_alpha[3][0]][Y + c_alpha[3][1]][Z + c_alpha[3][2]][k] = Mom[7] / (double)2 - Mom[8] / (double)2 - Mom[5] / (double)(2 * c_s2) + (Mom[0] * (c_s2 - 1)) / (double)(2 * sqr(c_s2)) - (Mom[1] * (c_s2 - 1)) / (double)(2 * c_s2) + (Mom[4] * (c_s2 - 1)) / (double)(2 * c_s2);
								pop_s[4][X + c_alpha[4][0]][Y + c_alpha[4][1]][Z + c_alpha[4][2]][k] = Mom[6] / (double)2 - Mom[8] / (double)2 - Mom[4] / (double)(2 * c_s2) + (Mom[0] * (c_s2 - 1)) / (double)(2 * sqr(c_s2)) - (Mom[2] * (c_s2 - 1)) / (double)(2 * c_s2) + (Mom[5] * (c_s2 - 1)) / (double)(2 * c_s2);
								pop_s[5][X + c_alpha[5][0]][Y + c_alpha[5][1]][Z + c_alpha[5][2]][k] = (Mom[0] + Mom[1] * c_s2 + Mom[2] * c_s2 + Mom[4] * c_s2 + Mom[5] * c_s2 + Mom[3] * sqr(c_s2) + Mom[6] * sqr(c_s2) + Mom[7] * sqr(c_s2) + Mom[8] * sqr(c_s2)) / (double)(4 * sqr(c_s2));
								pop_s[6][X + c_alpha[6][0]][Y + c_alpha[6][1]][Z + c_alpha[6][2]][k] = (Mom[0] - Mom[1] * c_s2 + Mom[2] * c_s2 + Mom[4] * c_s2 + Mom[5] * c_s2 - Mom[3] * sqr(c_s2) + Mom[6] * sqr(c_s2) - Mom[7] * sqr(c_s2) + Mom[8] * sqr(c_s2)) / (double)(4 * sqr(c_s2));
								pop_s[7][X + c_alpha[7][0]][Y + c_alpha[7][1]][Z + c_alpha[7][2]][k] = (Mom[0] - Mom[1] * c_s2 - Mom[2] * c_s2 + Mom[4] * c_s2 + Mom[5] * c_s2 + Mom[3] * sqr(c_s2) - Mom[6] * sqr(c_s2) - Mom[7] * sqr(c_s2) + Mom[8] * sqr(c_s2)) / (double)(4 * sqr(c_s2));
								pop_s[8][X + c_alpha[8][0]][Y + c_alpha[8][1]][Z + c_alpha[8][2]][k] = (Mom[0] + Mom[1] * c_s2 - Mom[2] * c_s2 + Mom[4] * c_s2 + Mom[5] * c_s2 - Mom[3] * sqr(c_s2) - Mom[6] * sqr(c_s2) + Mom[7] * sqr(c_s2) + Mom[8] * sqr(c_s2)) / (double)(4 * sqr(c_s2));
							}
						}
					}
				}
			}
		}
		return;
	}
}
/// ***************************************************** ///
/// MASS CORRECTOR (FOR LB SOLVER)                        ///
/// ***************************************************** ///
void Species_solver::mass_corrector(int time, Flow_solver* Flow, Thermal_solver* Thermal, Parallel_MPI* MPI_parallel) {
	double tau_k, pop_s_tot, gamma_0;
	if (MPI_parallel->processor_id != MASTER) {
		int X, Y, Z, alpha, k;
		for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (solid_species[{X, Y, Z}] == FALSE) {
						for (alpha = 0; alpha < Discrete_Velocity; alpha++) {
							pop_s_tot = 0;
							for (k = 0; k < Nb_spec; ++k) {
								gamma_0 = molar_mass_av_0 / Molar_mass[k];
								tau_k = c_s2 * Flow->density[{X, Y, Z}] * gamma_0 * diffusion_coefficient[{X, Y, Z, k}] / (molar_mass_av[{X, Y, Z}] / Molar_mass[k]) + 0.5;
								pop_s_tot = pop_s_tot + (tau_k - 0.5) * (pop_s[alpha][X + c_alpha[alpha][0]][Y + c_alpha[alpha][1]][Z + c_alpha[alpha][2]][k] - pop_old_s[alpha][X][Y][Z][k]);
							}
							for (k = 0; k < Nb_spec; ++k) {
								gamma_0 = molar_mass_av_0 / Molar_mass[k];
								tau_k = c_s2 * Flow->density[{X, Y, Z}] * gamma_0 * diffusion_coefficient[{X, Y, Z, k}] / (molar_mass_av[{X, Y, Z}] / Molar_mass[k]) + 0.5;
								pop_s[alpha][X + c_alpha[alpha][0]][Y + c_alpha[alpha][1]][Z + c_alpha[alpha][2]][k] += (mass_fraction[{X, Y, Z, k}] / tau_k) * pop_s_tot;
							}
						}
					}
				}
			}
		}
	}
	return;
}
/// ***************************************************** ///
/// UPDATE MASS FRACTION (WITH ADVECTION TERM) USING FD   ///
/// ***************************************************** ///
/* ****************Species Transport Equation****************
The Species mass balance equation in conservative form with correction velocity is given by:
    ∂(Yk)/∂t + ∂((ui + Vi_c)Yk) / ∂xi = ∂(Dk.Wk/W.∂Xk/∂xi)/∂xi + ωkdot/ρ
    Source: TNC book page 15. eq: 1.45
In terms of mass fraction:
        ∂(Yk)/∂t + ∂((ui + Vi_c)Yk) / ∂xi = ∂(Dk/W).∂(Yk*W/∂xi)/∂xi + ωkdot/ρ
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ∂(Yk)/∂t = - ∂((ui + Vi_c)Yk) / ∂xi + ∂(Dk/W).∂(Yk*W/∂xi)/∂xi + ωkdot/ρ !!!!!
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        advection term = - ∂((ui + Vi_c)Yk) / ∂xi
        diffusion term = ∂(Dk/W).∂(Yk*W/∂xi)/∂xi
        production term = ωkdot/ρ
where:
    ρ = density : Flow->rho_0
    Yk = mass fraction of Species k : mass_fraction[{X, Y, Z, k}]
    ui = velocity in the i-direction : Flow->velocity[{X, Y, Z, i}]
    Dk = diffusion coefficient of Species k : diffusion_coefficient[{X, Y, Z, k}]
    Wk = molar mass of Species k : Molar_mass[k] ; this value is constant
    W is the mean molecular weight : molar_mass_av[{X, Y, Z}]
    Xk = mole fraction of Species k : Xk = Yk * (W / Wk) . mass_fraction[{X, Y, Z, k}] * (molar_mass_av[{X, Y, Z}] / Molar_mass[k])
    ωkdot = production rate of Species k : Production[{X, Y, Z, k}]
    Vi_c = correction velocity
    Vi_c = Σ_k=1^N_sp(Dk.Wk/W.∂Xk/∂xi)

    FD_Euler : To compute base fluxes and initial updates.
    FD_Euler_Correction_Velocity : To adjust fluxes based on correction velocity.
    FD_HC_Euler : To finalize Species mass fraction updates using corrected fluxes.
*/
/* "FD_Euler" solves ∂(Yk)/∂t = - ∂((ui)Yk)/∂xi
        From the equation: ∂(Yk)/∂t = -∂((ui + Vi_c)Yk)/∂xi + ∂((Dk/W).(∂(Yk*W)/∂xi)/∂xi + ωkdot/ρ
        Modified: (FD_Euler) = -∂(Vi_c.Yk)/∂xi + ∂((Dk/W).∂(Yk*W)/∂xi)/∂xi + ωkdot/ρ

        This is the advection term in the Species mass balance equation.
        The advection term represents the convective transport of Species Yk due to the fluid Flow with velocity ui.
        This accounts for the movement of the Species along the spatial directions.
        ∂(ρYk)/∂t is implicitly solved as part of the overall solution process for the Species transport equation.
        The time derivative term reflects the change in mass fraction of Species Yk over time, and it is typically updated
        along with other terms in an iterative time-stepping scheme.
*/
void Species_solver::FD_Euler(int time, Flow_solver* Flow, Thermal_solver* Thermal, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		swap(previous_mass_fraction, mass_fraction);
		int k, X, Y, Z;
		double dY_x, dY_y, dY_z;
		for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (solid_species[{X, Y, Z}] == FALSE) {
						for (k = 0; k < Nb_spec; ++k) {
							double Y0 = previous_mass_fraction[{X, Y, Z, k}];
							double Y_xp = previous_mass_fraction[{X + 1, Y, Z, k}];
							double Y_xpp = previous_mass_fraction[{X + 2, Y, Z, k}];
							double Y_xn = previous_mass_fraction[{X - 1, Y, Z, k}];
							double Y_xnn = previous_mass_fraction[{X - 2, Y, Z, k}];
							double Y_yp = previous_mass_fraction[{X, Y + 1, Z, k}];
							double Y_ypp = previous_mass_fraction[{X, Y + 2, Z, k}];
							double Y_yn = previous_mass_fraction[{X, Y - 1, Z, k}];
							double Y_ynn = previous_mass_fraction[{X, Y - 2, Z, k}];
							double Y_zp = previous_mass_fraction[{X, Y, Z + 1, k}];
							double Y_zpp = previous_mass_fraction[{X, Y, Z + 2, k}];
							double Y_zn = previous_mass_fraction[{X, Y, Z - 1, k}];
							double Y_znn = previous_mass_fraction[{X, Y, Z - 2, k}];
							double vel_X = Flow->velocity[{X, Y, Z, 0}];
							double vel_Y = Flow->velocity[{X, Y, Z, 1}];
							double vel_Z = Flow->velocity[{X, Y, Z, 2}];

#if defined FD_UPWIND
							dY_x = FD::UPWIND1NONCONS(vel_X, Y_xn, Y0, Y_xp);
							dY_y = FD::UPWIND1NONCONS(vel_Y, Y_yn, Y0, Y_yp);
							dY_z = FD::UPWIND1NONCONS(vel_Z, Y_zn, Y0, Y_zp);
#endif  // defined
#if defined FD_UPWIND2
							dY_x = FD::UPWIND2NONCONS(vel_X, Y_xnn, Y_xn, Y0, Y_xp, Y_xpp);
							dY_y = FD::UPWIND2NONCONS(vel_Y, Y_ynn, Y_yn, Y0, Y_yp, Y_ypp);
							dY_z = FD::UPWIND2NONCONS(vel_Z, Y_znn, Y_zn, Y0, Y_zp, Y_zpp);
#endif  // defined
#if defined FD_CENTRAL
							dY_x = FD::CENTRALNONCONS(vel_X, Y_xn, Y_xp);
							dY_y = FD::CENTRALNONCONS(vel_Y, Y_yn, Y_yp);
							dY_z = FD::CENTRALNONCONS(vel_Z, Y_zn, Y_zp);
#endif  // defined
#if defined FD_CENTRAL4
							dY_x = FD::CENTRAL4NONCONS(vel_X, Y_xnn, Y_xn, Y0, Y_xp, Y_xpp);
							dY_y = FD::CENTRAL4NONCONS(vel_Y, Y_ynn, Y_yn, Y0, Y_yp, Y_ypp);
							dY_z = FD::CENTRAL4NONCONS(vel_Z, Y_znn, Y_zn, Y0, Y_zp, Y_zpp);
#endif  // defined
#if defined FD_WENO3
							dY_x = FD::WENO3NONCONS(vel_X, Y_xnn, Y_xn, Y0, Y_xp, Y_xpp);
							dY_y = FD::WENO3NONCONS(vel_Y, Y_ynn, Y_yn, Y0, Y_yp, Y_ypp);
							dY_z = FD::WENO3NONCONS(vel_Z, Y_znn, Y_zn, Y0, Y_zp, Y_zpp);
#endif  // defined
        // The code includes checks for solid boundaries (solid_species values equal to -1) to handle the advection term near solid objects. Central differencing is used when neighboring cells are solid.
							if (solid_species[{X + 1, Y, Z}] != -1 || solid_species[{X - 1, Y, Z}] != -1) {
								dY_x = FD::CENTRALNONCONS(vel_X, Y_xn, Y_xp);
							}
							if (solid_species[{X, Y + 1, Z}] != -1 || solid_species[{X, Y - 1, Z}] != -1) {
								dY_y = FD::CENTRALNONCONS(vel_Y, Y_yn, Y_yp);
							}
							if (solid_species[{X, Y, Z + 1}] != -1 || solid_species[{X, Y, Z - 1}] != -1) {
								dY_z = FD::CENTRALNONCONS(vel_Z, Y_zn, Y_zp);
							}
							if (Dimension < 2) dY_y = 0;
							if (Dimension < 3) dY_z = 0;
							mass_fraction[{X, Y, Z, k}] = Y0 - (dY_x + dY_y + dY_z);
						}
					}
				}
			}
		}
		return;
	}
}
/// ***************************************************** ///
/// ADD SPECIES FIELD DIFFUSION TO FD SOLVER (HC)       ///
/// ***************************************************** ///
/* "FD_HC_Euler" solves ∂(Yk)/∂t = ∂((Dk * Wk / W) ∂Xk/∂xi) / ∂xi + ωkdot / ρ
    where,
        Dk = diffusion coefficient of Species k
        Wk = molar mass of Species k (Molar_mass[k])
        W = mean molecular weight (molar_mass_av[{X, Y, Z}])
        Xk = mole fraction of Species k (mass_fraction[{X, Y, Z, k}] * (molar_mass_av[{X, Y, Z}] / Molar_mass[k]))
        ωkdot = production rate of Species k (Production[{X, Y, Z, k}])
        ρ = density (Flow->rho_0)
    Modified: (FD_Euler) + ∂(Vi_c.Yk)/∂xi = (FD_HC_Euler)
*/
void Species_solver::FD_HC_Euler(int time, Flow_solver* Flow, Thermal_solver* Thermal, Parallel_MPI* MPI_parallel, int VC) {
	if (MPI_parallel->processor_id != MASTER) {
		int k, kk, X, Y, Z;
		double Vd_left, Vd_right;
		for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (solid_species[{X, Y, Z}] == FALSE) {
						for (k = 0; k < Nb_spec; ++k) {
							Flux[X][Y][Z][k][0] = 0;
							Flux[X][Y][Z][k][1] = 0;
							Flux[X][Y][Z][k][2] = 0;
						}
						for (k = 0; k < Nb_spec; ++k) {
							double X0 = previous_mass_fraction[{X, Y, Z, k}] * molar_mass_av[{X, Y, Z}];
							double X_xp = previous_mass_fraction[{X + 1, Y, Z, k}] * molar_mass_av[{X + 1, Y, Z}];
							double X_xpp = previous_mass_fraction[{X + 2, Y, Z, k}] * molar_mass_av[{X + 2, Y, Z}];
							double X_xn = previous_mass_fraction[{X - 1, Y, Z, k}] * molar_mass_av[{X - 1, Y, Z}];
							double X_xnn = previous_mass_fraction[{X - 2, Y, Z, k}] * molar_mass_av[{X - 2, Y, Z}];
							double X_yp = previous_mass_fraction[{X, Y + 1, Z, k}] * molar_mass_av[{X, Y + 1, Z}];
							double X_ypp = previous_mass_fraction[{X, Y + 2, Z, k}] * molar_mass_av[{X, Y + 2, Z}];
							double X_yn = previous_mass_fraction[{X, Y - 1, Z, k}] * molar_mass_av[{X, Y - 1, Z}];
							double X_ynn = previous_mass_fraction[{X, Y - 2, Z, k}] * molar_mass_av[{X, Y - 2, Z}];
							double X_zp = previous_mass_fraction[{X, Y, Z + 1, k}] * molar_mass_av[{X, Y, Z + 1}];
							double X_zpp = previous_mass_fraction[{X, Y, Z + 2, k}] * molar_mass_av[{X, Y, Z + 2}];
							double X_zn = previous_mass_fraction[{X, Y, Z - 1, k}] * molar_mass_av[{X, Y, Z - 1}];
							double X_znn = previous_mass_fraction[{X, Y, Z - 2, k}] * molar_mass_av[{X, Y, Z - 2}];
							/* D*k is defined as D*k = rho * Dk / Mbar */
							double D = diffusion_coefficient[{X, Y, Z, k}] / molar_mass_av[{X, Y, Z}];
							double D_xp = diffusion_coefficient[{X + 1, Y, Z, k}] / molar_mass_av[{X + 1, Y, Z}];
							double D_xpp = diffusion_coefficient[{X + 2, Y, Z, k}] / molar_mass_av[{X + 2, Y, Z}];
							double D_xn = diffusion_coefficient[{X - 1, Y, Z, k}] / molar_mass_av[{X - 1, Y, Z}];
							double D_xnn = diffusion_coefficient[{X - 2, Y, Z, k}] / molar_mass_av[{X - 2, Y, Z}];
							double D_yp = diffusion_coefficient[{X, Y + 1, Z, k}] / molar_mass_av[{X, Y + 1, Z}];
							double D_ypp = diffusion_coefficient[{X, Y + 2, Z, k}] / molar_mass_av[{X, Y + 2, Z}];
							double D_yn = diffusion_coefficient[{X, Y - 1, Z, k}] / molar_mass_av[{X, Y - 1, Z}];
							double D_ynn = diffusion_coefficient[{X, Y - 2, Z, k}] / molar_mass_av[{X, Y - 2, Z}];
							double D_zp = diffusion_coefficient[{X, Y, Z + 1, k}] / molar_mass_av[{X, Y, Z + 1}];
							double D_zpp = diffusion_coefficient[{X, Y, Z + 2, k}] / molar_mass_av[{X, Y, Z + 2}];
							double D_zn = diffusion_coefficient[{X, Y, Z - 1, k}] / molar_mass_av[{X, Y, Z - 1}];
							double D_znn = diffusion_coefficient[{X, Y, Z - 2, k}] / molar_mass_av[{X, Y, Z - 2}];
#if defined FD_CENTRAL4 || defined FD_WENO3
							/* Compute left and right fluxes X-direction*/
							if (solid_species[{X + 1, Y, Z}] == -1 && solid_species[{X - 1, Y, Z}] == -1) {
								Vd_right = FD::CENTRAL4FLUX(X_xn, X0, X_xp, X_xpp, D_xn, D, D_xp, D_xpp);  //(Dk * Wk) * ∂Xk/∂xi at the right interface
								Vd_left = FD::CENTRAL4FLUX(X_xnn, X_xn, X0, X_xp, D_xnn, D_xn, D, D_xp);   //(Dk * Wk) * ∂Xk/∂xi at the left interface
							}
							if (solid_species[{X + 1, Y, Z}] != -1 || solid_species[{X - 1, Y, Z}] != -1) {
								Vd_right = FD::CENTRAL2FLUX(X0, X_xp, D, D_xp);
								Vd_left = FD::CENTRAL2FLUX(X_xn, X0, D_xn, D);
							}
#endif
#if defined FD_UPWIND || defined FD_UPWIND2 || defined FD_CENTRAL
							Vd_right = FD::CENTRAL2FLUX(X0, X_xp, D, D_xp);
							Vd_left = FD::CENTRAL2FLUX(X_xn, X0, D_xn, D);
#endif
							// Compute net flux  X-direction
							Flux[X][Y][Z][k][0] += (Vd_right - Vd_left);  // Update flux term: ∂((Dk * Wk / W) ∂Xk/∂xi) / ∂xi represents the divergence of the diffusive flux
							if (VC == 1) {
								for (kk = 0; kk < Nb_spec; ++kk) {
									Flux[X][Y][Z][kk][0] -= (previous_mass_fraction[{X + 1, Y, Z, kk}] * Vd_right - previous_mass_fraction[{X, Y, Z, kk}] * Vd_left);
								}
							}
							/* Compute left and right fluxes Y-direction*/
#if defined FD_CENTRAL4 || defined FD_WENO3
							if (solid_species[{X, Y + 1, Z}] == -1 && solid_species[{X, Y - 1, Z}] == -1) {
								Vd_right = FD::CENTRAL4FLUX(X_yn, X0, X_yp, X_ypp, D_yn, D, D_yp, D_ypp);
								Vd_left = FD::CENTRAL4FLUX(X_ynn, X_yn, X0, X_yp, D_ynn, D_yn, D, D_yp);
							}
							if (solid_species[{X, Y + 1, Z}] != -1 || solid_species[{X, Y - 1, Z}] != -1) {
								Vd_right = FD::CENTRAL2FLUX(X0, X_yp, D, D_yp);
								Vd_left = FD::CENTRAL2FLUX(X_yn, X0, D_yn, D);
							}
#endif
#if defined FD_UPWIND || defined FD_UPWIND2 || defined FD_CENTRAL
							Vd_right = FD::CENTRAL2FLUX(X0, X_yp, D, D_yp);
							Vd_left = FD::CENTRAL2FLUX(X_yn, X0, D_yn, D);
#endif
							Flux[X][Y][Z][k][1] += (Vd_right - Vd_left);
							if (VC == 1) {
								for (kk = 0; kk < Nb_spec; ++kk) {
									Flux[X][Y][Z][kk][1] -= (previous_mass_fraction[{X, Y + 1, Z, kk}] * Vd_right - previous_mass_fraction[{X, Y, Z, kk}] * Vd_left);
								}
							}
							/* Compute left and right fluxes Z-direction*/
#if defined FD_CENTRAL4 || defined FD_WENO3
							if (solid_species[{X, Y, Z + 1}] == -1 && solid_species[{X, Y, Z - 1}] == -1) {
								Vd_right = FD::CENTRAL4FLUX(X_zn, X0, X_zp, X_zpp, D_zn, D, D_zp, D_zpp);
								Vd_left = FD::CENTRAL4FLUX(X_znn, X_zn, X0, X_zp, D_znn, D_zn, D, D_zp);
							}
							if (solid_species[{X, Y, Z + 1}] != -1 || solid_species[{X, Y, Z - 1}] != -1) {
								Vd_right = FD::CENTRAL2FLUX(X0, X_zp, D, D_zp);
								Vd_left = FD::CENTRAL2FLUX(X_zn, X0, D_zn, D);
							}
#endif
#if defined FD_UPWIND || defined FD_UPWIND2 || defined FD_CENTRAL
							Vd_right = FD::CENTRAL2FLUX(Z0, Z_zp, D, D_zp);
							Vd_left = FD::CENTRAL2FLUX(Z_zn, Z0, D_zn, D);
#endif
							Flux[X][Y][Z][k][2] += (Vd_right - Vd_left);
							if (VC == 1) {
								for (kk = 0; kk < Nb_spec; ++kk) {
									Flux[X][Y][Z][kk][2] -= (previous_mass_fraction[{X, Y, Z + 1, kk}] * Vd_right - previous_mass_fraction[{X, Y, Z, kk}] * Vd_left);
								}
							}
						}
						for (k = 0; k < Nb_spec; ++k) {
							if (Dimension < 2) Flux[X][Y][Z][k][1] = 0;
							if (Dimension < 3) Flux[X][Y][Z][k][2] = 0;
							double flux_VY = (Flux[X][Y][Z][k][0] + Flux[X][Y][Z][k][1] + Flux[X][Y][Z][k][2] + Production[{X, Y, Z, k}]) / (Flow->rho_0 * Flow->density[{X, Y, Z}]);
							mass_fraction[{X, Y, Z, k}] += flux_VY;
							if (mass_fraction[{X, Y, Z, k}] < 0) mass_fraction[{X, Y, Z, k}] = 0;
#if defined LMNA_solver
							Flow->divU[{X, Y, Z}] += (molar_mass_av[{X, Y, Z}] / Molar_mass[k]) * flux_VY;
#endif  // defined
						}
					}
				}
			}
		}
		return;
	}
}
/// ******************************************************* ///
/// ADD SPECIES FIELD CORRECTION VELOCITY TO FD SOLVER      ///
/// ******************************************************* ///
/* "FD_Euler_Correction_Velocity" solves ∂(Yk)/∂t = -∂(Vi_c.Yk)/∂xi => [∂(Yk)/∂t = -∂(Σ_k=1^N(Dk*Wk/W).∂Xk/∂xi)*Yk / ∂xi]
    From the equation: ∂(Yk)/∂t = -∂((uiYk) / ∂xi  - ∂((Vi_c)Yk)/∂xi + ∂(Dk.Wk/W.∂Xk/∂xi)/∂xi + ωkdot/ρ
    Modified: (FD_Euler) = FD_Euler_Correction_Velocity + (FD_Euler_Hirschfelder)

    Vi_c = Σ_k=1^N((Dk*Wk/W).∂(Yk*W/Wk)/∂xi)
    =∂(Vi_c)/∂xi = (Σ_k=1^N(Dk*Wk/W * ∂(Yk*W/Wk)/∂xi))

    This is the correction velocity term in the Species mass balance equation.
    The correction velocity term represents the effect of the concentration gradient of the Species on the fluid velocity.
    It accounts for how the Species concentration affects the fluid velocity.
    The correction velocity term is the product of the diffusion coefficient, the Species mass fraction, and the gradient of the Species mass fraction.

    Here Diffusion coefficient D = λ / (ρ * Cp_k) : λ = Thermal conductivity, ρ = density, Cp_k = specific heat capacity of Species k
    The equation is : ∂(Dk.Wk/W.∂Xk/∂xi)/∂xi => changing Xk to Yk => Xk = Yk*W/Wk ; Wk is constant, W is not constant
*/
void Species_solver::FD_Euler_Correction_Velocity(int time, Flow_solver* Flow, Thermal_solver* Thermal, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		int k, kk, X, Y, Z;
		double Vd_left, Vd_right;
		// Initialize variables for summation
		// Summation for the correction velocity
		for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (solid_species[{X, Y, Z}] == FALSE) {
						// Initialize fluxes for current cell and Species
						for (k = 0; k < Nb_spec; ++k) {
							Flux[X][Y][Z][k][0] = 0;
							Flux[X][Y][Z][k][1] = 0;
							Flux[X][Y][Z][k][2] = 0;
						}
						for (k = 0; k < Nb_spec; ++k) {
							// Get the current mass fraction
							// Current and neighboring mass fractions
							double X0 = previous_mass_fraction[{X, Y, Z, k}] * molar_mass_av[{X, Y, Z}];
							double X_xp = previous_mass_fraction[{X + 1, Y, Z, k}] * molar_mass_av[{X + 1, Y, Z}];
							double X_xpp = previous_mass_fraction[{X + 2, Y, Z, k}] * molar_mass_av[{X + 2, Y, Z}];
							double X_xn = previous_mass_fraction[{X - 1, Y, Z, k}] * molar_mass_av[{X - 1, Y, Z}];
							double X_xnn = previous_mass_fraction[{X - 2, Y, Z, k}] * molar_mass_av[{X - 2, Y, Z}];
							double X_yp = previous_mass_fraction[{X, Y + 1, Z, k}] * molar_mass_av[{X, Y + 1, Z}];
							double X_ypp = previous_mass_fraction[{X, Y + 2, Z, k}] * molar_mass_av[{X, Y + 2, Z}];
							double X_yn = previous_mass_fraction[{X, Y - 1, Z, k}] * molar_mass_av[{X, Y - 1, Z}];
							double X_ynn = previous_mass_fraction[{X, Y - 2, Z, k}] * molar_mass_av[{X, Y - 2, Z}];
							double X_zp = previous_mass_fraction[{X, Y, Z + 1, k}] * molar_mass_av[{X, Y, Z + 1}];
							double X_zpp = previous_mass_fraction[{X, Y, Z + 2, k}] * molar_mass_av[{X, Y, Z + 2}];
							double X_zn = previous_mass_fraction[{X, Y, Z - 1, k}] * molar_mass_av[{X, Y, Z - 1}];
							double X_znn = previous_mass_fraction[{X, Y, Z - 2, k}] * molar_mass_av[{X, Y, Z - 2}];
							/* D*k is defined as D*k = rho * Dk / Mbar */
							double D = diffusion_coefficient[{X, Y, Z, k}] / molar_mass_av[{X, Y, Z}];
							double D_xp = diffusion_coefficient[{X + 1, Y, Z, k}] / molar_mass_av[{X + 1, Y, Z}];
							double D_xpp = diffusion_coefficient[{X + 2, Y, Z, k}] / molar_mass_av[{X + 2, Y, Z}];
							double D_xn = diffusion_coefficient[{X - 1, Y, Z, k}] / molar_mass_av[{X - 1, Y, Z}];
							double D_xnn = diffusion_coefficient[{X - 2, Y, Z, k}] / molar_mass_av[{X - 2, Y, Z}];
							double D_yp = diffusion_coefficient[{X, Y + 1, Z, k}] / molar_mass_av[{X, Y + 1, Z}];
							double D_ypp = diffusion_coefficient[{X, Y + 2, Z, k}] / molar_mass_av[{X, Y + 2, Z}];
							double D_yn = diffusion_coefficient[{X, Y - 1, Z, k}] / molar_mass_av[{X, Y - 1, Z}];
							double D_ynn = diffusion_coefficient[{X, Y - 2, Z, k}] / molar_mass_av[{X, Y - 2, Z}];
							double D_zp = diffusion_coefficient[{X, Y, Z + 1, k}] / molar_mass_av[{X, Y, Z + 1}];
							double D_zpp = diffusion_coefficient[{X, Y, Z + 2, k}] / molar_mass_av[{X, Y, Z + 2}];
							double D_zn = diffusion_coefficient[{X, Y, Z - 1, k}] / molar_mass_av[{X, Y, Z - 1}];
							double D_znn = diffusion_coefficient[{X, Y, Z - 2, k}] / molar_mass_av[{X, Y, Z - 2}];

#if defined FD_CENTRAL4 || defined FD_WENO3
							/* Compute left and right fluxes X-direction*/
							if (solid_species[{X + 1, Y, Z}] == -1 && solid_species[{X - 1, Y, Z}] == -1) {
								Vd_right = FD::CENTRAL4FLUX(X_xn, X0, X_xp, X_xpp, D_xn, D, D_xp, D_xpp);  //(Dk * Wk) * ∂Xk/∂xi at the right interface
								Vd_left = FD::CENTRAL4FLUX(X_xnn, X_xn, X0, X_xp, D_xnn, D_xn, D, D_xp);   //(Dk * Wk) * ∂Xk/∂xi at the left interface
							}
							if (solid_species[{X + 1, Y, Z}] != -1 || solid_species[{X - 1, Y, Z}] != -1) {
								Vd_right = FD::CENTRAL2FLUX(X0, X_xp, D, D_xp);
								Vd_left = FD::CENTRAL2FLUX(X_xn, X0, D_xn, D);
							}
#endif
#if defined FD_UPWIND || defined FD_UPWIND2 || defined FD_CENTRAL
							Vd_right = FD::CENTRAL2FLUX(X0, X_xp, D, D_xp);
							Vd_left = FD::CENTRAL2FLUX(X_xn, X0, D_xn, D);
#endif
							// Compute net flux  X-direction
							for (kk = 0; kk < Nb_spec; ++kk) {
								double Yk_right = 0.5 * (previous_mass_fraction[{X + 1, Y, Z, kk}] + previous_mass_fraction[{X, Y, Z, kk}]);
								double Yk_left = 0.5 * (previous_mass_fraction[{X, Y, Z, kk}] + previous_mass_fraction[{X - 1, Y, Z, kk}]);
								Flux[X][Y][Z][kk][0] -= (Vd_right * Yk_right - Vd_left * Yk_left);
							}
							/* Compute left and right fluxes Y-direction*/
#if defined FD_CENTRAL4 || defined FD_WENO3
							if (solid_species[{X, Y + 1, Z}] == -1 && solid_species[{X, Y - 1, Z}] == -1) {
								Vd_right = FD::CENTRAL4FLUX(X_yn, X0, X_yp, X_ypp, D_yn, D, D_yp, D_ypp);
								Vd_left = FD::CENTRAL4FLUX(X_ynn, X_yn, X0, X_yp, D_ynn, D_yn, D, D_yp);
							}
							if (solid_species[{X, Y + 1, Z}] != -1 || solid_species[{X, Y - 1, Z}] != -1) {
								Vd_right = FD::CENTRAL2FLUX(X0, X_yp, D, D_yp);
								Vd_left = FD::CENTRAL2FLUX(X_yn, X0, D_yn, D);
							}
#endif
#if defined FD_UPWIND || defined FD_UPWIND2 || defined FD_CENTRAL
							Vd_right = FD::CENTRAL2FLUX(X0, X_yp, D, D_yp);
							Vd_left = FD::CENTRAL2FLUX(X_yn, X0, D_yn, D);
#endif
							for (kk = 0; kk < Nb_spec; ++kk) {
								double Yk_right = 0.5 * (previous_mass_fraction[{X, Y + 1, Z, kk}] + previous_mass_fraction[{X, Y, Z, kk}]);
								double Yk_left = 0.5 * (previous_mass_fraction[{X, Y, Z, kk}] + previous_mass_fraction[{X, Y - 1, Z, kk}]);
								Flux[X][Y][Z][kk][1] -= (Vd_right * Yk_right - Vd_left * Yk_left);
							}
							/* Compute left and right fluxes Z-direction*/
#if defined FD_CENTRAL4 || defined FD_WENO3
							if (solid_species[{X, Y, Z + 1}] == -1 && solid_species[{X, Y, Z - 1}] == -1) {
								Vd_right = FD::CENTRAL4FLUX(X_zn, X0, X_zp, X_zpp, D_zn, D, D_zp, D_zpp);
								Vd_left = FD::CENTRAL4FLUX(X_znn, X_zn, X0, X_zp, D_znn, D_zn, D, D_zp);
							}
							if (solid_species[{X, Y, Z + 1}] != -1 || solid_species[{X, Y, Z - 1}] != -1) {
								Vd_right = FD::CENTRAL2FLUX(X0, X_zp, D, D_zp);
								Vd_left = FD::CENTRAL2FLUX(X_zn, X0, D_zn, D);
							}
#endif
#if defined FD_UPWIND || defined FD_UPWIND2 || defined FD_CENTRAL
							Vd_right = FD::CENTRAL2FLUX(Z0, Z_zp, D, D_zp);
							Vd_left = FD::CENTRAL2FLUX(Z_zn, Z0, D_zn, D);
#endif
							for (kk = 0; kk < Nb_spec; ++kk) {
								double Yk_right = 0.5 * (previous_mass_fraction[{X, Y, Z + 1, kk}] + previous_mass_fraction[{X, Y, Z, kk}]);
								double Yk_left = 0.5 * (previous_mass_fraction[{X, Y, Z, kk}] + previous_mass_fraction[{X, Y, Z - 1, kk}]);
								Flux[X][Y][Z][kk][2] -= (Vd_right * Yk_right - Vd_left * Yk_left);
							}
						}
						/*	// Compute the correction velocity using central differences
						    Vi_c_x_right = FD::CENTRAL4FLUX(X_xn, X0, X_xp, X_xpp, D_xn, D, D_xp, D_xpp);
						    Vi_c_x_left = FD::CENTRAL4FLUX(X_xnn, X_xn, X0, X_xp, D_xnn, D_xn, D, D_xp);
						    Vi_c_y_right = FD::CENTRAL4FLUX(X_yn, X0, X_yp, X_ypp, D_yn, D, D_yp, D_ypp);
						    Vi_c_y_left = FD::CENTRAL4FLUX(X_ynn, X_yn, X0, X_yp, D_ynn, D_yn, D, D_yp);
						    Vi_c_z_right = FD::CENTRAL4FLUX(X_zn, X0, X_zp, X_zpp, D_zn, D, D_zp, D_zpp);
						    Vi_c_z_left = FD::CENTRAL4FLUX(X_znn, X_zn, X0, X_zp, D_znn, D_zn, D, D_zp);

						    // Interpolation of Mass fraction
						    double Yk_x_right = 0.5 * (previous_mass_fraction[{X + 1, Y, Z, k}] + previous_mass_fraction[{X, Y, Z, k}]);
						    double Yk_x_left = 0.5 * (previous_mass_fraction[{X, Y, Z, k}] + previous_mass_fraction[{X - 1, Y, Z, k}]);
						    double Yk_y_right = 0.5 * (previous_mass_fraction[{X, Y + 1, Z, k}] + previous_mass_fraction[{X, Y, Z, k}]);
						    double Yk_y_left = 0.5 * (previous_mass_fraction[{X, Y, Z, k}] + previous_mass_fraction[{X, Y - 1, Z, k}]);
						    double Yk_z_right = 0.5 * (previous_mass_fraction[{X, Y, Z + 1, k}] + previous_mass_fraction[{X, Y, Z, k}]);
						    double Yk_z_left = 0.5 * (previous_mass_fraction[{X, Y, Z, k}] + previous_mass_fraction[{X, Y, Z - 1, k}]);

						    //Product of Correction Velocity and Interpolated Mass Fraction:
						    double div_Vi_c_x_Yk = Vi_c_x_right * Yk_x_right - Vi_c_x_left * Yk_x_left;
						    double div_Vi_c_y_Yk = Vi_c_y_right * Yk_y_right - Vi_c_y_left * Yk_y_left;
						    double div_Vi_c_z_Yk = Vi_c_z_right * Yk_z_right - Vi_c_z_left * Yk_z_left;

						    // Summing up for all Species
						    Flux[X][Y][Z][k][0] -= div_Vi_c_x_Yk;  // Σ_k=1^N(Dk*Wk/W).∂Xk/∂x*Yk
						    Flux[X][Y][Z][k][1] -= div_Vi_c_y_Yk;  // Σ_k=1^N(Dk*Wk/W).∂Yk/∂y*Yk
						    Flux[X][Y][Z][k][2] -= div_Vi_c_z_Yk;  // Σ_k=1^N(Dk*Wk/W).∂Zk/∂z*Yk
						}*/

						// Calculate the divergence of the correction velocity flux for each Species
						for (int k = 0; k < Nb_spec; ++k) {
							// Initialize the fluxes for dimensions higher than the current problem dimensionality to zero
							if (Dimension < 2) Flux[X][Y][Z][k][1] = 0;  // Set Y-direction flux to 0 if the problem is 1D
							if (Dimension < 3) Flux[X][Y][Z][k][2] = 0;  // Set Z-direction flux to 0 if the problem is 2D
							double flux_VY = (Flux[X][Y][Z][k][0] + Flux[X][Y][Z][k][1] + Flux[X][Y][Z][k][2]) / (Flow->rho_0 * Flow->density[{X, Y, Z}]);
							mass_fraction[{X, Y, Z, k}] += flux_VY;
							//  Ensure that the updated mass fraction is non-negative; if it's negative, set it to zero
							if (mass_fraction[{X, Y, Z, k}] < 0) mass_fraction[{X, Y, Z, k}] = 0;
#if defined LMNA_solver
							// This term involves the ratio of the average molar mass to the molar mass of the Species k, scaled by the computed flux
							Flow->divU[{X, Y, Z}] += (molar_mass_av[{X, Y, Z}] / Molar_mass[k]) * flux_VY;
#endif  // defined
						}
					}
				}
			}
		}
		return;
	}
}
// onscreen function
void Species_solver::Check_Mass_Fraction_Conservation(Parallel_MPI* MPI_parallel) {
	bool is_conserved = true;

	// Loop through the grid cells
	for (int X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
		for (int Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
			for (int Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
				if (solid_species[{X, Y, Z}] == FALSE) {
					double sum_mass_fraction = 0.0;

					// Sum the mass fractions of all Species in the current cell
					for (int k = 0; k < Nb_spec; ++k) {
						sum_mass_fraction += mass_fraction[{X, Y, Z, k}];
					}

					// Check if the sum is 1 (allowing for some numerical tolerance)
					if (std::abs(sum_mass_fraction - 1.0) > 1e-6) {
						is_conserved = false;
						std::cout << "Mass fraction not conserved at cell (" << X << ", " << Y << ", " << Z << "): "
								  << "Sum = " << sum_mass_fraction << std::endl;
					}
				}
			}
		}
	}

	// Print a message if mass fraction is conserved in all cells
	if (is_conserved) {
		std::cout << "Mass fraction is conserved in all cells." << std::endl;
	}
}
/// ***************************************************** ///
/// UPDATE MASS FRACTION (WITH DIFFUSION TERM) USING FD   ///
/// ***************************************************** ///
/*	NOT USED IN REACTIVE FLOW
    solve the diffusion term or (a second-order spatial derivative of the mass fraction (∇^2.Yk)
of the Species mass balance equation for a single Species (since k=0)

FD_Euler_diffusion solves ∇·(ρVkYk) = ωk
where
- ∇·(ρVkYk) is the divergence of the mass flux due to diffusion.
- V_k is the mass flux vector for the k-th Species.
- Y_k is the mass fraction of the k-th Species.
- ωk is the source term due to chemical reactions.
*/
void Species_solver::FD_Euler_diffusion(int time, Flow_solver* Flow, Thermal_solver* Thermal, Parallel_MPI* MPI_parallel, int VC) {
	if (MPI_parallel->processor_id != MASTER) {
		int X, Y, Z;
		double grad2_S, q_phi_s, Hp, Hn;
		for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (solid_species[{X, Y, Z}] == FALSE) {
						// q_phi_s = (1.0 - Phase->phase[{X,Y,Z}]);
						q_phi_s = 1.0;

#if defined FD_UPWIND || defined FD_UPWIND2 || defined FD_CENTRAL
						grad2_S = 0;
						grad2_S = FD::CENTRAL2FLUX(previous_mass_fraction[{X, Y, Z, 0}], previous_mass_fraction[{X + 1, Y, Z, 0}], diffusion_coefficient[{X, Y, Z, 0}] * q_phi_s, diffusion_coefficient[{X + 1, Y, Z, 0}] * q_phi_s)
						          - FD::CENTRAL2FLUX(previous_mass_fraction[{X - 1, Y, Z, 0}], previous_mass_fraction[{X, Y, Z, 0}], diffusion_coefficient[{X - 1, Y, Z, 0}] * q_phi_s, diffusion_coefficient[{X, Y, Z, 0}] * q_phi_s);
						if (Dimension > 1) {
							grad2_S += (FD::CENTRAL2FLUX(previous_mass_fraction[{X, Y, Z, 0}], previous_mass_fraction[{X, Y + 1, Z, 0}], diffusion_coefficient[{X, Y, Z, 0}] * q_phi_s, diffusion_coefficient[{X, Y + 1, Z, 0}] * q_phi_s)
							            - FD::CENTRAL2FLUX(previous_mass_fraction[{X, Y - 1, Z, 0}], previous_mass_fraction[{X, Y, Z, 0}], diffusion_coefficient[{X, Y - 1, Z, 0}] * q_phi_s, diffusion_coefficient[{X, Y, Z, 0}] * q_phi_s));
						}
						if (Dimension > 2) {
							grad2_S += (FD::CENTRAL2FLUX(previous_mass_fraction[{X, Y, Z, 0}], previous_mass_fraction[{X, Y, Z + 1, 0}], diffusion_coefficient[{X, Y, Z, 0}] * q_phi_s, diffusion_coefficient[{X, Y, Z + 1, 0}] * q_phi_s)
							            - FD::CENTRAL2FLUX(previous_mass_fraction[{X, Y, Z - 1, 0}], previous_mass_fraction[{X, Y, Z, 0}], diffusion_coefficient[{X, Y, Z - 1, 0}] * q_phi_s, diffusion_coefficient[{X, Y, Z, 0}] * q_phi_s));
						}
#endif

#if defined FD_CENTRAL4 || defined FD_WENO3
						///  x-direction
						grad2_S = 0;
						if (solid_species[{X + 1, Y, Z}] == -1 && solid_species[{X - 1, Y, Z}] == -1) {
							Hp = FD::CENTRAL4FLUX(previous_mass_fraction[{X - 1, Y, Z, 0}], previous_mass_fraction[{X, Y, Z, 0}], previous_mass_fraction[{X + 1, Y, Z, 0}], previous_mass_fraction[{X + 2, Y, Z, 0}],
							                      diffusion_coefficient[{X - 1, Y, Z, 0}] * q_phi_s, diffusion_coefficient[{X, Y, Z, 0}] * q_phi_s, diffusion_coefficient[{X + 1, Y, Z, 0}] * q_phi_s, diffusion_coefficient[{X + 2, Y, Z, 0}] * q_phi_s);

							Hn = FD::CENTRAL4FLUX(previous_mass_fraction[{X - 2, Y, Z, 0}], previous_mass_fraction[{X - 1, Y, Z, 0}], previous_mass_fraction[{X, Y, Z, 0}], previous_mass_fraction[{X + 1, Y, Z, 0}],
							                      diffusion_coefficient[{X - 2, Y, Z, 0}] * q_phi_s, diffusion_coefficient[{X - 1, Y, Z, 0}] * q_phi_s, diffusion_coefficient[{X, Y, Z, 0}] * q_phi_s, diffusion_coefficient[{X + 1, Y, Z, 0}] * q_phi_s);

							grad2_S += (Hp - Hn);
						}
						if (solid_species[{X + 1, Y, Z}] != -1 || solid_species[{X - 1, Y, Z}] != -1) {
							grad2_S += (FD::CENTRAL2FLUX(previous_mass_fraction[{X, Y, Z, 0}], previous_mass_fraction[{X + 1, Y, Z, 0}], diffusion_coefficient[{X, Y, Z, 0}] * q_phi_s, diffusion_coefficient[{X + 1, Y, Z, 0}] * q_phi_s)
							            - FD::CENTRAL2FLUX(previous_mass_fraction[{X - 1, Y, Z, 0}], previous_mass_fraction[{X, Y, Z, 0}], diffusion_coefficient[{X - 1, Y, Z, 0}] * q_phi_s, diffusion_coefficient[{X, Y, Z, 0}] * q_phi_s));
						}
						///  y-direction
						if (Dimension > 1) {
							if (solid_species[{X, Y + 1, Z}] == -1 && solid_species[{X, Y - 1, Z}] == -1) {
								Hp = FD::CENTRAL4FLUX(previous_mass_fraction[{X, Y - 1, Z, 0}], previous_mass_fraction[{X, Y, Z, 0}], previous_mass_fraction[{X, Y + 1, Z, 0}], previous_mass_fraction[{X, Y + 2, Z, 0}],
								                      diffusion_coefficient[{X, Y - 1, Z, 0}] * q_phi_s, diffusion_coefficient[{X, Y, Z, 0}] * q_phi_s, diffusion_coefficient[{X, Y + 1, Z, 0}] * q_phi_s, diffusion_coefficient[{X, Y + 2, Z, 0}] * q_phi_s);

								Hn = FD::CENTRAL4FLUX(previous_mass_fraction[{X, Y - 2, Z, 0}], previous_mass_fraction[{X, Y - 1, Z, 0}], previous_mass_fraction[{X, Y, Z, 0}], previous_mass_fraction[{X, Y + 1, Z, 0}],
								                      diffusion_coefficient[{X, Y - 2, Z, 0}] * q_phi_s, diffusion_coefficient[{X, Y - 1, Z, 0}] * q_phi_s, diffusion_coefficient[{X, Y, Z, 0}] * q_phi_s, diffusion_coefficient[{X, Y + 1, Z, 0}] * q_phi_s);

								grad2_S += (Hp - Hn);
							}
							if (solid_species[{X, Y + 1, Z}] != -1 || solid_species[{X, Y - 1, Z}] != -1) {
								grad2_S += (FD::CENTRAL2FLUX(previous_mass_fraction[{X, Y, Z, 0}], previous_mass_fraction[{X, Y + 1, Z, 0}], diffusion_coefficient[{X, Y, Z, 0}] * q_phi_s, diffusion_coefficient[{X, Y + 1, Z, 0}] * q_phi_s)
								            - FD::CENTRAL2FLUX(previous_mass_fraction[{X, Y - 1, Z, 0}], previous_mass_fraction[{X, Y, Z, 0}], diffusion_coefficient[{X, Y - 1, Z, 0}] * q_phi_s, diffusion_coefficient[{X, Y, Z, 0}] * q_phi_s));
							}
						}
						///  z-direction
						if (Dimension > 2) {
							if (solid_species[{X, Y, Z + 1}] == -1 && solid_species[{X, Y, Z - 1}] == -1) {
								Hp = FD::CENTRAL4FLUX(previous_mass_fraction[{X, Y, Z - 1, 0}], previous_mass_fraction[{X, Y, Z, 0}], previous_mass_fraction[{X, Y, Z + 1, 0}], previous_mass_fraction[{X, Y, Z + 2, 0}],
								                      diffusion_coefficient[{X, Y, Z - 1, 0}] * q_phi_s, diffusion_coefficient[{X, Y, Z, 0}] * q_phi_s, diffusion_coefficient[{X, Y, Z + 1, 0}] * q_phi_s, diffusion_coefficient[{X, Y, Z + 2, 0}] * q_phi_s);

								Hn = FD::CENTRAL4FLUX(previous_mass_fraction[{X, Y, Z - 2, 0}], previous_mass_fraction[{X, Y, Z - 1, 0}], previous_mass_fraction[{X, Y, Z, 0}], previous_mass_fraction[{X, Y, Z + 1, 0}],
								                      diffusion_coefficient[{X, Y, Z - 2, 0}] * q_phi_s, diffusion_coefficient[{X, Y, Z - 1, 0}] * q_phi_s, diffusion_coefficient[{X, Y, Z, 0}] * q_phi_s, diffusion_coefficient[{X, Y, Z + 1, 0}] * q_phi_s);

								grad2_S += (Hp - Hn);
							}
							if (solid_species[{X, Y, Z + 1}] != -1 || solid_species[{X, Y, Z - 1}] != -1) {
								grad2_S += (FD::CENTRAL2FLUX(previous_mass_fraction[{X, Y, Z, 0}], previous_mass_fraction[{X, Y, Z + 1, 0}], diffusion_coefficient[{X, Y, Z, 0}] * q_phi_s, diffusion_coefficient[{X, Y, Z + 1, 0}] * q_phi_s)
								            - FD::CENTRAL2FLUX(previous_mass_fraction[{X, Y, Z - 1, 0}], previous_mass_fraction[{X, Y, Z, 0}], diffusion_coefficient[{X, Y, Z - 1, 0}] * q_phi_s, diffusion_coefficient[{X, Y, Z, 0}] * q_phi_s));
							}
						}
#endif

						mass_fraction[{X, Y, Z, 0}] += grad2_S + Production[{X, Y, Z, 0}];
						//    if (mass_fraction[{X,Y,Z,k}]<0) mass_fraction[{X,Y,Z,k}] = 0;
					}
				}
			}
		}
	}
}
/// ***************************************************** ///
/// ADD SPECIES FIELD DIFFUSION TO FD SOLVER (Fick)         ///
/// ***************************************************** ///
/* 3: "FD_Fick_Euler" solves ∂(Yk)/∂t = ∂(Dk.∇Yk)/∂xi => [∂(Dk.∇Yk)/∂xi]
                            ∂(Dk.∇Yk)/∂xi => Dk.∇²Yk + ∇Dk.∇Yk
    replacing mole fraction Xk with mass fraction = Yk*W/Wk ; Wk is constant, W is not constant
    The equation is : ∂(Dk.∇Yk)/∂xi => Dk.∇²Yk + ∇Dk.∇Yk
    Fick is used for binary diffusion, where the diffusion coefficient is a function of the Species mass fraction.
    Fick's law of diffusion states that the flux of a Species is proportional to the gradient of the Species mass fraction.
    The diffusion coefficient is a function of the Species mass fraction.
    The diffusion coefficient is the product of the Thermal conductivity, the Species mass fraction, and the gradient of the Species mass fraction.
    The equation is : ∂(Dk.∇Yk)/∂xi => Dk.∇²Yk + ∇Dk.∇Yk => [∂(Dk.∇Yk)/∂xi]

    From the equation: ∂(Yk)/∂t = ∂((uiYk) / ∂xi  + ∂((Vi_c)Yk)/∂xi + ∂(Dk.Wk/W.∂Xk/∂xi)/∂xi + ωkdot/ρ
*/
void Species_solver::FD_Fick_Euler(int time, Flow_solver* Flow, Thermal_solver* Thermal, Parallel_MPI* MPI_parallel, int VC) {
	if (MPI_parallel->processor_id != MASTER) {
		int k, kk, X, Y, Z;
		double Vd_left, Vd_right;
		for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (solid_species[{X, Y, Z}] == FALSE) {
						for (k = 0; k < Nb_spec; ++k) {
							Flux[X][Y][Z][k][0] = 0;
							Flux[X][Y][Z][k][1] = 0;
							Flux[X][Y][Z][k][2] = 0;
						}
						for (k = 0; k < Nb_spec; ++k) {
							/// COMPUTE HC FLUX X-DIRECTION
							Vd_right = (diffusion_coefficient[{X + 1, Y, Z, k}]) * (previous_mass_fraction[{X + 1, Y, Z, k}] - previous_mass_fraction[{X, Y, Z, k}]);
							Vd_left = (diffusion_coefficient[{X, Y, Z, k}]) * (previous_mass_fraction[{X, Y, Z, k}] - previous_mass_fraction[{X - 1, Y, Z, k}]);
							// V_right = -Dpk.∇Yp : diff_coefficient_right * (mass_fraction_right - mass_fraction_left)
							// V_left = -Dpk.∇Yp : diff_coefficient_left * (mass_fraction_right - mass_fraction_left)
							Flux[X][Y][Z][k][0] += (Vd_right - Vd_left);
							if (VC == 1) {
								for (kk = 0; kk < Nb_spec; ++kk) {
									Flux[X][Y][Z][kk][0] -= (previous_mass_fraction[{X + 1, Y, Z, kk}] * Vd_right - previous_mass_fraction[{X, Y, Z, kk}] * Vd_left);
									// updating the flux term based on the differences in mass fractions (previous_mass_fraction) multiplied by the differences in
									// diffusion velocities (Vd_right and Vd_left). While the code doesn't directly include the denominator term Yp,
									// the multiplication by the differences in mass fractions captures the essence of the gradient term ∇Yp.
									// The denominator term is implicitly included in the calculation of the diffusion velocities (Vd_right and Vd_left).
								}
							}
							/// COMPUTE HC FLUX Y-DIRECTION
							Vd_right = (diffusion_coefficient[{X, Y + 1, Z, k}]) * (previous_mass_fraction[{X, Y + 1, Z, k}] - previous_mass_fraction[{X, Y, Z, k}]);
							Vd_left = (diffusion_coefficient[{X, Y, Z, k}]) * (previous_mass_fraction[{X, Y, Z, k}] - previous_mass_fraction[{X, Y - 1, Z, k}]);
							Flux[X][Y][Z][k][1] += (Vd_right - Vd_left);
							if (VC == 1) {
								for (kk = 0; kk < Nb_spec; ++kk) {
									Flux[X][Y][Z][kk][1] -= (previous_mass_fraction[{X, Y + 1, Z, kk}] * Vd_right - previous_mass_fraction[{X, Y, Z, kk}] * Vd_left);
								}
							}
							/// COMPUTE HC FLUX Z-DIRECTION
							Vd_right = (diffusion_coefficient[{X, Y, Z + 1, k}]) * (previous_mass_fraction[{X, Y, Z + 1, k}] - previous_mass_fraction[{X, Y, Z, k}]);
							Vd_left = (diffusion_coefficient[{X, Y, Z, k}]) * (previous_mass_fraction[{X, Y, Z, k}] - previous_mass_fraction[{X, Y, Z - 1, k}]);
							Flux[X][Y][Z][k][2] += (Vd_right - Vd_left);
							if (VC == 1) {
								for (kk = 0; kk < Nb_spec; ++kk) {
									Flux[X][Y][Z][kk][2] -= (previous_mass_fraction[{X, Y, Z + 1, kk}] * Vd_right - previous_mass_fraction[{X, Y, Z, kk}] * Vd_left);
								}
							}
						}
						for (k = 0; k < Nb_spec; ++k) {
							mass_fraction[{X, Y, Z, k}] += (Flux[X][Y][Z][k][0] + Flux[X][Y][Z][k][1] + Flux[X][Y][Z][k][2] + Production[{X, Y, Z, k}]) / (Flow->rho_0 * Flow->density[{X, Y, Z}]);
#if defined LMNA_solver
							Flow->divU[{X, Y, Z}] += (molar_mass_av[{X, Y, Z}] / Molar_mass[k]) * (Flux[X][Y][Z][k][0] + Flux[X][Y][Z][k][1] + Flux[X][Y][Z][k][2] + Production[{X, Y, Z, k}]) / (Flow->rho_0 * Flow->density[{X, Y, Z}]);
#endif  // defined
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
/*	BOUNDARY CONDITIONS:
    Richardson Extrapolation:
    (0th-order) BC: T[i] = T[i-1]  (1 adjacent point)
    (1st-order) BC: T[i] = 2T[i-1] - T[i-2] (2 adjacent points)
    (2nd-order) BC: T[i] = (4/3)T[i-1] - (1/3)T[i-2] (2 adjacent points)
    (3rd-order) BC: T[i] = (18/11)T[i-1] - (9/11)T[i-2] + (2/11)T[i-3] (3 adjacent points)
    (4th-order) BC: T[i] = (48/25)T[i-1] - (36/25)T[i-2] + (16/25)T[i-3] - (3/25)T[i-4] (4 adjacent points)
    (5th-order) BC: T[i] = (300/137)T[i-1] - (300/137)T[i-2] + (200/137)T[i-3] - (75/137)T[i-4] + (12/137)T[i-5] (5 adjacent points)

    Order refers to the accuracy or the number of terms in the Taylor series expansion used to approximate the value of the variable.
    Zero-order accuracy means that the error is independent of the step size h. i.e, E=O(1).
    First-order accuracy means that the error is proportional to the step size h. i.e, E=O(h).
    Second-order accuracy means that the error is proportional to the square of the step size h. i.e, E=O(h^2).
    Third-order accuracy means that the error is proportional to the cube of the step size h. i.e, E=O(h^3).

    Gradient refers to the order of the derivative of the variable or the type of difference scheme used to compute the derivative.
    Zeroth-order gradient means that the variable is constant. i.e, ∂T/∂x = 0.
    First-order gradient means that the variable is linear. i.e, ∂T/∂x = O(1).
    Second-order gradient means that the variable is quadratic. i.e, ∂T/∂x = O(x^2).
    Third-order gradient means that the variable is cubic. i.e, ∂T/∂x = O(x^3).

    //Types of Boundary Conditions:
    Dirichlet BC: Df = -Df_bar: Specifies the value of a variable at the boundary. Eg: setting fixed temperature or mass fraction at the boundary.
    Neumann BC: Df = (w + w_bar) * Y_k - Df_bar: Specifies the value of the gradient of a variable at the boundary. Eg: setting fixed heat flux or mass flux at the boundary.
    Zero Gradient BC: Df = Df_bar: Specifies the value of the gradient of a variable at the boundary to be equal to the gradient of the variable at the cell adjacent to the boundary.
    Zero Flux BC: Df = 0: Specifies the value of a variable at the boundary to be zero. Eg: setting zero heat flux or mass flux at the boundary.
    Constant Flux BC: Df = Df_bar: Specifies the value of the gradient of a variable at the boundary to be equal to the gradient of the variable at the cell adjacent to the boundary.
    Periodic BC: Df = Df_bar: Specifies the value of a variable at the boundary to be equal to the value of the variable at the opposite boundary.

    X, Y, Z are the coordinates of the boundary cell.
    Xp, Yp, Zp are the coordinates of the cell adjacent to the boundary.
    c_alpha is the velocity vector of the discrete velocity alpha.
    pop_s is the distribution function. pop_s[alpha][X][Y][Z][kk] is the distribution function of Species kk at the cell (X, Y, Z) and discrete velocity alpha.
    pop_s[alpha_bar[alpha]][Xp][Yp][Zp][kk] is the distribution function of Species kk at the cell (Xp, Yp, Zp) and discrete velocity alpha_bar[alpha].
    Boundaries[k].n is the normal vector of the boundary.
    Boundaries[k].Y_k is the mass fraction of Species k at the boundary.
    Boundaries[k].k[kk] is the Species index of the boundary.
*/
void Species_solver::BC(int time, Flow_solver* Flow, Thermal_solver* Thermal, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		int bou;
		int X, Y, Z, k, kk, alpha, Xp, Yp, Zp;
		for (k = 0; k < Boundaries.size(); ++k) {
			X = Boundaries[k].X;
			Y = Boundaries[k].Y;
			Z = Boundaries[k].Z;
			bou = Boundaries[k].type;
			switch (bou) {
				/// 1: Zero mass fraction BC (Dirichlet BC) - Zeroth order.
				// Enforces zero mass fraction at the boundary by reflecting the Species population at the boundary to the cell adjacent to the boundary.
				// This ensures that the mass fraction of the Species at the boundary is zero, effectively mimicking a reflective boundary condition for the Species.
				case 1: {
					for (alpha = 1; alpha < Discrete_Velocity; alpha++) {
						// here "-" identifies the cell that lies in the opposite direction to the c_alpha vector
						Xp = (X - c_alpha[alpha][0]);
						Yp = (Y - c_alpha[alpha][1]);
						Zp = (Z - c_alpha[alpha][2]);
						if (solid_species[{Xp, Yp, Zp}] != -1) {              // Check if the cell adjacent to the boundary is a solid cell
							if (DOT(c_alpha[alpha], Boundaries[k].n) >= 0) {  // Check if the discrete velocity is pointing towards the boundary
								for (kk = 0; kk < Nb_spec; kk++) {
									pop_s[alpha][X][Y][Z][kk] = -pop_s[alpha_bar[alpha]][Xp][Yp][Zp][kk];  // Zero mass fraction on walls
								}
							}
						}
					}
					break;
				}
				/// 2: Non Zero mass fraction BC (Dirichlet BC) - First order.
				// Enforces a non-zero mass fraction at the boundary. The mass fraction at the boundary is set to a specified value based on a combination of weights and the specified mass fraction values for different Species. This ensures that specific non-zero concentrations are maintained at the boundary, which can simulate conditions like inlet or outlet boundaries with specified Species concentrations.
				case 2: {
					for (alpha = 1; alpha < Discrete_Velocity; alpha++) {
						Xp = (X - c_alpha[alpha][0]);
						Yp = (Y - c_alpha[alpha][1]);
						Zp = (Z - c_alpha[alpha][2]);
						if (solid_species[{Xp, Yp, Zp}] != -1) {
							if (DOT(c_alpha[alpha], Boundaries[k].n) >= 0) {
								for (kk = 0; kk < Nb_spec; kk++) {
									pop_s[alpha][X][Y][Z][kk] = (weight[alpha] + weight[alpha_bar[alpha]]) * Boundaries[k].Y_k[kk]
									                            - pop_s[alpha_bar[alpha]][Xp][Yp][Zp][kk];  /// Non-zero mass fraction on walls
								}
							}
						}
					}
					break;
				}
				/// 3: Zero Gradient BC (Neumann BC) - Zeroth order.
				case 3: {
					for (alpha = 1; alpha < Discrete_Velocity; alpha++) {
						Xp = (X - c_alpha[alpha][0]);
						Yp = (Y - c_alpha[alpha][1]);
						Zp = (Z - c_alpha[alpha][2]);
						if (solid_species[{Xp, Yp, Zp}] != -1) {
							if (DOT(c_alpha[alpha], Boundaries[k].n) > 0) {
								for (kk = 0; kk < Nb_spec; kk++) {
									// Implementing zero-gradient boundary condition
									pop_s[alpha][X][Y][Z][kk] = pop_s[alpha_bar[alpha]][Xp][Yp][Zp][kk];
									// if using shifted indexing for zero-gradient
									// pop_s[alpha][X][Y][Z][kk] = pop_s[alpha][X + Boundaries[k].n[0]][Y + Boundaries[k].n[1]][Z + Boundaries[k].n[2]][kk];
								}
							}
						}
					}
					break;
				}
				/// 4: Constant Flux BC (Neumann BC) - First order.
				case 4: {
					for (alpha = 1; alpha < Discrete_Velocity; alpha++) {
						Xp = (X - c_alpha[alpha][0]);
						Yp = (Y - c_alpha[alpha][1]);
						Zp = (Z - c_alpha[alpha][2]);
						if (solid_species[{Xp, Yp, Zp}] != -1) {
							if (DOT(c_alpha[alpha], Boundaries[k].n) >= 0) {
								for (kk = 0; kk < Nb_spec; kk++) {
									pop_s[alpha][X][Y][Z][kk] = pop_s[alpha_bar[alpha]][Xp][Yp][Zp][kk];
								}
							}
						}
					}
					break;
				}
				/// 102: Non Zero mass fraction BC (Dirichlet BC) - First order.
				case 102: {
					if (!curved_boundaries) {
						for (kk = 0; kk < Nb_spec; ++kk) {
							if (Boundaries[k].k[kk] == 1) {
								Xp = (X - Boundaries[k].n[0]);
								Yp = (Y - Boundaries[k].n[1]);
								Zp = (Z - Boundaries[k].n[2]);
								if (Boundaries[k].n[0] != 0) {
									mass_fraction[{Xp, Y, Z, kk}] = Boundaries[k].Y_k[kk];  // Boundaries[k].Y_k[kk] = mass fractions from Species Field Initial Conditions for inlet
									previous_mass_fraction[{Xp, Y, Z, kk}] = Boundaries[k].Y_k[kk];
									temp_mass_fraction[{Xp, Y, Z, kk}] = Boundaries[k].Y_k[kk];
								}
								if (Boundaries[k].n[1] != 0) {
									mass_fraction[{X, Yp, Z, kk}] = Boundaries[k].Y_k[kk];
									previous_mass_fraction[{X, Yp, Z, kk}] = Boundaries[k].Y_k[kk];
									temp_mass_fraction[{X, Yp, Z, kk}] = Boundaries[k].Y_k[kk];
								}
								if (Boundaries[k].n[2] != 0) {
									mass_fraction[{X, Y, Zp, kk}] = Boundaries[k].Y_k[kk];
									previous_mass_fraction[{X, Y, Zp, kk}] = Boundaries[k].Y_k[kk];
									temp_mass_fraction[{X, Y, Zp, kk}] = Boundaries[k].Y_k[kk];
								}
							}
						}
					}
					if (curved_boundaries) {
						for (kk = 0; kk < Nb_spec; ++kk) {
							if (Boundaries[k].k[kk] == 1) {
								// First get mass fraction at image point
								double Y_image = 0;
								double previous_Y_image = 0;
								for (int i = 0; i < Boundaries[k].X_Image_Int.size(); i++) {
									int Xpi = Boundaries[k].X_Image_Int[i];
									int Ypi = Boundaries[k].Y_Image_Int[i];
									int Zpi = Boundaries[k].Z_Image_Int[i];
									Y_image += Boundaries[k].W_Image_Int[i] * mass_fraction[{Xpi, Ypi, Zpi, kk}];
									previous_Y_image += Boundaries[k].W_Image_Int[i] * previous_mass_fraction[{Xpi, Ypi, Zpi, kk}];
								}
								if (Boundaries[k].n[0] != 0) {
									mass_fraction[{Xp, Y, Z, kk}] = mass_fraction[{X, Y, Z, kk}];
									previous_mass_fraction[{Xp, Y, Z, kk}] = previous_mass_fraction[{X, Y, Z, kk}];
									temp_mass_fraction[{Xp, Y, Z, kk}] = temp_mass_fraction[{X, Y, Z, kk}];
								}
								if (Boundaries[k].n[1] != 0) {
									mass_fraction[{X, Yp, Z, kk}] = mass_fraction[{X, Y, Z, kk}];
									previous_mass_fraction[{X, Yp, Z, kk}] = previous_mass_fraction[{X, Y, Z, kk}];
									temp_mass_fraction[{X, Yp, Z, kk}] = temp_mass_fraction[{X, Y, Z, kk}];
								}
								if (Boundaries[k].n[2] != 0) {
									mass_fraction[{X, Y, Zp, kk}] = mass_fraction[{X, Y, Z, kk}];
									previous_mass_fraction[{X, Y, Zp, kk}] = previous_mass_fraction[{X, Y, Z, kk}];
									temp_mass_fraction[{X, Y, Zp, kk}] = temp_mass_fraction[{X, Y, Z, kk}];
								}
								// Normalization of mass fractions
								double Y_tot = 0;
								for (int k = 0; k < Nb_spec; ++k) {
									Y_tot += mass_fraction[{Xp, Yp, Zp, k}];
								}
								if (Y_tot > 0) {
									for (int k = 0; k < Nb_spec; ++k) {
										mass_fraction[{Xp, Yp, Zp, k}] /= Y_tot;
									}
								}
							}
						}
						break;
					}
					break;
				}
				/// 104: Zero flux BC (Neumann BC) - First order.
				case 104: {
					if (!curved_boundaries) {
						for (int kk = 0; kk < Nb_spec; ++kk) {
							if (Boundaries[k].k[kk] == 1) {
								Xp = (X - Boundaries[k].n[0]);
								Yp = (Y - Boundaries[k].n[1]);
								Zp = (Z - Boundaries[k].n[2]);
								if (Boundaries[k].n[0] != 0) {
									mass_fraction[{Xp, Y, Z, kk}] = mass_fraction[{X, Y, Z, kk}];
									previous_mass_fraction[{Xp, Y, Z, kk}] = previous_mass_fraction[{X, Y, Z, kk}];
									temp_mass_fraction[{Xp, Y, Z, kk}] = temp_mass_fraction[{X, Y, Z, kk}];
								}
								if (Boundaries[k].n[1] != 0) {
									mass_fraction[{X, Yp, Z, kk}] = mass_fraction[{X, Y, Z, kk}];
									previous_mass_fraction[{X, Yp, Z, kk}] = previous_mass_fraction[{X, Y, Z, kk}];
									temp_mass_fraction[{X, Yp, Z, kk}] = temp_mass_fraction[{X, Y, Z, kk}];
								}
								if (Boundaries[k].n[2] != 0) {
									mass_fraction[{X, Y, Zp, kk}] = mass_fraction[{X, Y, Z, kk}];
									previous_mass_fraction[{X, Y, Zp, kk}] = previous_mass_fraction[{X, Y, Z, kk}];
									temp_mass_fraction[{X, Y, Zp, kk}] = temp_mass_fraction[{X, Y, Z, kk}];
								}
							}
						}
					}
					if (curved_boundaries) {
						for (int kk = 0; kk < Nb_spec; ++kk) {
							if (Boundaries[k].k[kk] == 1) {
								double Y_image = 0;
								double previous_Y_image = 0;
								for (int i = 0; i < Boundaries[k].X_Image_Int.size(); i++) {
									int Xpi = Boundaries[k].X_Image_Int[i];
									int Ypi = Boundaries[k].Y_Image_Int[i];
									int Zpi = Boundaries[k].Z_Image_Int[i];
									Y_image += Boundaries[k].W_Image_Int[i] * mass_fraction[{Xpi, Ypi, Zpi, kk}];
									previous_Y_image += Boundaries[k].W_Image_Int[i] * previous_mass_fraction[{Xpi, Ypi, Zpi, kk}];
								}
								if (Boundaries[k].n[0] != 0) {
									mass_fraction[{X, Y, Z, kk}] = Y_image;
									previous_mass_fraction[{X, Y, Z, kk}] = previous_Y_image;
								}
								if (Boundaries[k].n[1] != 0) {
									mass_fraction[{X, Y, Z, kk}] = Y_image;
									previous_mass_fraction[{X, Y, Z, kk}] = previous_Y_image;
								}
								if (Boundaries[k].n[2] != 0) {
									mass_fraction[{X, Y, Z, kk}] = Y_image;
									previous_mass_fraction[{X, Y, Z, kk}] = previous_Y_image;
								}
							}
						}
						// Normalization of mass fractions
						double Y_tot = 0;
						for (int k = 0; k < Nb_spec; ++k) {
							Y_tot += mass_fraction[{X, Y, Z, k}];
						}
						if (Y_tot > 0) {
							for (int k = 0; k < Nb_spec; ++k) {
								mass_fraction[{X, Y, Z, k}] /= Y_tot;
							}
						}
					}
					break;
				}
				/// 105: Linear interpolation BC (Dirichlet BC) - First order.
				case 105: {
					if (!curved_boundaries) {
						for (int kk = 0; kk < Nb_spec; ++kk) {
							if (Boundaries[k].k[kk] == 1) {
								int Xp = X + Boundaries[k].n[0];
								int Yp = Y + Boundaries[k].n[1];
								int Zp = Z + Boundaries[k].n[2];

								int Xp1 = X + 2 * Boundaries[k].n[0];
								int Yp1 = Y + 2 * Boundaries[k].n[1];
								int Zp1 = Z + 2 * Boundaries[k].n[2];

								if (Boundaries[k].n[0] != 0) {
									mass_fraction[{X, Y, Z, kk}] = 2.0 * mass_fraction[{Xp, Y, Z, kk}] - mass_fraction[{Xp1, Y, Z, kk}];
									previous_mass_fraction[{X, Y, Z, kk}] = 2.0 * previous_mass_fraction[{Xp, Y, Z, kk}] - previous_mass_fraction[{Xp1, Y, Z, kk}];
								}
								if (Boundaries[k].n[1] != 0) {
									mass_fraction[{X, Y, Z, kk}] = 2.0 * mass_fraction[{X, Yp, Z, kk}] - mass_fraction[{X, Yp1, Z, kk}];
									previous_mass_fraction[{X, Y, Z, kk}] = 2.0 * previous_mass_fraction[{X, Yp, Z, kk}] - previous_mass_fraction[{X, Yp1, Z, kk}];
								}
								if (Boundaries[k].n[2] != 0) {
									mass_fraction[{X, Y, Z, kk}] = 2.0 * mass_fraction[{X, Y, Zp, kk}] - mass_fraction[{X, Y, Zp1, kk}];
									previous_mass_fraction[{X, Y, Z, kk}] = 2.0 * previous_mass_fraction[{X, Y, Zp, kk}] - previous_mass_fraction[{X, Y, Zp1, kk}];
								}
							}
						}
					}
					if (curved_boundaries) {
						for (int kk = 0; kk < Nb_spec; ++kk) {
							if (Boundaries[k].k[kk] == 1) {
								// First get mass fraction at image point
								double Y_image = 0;
								double previous_Y_image = 0;
								for (int i = 0; i < Boundaries[k].X_Image_Int.size(); i++) {
									int Xpi = Boundaries[k].X_Image_Int[i];
									int Ypi = Boundaries[k].Y_Image_Int[i];
									int Zpi = Boundaries[k].Z_Image_Int[i];
									Y_image += Boundaries[k].W_Image_Int[i] * mass_fraction[{Xpi, Ypi, Zpi, kk}];
									previous_Y_image += Boundaries[k].W_Image_Int[i] * previous_mass_fraction[{Xpi, Ypi, Zpi, kk}];
								}
								// Next calculate the mass fraction using the image point interpolation
								double interpolated_mass_fraction = 0.0;
								double interpolated_previous_mass_fraction = 0.0;
								if (Boundaries[k].n[0] != 0) {
									interpolated_mass_fraction += 2.0 * Y_image - mass_fraction[{X, Y, Z, kk}];
									interpolated_previous_mass_fraction += 2.0 * previous_Y_image - previous_mass_fraction[{X, Y, Z, kk}];
								}
								if (Boundaries[k].n[1] != 0) {
									interpolated_mass_fraction += 2.0 * Y_image - mass_fraction[{X, Y, Z, kk}];
									interpolated_previous_mass_fraction += 2.0 * previous_Y_image - previous_mass_fraction[{X, Y, Z, kk}];
								}
								if (Boundaries[k].n[2] != 0) {
									interpolated_mass_fraction += 2.0 * Y_image - mass_fraction[{X, Y, Z, kk}];
									interpolated_previous_mass_fraction += 2.0 * previous_Y_image - previous_mass_fraction[{X, Y, Z, kk}];
								}
								// Update mass fractions
								mass_fraction[{X, Y, Z, kk}] = interpolated_mass_fraction;
								previous_mass_fraction[{X, Y, Z, kk}] = interpolated_previous_mass_fraction;
							}
						}
						// Normalize mass fractions
						double sum_mass_fraction = 0.0;
						double sum_previous_mass_fraction = 0.0;
						for (int kk = 0; kk < Nb_spec; ++kk) {
							if (Boundaries[k].k[kk] == 1) {
								sum_mass_fraction += mass_fraction[{X, Y, Z, kk}];
								sum_previous_mass_fraction += previous_mass_fraction[{X, Y, Z, kk}];
							}
						}
						if (sum_mass_fraction > 0.0 && sum_previous_mass_fraction > 0.0) {
							for (int kk = 0; kk < Nb_spec; ++kk) {
								if (Boundaries[k].k[kk] == 1) {
									mass_fraction[{X, Y, Z, kk}] /= sum_mass_fraction;
									previous_mass_fraction[{X, Y, Z, kk}] /= sum_previous_mass_fraction;
								}
							}
						}
					}
					break;
				}
				/// 106: Second order interpolation BC (Dirichlet BC) - Second order.
				case 106: {
					if (!curved_boundaries) {
						for (int kk = 0; kk < Nb_spec; ++kk) {
							if (Boundaries[k].k[kk] == 1) {
								int Xp = X + Boundaries[k].n[0];
								int Yp = Y + Boundaries[k].n[1];
								int Zp = Z + Boundaries[k].n[2];

								int Xp1 = X + 2 * Boundaries[k].n[0];
								int Yp1 = Y + 2 * Boundaries[k].n[1];
								int Zp1 = Z + 2 * Boundaries[k].n[2];

								if (Boundaries[k].n[0] != 0) {
									mass_fraction[{X, Y, Z, kk}] = (4.0 / 3.0) * mass_fraction[{Xp, Y, Z, kk}] - (1.0 / 3.0) * mass_fraction[{Xp1, Y, Z, kk}];
									previous_mass_fraction[{X, Y, Z, kk}] = (4.0 / 3.0) * previous_mass_fraction[{Xp, Y, Z, kk}] - (1.0 / 3.0) * previous_mass_fraction[{Xp1, Y, Z, kk}];
								}
								if (Boundaries[k].n[1] != 0) {
									mass_fraction[{X, Y, Z, kk}] = (4.0 / 3.0) * mass_fraction[{X, Yp, Z, kk}] - (1.0 / 3.0) * mass_fraction[{X, Yp1, Z, kk}];
									previous_mass_fraction[{X, Y, Z, kk}] = (4.0 / 3.0) * previous_mass_fraction[{X, Yp, Z, kk}] - (1.0 / 3.0) * previous_mass_fraction[{X, Yp1, Z, kk}];
								}
								if (Boundaries[k].n[2] != 0) {
									mass_fraction[{X, Y, Z, kk}] = (4.0 / 3.0) * mass_fraction[{X, Y, Zp, kk}] - (1.0 / 3.0) * mass_fraction[{X, Y, Zp1, kk}];
									previous_mass_fraction[{X, Y, Z, kk}] = (4.0 / 3.0) * previous_mass_fraction[{X, Y, Zp, kk}] - (1.0 / 3.0) * previous_mass_fraction[{X, Y, Zp1, kk}];
								}
							}
						}
					}
					if (curved_boundaries) {
						for (int kk = 0; kk < Nb_spec; ++kk) {
							if (Boundaries[k].k[kk] == 1) {
								// First get mass fraction at image point
								double Y_image = 0;
								double previous_Y_image = 0;
								for (int i = 0; i < Boundaries[k].X_Image_Int.size(); i++) {
									int Xpi = Boundaries[k].X_Image_Int[i];
									int Ypi = Boundaries[k].Y_Image_Int[i];
									int Zpi = Boundaries[k].Z_Image_Int[i];
									Y_image += Boundaries[k].W_Image_Int[i] * mass_fraction[{Xpi, Ypi, Zpi, kk}];
									previous_Y_image += Boundaries[k].W_Image_Int[i] * previous_mass_fraction[{Xpi, Ypi, Zpi, kk}];
								}

								// Next calculate the mass fraction using the image point interpolation
								double interpolated_mass_fraction = 0.0;
								double interpolated_previous_mass_fraction = 0.0;
								if (Boundaries[k].n[0] != 0) {
									interpolated_mass_fraction += (4.0 / 3.0) * Y_image - (1.0 / 3.0) * mass_fraction[{X, Y, Z, kk}];
									interpolated_previous_mass_fraction += (4.0 / 3.0) * previous_Y_image - (1.0 / 3.0) * previous_mass_fraction[{X, Y, Z, kk}];
								}
								if (Boundaries[k].n[1] != 0) {
									interpolated_mass_fraction += (4.0 / 3.0) * Y_image - (1.0 / 3.0) * mass_fraction[{X, Y, Z, kk}];
									interpolated_previous_mass_fraction += (4.0 / 3.0) * previous_Y_image - (1.0 / 3.0) * previous_mass_fraction[{X, Y, Z, kk}];
								}
								if (Boundaries[k].n[2] != 0) {
									interpolated_mass_fraction += (4.0 / 3.0) * Y_image - (1.0 / 3.0) * mass_fraction[{X, Y, Z, kk}];
									interpolated_previous_mass_fraction += (4.0 / 3.0) * previous_Y_image - (1.0 / 3.0) * previous_mass_fraction[{X, Y, Z, kk}];
								}
								mass_fraction[{X, Y, Z, kk}] = interpolated_mass_fraction;
								previous_mass_fraction[{X, Y, Z, kk}] = interpolated_previous_mass_fraction;
							}
						}
						double sum_mass_fraction = 0.0;
						double sum_previous_mass_fraction = 0.0;
						for (int kk = 0; kk < Nb_spec; ++kk) {
							if (Boundaries[k].k[kk] == 1) {
								sum_mass_fraction += mass_fraction[{X, Y, Z, kk}];
								sum_previous_mass_fraction += previous_mass_fraction[{X, Y, Z, kk}];
							}
						}
						if (sum_mass_fraction > 0.0 && sum_previous_mass_fraction > 0.0) {
							for (int kk = 0; kk < Nb_spec; ++kk) {
								if (Boundaries[k].k[kk] == 1) {
									mass_fraction[{X, Y, Z, kk}] /= sum_mass_fraction;
									previous_mass_fraction[{X, Y, Z, kk}] /= sum_previous_mass_fraction;
								}
							}
						}
					}
					break;
				}
				/// 107: Third order interpolation BC (Dirichlet BC) - Third order.
				case 107: {
					if (!curved_boundaries) {
						for (int kk = 0; kk < Nb_spec; ++kk) {
							if (Boundaries[k].k[kk] == 1) {
								int Xp = X + Boundaries[k].n[0];
								int Yp = Y + Boundaries[k].n[1];
								int Zp = Z + Boundaries[k].n[2];

								int Xp1 = X + 2 * Boundaries[k].n[0];
								int Yp1 = Y + 2 * Boundaries[k].n[1];
								int Zp1 = Z + 2 * Boundaries[k].n[2];

								int Xp2 = X + 3 * Boundaries[k].n[0];
								int Yp2 = Y + 3 * Boundaries[k].n[1];
								int Zp2 = Z + 3 * Boundaries[k].n[2];

								if (Boundaries[k].n[0] != 0) {
									mass_fraction[{X, Y, Z, kk}] = (18.0 / 11.0) * mass_fraction[{Xp, Y, Z, kk}] - (9.0 / 11.0) * mass_fraction[{Xp1, Y, Z, kk}] + (2.0 / 11.0) * mass_fraction[{Xp2, Y, Z, kk}];
									previous_mass_fraction[{X, Y, Z, kk}] = (18.0 / 11.0) * previous_mass_fraction[{Xp, Y, Z, kk}] - (9.0 / 11.0) * previous_mass_fraction[{Xp1, Y, Z, kk}] + (2.0 / 11.0) * previous_mass_fraction[{Xp2, Y, Z, kk}];
								}
								if (Boundaries[k].n[1] != 0) {
									mass_fraction[{X, Y, Z, kk}] = (18.0 / 11.0) * mass_fraction[{X, Yp, Z, kk}] - (9.0 / 11.0) * mass_fraction[{X, Yp1, Z, kk}] + (2.0 / 11.0) * mass_fraction[{X, Yp2, Z, kk}];
									previous_mass_fraction[{X, Y, Z, kk}] = (18.0 / 11.0) * previous_mass_fraction[{X, Yp, Z, kk}] - (9.0 / 11.0) * previous_mass_fraction[{X, Yp1, Z, kk}] + (2.0 / 11.0) * previous_mass_fraction[{X, Yp2, Z, kk}];
								}
								if (Boundaries[k].n[2] != 0) {
									mass_fraction[{X, Y, Z, kk}] = (18.0 / 11.0) * mass_fraction[{X, Y, Zp, kk}] - (9.0 / 11.0) * mass_fraction[{X, Y, Zp1, kk}] + (2.0 / 11.0) * mass_fraction[{X, Y, Zp2, kk}];
									previous_mass_fraction[{X, Y, Z, kk}] = (18.0 / 11.0) * previous_mass_fraction[{X, Y, Zp, kk}] - (9.0 / 11.0) * previous_mass_fraction[{X, Y, Zp1, kk}] + (2.0 / 11.0) * previous_mass_fraction[{X, Y, Zp2, kk}];
								}
							}
						}
					}
					if (curved_boundaries) {
						for (int kk = 0; kk < Nb_spec; ++kk) {
							if (Boundaries[k].k[kk] == 1) {
								// First get mass fraction at image points
								double Y_image = 0.0;
								double previous_Y_image = 0.0;
								for (int i = 0; i < Boundaries[k].X_Image_Int.size(); i++) {
									int Xpi = Boundaries[k].X_Image_Int[i];
									int Ypi = Boundaries[k].Y_Image_Int[i];
									int Zpi = Boundaries[k].Z_Image_Int[i];
									Y_image += Boundaries[k].W_Image_Int[i] * mass_fraction[{Xpi, Ypi, Zpi, kk}];
									previous_Y_image += Boundaries[k].W_Image_Int[i] * previous_mass_fraction[{Xpi, Ypi, Zpi, kk}];
								}
								double interpolated_mass_fraction = 0.0;
								double interpolated_previous_mass_fraction = 0.0;
								if (Boundaries[k].n[0] != 0) {
									interpolated_mass_fraction += (18.0 / 11.0) * Y_image - (9.0 / 11.0) * mass_fraction[{X, Y, Z, kk}];
									interpolated_previous_mass_fraction += (18.0 / 11.0) * previous_Y_image - (9.0 / 11.0) * previous_mass_fraction[{X, Y, Z, kk}];
								}
								if (Boundaries[k].n[1] != 0) {
									interpolated_mass_fraction += (18.0 / 11.0) * Y_image - (9.0 / 11.0) * mass_fraction[{X, Y, Z, kk}];
									interpolated_previous_mass_fraction += (18.0 / 11.0) * previous_Y_image - (9.0 / 11.0) * previous_mass_fraction[{X, Y, Z, kk}];
								}
								if (Boundaries[k].n[2] != 0) {
									interpolated_mass_fraction += (18.0 / 11.0) * Y_image - (9.0 / 11.0) * mass_fraction[{X, Y, Z, kk}];
									interpolated_previous_mass_fraction += (18.0 / 11.0) * previous_Y_image - (9.0 / 11.0) * previous_mass_fraction[{X, Y, Z, kk}];
								}
								mass_fraction[{X, Y, Z, kk}] = interpolated_mass_fraction;
								previous_mass_fraction[{X, Y, Z, kk}] = interpolated_previous_mass_fraction;
							}
						}
						// Normalize mass fractions
						double sum_mass_fraction = 0.0;
						double sum_previous_mass_fraction = 0.0;
						for (int kk = 0; kk < Nb_spec; ++kk) {
							if (Boundaries[k].k[kk] == 1) {
								sum_mass_fraction += mass_fraction[{X, Y, Z, kk}];
								sum_previous_mass_fraction += previous_mass_fraction[{X, Y, Z, kk}];
							}
						}
						if (sum_mass_fraction > 0.0 && sum_previous_mass_fraction > 0.0) {
							for (int kk = 0; kk < Nb_spec; ++kk) {
								if (Boundaries[k].k[kk] == 1) {
									mass_fraction[{X, Y, Z, kk}] /= sum_mass_fraction;
									previous_mass_fraction[{X, Y, Z, kk}] /= sum_previous_mass_fraction;
								}
							}
						}
					}
					break;
				}
				/// 108: Convective mass fraction BC (Dirichlet BC) - First order.
				/*
				This boundary condition is based on the work of Lallemand and Luo (2000).
				λ = -min(0, n.u) where n is the normal vector and u is the velocity vector.
				It applies a BC where Y_k is updated using a linear interpolation approach. It ensures that the mass fraction is conserved.
				It can be used in cases where the velocity is not zero at the boundary.
				*/
				case 108: {
					int Xp = X + Boundaries[k].n[0];
					int Yp = Y + Boundaries[k].n[1];
					int Zp = Z + Boundaries[k].n[2];

					for (int kk = 0; kk < Nb_spec; ++kk) {
						if (Boundaries[k].k[kk] == 1) {
							// Calculate lambda based on velocity and normal direction
							double lambda = -std::min((Boundaries[k].n[0] * Flow->velocity[0])
							                              + (Boundaries[k].n[1] * Flow->velocity[1])
							                              + (Boundaries[k].n[2] * Flow->velocity[2]),
							                          0.0);
							// Apply boundary condition using RHS approach
							if (1.0 + lambda != 0.0) {
								mass_fraction[{X, Y, Z, kk}] = (previous_mass_fraction[{X, Y, Z, kk}] + lambda * mass_fraction[{Xp, Yp, Zp, kk}]) / (1.0 + lambda);
							} else {
								// Handle division by zero or near-zero case
								mass_fraction[{X, Y, Z, kk}] = previous_mass_fraction[{X, Y, Z, kk}];
							}
						}
					}
					break;
				}
				/// 109: Diffusive mass fraction BC (Dirichlet BC) - First order.
				/*
				This boundary condition is based on the work of Lallemand and Luo (2000).
				It applies a BC where Y_k is updated using a linear interpolation approach. It ensures that the mass fraction is conserved.
				It can be used in cases where the diffusion coefficient is not zero at the boundary.
				*/
				case 109: {
					for (int kk = 0; kk < Nb_spec; ++kk) {
						if (Boundaries[k].k[kk] == 1) {
							Xp = (X + Boundaries[k].n[0]);
							Yp = (Y + Boundaries[k].n[1]);
							Zp = (Z + Boundaries[k].n[2]);
							// Calculate mass fraction update
							double correction_term = mass_fraction[{X, Y, Z, Boundaries[k].kp[kk]}] * Boundaries[k].Y_k[kk] / (double)diffusion_coefficient[{X, Y, Z, kk}];
							mass_fraction[{X, Y, Z, kk}] = mass_fraction[{Xp, Yp, Zp, kk}] - correction_term;
						}
					}
					break;
				}
			}
		}
	}
	return;
}
/// ***************************************************** ///
/// GET DISTRIBUTION FUNCTION MOMENTA                     ///
/// ***************************************************** ///
/*This function calculates the moments of the distribution function, which is basically the mass fraction.
It uses the discrete velocity model to calculate the moments.*/
void Species_solver::momenta(Flow_solver* Flow, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		int X, Y, Z, k, alpha;
		for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (solid_species[{X, Y, Z}] == FALSE) {
						V_c[{X, Y, Z, 0}] = 0;
						V_c[{X, Y, Z, 1}] = 0;
						V_c[{X, Y, Z, 2}] = 0;
						molar_mass_av[{X, Y, Z}] = 0;
						for (k = 0; k < Nb_spec; ++k) {
							previous_mass_fraction[{X, Y, Z, k}] = mass_fraction[{X, Y, Z, k}];
							mass_fraction[{X, Y, Z, k}] = 0;
							Flux[X][Y][Z][k][0] = 0;
							Flux[X][Y][Z][k][1] = 0;
							Flux[X][Y][Z][k][2] = 0;
							for (alpha = 0; alpha < Discrete_Velocity; alpha++) {
								mass_fraction[{X, Y, Z, k}] += pop_s[alpha][X][Y][Z][k];
							}
						}
					}
				}
			}
		}
	}
	return;
}
/// ***************************************************** ///
/// GET CHEMICAL REACTION SOURCE TERMS                    ///
/// ***************************************************** ///
void Species_solver::User_defined_production(Thermal_solver* Thermal, Flow_solver* Flow, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		double Coeffs_fi, Coeffs_ri;
		int X, Y, Z, i, k;
		double c_k, k_fi, k_ri;

		// Iterate over the computational domain
		for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					// Check if the current cell is not a solid Species
					if (solid_species[{X, Y, Z}] == FALSE) {
						// Initialize production arrays
						for (k = 0; k < Nb_spec; ++k) {
							Production[{X, Y, Z, k}] = 0.0;
							Thermal->Production[{X, Y, Z}] = 0.0;
						}
						// Iterate over all reactions
						for (i = 0; i < Nb_reac; ++i) {
							Coeffs_fi = 1.0;
							Coeffs_ri = 0.0;
							k_ri = 0.0;
							k_fi = 0.0;

							// Check if the reaction type is Arrhenius
							if (Reac_type[i] == "Arrhenius") {
								// Calculate the concentration terms for the reaction
								for (int kprim = 0; kprim < Nb_spec; ++kprim) {
									c_k = Flow->rho_0 * Flow->density[{X, Y, Z}] * mass_fraction[{X, Y, Z, kprim}] / (double)Molar_mass[kprim];
									// Adjust coefficients based on stoichiometric coefficients and reaction orders
									if (Stoechio_coeff_fi[i][kprim] > 0 && Reac_order_fi[i][kprim] != 0)
										Coeffs_fi *= pow(c_k, Reac_order_fi[i][kprim]);
									if (Stoechio_coeff_ri[i][kprim] > 0 && Reac_order_ri[i][kprim] != 0)
										Coeffs_ri *= pow(c_k, Reac_order_ri[i][kprim]);
								}

								// Compute the forward and reverse reaction rates
								k_fi = Reac_coeff[i][0] * pow(Thermal->temperature[{X, Y, Z}] * Thermal->T_0, Reac_coeff[i][1])
								       * exp(-Reac_coeff[i][2] / (R_GAS * (Thermal->T_0 * Thermal->temperature[{X, Y, Z}] + EPSILON)));
								k_ri = Reac_coeff[i][4];
							}
							// If a Michaelis-Menten reaction (commented out in this example, but you can uncomment if needed)
							/*
							if (Reac_type[i] == "Michaelis") {
							    double S = 0.0;
							    double E = 0.0;
							    for (int kprim = 0; kprim < Nb_spec; ++kprim) {
							        if (Reac_order[i][kprim] == 1 && mass_fraction[{X, Y, Z, kprim}] > 0) {
							            S = mass_fraction[{X, Y, Z, kprim}];
							        }
							        if (Reac_order[i][kprim] == -1 && mass_fraction[{X, Y, Z, kprim}] > 0) {
							            E = mass_fraction[{X, Y, Z, kprim}];
							        }
							    }
							    Coeffs_fi = Reac_coeff[i][0] * E * S / (Reac_coeff[i][1] + S);
							}
							*/

							// Update production rates in the Thermal and Species arrays
							Thermal->Production[{X, Y, Z}] += Reac_coeff[i][3] * (k_fi * Coeffs_fi - k_ri * Coeffs_ri) * global_parameters.D_t;
							for (k = 0; k < Nb_spec; ++k) {
								Production[{X, Y, Z, k}] += Molar_mass[k] * (Stoechio_coeff_ri[i][k] - Stoechio_coeff_fi[i][k]) * (k_fi * Coeffs_fi - k_ri * Coeffs_ri) * global_parameters.D_t;
							}
						}
					}
				}
			}
		}
	}
}
/// ***************************************************** ///
/// GET AVERAGE MOLAR MASS                                ///
/// ***************************************************** ///
/* This function calculates the average molar mass of the fluid at each fluid node. The average molar mass is calculated using the mass fractions of the Species at each node.
The average molar mass is calculated using the formula:
W = Σ (Y_k / M_k) / Σ (Y_k)
where:
- W is the average molar mass.
- Y_k is the mass fraction of Species k.
- M_k is the molar mass of Species k.
The function loops over all fluid nodes in the computational domain and calculates the average molar mass at each node.
The average molar mass is stored in the molar_mass_av array.
This function should be used after the mass fraction has been updated and whenever the average molar mass is needed.
*/
void Species_solver::Molar_mass_computation(Flow_solver* Flow, Thermal_solver* Thermal, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		unsigned int X, Y, Z;
		/// I - GO OVER ALL FLUID NODES
		for (X = 0; X < MPI_parallel->dev_end[0]; ++X) {
			for (Y = 0; Y < MPI_parallel->dev_end[1]; ++Y) {
				for (Z = 0; Z < MPI_parallel->dev_end[2]; ++Z) {
					if (solid_species[{X, Y, Z}] == FALSE) {  /// if a node has at least one fluid neighbor
						molar_mass_av[{X, Y, Z}] = 0;
						double total_mass_fraction = 0;
						for (unsigned k = 0; k < Nb_spec; ++k) {
							molar_mass_av[{X, Y, Z}] += (mass_fraction[{X, Y, Z, k}] / Molar_mass[k]);
							total_mass_fraction = total_mass_fraction + mass_fraction[{X, Y, Z, k}];
						}
						molar_mass_av[{X, Y, Z}] = total_mass_fraction / molar_mass_av[{X, Y, Z}];
					}
				}
			}
		}
		unsigned int Xp, Yp, Zp;
		for (unsigned i = 0; i < Boundaries.size(); ++i) {
			X = Boundaries[i].X;
			Y = Boundaries[i].Y;
			Z = Boundaries[i].Z;
			Xp = (X - Boundaries[i].n[0]);
			Yp = (Y - Boundaries[i].n[1]);
			Zp = (Z - Boundaries[i].n[2]);
			molar_mass_av[{Xp, Yp, Zp}] = 0;
			molar_mass_av[{X, Y, Z}] = 0;
			double total_mass_fraction_p = 0;
			double total_mass_fraction = 0;
			for (unsigned k = 0; k < Nb_spec; ++k) {
				molar_mass_av[{Xp, Yp, Zp}] += (mass_fraction[{Xp, Yp, Zp, k}] / Molar_mass[k]);
				total_mass_fraction_p += mass_fraction[{Xp, Yp, Zp, k}];
				molar_mass_av[{X, Y, Z}] += (mass_fraction[{X, Y, Z, k}] / Molar_mass[k]);
				total_mass_fraction += mass_fraction[{X, Y, Z, k}];
			}
			molar_mass_av[{Xp, Yp, Zp}] = total_mass_fraction_p / molar_mass_av[{Xp, Yp, Zp}];
			molar_mass_av[{X, Y, Z}] = total_mass_fraction / molar_mass_av[{X, Y, Z}];
		}
	}
}
/// ***************************************************** ///
/// SPONGE ZONE                                           ///
/// ***************************************************** ///
/* This function applies a sponge zone to the Species field. The sponge zone is a region in the computational domain where the diffusion coefficient is increased to
reduce the diffusion of Species. This is useful for simulating the effects of a porous medium or a solid boundary. The sponge zone is defined by a start and end
radius, a direction, and a diffusion coefficient. The diffusion coefficient is increased in the sponge zone by a factor of 2.

The equation used here is:
D_k = D_k + (Density * Diffusion_coeff) - D_k) * 0.5 * (sin(M_PI * (r - r_start) / (r_end - r_start) - M_PI / 2.) + 1.)
where:
- D_k is the diffusion coefficient of Species k.
- Density is the density of the fluid.
- Diffusion_coeff is the diffusion coefficient of the sponge zone.
- r is the distance from the center of the sponge zone.
- r_start is the start radius of the sponge zone.
- r_end is the end radius of the sponge zone.
- M_PI is the value of pi.

The steps to apply the sponge zone are as follows:
1. Loop over all fluid nodes in the computational domain.
2. Check if the node is in the sponge zone.
3. If the node is in the sponge zone, increase the diffusion coefficient by a factor of 2.
4. Repeat for all Species.
The sponge zone is applied in the x, y, or z direction based on the direction parameter. The start and end radius define the region where the sponge zone is applied.
The diffusion coefficient is increased by a factor of 2 in the sponge zone. The sponge zone is applied to the Species field to reduce the diffusion of Species in the
sponge zone. This is useful for simulating the effects of a porous medium or a solid boundary.
*/
void Species_solver::Sponge_zone(double r_start, double r_end, int direction, double diffusion_coeff, Flow_solver* Flow, stl_import* Geo_stl, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		for (int X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (int Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (int Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					double xc, yc, zc, rr;
					MPI_parallel->get_coordinates(X, Y, Z, Geo_stl->x_center, Geo_stl->y_center, Geo_stl->z_center, xc, yc, zc);
					if (direction == 0) rr = xc;
					if (direction == 1) rr = yc;
					if (direction == 2) rr = zc;
					if (solid_species[{X, Y, Z}] == FALSE && rr >= r_start && rr <= r_end) {
						double coefficient = Flow->density[{X, Y, Z}] * diffusion_coeff;
						for (int k = 0; k < Nb_spec; k++) {
							diffusion_coefficient[{X, Y, Z, k}] += ((coefficient - diffusion_coefficient[{X, Y, Z, k}]) * 0.5 * (sin(M_PI * (rr - r_start) / (r_end - r_start) - M_PI / 2.) + 1.));
						}
					}
				}
			}
		}
	}
}
/// ***************************************************** ///
/// APPLY MODEL FOR COMPUTATION OF DIFFUSION COEFF        ///
/// ***************************************************** ///
void Species_solver::Diffusion_Coefficient_computation(Flow_solver* Flow, Thermal_solver* Thermal, Parallel_MPI* MPI_parallel) {
	unsigned int X, Y, Z, Xp, Yp, Zp, k;
	const std::array<double, 5> Sc_k = {1.241, 0.728, 0.941, 0.537, 0.69};  // Schmidt numbers
	std::vector<double> Yk(Nb_spec);                                        // Mass fraction array
	if (MPI_parallel->processor_id != MASTER) {
		for (X = MPI_parallel->start_XYZ2[0] - 1; X <= MPI_parallel->end_XYZ2[0] + 1; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1] - 1; Y <= MPI_parallel->end_XYZ2[1] + 1; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2] - 1; Z <= MPI_parallel->end_XYZ2[2] + 1; ++Z) {
					if ((solid_species[{X, Y, Z}] + solid_species[{X + 1, Y, Z}] + solid_species[{X - 1, Y, Z}] + solid_species[{X, Y + 1, Z}] + solid_species[{X, Y - 1, Z}] + solid_species[{X, Y, Z + 1}] + solid_species[{X, Y, Z - 1}]) < 7) {
						for (k = 0; k < Nb_spec; ++k) {
							diffusion_coefficient[{X, Y, Z, k}] = global_parameters.D_t * Sutherland_species_diffusion(Thermal->temperature[{X, Y, Z}] * Thermal->T_0, Sc_k[k]) / (global_parameters.D_x * global_parameters.D_x);
						}
					}
				}
			}
		}
		for (const auto& boundary : Boundaries) {
			X = boundary.X;
			Y = boundary.Y;
			Z = boundary.Z;
			Xp = X - boundary.n[0];
			Yp = Y - boundary.n[1];
			Zp = Z - boundary.n[2];
			for (k = 0; k < Nb_spec; ++k) {
				diffusion_coefficient[{X, Y, Z, k}] = global_parameters.D_t * Sutherland_species_diffusion(Thermal->temperature[{X, Y, Z}] * Thermal->T_0, Sc_k[k]) / (global_parameters.D_x * global_parameters.D_x);

				diffusion_coefficient[{Xp, Yp, Zp, k}] = global_parameters.D_t * Sutherland_species_diffusion(Thermal->temperature[{Xp, Yp, Zp}] * Thermal->T_0, Sc_k[k]) / (global_parameters.D_x * global_parameters.D_x);
			}
		}
	}
}
/// ***************************************************** ///
/// WRITE RECOVERY FILE                                   ///
/// ***************************************************** ///
void Species_solver::Recovery_write(Parallel_MPI* MPI_parallel, int& t) {
	if (MPI_parallel->processor_id != MASTER) {
		std::stringstream str_line1;
		str_line1 << "Alborz_Results/recovery/recover_Species_" << t << "_" << MPI_parallel->processor_id << ".dat";
		std::string strstr_line1 = str_line1.str();
		ofstream output_recovery;
		output_recovery.open(strstr_line1.c_str(), std::ios::out | std::ios::binary);
		unsigned int X, Y, Z, k;

		for (X = 0; X < MPI_parallel->dev_end[0]; X++) {
			for (Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
				for (Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
					for (k = 0; k < Nb_spec; ++k) {
						//			        for (alpha = 0; alpha < Discrete_Velocity; ++alpha) {
						//                      output_recovery.write((char*)&(pop_s[alpha][X][Y][Z][k]), sizeof(double)* 1);
						//                        }
						output_recovery.write((char*)&(mass_fraction[{X, Y, Z, k}]), sizeof(double) * 1);
						output_recovery.write((char*)&(diffusion_coefficient[{X, Y, Z, k}]), sizeof(double) * 1);
					}
					output_recovery.write((char*)&(solid_species[{X, Y, Z}]), sizeof(int) * 1);
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
void Species_solver::Recovery_read(Parallel_MPI* MPI_parallel, int& t) {
	if (MPI_parallel->processor_id != MASTER) {
		std::stringstream str_line1;
		str_line1 << "Alborz_Results/recovery/recover_Species_" << t << "_" << MPI_parallel->processor_id << ".dat";
		std::string strstr_line1 = str_line1.str();
		std::ifstream intput_recovery(strstr_line1, std::ios::in | std::ios::binary);
		unsigned int X, Y, Z, k;

		for (X = 0; X < MPI_parallel->dev_end[0]; X++) {
			for (Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
				for (Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
					for (k = 0; k < Nb_spec; ++k) {
						//			            for (alpha = 0; alpha < Discrete_Velocity; ++alpha) {
						//                            intput_recovery.read((char*)&(pop_s[alpha][X][Y][Z][k]), sizeof(double));
						//                            }
						intput_recovery.read((char*)&(mass_fraction[{X, Y, Z, k}]), sizeof(double));
						previous_mass_fraction[{X, Y, Z, k}] = mass_fraction[{X, Y, Z, k}];
						intput_recovery.read((char*)&(diffusion_coefficient[{X, Y, Z, k}]), sizeof(double));
					}
					intput_recovery.read((char*)&(solid_species[{X, Y, Z}]), sizeof(int));
				}
			}
		}
		intput_recovery.close();
	}
	return;
}
void Species_solver::register_recovery(IO_interface& io) {
	io.add_field(mass_fraction, "species_solver_mass_fraction");
	// was previously just set to mass_fraction which could be done in a custom read task
	io.add_field(previous_mass_fraction, "species_solver_previous_mass_fraction");
	io.add_field(diffusion_coefficient, "species_solver_diffusion_coefficient");
	io.add_field(solid_species, "species_solver_solid");
}
/// ***************************************************** ///
/// EXCHANGE POPULATIONS      BETWEEN PROCESSORS AT       ///
/// INTERFACES                                            ///
/// ***************************************************** ///
void Species_solver::Data_Exchange(Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		///--------------------------------------------------------->  In Y-direction
		vector<int> pop_totop;
		vector<int> pop_tobottom;

		for (int alpha = 0; alpha < Discrete_Velocity; alpha++) {
			if (c_alpha[alpha][1] > 0) {
				pop_totop.push_back(alpha);
			}
			if (c_alpha[alpha][1] < 0) {
				pop_tobottom.push_back(alpha);
			}
		}
		static double* buf_totop = new double[MPI_parallel->dev_end[0] * MPI_parallel->dev_end[2] * Nb_spec * pop_totop.size()];
		static double* buf_tobottom = new double[MPI_parallel->dev_end[0] * MPI_parallel->dev_end[2] * Nb_spec * pop_tobottom.size()];
		static double* buf_fromtop = new double[MPI_parallel->dev_end[0] * MPI_parallel->dev_end[2] * Nb_spec * pop_totop.size()];
		static double* buf_frombottom = new double[MPI_parallel->dev_end[0] * MPI_parallel->dev_end[2] * Nb_spec * pop_tobottom.size()];
		// Prepare messages to be sent
		for (int i = 0; i < pop_tobottom.size(); i++) {
			for (int X = 0; X < MPI_parallel->dev_end[0]; X++) {
				for (int Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
					memcpy(buf_totop + i * MPI_parallel->dev_end[0] * MPI_parallel->dev_end[2] * Nb_spec + X * MPI_parallel->dev_end[2] * Nb_spec + Z * Nb_spec,
					       &(pop_s[pop_totop[i]][X][MPI_parallel->end_XYZ2[1] + 1][Z][0]), sizeof(double) * Nb_spec);
					memcpy(buf_tobottom + i * MPI_parallel->dev_end[0] * MPI_parallel->dev_end[2] * Nb_spec + X * MPI_parallel->dev_end[2] * Nb_spec + Z * Nb_spec,
					       &(pop_s[pop_tobottom[i]][X][1][Z][0]), sizeof(double) * Nb_spec);
				}
			}
		}
		int Bottom_neighbour, Top_neighbour;
		Top_neighbour = MPI_parallel->proc_arrangement[MPI_parallel->proc_position[0]][(MPI_parallel->proc_position[1] + 1) % MPI_parallel->Np_Y][MPI_parallel->proc_position[2]];
		Bottom_neighbour = MPI_parallel->proc_arrangement[MPI_parallel->proc_position[0]][(MPI_parallel->proc_position[1] + MPI_parallel->Np_Y - 1) % MPI_parallel->Np_Y][MPI_parallel->proc_position[2]];
		////Send-recv all totop+fromt
		MPI_Sendrecv(buf_totop, pop_totop.size() * MPI_parallel->dev_end[0] * MPI_parallel->dev_end[2] * Nb_spec, MPI_DOUBLE, Top_neighbour, LTAG,
		             buf_frombottom, pop_tobottom.size() * MPI_parallel->dev_end[0] * MPI_parallel->dev_end[2] * Nb_spec, MPI_DOUBLE, Bottom_neighbour, LTAG,
		             MPI_COMM_WORLD, &status);
		// Send-recv all toleft+fromleft
		MPI_Sendrecv(buf_tobottom, pop_tobottom.size() * MPI_parallel->dev_end[0] * MPI_parallel->dev_end[2] * Nb_spec, MPI_DOUBLE, Bottom_neighbour, RTAG,
		             buf_fromtop, pop_totop.size() * MPI_parallel->dev_end[0] * MPI_parallel->dev_end[2] * Nb_spec, MPI_DOUBLE, Top_neighbour, RTAG,
		             MPI_COMM_WORLD, &status);
		for (int i = 0; i < pop_totop.size(); i++) {
			for (int X = 0; X < MPI_parallel->dev_end[0]; X++) {
				for (int Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
					memcpy(&pop_s[pop_totop[i]][X][2][Z][0], buf_frombottom + i * MPI_parallel->dev_end[0] * MPI_parallel->dev_end[2] * Nb_spec + X * MPI_parallel->dev_end[2] * Nb_spec + Z * Nb_spec, sizeof(double) * Nb_spec);
					memcpy(&pop_s[pop_tobottom[i]][X][MPI_parallel->actual_rows_XYZ[1] + 1][Z][0], buf_fromtop + i * MPI_parallel->dev_end[0] * MPI_parallel->dev_end[2] * Nb_spec + X * MPI_parallel->dev_end[2] * Nb_spec + Z * Nb_spec, sizeof(double) * Nb_spec);
				}
			}
		}
		/// Number of populations to exchange
		vector<int> pop_toleft;
		vector<int> pop_toright;

		for (int alpha = 0; alpha < Discrete_Velocity; alpha++) {
			if (c_alpha[alpha][0] > 0) {
				pop_toright.push_back(alpha);
			}
			if (c_alpha[alpha][0] < 0) {
				pop_toleft.push_back(alpha);
			}
		}
		///////// Exchange data with neighbors /////////
		static double* buf_toleft = new double[MPI_parallel->dev_end[1] * MPI_parallel->dev_end[2] * Nb_spec * pop_toleft.size()];
		static double* buf_toright = new double[MPI_parallel->dev_end[1] * MPI_parallel->dev_end[2] * Nb_spec * pop_toright.size()];
		static double* buf_fromleft = new double[MPI_parallel->dev_end[1] * MPI_parallel->dev_end[2] * Nb_spec * pop_toright.size()];
		static double* buf_fromright = new double[MPI_parallel->dev_end[1] * MPI_parallel->dev_end[2] * Nb_spec * pop_toleft.size()];
		///--------------------------------------------------------->  In X-direction
		// Prepare messages to be sent
		for (int i = 0; i < pop_toleft.size(); i++) {
			for (int Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
				for (int Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
					memcpy(buf_toright + i * MPI_parallel->dev_end[1] * MPI_parallel->dev_end[2] * Nb_spec + Y * MPI_parallel->dev_end[2] * Nb_spec + Z * Nb_spec,
					       &pop_s[pop_toright[i]][MPI_parallel->end_XYZ2[0] + 1][Y][Z][0], sizeof(double) * Nb_spec);
					memcpy(buf_toleft + i * MPI_parallel->dev_end[1] * MPI_parallel->dev_end[2] * Nb_spec + Y * MPI_parallel->dev_end[2] * Nb_spec + Z * Nb_spec,
					       &pop_s[pop_toleft[i]][1][Y][Z][0], sizeof(double) * Nb_spec);
				}
			}
		}
		int Left_neighbour, Right_neighbour;
		Right_neighbour = MPI_parallel->proc_arrangement[(MPI_parallel->proc_position[0] + 1) % MPI_parallel->Np_X][MPI_parallel->proc_position[1]][MPI_parallel->proc_position[2]];
		Left_neighbour = MPI_parallel->proc_arrangement[(MPI_parallel->proc_position[0] + MPI_parallel->Np_X - 1) % MPI_parallel->Np_X][MPI_parallel->proc_position[1]][MPI_parallel->proc_position[2]];
		////Send-recv all toright+fromleft
		MPI_Sendrecv(buf_toright, pop_toright.size() * MPI_parallel->dev_end[1] * MPI_parallel->dev_end[2] * Nb_spec, MPI_DOUBLE, Right_neighbour, LTAG,
		             buf_fromleft, pop_toleft.size() * MPI_parallel->dev_end[1] * MPI_parallel->dev_end[2] * Nb_spec, MPI_DOUBLE, Left_neighbour, LTAG,
		             MPI_COMM_WORLD, &status);
		// Send-recv all toleft+fromright
		MPI_Sendrecv(buf_toleft, pop_toleft.size() * MPI_parallel->dev_end[1] * MPI_parallel->dev_end[2] * Nb_spec, MPI_DOUBLE, Left_neighbour, RTAG,
		             buf_fromright, pop_toleft.size() * MPI_parallel->dev_end[1] * MPI_parallel->dev_end[2] * Nb_spec, MPI_DOUBLE, Right_neighbour, RTAG,
		             MPI_COMM_WORLD, &status);
		// Postprocess messages
		for (int i = 0; i < pop_toleft.size(); i++) {
			for (int Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
				for (int Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
					memcpy(&pop_s[pop_toright[i]][2][Y][Z][0], buf_fromleft + i * MPI_parallel->dev_end[1] * MPI_parallel->dev_end[2] * Nb_spec + Y * MPI_parallel->dev_end[2] * Nb_spec + Z * Nb_spec, sizeof(double) * Nb_spec);
					memcpy(&pop_s[pop_toleft[i]][MPI_parallel->actual_rows_XYZ[0] + 1][Y][Z][0], buf_fromright + i * MPI_parallel->dev_end[1] * MPI_parallel->dev_end[2] * Nb_spec + Y * MPI_parallel->dev_end[2] * Nb_spec + Z * Nb_spec, sizeof(double) * Nb_spec);
				}
			}
		}
		///--------------------------------------------------------->  In Z-direction
		vector<int> pop_tofront;
		vector<int> pop_torear;

		for (int alpha = 0; alpha < Discrete_Velocity; alpha++) {
			if (c_alpha[alpha][2] > 0) {
				pop_tofront.push_back(alpha);
			}
			if (c_alpha[alpha][2] < 0) {
				pop_torear.push_back(alpha);
				// std::cout << alpha << " ";
			}
		}
		static double* buf_tofront = new double[MPI_parallel->dev_end[0] * MPI_parallel->dev_end[1] * Nb_spec * pop_tofront.size()];
		static double* buf_torear = new double[MPI_parallel->dev_end[0] * MPI_parallel->dev_end[1] * Nb_spec * pop_torear.size()];
		static double* buf_fromfront = new double[MPI_parallel->dev_end[0] * MPI_parallel->dev_end[1] * Nb_spec * pop_tofront.size()];
		static double* buf_fromrear = new double[MPI_parallel->dev_end[0] * MPI_parallel->dev_end[1] * Nb_spec * pop_torear.size()];
		// Prepare messages to be sent
		for (int i = 0; i < pop_tofront.size(); i++) {
			for (int X = 0; X < MPI_parallel->dev_end[0]; X++) {
				for (int Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
					memcpy(buf_tofront + i * MPI_parallel->dev_end[0] * MPI_parallel->dev_end[1] * Nb_spec + X * MPI_parallel->dev_end[1] * Nb_spec + Y * Nb_spec,
					       &pop_s[pop_tofront[i]][X][Y][MPI_parallel->end_XYZ2[2] + 1][0], sizeof(double) * Nb_spec);
					memcpy(buf_torear + i * MPI_parallel->dev_end[0] * MPI_parallel->dev_end[1] * Nb_spec + X * MPI_parallel->dev_end[1] * Nb_spec + Y * Nb_spec,
					       &pop_s[pop_torear[i]][X][Y][1][0], sizeof(double) * Nb_spec);
				}
			}
		}
		int Rear_neighbour, Front_neighbour;
		Front_neighbour = MPI_parallel->proc_arrangement[MPI_parallel->proc_position[0]][MPI_parallel->proc_position[1]][(MPI_parallel->proc_position[2] + 1) % MPI_parallel->Np_Z];
		Rear_neighbour = MPI_parallel->proc_arrangement[MPI_parallel->proc_position[0]][MPI_parallel->proc_position[1]][(MPI_parallel->proc_position[2] + MPI_parallel->Np_Z - 1) % MPI_parallel->Np_Z];
		////Send-recv all totop+fromt
		MPI_Sendrecv(buf_tofront, pop_tofront.size() * MPI_parallel->dev_end[0] * MPI_parallel->dev_end[1] * Nb_spec, MPI_DOUBLE, Front_neighbour, LTAG,
		             buf_fromrear, pop_torear.size() * MPI_parallel->dev_end[0] * MPI_parallel->dev_end[1] * Nb_spec, MPI_DOUBLE, Rear_neighbour, LTAG,
		             MPI_COMM_WORLD, &status);
		// Send-recv all toleft+fromleft
		MPI_Sendrecv(buf_torear, pop_torear.size() * MPI_parallel->dev_end[0] * MPI_parallel->dev_end[1] * Nb_spec, MPI_DOUBLE, Rear_neighbour, RTAG,
		             buf_fromfront, pop_tofront.size() * MPI_parallel->dev_end[0] * MPI_parallel->dev_end[1] * Nb_spec, MPI_DOUBLE, Front_neighbour, RTAG,
		             MPI_COMM_WORLD, &status);
		for (int i = 0; i < pop_tofront.size(); i++) {
			for (int X = 0; X < MPI_parallel->dev_end[0]; X++) {
				for (int Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
					memcpy(&pop_s[pop_tofront[i]][X][Y][2][0], buf_fromrear + i * MPI_parallel->dev_end[0] * MPI_parallel->dev_end[1] * Nb_spec + X * MPI_parallel->dev_end[1] * Nb_spec + Y * Nb_spec, sizeof(double) * Nb_spec);
					memcpy(&pop_s[pop_torear[i]][X][Y][MPI_parallel->actual_rows_XYZ[2] + 1][0], buf_fromfront + i * MPI_parallel->dev_end[0] * MPI_parallel->dev_end[1] * Nb_spec + X * MPI_parallel->dev_end[1] * Nb_spec + Y * Nb_spec, sizeof(double) * Nb_spec);
				}
			}
		}
	}
}
/// ***************************************************** ///
/// EXCHANGE MACROSCOPIC VARS BETWEEN PROCESSORS AT       ///
/// INTERFACES                                            ///
/// ***************************************************** ///
void Species_solver::Data_Exchange_Macroscopic(Parallel_MPI* MPI_parallel) {
	/*	if (MPI_parallel->processor_id != MASTER) {
	    macroscopic_group.exchange_data();
	}*/
	// we cant use the macroscopic_group here because only mass_fraction[...,0] should be synchronized
	unsigned int X, Y, Z;
	if (MPI_parallel->processor_id != MASTER) {
		int Nparams = Nb_spec * 2;
		///--------------------------------------------------------->  In Y-direction
		static double* buf_totop_macro_s = new double[MPI_parallel->dev_end[0] * MPI_parallel->dev_end[2] * Nparams];
		static double* buf_tobottom_macro_s = new double[MPI_parallel->dev_end[0] * MPI_parallel->dev_end[2] * Nparams];
		static double* buf_fromtop_macro_s = new double[MPI_parallel->dev_end[0] * MPI_parallel->dev_end[2] * Nparams];
		static double* buf_frombottom_macro_s = new double[MPI_parallel->dev_end[0] * MPI_parallel->dev_end[2] * Nparams];
		// Prepare messages to be sent
		for (X = 0; X < MPI_parallel->dev_end[0]; X++) {
			for (Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
				memcpy(buf_totop_macro_s + X * MPI_parallel->dev_end[2] * Nparams + Z * Nparams + 0,
				       &mass_fraction[{X, MPI_parallel->end_XYZ2[1], Z, 0}], Nb_spec * sizeof(double));
				memcpy(buf_totop_macro_s + X * MPI_parallel->dev_end[2] * Nparams + Z * Nparams + Nb_spec,
				       &mass_fraction[{X, MPI_parallel->end_XYZ2[1] - 1, Z, 0}], Nb_spec * sizeof(double));

				memcpy(buf_tobottom_macro_s + X * MPI_parallel->dev_end[2] * Nparams + Z * Nparams + 0,
				       &mass_fraction[{X, 2, Z, 0}], Nb_spec * sizeof(double));
				memcpy(buf_tobottom_macro_s + X * MPI_parallel->dev_end[2] * Nparams + Z * Nparams + Nb_spec,
				       &mass_fraction[{X, 3, Z, 0}], Nb_spec * sizeof(double));
			}
		}
		unsigned int Bottom_neighbour, Top_neighbour;
		Top_neighbour = MPI_parallel->proc_arrangement[MPI_parallel->proc_position[0]][(MPI_parallel->proc_position[1] + 1) % MPI_parallel->Np_Y][MPI_parallel->proc_position[2]];
		Bottom_neighbour = MPI_parallel->proc_arrangement[MPI_parallel->proc_position[0]][(MPI_parallel->proc_position[1] + MPI_parallel->Np_Y - 1) % MPI_parallel->Np_Y][MPI_parallel->proc_position[2]];
		////Send-recv all totop+fromt
		MPI_Sendrecv(buf_totop_macro_s, Nparams * MPI_parallel->dev_end[0] * MPI_parallel->dev_end[2], MPI_DOUBLE, Top_neighbour, LTAG,
		             buf_frombottom_macro_s, Nparams * MPI_parallel->dev_end[0] * MPI_parallel->dev_end[2], MPI_DOUBLE, Bottom_neighbour, LTAG,
		             MPI_COMM_WORLD, &status);
		// Send-recv all toleft+fromleft
		MPI_Sendrecv(buf_tobottom_macro_s, Nparams * MPI_parallel->dev_end[0] * MPI_parallel->dev_end[2], MPI_DOUBLE, Bottom_neighbour, RTAG,
		             buf_fromtop_macro_s, Nparams * MPI_parallel->dev_end[0] * MPI_parallel->dev_end[2], MPI_DOUBLE, Top_neighbour, RTAG,
		             MPI_COMM_WORLD, &status);

		for (X = 0; X < MPI_parallel->dev_end[0]; X++) {
			for (Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
				memcpy(&mass_fraction[{X, 1, Z, 0}],
				       buf_frombottom_macro_s + X * MPI_parallel->dev_end[2] * Nparams + Z * Nparams + 0,
				       Nb_spec * sizeof(double));
				memcpy(&mass_fraction[{X, 0, Z, 0}],
				       buf_frombottom_macro_s + X * MPI_parallel->dev_end[2] * Nparams + Z * Nparams + Nb_spec,
				       Nb_spec * sizeof(double));

				memcpy(&mass_fraction[{X, MPI_parallel->actual_rows_XYZ[1] + 2, Z, 0}],
				       buf_fromtop_macro_s + X * MPI_parallel->dev_end[2] * Nparams + Z * Nparams + 0,
				       Nb_spec * sizeof(double));
				memcpy(&mass_fraction[{X, MPI_parallel->actual_rows_XYZ[1] + 3, Z, 0}],
				       buf_fromtop_macro_s + X * MPI_parallel->dev_end[2] * Nparams + Z * Nparams + Nb_spec,
				       Nb_spec * sizeof(double));
			}
		}
		///////// Exchange data with neighbors /////////
		static double* buf_toleft_macro_s = new double[MPI_parallel->dev_end[1] * MPI_parallel->dev_end[2] * Nparams];
		static double* buf_toright_macro_s = new double[MPI_parallel->dev_end[1] * MPI_parallel->dev_end[2] * Nparams];
		static double* buf_fromleft_macro_s = new double[MPI_parallel->dev_end[1] * MPI_parallel->dev_end[2] * Nparams];
		static double* buf_fromright_macro_s = new double[MPI_parallel->dev_end[1] * MPI_parallel->dev_end[2] * Nparams];
		///--------------------------------------------------------->  In X-direction
		// Prepare messages to be sent
		for (Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
			for (Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
				memcpy(buf_toright_macro_s + Y * MPI_parallel->dev_end[2] * Nparams + Z * Nparams + 0,
				       &mass_fraction[{MPI_parallel->end_XYZ2[0], Y, Z, 0}], Nb_spec * sizeof(double));
				memcpy(buf_toright_macro_s + Y * MPI_parallel->dev_end[2] * Nparams + Z * Nparams + Nb_spec,
				       &mass_fraction[{MPI_parallel->end_XYZ2[0] - 1, Y, Z, 0}], Nb_spec * sizeof(double));

				memcpy(buf_toleft_macro_s + Y * MPI_parallel->dev_end[2] * Nparams + Z * Nparams + 0,
				       &mass_fraction[{2, Y, Z, 0}], Nb_spec * sizeof(double));
				memcpy(buf_toleft_macro_s + Y * MPI_parallel->dev_end[2] * Nparams + Z * Nparams + Nb_spec,
				       &mass_fraction[{3, Y, Z, 0}], Nb_spec * sizeof(double));
			}
		}
		unsigned int Left_neighbour, Right_neighbour;
		Right_neighbour = MPI_parallel->proc_arrangement[(MPI_parallel->proc_position[0] + 1) % MPI_parallel->Np_X][MPI_parallel->proc_position[1]][MPI_parallel->proc_position[2]];
		Left_neighbour = MPI_parallel->proc_arrangement[(MPI_parallel->proc_position[0] + MPI_parallel->Np_X - 1) % MPI_parallel->Np_X][MPI_parallel->proc_position[1]][MPI_parallel->proc_position[2]];
		////Send-recv all toright+fromleft
		MPI_Sendrecv(buf_toright_macro_s, Nparams * MPI_parallel->dev_end[1] * MPI_parallel->dev_end[2], MPI_DOUBLE, Right_neighbour, LTAG,
		             buf_fromleft_macro_s, Nparams * MPI_parallel->dev_end[1] * MPI_parallel->dev_end[2], MPI_DOUBLE, Left_neighbour, LTAG,
		             MPI_COMM_WORLD, &status);
		// Send-recv all toleft+fromright
		MPI_Sendrecv(buf_toleft_macro_s, Nparams * MPI_parallel->dev_end[1] * MPI_parallel->dev_end[2], MPI_DOUBLE, Left_neighbour, RTAG,
		             buf_fromright_macro_s, Nparams * MPI_parallel->dev_end[1] * MPI_parallel->dev_end[2], MPI_DOUBLE, Right_neighbour, RTAG,
		             MPI_COMM_WORLD, &status);
		// Postprocess messages
		for (Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
			for (Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
				memcpy(&mass_fraction[{1, Y, Z, 0}], buf_fromleft_macro_s + Y * MPI_parallel->dev_end[2] * Nparams + Z * Nparams + 0, Nb_spec * sizeof(double));
				memcpy(&mass_fraction[{0, Y, Z, 0}], buf_fromleft_macro_s + Y * MPI_parallel->dev_end[2] * Nparams + Z * Nparams + Nb_spec, Nb_spec * sizeof(double));

				memcpy(&mass_fraction[{MPI_parallel->actual_rows_XYZ[0] + 2, Y, Z, 0}],
				       buf_fromright_macro_s + Y * MPI_parallel->dev_end[2] * Nparams + Z * Nparams + 0, Nb_spec * sizeof(double));
				memcpy(&mass_fraction[{MPI_parallel->actual_rows_XYZ[0] + 3, Y, Z, 0}],
				       buf_fromright_macro_s + Y * MPI_parallel->dev_end[2] * Nparams + Z * Nparams + Nb_spec, Nb_spec * sizeof(double));
			}
		}
		///--------------------------------------------------------->  In Z-direction
		static double* buf_tofront_macro_s = new double[MPI_parallel->dev_end[0] * MPI_parallel->dev_end[1] * Nparams];
		static double* buf_torear_macro_s = new double[MPI_parallel->dev_end[0] * MPI_parallel->dev_end[1] * Nparams];
		static double* buf_fromfront_macro_s = new double[MPI_parallel->dev_end[0] * MPI_parallel->dev_end[1] * Nparams];
		static double* buf_fromrear_macro_s = new double[MPI_parallel->dev_end[0] * MPI_parallel->dev_end[1] * Nparams];
		// Prepare messages to be sent
		for (X = 0; X < MPI_parallel->dev_end[0]; X++) {
			for (Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
				memcpy(buf_tofront_macro_s + X * MPI_parallel->dev_end[1] * Nparams + Y * Nparams + 0,
				       &mass_fraction[{X, Y, MPI_parallel->end_XYZ2[2], 0}], Nb_spec * sizeof(double));
				memcpy(buf_tofront_macro_s + X * MPI_parallel->dev_end[1] * Nparams + Y * Nparams + Nb_spec,
				       &mass_fraction[{X, Y, MPI_parallel->end_XYZ2[2] - 1, 0}], Nb_spec * sizeof(double));

				memcpy(buf_torear_macro_s + X * MPI_parallel->dev_end[1] * Nparams + Y * Nparams + 0,
				       &mass_fraction[{X, Y, 2, 0}], Nb_spec * sizeof(double));
				memcpy(buf_torear_macro_s + X * MPI_parallel->dev_end[1] * Nparams + Y * Nparams + Nb_spec,
				       &mass_fraction[{X, Y, 3, 0}], Nb_spec * sizeof(double));
			}
		}
		unsigned int Rear_neighbour, Front_neighbour;
		Front_neighbour = MPI_parallel->proc_arrangement[MPI_parallel->proc_position[0]][MPI_parallel->proc_position[1]][(MPI_parallel->proc_position[2] + 1) % MPI_parallel->Np_Z];
		Rear_neighbour = MPI_parallel->proc_arrangement[MPI_parallel->proc_position[0]][MPI_parallel->proc_position[1]][(MPI_parallel->proc_position[2] + MPI_parallel->Np_Z - 1) % MPI_parallel->Np_Z];
		////Send-recv all totop+fromt
		MPI_Sendrecv(buf_tofront_macro_s, Nparams * MPI_parallel->dev_end[0] * MPI_parallel->dev_end[1], MPI_DOUBLE, Front_neighbour, LTAG,
		             buf_fromrear_macro_s, Nparams * MPI_parallel->dev_end[0] * MPI_parallel->dev_end[1], MPI_DOUBLE, Rear_neighbour, LTAG,
		             MPI_COMM_WORLD, &status);
		// Send-recv all toleft+fromleft
		MPI_Sendrecv(buf_torear_macro_s, Nparams * MPI_parallel->dev_end[0] * MPI_parallel->dev_end[1], MPI_DOUBLE, Rear_neighbour, RTAG,
		             buf_fromfront_macro_s, Nparams * MPI_parallel->dev_end[0] * MPI_parallel->dev_end[1], MPI_DOUBLE, Front_neighbour, RTAG,
		             MPI_COMM_WORLD, &status);
		for (X = 0; X < MPI_parallel->dev_end[0]; X++) {
			for (Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
				memcpy(&mass_fraction[{X, Y, 1, 0}], buf_fromrear_macro_s + X * MPI_parallel->dev_end[1] * Nparams + Y * Nparams + 0, Nb_spec * sizeof(double));
				memcpy(&mass_fraction[{X, Y, 0, 0}], buf_fromrear_macro_s + X * MPI_parallel->dev_end[1] * Nparams + Y * Nparams + Nb_spec, Nb_spec * sizeof(double));

				memcpy(&mass_fraction[{X, Y, MPI_parallel->actual_rows_XYZ[2] + 2, 0}],
				       buf_fromfront_macro_s + X * MPI_parallel->dev_end[1] * Nparams + Y * Nparams + 0, Nb_spec * sizeof(double));
				memcpy(&mass_fraction[{X, Y, MPI_parallel->actual_rows_XYZ[2] + 3, 0}],
				       buf_fromfront_macro_s + X * MPI_parallel->dev_end[1] * Nparams + Y * Nparams + Nb_spec, Nb_spec * sizeof(double));
			}
		}
	}
}
/// ***************************************************** ///
/// COMPUTE MAXIMUM PROD RATE                             ///
/// ***************************************************** ///
// For each Species (Nb_spec), check if the previous mass fraction is above a certain threshold (1e-5)
// and if the production rate for that Species is negative (Production[{X, Y, Z, k}] < 0).
// If these conditions are met, compute the consumption rate and update the maximum and minimum consumption rates.
double Species_solver::Consumption_rate_monitor(Flow_solver* Flow, Parallel_MPI* MPI_parallel, int time, unsigned int output) {
	double prod_max = 0;
	double prod;
	double prod_min = 0;
	double prod_max_global, prod_min_global;
	if (MPI_parallel->processor_id != MASTER) {
		for (int X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (int Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (int Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (solid_species[{X, Y, Z}] == FALSE) {
						for (int k = 0; k < Nb_spec; k++) {
							if (previous_mass_fraction[{X, Y, Z, k}] > 1e-5 && Production[{X, Y, Z, k}] < 0) {
								prod = fabs((Production[{X, Y, Z, k}] / (Flow->rho_0 * Flow->density[{X, Y, Z}])) / previous_mass_fraction[{X, Y, Z, k}]);
								prod_max = std::max(prod_max, prod);
								prod_min = std::min(prod_min, prod);
							}
						}
					}
				}
			}
		}
	}
	MPI_Allreduce(&prod_max, &prod_max_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(&prod_min, &prod_min_global, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
	if (MPI_parallel->processor_id == MASTER && output == 1) {
		/// Create filename
		stringstream output_filename;
		output_filename << "Alborz_Results/debug/ProdctionRateMinMax.dat";
		ofstream output_file;
		/// Open file
		output_file.open(output_filename.str().c_str(), fstream::app);
		/// Write data
		output_file << time << " ";  // time step
		output_file << setprecision(30) << prod_min_global << " " << prod_max_global << endl;
		/// Close file
		output_file.close();
	}
	return prod_max_global;
}
double Species_solver::Consumption_rate_monitor_each_species(Flow_solver* Flow, Parallel_MPI* MPI_parallel, int time, unsigned int output) {
	std::vector<double> prod_values(Nb_spec, 0.0);  // To store production rates for each Species
	std::vector<double> prod_global(Nb_spec, 0.0);  // To store global production rates for each Species
	if (MPI_parallel->processor_id != MASTER) {
		for (int X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (int Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (int Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (solid_species[{X, Y, Z}] == FALSE) {
						for (int k = 0; k < Nb_spec; k++) {
							if (previous_mass_fraction[{X, Y, Z, k}] > 1e-5 && Production[{X, Y, Z, k}] < 0) {
								double prod = fabs((Production[{X, Y, Z, k}] / (Flow->rho_0 * Flow->density[{X, Y, Z}])) / previous_mass_fraction[{X, Y, Z, k}]);
								prod_values[k] += prod;  // local vector that stores the partial production rates for each Species on each processor.
							}
						}
					}
				}
			}
		}
	}
	MPI_Allreduce(prod_values.data(), prod_global.data(), Nb_spec, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);  // global sum of production rates for each Species.
	if (MPI_parallel->processor_id == MASTER && output == 1) {
		/// Create filename
		stringstream output_filename;
		output_filename << "Alborz_Results/debug/SpeciesProductionRates.dat";
		ofstream output_file;
		/// Open file
		output_file.open(output_filename.str().c_str(), fstream::app);

		// Write header if the file is being created for the first time
		if (output_file.tellp() == 0) {
			output_file << "TimeStep ";
			for (int k = 0; k < Nb_spec; ++k) {
				output_file << "\t" << species_name_RG[k];
			}
			output_file << std::endl;
		}

		/// Write data
		output_file << time;
		for (int k = 0; k < Nb_spec; ++k) {
			output_file << "\t" << setprecision(15) << prod_global[k];  // total production rate of a specific Species across all processors after the reduction operation.
		}
		output_file << endl;

		/// Close file
		output_file.close();
	}

	return 0.0;  // Change the return type as needed
}
/// ***************************************************** ///
/// COMPUTE MAXIMUM FO IN DOMAIN                          ///
/// ***************************************************** ///
double Species_solver::Fo_monitor(Flow_solver* Flow, Parallel_MPI* MPI_parallel, int time, unsigned int output) {
	double D_max = -1;
	double D_min = 1e16;
	double D_max_global, D_min_global;
	if (MPI_parallel->processor_id != MASTER) {
		int X, Y, Z, k;
		for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (solid_species[{X, Y, Z}] == FALSE) {
						for (k = 0; k < Nb_spec; k++) {
							D_max = std::max(D_max, diffusion_coefficient[{X, Y, Z, k}] / Flow->density[{X, Y, Z}]);
							D_min = std::min(D_min, diffusion_coefficient[{X, Y, Z, k}] / Flow->density[{X, Y, Z}]);
						}
					}
				}
			}
		}
	}
	MPI_Allreduce(&D_max, &D_max_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(&D_min, &D_min_global, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
	if (MPI_parallel->processor_id == MASTER && output == 1) {
		/// Create filename
		stringstream output_filename;
		output_filename << "Alborz_Results/debug/SpeciesDiffusionMinMax.dat";
		ofstream output_file;
		/// Open file
		output_file.open(output_filename.str().c_str(), fstream::app);
		/// Write data
		output_file << time << " ";  // time step
		output_file << setprecision(30) << D_min_global << " " << D_max_global << endl;
		/// Close file
		output_file.close();
	}
	return D_max_global;
}
/// ***************************************************** ///
/// INITIAL CONDITIONS: USER-DEFINED VALUES (FROM INPUT   ///
/// FILE)                                                 ///
/// ***************************************************** ///
void Inline_User_Defined(Vector_field& Y_k, Vector_field& diffusion_coefficient, Solid_field& solid, const std::vector<std::string>& species_names, double N_x, double N_y, double N_z, double Nb_spec, double c_s2, const std::string& filename, int Zones, Parallel_MPI* MPI_parallel) {
    if (MPI_parallel->is_master()) {
        return;
    }
    const std::vector<double> initial_vec(Nb_spec, 0.0);
    std::vector<int> type(Zones + 1, 0);
    Ini_s.resize(Zones + 1, initial_vec);
    Ini_D.resize(Zones + 1, initial_vec);
    type[0] = 1;  /// -----> Solids are set to +1
    std::vector<Initial_field_slice> special_volumes;

    /// Open input file
    ifstream input_file_s(filename + ".dat", ios::binary);
    find_line_after_header(input_file_s, "c\tSpecies Field Initial Conditions");
    if (MPI_parallel->processor_id == (MASTER + 1)) {
        std::cout << "SPECIES: User defined Initial conditions \n";
        std::cout << "================================ \n";
    }
    for (unsigned i = 0; i < Zones; i++) {
        int counter_species;
        std::vector<int> index_nonzero;
        std::string species_names_temp;
        find_line_after_comment(input_file_s);
        int index;
        bool is_extra = false;
        input_file_s >> index;

        if (index < 0) {
            is_extra = true;
            index = Ini_s.size();
            Ini_s.push_back(initial_vec);
            Ini_D.push_back(initial_vec);
            type.push_back(0);
            --i;
        }
        input_file_s >> type[index];
        input_file_s >> counter_species;
        find_line_after_comment(input_file_s);
        for (unsigned j = 0; j < counter_species; j++) {
            input_file_s >> species_names_temp;
            auto it = std::find(species_names.begin(), species_names.end(), species_names_temp);
            if (it != species_names.end()) {
                const int k = std::distance(species_names.begin(), it);
                input_file_s >> Ini_s[index][k];
                index_nonzero.push_back(k);
            } else {
                ERROR_ABORT("[Error] Unknown Species \"" << species_names_temp << "\" "
                                                         << " in initial conditions.");
            }
        }

        find_line_after_comment(input_file_s);
        for (unsigned j = 0; j < counter_species; j++) {
            input_file_s >> species_names_temp;
            auto it = std::find(species_names.begin(), species_names.end(), species_names_temp);
            if (it != species_names.end()) {
                const int k = std::distance(species_names.begin(), it);
                input_file_s >> Ini_D[index][k];
                Ini_D[index][k] = (global_parameters.D_t * Ini_D[index][k] / (global_parameters.D_x * global_parameters.D_x));
            } else {
                ERROR_ABORT("[Error] Unknown Species \"" << species_names_temp << "\" "
                                                         << " in initial conditions.");
            }
        }

        if (is_extra) {
            special_volumes.push_back({});
            Initial_field_slice& slice = special_volumes.back();
            slice.index = index;
        }

        if (MPI_parallel->processor_id == (MASTER + 1)) {
            std::cout << "Zone : " << index << "\t Type : " << type[index] << std::endl;
            std::cout << "Initial non-zero mass fraction : " << std::endl;
            for (int k = 0; k < counter_species; ++k) {
                std::cout << species_names[index_nonzero[k]] << " " << Ini_s[index][index_nonzero[k]] << "\t";
                if ((k + 1) % 6 == 0) std::cout << "\n";
            }
            std::cout << endl;
            std::cout << "Tau_s : " << std::endl;
            for (int k = 0; k < counter_species; ++k) {
                std::cout << species_names[index_nonzero[k]] << " " << Ini_D[index][index_nonzero[k]] << "\t";
                if ((k + 1) % 6 == 0) std::cout << "\n";
            }
            std::cout << endl;
        }
    }
    input_file_s.close();
    for (unsigned X = 0; X < MPI_parallel->dev_end[0]; X++) {
        for (unsigned Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
            for (unsigned Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
                for (int k = 0; k < Nb_spec; k++) {
                    diffusion_coefficient[{X, Y, Z, k}] = Ini_D[solid[{X, Y, Z}]][k];
                    Y_k[{X, Y, Z, k}] = Ini_s[solid[{X, Y, Z}]][k];
                }
                solid[{X, Y, Z}] = type[solid[{X, Y, Z}]];
            }
        }
    }
}
/// ***************************************************** ///
/// INITIAL CONDITIONS: 3-D COMPRESSIBLE TAYLOR-GREEN     ///
/// (INPUT FATA FILE NEEDED) VORTEX                       ///
/// ***************************************************** ///
void TGV3Dcold_species(Vector_field& Y_k, Vector_field& diffusion_coefficient, Solid_field& solid, const std::vector<std::string>& species_names, double N_x, double N_y, double N_z, double Nb_spec, double c_s2, const std::string& filename, int Zones, Parallel_MPI* MPI_parallel) {
    std::string file_address;
    int N_grid;

    if (MPI_parallel->processor_id != MASTER) {
        unsigned int X, Y, Z, k, i, Nb;
        double xx;
        Nb = Nb_spec;

        // Initialize arrays to read data from file
        std::vector<std::vector<double>> Y_initial(Nb_spec);
        std::vector<double> x_initial;

        // Read the file address from the header file
        std::ifstream input_file(filename + ".dat", std::ios::binary);
        find_line_after_header(input_file, "c\tSpecies Initial Profile File");
        input_file >> N_grid >> file_address;
        input_file.close();

        // Read data from the extracted filename
        std::ifstream data_file(file_address, std::ios::in | std::ios::binary);
        x_initial.resize(N_grid);
        for (k = 0; k < Nb_spec; ++k) {
            Y_initial[k].resize(N_grid);
        }
        for (X = 0; X < N_grid; ++X) {
            data_file >> x_initial[X];
            x_initial[X] *= 0.01;
            for (k = 0; k < Nb_spec; ++k) {
                data_file >> Y_initial[k][X];
            }
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
                    solid[{X, Y, Z}] = -1;
                    for (k = 0; k < Nb; ++k) {
                        diffusion_coefficient[{X, Y, Z, k}] = 1e-5;
                        Y_k[{X, Y, Z, k}] = Y_initial[k][index] + (xx - x_initial[index]) * (Y_initial[k][index + 1] - Y_initial[k][index]) / (x_initial[index + 1] - x_initial[index]);
                    }
                }
            }
        }

        if (MPI_parallel->processor_id == (MASTER + 1)) {
            std::cout << "SPECIES: non-reacting TGV-3D initial conditions \n";
            std::cout << "================================ \n";
            std::cout << "Species file address: " << file_address << std::endl;
            std::cout << "N_grid: " << N_grid << std::endl;
            std::cout << "================================ \n";
        }
    }
}
/// ***************************************************** ///
/// INITIAL CONDITIONS: 3-D REACTING TAYLOR-GREEN         ///
/// (INPUT FATA FILE NEEDED) VORTEX                       ///
/// ***************************************************** ///
void TGV3Dreacting_species(Vector_field& Y_k, Vector_field& diffusion_coefficient, Scalar_field& molar_mass_av, Solid_field& solid, std::vector<std::string>& species_names, double N_x, double N_y, double N_z, double Nb_spec, double c_s2, double& T_eq, const std::string& filename, int Zones, Parallel_MPI* MPI_parallel, Flow_solver* Flow, Thermal_solver* Thermal, Thermo_chemistry_cantera* thermo_chemistry, Species_solver* Species) {
	if (MPI_parallel->processor_id != MASTER) {
		int Nb = Nb_spec;
		double stiffness, radius, xflamepos, ini_temp;
		std::vector<double> massf_in(Nb), massf_out(Nb), diff_co_in(Nb), diff_co_out(Nb);
		std::string input_filename = filename + ".dat";
		std::ifstream input_file(input_filename.c_str(), std::ios::binary);
		input_file.seekg(0, std::ios::beg);
		find_line_after_header(input_file, "c\tSpecies Initial Profile File");
		find_line_after_comment(input_file);
		input_file >> stiffness >> radius >> xflamepos >> ini_temp;
		find_line_after_comment(input_file);
		for (unsigned int k = 0; k < Nb; ++k) {
			input_file >> species_names[k] >> massf_in[k];
		}
		find_line_after_comment(input_file);
		for (unsigned int k = 0; k < Nb; ++k) {
			input_file >> species_names[k] >> diff_co_in[k];
		}
		find_line_after_comment(input_file);
		for (unsigned int k = 0; k < Nb; ++k) {
			input_file >> species_names[k] >> massf_out[k];
		}
		find_line_after_comment(input_file);
		for (unsigned int k = 0; k < Nb; ++k) {
			input_file >> species_names[k] >> diff_co_out[k];
		}
		input_file.close();
		// Initialize fields based on the imported data
		for (unsigned int X = 0; X < MPI_parallel->dev_end[0]; ++X) {
			double xx = fmod(X - MPI_parallel->start_XYZ2[0] + MPI_parallel->start_XYZ[0] + global_parameters.Nx, global_parameters.Nx);
			double stiffness2 = 100 * stiffness / global_parameters.Nx;
			double flamepos = xflamepos * global_parameters.Nx;
			double radial_dist = sqrt(pow(xx - flamepos, 2));
			double ref_tanh = 0.5 * (1.0 + tanh(stiffness2 * (radial_dist - radius) / radius));
			for (unsigned int Y = 0; Y < MPI_parallel->dev_end[1]; ++Y) {
				for (unsigned int Z = 0; Z < MPI_parallel->dev_end[2]; ++Z) {
					solid[{X, Y, Z}] = -1;  // Mark as solid
					std::vector<double> Y_temp(Nb);
					double test_one = 0.0;
					// Calculate and normalize mass fractions
					for (unsigned int j = 0; j < Nb; ++j) {
						Y_k[{X, Y, Z, j}] = massf_in[j] * (1.0 - ref_tanh) + massf_out[j] * ref_tanh;  // eq: Yk = Yk1*(1-tanh) + Yk2*tanh
						test_one += Y_k[{X, Y, Z, j}];
						diffusion_coefficient[{X, Y, Z, j}] = diff_co_in[j];
					}
					for (unsigned int j = 0; j < Nb; ++j) {
						Y_k[{X, Y, Z, j}] /= test_one;
						Y_temp[j] = Y_k[{X, Y, Z, j}];  // Store normalized mass fractions for Cantera
					}
					double T_in = ini_temp;
					double P = Flow->p_th_0;
#if defined compressible
					P = Flow->density[{X, Y, Z}] * Flow->rho_0 * R_GAS * T_in / (molar_mass_av[{X, Y, Z}]);
#endif
					thermo_chemistry->sol->thermo()->setState_TPY(T_in, P, Y_temp.data());
					thermo_chemistry->sol->thermo()->equilibrate("HP", "auto", 1.0e-4);
					// Mass fractions initialization
					std::vector<double> massf_cantera(Nb);
					thermo_chemistry->sol->thermo()->getMassFractions(massf_cantera.data());
					for (unsigned int j = 0; j < Nb; ++j) {
						Y_k[{X, Y, Z, j}] = massf_cantera[j];
					}
					molar_mass_av[{X, Y, Z}] = thermo_chemistry->sol->thermo()->meanMolecularWeight() * 1e-3;
					T_eq = thermo_chemistry->sol->thermo()->temperature();
					Thermal->temperature[{X, Y, Z}] = T_eq;
					Thermal->c_p[{X, Y, Z}] = thermo_chemistry->sol->thermo()->cp_mass();
					Thermal->thermal_diffusion_coefficient[{X, Y, Z}] = thermo_chemistry->sol->transport()->thermalConductivity();
					solid[{X, Y, Z}] = -1;
				}
			}
		}
		if (MPI_parallel->processor_id == (MASTER + 1)) {
			std::cout << "SPECIES: reacting TGV-3D initial conditions \n";
			std::cout << "================================ \n";
			std::cout << "Species file address: " << filename << std::endl;
			std::cout << "Stiffness: " << stiffness << std::endl;
			std::cout << "Radius: " << radius << std::endl;
			std::cout << "Flame Position: " << xflamepos << std::endl;
			std::cout << "Initial Temperature: " << ini_temp << std::endl;
			std::cout << "================================ \n";
		}
	}
}
/// ***************************************************** ///
/// INITIAL CONDITIONS: GAUSSIAN HILL                     ///
/// ***************************************************** ///
void Gaussian_species(Vector_field& Y_k, Vector_field& diffusion_coefficient, Solid_field& solid, const std::vector<std::string>& species_names, double N_x, double N_y, double N_z, double Nb_spec, double c_s2, const std::string& filename, int Zones, Parallel_MPI* MPI_parallel) {
    ///----------------------------------------------------------------------///
    ///                        COMMAND LINE INPUT                            ///
    ///----------------------------------------------------------------------///
    if (MPI_parallel->processor_id != MASTER) {
        if (MPI_parallel->processor_id == (MASTER + 1)) {
            std::cout << "SPECIES: Gaussian Hill conditions \n";
            std::cout << "================================= \n";
        }
        /// Open input file
        ifstream input_file(filename + ".dat", ios::binary);
        find_line_after_header(input_file, "c\tSpecies Field Gaussian Initial Conditions");
        find_line_after_comment(input_file);
        double dim, sigma0, x_0, y_0, z_0, Ini_D;
        input_file >> dim >> sigma0 >> x_0 >> y_0 >> z_0 >> Ini_D;
        input_file.close();

        Ini_D = Ini_D * (global_parameters.D_t / sqr(global_parameters.D_x));

        if (MPI_parallel->processor_id == (MASTER + 1)) {
            std::cout << "D : " << dim << "\t Sigma : " << sigma0 << "\n"
                      << "x0 : " << x_0 << "\t y0 : "
                      << "\t z0 : " << z_0 << "\n"
                      << "Diffusion coefficient: " << Ini_D << std::endl;
        }

        for (int X = 0; X < MPI_parallel->dev_end[0]; X++) {
            double xx = fmod(X - MPI_parallel->start_XYZ2[0] + MPI_parallel->start_XYZ[0] + N_x, N_x) * global_parameters.D_x;
            for (int Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
                double yy = fmod(Y - MPI_parallel->start_XYZ2[1] + MPI_parallel->start_XYZ[1] + N_y, N_y) * global_parameters.D_x;
                for (int Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
                    // double zz = fmod(Z - MPI_parallel->start_XYZ2[2] + MPI_parallel->start_XYZ[2] + N_z, N_z) * global_parameters.D_x;
                    diffusion_coefficient[{X, Y, Z, 0}] = Ini_D;
                    double distance = sqr(xx - x_0) + sqr(yy - y_0);
                    if (dim > 2) distance += sqr(yy - z_0);
                    Y_k[{X, Y, Z, 0}] = std::exp(-0.5 * distance / sqr(sigma0));
                    solid[{X, Y, Z}] = -1;
                }
            }
        }
    }
}
/// ***************************************************** ///
/// GET THERMAL DIFFUSION USING SUTHERLAND MODEL          ///
/// ***************************************************** ///
double Sutherland_species_diffusion(double T, double Sc_k) {
	//	constexpr double S = 110.5;          /* [S] = K */
	constexpr double T_star = 298.;      /* [T_star] = K */
	constexpr double mu_star = 1.782e-5; /* [mu_star] = kg.m/s^3.K */

	const double D_k = (mu_star / Sc_k) * pow(T / (double)T_star, 0.69);  // * (T_star + S)/(double)(T + S);
	return D_k;
}
//=============================================================================================================================
//=============================================================================================================================
Species_solver::~Species_solver() {
}
#if defined IMPLEX && !defined REGATH_LIB
//============================================================================================================================
//                     GLOBAL FUNCTIONS FOR THE IMPLICIT SOURCE TERM SOLVER
//============================================================================================================================
///--------------------------------------------------------
///  This is a dummy function
///--------------------------------------------------------
void Jacobian(int* n, double* x, double* y, double* dfy,
              int* ldfy, double* rpar, double* ipar) {
}
///--------------------------------------------------------
///  This is a dummy function
///--------------------------------------------------------
void Mass(int* n, double* am, int* lmas, int* rpar, int* ipar) {
}
///--------------------------------------------------------
///  This is a dummy function
///--------------------------------------------------------
void solout(int* n, double* am, int* lmas, int* rpar, int* ipar) {
}
//-------------------------------------------------------------------------------
//              Right Hand-side of the system of equation
//-------------------------------------------------------------------------------
///---------------------------------------------------------
///   This function establishes the matrix corresponding
///   to dy[i]/dt = f[i]
///---------------------------------------------------------
void ImplicitSource(int* n, double* x, double* y, double* fy,
                    double* rpar, int* ipar) {
	double Coeffs_fi, Coeffs_ri;
	double S, E;
	int i, k, kprim;
	double c_k, k_fi, k_ri;

	ImSolver1.T = y[ImSolver1.Nb_spec];
	fy[ImSolver1.Nb_spec] = 0;

	for (k = 0; k < ImSolver1.Nb_spec; ++k) {
		fy[k] = 0;
		ImSolver1.Y_k[k] = y[k];
		ImSolver1.C[k] = y[k] * ImSolver1.Rho / (ImSolver1.M[k]);
		for (i = 0; i < ImSolver1.Nb_reac; ++i) {
			Coeffs_fi = 1;
			Coeffs_ri = 1;
			/// If an Arrhenius-type reaction then run this part
			if (ImSolver1.Reac_type[i] == "Arrhenius") {
				for (int kprim = 0; kprim < ImSolver1.Nb_spec; ++kprim) {
					c_k = ImSolver1.Rho * y[kprim] / (double)ImSolver1.M[kprim];
					/// IF THE STOECHIOMETRIC COEFFICIENT IS NEGATIVE, IT IS A REACTANTS
					/// THEREFORE IT WILL AFFECT FORWARD REACTION RATE
					if (ImSolver1.Stoechio_coeff[i][kprim] < 0) Coeffs_fi *= pow(c_k, ImSolver1.Reac_order[i][kprim]);
					/// ENZYMES INTERVEENING IN THE REACTION RATES
					if (ImSolver1.Stoechio_coeff[i][kprim] == 0 && ImSolver1.Reac_order[i][kprim] < 0) Coeffs_fi *= pow(c_k, fabs(ImSolver1.Reac_order[i][kprim]));
					/// IF THE STOECHIOMETRIC COEFFICIENT IS POSITIVE, IT IS A PRODUCT
					/// THEREFORE IT WILL AFFECT BACKWARD REACTION RATE
					if (ImSolver1.Stoechio_coeff[i][kprim] > 0) Coeffs_ri *= pow(c_k, ImSolver1.Reac_order[i][kprim]);
				}
				k_fi = ImSolver1.Reac_coeff[i][0] * pow(ImSolver1.T, ImSolver1.Reac_coeff[i][1])
				       * exp(-ImSolver1.Reac_coeff[i][2] / (R_GAS * ImSolver1.T));
				k_ri = ImSolver1.Reac_coeff[i][4];

				fy[ImSolver1.Nb_spec] += ImSolver1.Reac_coeff[i][3] * (k_fi * Coeffs_fi - k_ri * Coeffs_ri) / (ImSolver1.Cp * ImSolver1.Rho);
				fy[k] += ImSolver1.M[k] * ImSolver1.Stoechio_coeff[i][k] * (k_fi * Coeffs_fi - k_ri * Coeffs_ri) / ImSolver1.Rho;
			}
			/// If a Michaelis\Menten reaction then run this part
			S = 0;
			E = 0;
			if (ImSolver1.Reac_type[i] == "Michaelis") {
				for (kprim = 0; kprim < ImSolver1.Nb_spec; ++kprim) {
					if (ImSolver1.Reac_order[i][kprim] == 1 && ImSolver1.Y_k[kprim] > 0) {
						S = ImSolver1.Y_k[kprim];
					}
					if (ImSolver1.Reac_order[i][kprim] == -1 && ImSolver1.Y_k[kprim] > 0) {
						E = ImSolver1.Y_k[kprim];
					}
				}
				Coeffs_fi = ImSolver1.Reac_coeff[i][0] * E * S / (ImSolver1.Reac_coeff[i][1] + S);
				if (Coeffs_fi < 0) {
					std::cout << " NOOOOOOOOOOO ";
				}
				fy[k] += (ImSolver1.M[k] * ImSolver1.Stoechio_coeff[i][k] * Coeffs_fi) / ImSolver1.Rho;
			}
		}
	}
}
void GetDataImSolver(double* Y_k, double T, double Rho, double Cp) {
	for (int i = 0; i < ImSolver1.Nb_spec; ++i) {
		y[i] = Y_k[i];
	}
	ImSolver1.Rho = Rho;
	ImSolver1.Cp = Cp;
	y[ImSolver1.Nb_spec] = T;
}
void PutDataBackImSolver(double* Y_k, double* w_k, double T, double& w_T) {
	///______________________________________________________________________________________________________///
	///                                           PUT DATA BACK                                              ///
	///______________________________________________________________________________________________________///
	for (int i = 0; i < ImSolver1.Nb_spec; ++i) {
		w_k[i] = y[i] - Y_k[i];
	}
	w_T = y[ImSolver1.Nb_spec] - T;
}
void Stiff_Source(Species_solver* Species, Flow_solver* Flow, Thermal_solver* Thermal, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		int X, Y, Z;
		for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (Species->solid_species[{X, Y, Z}] == FALSE) {
						Imsolver_t = 0;
						Imsolver_t_end = global_parameters.D_t;
						Imsolver_h = global_parameters.D_t;
						GetDataImSolver(Species->mass_fraction[X][Y][Z], Thermal->T_0 * Thermal->temperature[{X, Y, Z}], Flow->rho_0 * Flow->density[{X, Y, Z}], Thermal->c_p[{X, Y, Z}]);
						RADAU(&npp, ImplicitSource, &Imsolver_t, y, &Imsolver_t_end, &Imsolver_h,
						      &rtoler, &atoler, &itoler,
						      Jacobian, &ijac, &mljac, &mujac,
						      Mass, &imas, &mlmas, &mumas,
						      solout, &iout,
						      work, &lwork, iwork, &liwork, &rpar, &ipar, &idid);
						PutDataBackImSolver(Species->mass_fraction[X][Y][Z], Species->Production[{X, Y, Z}], Thermal->temperature[{X, Y, Z}], Thermal->Production[{X, Y, Z}]);
					}
				}
			}
		}
	}
}
#endif  // defined

Thickened_flame::Thickened_flame() {};
Thickened_flame::~Thickened_flame() {};
void Thickened_flame::General_data_input(const std::string& filename, Parallel_MPI* MPI_parallel) {
	int column_width = 40;
	/// Open input file
	string input_filename(filename);
	input_filename += ".dat";
	ifstream input_file;  // File is open for READING
	input_file.open(input_filename.c_str(), ios::binary);

	input_file.clear();
	input_file.seekg(0, ios::beg);
	find_line_after_header(input_file, "c\tThickened Flame Model");
	find_line_after_comment(input_file);
	input_file >> delta_0 >> delta_1 >> S_L >> T_fresh >> T_burnt >> sigma;
	delta_0 /= global_parameters.D_x;
	delta_1 /= global_parameters.D_x;
	filter_width /= global_parameters.D_x;
	S_L *= (global_parameters.D_t / global_parameters.D_x);
	F = delta_1 / delta_0;
	input_file.close();
	if (MPI_parallel->processor_id == MASTER + 1) {
		std::cout << "Tickened Flame" << endl;
		std::cout << "=====================" << endl;
		std::cout << setw(column_width) << left << "Flame real thickness : " << delta_0 << endl;
		std::cout << setw(column_width) << left << "Thickness flame thickness : " << delta_1 << endl;
		std::cout << setw(column_width) << left << "Flame speed : " << S_L << endl;
		std::cout << setw(column_width) << left << "F : " << F << endl;
		std::cout << setw(column_width) << left << "Tf : " << T_fresh << endl;
		std::cout << setw(column_width) << left << "Tb : " << T_burnt << endl;
		std::cout << setw(column_width) << left << "Sigma : " << sigma << endl;
		std::cout << endl;
	}
	Memory_allocation(MPI_parallel);
}
void Thickened_flame::Memory_allocation(Parallel_MPI* MPI_parallel) {
	unsigned int X, Y, Z;
	if (MPI_parallel->processor_id != MASTER) {
		Efficiency = new double**[MPI_parallel->dev_end[0]];
		for (X = 0; X < MPI_parallel->dev_end[0]; ++X) {
			Efficiency[X] = new double*[MPI_parallel->dev_end[1]];
			for (Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
				Efficiency[X][Y] = new double[MPI_parallel->dev_end[2]];
				for (Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
					Efficiency[X][Y][Z] = 1.;
				}
			}
		}
	}
}
void Thickened_flame::Get_efficiency(Flow_solver* Flow, Thermal_solver* Thermal, Species_solver* Species, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		/// I - GO OVER ALL FLUID NODES
		unsigned int X, Y, Z;
		for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (solid_species[{X, Y, Z}] == FALSE) {
						/* subgrid fluctuating velocity */
						// int Xtemp = X; int Ytemp = Y; int Ztemp = Z;
						// data lapU0 =    Flow->velocity[X+1][Y][Z][0] - 2.*Flow->velocity[X][Y][Z][0] + Flow->velocity[X-1][Y][Z][0]
						//               + Flow->velocity[X][Y+1][Z][0] - 2.*Flow->velocity[X][Y][Z][0] + Flow->velocity[X][Y-1][Z][0]
						//               + Flow->velocity[X][Y][Z+1][0] - 2.*Flow->velocity[X][Y][Z][0] + Flow->velocity[X][Y][Z-1][0];
						// data lapV0 =    Flow->velocity[X+1][Y][Z][1] - 2.*Flow->velocity[X][Y][Z][1] + Flow->velocity[X-1][Y][Z][1]
						//               + Flow->velocity[X][Y+1][Z][1] - 2.*Flow->velocity[X][Y][Z][1] + Flow->velocity[X][Y-1][Z][1]
						//               + Flow->velocity[X][Y][Z+1][1] - 2.*Flow->velocity[X][Y][Z][1] + Flow->velocity[X][Y][Z-1][1];
						// data lapW0 =    Flow->velocity[X+1][Y][Z][2] - 2.*Flow->velocity[X][Y][Z][2] + Flow->velocity[X-1][Y][Z][2]
						//               + Flow->velocity[X][Y+1][Z][2] - 2.*Flow->velocity[X][Y][Z][2] + Flow->velocity[X][Y-1][Z][2]
						//               + Flow->velocity[X][Y][Z+1][2] - 2.*Flow->velocity[X][Y][Z][2] + Flow->velocity[X][Y][Z-1][2];
						// X = Xtemp + 1; Y = Ytemp; Z = Ztemp;
						// data lapUxp =   Flow->velocity[X+1][Y][Z][0] - 2.*Flow->velocity[X][Y][Z][0] + Flow->velocity[X-1][Y][Z][0]
						//               + Flow->velocity[X][Y+1][Z][0] - 2.*Flow->velocity[X][Y][Z][0] + Flow->velocity[X][Y-1][Z][0]
						//               + Flow->velocity[X][Y][Z+1][0] - 2.*Flow->velocity[X][Y][Z][0] + Flow->velocity[X][Y][Z-1][0];
						// data lapVxp =   Flow->velocity[X+1][Y][Z][1] - 2.*Flow->velocity[X][Y][Z][1] + Flow->velocity[X-1][Y][Z][1]
						//               + Flow->velocity[X][Y+1][Z][1] - 2.*Flow->velocity[X][Y][Z][1] + Flow->velocity[X][Y-1][Z][1]
						//               + Flow->velocity[X][Y][Z+1][1] - 2.*Flow->velocity[X][Y][Z][1] + Flow->velocity[X][Y][Z-1][1];
						// data lapWxp =   Flow->velocity[X+1][Y][Z][2] - 2.*Flow->velocity[X][Y][Z][2] + Flow->velocity[X-1][Y][Z][2]
						//               + Flow->velocity[X][Y+1][Z][2] - 2.*Flow->velocity[X][Y][Z][2] + Flow->velocity[X][Y-1][Z][2]
						//               + Flow->velocity[X][Y][Z+1][2] - 2.*Flow->velocity[X][Y][Z][2] + Flow->velocity[X][Y][Z-1][2];
						// X = Xtemp - 1; Y = Ytemp; Z = Ztemp;
						// data lapUxn =   Flow->velocity[X+1][Y][Z][0] - 2.*Flow->velocity[X][Y][Z][0] + Flow->velocity[X-1][Y][Z][0]
						//               + Flow->velocity[X][Y+1][Z][0] - 2.*Flow->velocity[X][Y][Z][0] + Flow->velocity[X][Y-1][Z][0]
						//               + Flow->velocity[X][Y][Z+1][0] - 2.*Flow->velocity[X][Y][Z][0] + Flow->velocity[X][Y][Z-1][0];
						// data lapVxn =   Flow->velocity[X+1][Y][Z][1] - 2.*Flow->velocity[X][Y][Z][1] + Flow->velocity[X-1][Y][Z][1]
						//               + Flow->velocity[X][Y+1][Z][1] - 2.*Flow->velocity[X][Y][Z][1] + Flow->velocity[X][Y-1][Z][1]
						//               + Flow->velocity[X][Y][Z+1][1] - 2.*Flow->velocity[X][Y][Z][1] + Flow->velocity[X][Y][Z-1][1];
						// data lapWxn =   Flow->velocity[X+1][Y][Z][2] - 2.*Flow->velocity[X][Y][Z][2] + Flow->velocity[X-1][Y][Z][2]
						//               + Flow->velocity[X][Y+1][Z][2] - 2.*Flow->velocity[X][Y][Z][2] + Flow->velocity[X][Y-1][Z][2]
						//               + Flow->velocity[X][Y][Z+1][2] - 2.*Flow->velocity[X][Y][Z][2] + Flow->velocity[X][Y][Z-1][2];
						// X = Xtemp; Y = Ytemp + 1; Z = Ztemp;
						// data lapUyp =   Flow->velocity[X+1][Y][Z][0] - 2.*Flow->velocity[X][Y][Z][0] + Flow->velocity[X-1][Y][Z][0]
						//               + Flow->velocity[X][Y+1][Z][0] - 2.*Flow->velocity[X][Y][Z][0] + Flow->velocity[X][Y-1][Z][0]
						//               + Flow->velocity[X][Y][Z+1][0] - 2.*Flow->velocity[X][Y][Z][0] + Flow->velocity[X][Y][Z-1][0];
						// data lapVyp =   Flow->velocity[X+1][Y][Z][1] - 2.*Flow->velocity[X][Y][Z][1] + Flow->velocity[X-1][Y][Z][1]
						//               + Flow->velocity[X][Y+1][Z][1] - 2.*Flow->velocity[X][Y][Z][1] + Flow->velocity[X][Y-1][Z][1]
						//               + Flow->velocity[X][Y][Z+1][1] - 2.*Flow->velocity[X][Y][Z][1] + Flow->velocity[X][Y][Z-1][1];
						// data lapWyp =   Flow->velocity[X+1][Y][Z][2] - 2.*Flow->velocity[X][Y][Z][2] + Flow->velocity[X-1][Y][Z][2]
						//               + Flow->velocity[X][Y+1][Z][2] - 2.*Flow->velocity[X][Y][Z][2] + Flow->velocity[X][Y-1][Z][2]
						//               + Flow->velocity[X][Y][Z+1][2] - 2.*Flow->velocity[X][Y][Z][2] + Flow->velocity[X][Y][Z-1][2];
						// X = Xtemp; Y = Ytemp - 1; Z = Ztemp;
						// data lapUyn =   Flow->velocity[X+1][Y][Z][0] - 2.*Flow->velocity[X][Y][Z][0] + Flow->velocity[X-1][Y][Z][0]
						//               + Flow->velocity[X][Y+1][Z][0] - 2.*Flow->velocity[X][Y][Z][0] + Flow->velocity[X][Y-1][Z][0];
						//               + Flow->velocity[X][Y][Z+1][0] - 2.*Flow->velocity[X][Y][Z][0] + Flow->velocity[X][Y][Z-1][0];
						// data lapVyn =   Flow->velocity[X+1][Y][Z][1] - 2.*Flow->velocity[X][Y][Z][1] + Flow->velocity[X-1][Y][Z][1]
						//               + Flow->velocity[X][Y+1][Z][1] - 2.*Flow->velocity[X][Y][Z][1] + Flow->velocity[X][Y-1][Z][1]
						//               + Flow->velocity[X][Y][Z+1][1] - 2.*Flow->velocity[X][Y][Z][1] + Flow->velocity[X][Y][Z-1][1];
						// data lapWyn =   Flow->velocity[X+1][Y][Z][2] - 2.*Flow->velocity[X][Y][Z][2] + Flow->velocity[X-1][Y][Z][2]
						//               + Flow->velocity[X][Y+1][Z][2] - 2.*Flow->velocity[X][Y][Z][2] + Flow->velocity[X][Y-1][Z][2]
						//               + Flow->velocity[X][Y][Z+1][2] - 2.*Flow->velocity[X][Y][Z][2] + Flow->velocity[X][Y][Z-1][2];
						// X = Xtemp; Y = Ytemp; Z = Ztemp + 1;
						// data lapUzp =   Flow->velocity[X+1][Y][Z][0] - 2.*Flow->velocity[X][Y][Z][0] + Flow->velocity[X-1][Y][Z][0]
						//               + Flow->velocity[X][Y+1][Z][0] - 2.*Flow->velocity[X][Y][Z][0] + Flow->velocity[X][Y-1][Z][0]
						//               + Flow->velocity[X][Y][Z+1][0] - 2.*Flow->velocity[X][Y][Z][0] + Flow->velocity[X][Y][Z-1][0];
						// data lapVzp =   Flow->velocity[X+1][Y][Z][1] - 2.*Flow->velocity[X][Y][Z][1] + Flow->velocity[X-1][Y][Z][1]
						//               + Flow->velocity[X][Y+1][Z][1] - 2.*Flow->velocity[X][Y][Z][1] + Flow->velocity[X][Y-1][Z][1]
						//               + Flow->velocity[X][Y][Z+1][1] - 2.*Flow->velocity[X][Y][Z][1] + Flow->velocity[X][Y][Z-1][1];
						// data lapWzp =   Flow->velocity[X+1][Y][Z][2] - 2.*Flow->velocity[X][Y][Z][2] + Flow->velocity[X-1][Y][Z][2]
						//               + Flow->velocity[X][Y+1][Z][2] - 2.*Flow->velocity[X][Y][Z][2] + Flow->velocity[X][Y-1][Z][2]
						//               + Flow->velocity[X][Y][Z+1][2] - 2.*Flow->velocity[X][Y][Z][2] + Flow->velocity[X][Y][Z-1][2];
						// X = Xtemp; Y = Ytemp; Z = Ztemp + 1;
						// data lapUzn =   Flow->velocity[X+1][Y][Z][0] - 2.*Flow->velocity[X][Y][Z][0] + Flow->velocity[X-1][Y][Z][0]
						//               + Flow->velocity[X][Y+1][Z][0] - 2.*Flow->velocity[X][Y][Z][0] + Flow->velocity[X][Y-1][Z][0]
						//               + Flow->velocity[X][Y][Z+1][0] - 2.*Flow->velocity[X][Y][Z][0] + Flow->velocity[X][Y][Z-1][0];
						// data lapVzn =   Flow->velocity[X+1][Y][Z][1] - 2.*Flow->velocity[X][Y][Z][1] + Flow->velocity[X-1][Y][Z][1]
						//               + Flow->velocity[X][Y+1][Z][1] - 2.*Flow->velocity[X][Y][Z][1] + Flow->velocity[X][Y-1][Z][1]
						//               + Flow->velocity[X][Y][Z+1][1] - 2.*Flow->velocity[X][Y][Z][1] + Flow->velocity[X][Y][Z-1][1];
						// data lapWzn =   Flow->velocity[X+1][Y][Z][2] - 2.*Flow->velocity[X][Y][Z][2] + Flow->velocity[X-1][Y][Z][2]
						//               + Flow->velocity[X][Y+1][Z][2] - 2.*Flow->velocity[X][Y][Z][2] + Flow->velocity[X][Y-1][Z][2]
						//               + Flow->velocity[X][Y][Z+1][2] - 2.*Flow->velocity[X][Y][Z][2] + Flow->velocity[X][Y][Z-1][2];
						// X = Xtemp; Y = Ytemp; Z = Ztemp;

						// data rotX = 0.5 * (lapWyp - lapWyn) - 0.5 * (lapVzp - lapVzn);
						// data rotY = 0.5 * (lapUzp - lapUzn) - 0.5 * (lapWxp - lapWxn);
						// data rotZ = 0.5 * (lapUyp - lapUyn) - 0.5 * (lapVxp - lapVxn);
						// data u_p = 2.*sqrt(pow(rotX,2) + pow(rotY,2) + pow(rotZ,2) );
						/* Smago constant */ /* model constants */
						// data C_k = 1.5; data b = 1.4; data gamma = 0.5;
						/* subgrid Reynolds */
						// data Re_delta = 4. * F * (u_p/S_L);
						// data a = 0.6 + 0.2 * exp(-0.1*u_p/S_L) - 0.2 * exp(-0.01*F);
						// data f_u = 4. * sqrt(27.*C_k/110.) * (18.*C_k/55.) * pow(u_p/S_L, 2);
						// data f_Re = sqrt( (9./55.) * exp(-1.5*C_k*pow(M_PI, 4./3.)/Re_delta ) ) * sqrt(Re_delta);
						// data f_delta = sqrt( (27./110.)*C_k*pow(M_PI,4./3.) * ( pow(F,4./3.) - 1.) );
						// data Gamma_delta = pow( pow(pow( pow(f_u, -a) + pow(f_delta, -a) , -1./a), -b) + pow(f_Re, -b) , -1./b);
						double X_delta = 1.;  // pow(1. + MIN(F - 1., Gamma_delta*u_p/S_L) , gamma);
						Efficiency[X][Y][Z] = X_delta;
					}
				}
			}
		}
	}
}
void Thickened_flame::Data_Exchange(Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		unsigned int Nparams = MPI_parallel->buffer_size * 1;
		///--------------------------------------------------------->  In Y-direction
		static double* buf_totop_TF = new double[MPI_parallel->dev_end[0] * MPI_parallel->dev_end[2] * Nparams];
		static double* buf_tobottom_TF = new double[MPI_parallel->dev_end[0] * MPI_parallel->dev_end[2] * Nparams];
		static double* buf_fromtop_TF = new double[MPI_parallel->dev_end[0] * MPI_parallel->dev_end[2] * Nparams];
		static double* buf_frombottom_TF = new double[MPI_parallel->dev_end[0] * MPI_parallel->dev_end[2] * Nparams];
		// Prepare messages to be sent
		for (unsigned int X = 0; X < MPI_parallel->dev_end[0]; X++) {
			for (unsigned int Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
				memcpy(buf_totop_TF + X * MPI_parallel->dev_end[2] * Nparams + Z * Nparams + 0,
				       &Efficiency[X][MPI_parallel->end_XYZ2[1]][Z], sizeof(double));
				memcpy(buf_totop_TF + X * MPI_parallel->dev_end[2] * Nparams + Z * Nparams + 1,
				       &Efficiency[X][MPI_parallel->end_XYZ2[1] - 1][Z], sizeof(double));

				memcpy(buf_tobottom_TF + X * MPI_parallel->dev_end[2] * Nparams + Z * Nparams + 0,
				       &Efficiency[X][MPI_parallel->start_XYZ2[1]][Z], sizeof(double));
				memcpy(buf_tobottom_TF + X * MPI_parallel->dev_end[2] * Nparams + Z * Nparams + 1,
				       &Efficiency[X][MPI_parallel->start_XYZ2[1] + 1][Z], sizeof(double));
			}
		}
		int Bottom_neighbour, Top_neighbour;
		Top_neighbour = MPI_parallel->proc_arrangement[MPI_parallel->proc_position[0]][(MPI_parallel->proc_position[1] + 1) % MPI_parallel->Np_Y][MPI_parallel->proc_position[2]];
		Bottom_neighbour = MPI_parallel->proc_arrangement[MPI_parallel->proc_position[0]][(MPI_parallel->proc_position[1] + MPI_parallel->Np_Y - 1) % MPI_parallel->Np_Y][MPI_parallel->proc_position[2]];
		////Send-recv all totop+fromt
		MPI_Sendrecv(buf_totop_TF, Nparams * MPI_parallel->dev_end[0] * MPI_parallel->dev_end[2], MPI_DOUBLE, Top_neighbour, LTAG,
		             buf_frombottom_TF, Nparams * MPI_parallel->dev_end[0] * MPI_parallel->dev_end[2], MPI_DOUBLE, Bottom_neighbour, LTAG,
		             MPI_COMM_WORLD, &status);
		// Send-recv all toleft+fromleft
		MPI_Sendrecv(buf_tobottom_TF, Nparams * MPI_parallel->dev_end[0] * MPI_parallel->dev_end[2], MPI_DOUBLE, Bottom_neighbour, RTAG,
		             buf_fromtop_TF, Nparams * MPI_parallel->dev_end[0] * MPI_parallel->dev_end[2], MPI_DOUBLE, Top_neighbour, RTAG,
		             MPI_COMM_WORLD, &status);

		for (unsigned int X = 0; X < MPI_parallel->dev_end[0]; X++) {
			for (unsigned int Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
				memcpy(&Efficiency[X][MPI_parallel->start_XYZ2[1] - 1][Z],
				       buf_frombottom_TF + X * MPI_parallel->dev_end[2] * Nparams + Z * Nparams + 0,
				       sizeof(double));
				memcpy(&Efficiency[X][MPI_parallel->start_XYZ2[1] - 2][Z],
				       buf_frombottom_TF + X * MPI_parallel->dev_end[2] * Nparams + Z * Nparams + 1,
				       sizeof(double));

				memcpy(&Efficiency[X][MPI_parallel->end_XYZ2[1] + 1][Z],
				       buf_fromtop_TF + X * MPI_parallel->dev_end[2] * Nparams + Z * Nparams + 0,
				       sizeof(double));
				memcpy(&Efficiency[X][MPI_parallel->end_XYZ2[1] + 2][Z],
				       buf_fromtop_TF + X * MPI_parallel->dev_end[2] * Nparams + Z * Nparams + 1,
				       sizeof(double));
			}
		}
		///////// Exchange data with neighbors /////////
		static double* buf_toleft_TF = new double[MPI_parallel->dev_end[1] * MPI_parallel->dev_end[2] * Nparams];
		static double* buf_toright_TF = new double[MPI_parallel->dev_end[1] * MPI_parallel->dev_end[2] * Nparams];
		static double* buf_fromleft_TF = new double[MPI_parallel->dev_end[1] * MPI_parallel->dev_end[2] * Nparams];
		static double* buf_fromright_TF = new double[MPI_parallel->dev_end[1] * MPI_parallel->dev_end[2] * Nparams];
		///--------------------------------------------------------->  In X-direction
		// Prepare messages to be sent
		for (unsigned int Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
			for (unsigned int Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
				memcpy(buf_toright_TF + Y * MPI_parallel->dev_end[2] * Nparams + Z * Nparams + 0,
				       &Efficiency[MPI_parallel->end_XYZ2[0]][Y][Z], sizeof(double));
				memcpy(buf_toright_TF + Y * MPI_parallel->dev_end[2] * Nparams + Z * Nparams + 1,
				       &Efficiency[MPI_parallel->end_XYZ2[0] - 1][Y][Z], sizeof(double));

				memcpy(buf_toleft_TF + Y * MPI_parallel->dev_end[2] * Nparams + Z * Nparams + 0,
				       &Efficiency[MPI_parallel->start_XYZ2[0]][Y][Z], sizeof(double));
				memcpy(buf_toleft_TF + Y * MPI_parallel->dev_end[2] * Nparams + Z * Nparams + 1,
				       &Efficiency[MPI_parallel->start_XYZ2[0] + 1][Y][Z], sizeof(double));
			}
		}
		int Left_neighbour, Right_neighbour;
		Right_neighbour = MPI_parallel->proc_arrangement[(MPI_parallel->proc_position[0] + 1) % MPI_parallel->Np_X][MPI_parallel->proc_position[1]][MPI_parallel->proc_position[2]];
		Left_neighbour = MPI_parallel->proc_arrangement[(MPI_parallel->proc_position[0] + MPI_parallel->Np_X - 1) % MPI_parallel->Np_X][MPI_parallel->proc_position[1]][MPI_parallel->proc_position[2]];
		////Send-recv all toright+fromleft
		MPI_Sendrecv(buf_toright_TF, Nparams * MPI_parallel->dev_end[1] * MPI_parallel->dev_end[2], MPI_DOUBLE, Right_neighbour, LTAG,
		             buf_fromleft_TF, Nparams * MPI_parallel->dev_end[1] * MPI_parallel->dev_end[2], MPI_DOUBLE, Left_neighbour, LTAG,
		             MPI_COMM_WORLD, &status);
		// Send-recv all toleft+fromright
		MPI_Sendrecv(buf_toleft_TF, Nparams * MPI_parallel->dev_end[1] * MPI_parallel->dev_end[2], MPI_DOUBLE, Left_neighbour, RTAG,
		             buf_fromright_TF, Nparams * MPI_parallel->dev_end[1] * MPI_parallel->dev_end[2], MPI_DOUBLE, Right_neighbour, RTAG,
		             MPI_COMM_WORLD, &status);
		// Postprocess messages
		for (int Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
			for (int Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
				memcpy(&Efficiency[MPI_parallel->start_XYZ2[0] - 1][Y][Z], buf_fromleft_TF + Y * MPI_parallel->dev_end[2] * Nparams + Z * Nparams + 0, sizeof(double));
				memcpy(&Efficiency[MPI_parallel->start_XYZ2[0] - 2][Y][Z], buf_fromleft_TF + Y * MPI_parallel->dev_end[2] * Nparams + Z * Nparams + 8, sizeof(double));

				memcpy(&Efficiency[MPI_parallel->end_XYZ2[0] + 1][Y][Z], buf_fromright_TF + Y * MPI_parallel->dev_end[2] * Nparams + Z * Nparams + 0, sizeof(double));
				memcpy(&Efficiency[MPI_parallel->end_XYZ2[0] + 2][Y][Z], buf_fromright_TF + Y * MPI_parallel->dev_end[2] * Nparams + Z * Nparams + 8, sizeof(double));
			}
		}
		///--------------------------------------------------------->  In Z-direction
		static double* buf_tofront_TF = new double[MPI_parallel->dev_end[0] * MPI_parallel->dev_end[1] * Nparams];
		static double* buf_torear_TF = new double[MPI_parallel->dev_end[0] * MPI_parallel->dev_end[1] * Nparams];
		static double* buf_fromfront_TF = new double[MPI_parallel->dev_end[0] * MPI_parallel->dev_end[1] * Nparams];
		static double* buf_fromrear_TF = new double[MPI_parallel->dev_end[0] * MPI_parallel->dev_end[1] * Nparams];
		// Prepare messages to be sent
		for (int X = 0; X < MPI_parallel->dev_end[0]; X++) {
			for (int Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
				memcpy(buf_tofront_TF + X * MPI_parallel->dev_end[1] * Nparams + Y * Nparams + 0,
				       &Efficiency[X][Y][MPI_parallel->end_XYZ2[2]], sizeof(double));
				memcpy(buf_tofront_TF + X * MPI_parallel->dev_end[1] * Nparams + Y * Nparams + 8,
				       &Efficiency[X][Y][MPI_parallel->end_XYZ2[2] - 1], sizeof(double));

				memcpy(buf_torear_TF + X * MPI_parallel->dev_end[1] * Nparams + Y * Nparams + 0,
				       &Efficiency[X][Y][MPI_parallel->start_XYZ2[2]], sizeof(double));
				memcpy(buf_torear_TF + X * MPI_parallel->dev_end[1] * Nparams + Y * Nparams + 8,
				       &Efficiency[X][Y][MPI_parallel->start_XYZ2[2] + 1], sizeof(double));
			}
		}
		int Rear_neighbour, Front_neighbour;
		Front_neighbour = MPI_parallel->proc_arrangement[MPI_parallel->proc_position[0]][MPI_parallel->proc_position[1]][(MPI_parallel->proc_position[2] + 1) % MPI_parallel->Np_Z];
		Rear_neighbour = MPI_parallel->proc_arrangement[MPI_parallel->proc_position[0]][MPI_parallel->proc_position[1]][(MPI_parallel->proc_position[2] + MPI_parallel->Np_Z - 1) % MPI_parallel->Np_Z];
		////Send-recv all totop+fromt
		MPI_Sendrecv(buf_tofront_TF, Nparams * MPI_parallel->dev_end[0] * MPI_parallel->dev_end[1], MPI_DOUBLE, Front_neighbour, LTAG,
		             buf_fromrear_TF, Nparams * MPI_parallel->dev_end[0] * MPI_parallel->dev_end[1], MPI_DOUBLE, Rear_neighbour, LTAG,
		             MPI_COMM_WORLD, &status);
		// Send-recv all toleft+fromleft
		MPI_Sendrecv(buf_torear_TF, Nparams * MPI_parallel->dev_end[0] * MPI_parallel->dev_end[1], MPI_DOUBLE, Rear_neighbour, RTAG,
		             buf_fromfront_TF, Nparams * MPI_parallel->dev_end[0] * MPI_parallel->dev_end[1], MPI_DOUBLE, Front_neighbour, RTAG,
		             MPI_COMM_WORLD, &status);
		for (int X = 0; X < MPI_parallel->dev_end[0]; X++) {
			for (int Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
				memcpy(&Efficiency[X][Y][MPI_parallel->start_XYZ2[2] - 1], buf_fromrear_TF + X * MPI_parallel->dev_end[1] * Nparams + Y * Nparams + 0, sizeof(double));
				memcpy(&Efficiency[X][Y][MPI_parallel->start_XYZ2[2] - 2], buf_fromrear_TF + X * MPI_parallel->dev_end[1] * Nparams + Y * Nparams + 8, sizeof(double));

				memcpy(&Efficiency[X][Y][MPI_parallel->end_XYZ2[2] + 1], buf_fromfront_TF + X * MPI_parallel->dev_end[1] * Nparams + Y * Nparams + 0, sizeof(double));
				memcpy(&Efficiency[X][Y][MPI_parallel->end_XYZ2[2] + 2], buf_fromfront_TF + X * MPI_parallel->dev_end[1] * Nparams + Y * Nparams + 8, sizeof(double));
			}
		}
	}
}
void Thickened_flame::Apply_filter(Flow_solver* Flow, Thermal_solver* Thermal, Species_solver* Species, Parallel_MPI* MPI_parallel) {
	Get_efficiency(Flow, Thermal, Species, MPI_parallel);
	Data_Exchange(MPI_parallel);
	if (MPI_parallel->processor_id != MASTER) {
		/// I - GO OVER ALL FLUID NODES
		for (int X = 0; X < MPI_parallel->dev_end[0]; ++X) {
			for (int Y = 0; Y < MPI_parallel->dev_end[1]; ++Y) {
				for (int Z = 0; Z < MPI_parallel->dev_end[2]; ++Z) {
					if (solid_species[{X, Y, Z}] == FALSE) {
						double c_prog = (T_burnt / Thermal->T_0 - Thermal->temperature[{X, Y, Z}]) / (T_burnt / Thermal->T_0 - T_fresh / Thermal->T_0);
						c_prog = MAX(MIN(c_prog, 1.), 0.);
						// double Omega = 16. * pow(c_prog * (1 - c_prog), 2);
						double F_adaptiv = F;  // 1. + (F - 1.) * tanh(sigma*Omega);
						double X_delta = Efficiency[X][Y][Z];
						for (int k = 0; k < Species->Nb_spec; ++k) {
							Species->diffusion_coefficient[{X, Y, Z, k}] *= (X_delta * F_adaptiv);
							Species->Production[{X, Y, Z, k}] *= (X_delta / F_adaptiv);
						}
						Thermal->thermal_diffusion_coefficient[{X, Y, Z}] *= (X_delta * F_adaptiv);
						Thermal->Production[{X, Y, Z}] *= (X_delta / F_adaptiv);
					}
				}
			}
		}
		for (int i = 0; i < Species->Boundaries.size(); ++i) {
			int X = Species->Boundaries[i].X;
			int Y = Species->Boundaries[i].Y;
			int Z = Species->Boundaries[i].Z;
			int Xp = X - Species->Boundaries[i].n[0];
			int Yp = Y - Species->Boundaries[i].n[1];
			int Zp = Z - Species->Boundaries[i].n[2];
			for (int k = 0; k < Species->Nb_spec; ++k) {
				Species->diffusion_coefficient[{X, Y, Z, k}] = Species->diffusion_coefficient[{Xp, Yp, Zp, k}];
				Species->Production[{X, Y, Z, k}] = 0;
			}
		}
		for (int i = 0; i < Thermal->Boundaries.size(); ++i) {
			int X = Thermal->Boundaries[i].X;
			int Y = Thermal->Boundaries[i].Y;
			int Z = Thermal->Boundaries[i].Z;
			int Xp = X - Thermal->Boundaries[i].n[0];
			int Yp = Y - Thermal->Boundaries[i].n[1];
			int Zp = Z - Thermal->Boundaries[i].n[2];
			Thermal->thermal_diffusion_coefficient[{X, Y, Z}] = Thermal->thermal_diffusion_coefficient[{Xp, Yp, Zp}];
			Thermal->Production[{X, Y, Z}] = 0.0;
		}
	}
}

#if defined IMPLEX && !defined EXT_LIB
//============================================================================================================================
//                     GLOBAL FUNCTIONS FOR THE IMPLICIT SOURCE TERM SOLVER
//============================================================================================================================
///--------------------------------------------------------
///  This is a dummy function
///--------------------------------------------------------
void Jacobian(int* n, double* x, double* y, double* dfy,
              int* ldfy, double* rpar, double* ipar) {
}
///--------------------------------------------------------
///  This is a dummy function
///--------------------------------------------------------
void Mass(int* n, double* am, int* lmas, int* rpar, int* ipar) {
}
///--------------------------------------------------------
///  This is a dummy function
///--------------------------------------------------------
void solout(int* n, double* am, int* lmas, int* rpar, int* ipar) {
}
//-------------------------------------------------------------------------------
//              Right Hand-side of the system of equation
//-------------------------------------------------------------------------------
///---------------------------------------------------------
///   This function establishes the matrix corresponding
///   to dy[i]/dt = f[i]
///---------------------------------------------------------
void ImplicitSource(int* n, double* x, double* y, double* fy,
                    double* rpar, int* ipar) {
	double Coeffs_fi, Coeffs_ri;
	double S, E;
	int i, k, kprim;
	double c_k, k_fi, k_ri;

	ImSolver1.T = y[ImSolver1.Nb_spec];
	fy[ImSolver1.Nb_spec] = 0;

	for (k = 0; k < ImSolver1.Nb_spec; ++k) {
		fy[k] = 0;
		ImSolver1.Y_k[k] = y[k];
		ImSolver1.C[k] = y[k] * ImSolver1.Rho / (ImSolver1.M[k]);
		for (i = 0; i < ImSolver1.Nb_reac; ++i) {
			Coeffs_fi = 1;
			Coeffs_ri = 1;
			/// If an Arrhenius-type reaction then run this part
			if (ImSolver1.Reac_type[i] == "Arrhenius") {
				for (int kprim = 0; kprim < ImSolver1.Nb_spec; ++kprim) {
					c_k = ImSolver1.Rho * y[kprim] / (double)ImSolver1.M[kprim];
					/// IF THE STOECHIOMETRIC COEFFICIENT IS NEGATIVE, IT IS A REACTANTS
					/// THEREFORE IT WILL AFFECT FORWARD REACTION RATE
					if (ImSolver1.Stoechio_coeff[i][kprim] < 0) Coeffs_fi *= pow(c_k, ImSolver1.Reac_order[i][kprim]);
					/// ENZYMES INTERVEENING IN THE REACTION RATES
					if (ImSolver1.Stoechio_coeff[i][kprim] == 0 && ImSolver1.Reac_order[i][kprim] < 0) Coeffs_fi *= pow(c_k, fabs(ImSolver1.Reac_order[i][kprim]));
					/// IF THE STOECHIOMETRIC COEFFICIENT IS POSITIVE, IT IS A PRODUCT
					/// THEREFORE IT WILL AFFECT BACKWARD REACTION RATE
					if (ImSolver1.Stoechio_coeff[i][kprim] > 0) Coeffs_ri *= pow(c_k, ImSolver1.Reac_order[i][kprim]);
				}
				k_fi = ImSolver1.Reac_coeff[i][0] * pow(ImSolver1.T, ImSolver1.Reac_coeff[i][1])
				       * exp(-ImSolver1.Reac_coeff[i][2] / (R_GAS * ImSolver1.T));
				k_ri = ImSolver1.Reac_coeff[i][4];

				fy[ImSolver1.Nb_spec] += ImSolver1.Reac_coeff[i][3] * (k_fi * Coeffs_fi - k_ri * Coeffs_ri) / (ImSolver1.Cp * ImSolver1.Rho);
				fy[k] += ImSolver1.M[k] * ImSolver1.Stoechio_coeff[i][k] * (k_fi * Coeffs_fi - k_ri * Coeffs_ri) / ImSolver1.Rho;
			}
			/// If a Michaelis\Menten reaction then run this part
			S = 0;
			E = 0;
			if (ImSolver1.Reac_type[i] == "Michaelis") {
				for (kprim = 0; kprim < ImSolver1.Nb_spec; ++kprim) {
					if (ImSolver1.Reac_order[i][kprim] == 1 && ImSolver1.Y_k[kprim] > 0) {
						S = ImSolver1.Y_k[kprim];
					}
					if (ImSolver1.Reac_order[i][kprim] == -1 && ImSolver1.Y_k[kprim] > 0) {
						E = ImSolver1.Y_k[kprim];
					}
				}
				Coeffs_fi = ImSolver1.Reac_coeff[i][0] * E * S / (ImSolver1.Reac_coeff[i][1] + S);
				if (Coeffs_fi < 0) {
					std::cout << " NOOOOOOOOOOO ";
				}
				fy[k] += (ImSolver1.M[k] * ImSolver1.Stoechio_coeff[i][k] * Coeffs_fi) / ImSolver1.Rho;
			}
		}
	}
}
void GetDataImSolver(double* Y_k, double T, double Rho, double Cp) {
	for (int i = 0; i < ImSolver1.Nb_spec; ++i) {
		y[i] = Y_k[i];
	}
	ImSolver1.Rho = Rho;
	ImSolver1.Cp = Cp;
	y[ImSolver1.Nb_spec] = T;
}
void PutDataBackImSolver(double* Y_k, double* w_k, double T, double& w_T) {
	///______________________________________________________________________________________________________///
	///                                           PUT double BACK                                              ///
	///______________________________________________________________________________________________________///
	for (int i = 0; i < ImSolver1.Nb_spec; ++i) {
		w_k[i] = y[i] - Y_k[i];
	}
	w_T = y[ImSolver1.Nb_spec] - T;
}
void Stiff_Source(Species_solver* Species, Flow_solver* Flow, Thermal_solver* Thermal, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		int X, Y, Z;
		for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (Species->is_solid[X][Y][Z] == FALSE) {
						Imsolver_t = 0;
						Imsolver_t_end = global_parameters.D_t;
						Imsolver_h = global_parameters.D_t;
						GetDataImSolver(Species->mass_fraction[X][Y][Z], Thermal->T_0 * Thermal->temperature[X][Y][Z], Flow->rho_0 * Flow->density[X][Y][Z], Thermal->c_p[X][Y][Z]);
						RADAU(&npp, ImplicitSource, &Imsolver_t, y, &Imsolver_t_end, &Imsolver_h,
						      &rtoler, &atoler, &itoler,
						      Jacobian, &ijac, &mljac, &mujac,
						      Mass, &imas, &mlmas, &mumas,
						      solout, &iout,
						      work, &lwork, iwork, &liwork, &rpar, &ipar, &idid);
						PutDataBackImSolver(Species->mass_fraction[X][Y][Z], Species->Production[X][Y][Z], Thermal->temperature[X][Y][Z], Thermal->Production[X][Y][Z]);
					}
				}
			}
		}
	}
}
#endif  // defined

void Species_solver::species_mole_fractions(unsigned int time_step, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->is_master()) {
		return;
	}
	ofstream output_file("Alborz_Results/debug/species_mole_fractions.dat", fstream::app);
	output_file << time_step << " ";

	double total_molar_fractions[Nb_spec] = {0.0};
	double total_mass_fraction[Nb_spec] = {0.0};
	int total_nodes = 0;
	// Calculate cumulative sums and count total nodes
	for (int X = 0; X < MPI_parallel->dev_end[0]; ++X) {
		for (int Y = 0; Y < MPI_parallel->dev_end[1]; ++Y) {
			for (int Z = 0; Z < MPI_parallel->dev_end[2]; ++Z) {
				for (int k = 0; k < Nb_spec; ++k) {
					double mass_fraction_k = mass_fraction[{X, Y, Z, k}];
					double molar_mass_k = Molar_mass[k];
					total_molar_fractions[k] += mass_fraction_k / molar_mass_k;
					total_mass_fraction[k] += mass_fraction_k;
				}
				total_nodes++;
			}
		}
	}
	// Calculate the sum of (w_i / M_i) for all Species
	double total_sum_wi_Mi = 0.0;
	for (int k = 0; k < Nb_spec; ++k) {
		total_sum_wi_Mi += total_mass_fraction[k] / Molar_mass[k];
	}

	// Calculate the mole fractions for each Species and write to the file
	for (int k = 0; k < Nb_spec; ++k) {
		double mole_fraction = (total_mass_fraction[k] / Molar_mass[k]) / total_sum_wi_Mi;
		output_file << mole_fraction << " ";
	}

	for (int k = 0; k < Nb_spec; ++k) {
		double average_mass_fraction = total_mass_fraction[k] / total_nodes;
		output_file << average_mass_fraction << " ";
	}

	output_file << endl;
	output_file.close();
}
void Species_solver::mass_conservation_report(int time, Flow_solver* Flow, Thermal_solver* Thermal, Parallel_MPI* MPI_parallel) {
	int k, X, Y, Z;
	double sumY = 0.0;
	double sumYV = 0.0;
	double correction_factor = 0.0;
	if (MPI_parallel->processor_id != MASTER) {
		for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (solid_species[{X, Y, Z}] == FALSE) {
						// Check and correct the condition ∑(Y_k * V_k) = 0
						for (k = 0; k < Nb_spec; ++k) {
							sumY += mass_fraction[{X, Y, Z, k}];
							sumYV += mass_fraction[{X, Y, Z, k}] * (Flux[X][Y][Z][k][0] + Flux[X][Y][Z][k][1] + Flux[X][Y][Z][k][2]);
						}
						// Correct if necessary
						// if (sumYV != 0.0) {
						//	for (k = 0; k < Nb_spec; ++k) {
						// double correction_factor = -sumYV / (Flux[X][Y][Z][k][0] + Flux[X][Y][Z][k][1] + Flux[X][Y][Z][k][2]);
						// mass_fraction[{X, Y, Z, k}] += correction_factor * (Flux[X][Y][Z][k][0] + Flux[X][Y][Z][k][1] + Flux[X][Y][Z][k][2]);
						//	}
						//}
					}
				}
			}
		}
	}
	if (MPI_parallel->processor_id == MASTER) {
		/// Create filename
		stringstream output_filename;
		output_filename << "Alborz_Results/debug/massconservation.dat";
		ofstream output_file;
		/// Open file
		output_file.open(output_filename.str().c_str(), fstream::app);
		/// Write data
		output_file << setprecision(30) << time << "\t";  // time step
		output_file << setprecision(30) << "SumY: " << sumY << std::endl
					<< "SumYV: " << sumYV << correction_factor << std::endl;
		/// Close file
		output_file.close();
	}
	return;
}
void Species_solver::calculateLewisNumber(Flow_solver* Flow, Thermal_solver* Thermal, Species_solver* Species, Parallel_MPI* MPI_parallel, int output, int time) {
	std::vector<double> cal_lewis_nu_local(Nb_spec, 0.0);
	std::vector<double> cal_lewis_nu_global(Nb_spec, 0.0);

	int total_cells_processed = 0;  // Total number of cells processed across all processors

	if (MPI_parallel->processor_id != MASTER) {
		for (int X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (int Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (int Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (solid_species[{X, Y, Z}] == FALSE) {
						++total_cells_processed;  // Increment the count for each valid cell processed
						for (int k = 0; k < Nb_spec; k++) {
							double lewis = Thermal->thermal_diffusion_coefficient[{X, Y, Z}] / (diffusion_coefficient[{X, Y, Z, k}] * Thermal->c_p[{X, Y, Z}]);
							cal_lewis_nu_local[k] += lewis;
						}
					}
				}
			}
		}
	}

	// Sum up the total number of cells processed across all processors
	MPI_Allreduce(MPI_IN_PLACE, &total_cells_processed, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

	// Sum up the local Lewis numbers across all processors
	MPI_Allreduce(cal_lewis_nu_local.data(), cal_lewis_nu_global.data(), Nb_spec, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	// Calculate average Lewis numbers
	for (int k = 0; k < Nb_spec; ++k) {
		cal_lewis_nu_global[k] /= total_cells_processed;
	}

	// Write average Lewis numbers to file
	if (MPI_parallel->processor_id == MASTER && output == 1) {
		/// Create filename
		stringstream output_filename;
		output_filename << "Alborz_Results/debug/lewis_number.dat";
		ofstream output_file;
		/// Open file
		output_file.open(output_filename.str().c_str(), fstream::app);

		// Write header if the file is being created for the first time
		if (output_file.tellp() == 0) {
			output_file << "TimeStep ";
			for (int k = 0; k < Nb_spec; ++k) {
				output_file << "\t" << species_name_RG[k];
			}
			output_file << std::endl;
		}

		/// Write data
		output_file << time;
		for (int k = 0; k < Nb_spec; ++k) {
			output_file << "\t" << setprecision(15) << cal_lewis_nu_global[k];
		}
		output_file << endl;

		/// Close file
		output_file.close();
	}
}

void Species_solver::initialize_field_FD_TGV_reactive(Vector_field& Y_k, Vector_field& diffusion_coefficient, Scalar_field& molar_mass_av, Solid_field& solid, Geometry* Geo, stl_import* Geo_stl, Flow_solver* Flow, Thermal_solver* Thermal, Parallel_MPI* MPI_parallel, const std::string& filename, Thermo_chemistry_cantera* thermo_chemistry) {
	const unsigned int Nb = Nb_spec;
	COMP = 0;
	if (MPI_parallel->processor_id != MASTER) {
		// Initialize solid Species based on STL domain if available
		if (Geo_stl->flag == 1) {
			for (unsigned int X = 0; X < MPI_parallel->dev_end[0]; ++X) {
				for (unsigned int Y = 0; Y < MPI_parallel->dev_end[1]; ++Y) {
					for (unsigned int Z = 0; Z < MPI_parallel->dev_end[2]; ++Z) {
						solid[{X, Y, Z}] = Geo_stl->domain[{X, Y, Z}];
						Thermal->solid_thermal_type[{X, Y, Z}] = Geo_stl->domain[{X, Y, Z}];
					}
				}
			}
		}
		// Reading initial conditions from the file
		double stiffness, xflamepos, ini_temp, radius;
		double flamepos, radial_dist, ref_tanh, xx;
		bool is_reactive = true;
		std::vector<double> composition_in(Nb), composition_out(Nb);
		species_name_RG.resize(Nb_spec);
		std::string input_filename = filename + ".dat";
		std::ifstream input_file(input_filename.c_str(), std::ios::binary);
		input_file.seekg(0, std::ios::beg);
		find_line_after_header(input_file, "c\tSpecies Initial Profile File");
		find_line_after_comment(input_file);
		// Read general parameters and fraction type indicator
		char fraction_type;
		input_file >> stiffness >> xflamepos >> ini_temp >> fraction_type;
		// Load mole fractions or mass fractions
		find_line_after_comment(input_file);
		for (unsigned int k = 0; k < Nb_spec; ++k) {
			input_file >> species_name_RG[k] >> composition_in[k];
		}
		find_line_after_comment(input_file);
		for (unsigned int k = 0; k < Nb_spec; ++k) {
			input_file >> species_name_RG[k] >> composition_out[k];
		}
		input_file.close();
		// Initialize fields based on the imported data
		for (unsigned int X = 0; X < MPI_parallel->dev_end[0]; ++X) {
			xx = fmod(X - MPI_parallel->start_XYZ2[0] + MPI_parallel->start_XYZ[0] + global_parameters.Nx, global_parameters.Nx) * global_parameters.D_x;
			
			flamepos = xflamepos * global_parameters.Nx * global_parameters.D_x;
			radius = global_parameters.Nx * global_parameters.D_x / 8.0;
			radial_dist = fabs(xx - flamepos);
			ref_tanh = 0.5 * (1.0 + tanh(stiffness * (radial_dist - radius) / radius));

			for (unsigned int Y = 0; Y < MPI_parallel->dev_end[1]; ++Y) {
				for (unsigned int Z = 0; Z < MPI_parallel->dev_end[2]; ++Z) {
					solid_species[{X, Y, Z}] = -1;  // Mark as solid
					std::vector<double> Y_temp(Nb);
					double test_one = 0.0;
					// Calculate and normalize mass fractions
					for (unsigned int j = 0; j < Nb; ++j) {
						Y_k[{X, Y, Z, j}] = composition_in[j] * (1.0 - ref_tanh) + composition_out[j] * ref_tanh;
						test_one += Y_k[{X, Y, Z, j}];
						diffusion_coefficient[{X, Y, Z, j}] = 1e-5;
					}
					for (unsigned int j = 0; j < Nb; ++j) {
						Y_k[{X, Y, Z, j}] /= test_one;
						Y_temp[j] = Y_k[{X, Y, Z, j}];  // Store normalized mass fractions for Cantera
					}
					double T_in = ini_temp;
					double P = Flow->p_th_0;
#if defined compressible
					P = Flow->density[{X, Y, Z}] * Flow->rho_0 * R_GAS * T_in / (molar_mass_av[{X, Y, Z}]);
#endif
					thermo_chemistry->sol->thermo()->setState_TPY(T_in, P, Y_temp.data());
					thermo_chemistry->sol->thermo()->equilibrate("HP");
					// Mass fractions initialization
					std::vector<double> massf_cantera(Nb);
					thermo_chemistry->sol->thermo()->getMassFractions(massf_cantera.data());
					for (unsigned int j = 0; j < Nb; ++j) {
						Y_k[{X, Y, Z, j}] = massf_cantera[j];
					}
					molar_mass_av[{X, Y, Z}] = thermo_chemistry->sol->thermo()->meanMolecularWeight() * 1e-3;
					double T_eq = thermo_chemistry->sol->thermo()->temperature();
					Thermal->temperature[{X, Y, Z}] = T_eq;
					Thermal->c_p[{X, Y, Z}] = thermo_chemistry->sol->thermo()->cp_mass();
					Thermal->thermal_diffusion_coefficient[{X, Y, Z}] = thermo_chemistry->sol->transport()->thermalConductivity();
					solid[{X, Y, Z}] = -1;
					V_c[{X, Y, Z, 0}] = 0;
					V_c[{X, Y, Z, 1}] = 0;
					V_c[{X, Y, Z, 2}] = 0;
					for (unsigned int k = 0; k < Nb; ++k) {
						Production[{X, Y, Z, k}] = 0.;
						Flux[X][Y][Z][k][0] = 0.;
						Flux[X][Y][Z][k][1] = 0.;
						Flux[X][Y][Z][k][2] = 0.;
						previous_mass_fraction[{X, Y, Z, k}] = Y_k[{X, Y, Z, k}];
					}
					molar_mass_av[{X, Y, Z}] = 1.0 / molar_mass_av[{X, Y, Z}];  // right now the unit of mm_av is kg/mol and we need it in mol/kg for the solver
				}
			}
		}
		if (MPI_parallel->processor_id == (MASTER + 1)) {
			std::cout << "SPECIES and THERMAL: reacting TGV-3D initial conditions \n";
			std::cout << "================================ \n";
			std::cout << "Stiffness: " << stiffness << std::endl;
			std::cout << "Radius: " << radius << std::endl;
			std::cout << "Flame Position: " << xflamepos << " which is: " << flamepos << " m " << std::endl;
			std::cout << "Initial Temperature: " << ini_temp << std::endl;
			std::cout << "The equation used for the initial profile is: Yk = Yk1*(1-tanh) + Yk2*tanh \n";
			std::cout << "Where tanh = 0.5 * (1 + tanh(stiffness2 * (radial_dist - radius) / radius)) \n";
			std::cout << "================================ \n";
		}
	}
}