#ifndef REGATH_INTERFACE_H_INCLUDED
#define REGATH_INTERFACE_H_INCLUDED

#ifdef REGATH_LIB

#include "Flow_solver.h"
#include "Thermal_solver.h"
#include "Species_solver.h"
#include "ALBORZ_SETTINGS.h"
#include "Fluid_read_write.h"
#include "ALBORZ_Macros.h"
#include "utils/Config_utils.h"
#if defined IMPLEX
#include "Radau/radau.h"
#endif
/// **********************************************************************************************************************************
///                               EXTERN FORTRAN FUNCTIONS FROM REGATH
/// **********************************************************************************************************************************
//                               CALLED FROM REGATH_INTERFACE ( mod_regath_interface )
//                               The code calling these function should be compiled with all of
//                               the REGATH libraries
extern "C" {
extern void __mod_regath_test_MOD_aaa(int*);
extern void __mod_regath_library_MOD_regath_initialize_thermo(int*, int*, char (*)[17], int*, int*, int*, int*);
extern void __mod_regath_interface_MOD_open_file(int*, int*, char (*)[30], char (*)[30]);
extern void __mod_regath_library_MOD_regath_universal_constants(double*, double*, double*);
extern void __mod_regath_library_MOD_regath_initialize_transport(int*, int*);
extern void __mod_regath_library_MOD_regath_species_number(char (*)[17], int*, int*);
extern void __mod_regath_interface_MOD_regath_species_names_cpp(char*, int*);
extern void __mod_regath_interface_MOD_regath_spec_mole_weight_cpp(int*, double*);
/// EVALUATION OF SPECIFIC HEAT CAPACITIES AND ENTHALPIES
extern void __mod_regath_interface_MOD_regath_cp_mixt_mass_cpp(int*, int*, double*, double*, double*);
extern void __mod_regath_interface_MOD_regath_cp_spec_mass_cpp(int*, int*, double*, double*);

extern void __mod_regath_interface_MOD_regath_cv_mixt_mass_cpp(int*, int*, double*, double*, double*);
extern void __mod_regath_interface_MOD_regath_enthalpy_spec_mass_cpp(int*, int*, double*, double*);
extern void __mod_regath_interface_MOD_regath_int_energy_spec_mass_cpp(int*, int*, double*, double*);
/// EVALUATION OF MIXITURE AVERAGE MOLAR MASS
extern void __mod_regath_interface_MOD_regath_mixt_mole_weight_cpp(int*, int*, double*, double*);
/// CONVERSION OF MASS TO MOLE FRACTION AND CONCENTRATION
extern void __mod_regath_interface_MOD_regath_mass_to_mole_frac_cpp(int*, int*, double*, double*);
extern void __mod_regath_interface_MOD_regath_mole_to_mass_frac_cpp(int*, int*, double*, double*);
extern void __mod_regath_interface_MOD_regath_mass_to_concentration_cpp(int*, int*, double*, double*, double*, double*);
/// EVALUATION OF SPECIES AND ENERGY PRODUCTION RATE
extern void __mod_regath_interface_MOD_regath_production_rate_from_y_cpp(int*, int*, double*, double*, double*, double*);
extern void __mod_regath_interface_MOD_regath_production_rate_cpp(int*, int*, double*, double*, double*);
/// EVALUATION OF THERMAL CONDUCTIVITY
extern void __mod_regath_interface_MOD_regath_conductivity_mixt_cpp(int*, int*, double*, double*, double*);
/// EVALUATION OF VISCOSITY
extern void __mod_regath_interface_MOD_regath_viscosity_polynomial_mixt_cpp(int*, int*, double*, double*, double*);  /// last argument is dynamic viscosity
extern void __mod_regath_interface_MOD_regath_viscosity_wilke_mixt_cpp(int*, int*, double*, double*, double*);       /// last argument is dynamic viscosity
/// EVALUATION OF SPECIES DIFFUSION COEFFICIENTS
extern void __mod_regath_interface_MOD_regath_diffusivity_spec_in_mixt_cpp(int*, int*, double*, double*, double*, double*);
extern void __mod_regath_interface_MOD_regath_diffusivity_binary_coefs_cpp(int*, int*, double*, double*, double*);

extern void __mod_regath_interface_MOD_ltstep(int*, int*, double*, double*, double*, double*, double*, double*, double*);
}

struct temp_data {
	double *Y_temp, *density_temp, *enthalpy_temp, *C_temp, *T_temp, *Cp_temp, *M_temp, *w_temp, M_bar_temp;
	int Nb_spec, np;
};
#if defined IMPLEX && defined REGATH_LIB
temp_data ImSolver;
///_________________________________________________________________________________________________///
///                                  RADAU IMPLICIT SOLVER INITIALIZATION                           ///
///_________________________________________________________________________________________________///
double Imsolver_t, Imsolver_t_end, Imsolver_h;
int npp;  // This is dimension of the implicit problem = (Nb_spec+1)global_parameters.Nx*global_parameters.Ny
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
	ImSolver.T_temp[0] = y[ImSolver.Nb_spec];
	for (int i = 0; i < ImSolver.Nb_spec; ++i) {
		ImSolver.Y_temp[i] = y[i];
		if (ImSolver.Y_temp[i] < 0) ImSolver.Y_temp[i] = 1e-50;
		ImSolver.C_temp[i] = ImSolver.Y_temp[i] * ImSolver.density_temp[0] / ImSolver.M_temp[i];
	}
	///______________________________________________________________________________________________________///
	///                           GET THERMOCHEMICAL PROPERTIES                                              ///
	///______________________________________________________________________________________________________///
	__mod_regath_interface_MOD_regath_production_rate_cpp(&(ImSolver.Nb_spec), &(ImSolver.np), &(ImSolver.T_temp[0]), ImSolver.C_temp, ImSolver.w_temp);
	__mod_regath_interface_MOD_regath_cp_mixt_mass_cpp(&(ImSolver.Nb_spec), &(ImSolver.np), &(ImSolver.T_temp[0]), ImSolver.Y_temp, &(ImSolver.Cp_temp[0]));
	__mod_regath_interface_MOD_regath_enthalpy_spec_mass_cpp(&(ImSolver.Nb_spec), &(ImSolver.np), &(ImSolver.T_temp[0]), ImSolver.enthalpy_temp);
	///______________________________________________________________________________________________________///
	///                             PUT DATA INTO FIELD VARIABLE                                             ///
	///______________________________________________________________________________________________________///
	fy[ImSolver.Nb_spec] = 0;
	for (int i = 0; i < ImSolver.Nb_spec; ++i) {
		fy[i] = ImSolver.M_temp[i] * ImSolver.w_temp[i] / ImSolver.density_temp[0];
		fy[ImSolver.Nb_spec] -= (ImSolver.enthalpy_temp[i] * fy[i]);
	}
	fy[ImSolver.Nb_spec] *= 1 / (double)(ImSolver.Cp_temp[0]);
}
void GetDataImSolver(double* Y_k, double T, double Rho) {
	for (int i = 0; i < ImSolver.Nb_spec; ++i) {
		y[i] = Y_k[i];
	}
	ImSolver.density_temp[0] = Rho;
	y[ImSolver.Nb_spec] = T;
}
void PutDataBackImSolver(double* Y_k, double* w_k, double T, double& w_T) {
	///______________________________________________________________________________________________________///
	///                                           PUT DATA BACK                                              ///
	///______________________________________________________________________________________________________///
	for (int i = 0; i < ImSolver.Nb_spec; ++i) {
		w_k[i] = y[i] - Y_k[i];
	}
	w_T = (y[ImSolver.Nb_spec] - T);
}
/*
This function is used to solve the stiff source term using the RADAU5 implicit solver from the ODEPACK library
It solves the system of ODEs dy[i]/dt = f[i] where f[i] is the right hand side of the system of equations
The system of equations is solved for each node in the domain
*/
void StiffSource(Species_solver* Species, Flow_solver* Flow, Thermal_solver* Thermal, Parallel_MPI* MPI_parallel) {
	if (processor_id != MASTER) {
		double T, rho, P;
		int X, Y, Z, k, alpha, xx;
		for (X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (Flow->is_solid[{X, Y, Z}] == FALSE) {  /// if a node has at least one fluid neighbor
						T = Thermal->temperature[{X, Y, Z}] * Thermal->T_0;
						rho = Flow->density[{X, Y, Z}] * Flow->rho_0;
						P = Flow->p_th;
						Imsolver_t = 0;
						Imsolver_t_end = global_parameters.D_t;
						Imsolver_h = global_parameters.D_t;
						Thermal->Production[{X,Y,Z}] = 0;
						GetDataImSolver(Species->mass_fraction[{X, Y, Z}], T, rho); /// Get the initial values for the Species mass fractions
						RADAU5(&npp, ImplicitSource, &Imsolver_t, y, &Imsolver_t_end, &Imsolver_h,
						       &rtoler, &atoler, &itoler,
						       Jacobian, &ijac, &mljac, &mujac,
						       Mass, &imas, &mlmas, &mumas,
						       solout, &iout,
						       work, &lwork, iwork, &liwork, &rpar, &ipar, &idid);
						PutDataBackImSolver(Species->mass_fraction[{X, Y, Z}], Species->Production[{X,Y,Z}], T, Thermal->Production[{X,Y,Z}]);
						///______________________________________________________________________________________________________///
						///                           GET THERMOCHEMICAL PROPERTIES                                              ///
						///______________________________________________________________________________________________________///
						__mod_regath_interface_MOD_regath_cp_mixt_mass_cpp(&Species->Nb_spec, &npp, &T, Species->mass_fraction[{X, Y, Z}], &(Thermal->c_p[{X, Y, Z}]));
						Thermal->Production[{X,Y,Z}] = Thermal->Production[{X,Y,Z}] * Flow->density[{X, Y, Z}] * Flow->rho_0 * Thermal->c_p[{X, Y, Z}];
						for (k = 0; k < Species->Nb_spec; k++) {
							Species->Production[{X,Y,Z,k}] = Species->Production[{X,Y,Z,k}] * Flow->density[{X, Y, Z}] * Flow->rho_0;
						}
					}
				}
			}
		}
		for (X = MPI_parallel->start_XYZ2[0] - 1; X <= MPI_parallel->end_XYZ2[0] + 1; ++X) {
			for (Y = MPI_parallel->start_XYZ2[1] - 1; Y <= MPI_parallel->end_XYZ2[1] + 1; ++Y) {
				for (Z = MPI_parallel->start_XYZ2[2] - 1; Z <= MPI_parallel->end_XYZ2[2] + 1; ++Z) {
					if ((Flow->is_solid[{X, Y, Z}]
					     + Flow->is_solid[{X + 1, Y, Z}] + Flow->is_solid[{X - 1, Y, Z}]
					     + Flow->is_solid[{X, Y + 1, Z}] + Flow->is_solid[{X, Y - 1, Z}]
					     + Flow->is_solid[{X, Y, Z + 1}] + Flow->is_solid[{X, Y, Z - 1}])
					    < 7) {  /// if a node has at least one fluid neighbor
						__mod_regath_interface_MOD_regath_mixt_mole_weight_cpp(&Species->Nb_spec, &npp, Species->mass_fraction[{X, Y, Z}], &(Species->molar_mass_av[{X,Y,Z}]));
						__mod_regath_interface_MOD_regath_cp_mixt_mass_cpp(&Species->Nb_spec, &npp, &T, Species->mass_fraction[{X, Y, Z}], &(Thermal->c_p[{X, Y, Z}]));
					}
				}
			}
		}
	}
}
#endif
class ThermoChemistry {
private:
	int first_run = 0;
	int lout_user_thermo = 6, link_thermo = 15, Nb_elem, Nb_reac, Nb_polynom, lout_user_transp = 6, link_transport = 25, np;
	double Gas_const, Gas_const_cal;
	char cstring[17] = "SI              ";
	double *C_k, *h_k, *X_k;
	double T, P;
	double* Le;
	double cp_temp;

public:
	double p_atm, *mol_mass_RG;
	int Nb_spec;
	void Transport_properties_REGATH(Flow_solver* Flow, Thermal_solver* Thermal, Species_solver* Species, Parallel_MPI* MPI_parallel) {
		if (MPI_parallel->processor_id != MASTER) {
			int X, Y, Z, k;
			///________________________________________________________________________________________________
			///                 REGTATH: compute Transport properties for each Species / and the mixture
			///________________________________________________________________________________________________
			for (X = 0; X < MPI_parallel->dev_end[0]; ++X) {
				for (Y = 0; Y < MPI_parallel->dev_end[1]; ++Y) {
					for (Z = 0; Z < MPI_parallel->dev_end[2]; ++Z) {
						if (Flow->is_solid[{X, Y, Z}] == FALSE) {  /// if a node has at least one fluid neighbor
							                                       ///______________________________________________________________________________________________________///
							                                       ///                              TRANSFER DATA TO TEMPORARY ARRAYS                                       ///
							                                       ///______________________________________________________________________________________________________///
							T = Thermal->temperature[{X, Y, Z}] * Thermal->T_0;
#if defined compressible
							P = Flow->density[{X, Y, Z}] * Flow->rho_0 * R_GAS * T / (Species->molar_mass_av[{X,Y,Z}]);
#endif  // defined
#if defined LMNA_solver
							P = Flow->p_th_0;
#endif  // defined
        ///______________________________________________________________________________________________________///
        ///                           GET TRANSPORT PROPERTIES                                                   ///
        ///______________________________________________________________________________________________________///
							//__mod_regath_interface_MOD_regath_mass_to_mole_frac_cpp(&Nb_spec, &np, Species->mass_fraction[{X, Y, Z}], X_k);
							__mod_regath_interface_MOD_regath_mass_to_mole_frac_cpp(&Nb_spec, &np, &(Species->mass_fraction[{X, Y, Z}]), X_k);
							__mod_regath_interface_MOD_regath_conductivity_mixt_cpp(&Nb_spec, &np, &T, X_k, &(Thermal->diffusion_coefficient[{X,Y,Z}]));
#if defined Polynomial_viscosity
							__mod_regath_interface_MOD_regath_viscosity_polynomial_mixt_cpp(&Nb_spec, &np, &T, X_k, &(Flow->viscosity[{X, Y, Z}]));
#endif
#if defined Wilke_viscosity
							__mod_regath_interface_MOD_regath_viscosity_wilke_mixt_cpp(&Nb_spec, &np, &T, X_k, &(Flow->viscosity[{X, Y, Z}]));
#endif
#if defined MixtureAveraged_Species_diffusion
							__mod_regath_interface_MOD_regath_diffusivity_spec_in_mixt_cpp(&Nb_spec, &np, &P, &T, X_k, Species->diffusion_coefficient[{X,Y,Z}]);
#endif
							///______________________________________________________________________________________________________///
							///                           NON-DIMENSIONALIZE                                                         ///
							///______________________________________________________________________________________________________///
							Flow->viscosity[{X, Y, Z}] *= (global_parameters.D_t / (global_parameters.D_x * global_parameters.D_x * Flow->density[{X, Y, Z}] * Flow->rho_0));
							Thermal->diffusion_coefficient[{X,Y,Z}] *= (global_parameters.D_t / (global_parameters.D_x * global_parameters.D_x));
							for (k = 0; k < Species->Nb_spec; k++) {
#if defined UnitLewis_Species_diffusion
								Species->diffusion_coefficient[{X,Y,Z,k}] = Thermal->diffusion_coefficient[{X,Y,Z}] / (Thermal->c_p[{X, Y, Z}]);
#endif
#if defined ConstLewis_Species_diffusion
								Species->diffusion_coefficient[{X,Y,Z,k}] = Thermal->diffusion_coefficient[{X,Y,Z}] / (Le[k] * Thermal->c_p[{X, Y, Z}]);
#endif
#if defined MixtureAveraged_Species_diffusion
								Species->diffusion_coefficient[{X,Y,Z,k}] *= Flow->rho_0 * Flow->density[{X, Y, Z}] * (global_parameters.D_t / (global_parameters.D_x * global_parameters.D_x));
#endif
							}
						}
					}
				}
			}
			//unsigned Xp, Yp, Zp;
			for (unsigned int i = 0; i < Flow->boundaries.size(); ++i) {
				int X = Flow->boundaries[i].X;
				int Y = Flow->boundaries[i].Y;
				int Z = Flow->boundaries[i].Z;
				const int Xp = (X - Flow->boundaries[i].n[0]);
				const int Yp = (Y - Flow->boundaries[i].n[1]);
				const int Zp = (Z - Flow->boundaries[i].n[2]);
				///______________________________________________________________________________________________________///
				///                              TRANSFER DATA TO TEMPORARY ARRAYS                                       ///
				///______________________________________________________________________________________________________///
				T = Thermal->temperature[{Xp, Yp, Zp}] * Thermal->T_0;
				P = Flow->density[{X, Y, Z}] * Flow->rho_0 * R_GAS * T / (Species->molar_mass_av[{X,Y,Z}]);
				///______________________________________________________________________________________________________///
				///                           GET TRANSPORT PROPERTIES                                                   ///
				///______________________________________________________________________________________________________///
				__mod_regath_interface_MOD_regath_mass_to_mole_frac_cpp(&Nb_spec, &np, &(Species->mass_fraction[{Xp,Yp,Zp}]), X_k);
				__mod_regath_interface_MOD_regath_conductivity_mixt_cpp(&Nb_spec, &np, &T, X_k, &(Thermal->diffusion_coefficient[{Xp,Yp,Zp}]));
#if defined Polynomial_viscosity
				__mod_regath_interface_MOD_regath_viscosity_polynomial_mixt_cpp(&Nb_spec, &np, &T, X_k, &(Flow->viscosity[{Xp, Yp, Zp}]));
#endif
#if defined Wilke_viscosity
				__mod_regath_interface_MOD_regath_viscosity_wilke_mixt_cpp(&Nb_spec, &np, &T, X_k, &(Flow->viscosity[{Xp, Yp, Zp}]));
#endif
#if defined MixtureAveraged_Species_diffusion
				__mod_regath_interface_MOD_regath_diffusivity_spec_in_mixt_cpp(&Nb_spec, &np, &P, &T, X_k, Species->diffusion_coefficient[{Xp,Yp,Zp}]);
#endif
				///______________________________________________________________________________________________________///
				///                           NON-DIMENSIONALIZE                                                         ///
				///______________________________________________________________________________________________________///
				Flow->viscosity[{Xp, Yp, Zp}] *= (global_parameters.D_t / (global_parameters.D_x * global_parameters.D_x * Flow->density[{Xp, Yp, Zp}] * Flow->rho_0));
				Thermal->diffusion_coefficient[{Xp,Yp,Zp}] *= (global_parameters.D_t / (global_parameters.D_x * global_parameters.D_x));
				for (k = 0; k < Species->Nb_spec; k++) {
#if defined UnitLewis_Species_diffusion
					Species->diffusion_coefficient[{Xp,Yp,Zp,k}] = Thermal->diffusion_coefficient[{Xp,Yp,Zp}] / (Thermal->c_p[{X, Y, Z}]);
#endif
#if defined ConstLewis_Species_diffusion
					Species->diffusion_coefficient[{Xp,Yp,Zp,k}] = Thermal->diffusion_coefficient[{Xp,Yp,Zp}] / (Le[k] * Thermal->c_p[{X, Y, Z}]);
#endif
#if defined MixtureAveraged_Species_diffusion
					Species->diffusion_coefficient[{Xp,Yp,Zp,k}] *= Flow->rho_0 * Flow->density[{X, Y, Z}] * (global_parameters.D_t / (global_parameters.D_x * global_parameters.D_x));
#endif
				}
			}
		}
	}
	void Thermo_properties_REGATH(Flow_solver* Flow, Thermal_solver* Thermal, Species_solver* Species, Parallel_MPI* MPI_parallel) {
		if (MPI_parallel->processor_id != MASTER) {
			int X, Y, Z, k;
			for (X = 0; X < MPI_parallel->dev_end[0]; ++X) {
				for (Y = 0; Y < MPI_parallel->dev_end[1]; ++Y) {
					for (Z = 0; Z < MPI_parallel->dev_end[2]; ++Z) {
						if (Flow->is_solid[{X, Y, Z}] == FALSE) {  /// if a node has at least one fluid neighbor
							                                       ///______________________________________________________________________________________________________///
							                                       ///                              TRANSFER DATA TO TEMPORARY ARRAYS                                       ///
							                                       ///______________________________________________________________________________________________________///
							for (k = 0; k < Nb_spec; ++k) {
								C_k[k] = Flow->rho_0 * Flow->density[{X, Y, Z}] * Species->mass_fraction[{X,Y,Z,k}] / Species->Molar_mass[k];
								if (C_k[k] < 0) C_k[k] = 0;
								Species->Production[{X,Y,Z,k}] = 0;
							}
							Thermal->Production[{X,Y,Z}] = 0;
							T = Thermal->temperature[{X, Y, Z}] * Thermal->T_0;
							///______________________________________________________________________________________________________///
							///                           GET THERMOCHEMICAL PROPERTIES                                              ///
							///______________________________________________________________________________________________________///
							__mod_regath_interface_MOD_regath_mixt_mole_weight_cpp(&Nb_spec, &np, &(Species->mass_fraction[{X, Y, Z}]), &(Species->molar_mass_av[{X,Y,Z}]));
							__mod_regath_interface_MOD_regath_production_rate_cpp(&Nb_spec, &np, &T, C_k, &(Species->Production[{X,Y,Z}]));
							__mod_regath_interface_MOD_regath_cp_mixt_mass_cpp(&Nb_spec, &np, &T, &(Species->mass_fraction[{X, Y, Z}]), &(Thermal->c_p[{X, Y, Z}]));
							__mod_regath_interface_MOD_regath_enthalpy_spec_mass_cpp(&Nb_spec, &np, &T, h_k);
							///______________________________________________________________________________________________________///
							///                             PUT DATA INTO FIELD VARIABLE                                             ///
							///______________________________________________________________________________________________________///
							for (k = 0; k < Nb_spec; ++k) {
								Species->Production[{X,Y,Z,k}] *= (global_parameters.D_t * Species->Molar_mass[k]);
								Thermal->Production[{X,Y,Z}] -= (h_k[k] * Species->Production[{X,Y,Z,k}]);
							}
						}
					}
				}
			}
			unsigned Xp, Yp, Zp;
			for (unsigned int i = 0; i < Flow->boundaries.size(); ++i) {
				X = Flow->boundaries[i].X;
				Y = Flow->boundaries[i].Y;
				Z = Flow->boundaries[i].Z;
				Xp = (X - Flow->boundaries[i].n[0]);
				Yp = (Y - Flow->boundaries[i].n[1]);
				Zp = (Z - Flow->boundaries[i].n[2]);
				///______________________________________________________________________________________________________///
				///                              TRANSFER DATA TO TEMPORARY ARRAYS                                       ///
				///______________________________________________________________________________________________________///
				for (k = 0; k < Nb_spec; ++k) {
					C_k[k] = Flow->rho_0 * Flow->density[{Xp, Yp, Zp}] * Species->mass_fraction[{Xp,Yp,Zp,k}] / Species->Molar_mass[k];
					if (C_k[k] < 0) C_k[k] = 0;
					Species->Production[{Xp,Yp,Zp,k}] = 0;
				}
				Thermal->Production[{Xp,Yp,Zp}] = 0;
				T = Thermal->temperature[{Xp, Yp, Zp}] * Thermal->T_0;
				///______________________________________________________________________________________________________///
				///                           GET THERMOCHEMICAL PROPERTIES                                              ///
				///______________________________________________________________________________________________________///
				__mod_regath_interface_MOD_regath_mixt_mole_weight_cpp(&Nb_spec, &np, &(Species->mass_fraction[{Xp,Yp,Zp}]), &(Species->molar_mass_av[{Xp,Yp,Zp}]));
				__mod_regath_interface_MOD_regath_production_rate_cpp(&Nb_spec, &np, &T, C_k, &(Species->Production[{Xp,Yp,Zp}]));
				__mod_regath_interface_MOD_regath_cp_mixt_mass_cpp(&Nb_spec, &np, &T, &(Species->mass_fraction[{Xp,Yp,Zp}]), &(Thermal->c_p[{Xp, Yp, Zp}]));
				__mod_regath_interface_MOD_regath_enthalpy_spec_mass_cpp(&Nb_spec, &np, &T, h_k);
				///______________________________________________________________________________________________________///
				///                             PUT DATA INTO FIELD VARIABLE                                             ///
				///______________________________________________________________________________________________________///
				for (k = 0; k < Nb_spec; ++k) {
					Species->Production[{Xp,Yp,Zp,k}] *= 0;
					Thermal->Production[{X,Y,Z}] -= 0;
				}
			}
		}
	}
	void Initialisation(Species_solver* Species, std::string filename, Parallel_MPI* MPI_parallel) {
		np = 1;
		char Link[30], LinkTP[30];
		std::string species_name;
		///__________________________________________________________________________________________________///
		///                            REGATH INITIALIZATION                                                 ///
		///__________________________________________________________________________________________________///
		/// Open input file
		string input_filename(filename);
		input_filename += ".dat";
		ifstream input_file;  // File is open for READING
		input_file.open(input_filename.c_str(), ios::binary);
		string Line1;
		//	char comment_indicator = 'k';

		input_file.clear();
		input_file.seekg(0, ios::beg);
		find_line_after_header(input_file, "c\tREGATH Libraries");
		find_line_after_comment(input_file);
		input_file >> Link;
		input_file >> LinkTP;
		if (MPI_parallel->processor_id == MASTER) {
			std::cout << "Name of REGATH libraries : " << Link << ", " << LinkTP << std::endl;
		}
		__mod_regath_interface_MOD_open_file(&link_thermo, &link_transport, &Link, &LinkTP);
		__mod_regath_library_MOD_regath_initialize_thermo(&lout_user_thermo, &link_thermo, &cstring,
		                                                  &Nb_elem, &Nb_spec, &Nb_reac, &Nb_polynom);
		__mod_regath_library_MOD_regath_initialize_transport(&lout_user_transp, &link_transport);
		///_________________________________________________________________________________________________///
		///                            TEMPORARY ARRAY INITIALIZATION                                       ///
		///_________________________________________________________________________________________________///
		char** species_name_RG;
		char* species_name_temp_RG;
		species_name_temp_RG = new char[Nb_spec * 16];
		__mod_regath_interface_MOD_regath_species_names_cpp(species_name_temp_RG, &Nb_spec);
		species_name_RG = new char*[Nb_spec];
		Species->species_name_RG.resize(Nb_spec);
		for (unsigned int i = 0; i < Nb_spec; ++i) {
			species_name_RG[i] = new char[16];
			for (unsigned int j = 0; j < 15; ++j) {
				species_name_RG[i][j] = species_name_temp_RG[(i * 16) + j];
				if (species_name_temp_RG[(i * 16) + j] == ' ') {
					species_name_RG[i][j] = '\0';
				}
			}
			Species->species_name_RG[i] = std::string(species_name_RG[i]);
		}
#if defined ConstLewis_Species_diffusion
		Le = new double[Nb_spec];
		find_line_after_comment(input_file);
		for (unsigned int j = 0; j < Nb_spec; j++) {
			input_file >> species_name;
			for (unsigned int k = 0; k < Nb_spec; k++) {
				if (!species_name.compare(Species->species_name_RG[k])) {
					input_file >> Le[k];
				}
			}
		}
#endif
		input_file.close();
#if defined IMPLEX
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
#endif
		mol_mass_RG = new double[Nb_spec];
		C_k = new double[Nb_spec];
		h_k = new double[Nb_spec];
		X_k = new double[Nb_spec];
#if defined IMPLEX
		ImSolver.np = 1;
		ImSolver.C_temp = new double[Nb_spec];
		ImSolver.density_temp = new double[1];
		ImSolver.enthalpy_temp = new double[Nb_spec];
		ImSolver.Y_temp = new double[Nb_spec];
		ImSolver.T_temp = new double[1];
		ImSolver.Cp_temp = new double[1];
		ImSolver.M_temp = new double[Nb_spec];
		ImSolver.w_temp = new double[Nb_spec];
#endif
		///_________________________________________________________________________________________________///
		///                                  SPECIES NAME INITIALIZATION                                    ///
		///_________________________________________________________________________________________________///
		if (MPI_parallel->processor_id == MASTER) {
			std::cout << "SPECIES NUM " << Nb_spec << "\n";
			for (int i = 0; i < Nb_spec; ++i) {
				std::cout << i << "\t" << Species->species_name_RG[i];
#if defined ConstLewis_Species_diffusion
				std::cout << "\t" << Le[i];
#endif
				std::cout << "\n";
			}
		}
		Species->Nb_spec = Nb_spec;
		__mod_regath_interface_MOD_regath_spec_mole_weight_cpp(&Nb_spec, mol_mass_RG);
		__mod_regath_library_MOD_regath_universal_constants(&Gas_const, &Gas_const_cal, &p_atm);
		Species->Molar_mass.resize(Nb_spec);
		for (int k = 0; k < Nb_spec; k++) {
			Species->Molar_mass[k] = mol_mass_RG[k];
		}
#if defined IMPLEX
		ImSolver.Nb_spec = Nb_spec;
		ImSolver.M_temp = mol_mass_RG;
#endif  // defined
	}
};

#endif // defined REGATH_LIB
#endif  // REGATH_INTERFACE_H_INCLUDED
