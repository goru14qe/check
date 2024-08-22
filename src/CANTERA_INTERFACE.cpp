#include "CANTERA_INTERFACE.h"
#include "Flow_solver.h"
#include "Thermal_solver.h"
#include "Species_solver.h"
#include "ALBORZ_SETTINGS.h"
#include "Fluid_read_write.h"
/* These headers contain cantera functions */
#define CT_SKIP_DEPRECATION_WARNINGS
#include "cantera/thermo.h"
#include "cantera/transport.h"
#include "cantera/kinetics/GasKinetics.h"
#include "cantera/thermo/Phase.h"
#include "cantera/base/Solution.h"

#include "ALBORZ_Macros.h"
#include "utils/Config_utils.h"

using namespace Cantera;

//	initializes the Cantera solution instance, and gets the number of Species and reactions from Cantera.	//
//	also sets the Species names and molar masses.															//
void Thermo_chemistry_cantera::Initialisation(Species_solver* Species, const std::string& filename, Parallel_MPI* MPI_parallel) {
	using namespace std;
	/* open ALBORZ input file */
	ifstream input_file(filename + ".dat", ios::binary);
	/* Find cantera header in ALBORZ input file */
	find_line_after_header(input_file, "c\tCANTERA Libraries");
	find_line_after_comment(input_file);
	input_file >> thermochemistry_file >> chemistry >> transport;
	input_file.close();
	thermochemistry_file = get_current_working_directory() + "//" + thermochemistry_file;
	/* initiate cantera solution instance */
	sol = Cantera::newSolution(thermochemistry_file, chemistry, transport);
	/* Get number of Species and reactions from Cantera*/
	Nb_spec = sol->thermo()->nSpecies();
	Nb_reac = sol->kinetics()->nReactions();
	Species->Nb_spec = Nb_spec;
	Species->species_name_RG.resize(Nb_spec);
	Species->Molar_mass.resize(Nb_spec);
	temp = new double[Nb_spec];
	for (int k = 0; k < Nb_spec; ++k) {
		Species->species_name_RG[k] = sol->thermo()->speciesName(k);
		Species->Molar_mass[k] = sol->thermo()->molecularWeight(k) * 1e-3;
	}
}

//	calculates the thermo-chemical properties using Cantera libraries for thermodynamics and chemistry.	//
//	incorporating the Species production rates and molar masses.										//
void Thermo_chemistry_cantera::Thermo_properties(Flow_solver* Flow, Thermal_solver* Thermal, Species_solver* Species, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		/* Bulk nodes */
		for (int X = 0; X < MPI_parallel->dev_end[0]; ++X) {
			for (int Y = 0; Y < MPI_parallel->dev_end[1]; ++Y) {
				for (int Z = 0; Z < MPI_parallel->dev_end[2]; ++Z) {
					if (Species->solid_species[{X, Y, Z}] == FALSE || Flow->is_solid[{X, Y, Z}] == FALSE || Thermal->solid_thermal_type[{X, Y, Z}] == FALSE) {
						double sum = 0;
						for (int k = 0; k < Nb_spec; ++k) {
							Species->Production[{X, Y, Z, k}] = 0;
							temp[k] = std::max(Species->mass_fraction[{X, Y, Z, k}], 0.0);  // Ensure that the mass fractions are non-negative
							sum += temp[k];                                                 // Sum of mass fractions
						}
						for (int k = 0; k < Nb_spec; ++k) {
							temp[k] /= sum;  // Normalize mass fractions to sum to 1
						}
						/* Get temperature in Kelvins */
						double T = Thermal->temperature[{X, Y, Z}] * Thermal->T_0;
						double P = Flow->p_th_0;
#if defined compressible
						P = Flow->density[{X, Y, Z}] * Flow->rho_0 * R_GAS * T / (Species->molar_mass_av[{X, Y, Z}]);
#endif
						// Set state in Cantera
						sol->thermo()->setState_TPY(T, P, temp);
						// Retrieve properties from Cantera
						Species->molar_mass_av[{X, Y, Z}] = sol->thermo()->meanMolecularWeight() * 1e-3;  // Mean molecular weight in kg/mol
						/* return units of cantera: kmol/m3/s */
						sol->kinetics()->getNetProductionRates(&Species->Production[{X, Y, Z}]);  // Net production rates in kmol/m^3/s
						Thermal->c_p[{X, Y, Z}] = sol->thermo()->cp_mass();                   // Specific heat capacity in J/(kg*K)
						std::vector<double> h_k(Nb_spec);
						sol->thermo()->getPartialMolarEnthalpies(h_k.data());  // Partial molar enthalpies in J/mol

						// Convert units and scale production rates
						for (int k = 0; k < Nb_spec; ++k) {
							// Production rates need to be scaled to match the mass-based system if necessary
							Species->Production[{X, Y, Z, k}] *= (1e3 * global_parameters.D_t * Species->Molar_mass[k]);  // Convert from kmol/m^3/s to kg/m^3/s
							                                                                                              // Thermal->Production[{X, Y, Z}] -= h_k[k] * Species->Production[{X, Y, Z, k}] / Species->Molar_mass[k];  // Update temperature production
						}
					}
				}
			}
		}
		/* #### BOUNDARY CONDITION #### */	
		/*for (int k = 0; k < Species->Boundaries.size(); ++k) {
			int X = Species->Boundaries[k].X;
			int Y = Species->Boundaries[k].Y;
			int Z = Species->Boundaries[k].Z;
			int Xp, Yp, Zp;
			// Inlet boundary condition (type 102) - second order accuracy
			if (Species->Boundaries[k].type == 102) {
				double M_av = 0.0;
				//Get temperature in Kelvins
				double T = Thermal->temperature[{X, Y, Z}] * Thermal->T_0;
				if (Thermal->Boundaries[k].type == 102)
					T = Thermal->Boundaries[k].T * Thermal->T_0;
				double P = Flow->p_th_0;
#if defined compressible
				P = Flow->density[{X, Y, Z}] * Flow->rho_0 * R_GAS * T / (Species->molar_mass_av[{X, Y, Z}]);
#endif  // defined
				double* Y_k_holder = &(Species->Boundaries[k].Y_k[0]);
				//Y_k[0] : because the mass fractions are stored in the same order as the Species names in Cantera.
				//and we are pointing to the first element of the mass fractions array.
				sol->thermo()->setState_TPY(T, P, Y_k_holder);
				// Get W_av, Cp_av and omega_k in SI units
				// Return units of Cantera: kg/kmol
				M_av = sol->thermo()->meanMolecularWeight() * 1e-3;
				if (!Species->curved_boundaries) {
					Xp = X - Species->Boundaries[k].n[0];
					Yp = Y - Species->Boundaries[k].n[1];
					Zp = Z - Species->Boundaries[k].n[2];
					Species->molar_mass_av[{Xp, Yp, Zp}] = 2.0 * M_av - Species->molar_mass_av[{X, Y, Z}];  // Second order accuracy
					// Species->molar_mass_av[{X, Y, Z}] = M_av;													// First order accuracy
					//Species->molar_mass_av[{X, Y, Z}] = 0.5 * (Species->molar_mass_av[{Xp, Yp, Zp}] + M_av);	//Central difference Scheme
					//Species->molar_mass_av[{X, Y, Z}] = Species->molar_mass_av[{Xp, Yp, Zp}] + α * (M_av - Species->molar_mass_av[{Xp, Yp, Zp}]); // Upwind Scheme here α = factor depending on the Flow direction
					
				}
				if (Species->curved_boundaries) {
					double M_av_image = 0.0;
					double total_weight = 0.0;
					for (int ip = 0; ip < Species->Boundaries[k].X_Image_Int.size(); ++ip) {
						int X_img = Species->Boundaries[k].X_Image_Int[ip];
						int Y_img = Species->Boundaries[k].Y_Image_Int[ip];
						int Z_img = Species->Boundaries[k].Z_Image_Int[ip];
						M_av_image += Species->Boundaries[k].W_Image_Int[ip] * Species->molar_mass_av[{X_img, Y_img, Z_img}];
						total_weight += Species->Boundaries[k].W_Image_Int[ip];
					}
					if (total_weight > 0) {
						M_av_image /= total_weight;  // Average molar mass at image points
						Species->molar_mass_av[{X, Y, Z}] = 2.0 * M_av - M_av_image;
					}
				}
			}
			// Outlet boundary condition (type 104) - 0th order extrapolation
			if (Species->Boundaries[k].type == 104) {
				Xp = X - Species->Boundaries[k].n[0];
				Yp = Y - Species->Boundaries[k].n[1];
				Zp = Z - Species->Boundaries[k].n[2];
				if (!Species->curved_boundaries) {
					Species->molar_mass_av[{Xp, Yp, Zp}] = Species->molar_mass_av[{X, Y, Z}];
				}
				if (Species->curved_boundaries) {
					double molar_mass_image = 0.0;
					for (int ip = 0; ip < Species->Boundaries[k].X_Image_Int.size(); ++ip) {
						int X_img = Species->Boundaries[k].X_Image_Int[ip];
						int Y_img = Species->Boundaries[k].Y_Image_Int[ip];
						int Z_img = Species->Boundaries[k].Z_Image_Int[ip];
						molar_mass_image += Species->Boundaries[k].W_Image_Int[ip] * Species->molar_mass_av[{X_img, Y_img, Z_img}];
					}
					Species->molar_mass_av[{X, Y, Z}] = molar_mass_image;
				}
			}
			// Outlet boundary condition (type 105) - 1st order extrapolation
			if (Species->Boundaries[k].type == 105) {
				int Xp = X - Species->Boundaries[k].n[0];
				int Yp = Y - Species->Boundaries[k].n[1];
				int Zp = Z - Species->Boundaries[k].n[2];

				int Xp1 = X - 2 * Species->Boundaries[k].n[0];
				int Yp1 = Y - 2 * Species->Boundaries[k].n[1];
				int Zp1 = Z - 2 * Species->Boundaries[k].n[2];
				if (!Species->curved_boundaries) {
					Species->molar_mass_av[{X, Y, Z}] = 2.0 * Species->molar_mass_av[{Xp, Yp, Zp}] - Species->molar_mass_av[{Xp1, Yp1, Zp1}];
				}
				if (Species->curved_boundaries) {
					double M_av_Image = 0.0;
					double total_weight = 0.0;
					for (int ip = 0; ip < Species->Boundaries[k].X_Image_Int.size(); ++ip) {
						int X_img = Species->Boundaries[k].X_Image_Int[ip];
						int Y_img = Species->Boundaries[k].Y_Image_Int[ip];
						int Z_img = Species->Boundaries[k].Z_Image_Int[ip];
						M_av_Image += Species->Boundaries[k].W_Image_Int[ip] * Species->molar_mass_av[{X_img, Y_img, Z_img}];
						total_weight += Species->Boundaries[k].W_Image_Int[ip];
					}
					if (total_weight > 0) {
						M_av_Image /= total_weight;  // Average molar mass at image points
						Species->molar_mass_av[{X, Y, Z}] = 2.0 * M_av_Image - Species->molar_mass_av[{Xp1, Yp1, Zp1}];
					}
				}
			}
			// Outlet boundary condition (type 106) - 2nd order extrapolation
			if (Species->Boundaries[k].type == 106) {
				int Xp = X - Species->Boundaries[k].n[0];
				int Yp = Y - Species->Boundaries[k].n[1];
				int Zp = Z - Species->Boundaries[k].n[2];

				int Xp1 = X - 2 * Species->Boundaries[k].n[0];
				int Yp1 = Y - 2 * Species->Boundaries[k].n[1];
				int Zp1 = Z - 2 * Species->Boundaries[k].n[2];

				if (!Species->curved_boundaries) {
					Species->molar_mass_av[{X, Y, Z}] = (4.0 / 3.0) * Species->molar_mass_av[{Xp, Yp, Zp}] - (1.0 / 3.0) * Species->molar_mass_av[{Xp1, Yp1, Zp1}];
				}
				if (Species->curved_boundaries) {
					double M_av_Image = 0.0;
					double total_weight = 0.0;
					for (int ip = 0; ip < Species->Boundaries[k].X_Image_Int.size(); ++ip) {
						int X_img = Species->Boundaries[k].X_Image_Int[ip];
						int Y_img = Species->Boundaries[k].Y_Image_Int[ip];
						int Z_img = Species->Boundaries[k].Z_Image_Int[ip];
						M_av_Image += Species->Boundaries[k].W_Image_Int[ip] * Species->molar_mass_av[{X_img, Y_img, Z_img}];
						total_weight += Species->Boundaries[k].W_Image_Int[ip];
					}
					if (total_weight > 0) {
						M_av_Image /= total_weight;  // Average molar mass at image points
						Species->molar_mass_av[{X, Y, Z}] = (4.0 / 3.0) * M_av_Image - (1.0 / 3.0) * Species->molar_mass_av[{Xp1, Yp1, Zp1}];
					}
				}
			}
		}*/
	}
}

//	calculates the diffusion coefficients, viscosity, and Thermal conductivity	//
//	using Cantera libraries for thermodynamics and chemistry.					//
void Thermo_chemistry_cantera::Transport_properties(Flow_solver* Flow, Thermal_solver* Thermal, Species_solver* Species, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		/* Bulk nodes */
		for (int X = 0; X < MPI_parallel->dev_end[0]; ++X) {
			for (int Y = 0; Y < MPI_parallel->dev_end[1]; ++Y) {
				for (int Z = 0; Z < MPI_parallel->dev_end[2]; ++Z) {
					// Skip if the point is a solid or has a solid Thermal type
					if (Species->solid_species[{X, Y, Z}] == FALSE || Flow->is_solid[{X, Y, Z}] == FALSE || Thermal->solid_thermal_type[{X, Y, Z}] == FALSE) {
						// Compute temperature and pressure
						double T = Thermal->temperature[{X, Y, Z}] * Thermal->T_0;
						double P = Flow->p_th_0;
#if defined compressible
						P = Flow->density[{X, Y, Z}] * Flow->rho_0 * R_GAS * T / (Species->molar_mass_av[{X, Y, Z}]);
#endif  // defined
        // Set state in Cantera and compute transport properties
						sol->thermo()->setState_TPY(T, P, &Species->mass_fraction[{X, Y, Z}]);
						Thermal->thermal_diffusion_coefficient[{X, Y, Z}] = sol->transport()->thermalConductivity();
						Flow->viscosity[{X, Y, Z}] = sol->transport()->viscosity();
						sol->transport()->getMixDiffCoeffs(&Species->diffusion_coefficient[{X, Y, Z}]);

						// Apply scaling factors based on grid spacing and densities
						Flow->viscosity[{X, Y, Z}] *= (global_parameters.D_t / (global_parameters.D_x * global_parameters.D_x * Flow->density[{X, Y, Z}] * Flow->rho_0));
						Thermal->thermal_diffusion_coefficient[{X, Y, Z}] *= (global_parameters.D_t / (global_parameters.D_x * global_parameters.D_x));

						// Update diffusion coefficients for each Species
						for (int k = 0; k < Species->Nb_spec; ++k) {
#if defined UnitLewis_Species_diffusion  // Lewis No. = 1 => D_i = D_o / c_p , where D_o is the diffusion coefficient from Cantera and c_p is the specific heat capacity at constant pressure.
							Species->diffusion_coefficient[{X, Y, Z, k}] *= Thermal->thermal_diffusion_coefficient[{X, Y, Z}] / Thermal->c_p[{X, Y, Z}];
#endif
#if defined ConstLewis_Species_diffusion  // Lewis No. = constant => D_i = D_o / (Le * c_p) , where D_o is the diffusion coefficient from Cantera and c_p is the specific heat capacity at constant pressure and Le is the Lewis number.
							Species->diffusion_coefficient[{X, Y, Z, k}] = Thermal->thermal_diffusion_coefficient[{X, Y, Z}] / (Species->Le[k] * Thermal->c_p[{X, Y, Z}]);
#endif
#if defined MixtureAveraged_Species_diffusion  // Mixture averaged diffusion coefficient => D_i = D_o * (rho * rho_0 * D_t / (D_x * D_x)).
							Species->diffusion_coefficient[{X, Y, Z, k}] *= Flow->rho_0 * Flow->density[{X, Y, Z}] * (global_parameters.D_t / (global_parameters.D_x * global_parameters.D_x));
#endif
#if defined Schmidt_Number_Diffusion  // Schmidt Number diffusion coefficient => D_i = (nu / (Sc * c_p)) , where nu is the viscosity, Sc is the Schmidt number and c_p is the specific heat capacity at constant pressure.
							double nu = Flow->viscosity[{X, Y, Z}];
							double alpha = Thermal->thermal_diffusion_coefficient[{X, Y, Z}] / Thermal->c_p[{X, Y, Z}];
							Species->diffusion_coefficient[{X, Y, Z, k}] = nu / (Species->Schmidt_Number[k] * alpha);
#endif
#if defined Turbulent_Diffusion                                        // Turbulent diffusion coefficient => D_i = (nu_t / (Sc * alpha)) , where nu_t is the turbulent viscosity, Sc is the Schmidt number and alpha is the Thermal diffusivity.
							double nu_t = Flow->viscosity[{X, Y, Z}];  // Turbulent viscosity
							double alpha = Thermal->thermal_diffusion_coefficient[{X, Y, Z}] / Thermal->c_p[{X, Y, Z}];
							Species->diffusion_coefficient[{X, Y, Z, k}] += nu_t / (Species->Schmidt_Number[k] * alpha);  // Adjust with turbulent diffusion
#endif
#if defined Knudsen_Diffusion                   // Knudsen diffusion coefficient => D_i = (nu / (d_pore / 3)) , where nu is the viscosity and d_pore is the diameter of the pores.
							double d_pore = 1;  // Replace with global_parameters.pore_diameter if available
							double nu = Flow->viscosity[{X, Y, Z}];
							Species->diffusion_coefficient[{X, Y, Z, k}] = nu / (d_pore / 3.0);  // Knudsen diffusion adjustment
#endif
						}
					}
				}
			}
		}
		// #### BOUNDARY CONDITION #### //
		//Species BC//
		/*for (int k = 0; k < Species->Boundaries.size(); ++k) {
			int X = Species->Boundaries[k].X;
			int Y = Species->Boundaries[k].Y;
			int Z = Species->Boundaries[k].Z;
			int Xp, Yp, Zp;
			// Inlet boundary condition (type 102)
			if (Species->Boundaries[k].type == 102) {
				double T = Thermal->temperature[{X, Y, Z}] * Thermal->T_0;
				if (Thermal->Boundaries[k].type == 102)
					T = Thermal->Boundaries[k].T * Thermal->T_0;
				double P = Flow->p_th_0;
#if defined compressible
				P = Flow->density[{X, Y, Z}] * Flow->rho_0 * R_GAS * T / (Species->molar_mass_av[{X, Y, Z}]);
#endif                                                                  // defined compressible
				double* Y_k_holder = &(Species->Boundaries[k].Y_k[0]);  // Mass fractions at the boundary
				sol->thermo()->setState_TPY(T, P, Y_k_holder);          // Set state in Cantera
				sol->transport()->getMixDiffCoeffs(temp);               // Get diffusion coefficients
				for (int k = 0; k < Species->Nb_spec; ++k) {
#if defined UnitLewis_Species_diffusion  // Lewis No. = 1 => D_i = D_o / c_p , where D_o is the diffusion coefficient from Cantera and c_p is the specific heat capacity at constant pressure.
					temp[k] *= (Flow->rho_0 * Flow->density[{X, Y, Z}] * (global_parameters.D_t / (global_parameters.D_x * global_parameters.D_x))) / Thermal->c_p[{X, Y, Z}];
#endif
#if defined ConstLewis_Species_diffusion  // Lewis No. = constant => D_i = D_o / (Le * c_p) , where D_o is the diffusion coefficient from Cantera and c_p is the specific heat capacity at constant pressure and Le is the Lewis number.
					temp[k] /= (Species->Le[k] * Thermal->c_p[{X, Y, Z}]);
#endif
#if defined MixtureAveraged_Species_diffusion  // Mixture averaged diffusion coefficient => D_i = D_o * (rho * rho_0 * D_t / (D_x * D_x)).
					temp[k] *= Flow->rho_0 * Flow->density[{X, Y, Z}] * (global_parameters.D_t / (global_parameters.D_x * global_parameters.D_x));
#endif
#if defined Schmidt_Number_Diffusion  // Schmidt Number diffusion coefficient => D_i = (nu / (Sc * c_p)) , where nu is the viscosity, Sc is the Schmidt number and c_p is the specific heat capacity at constant pressure.
					double nu = Flow->viscosity[{X, Y, Z}];
					double alpha = Thermal->thermal_diffusion_coefficient[{X, Y, Z}] / Thermal->c_p[{X, Y, Z}];
					temp[k] = nu / (Species->Schmidt_Number[k] * alpha);
#endif
#if defined Turbulent_Diffusion                                // Turbulent diffusion coefficient => D_i = (nu_t / (Sc * alpha)) , where nu_t is the turbulent viscosity, Sc is the Schmidt number and alpha is the Thermal diffusivity.
					double nu_t = Flow->viscosity[{X, Y, Z}];  // Turbulent viscosity
					double alpha = Thermal->thermal_diffusion_coefficient[{X, Y, Z}] / Thermal->c_p[{X, Y, Z}];
					temp[k] += nu_t / (Species->Schmidt_Number[k] * alpha);  // Adjust with turbulent diffusion
#endif
#if defined Knudsen_Diffusion           // Knudsen diffusion coefficient => D_i = (nu / (d_pore / 3)) , where nu is the viscosity and d_pore is the diameter of the pores.
					double d_pore = 1;  // Replace with global_parameters.pore_diameter if available
					double nu = Flow->viscosity[{X, Y, Z}];
					temp[k] = nu / (d_pore / 3.0);  // Knudsen diffusion adjustment
#endif
				}
				if (!Species->curved_boundaries) {
					Xp = X - Species->Boundaries[k].n[0];
					Yp = Y - Species->Boundaries[k].n[1];
					Zp = Z - Species->Boundaries[k].n[2];
					for (int k = 0; k < Species->Nb_spec; ++k) {
						// Apply second-order accuracy (central difference scheme)
						Species->diffusion_coefficient[{Xp, Yp, Zp, k}] = 2.0 * temp[k] - Species->diffusion_coefficient[{X, Y, Z, k}];
						// Species->diffusion_coefficient[{X, Y, Z, k}] = temp[k];  // First-order accuracy
						// Species->diffusion_coefficient[{X, Y, Z, k}] = 0.5 * (Species->diffusion_coefficient[{Xp, Yp, Zp, k}] + temp[k]);  // Central difference scheme
						// Species->diffusion_coefficient[{X, Y, Z, k}] = Species->diffusion_coefficient[{Xp, Yp, Zp, k}] + α * (temp[k] - Species->diffusion_coefficient[{Xp, Yp, Zp, k}]);  // Upwind scheme
					}
				}
				if (Species->curved_boundaries) {
					double D_image = 0.0;
					double total_weight = 0.0;
					for (int ip = 0; ip < Species->Boundaries[k].X_Image_Int.size(); ++ip) {
						int X_img = Species->Boundaries[k].X_Image_Int[ip];
						int Y_img = Species->Boundaries[k].Y_Image_Int[ip];
						int Z_img = Species->Boundaries[k].Z_Image_Int[ip];
						D_image += Species->Boundaries[k].W_Image_Int[ip] * Species->diffusion_coefficient[{X_img, Y_img, Z_img, k}];
						total_weight += Species->Boundaries[k].W_Image_Int[ip];
					}
					if (total_weight > 0) {
						D_image /= total_weight;  // Average diffusion coefficient at image points
						for (int k = 0; k < Species->Nb_spec; ++k) {
							// Apply second-order accuracy for curved boundaries
							Species->diffusion_coefficient[{X, Y, Z, k}] = 2.0 * temp[k] - D_image;
						}
					}
				}
			}
			// Outlet boundary condition (type 104)
			if (Species->Boundaries[k].type == 104) {
				Xp = X - Species->Boundaries[k].n[0];
				Yp = Y - Species->Boundaries[k].n[1];
				Zp = Z - Species->Boundaries[k].n[2];
				if (!Species->curved_boundaries) {
					for (int kk = 0; kk < Species->Nb_spec; ++kk) {
						Species->diffusion_coefficient[{Xp, Yp, Zp, kk}] = Species->diffusion_coefficient[{X, Y, Z, kk}];
					}
				}
				if (Species->curved_boundaries) {
					for (int kk = 0; kk < Species->Nb_spec; ++kk) {
						double diff_coeff_image = 0.0;
						double total_weight = 0.0;
						for (int ip = 0; ip < Species->Boundaries[k].X_Image_Int.size(); ++ip) {
							int X_img = Species->Boundaries[k].X_Image_Int[ip];
							int Y_img = Species->Boundaries[k].Y_Image_Int[ip];
							int Z_img = Species->Boundaries[k].Z_Image_Int[ip];
							diff_coeff_image += Species->Boundaries[k].W_Image_Int[ip] * Species->diffusion_coefficient[{X_img, Y_img, Z_img, kk}];
							total_weight += Species->Boundaries[k].W_Image_Int[ip];
						}
						Species->diffusion_coefficient[{X, Y, Z, kk}] = diff_coeff_image;
					}
				}
			}
			// Outlet boundary condition (type 105)
			if (Species->Boundaries[k].type == 105) {
				Xp = X - Species->Boundaries[k].n[0];
				Yp = Y - Species->Boundaries[k].n[1];
				Zp = Z - Species->Boundaries[k].n[2];

				int Xp1 = X - 2 * Species->Boundaries[k].n[0];
				int Yp1 = Y - 2 * Species->Boundaries[k].n[1];
				int Zp1 = Z - 2 * Species->Boundaries[k].n[2];
				if (!Species->curved_boundaries) {
					for (int kk = 0; kk < Species->Nb_spec; ++kk) {
						Species->diffusion_coefficient[{X, Y, Z, kk}] = 2.0 * Species->diffusion_coefficient[{Xp, Yp, Zp, kk}] - Species->diffusion_coefficient[{Xp1, Yp1, Zp1, kk}];
					}
				}
				if (Species->curved_boundaries) {
					double Diff_image = 0.0;
					double total_weight = 0.0;
					for (int ip = 0; ip < Species->Boundaries[k].X_Image_Int.size(); ++ip) {
						int X_img = Species->Boundaries[k].X_Image_Int[ip];
						int Y_img = Species->Boundaries[k].Y_Image_Int[ip];
						int Z_img = Species->Boundaries[k].Z_Image_Int[ip];
						for (int kk = 0; kk < Species->Nb_spec; ++kk) {
							Diff_image += Species->Boundaries[k].W_Image_Int[ip] * Species->diffusion_coefficient[{X_img, Y_img, Z_img, kk}];
						}
						total_weight += Species->Boundaries[k].W_Image_Int[ip];
					}
					if (total_weight > 0) {
						Diff_image /= total_weight;
						for (int kk = 0; kk < Species->Nb_spec; ++kk) {
							Species->diffusion_coefficient[{X, Y, Z, kk}] = 2.0 * Diff_image - Species->diffusion_coefficient[{Xp1, Yp1, Zp1, kk}];
						}
					}
				}
			}
			// Outlet boundary condition (type 106)
			if (Species->Boundaries[k].type == 106) {
				Xp = X - Species->Boundaries[k].n[0];
				Yp = Y - Species->Boundaries[k].n[1];
				Zp = Z - Species->Boundaries[k].n[2];

				int Xp1 = X - 2 * Species->Boundaries[k].n[0];
				int Yp1 = Y - 2 * Species->Boundaries[k].n[1];
				int Zp1 = Z - 2 * Species->Boundaries[k].n[2];
				if (!Species->curved_boundaries) {
					for (int kk = 0; kk < Species->Nb_spec; ++kk) {
						Species->diffusion_coefficient[{X, Y, Z, kk}] = (4.0 / 3.0) * Species->diffusion_coefficient[{Xp, Yp, Zp, kk}] - (1.0 / 3.0) * Species->diffusion_coefficient[{Xp1, Yp1, Zp1, kk}];
					}
				}
				if (Species->curved_boundaries) {
					double Diff_image = 0.0;
					double total_weight = 0.0;
					for (int ip = 0; ip < Species->Boundaries[k].X_Image_Int.size(); ++ip) {
						int X_img = Species->Boundaries[k].X_Image_Int[ip];
						int Y_img = Species->Boundaries[k].Y_Image_Int[ip];
						int Z_img = Species->Boundaries[k].Z_Image_Int[ip];
						for (int kk = 0; kk < Species->Nb_spec; ++kk) {
							Diff_image += Species->Boundaries[k].W_Image_Int[ip] * Species->diffusion_coefficient[{X_img, Y_img, Z_img, kk}];
						}
						total_weight += Species->Boundaries[k].W_Image_Int[ip];
					}
					if (total_weight > 0) {
						Diff_image /= total_weight;
						for (int kk = 0; kk < Species->Nb_spec; ++kk) {
							Species->diffusion_coefficient[{X, Y, Z, kk}] = (4.0 / 3.0) * Diff_image - (1.0 / 3.0) * Species->diffusion_coefficient[{Xp1, Yp1, Zp1, kk}];
						}
					}
				}
			}
		}*/
		/* #### BOUNDARY CONDITION #### */
		
	/* Next temperature boundaries */
	/*	for (int k = 0; k < Thermal->Boundaries.size(); ++k) {
			int X = Thermal->Boundaries[k].X;
			int Y = Thermal->Boundaries[k].Y;
			int Z = Thermal->Boundaries[k].Z;
			int Xp, Yp, Zp;
			// Inlet boundary condition (type 102)
			if (Thermal->Boundaries[k].type == 102) {
				double Diff = 0.0;
				double T = Thermal->Boundaries[k].T * Thermal->T_0;
				double P = Flow->p_th_0;
#if defined compressible
				P = Flow->density[{X, Y, Z}] * Flow->rho_0 * R_GAS * T / Species->molar_mass_av[{X, Y, Z}];
#endif                                                                                              // defined
				double* Y_k_holder = &(Species->mass_fraction[{X, Y, Z, 0}]);               // Full array pointer
				if (Species->Boundaries[k].type == 102) {
					Y_k_holder = &(Species->Boundaries[k].Y_k[0]);  						// Mass fractions at the boundary
				}
				sol->thermo()->setState_TPY(T, P, Y_k_holder);                                      // Set state
				Diff = sol->transport()->thermalConductivity();                                     // Get Thermal conductivity
				Diff *= (global_parameters.D_t / (global_parameters.D_x * global_parameters.D_x));  // Scale Thermal conductivity
				if (!Thermal->curved_boundaries) {
					Xp = X - Thermal->Boundaries[k].n[0];
					Yp = Y - Thermal->Boundaries[k].n[1];
					Zp = Z - Thermal->Boundaries[k].n[2];
					// Second-order accuracy using central difference scheme
					Thermal->thermal_diffusion_coefficient[{Xp, Yp, Zp}] = 2.0 * Diff - Thermal->thermal_diffusion_coefficient[{X, Y, Z}];
					// First-order accuracy:
					// Thermal->thermal_diffusion_coefficient[{X, Y, Z}] = Diff;
					// Central difference scheme:
					// Thermal->thermal_diffusion_coefficient[{X, Y, Z}] = 0.5 * (Thermal->thermal_diffusion_coefficient[{Xp, Yp, Zp}] + Diff);
					// Upwind scheme:
					// double alpha = ...; // Define alpha based on Flow direction
					// Thermal->thermal_diffusion_coefficient[{X, Y, Z}] = Thermal->thermal_diffusion_coefficient[{Xp, Yp, Zp}] + alpha * (Diff - Thermal->thermal_diffusion_coefficient[{Xp, Yp, Zp}]);
				}
				if (Thermal->curved_boundaries) {
					double Diff_image = 0.0;
					double total_weight = 0.0;
					for (int ip = 0; ip < Thermal->Boundaries[k].X_Image_Int.size(); ++ip) {
						int X_img = Thermal->Boundaries[k].X_Image_Int[ip];
						int Y_img = Thermal->Boundaries[k].Y_Image_Int[ip];
						int Z_img = Thermal->Boundaries[k].Z_Image_Int[ip];
						Diff_image += Thermal->Boundaries[k].W_Image_Int[ip] * Thermal->thermal_diffusion_coefficient[{X_img, Y_img, Z_img}];
						total_weight += Thermal->Boundaries[k].W_Image_Int[ip];
					}
					if (total_weight > 0) {
						Diff_image /= total_weight;  // Average Thermal conductivity at image points
						// Second-order accuracy using central difference scheme
						Thermal->thermal_diffusion_coefficient[{X, Y, Z}] = 2.0 * Diff - Diff_image;
					}
				}
			}
			// Outlet boundary condition (type 104)
			if (Thermal->Boundaries[k].type == 104) {
				if (!Thermal->curved_boundaries) {
					Xp = X - Thermal->Boundaries[k].n[0];
					Yp = Y - Thermal->Boundaries[k].n[1];
					Zp = Z - Thermal->Boundaries[k].n[2];
					Thermal->thermal_diffusion_coefficient[{Xp, Yp, Zp}] = Thermal->thermal_diffusion_coefficient[{X, Y, Z}];
				}
				if (Thermal->curved_boundaries) {
					double diff_coeff_image = 0.0;
					// double total_weight = 0.0;
					for (int ip = 0; ip < Thermal->Boundaries[k].X_Image_Int.size(); ++ip) {
						int X_img = Thermal->Boundaries[k].X_Image_Int[ip];
						int Y_img = Thermal->Boundaries[k].Y_Image_Int[ip];
						int Z_img = Thermal->Boundaries[k].Z_Image_Int[ip];
						diff_coeff_image += Thermal->Boundaries[k].W_Image_Int[ip] * Thermal->thermal_diffusion_coefficient[{X_img, Y_img, Z_img}];
						// total_weight += Thermal->Boundaries[k].W_Image_Int[ip];
					}
					// if (total_weight > 0) {
					//	diff_coeff_image /= total_weight;  // Average diffusion coefficient at image points
					Thermal->thermal_diffusion_coefficient[{X, Y, Z}] = diff_coeff_image;
					//}
				}
			}
			// Outlet boundary condition (type 105)
			if (Thermal->Boundaries[k].type == 105) {
				Xp = (X - Thermal->Boundaries[k].n[0]);
				Yp = (Y - Thermal->Boundaries[k].n[1]);
				Zp = (Z - Thermal->Boundaries[k].n[2]);

				int Xp1 = (X - 2 * Thermal->Boundaries[k].n[0]);
				int Yp1 = (Y - 2 * Thermal->Boundaries[k].n[1]);
				int Zp1 = (Z - 2 * Thermal->Boundaries[k].n[2]);

				if (!Thermal->curved_boundaries) {
					Thermal->thermal_diffusion_coefficient[{X, Y, Z}] = 2.0 * Thermal->thermal_diffusion_coefficient[{Xp, Yp, Zp}] - Thermal->thermal_diffusion_coefficient[{Xp1, Yp1, Zp1}];
				}
				if (Thermal->curved_boundaries) {
					double Diff_image = 0.0;
					double total_weight = 0.0;
					for (int ip = 0; ip < Thermal->Boundaries[k].X_Image_Int.size(); ++ip) {
						int X_img = Thermal->Boundaries[k].X_Image_Int[ip];
						int Y_img = Thermal->Boundaries[k].Y_Image_Int[ip];
						int Z_img = Thermal->Boundaries[k].Z_Image_Int[ip];
						Diff_image += Thermal->Boundaries[k].W_Image_Int[ip] * Thermal->thermal_diffusion_coefficient[{X_img, Y_img, Z_img}];
						total_weight += Thermal->Boundaries[k].W_Image_Int[ip];
					}
					if (total_weight > 0) {
						Diff_image /= total_weight;  // Average diffusion coefficient at image points
						Thermal->thermal_diffusion_coefficient[{X, Y, Z}] = 2.0 * Diff_image - Thermal->thermal_diffusion_coefficient[{Xp1, Yp1, Zp1}];
					}
				}
			}
			// Outlet boundary condition (type 106)
			if (Thermal->Boundaries[k].type == 106) {
				Xp = X - Thermal->Boundaries[k].n[0];
				Yp = Y - Thermal->Boundaries[k].n[1];
				Zp = Z - Thermal->Boundaries[k].n[2];

				int Xp1 = X - 2 * Thermal->Boundaries[k].n[0];
				int Yp1 = Y - 2 * Thermal->Boundaries[k].n[1];
				int Zp1 = Z - 2 * Thermal->Boundaries[k].n[2];
				if (!Thermal->curved_boundaries) {
					Thermal->thermal_diffusion_coefficient[{X, Y, Z}] = (4.0 / 3.0) * Thermal->thermal_diffusion_coefficient[{Xp, Yp, Zp}] - (1.0 / 3.0) * Thermal->thermal_diffusion_coefficient[{Xp1, Yp1, Zp1}];
				}
				if (Thermal->curved_boundaries) {
					double Diff_image = 0.0;
					double total_weight = 0.0;
					for (int ip = 0; ip < Thermal->Boundaries[k].X_Image_Int.size(); ++ip) {
						int X_img = Thermal->Boundaries[k].X_Image_Int[ip];
						int Y_img = Thermal->Boundaries[k].Y_Image_Int[ip];
						int Z_img = Thermal->Boundaries[k].Z_Image_Int[ip];
						Diff_image += Thermal->Boundaries[k].W_Image_Int[ip] * Thermal->thermal_diffusion_coefficient[{X_img, Y_img, Z_img}];
						total_weight += Thermal->Boundaries[k].W_Image_Int[ip];
					}
					if (total_weight > 0) {
						Diff_image /= total_weight;  // Average diffusion coefficient at image points
						Thermal->thermal_diffusion_coefficient[{X, Y, Z}] = (4.0 / 3.0) * Diff_image - (1.0 / 3.0) * Thermal->thermal_diffusion_coefficient[{Xp1, Yp1, Zp1}];
					}
				}
			}
		}*/
	}
}

//	calculates the heat production rate using Cantera libraries for thermodynamics and chemistry.	//
//	incorporating the Species production rates and molar masses.									//
void Thermo_chemistry_cantera::Heat_production(Flow_solver* Flow, Thermal_solver* Thermal, Species_solver* Species, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		for (int X = 0; X < MPI_parallel->dev_end[0]; ++X) {
			for (int Y = 0; Y < MPI_parallel->dev_end[1]; ++Y) {
				for (int Z = 0; Z < MPI_parallel->dev_end[2]; ++Z) {
					// Check if the point is not solid
					if (Thermal->solid_thermal_type[{X, Y, Z}] == FALSE) {
						// Initialize production term
						Thermal->Production[{X, Y, Z}] = 0.0;
						// Compute temperature and pressure
						double T = Thermal->temperature[{X, Y, Z}] * Thermal->T_0;
						double P = Flow->p_th_0;
#if defined compressible
						// Compute pressure for compressible case
						P = Flow->density[{X, Y, Z}] * Flow->rho_0 * R_GAS * T / (Species->molar_mass_av[{X, Y, Z}]);
#endif  // defined
        // Set state in Cantera
						sol->thermo()->setState_TPY(T, P, &Species->mass_fraction[{X, Y, Z}]);

						// Get partial molar enthalpies
						sol->thermo()->getPartialMolarEnthalpies(temp);

						// Calculate production term
						for (int k = 0; k < Nb_spec; ++k) {
							// Convert to proper units and update production term
							temp[k] *= 1e-3 / Species->Molar_mass[k];
							Thermal->Production[{X, Y, Z}] -= (temp[k] * Species->Production[{X, Y, Z, k}]);
						}
					}
				}
			}
		}
		/* #### BOUNDARY CONDITION #### */
/*		for (int k = 0; k < Thermal->Boundaries.size(); ++k) {
			int X = Thermal->Boundaries[k].X;
			int Y = Thermal->Boundaries[k].Y;
			int Z = Thermal->Boundaries[k].Z;
			int Xp, Yp, Zp;
			// Inlet boundary condition (type 102)
			if (Thermal->Boundaries[k].type == 102) {
				double T = Thermal->temperature[{X, Y, Z}] * Thermal->T_0;
				double P = Flow->p_th_0;
#if defined compressible
				P = Flow->density[{X, Y, Z}] * Flow->rho_0 * R_GAS * T / Species->molar_mass_av[{X, Y, Z}];
#endif  // defined compressible

				double* Y_k_holder = &(Species->mass_fraction[{X, Y, Z}]); // Full array pointer
				sol->thermo()->setState_TPY(T, P, Y_k_holder);          // Set state
				sol->thermo()->getPartialMolarEnthalpies(temp);

				// Initialize production term
				temp[k] = 0.0;
				// Compute production term based on partial molar enthalpies and Species' production rates
				for (int k = 0; k < Nb_spec; ++k) {
					temp[k] = temp[k] * 1e-3 / Species->Molar_mass[k];  // Convert enthalpies to proper units
					temp[k] -= (temp[k] * Species->Production[{X, Y, Z, k}]);
				}
				// Apply boundary condition for production term
				if (!Thermal->curved_boundaries) {
					Xp = X - Thermal->Boundaries[k].n[0];  // Xp, Yp, Zp: outside points
					Yp = Y - Thermal->Boundaries[k].n[1];
					Zp = Z - Thermal->Boundaries[k].n[2];
					// Central difference scheme for second-order accuracy
					Thermal->Production[{Xp, Yp, Zp}] = 2.0 * temp[k] - Thermal->Production[{X, Y, Z}];
					// Uncomment for other schemes:
					// First-order accuracy:
					// Thermal->Production[{X, Y, Z}] = Thermal->Production[{Xp, Yp, Zp}];
					// Central difference scheme:
					// Thermal->Production[{X, Y, Z}] = 0.5 * (Thermal->Production[{Xp, Yp, Zp}] + Thermal->Production[{X, Y, Z}]);
					// Upwind scheme (α needs to be defined based on Flow direction):
					// double alpha = ...; // Define alpha based on Flow direction
					// Thermal->Production[{X, Y, Z}] = Thermal->Production[{Xp, Yp, Zp}] + alpha * (Thermal->Production[{X, Y, Z}] - Thermal->Production[{Xp, Yp, Zp}]);
				}
				if (Thermal->curved_boundaries) {
					double Prod_image = 0.0;
					double total_weight = 0.0;
					// Average production term at image points for curved boundaries
					for (int ip = 0; ip < Thermal->Boundaries[k].X_Image_Int.size(); ++ip) {
						int X_img = Thermal->Boundaries[k].X_Image_Int[ip];
						int Y_img = Thermal->Boundaries[k].Y_Image_Int[ip];
						int Z_img = Thermal->Boundaries[k].Z_Image_Int[ip];
						Prod_image += Thermal->Boundaries[k].W_Image_Int[ip] * Thermal->Production[{X_img, Y_img, Z_img}];
						total_weight += Thermal->Boundaries[k].W_Image_Int[ip];
					}
					if (total_weight > 0) {
						Prod_image /= total_weight;  // Average production term
						// Central difference scheme for second-order accuracy
						Thermal->Production[{X, Y, Z}] = 0.0;  // 2.0 * Thermal->Production[{X, Y, Z}] - Prod_image;
					}
				}
			}
			// Outlet boundary condition (type 104) - 0th order extrapolation
			if (Thermal->Boundaries[k].type == 104) {
				Xp = X - Thermal->Boundaries[k].n[0];
				Yp = Y - Thermal->Boundaries[k].n[1];
				Zp = Z - Thermal->Boundaries[k].n[2];
				if (!Thermal->curved_boundaries) {
					Thermal->Production[{Xp, Yp, Zp}] = Thermal->Production[{X, Y, Z}];
				}
				if (Thermal->curved_boundaries) {
					double Prod_image = 0.0;
					for (int ip = 0; ip < Thermal->Boundaries[k].X_Image_Int.size(); ++ip) {
						int X_img = Thermal->Boundaries[k].X_Image_Int[ip];
						int Y_img = Thermal->Boundaries[k].Y_Image_Int[ip];
						int Z_img = Thermal->Boundaries[k].Z_Image_Int[ip];
						Prod_image += Thermal->Boundaries[k].W_Image_Int[ip] * Thermal->Production[{X_img, Y_img, Z_img}];
					}
					Thermal->Production[{X, Y, Z}] = 0.0;  // Prod_image;
				}
			}
			// Outlet boundary condition (type 105) - 1st order extrapolation
			if (Thermal->Boundaries[k].type == 105) {
				Xp = (X - Thermal->Boundaries[k].n[0]);
				Yp = (Y - Thermal->Boundaries[k].n[1]);
				Zp = (Z - Thermal->Boundaries[k].n[2]);
				int Xp1 = (X - 2 * Thermal->Boundaries[k].n[0]);
				int Yp1 = (Y - 2 * Thermal->Boundaries[k].n[1]);
				int Zp1 = (Z - 2 * Thermal->Boundaries[k].n[2]);
				if (!Thermal->curved_boundaries) {
					Thermal->Production[{X, Y, Z}] = 2.0 * Thermal->Production[{Xp, Yp, Zp}] - Thermal->Production[{Xp1, Yp1, Zp1}];
				}
				if (Thermal->curved_boundaries) {
					double Prod_image = 0.0;
					double total_weight = 0.0;
					for (int ip = 0; ip < Thermal->Boundaries[k].X_Image_Int.size(); ++ip) {
						int X_img = Thermal->Boundaries[k].X_Image_Int[ip];
						int Y_img = Thermal->Boundaries[k].Y_Image_Int[ip];
						int Z_img = Thermal->Boundaries[k].Z_Image_Int[ip];
						Prod_image += Thermal->Boundaries[k].W_Image_Int[ip] * Thermal->Production[{X_img, Y_img, Z_img}];
						total_weight += Thermal->Boundaries[k].W_Image_Int[ip];
					}
					if (total_weight > 0) {
						Prod_image /= total_weight;  // Average temperature at image points
						Thermal->Production[{X, Y, Z}] = 2.0 * Prod_image - Thermal->Production[{Xp1, Yp1, Zp1}];
					}
				}
			}
			// Outlet boundary condition (type 106) - 2nd order extrapolation
			if (Thermal->Boundaries[k].type == 106) {
				Xp = X - Thermal->Boundaries[k].n[0];
				Yp = Y - Thermal->Boundaries[k].n[1];
				Zp = Z - Thermal->Boundaries[k].n[2];
				int Xp1 = X - 2 * Thermal->Boundaries[k].n[0];
				int Yp1 = Y - 2 * Thermal->Boundaries[k].n[1];
				int Zp1 = Z - 2 * Thermal->Boundaries[k].n[2];
				if (!Thermal->curved_boundaries) {
					Thermal->Production[{X, Y, Z}] = (4.0 / 3.0) * Thermal->Production[{Xp, Yp, Zp}] - (1.0 / 3.0) * Thermal->Production[{Xp1, Yp1, Zp1}];
				}
				if (Thermal->curved_boundaries) {
					double Prod_image = 0.0;
					double total_weight = 0.0;
					for (int ip = 0; ip < Thermal->Boundaries[k].X_Image_Int.size(); ++ip) {
						int X_img = Thermal->Boundaries[k].X_Image_Int[ip];
						int Y_img = Thermal->Boundaries[k].Y_Image_Int[ip];
						int Z_img = Thermal->Boundaries[k].Z_Image_Int[ip];
						Prod_image += Thermal->Boundaries[k].W_Image_Int[ip] * Thermal->Production[{X_img, Y_img, Z_img}];
						total_weight += Thermal->Boundaries[k].W_Image_Int[ip];
					}
					if (total_weight > 0) {
						Prod_image /= total_weight;  // Average temperature at image points
						Thermal->Production[{X, Y, Z}] = (4.0 / 3.0) * Prod_image - (1.0 / 3.0) * Thermal->Production[{Xp1, Yp1, Zp1}];
					}
				}
			}
		}*/
	}
}

//	writes a report file for Species and chemistry, including the number of Species,	//
//	reactions, and the Species molar masses.											//
void Thermo_chemistry_cantera::write_report(const std::string& base_path, const Species_solver& Species, Parallel_MPI& MPI_parallel) const {
	if (!MPI_parallel.is_master()) {
		return;
	}
	/* Write a report file for Species and chemistry */
	std::ofstream CANTERA_report;
	CANTERA_report.open("Alborz_Results/Data/Cantera_report.dat", fstream::trunc);
	CANTERA_report << "Name of CANTERA library : " << thermochemistry_file << ", " << chemistry << ", " << transport << std::endl;
	/* Put in file header */
	int column_width = 30;
	CANTERA_report << std::setw(column_width) << left << "index";
	CANTERA_report << std::setw(column_width) << left << "Species";
	CANTERA_report << std::setw(column_width) << left << "Mk[kg/mol]";
	CANTERA_report << std::endl;
	/* Write double here */
	for (int k = 0; k < Nb_spec; ++k) {
		CANTERA_report << std::setw(column_width) << left << k;
		CANTERA_report << std::setw(column_width) << left << Species.species_name_RG[k];
		CANTERA_report << std::setw(column_width) << left << Species.Molar_mass[k];
		CANTERA_report << std::endl;
	}
	CANTERA_report << "List of reactions" << std::endl;
	column_width = 50;
	CANTERA_report << std::setw(column_width) << left << "index";
	CANTERA_report << std::setw(column_width) << left << "Reactions";
	CANTERA_report << std::endl;
	for (int k = 0; k < Nb_reac; ++k) {
		CANTERA_report << std::setw(column_width) << left << k;
		CANTERA_report << std::setw(column_width) << left << sol->kinetics()->reactionString(k) << std::endl;
	}
	CANTERA_report.close();
}

void Thermo_chemistry_cantera::compute_cp_k(unsigned int* Nb_spec, double* temperature, double* pressure, double* mass_fractions, double* cp_k) {
	sol->thermo()->setState_TPY(*temperature, *pressure, mass_fractions);
	// Retrieve specific heat capacities (J/(kg*K))
	std::vector<double> cp_k_vec(*Nb_spec);
	sol->thermo()->getPartialMolarCp(cp_k_vec.data());
	// Convert from J/(kmol*K) to J/(kg*K)
	for (int i = 0; i < *Nb_spec; ++i) {
		double molecular_weight = sol->thermo()->molecularWeight(i);
		cp_k[i] = cp_k_vec[i] / molecular_weight;
		
	}
}

/*
Choice between concentration (C_k)and mass fraction (Y_k) for Species thermo properties function
C_k = rho_0 * rho * Y_k / M_k
where ρ is the density of the mixture,Yk is the mass fraction of Species Mk is the molar mass of Species 𝑘.
Advantages:
Direct Use in Rate Equations: Concentrations are directly used in reaction rate equations, which often depend on the number of molecules per unit volume.
Normalization: Concentration inherently ensures that all Species are considered in terms of their actual amounts per unit volume.
Species Production Rates: Calculating production rates typically requires concentrations.
Disadvantages:
Density Dependency: Concentration depends on the density, which can add complexity, especially in compressible flows where density varies significantly.

Y_k = C_k * M_k / (rho_0 * rho)
where ρ is the density of the mixture, Yk is the mass fraction of Species Mk is the molar mass of Species 𝑘.
Advantages:
Mass Conservation: Mass fractions ensure that the sum of all Species mass fractions is equal to one, which is a requirement for mass conservation.
Simplicity: Mass fractions are simpler to work with in many cases, especially when dealing with mass transfer.
Density Independence: Mass fractions are independent of the density, which can simplify the implementation in compressible flows.
Normalization: Mass fractions are normalized by the total mass of the mixture, which can simplify the implementation in some cases.
Disadvantages:
Direct Use in Rate Equations: Mass fractions are not directly used in reaction rate equations, which often depend on the number of molecules per unit volume.
Species Production Rates: Calculating production rates typically requires concentrations.
*/