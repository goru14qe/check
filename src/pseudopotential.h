#ifndef PSEUDOPOTENTIAL_H
#define PSEUDOPOTENTIAL_H

#include "ALBORZ_GlobalVariables.h"
#include "ALBORZ_SETTINGS.h"
#include "Parallel.h"
/// THIS IS JUST AN EXAMPLE
typedef void (*PseudoPotentialPressure)(double& rho, double& reduced_temperature, double& pressure, double& cs2);
typedef void (*FreeEnergy_Ini)(double*** kappa_s, const Solid_field& solid, double N_x, double N_y, double N_z, double dx, double dt, std::string filename, int Zones, Parallel_MPI* MPI_parallel);

class Flow_solver;
class Thermal_solver;
class Species_solver;
class Geometry;
class stl_import;

class pseudopotential {
public:
	double*** Psi;
	double G, G1, G2, Gw;

	pseudopotential();
	void General_data_input(std::string filename, Parallel_MPI* MPI_parallel);
	void Memory_allocation(Parallel_MPI* MPI_parallel);
	void initialize_field(Geometry* Geo, stl_import* Geo_stl, Parallel_MPI* MPI_parallel, int& tot_sol, std::string filename);
	void Get_potential(PseudoPotentialPressure get_pressure, Flow_solver* Flow, Thermal_solver* Thermal, Parallel_MPI* MPI_parallel);
	void Store_pressure(PseudoPotentialPressure get_pressure, Flow_solver* Flow, Thermal_solver* Thermal, Parallel_MPI* MPI_parallel);
	void Get_force_1storder(Flow_solver* Flow, Parallel_MPI* MPI_parallel);
	void Get_force_1storder_improved(Flow_solver* Flow, Parallel_MPI* MPI_parallel);
	void Get_force_2ndorder(Flow_solver* Flow, Parallel_MPI* MPI_parallel);
	void Get_wall_force_2ndorder(Flow_solver* Flow, Parallel_MPI* MPI_parallel);
	virtual ~pseudopotential();

protected:
private:
};

class free_energy {
public:
	double*** L_rho;
	double*** Psi;
	double*** kappa_s;
	double kappa;

	free_energy();
	void Memory_allocation(Parallel_MPI*);
	void initialize_field(Geometry*, stl_import*, FreeEnergy_Ini Ini_Field, Parallel_MPI*, int&, std::string);
	void Store_pressure(PseudoPotentialPressure get_pressure, Flow_solver* Flow, Thermal_solver* Thermal, Parallel_MPI* MPI_parallel);
	void Kortweg_stress(PseudoPotentialPressure, Flow_solver*, Thermal_solver*, Parallel_MPI*);
	void wall_interaction(Flow_solver*, Parallel_MPI*);
	virtual ~free_energy();

protected:
private:
};

void SC_pressure(double& rho, double& reduced_temperature, double& pressure, double& cs2);  /* CARNAHAN STARLING EoS */
void CS_pressure(double& rho, double& reduced_temperature, double& pressure, double& cs2);  /* CARNAHAN STARLING EoS */
void RK_pressure(double& rho, double& reduced_temperature, double& pressure, double& cs2);  /* REDLICH KWONG EoS     */
void PR_pressure(double& rho, double& reduced_temperature, double& pressure, double& cs2);  /* PENG ROBINSON EoS     */
void VdW_pressure(double& rho, double& reduced_temperature, double& pressure, double& cs2); /* VANDERWAALS EoS       */

void USERDEFINED(const Solid_field& solid, double N_x, double N_y, double N_z, double dx, double dt, std::string filename, int Zones, Parallel_MPI* MPI_parallel);
void USERDEFINED_FREE_ENERGY(double*** kappa_s, const Solid_field& solid, double N_x, double N_y, double N_z, double dx, double dt, std::string filename, int Zones, Parallel_MPI* MPI_parallel);

void smoothen_density_multiphase(Flow_solver* Flow, Parallel_MPI* MPI_parallel, unsigned int Nt, double D);

void Diffusion_Coefficient_density_dependent(int time, double rho_L, double rho_G, double nu_L, double nu_G, Flow_solver* Flow, Parallel_MPI* MPI_parallel);

void initialize_multiphase_pseudopotential(Flow_solver* Flow, pseudopotential* MultiPhase, Thermal_solver* Thermal, Species_solver* Species, Geometry* Geo, Parallel_MPI* MPI_parallel, unsigned int Nt);
#endif  // PSEUDOPOTENTIAL_H
