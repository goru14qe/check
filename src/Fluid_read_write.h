#ifndef FLUID_READ_WRITE_H
#define FLUID_READ_WRITE_H

#include <iostream>
#include <fstream>
#include <string>
#include <stdint.h>
#include <sstream>
#include <fstream>  // file stream
#include "Parallel.h"
#include "utils/Config_utils.h"
#include "Tensor.h"


using namespace std;

class Thermal_solver;
class Flow_solver;
class Species_solver;
class Phase_Field;
class Particle_sim;
class Geometry;
class stl_import;
class Fluid_read_write {
public:
	double KE;
	double HE;
	double KE_global;
	double HE_global;
	double KE2;
	double KE2_global;
	double Enstrophy;
	double Enstrophy2;
	double Enstrophy2_global;
	double residual_flow;
	double residual_thermal;
	double residual_species;

	Fluid_read_write();
	void AverageEntropy(Flow_solver* Flow, Parallel_MPI* MPI_parallel, int time);
	void AverageKineticEnergy(Flow_solver* Flow, Parallel_MPI* MPI_parallel, int time);
	void AverageEnstrophy(Flow_solver* Flow, Parallel_MPI* MPI_parallel, int time);
	void get_sim_data(Geometry*, stl_import* Geo_stl, std::string, Parallel_MPI*);
	// Computes the relative L1 distance between field and field_old. Expects the fields to be non-negative.
	void automatic_input_file_update_for_Neumann(std::string, Parallel_MPI*);
	~Fluid_read_write();
};
class DataSampling
{
	public:
	int Point_number;
	int frequency;
	std::vector<int> X_monitor, Y_monitor, Z_monitor, index_monitor;
	void Save_VelPoint(std::string filename, stl_import* Geo_stl, Parallel_MPI* MPI_parallel);
	void Out_VelPoint(const Scalar_field& field, stl_import* Geo_stl, Parallel_MPI* MPI_parallel, int &t);
};
void CFL_monitor(Flow_solver*, Parallel_MPI*, unsigned int);
void Viscosity_monitor(Flow_solver*, Thermal_solver*, Species_solver*, Parallel_MPI*, int);
bool check_L2_residual(const Scalar_field& field, const Scalar_field& field_old, const Solid_field& solid, double threshold, const Parallel_MPI& MPI_parallel, int time, const std::string& field_name);
void integrate_domain(const Scalar_field& field, const Solid_field& solid, const Parallel_MPI& MPI_parallel, int time, std::string keyword);
void Diffusion_species_monitor(Species_solver*, Flow_solver*, Parallel_MPI*, int);
//void Check_Mass_Fraction_Conservation(int, Species_solver*, Parallel_MPI*);
void Thermal_diffusion_monitor(Flow_solver*, Thermal_solver*, Parallel_MPI*, int);
void temperature_monitor(Flow_solver*, Thermal_solver*, Species_solver*, Parallel_MPI*, int);
void Energy_monitor(Flow_solver*, Thermal_solver*, Species_solver*, Parallel_MPI*, int);
#endif
