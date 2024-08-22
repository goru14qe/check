#ifndef SECTIONAL_SOLVER_H
#define SECTIONAL_SOLVER_H

#include <vector>
#include "Parallel.h"
#include "Fluid_read_write.h"
class Geometry;
class stl_import;
class Flow_solver;
class Thermal_solver;

typedef void (*Sectional_Ini)(double**** Ysection, int*** solid, std::vector<double> Min_radius,
                              std::vector<double> Max_radius, double N_x, double N_y, double N_z, unsigned int Nb_sections,
                              std::string filename, int Zones, Parallel_MPI* MPI_parallel);
typedef void (*Stencil_Definition)(unsigned int& Dimension, unsigned int& Discrete_Velocity, std::vector<double>& w_alpha,
                                   std::vector<std::vector<int>>& c_alpha, std::vector<unsigned int>& alpha_bar, double& c_s2);

struct sectional_boundary_data {
	unsigned int X, Y, Z;
	unsigned int type;
	std::vector<double> Y_section;
	int n[3];
	std::vector<int> directions;  /// directions where the boundary condition is applied
	                              /// -1 : do not apply, +1 : apply
};

class Sectional_solver {
private:
public:
	friend class Fluid_read_write;
	/// friend class Thermo_chemistry_cantera;

	/// -------------> Lattice Parameters
	unsigned int Dimension, Discrete_Velocity;
	std::vector<double> weight;
	std::vector<double> weight_2;
	std::vector<std::vector<int>> c_alpha;
	std::vector<unsigned int> alpha_bar;
	double c_s2;
	/// -------------> sections properties
	unsigned int Nb_sections;
	std::vector<double> Min_radius;
	std::vector<double> Max_radius;
	/// -------------> Boundary data
	std::vector<sectional_boundary_data> Boundaries;
	std::vector<double*> Boundaries_fractions;
	/// -------------> Distribution function moments
	double**** Y_section;
	double**** previous_Y_section;
	double**** Production;
	int*** solid_sectional;

	Sectional_solver();
	void General_data_input(std::string filename, Parallel_MPI* MPI_parallel);
	void Memory_allocation_FD(Stencil_Definition, Parallel_MPI*);
	void initialize_field_FD(Geometry*, stl_import*, Sectional_Ini, Flow_solver*, Thermal_solver*, Parallel_MPI*, std::string);
	void initialize_BC(Geometry*, stl_import*, Parallel_MPI*, std::string);

	void FD(int, Flow_solver*, Thermal_solver*, Parallel_MPI*);
	void BC(int, Flow_solver*, Thermal_solver*, Parallel_MPI*);

	void Data_Exchange_Macroscopic(Parallel_MPI*);

	void swap_endian(char* buffer, size_t size);
	void swap_endian(char* buffer, char* buffer2, size_t size);
	void write_bigendian(std::ofstream& file, char* buffer, size_t count, size_t size);
	template <typename T>
	void write_bigendian(std::ofstream& file, T* buffer, size_t count);
	void write_vtk(int time, int t_vtk, Geometry* Geo, Parallel_MPI* MPI_parallel);

	~Sectional_solver();
};

void Inline_User_Defined(double**** Y_section, int*** solid, std::vector<double> Min_radius,
                         std::vector<double> Max_radius, double N_x, double N_y, double N_z, unsigned int Nb_sections,
                         std::string filename, int Zones, Parallel_MPI* MPI_parallel);

#endif  // LBM_SOLVER_SPECIES_H
