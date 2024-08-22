#ifndef CANTERA_INTERFACE_H_INCLUDED
#define CANTERA_INTERFACE_H_INCLUDED

#include <string>
#include <memory>

class Species_solver;
class Flow_solver;
class Thermal_solver;
class Parallel_MPI;

namespace Cantera {
class Solution;
}

class Thermo_chemistry_cantera {
private:
	/* file containing thermo-chemistry double */
	/* yaml or xml file format */
	std::string thermochemistry_file;
	/* chemistry and transport model */
	std::string chemistry, transport;
	std::string species_name;
	/* Number of Species */
	int Nb_spec;
	/* Number of reactions */
	int Nb_reac;

	/* A temporary double holder of size N_sp */
	double* temp;
	double *C_k, *h_k, *X_k;
	double *mol_mass_RG;
	double* Lewis_number;

public:
	/* Cantera sol object containin all functions pointers */
	std::shared_ptr<Cantera::Solution> sol;
	void Initialisation(Species_solver* Species, const std::string& filename, Parallel_MPI* MPI_parallel);
	void Thermo_properties(Flow_solver* Flow, Thermal_solver* Thermal, Species_solver* Species, Parallel_MPI* MPI_parallel);
	void Transport_properties(Flow_solver* Flow, Thermal_solver* Thermal, Species_solver* Species, Parallel_MPI* MPI_parallel);
	void Heat_production(Flow_solver* Flow, Thermal_solver* Thermal, Species_solver* Species, Parallel_MPI* MPI_parallel);
	void write_report(const std::string& base_path, const Species_solver& Species, Parallel_MPI& MPI_parallel) const;
	void compute_cp_k(unsigned int* Nb_spec, double* temperature, double* pressure, double* mass_fractions, double* cp_k);
};
#endif  // CANTERA_INTERFACE_H
