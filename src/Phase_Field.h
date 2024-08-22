#ifndef PHASE_FIELD_H
#define PHASE_FIELD_H

#include "ALBORZ_GlobalVariables.h"
#include "ALBORZ_SETTINGS.h"
#include "Parallel.h"
#include <vector>

typedef void (*Phase_Ini)(Scalar_field& Phase, Solid_field& solid, double N_x, double N_y, double N_z, const std::string& filename, int Zones, Parallel_MPI*);  // added W_zero in phase-field new
typedef void (*Stencil_Definition)(unsigned int& Dimension, unsigned int& Discrete_Velocity, std::vector<double>& w_alpha,
                                   std::vector<std::vector<int>>& c_alpha, std::vector<unsigned int>& alpha_bar, double& c_s2);
typedef void (*Crystal_Anisotropy)(double* n, double* NN, double& a_s, double* epsilon);

class Flow_solver;
class Thermal_solver;
class Species_solver;
class Geometry;
class stl_import;
class IO_interface;

struct phase_boundary_data {
	unsigned int X, Y, Z;
	unsigned int type;
	double Phase;
	int n[3];
};

class Phase_Field {
private:
	Vector_field pop_p, pop_old_p;  // LBM populations of temperature (old and new)
	double* pop_eq_p = nullptr;     // equilibrium populations of temperature
public:
	friend class Fluid_read_write;

	unsigned int Dimension, Discrete_Velocity;
	std::vector<double> weight;
	std::vector<double> weight_2;
	std::vector<std::vector<int>> c_alpha;
	std::vector<unsigned int> alpha_bar;
	double c_s2;
	double Gamma;

	//(double Xi_phase;//Thickness of the phase field
	double W_zero;      /// thickness of the interface   New phase-field added
	double tau_zero;    /// relaxation time   New phase-field added
	double M_phase;     /// SQ(W_zero)/tau_zero   New phase-field added
	double epsilon[3];  /// the strength of the anisotropy function  New phase-field added
	double lambda;      /// coupling of the temperature and concentration  New phase-field added
	double MC_inf;      /// the slope of the liquidious line in the phase diagram  New phase-field added
	double D_sup_sat;   /// saturation coefficient New added
	double K_sup_sat;   /// partition coefficient New added
	double L_sat;       /// depletion rate of water molecules in vapor %%%Tan%%%
	double center_x;
	double center_y;
	double center_z;
	double lambdal;  /// Thermal diffusivity for liquid
	double lambdas;  /// Thermal diffusivity for solid
	double D_S;      /// diffusion coefficient for Species

	std::vector<phase_boundary_data> Boundaries;
	Scalar_field phase;            // Second-order moment of the distribution function
	Scalar_field previous_phase;   // fluid temperature
	Scalar_field Production;       // Heat release due to chemical reaction
	Solid_field solid_phase_type;  // Thermal Boundary condition for solid nodes. 0) Fluid, 1) Solid with constant temperature, 2) Solid adiabatic, 3) Solid conjugate
	Scalar_field omega_p;          //= 1. / tau_t; //For fluid nodes
	Data_exchange_group pop_group;
	Data_exchange_group macroscopic_group;
	double n_p[3];
	double tet;
	double Radi;
	double Gamma_phase[3];  /// Phenomenological parameter which controls the horizontal and vertical growth %%%Tan%%%

	Phase_Field();
	void General_data_input(std::string, Parallel_MPI*);
	void Memory_allocation(Stencil_Definition, Parallel_MPI*);
	void initialize_p(Geometry*, stl_import*, Phase_Ini, Parallel_MPI*, std::string);
	void initialize_pop_eq_p(Parallel_MPI*, std::string);
	void initialize_BC_p(Geometry*, stl_import*, Parallel_MPI*, std::string);
	void equilibrium_p(double, double, double*);                              // compute the equilibrium populations of temperature from the fluid density and velocity
	void Crystal_DW(int, Thermal_solver*, Species_solver*, Parallel_MPI*);    /// Phase-source term from Younsi
	void Crystal_DW_T(int, Thermal_solver*, Species_solver*, Parallel_MPI*);  /// Phase-source term from Younsi
	void Crystal_DW_S(int, Thermal_solver*, Species_solver*, Parallel_MPI*);  /// Phase-source term from Younsi
	void Crystal_DW_snow(int time, Species_solver*, Parallel_MPI*);           /// Phase-source term from from Demange (Snowflakes)
	void Crystal_Heat(int, Thermal_solver*, Parallel_MPI*);                   /// Temperature-source term from Younsi
	void Thermal_Heat(int, Thermal_solver*, Parallel_MPI*);                   /// Temperature-source term from Mandelic acid
	void Crystal_Species(int, Species_solver*, Parallel_MPI*);                /// General Species-source term
	void Crystal_Specie_snow(int, Species_solver*, Parallel_MPI*);            /// Species-source term from Demange (Snowflakes)

	void Diffusion_Coefficient_computation_T(int, Thermal_solver*, Parallel_MPI*);
	void Diffusion_Coefficient_computation_S(int, Species_solver*, Parallel_MPI*);

	void Force_on_Fluid(int, Flow_solver*, Parallel_MPI*);
	void Reset_Velocity(int, Flow_solver*, Parallel_MPI*);
	void LBMNONCONS_p(int, Crystal_Anisotropy, Parallel_MPI*);  // perform LBM Thermal operations Crystal_DW new added
	void BC_p(int, Flow_solver*, Parallel_MPI*);
	void momenta_p(Parallel_MPI*);                              // compute temperature from the populations

	void Data_Exchange(Parallel_MPI*);
	void Data_Exchange_Macroscopic(Parallel_MPI* MPI_parallel);
	bool check_residual(double, Parallel_MPI*, int);

	void Recovery_write(Parallel_MPI*, int&);
	void Recovery_read(Parallel_MPI*, int&);
	void register_recovery(IO_interface& io);
	~Phase_Field();
};

void Inline_User_Defined_p(Scalar_field&, Solid_field&, double, double, double, const std::string&, int, Parallel_MPI*);  // added W_zero in phase-field new

void Anisotropy_multi(double*, double*, double&, double*);
void Tetrahedral_2D(double*, double*, double&, double*);  /// or six dendrites along coordinates in X, Y, Z directions in 3D
void Hexahedral_2D(double*, double*, double&, double*);   /// new phase-field added
void snow_vapor(double*, double*, double&, double*);
void twodendrite(double*, double*, double&, double*);
void Isotropic(double*, double*, double&, double*);
#endif

// #endif
