#ifndef THERMAL_SOLVER_H
#define THERMAL_SOLVER_H

#include "ALBORZ_GlobalVariables.h"
#include "ALBORZ_SETTINGS.h"
#include "Parallel.h"
#include "Geometry.h"
#include "Thermal_solver.h"
#include "Tensor.h"
#include <vector>

class Geometry;
class stl_import;
class Flow_solver;
class Phase_Field;
class IO_interface;
class Thermo_chemistry_cantera;
class Species_solver;
class Thermal_solver;

extern std::vector<double> Ini_T;
extern std::vector<double> Ini_cp;
extern std::vector<double> Ini_lambda;

typedef void (*Temperature_Ini)(Scalar_field& T, Scalar_field& c_p, Scalar_field& omega_t, Solid_field& solid, double N_x, double N_y, double N_z, double cs_2, double T_0, double E_0, double rho_0, const std::string& filename, int Zones,
                                Parallel_MPI* MPI_parallel);
typedef void (*Stencil_Definition)(unsigned int& Dimension, unsigned int& Discrete_Velocity, std::vector<double>& w_alpha,
                                   std::vector<std::vector<int>>& c_alpha, std::vector<unsigned int>& alpha_bar, double& c_s2);
typedef double (*Enthalpy)(double& T);

void NewtonRaphson(double&, double, double, double, double, double, double);
double EnthalpyTemp(double T, double cp);
double HeatCapacity(double T, double cp_in);
double Sutherland_thermal_diffusion(double nu_star, double T_star, double S, double T, double Pr);

struct thermal_boundary_data {
	unsigned int X, Y, Z;
	unsigned int type;
	double T;
	int n[3];
	std::vector<int> directions;  /// directions where the boundary condition is applied
	                              /// -1 : do not apply, +1 : apply

	int V_in, V_out;
	double normal_distance = -1;
	/* get normalized normal vector */
    double get_nx(){return n[0]/sqrt(pow(n[0], 2) + pow(n[1], 2) + pow(n[2], 2) );}
    double get_ny(){return n[1]/sqrt(pow(n[0], 2) + pow(n[1], 2) + pow(n[2], 2) );}
    double get_nz(){return n[2]/sqrt(pow(n[0], 2) + pow(n[1], 2) + pow(n[2], 2) );}
	/* get coordinate of image point */
	/* G---|---I */
	double X_Image, Y_Image, Z_Image;
    void get_image(){X_Image = X + 2.*n[0]; Y_Image = Y + 2.*n[1]; Z_Image = Z + 2.*n[2];}
	std::vector<double> X_Image_Int, Y_Image_Int, Z_Image_Int, W_Image_Int;
	/* Get weights for data reconstruction on Image node */
	/* Inverse distance function */
	void get_image_int_weights(){
		double u = 2;
		bool exact = false;
		int index = -1;
		W_Image_Int.resize(X_Image_Int.size());
		double W_tot = 0;
		for(int i=0; i<X_Image_Int.size(); i++){
			double di = sqrt( pow(X_Image_Int[i]-X_Image,2) + pow(Y_Image_Int[i]-Y_Image,2) + pow(Z_Image_Int[i]-Z_Image,2) );
			W_tot += pow(di,-u);
			if (di == 0) {exact = true; index = i;}
		}
		for(int i=0; i<X_Image_Int.size(); i++){
			double di = sqrt( pow(X_Image_Int[i]-X_Image,2) + pow(Y_Image_Int[i]-Y_Image,2) + pow(Z_Image_Int[i]-Z_Image,2) );
			W_Image_Int[i] = pow(di,-u)/W_tot;
			if (exact && i!=index) W_Image_Int[i] = 0;
			if (exact && i==index) W_Image_Int[i] = 1.;
		}
	}
	bool check_weights(){
		double W_tot = 0;
		bool check = true;
		for(int i=0; i<X_Image_Int.size(); i++) W_tot += W_Image_Int[i];
		if( fabs(W_tot-1.)>1e-12 ) check = false;
		return check;
	}
	double get_tot_weights(){
		double W_tot = 0;
		for(int i=0; i<X_Image_Int.size(); i++) W_tot += W_Image_Int[i];
		return W_tot;
	}
};
class Thermal_solver {
private:
	int az;
	Vector_field pop_t;  // LBM populations of temperature (old and new)
	Vector_field pop_old_t;
	double* pop_eq;  // equilibrium populations of temperature
public:
	friend class Fluid_read_write;
	Thermo_chemistry_cantera* thermo_chemistry;
	unsigned int Dimension, Discrete_Velocity;      // DdQq, Dimension = d, and Discrete_Velocity = q
	std::vector<double> weight;                     // Standard weights
	std::vector<double> weight_2;                   // second set of weight functions for non-linear equation solver
	std::vector<std::vector<int>> c_alpha;          // unit vector for each stencil discrete velocity
	std::vector<unsigned int> alpha_bar;            // index of opposite discrete velocity vector
	double c_s2;                                    // inverse of square of non-dimensional sound-speed
	double Gamma;                                   // Free parameter for non-linear equation solver, set to one for simple heat equation
	double VAR_switch;                              // Choice of zeroth-order moment
	double T_0;                                     // Temperature non-dimensionalization parameter, units in K
	double E_0;                                     // Energy non-dimensionalization parameter, units in
	std::vector<thermal_boundary_data> Boundaries;  // List of boundary nodes with corresponding data
	bool curved_boundaries = false;
	Scalar_field energy;                            // Zeroth-order moment of the distribution function
	Scalar_field energy_previous;
	Scalar_field solid_particle;
	Scalar_field initial_CP;
	Scalar_field temperature;           // Second-order moment of the distribution function
	Scalar_field previous_temperature;  // Previous value of zeroth-order moment, for convergence study
	Scalar_field temp_temperature;
	Scalar_field force_thermal;          // force term for Thermal lbm
	Scalar_field temp_force_thermal;     // For particle solver
	Scalar_field Production;             // Heat release due to chemical reaction
	Solid_field solid_thermal_type;      // Type of node ,-1 in simulation domain, +1 outside simulation domain
	Scalar_field thermal_diffusion_coefficient;  // = 1. / tau_t; //For fluid nodes
	Scalar_field c_p;                    // Heat capacity
	
	double Flux[3];                      // Flux computed for the purposes of the LKS formulation
	double A;
	double gbeta;       // In NATURAL convection: Ra=Gr*Pr=gbeta*dT*H^3/(alpha_t*nu)==>gbeta=alpha_t*nu*Ra/(dT*H^3). If natural convection is not important, set gbeta=0.0. Negative gbeta=g works in positive direction
	double T_infinity;  // reference temperature at infinity in case of natural convection
	Data_exchange_group pop_group;
	Data_exchange_group macroscopic_group;

	Thermal_solver();
	void General_data_input(std::string, Parallel_MPI*);
	// Returns the average in the local domain or the global average in the master process.
	double average_temp(const Parallel_MPI& MPI_parallel) const;
	void temp_monitor(unsigned int time_step, Flow_solver*, const Parallel_MPI& MPI_parallel);                    // GENERAL SIMULATION DATA INPUT
	void Memory_allocation(Stencil_Definition, Parallel_MPI*);                                                    // MEMORY ALLOCATION FOR VARIABLES
	void Memory_allocation_FD(Stencil_Definition, Parallel_MPI*);                                                 // MEMORY ALLOCATION FOR VARIABLES (FD SOLVER)
	void initialize_field(Geometry*, stl_import*, Temperature_Ini, Flow_solver*, Parallel_MPI*, std::string);     // INITIALIZATION OF ENERGY FIELD
	void initialize_field_FD(Geometry*, stl_import*, Temperature_Ini, Flow_solver*, Parallel_MPI*, std::string);  // INITIALIZATION OF ENERGY FIELD (FD SOLVER)
	void initialize_field_FD_TGV_temp_reactive(Scalar_field&, Scalar_field&, Scalar_field&, Solid_field&, Geometry*, stl_import*, Flow_solver*, Species_solver*, Parallel_MPI*, std::string, Thermo_chemistry_cantera*);

	void initialize_pop_eq(Flow_solver*, Parallel_MPI*, std::string);
	void initialize_pop_eq_crystal(Flow_solver*, Parallel_MPI*, const std::string&);  // INITIALIZATION OF POPULATION WITH EQUILIBRIUM                                     // INITIALIZATION OF POPULATION WITH EQUILIBRIUM
	void initialize_BC(Geometry*, stl_import*, Parallel_MPI*, std::string);           // IN_MEMORY STORAGE OF BOUNDARY NODES
	void Initialize_curved_boundaries_FD(stl_import*, Parallel_MPI*);
	void equilibrium_crystal(double&, double*, double*);
	void equilibrium(double&, double&, double&, double*, double*);  // CLASSICAL LINEAR EQUILIBRIUM FUNCTION
	void LBM_SRT(int, Flow_solver*, Parallel_MPI*);                 // LBM STEP WITH SRT FORMULATION
	void LBM_SRT_crystal(int, Flow_solver*, Parallel_MPI*);
	void LBM_MRT(int, Flow_solver*, Parallel_MPI*);
	void FD_Euler(int, Flow_solver*, Species_solver*, Parallel_MPI*);  // FINITE DIFFERENCE SOLVER
	void FD_Euler_diffusion(int, Flow_solver*, Species_solver*, Parallel_MPI*);
	void FD_Euler_species_diffusion_enthalpy(int, Flow_solver*, Species_solver*, Parallel_MPI*);
	void FD_Euler_pressure(int, Flow_solver*, Species_solver*, Parallel_MPI*);
	void FD_ugradT(int, Flow_solver*, Species_solver*, Parallel_MPI*);
	void BC(int, Flow_solver*, Parallel_MPI*);  // APPLICATION OF BOUNDARY CONDITIONS
	void momenta(Flow_solver*, Parallel_MPI*);  // COMPUTATION OF MOMENTA
	void Sponge_zone(double, double, int, Flow_solver*, stl_import*, Parallel_MPI*);
	void momenta_crystal(Parallel_MPI*);
	void momenta_FD(Flow_solver*, Parallel_MPI*);
	void viscous_heating(Flow_solver*, Species_solver*, Parallel_MPI*);
	void Boussinesq_force(int, Flow_solver*, Thermal_solver*, Parallel_MPI*);
	void Data_Exchange(Parallel_MPI*);                        // DATA EXCHANGE BETWEEN WORKER PROCESSORS
	void Data_Exchange_Macroscopic(Parallel_MPI*);            // MACROSCOPIC DATA EXCHANGE BETWEEN WORKER PROCESSORS
	void Recovery_write(Parallel_MPI* MPI_parallel, int& t);  // WRITING OF RECOVERY FILE
	void Recovery_read(Parallel_MPI* MPI_parallel, int& t);   // READING OF RECOVERY FILE
	void register_recovery(class IO_interface& io);
	void check_residual(Parallel_MPI*, int);  // COMPUTATION AND EVALUATION OF RESIDUAL STATE
	void Diffusion_Coefficient_computation(Flow_solver*, Species_solver*, Parallel_MPI*);  // COMPUTATION AND EVALUATION OF THERMAL DIFFUSION
	void swap_endian(char* buffer, size_t size);
	void swap_endian(char* buffer, char* buffer2, size_t size);
	void write_bigendian(std::ofstream& file, char* buffer, size_t count, size_t size);
	template <typename T>
	void write_bigendian(std::ofstream& file, T* buffer, size_t count);
	void write_vtk(int time, int t_vtk, stl_import* Geo, Parallel_MPI* MPI_parallel);
	double Fo_monitor(Flow_solver*, Parallel_MPI*, int, unsigned int);
	double calculateThermalDiffusivity(int time, Thermal_solver*, Flow_solver*, Parallel_MPI*);
	// Computes the average temperature at non-solid nodes.
	~Thermal_solver();
};
class Heat_source {
private:
public:
	std::vector<double> Xs, Ys, Zs, Tstart, Tend, Rs, Es;
	std::vector<double> Rsigma, Tsigma;
	void read_input(std::string, double, Parallel_MPI*);  // GENERAL SIMULATION DATA INPUT
	void Add_heat(Thermal_solver*, Parallel_MPI*, int, int, int, int);
};

void Inline_User_Defined(Scalar_field&, Scalar_field&,
                         Scalar_field&, Solid_field&, double, double,
                         double, double, double, double, double, const std::string&, int, Parallel_MPI*);  // USER-DEFINED INITIAL CONDITIONS
void TGV3Dcold_thermal(Scalar_field&, Scalar_field&,
                       Scalar_field&, Solid_field&, double, double,
                       double, double, double, double, double, const std::string&, int, Parallel_MPI*);  // USER-DEFINED INITIAL CONDITIONS
void TGV3Dreacting_thermal(Scalar_field&, Scalar_field&,
						   Scalar_field&, Solid_field&, double, double,
                           double, double, double, double, double, const std::string&, int, Parallel_MPI*);
void Gaussian_thermal(Scalar_field&, Scalar_field&,
                      Scalar_field&, Solid_field&, double, double,
                      double, double, double, double, double, const std::string&, int, Parallel_MPI*);
#endif
