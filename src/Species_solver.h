#ifndef LBM_SOLVER_SPECIES_H
#define LBM_SOLVER_SPECIES_H

#include <vector>
#include "Parallel.h"
#include "Geometry.h"

class Geometry;
class stl_import;
class Flow_solver;
class Thermal_solver;
class Phase_Field;
class IO_interface;
class Thermo_chemistry_cantera;
class Species_solver;

extern std::vector<std::vector<double>> Ini_s;
extern std::vector<std::vector<double>> Ini_D;

typedef void (*Species_Ini)(Vector_field& Y_k, Vector_field& omega_s, Solid_field& solid, const std::vector<std::string>& species_names, double N_x, double N_y, double N_z, double Nb_spec,
                            double c_s2, const std::string& filename, int Zones, Parallel_MPI* MPI_parallel);
typedef void (*Stencil_Definition)(unsigned int& Dimension, unsigned int& Discrete_Velocity, std::vector<double>& w_alpha,
                                   std::vector<std::vector<int>>& c_alpha, std::vector<unsigned int>& alpha_bar, double& c_s2);

double Sutherland_species_diffusion(double T, double Sc_k);
//void Thrombus_growth(Species_solver*, Flow_solver*, double*** Porosity, unsigned int tm, std::string target_species, double treshhold, double epsilon, Parallel_MPI* MPI_parallel);
extern double nu;
extern double alpha;

struct temp_Data_1 {
	/// ---------------------
	///      REACTIONS
	/// ---------------------
	std::vector<std::vector<double>> Stoechio_coeff;
	std::vector<std::string> Reac_type;
	std::vector<std::vector<double>> Reac_order;
	std::vector<std::vector<double>> Reac_coeff;
	double *Y_k, T, *w_k, w_T, *M, *C, Cp, *Molfrac;
	double Rho;
	int Nb_spec, Nb_reac;
};
struct species_boundary_data {
	/* coordinates of ghost node */
	unsigned int X, Y, Z;
	/* BC type */
	unsigned int type;
	/* inner and outter volume file index */
	int V_in, V_out;
	std::vector<double> Y_k;
	double Y_av;
	std::vector<double> Molfrac;
	
	std::vector<unsigned int> k;  /// Surface reaction rate constant
	std::vector<unsigned int> kp;
	int n[3];
	std::vector<int> directions;  /// directions where the boundary condition is applied
	                              /// -1 : do not apply, +1 : apply
								  /* coordinates of ghost node */
	/* distance to closest surface */
	double normal_distance = -1;
	/* get normalized normal vector */
	double get_nx() { return n[0] / sqrt(pow(n[0], 2) + pow(n[1], 2) + pow(n[2], 2)); }
	double get_ny() { return n[1] / sqrt(pow(n[0], 2) + pow(n[1], 2) + pow(n[2], 2)); }
	double get_nz() { return n[2] / sqrt(pow(n[0], 2) + pow(n[1], 2) + pow(n[2], 2)); }
	/* get coordinate of image point */
	/* G---|---I */
	double X_Image, Y_Image, Z_Image;
	void get_image() {
		X_Image = X + 2. * n[0];
		Y_Image = Y + 2. * n[1];
		Z_Image = Z + 2. * n[2];
	}
	std::vector<double> X_Image_Int, Y_Image_Int, Z_Image_Int, W_Image_Int;
	/* Get weights for data reconstruction on Image node */
	/* Inverse distance function */
	void get_image_int_weights() {
		double u = 2;
		bool exact = false;
		int index = -1;
		W_Image_Int.resize(X_Image_Int.size());
		double W_tot = 0;
		for (int i = 0; i < X_Image_Int.size(); i++) {
			double di = sqrt(pow(X_Image_Int[i] - X_Image, 2) + pow(Y_Image_Int[i] - Y_Image, 2) + pow(Z_Image_Int[i] - Z_Image, 2));
			W_tot += pow(di, -u);
			if (di == 0) {
				exact = true;
				index = i;
			}
		}
		for (int i = 0; i < X_Image_Int.size(); i++) {
			double di = sqrt(pow(X_Image_Int[i] - X_Image, 2) + pow(Y_Image_Int[i] - Y_Image, 2) + pow(Z_Image_Int[i] - Z_Image, 2));
			W_Image_Int[i] = pow(di, -u) / W_tot;
			if (exact && i != index) W_Image_Int[i] = 0;
			if (exact && i == index) W_Image_Int[i] = 1.;
		}
	}
	bool check_weights() {
		double W_tot = 0;
		bool check = true;
		for (int i = 0; i < X_Image_Int.size(); i++)
			W_tot += W_Image_Int[i];
		if (fabs(W_tot - 1.) > 1e-12) check = false;
		return check;
	}
	double get_tot_weights() {
		double W_tot = 0;
		for (int i = 0; i < X_Image_Int.size(); i++)
			W_tot += W_Image_Int[i];
		return W_tot;
	}
};
#if defined IMPLEX && !defined REGATH_LIB
void Jacobian(int* n, double* x, double* y, double* dfy,
              int* ldfy, double* rpar, double* ipar);
void Mass(int* n, double* am, int* lmas, int* rpar, int* ipar);
void solout(int* n, double* am, int* lmas, int* rpar, int* ipar);
void ImplicitSource(int* n, double* x, double* y, double* fy,
                    double* rpar, int* ipar);
void GetDataImSolver(double* Y_k, double T, double Rho, double Cp);
void PutDataBackImSolver(double* Y_k, double* w_k, double T, double& w_T);
void Stiff_Source(Species_solver* Species, Flow_solver* Flow, Thermal_solver* Thermal, Parallel_MPI*);
#endif

class Species_solver {
private:
	/// -------------> Populations
	double* pop_eq_s;
	double* Gamma;  // This coefficient rescales the rest particles weight to allow for lower or higher schmidt number simulations
	double* Theta;  //
	double *****pop_s = nullptr, *****pop_old_s = nullptr, *****Buffer = nullptr;
	double alpha_1, alpha_2, c_1, c_2;

	unsigned int COMP;

	unsigned int passive = 2;

	void Diffusion_velocity_correction(Flow_solver*, int, int, int);
	void Diffusion_velocity_correction_predcor(Flow_solver*, Parallel_MPI*);
	void Diffusion_velocity(Flow_solver*, int, int, int);
	void Maxwell_effective_diffusion(int, int, int, double);

public:
	friend class Fluid_read_write;
	friend class Thermo_chemistry_cantera;

	/// -------------> Lattice Parameters
	unsigned int Dimension, Discrete_Velocity;
	std::vector<double> weight;
	std::vector<double> weight_2;
	std::vector<std::vector<int>> c_alpha;
	std::vector<unsigned int> alpha_bar;
	double c_s2;
	/// -------------> User-defined chemical scheme parameters
	unsigned int Nb_spec, Nb_reac;
	std::vector<std::vector<double>> Stoechio_coeff_fi;
	std::vector<std::vector<double>> Stoechio_coeff_ri;
	std::vector<std::string> Reac_type;
	std::vector<std::vector<double>> Reac_order_fi;
	std::vector<std::vector<double>> Reac_order_ri;
	std::vector<std::vector<double>> Reac_coeff;
	double Cp;
	double* cp_k;
	/// -------------> Species properties
	double molar_mass_av_0;
	Scalar_field molar_mass_av;
	double T_eq;
	Scalar_field temp_T_eq;
	Scalar_field temp_cp_eq;
	Scalar_field temp_diffusion_coefficient_eq;
	std::vector<double> Molar_mass;
	std::vector<double> Le;
	std::vector<double> Sc;
	std::vector<double> D_k;
	std::vector<double> Diffusion_Coefficient;
	std::vector<std::string> species_name_RG;
	double* Diffusion_Ref;
	/// -------------> Boundary data
	bool curved_boundaries = false;
	std::vector<species_boundary_data> Boundaries;
	std::vector<double*> Boundaries_fractions;
	/// -------------> Distribution function moments
	Vector_field mass_fraction;
	Vector_field previous_mass_fraction;
	Vector_field temp_mass_fraction;
	Vector_field V_c;
	Vector_field Y_k;
	Vector_field diffusion_coefficient;
	Vector_field Molfrac;
	double***** Flux;
	Vector_field Production;
	Solid_field solid_species;
	double sumY, sumYV, correction_factor;
	Thermo_chemistry_cantera* thermo_chemistry;
	std::string diffusion_model;
	
	/* calculation of diffusion coefficient */
	/*static inline double const_D(double &Dk, double const &nu, double const &alpha) { return Dk; }
	static inline double const_Le(double &Le, double const& nu, double const& alpha){return alpha/Le;}
	static inline double const_Sc(double &Sc, double const& nu, double const& alpha){return nu/Sc;}
	double (*poin2diff)(double&, double const&, double const&);
	*/
	// see Data_Exchange_Macroscopic
	//	Data_exchange_group macroscopic_group;

	Species_solver();
	void General_data_input(const std::string& filename, Parallel_MPI* MPI_parallel);
	void initialize_reactions(const std::string& filename, Parallel_MPI* MPI_parallel);
	void Memory_allocation(Stencil_Definition, Parallel_MPI*);
	void Memory_allocation_FD(Stencil_Definition, Parallel_MPI*);
	void initialize_field(Geometry*, stl_import*, Species_Ini, Flow_solver*, Parallel_MPI*, const std::string&);
	void initialize_field_FD(Geometry*, stl_import*, Species_Ini, Flow_solver*, Thermal_solver*, Parallel_MPI*, const std::string&);
	void initialize_field_FD_TGV_reactive(Vector_field&, Vector_field&, Scalar_field&, Solid_field&, Geometry*, stl_import*, Flow_solver*, Thermal_solver*, Parallel_MPI*, const std::string&, Thermo_chemistry_cantera*);
	void initialize_pop_eq(Flow_solver*, Parallel_MPI*, const std::string&);
	void initialize_pop_eq_snow(Flow_solver*, Phase_Field*, Parallel_MPI*, const std::string&);
	void initialize_BC(Geometry*, stl_import*, Parallel_MPI*, const std::string&);
	void Initialize_curved_boundaries_FD(stl_import*, Parallel_MPI*);
	void equilibrium(double&, double&, double&, double*, double*);
	void equilibrium_snow(double&, double*, double, double, double);
	double meq_s(double, double, double, int, double);
	double Compute_gradient(double*, double*, double, double, double, double, int);
	void LBM_SRT(int, Flow_solver*, Thermal_solver*, Parallel_MPI*);
	void LBM_SRT_snow(int, Flow_solver*, Thermal_solver*, Phase_Field*, Parallel_MPI*);
	void LBM_MRT(int, Flow_solver*, Thermal_solver*, Parallel_MPI*);
	void mass_corrector(int, Flow_solver*, Thermal_solver*, Parallel_MPI*);
	void LBM_CM_MRT(int, Flow_solver*, Thermal_solver*, Parallel_MPI*);
	void FD_Euler(int, Flow_solver*, Thermal_solver*, Parallel_MPI*);
	void FD_Euler_diffusion(int, Flow_solver*, Thermal_solver*, Parallel_MPI*, int);
	void FD_HC_Euler(int, Flow_solver*, Thermal_solver*, Parallel_MPI*, int);
	void FD_Euler_Correction_Velocity(int, Flow_solver*, Thermal_solver*, Parallel_MPI*);
	void FD_HC_SSP_RK2_step1(int, Flow_solver*, Thermal_solver*, Parallel_MPI*);
	void FD_HC_SSP_RK2_step2(int, Flow_solver*, Thermal_solver*, Parallel_MPI*);
	void FD_Fick_Euler(int, Flow_solver*, Thermal_solver*, Parallel_MPI*, int);
	void FD_Fick_SSP_RK2_step1(int, Flow_solver*, Thermal_solver*, Parallel_MPI*);
	void FD_Fick_SSP_RK2_step2(int, Flow_solver*, Thermal_solver*, Parallel_MPI*);
	void LBM_COMP(int, Flow_solver*, Thermal_solver*, Parallel_MPI*);
	void BC(int, Flow_solver*, Thermal_solver*, Parallel_MPI*);
	void momenta(Flow_solver*, Parallel_MPI*);
	void species_mole_fractions(unsigned int, Parallel_MPI* MPI_parallel);
	void mass_conservation_report(int, Flow_solver*, Thermal_solver*, Parallel_MPI*);
	void Check_Mass_Fraction_Conservation(Parallel_MPI*);
	void Spatial_filter(double);
	void Recovery_write(Parallel_MPI* MPI_parallel, int& t);
	void Recovery_read(Parallel_MPI* MPI_parallel, int& t);
	void register_recovery(IO_interface& io);
	void User_defined_production(Thermal_solver*, Flow_solver*, Parallel_MPI*);
	void Molar_mass_computation(Flow_solver*, Thermal_solver*, Parallel_MPI*);
	void Sponge_zone(double, double, int, double, Flow_solver*, stl_import*, Parallel_MPI*);
	void Diffusion_Coefficient_computation(Flow_solver*, Thermal_solver*, Parallel_MPI*);
	void Data_Exchange(Parallel_MPI*);
	void Data_Exchange_Macroscopic(Parallel_MPI*);
	double Consumption_rate_monitor(Flow_solver*, Parallel_MPI*, int, unsigned int);
	double Consumption_rate_monitor_each_species(Flow_solver*, Parallel_MPI*, int, unsigned int);
	double Fo_monitor(Flow_solver*, Parallel_MPI*, int, unsigned int);
	void calculateLewisNumber(Flow_solver*, Thermal_solver*, Species_solver*, Parallel_MPI*, int output, int time);

	bool check_residual();
	~Species_solver();
};

class Thickened_flame{
    friend class Species_solver;
	public:
	Solid_field solid_species;
	/* flame laminar thickness, thickened flame thickness, and flame speed */
	double delta_0, delta_1, F, S_L;
	/* fresh and burnt gas temperature for dynamic thickening*/
	double T_fresh, T_burnt;
	/* Thickening region size */
	double sigma;
	/* Filter width for velocity fluctuation computation */
	double filter_width;

	double ***Efficiency;
	Thickened_flame();
	~Thickened_flame();

	void General_data_input(const std::string& filename, Parallel_MPI* MPI_parallel);
	void Memory_allocation(Parallel_MPI* MPI_parallel);
	void Get_efficiency(Flow_solver*, Thermal_solver*, Species_solver*, Parallel_MPI*);

	void Data_Exchange (Parallel_MPI* MPI_parallel);
	void Apply_filter(Flow_solver*, Thermal_solver*, Species_solver*, Parallel_MPI*);
};

void Inline_User_Defined(Vector_field&, Vector_field&, Solid_field&, const std::vector<std::string>&, double, double, double, double, double, const std::string&, int, Parallel_MPI*);
void TGV3Dcold_species(Vector_field&, Vector_field&, Solid_field&, const std::vector<std::string>&, double, double, double, double, double, const std::string&, int, Parallel_MPI*);
void TGV3Dreacting_species(Vector_field&, Vector_field&, Solid_field&, const std::vector<std::string>&, double, double, double, double, double, const std::string&, int, Parallel_MPI*);
void Gaussian_species(Vector_field&, Vector_field&, Solid_field&, const std::vector<std::string>&, double, double, double, double, double, const std::string&, int, Parallel_MPI*);

#endif  // LBM_SOLVER_SPECIES_H
