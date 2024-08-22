#ifndef FLOW_SOLVER_H
#define FLOW_SOLVER_H

#include "ALBORZ_GlobalVariables.h"
#include "ALBORZ_SETTINGS.h"
#include "Parallel.h"
#include "Tensor.h"
#include "Lattice_ops.h"
#include "Geometry.h"
#include <vector>
#include <cassert>
#include <map>
#include <cmath>
#include <random>

typedef void (*Flow_Ini)(Vector_field& u, Scalar_field& dens, Vector_field& force, Scalar_field& omega, Solid_field& solid, double N_x, double N_y, double N_z, double dx, double dt, double rho_0, double c_s2, const std::string& filename, int Zones, Parallel_MPI* MPI_parallel);
typedef std::map<std::string, Flow_Ini> FLOWINIMap;

typedef void (*Stencil_Definition)(unsigned int& Dimension, unsigned int& Discrete_Velocity, std::vector<double>& w_alpha,
                                   std::vector<std::vector<int>>& c_alpha, std::vector<unsigned int>& alpha_bar, double& c_s2);
double Sutherland_viscosity(double nu_star, double T_star, double S, double T);

class Geometry;
class stl_import;
class Thermal_solver;
class Species_solver;
class Fluid_read_write;
class FLOWINI;
class Flow_solver;
class IO_interface;
///-------------------------------------------------------------------///-----------------------------///
///             THIS STRUCTURE HOLDS DATA RELATIVE TO BOUNDARY NODES  ///                             ///
///-------------------------------------------------------------------///-----------------------------///

// available boundary conditions
enum struct Fluid_BC_type : unsigned {
	WALL_WEAK = 1,           // Wall BC (weakly-compressible solver)
	VELOCITY_WEAK = 2,       // Velocity BC (weakly-compressible solver)
	PRESSURE_WEAK = 3,       // Pressure BC (weakly-compressible solver)
	ZERO_GRAD_1_WEAK = 4,    // Zero-gradient (1st-order) BC (weakly-compressible solver)
	ZERO_GRAD_2_WEAK = 5,    // Zero-gradient (2nd-order) BC (weakly-compressible solver)
	VELOCITY_NEQ = 6,        // Velocity BC (non-equilibrium extrapolation)
	PRESSURE_NEQ = 7,        // Pressure BC (non-equilibrium extrapolation)
	VELOCITY_EQ = 8,         // Velocity BC (equilibrium)
	PRESSURE_EQ = 9,         // Pressure BC (equilibrium)
	VELOCITY_LMNA = 12,      // Velocity BC (LMNA solver)
	PRESSURE_LMNA = 13,      // Pressure BC (LMNA solver)
	NON_REFLECTING_OUTFLOW = 14,  // Non-reflecting open boundary (LMNA solver)
	VELOCITY_NEQ_LMNA = 16,  // Velocity BC (non-equilibrium extrapolation) (LMNA solver)
	PRESSURE_NEQ_LMNA = 17,  // Pressure BC (non-equilibrium extrapolation) (LMNA solver)
	CONVECTIVE_LMNA = 18,   // Zero-gradient (1st-order) BC that actually enforces a 0 velocity gradient ot the last fluid node with variable density
	SYMMETRICAL_LMNA = 0,    // Symmetrical boundary condition (LMNA solver)
};

struct Flow_solid_boundary_node {
	Index_vec3 idx;
	Flat_index flat_idx;
	Index_vec3 n;
	Vec3 v;                        // velocity at boundary
	std::vector<int> directions;   // directions where the boundary condition is applied
	std::vector<double> distance;  // distance to wall for curved boundaries normalized by the length of c_alpha
	                               // not actually needed currently
	                               // proper implementation would also require modifications to Flow_boundary_data::move_nodes
								   //	std::vector<size_t> fluid_boundary_idx;  // index for fluid neighbors in Flow_boundary_data::node_data
};

struct Flow_fluid_boundary_node {
	enum Img_point_flag : unsigned {
		FIRST = 1 << 0,
		SECOND = 1 << 1
	};

	Index_vec3 idx;
	Flat_index flat_idx;
	Vec3 normal;
	Vec3 v;                                  // velocity at boundary
	double distance;                         // with curved boundary shortest distance to wall, otherwise 1 (even when the surface normal is diagonal)
	interpolation::Stencil img_stencil;      // next node in surface normal direction from the fluid node
	interpolation::Stencil img_stencil_2;    // next node + 1 in surface normal direction from the fluid node
	Index_vec3 n;                            // simple normal pointing away from the surface
	std::vector<int> directions;             // directions with solid neighbors
	std::vector<size_t> solid_boundary_idx;  // index for solid neighbors in Flow_boundary_data::fluid_node_data
};

struct Flow_boundary_data {
	// get velocity with added turbulence if enabled
	Vec3 get_velocity() const;
	Vec3 get_turbulence() const;

	// removes nodes (both solid and fluid) indicated by indices for fluid_node_data
	void remove_nodes(std::vector<size_t> fluid_idx);

	// move over solid nodes to dest and update fluid nodes of dest accordingly
	void move_nodes(const std::vector<size_t>& fluid_idx, const std::vector<size_t>& fluid_idx_dest, Flow_boundary_data& dest);

	Fluid_BC_type type;
	int in_zone;     // index of the fluid zone
	int out_zone;    // index of the solid zone
	unsigned index;  // index of this BC according to the input config

	int X, Y, Z;
	double w_c;		/// filter coefficient
    int filtered;
    double *pop_filtered = nullptr;	 /// previous step populations for filtered boundary

	// data that is the same for every node
	double turbulence_intensity = 0.0;  // strength of noise added for velocity
	Vec3 v = {};                        // velocity at boundary; this value is only used during initialization
	double p = 0.0;                     // pressure at boundary

	std::vector<Flow_solid_boundary_node> node_data;        // list of solid nodes of this boundary
	std::vector<Flow_fluid_boundary_node> fluid_node_data;  // list of fluid nodes of this boundary
};

struct FlowResults {
	double avg_pres;
	double avg_den;
	double avg_vel;
	double max_pres;
	double max_den;
	double max_vel;
};

class Time_dependent_boundary {
private:
public:
	friend class Flow_solver;
	unsigned int number_of_BC;
	unsigned int number_of_data_points;
	std::vector<unsigned int> index_of_BC;
	std::vector<std::string> data_filename_of_BC;
	Tensor<Vec3, 2> velocity;
	Tensor<double, 2> t;

	void data_input(const std::string& filename, const Parallel_MPI& MPI_parallel);
	void set_values(unsigned int tm, Flow_solver& Flow, const Parallel_MPI& MPI_parallel) const;
};

class Non_uniform_boundary {
public:
	friend class Flow_solver;
	unsigned int number_of_BC;
	double x_ref, y_ref, z_ref;

	std::vector<unsigned int> index_of_BC;
	std::vector<int> Dimension;
	std::vector<Vec3> center;
	std::vector<Vec3> max_velocity;
	std::vector<double> radii;
	std::vector<double> thickness;

	enum struct Type {
		POISEULLE,
		TANH,
		COUNT
	};
	constexpr static std::array<const char*, static_cast<size_t>(Type::COUNT)> type_names = {
		"Poiseuille",
		"Tanh"};

	std::vector<Type> type;

	void data_input(const std::string& filename, const Parallel_MPI& MPI_parallel);
	void set_values(Flow_solver& Flow, const stl_import& geo_stl, const Parallel_MPI& MPI_parallel) const;

	Vec3 get_velocity_by_node(size_t j, const Index_vec3& idx, const stl_import& geo_stl, const Parallel_MPI& MPI_parallel) const;
	Vec3 get_velocity(int j, double distance) const;
};

///-------------------------------------------------------------------///-----------------------------///
///                            FLUID CLASS                            ///                             ///
///-------------------------------------------------------------------///-----------------------------///
class Flow_solver {
private:
public:
	friend class Fluid_read_write;
	friend class Porous;
	friend class time_dependent_boundary;

	std::string initial_condition_type;
	///-------------------------------------------------------------------///-----------------------------///
	///                        DATA DEFINIG THE STENCIL                   ///                             ///
	///-------------------------------------------------------------------///-----------------------------///
	unsigned int Dimension;                 /// Physical dimension of the simulation
	unsigned Discrete_Velocity;             /// Number of discrete velocities in stencil
	double c_s2;                            /// Represents 1/cs^2
	std::vector<double> weight;             /// weights associated to each velocity
	std::vector<std::vector<int>> c_alpha;  /// definition of stencil velocities
	std::vector<unsigned int> alpha_bar;    /// index of opposite stencil velocity
	std::vector<Flat_index> c_alpha_offsets;
	///-------------------------------------------------------------------///-----------------------------///
	///                        TRT PARAMETERS                             ///                             ///
	///-------------------------------------------------------------------///-----------------------------///
	double tau, nu, nu_0;
	///-------------------------------------------------------------------///-----------------------------///
	///                        ARRAY OF DATA (POPULATIONS)                ///                             ///
	///-------------------------------------------------------------------///-----------------------------///
	Population_field pop;      /// current time-step populations
	Population_field pop_old;  /// previous time-step populations
	double* pop_eq;            /// equilibrium populations
	double* pop_w;             /// equilibrium populations (used as temporary array for boundary conditions)
	double* pop_neq;
	double* pop_temp;
	///-------------------------------------------------------------------///-----------------------------///
	///                        ARRAY OF DATA (MOMENTS)                    ///                             ///
	///-------------------------------------------------------------------///-----------------------------///
	Solid_field is_solid;                      /// geometry
	Scalar_field velocity_magnitude;           /// current time-step velocity magnitudes
	Scalar_field previous_velocity_magnitude;  /// previous time-step velocity magnitudes
	Scalar_field density;                      /// current time-step densities
	Scalar_field previous_density;
	Scalar_field pressure;           /// current time-step pressures
	Scalar_field previous_pressure;  /// previous time-step pressures
	Vector_field velocity;           /// current time-step velocities
	Vector_field previous_velocity;  /// previous time-step velocities
	Scalar_field viscosity;          /// current time-step viscosities
	Vector_field force;              /// current time-step force
	Vector_field temp_force;         /// temporary array for force
	Scalar_field divU;               /// current time-step divU (only used for LMNA solver)
	Scalar_field previous_divU;
	Vector_field velocity_corrections;
	Non_solid_update non_solid_lattice;
	std::array<std::array<double, 3>, 3> Stress;  /// current step stress tensor
	Scalar_field alpha_entropic;                  /// Entropic stabilizer parameter, only active if DEBUG_MODE is on
	Data_exchange_group pop_group;
	Data_exchange_group macroscopic_group;
	///-------------------------------------------------------------------///-----------------------------///
	///                        GENERAL DATA                               ///                             ///
	///-------------------------------------------------------------------///-----------------------------///
	double gravity[3];                           /// gravity
	double rho_0;                                /// reference density (used for non-dimensionalization
	double M_av;                                 /// reference molar mass (used for initial conditions of LMNA)
	double mass_0, p_th_0, p_th, p_th_previous;  /// background thermodynamic pressure, p_th_0 : Initial, p_th : current step, p_th_previous : previous step, mass_0: Initial mass
	double physical_time = 0;                    /// holds the physical time
	double fluid_constant_viscosity;             /// Only used when fluid viscosity is constant
	                                             ///-------------------------------------------------------------------///-----------------------------///
	                                             ///                        BOUNDARY CONDITIONS                        ///                             ///
	                                             ///-------------------------------------------------------------------///-----------------------------///

	std::vector<Flow_boundary_data> boundaries;
	bool curved_bounce_back = false;
	///-------------------------------------------------------------------///-----------------------------///
	///                        PARTICLE DATA                              ///                             ///
	///-------------------------------------------------------------------///-----------------------------///
	struct solid_node {
		/// Constructor
		solid_node() {
			x = 0;
			y = 0;
		}
		/// Elements
		double x;  // current x-position
		double y;  // current y-position
	};
	struct solid_surface {
		/// Constructor
		solid_surface() {
			num_surf_nodes = surface_num_nodes;
			node_surf = new solid_node[num_surf_nodes];
		}

		/// Elements
		int num_surf_nodes;     // number of surface nodes on solid structure
		solid_node* node_surf;  // center node
	};
	solid_surface solid;  // create solid surface tracer
	int mkmk = 5;
	///-------------------------------------------------------------------///-----------------------------///
	///                            FUNCTIONS                              ///                             ///
	///-------------------------------------------------------------------///-----------------------------///
public:
	Flow_solver();
	void Update_physical_time(unsigned int, unsigned int, Parallel_MPI*);
	// void temp_monitor(unsigned int time_step, const Thermal_solver& Thermal, const Parallel_MPI& MPI_parallel);
	void Recovery_read_physical_time(unsigned int, unsigned int);
	void General_data_input(const std::string& filename,                    /// function reading input file
	                        Parallel_MPI* MPI_parallel);                    ///
	void initialize_field(Geometry*, stl_import*,                           /// function initializing macroscopic variables
	                      Flow_Ini, Thermal_solver*, Species_solver*,       ///
	                      Parallel_MPI*, std::string);                      ///
	void initialize_pop_eq(Parallel_MPI*,                                   /// initialize populations at equilibrium
	                       Thermal_solver*, Species_solver*, std::string);  ///
	void initialize_pop_eq_LMNA(Parallel_MPI*,                              /// initialize populations at equilibrium (LMNA)
	                            std::string);                               ///
	void initialize_pop_grad(Parallel_MPI*,                                 /// initialize populations at non-equilibrium
	                         Thermal_solver*, Species_solver*, std::string);
	void initialize_rho_LMNA(Parallel_MPI*,
	                         Thermal_solver*, Species_solver*);
	void initialize_rho_compressible(Parallel_MPI*,
	                                 Thermal_solver*, Species_solver*);
	void initialize_p_th_LMNA(Parallel_MPI*, Thermal_solver*, Species_solver*);  /// initialize thermodynamic pressure
	void Memory_allocation(Stencil_Definition, Parallel_MPI*);                   /// initialize arrays (allocate memory to dynamic arrays)
	void initialize_BC(Geometry*, stl_import*,                                   /// initialize boundary conditions
	                   Parallel_MPI*, const std::string& filename);              ///
	void initialize_curved_boundaries(const stl_import& geo_stl, const Parallel_MPI& MPI_parallel);
	void equilibrium(double, double, const double*, double*);  /// compute the equilibrium populations from the fluid density and velocity
	void LBM_CUMULANT_LMNA(int, Thermal_solver*, Species_solver*, Geometry*, Parallel_MPI*);
	void LBM_CM_MRT_LMNA(int, Thermal_solver*, Species_solver*, Parallel_MPI*); // perform LBM operations
	void LBM_CM_MRT(int, Thermal_solver*, Species_solver*, Geometry*, Parallel_MPI*);
	void LBM_MRT_LMNA(int, Thermal_solver*, Species_solver*, Geometry*, Parallel_MPI*); // perform LBM operations
	void LBM_SRT_LMNA(int, Thermal_solver*, Species_solver*, Geometry*, Parallel_MPI*); // perform LBM operations
	void BC(int, Thermal_solver*, Species_solver*, Parallel_MPI*);  // perform boundary conditions operations

	void BC_filter(int, Parallel_MPI*);
	void Sponge_zone(double, double, int, stl_import*, Parallel_MPI*);
	void momenta(int, Parallel_MPI*, Thermal_solver*, Species_solver*);       // compute fluid density and velocity from the populations
	void momenta_LMNA(int, Parallel_MPI*, Thermal_solver*, Species_solver*);  // compute fluid density and velocity from the populations
	void update_p_th_LMNA(int, Parallel_MPI*, Thermal_solver*, Species_solver*);
	void update_density_LMNA(int, Parallel_MPI*, Thermal_solver*, Species_solver*);
	void swap_divergence_LMNA(int, Parallel_MPI*, Thermal_solver*, Species_solver*);
	void Stress_tensor(unsigned X, unsigned int Y, unsigned int Z);
	void register_recovery(IO_interface& io, const Parallel_MPI& MPI_parallel);
	void write_curved_boundary_data(const std::string& base_name, const Parallel_MPI& MPI_parallel, const stl_import& geo_stl);
	void Data_Exchange(Parallel_MPI*);
	void Data_Exchange_Macroscopic(Parallel_MPI*);

	void Diffusion_Coefficient_computation(Thermal_solver*, Species_solver*, Parallel_MPI*);
	void Diffusion_Coefficient_const_viscosity(Thermal_solver*, Species_solver*, Parallel_MPI*);

	void simulation_divergence_monitor(Parallel_MPI* MPI_parallel);
	FlowResults average_values(const Parallel_MPI& MPI_parallel) const;
	~Flow_solver();

private:
	// Currently the functions need atleast one parameter so that
	// the variadic macro DISPATCH_BY_STENCIL can be used.
	template <int Dim, int Discrete_velocity>
	void LBM_CM_MRT_impl(const Parallel_MPI& MPI_parallel, const Thermal_solver& Thermal);

	template <int Dim, int Discrete_velocity>
	void momenta_impl(const Parallel_MPI& MPI_parallel);

	void initialize_boundary_corners();
};

double FROB(double** A, double** B);
void Flow_rate(double* m_dot, Flow_solver*, Parallel_MPI*);
void Fix_flow_rate(double* m_dot, double* m_dot_in, double* n, Flow_solver*, Parallel_MPI*);
void add_vortex(std::string, Flow_solver* Flow, Parallel_MPI* MPI_parallel, stl_import* Geo, unsigned int N_x, unsigned int N_y, unsigned int N_z);
double passot_pouquet_spec(double k, double ke, double urms);
void add_HIT(std::string, Flow_solver* Flow, Parallel_MPI* MPI_parallel, stl_import* Geo, unsigned int N_x, unsigned int N_y, unsigned int N_z);
void PIPE3D(std::string, Flow_solver*, Parallel_MPI*, stl_import* Geo);
void PIPE3DFORCE(Flow_solver* Flow, Parallel_MPI* MPI_parallel, stl_import* Geo);
void time_dependent_force(Flow_solver* Flow, Parallel_MPI* MPI_parallel);

///   Mapping Stencil initialization functions
class FLOWINI {
public:
	FLOWINIMap FLOWINIFunction;
	void Initialize();
	FLOWINI();
	~FLOWINI();

private:
};
// typedef void (*Flow_Ini)(Vector_field& u, Scalar_field& dens, Vector_field& force, Scalar_field& omega, Scalar_field& solid, double N_x, double N_y, double N_z, double dx, double dt, double rho_0, double c_s2, const std::string& filename, int Zones, Parallel_MPI* MPI_parallel);
void KIDA3D(Vector_field&, Scalar_field&, Vector_field&, Scalar_field&, Solid_field&, double, double, double, double, double, double, double, const std::string&, int, Parallel_MPI* MPI_parallel);
void TAYLORGREEN3D(Vector_field&, Scalar_field&, Vector_field&, Scalar_field&, Solid_field&, double, double, double, double, double, double, double, const std::string&, int, Parallel_MPI* MPI_parallel);
void TAYLORGREEN2D(Vector_field&, Scalar_field&, Vector_field&, Scalar_field&, Solid_field&, double, double, double, double, double, double, double, const std::string&, int, Parallel_MPI* MPI_parallel);
void CONVECTEDVORTEX(Vector_field&, Scalar_field&, Vector_field&, Scalar_field&, Solid_field&, double, double, double, double, double, double, double, const std::string&, int, Parallel_MPI* MPI_parallel);
void PERIODICSHEARLAYER(Vector_field&, Scalar_field&, Vector_field&, Scalar_field&, Solid_field&, double, double, double, double, double, double, double, const std::string&, int, Parallel_MPI* MPI_parallel);
void ACOUSTICWAVES(Vector_field&, Scalar_field&, Vector_field&, Scalar_field&, Solid_field&, double, double, double, double, double, double, double, const std::string&, int, Parallel_MPI* MPI_parallel);
void SHEARWAVEDECAY(Vector_field&, Scalar_field&, Vector_field&, Scalar_field&, Solid_field&, double, double, double, double, double, double, double, const std::string&, int, Parallel_MPI* MPI_parallel);
void USERDEFINED(Vector_field&, Scalar_field&, Vector_field&, Scalar_field&, Solid_field&, double, double, double, double, double, double, double, const std::string&, int, Parallel_MPI* MPI_parallel);
void USERDEFINEDFLUCT(Vector_field&, Scalar_field&, Vector_field&, Scalar_field&, Solid_field&, double, double, double, double, double, double, double, const std::string&, int, Parallel_MPI* MPI_parallel);

#endif
