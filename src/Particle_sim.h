#ifndef PARTICLE_SIM_H
#define PARTICLE_SIM_H

#include "ALBORZ_GlobalVariables.h"
#include "ALBORZ_SETTINGS.h"
#include "Parallel.h"

#include <fstream>   // file stream
#include <iostream>  // std::cout
#include <sstream>

#include <vector>
using std::vector;
using namespace std;

class Thermal_solver;
class Flow_solver;
class stl_import;
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
struct node_struct {
	/// Constructor

	node_struct() {
	}

	/// Elements

	double x;  // current x-position
	double y;  // current y-position
	double z;  // current y-position

	double vel_x;  // node velocity (x-component)
	double vel_y;  // node velocity (y-component)
	double vel_z;  // node velocity (y-component)

	double force_x;  // node force (x-component)
	double force_y;  // node force (y-component)
	double force_z;  // node force (y-component)

	double den;            // Density at boundary point
	double force_thermal;  // node force due to Thermal effect (it is not in Newton)
	double temperature;    // node force due to Thermal effect (it is not in Newton)
	double x_pos_1;        // Node x - location in BODY - fixed coordinate with respect to particle center.It does not change during time steps
	double y_pos_1;        // Node y - location in BODY - fixed coordinate with respect to particle center.It does not change during time steps
	double z_pos_1;        // Node z - location in BODY - fixed coordinate with respect to particle center.It does not change during time steps
};
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
struct center_struct {
	/// Elements
	double x;                 // current x-position
	double y;                 // current y-position
	double z;                 // current z-position
	double vel_x;             // center velocity (x-component)
	double vel_y;             // center velocity (y-component)
	double vel_z;             // center velocity (z-component)
	double den;               // Density at boundary point
	double omgX;              // center rotational speed
	double omgY;              // center rotational speed
	double omgZ;              // center rotational speed
	double tetaX;             // Angle x
	double tetaY;             // Angle y
	double tetaZ;             // Angle z
	double delta_teta_X;      // Particle change in rotation angle
	double delta_teta_Y;      // Particle change in rotation angle
	double delta_teta_Z;      // Particle change in rotation angle
	double temperature;       // node force due to Thermal effect (it is not in Newton)
	double temp_temperat[2];  // Temperature of particle at two previous time steps
	double force_thermal;     // center force due to Thermal effect (it is not in Newton)
	double previous_vel_x;    // Velocity x in previous time step
	double previous_vel_y;    // Velocity y in previous time step
	double previous_vel_z;    // Velocity y in previous time step
	double deter;             // determinant of transformations matrix for spheroid
	double quater[10];
	double tempx[2];                               // Translational speed of particle at two previous time steps
	double tempy[2];                               // Translational speed of particle at two previous time steps
	double tempz[2];                               // Translational speed of particle at two previous time steps
	double temp_omgX[2];                           // Rotational speed of particle at two previous time steps
	double temp_omgY[2];                           // Rotational speed of particle at two previous time steps
	double temp_omgZ[2];                           // Rotational speed of particle at two previous time steps	double temp_temperat[2]; // Temperature of particle at two previous time steps
	double Omg_prime_X, Omg_prime_Y, Omg_prime_Z;  // ELLIPSOID only: Rotational velocity in body-fixed coordinate
	double Qu0, Qu1, Qu2, Qu3;                     // Quaternions
	double temp_part;                              // At first iteration it is Initial particle temperature and then take the value of particle temperature at pervious iteration
	double fx_surf, fy_surf, fz_surf;              // Total force on particle center
	double t1_x;                                   // Torque on particle center from fluid
	double t1_y;                                   // Torque on particle center from fluid
	double t1_z;                                   // Torque on particle center from fluid
	double t2_x;                                   // Torque on particle center from fluid
	double t2_y;                                   // Torque on particle center from fluid
	double t2_z;                                   // Torque on particle center from fluid
	double t3_x;                                   // Torque on particle center due to collision with other particles or walls--> Just for Ellipsoid
	double t3_y;                                   // Torque on particle center due to collision with other particles or walls--> Just for Ellipsoid
	double t3_z;                                   // Torque on particle center due to collision with other particles or walls--> Just for Ellipsoid
	double f3x, f3y, f3z;                          // Total particle-particle Collision force on particle center
	double f4x, f4y, f4z;                          // Total particle-wall Collision force on particle center
	double displacement_verlet;                    // Displacement of particle sqrt((x-x0)^2+(y-y0)^2+(z-z0)^2) between two Verlet list update
	double heat_surf;
};

/// Structure for object (either cylinder or ellipse)
struct particle_struct {
	/// Constructor
	particle_struct() {
	}

	/// Elements
	int num_nodes;          // number of surface nodes
	center_struct* center;  // center node
	node_struct* node;      // list of nodes
	double radius;
	double radius_2;
};

/// ---------------------------------------------------------------------------------------------
/// ----------------------------- PARTICLE CLASS     --------------------------------------------

class Particle_sim {
private:
public:
	double center_y1, center_z1, radius1, ystar1, ustar1, Yc, Zc, PR;
	double xbegin;
	double ybegin;
	double zbegin;
	int num_particles;           // number of particles.
	int particle_num_nodes;      // number of surface nodes
	double particle_radius;      // radius of a circular cylinder OR the radius along the X-axis for a "ELLIPSE"
	double particle_center_x;    // center position (x-component) of first particle
	double particle_center_y;    // center position (y-component) of first particle
	double particle_center_z;    // center position (y-component) of first particle
	double particle_gravity_x;   // This force is applied to particles according to density ratio
	double particle_gravity_y;   // This force is applied to particles according to density ratio
	double particle_gravity_z;   // This force is applied to particles according to density ratio
	double den_ratio;            // Particle to fluid density ratio. Note that in below simulations denisty of fluid is assumed to be 1.0
	double specific_heat_ratio;  // Particle to fluid specific heat ratio.
	double particle_radius_2;    // Just used for "Spheroid"
	double init_ang;             // Initial angle of ELLIPSE with respect to X-dir in clock-wise direction (Jahate Mosalasati)
	double offset_x;             // Distance of particles from each other in x direction.
	double offset_y;             // Distance of particles from each other in y direction.
	double offset_z;             // Distance of particles from each other in y direction.
	double particle_temperature;
	double particle_heat_source;
	int num_strip;
	double cir_factor;
	double ini_v[3];

	////Particle collision parameters
	double r_buffer;  // Preferably = (r_cutoff+max(particle_radius,particle_radius_2))
	double r_cutoff;  // Preferably >= [2*max(particle_radius,particle_radius_2)+collision_threshold]
	double cij;
	double ep;       // Stiffness factor for particle-particle collision
	double Ep;       // Stiffness factor for particle-particle collision usually smaller than ep
	double epw;      // Stiffness factor for particle-wall collision
	double Epw;      // Stiffness factor for particle-wall collision usually smaller than epw
	double th;       // Threshold for particle-particle collision
	double thw;      // Threshold for particle-wall collision
	double th_lub;   // Lubrication force threshold (larger than th)
	double thw_lub;  // Wall lubrication force threshold (larger than thw)

	double* area = NULL;
	double center_old_x, center_old_y, center_old_z;
	particle_struct* particle = NULL;  // create immersed object
	int* All_OUT = NULL;
	int* Particle_IN = NULL;
	int* current_par_center = NULL;
	int* current_par_centerx = NULL;
	int* current_par_centery = NULL;
	int* current_par_centerz = NULL;
	std::vector<int> node_in;  // If a node is within a range devoted to a processor ==> node_in[i] shows the index of that node
	int* number_nodes_in = NULL;
	int** Node_IN = NULL;           // If a node is within a range devoted to a processor ==> Node_IN=TRUE
	int** current_node_loc = NULL;  // It tells us the processor at which node is located (before updating by Newton's eq)
	double particle_mass, particle_area;
	double mom_inertia_x, mom_inertia_y, mom_inertia_z;
	int* triangleset1 = NULL;  // Location of first triangle node. Just used for Particle_VTK to make a surface (Spheroid)
	int* triangleset2 = NULL;  // Location of first triangle node. Just used for Particle_VTK to make a surface
	int* triangleset3 = NULL;  // Location of first triangle node. Just used for Particle_VTK to make a surface
	int number_triangles;      // number of triangles created for vtk file of particle

	// Next 10 lines parameters are just for MOVING_SPHEROID to model its rotation
	double* T_prime_X;  // Torques in body-fixed coordinate
	double* T_prime_Y;
	double* T_prime_Z;
	double* euler_teta;  // Euler angles
	double* euler_fi;    // Euler angles
	double* euler_sai;   // Euler angles
	double sumq;

	int* proc_extend;
	int* proc_extend1;
	int* proc_extend2;

	vector<vector<int>> verlet_list;

public:
	Particle_sim();
	void General_data_input(Thermal_solver*, const std::string&, Parallel_MPI* MPI_parallel);
	void Particle_initialize(const std::string&, stl_import*, Parallel_MPI*, int);
	int find_processor_index(double,double,double, Parallel_MPI*, int*, int*, int*);
	void interpolate(particle_struct&, Thermal_solver*, Flow_solver*, const vector<int>&, int, int, int, int);
	void force_calc(particle_struct, int);
	void spread(particle_struct&, Thermal_solver*, Flow_solver*, vector<int>, int, int, int);
	void Data_Slave_to_Master_rcv(particle_struct particle[], int, Parallel_MPI*);
	void collision_sphere(particle_struct[], int, double);    // Collision forces for moving spheres
	void collision_spheroid(particle_struct[], int, double);  // Collision forces for moving Spheroidal objects
	void Tasks_Before_Newton(particle_struct particle[], Thermal_solver*, Flow_solver*, Parallel_MPI*, int);
	void Tasks_After_Newton(particle_struct particle[], int, Parallel_MPI*);
	void update_location(particle_struct particle[], int t, int[], Parallel_MPI*);
	void check_zero(double&, double);
	void rungekutta(double&, double&, double&, double, double, double, double, double, double, double&, double&, double&, double&);  // Runge-Kutta at each time step for the spheroid rotation
	void quaternion(double, double, double, double, int, double[]);                                                                  // Calculation of quaternion matrix elements. Eq2.13 Article 876 Huang 2012
	double determinant(int, double[]);
	void comp_rot_speed(double, double&, double&, double&, int, double[], double, double, double);                          // update rotational speed. Just for SPHEROID
	void update_rot_spheroid(double, double&, double&, double&, double[], double, double, double, double, double, double);  // update initial position of nodes if Euler_angles are not initially zero. Just for SPHEROID
	void normal_vec_spheroid(double, double&, double&, double&, int, double[], double, double, double);                     // Calculates NORMAL vector at each surface point (in INERTIAL coordinate) of spheroid . Used to calculate collision forces
	void write_particle_vtk(const std::string& base_path, int tm, particle_struct, int);                                    // write the particle state to the disk as VTK file
	void write_particle_data(const std::string& base_path, int tm, particle_struct, int);                                   // write particle data to the disk (drag/lift, center position)
	void write_particle_recovery(const std::string& base_path, int tm, particle_struct, int);
	void read_particle_recovery(const std::string& base_path, int tm, particle_struct[]);
	void write_particle_parameters(Parallel_MPI*);
	void update_verlet(particle_struct[], int);

	void register_recovery(class IO_interface& io_interface, Parallel_MPI& parallel_MPI);

	~Particle_sim();

private:
	static std::string make_recovery_name(const std::string& base_path, int tm, int npn);
};

#endif
