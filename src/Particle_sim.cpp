//#include "stdafx.h"
#define _USE_MATH_DEFINES
#include <cmath>
#include <fstream>  // file stream
#include <sstream>  // string streams
#include <cstdlib>  // standard library
#include <iomanip>  // For set precision. From ver 28
#include <iostream> // for the use of 'cout'
#include <cstring> // for memcpy
#include <vector>

#include "Particle_sim.h"
#include "ALBORZ_Macros.h"
#include "ALBORZ_SETTINGS.h"
#include "ALBORZ_GlobalVariables.h"
#include "Geometry.h"
#include "Flow_solver.h"
#include "Thermal_solver.h"
#include "io/IO_interface.h"
#include "utils/Config_utils.h"
/// #include <conio.h>
using namespace std;

//=============================================================================================================================
//=============================================================================================================================
//-----------------------------------------------------------------------------------------------------------------------------
Particle_sim::Particle_sim() {
}
//=============================================================================================================================
//=============================================================================================================================
//-----------------------------------------------------------------------------------------------------------------------------
// Subroutine: get_sim_data_particle
//-----------------------------------------------------------------------------------------------------------------------------
void Particle_sim::General_data_input(Thermal_solver* Thermal, const std::string& filename, Parallel_MPI* MPI_parallel) {
	string input_filename(filename);
	input_filename += ".dat";
	ifstream input_file;  // File is open for READING
	                      /// Open file
	input_file.open(input_filename.c_str(), ios::binary);
	string Line1;

	find_line_after_header(input_file, "c\tParticle data");
	find_line_after_comment(input_file);

	input_file >> num_particles >> particle_num_nodes >> particle_radius >> particle_center_x >> particle_center_y >> particle_center_z >> ini_v[0] >> ini_v[1] >> ini_v[2];
	ini_v[0] *= (global_parameters.D_t / global_parameters.D_x);
	ini_v[1] *= (global_parameters.D_t / global_parameters.D_x);
	ini_v[2] *= (global_parameters.D_t / global_parameters.D_x);
	// std::getline(input_file, Line1);

	find_line_after_comment(input_file);
	input_file >> particle_gravity_x >> particle_gravity_y >> particle_gravity_z >> den_ratio >> specific_heat_ratio >> particle_radius_2;
	particle_gravity_x *= (sqr(global_parameters.D_t) / global_parameters.D_x);
	particle_gravity_y *= (sqr(global_parameters.D_t) / global_parameters.D_x);
	particle_gravity_z *= (sqr(global_parameters.D_t) / global_parameters.D_x);

	find_line_after_comment(input_file);
	input_file >> offset_x >> offset_y >> offset_z;

	find_line_after_comment(input_file);
	input_file >> num_strip >> cir_factor;

	find_line_after_comment(input_file);
	input_file >> particle_temperature >> particle_heat_source;
	particle_temperature /= Thermal->T_0;

	find_line_after_header(input_file, "c\tParticle collision parameters");
	find_line_after_comment(input_file);
	input_file >> cij >> ep >> Ep >> epw >> Epw >> th >> thw >> th_lub >> thw_lub >> r_buffer >> r_cutoff;

	input_file.close();

	if (MPI_parallel->processor_id == MASTER + 1) {
		std::cout << "Particle field parameters" << endl;
		std::cout << "=====================" << endl;
		std::cout << "Number of particles : " << num_particles << endl;
		std::cout << "Particle radius : " << particle_radius << endl;
		std::cout << "Particle gravity : " << particle_gravity_x << "\t" << particle_gravity_y << "\t" << particle_gravity_z << endl;
		std::cout << "Particle temperature : " << particle_temperature << endl;
	}

	return;
}
//=============================================================================================================================
//=============================================================================================================================
//-----------------------------------------------------------------------------------------------------------------------------
// Subroutine: Particle_initialize()
//-----------------------------------------------------------------------------------------------------------------------------

void Particle_sim::Particle_initialize(const std::string& filename, stl_import* Geo_stl, Parallel_MPI* MPI_parallel, int tstart) {
#if defined MOVING_SPHERE || defined STATIONARY_SPHERE
	particle_num_nodes = int(M_PI / 3. * (3 * SQ(2 * particle_radius) + 1));  // Uhlmann (2005) recommendation for number of surface nodes for sphere																		   //int particle_num_nodes = 2500; // Manual adjustment of number of surface nodes for sphere
#endif
#if defined MOVING_SPHEROID || defined STATIONARY_SPHEROID
	particle_num_nodes = 6000;  // Important. This parameter is just for initial adjustment. The actual particle_num_nodes for MOVING_SPHEROID will be specified later.
#endif
	area = new double[particle_num_nodes];
	particle = new particle_struct[num_particles];

	for (int i = 0; i < num_particles; i++) {
		particle[i].node = new node_struct[particle_num_nodes];
		particle[i].center = new center_struct[1];
		particle[i].num_nodes = particle_num_nodes;  // For spheroid: This loop is with 6000 nodes and will change later
	}

	euler_teta = new double[num_particles];
	euler_fi = new double[num_particles];
	euler_sai = new double[num_particles];

	verlet_list.resize(num_particles);
	for (int i = 0; i < num_particles; ++i) {
		verlet_list[i].resize(0);  // 0 will change later
	}

	int par_limit = 0;  // maximum limit of particle in each side
	int par_limit1 = 0;  // maximum limit of particle in each side
	int par_limit2 = 0;  // maximum limit of particle in each side
#if defined MOVING_SPHERE
	par_limit = static_cast<int>(particle_radius / MPI_parallel->avg_nod_per_process[0]) + 1;
	par_limit1 = static_cast<int>(particle_radius / MPI_parallel->avg_nod_per_process[1]) + 1;
	par_limit2 = static_cast<int>(particle_radius / MPI_parallel->avg_nod_per_process[2]) + 1;																					   

#endif
#if defined MOVING_SPHEROID
	par_limit = static_cast<int>(MAX(particle_radius, particle_radius_2) / MPI_parallel->avg_nod_per_process[0]) + 1;
	par_limit1 = static_cast<int>(MAX(particle_radius, particle_radius_2) / MPI_parallel->avg_nod_per_process[1]) + 1;
	par_limit2 = static_cast<int>(MAX(particle_radius, particle_radius_2) / MPI_parallel->avg_nod_per_process[2]) + 1;																											  
#endif

	proc_extend = new int[MPI_parallel->num_processors];
	proc_extend1 = new int[MPI_parallel->num_processors];
	proc_extend2 = new int[MPI_parallel->num_processors];												  

	proc_extend[MPI_parallel->processor_id] = par_limit + 2;  // 2 means that send particle data to 2 other processor + par_limit on each side.
	proc_extend1[MPI_parallel->processor_id] = par_limit1 + 2;//2 means that send particle data to 2 other processor + par_limit on each side.
	proc_extend2[MPI_parallel->processor_id] = par_limit2 + 2;//2 means that send particle data to 2 other proces
	if (proc_extend[MPI_parallel->processor_id] > (MPI_parallel->Np_X )) {
		proc_extend[MPI_parallel->processor_id] = MPI_parallel->Np_X ;
	}
		if (proc_extend1[MPI_parallel->processor_id] > (MPI_parallel->Np_Y ))
	{
		proc_extend1[MPI_parallel->processor_id] = MPI_parallel->Np_Y ;
	}

		if (proc_extend2[MPI_parallel->processor_id] > (MPI_parallel->Np_Z ))
	{
		proc_extend2[MPI_parallel->processor_id] = MPI_parallel->Np_Z ;
	}

	if (tstart == 0) {
		/// ---------------------------------------------------------------------------------------------
		/// -------------------------- PARTICLE MASS AND MOMENT OF INERTIA ------------------------------
		/// ---------------------------------------------------------------------------------------------
#if defined MOVING_SPHERE
		double parvol = 4.0 / 3.0 * M_PI * particle_radius * particle_radius * particle_radius;  // Particle volume
		particle_mass = (den_ratio * 1.0) * parvol;                                              // Particle mass
		particle_area = M_PI * (12.0 * SQ(particle_radius) + 1.0) / 3.0;                         // Eq. 2.14 Lucci et. al. J. Fluid Mech. (2010), vol. 650, pp. 5�55.
		mom_inertia_x = 2.0 / 5.0 * particle_mass * SQ(particle_radius);                         // Moment of inertia
		mom_inertia_y = 2.0 / 5.0 * particle_mass * SQ(particle_radius);
		mom_inertia_z = 2.0 / 5.0 * particle_mass * SQ(particle_radius);
#endif

#if defined MOVING_SPHEROID || defined STATIONARY_SPHEROID
		double parvol = 4.0 / 3.0 * M_PI * particle_radius * particle_radius_2 * particle_radius_2;  // Particle volume
		particle_mass = (den_ratio * 1.0) * parvol;

		cc = particle_radius;
		aa = particle_radius_2;
		hh = sqrt(MAX((aa - cc), (cc - aa)) * (aa + cc) / pow(cc, 4));
		zz = particle_radius * hh;
		if (cc > aa) {
			particle_area = 2 * M_PI * aa * 2 * (asin(zz) / 2. + (zz * sqrt(1 - SQ(zz))) / 2.) / hh;
		} else {
			particle_area = 2 * M_PI * aa * 2 * (asinh(zz) / 2. + (zz * sqrt(1 + SQ(zz))) / 2.) / hh;
		}
		mom_inertia_x = 1.0 / 5.0 * particle_mass * (SQ(particle_radius_2) + SQ(particle_radius_2));
		mom_inertia_y = 1.0 / 5.0 * particle_mass * (SQ(particle_radius) + SQ(particle_radius_2));
		mom_inertia_z = 1.0 / 5.0 * particle_mass * (SQ(particle_radius) + SQ(particle_radius_2));
#endif
#if defined Corrected_Radius && defined Flow_With_Particle
		double corrected_radius_2 = pow((pow(particle_radius_2, 3) + pow(particle_radius_2 - 1.0, 3)) / 2.0, 1.0 / 3.0);

		double corrected_radius = pow((pow(particle_radius, 3) + pow(particle_radius - 1.0, 3)) / 2.0, 1.0 / 3.0);
		particle_radius = corrected_radius;      // If corrected-radius is defined, from now on particle-radius represents corrected radius
		particle_radius_2 = corrected_radius_2;  // If corrected-radius is defined, from now on particle-radius_2 represents corrected radius
#endif
	}
	///////// Particles initialize
	for (size_t i = 0; i < num_particles; i++) {
		particle[i].radius = particle_radius;
		particle[i].radius_2 = particle_radius_2;
		// particle[i].center[0].x = 0;//Will change during particle setup
		// particle[i].center[0].y = 0;
		// particle[i].center[0].z = 0;
		particle[i].center[0].vel_x = ini_v[0];
		particle[i].center[0].vel_y = ini_v[1];
		particle[i].center[0].vel_z = ini_v[2];
		particle[i].center[0].den = 0;
		particle[i].center[0].temperature = particle_temperature;
		particle[i].center[0].force_thermal = 0;
		particle[i].center[0].omgX = 0.0;
		particle[i].center[0].omgY = 0.0;
		particle[i].center[0].omgZ = 0.0;
		particle[i].center[0].previous_vel_x = 0.0;
		particle[i].center[0].previous_vel_y = 0.0;
		particle[i].center[0].previous_vel_z = 0.0;
		particle[i].center[0].tetaX = 0.0;
		particle[i].center[0].tetaY = 0.0;
		particle[i].center[0].tetaZ = 0.0;
		particle[i].center[0].tempx[0] = 0.0;
		particle[i].center[0].tempx[1] = 0.0;
		particle[i].center[0].tempy[0] = 0.0;
		particle[i].center[0].tempy[1] = 0.0;
		particle[i].center[0].tempz[0] = 0.0;
		particle[i].center[0].tempz[1] = 0.0;
		particle[i].center[0].temp_omgX[0] = 0.0;
		particle[i].center[0].temp_omgX[1] = 0.0;
		particle[i].center[0].temp_omgY[0] = 0.0;
		particle[i].center[0].temp_omgY[1] = 0.0;
		particle[i].center[0].temp_omgZ[0] = 0.0;
		particle[i].center[0].temp_omgZ[1] = 0.0;
		particle[i].center[0].temp_temperat[0] = particle_temperature;
		particle[i].center[0].temp_temperat[1] = particle_temperature;
		particle[i].center[0].temp_part = particle_temperature;
		particle[i].center[0].delta_teta_X = 0;
		particle[i].center[0].delta_teta_Y = 0;
		particle[i].center[0].delta_teta_Z = 0;
		particle[i].center[0].fx_surf = 0;
		particle[i].center[0].fy_surf = 0;
		particle[i].center[0].fz_surf = 0;
		particle[i].center[0].t1_x = 0;
		particle[i].center[0].t1_y = 0;
		particle[i].center[0].t1_z = 0;
		particle[i].center[0].t2_x = 0;
		particle[i].center[0].t2_y = 0;
		particle[i].center[0].t2_z = 0;
		particle[i].center[0].t3_x = 0;
		particle[i].center[0].t3_y = 0;
		particle[i].center[0].t3_z = 0;
		particle[i].center[0].displacement_verlet = 0;
		particle[i].center[0].heat_surf = 0;
		particle[i].center[0].f3x = 0;
		particle[i].center[0].f3y = 0;
		particle[i].center[0].f3z = 0;
		particle[i].center[0].f4x = 0;
		particle[i].center[0].f4y = 0;
		particle[i].center[0].f4z = 0;
		particle[i].center[0].Omg_prime_X = 0.0;  // Angular velocity in body-fixed coordinate. Just used for moving-spheroid. Must be ZERO initially
		particle[i].center[0].Omg_prime_Y = 0.0;  // Angular velocity in body-fixed coordinate. Just used for moving-spheroid. Must be ZERO initially
		particle[i].center[0].Omg_prime_Z = 0.0;  // Angular velocity in body-fixed coordinate. Just used for moving-spheroid. Must be ZERO initially
		euler_teta[i] = 0.0;                      // Euler angles (radian). You can adjust them. Important: when all Euler angles are zero, particle_radius is along X-axis and particle_radius_2 is along Y and Z-axis.
		euler_fi[i] = 0.0;                        // Euler angles (radian). You can adjust them. Important: when all Euler angles are zero, particle_radius is along X-axis and particle_radius_2 is along Y and Z-axis.
		euler_sai[i] = 0.0;                       // Euler angles (radian). You can adjust them. Important: when all Euler angles are zero, particle_radius is along X-axis and particle_radius_2 is along Y and Z-axis.
		particle[i].center[0].Qu0 = cos(0.5 * euler_teta[i]) * cos(0.5 * (euler_fi[i] + euler_sai[i]));
		particle[i].center[0].Qu1 = sin(0.5 * euler_teta[i]) * cos(0.5 * (euler_fi[i] - euler_sai[i]));
		particle[i].center[0].Qu2 = sin(0.5 * euler_teta[i]) * sin(0.5 * (euler_fi[i] - euler_sai[i]));
		particle[i].center[0].Qu3 = cos(0.5 * euler_teta[i]) * sin(0.5 * (euler_fi[i] + euler_sai[i]));
		for (size_t j = 0; j < particle_num_nodes; j++) {  // For spheroid: This loop is performed with particle_num_nodes = 6000
			particle[i].node[j].vel_x = 0;
			particle[i].node[j].vel_y = 0;
			particle[i].node[j].vel_z = 0;
			particle[i].node[j].force_x = 0;
			particle[i].node[j].force_y = 0;
			particle[i].node[j].force_z = 0;
			particle[i].node[j].den = 0;
			particle[i].node[j].temperature = 0;
			particle[i].node[j].force_thermal = 0;
		}
	}

	triangleset1 = new int[24000];  // Total number of triangles is particle_num_nodesx4. Therefore, it must be less than 24000
	triangleset2 = new int[24000];
	triangleset3 = new int[24000];
	xbegin = Geo_stl->x_center;
	ybegin = Geo_stl->y_center;
	zbegin = Geo_stl->z_center;

#if defined X_Pipe
	string input_filename(filename);
	input_filename += ".dat";
	ifstream input_file;  // File is open for READING
	input_file.open(input_filename.c_str(), ios::binary);
	string Line1;
	char comment_indicator = 'k';
	int beg_line;
	input_file.clear();
	input_file.seekg(0, ios::beg);
	do {
		std::getline(input_file, Line1);
	} while (Line1.find("c\tPipe Initialization") == string::npos);
	std::getline(input_file, Line1);
	do {
		beg_line = input_file.tellg();
		input_file >> comment_indicator;
		if (comment_indicator == '#') {
			std::getline(input_file, Line1);
		}
	} while (comment_indicator == '#');
	input_file.seekg(beg_line);
	/// READ PARAMETERS (in S.I. units)
	input_file >> center_y1 >> center_z1 >> radius1;
	/// PRINT OUT PARAMETERS (in LB units)
	if (MPI_parallel->processor_id == (MASTER + 1)) {
		std::cout << "particle Pipe Initial conditions \n";
		std::cout << "pipe Radius : " << radius1 / global_parameters.D_x << std::endl;
		std::cout << " pipe Center(y,z) : " << abs(ybegin - center_y1) / global_parameters.D_x + 1.0 << "\t" << abs(zbegin - center_z1) / global_parameters.D_x + 1.0 << std::endl;
	}

	Yc = abs(ybegin - center_y1) / global_parameters.D_x + 1;
	Zc = abs(zbegin - center_z1) / global_parameters.D_x + 1;
	PR = radius1 / global_parameters.D_x;
	input_file.close();
#endif

	// #if defined Recovery_OFF
	if (tstart == 0) {
		/// ********************************
		/// CREATING OBJECTS (1st PARTICLE)
		/// ********************************
		// Defining particles initial positions. (Based on Shape)
		particle[0].center[0].x = particle_center_x;
		particle[0].center[0].y = particle_center_y;
		particle[0].center[0].z = particle_center_z;

#if defined MOVING_SPHERE || defined STATIONARY_SPHERE
		double fik = 0.0;
		for (int n = 0; n < particle[0].num_nodes; ++n) {
			double ck, tetak;  // parameters for the uniform distribution of points on the sphere
			                                //  Distributing points on a sphere according to:
			                                // Eq. 2.10 , 2.11 Lucci et. al. J. Fluid Mech. (2010), vol. 650, pp. 5�55.
			ck = -1.0 + 2.0 * (double)n / (double)(particle_num_nodes - 1.0);
			tetak = acos(ck);
			if (n == 0 || n == particle_num_nodes - 1) {
				fik = 0.0;
			} else {
				fik += 3.6 / sqrt(particle_num_nodes) / sqrt(1.0 - SQ(ck));
			}
			particle[0].node[n].x = particle[0].center[0].x + particle_radius * cos(fik) * sin(tetak);
			particle[0].node[n].y = particle[0].center[0].y + particle_radius * sin(fik) * sin(tetak);
			particle[0].node[n].z = particle[0].center[0].z + particle_radius * cos(tetak);
			// Eq. 2.14 Lucci et. al. J. Fluid Mech. (2010), vol. 650, pp. 5�55.
			area[n] = M_PI * (12.0 * SQ(particle_radius) + 1.0) / 3.0 / (double)particle_num_nodes;  // area belonging to a node
		}
#endif

#if defined MOVING_SPHEROID || defined STATIONARY_SPHEROID
		// Distributing points on a spheroid.
		int kkk, mmm;
		double perimeter;
		double cc = particle_radius;
		double aa = particle_radius_2;
		perimeter = M_PI * (3.0 * (particle_radius + particle_radius_2) - sqrt((3.0 * particle_radius + particle_radius_2) * (particle_radius + 3.0 * particle_radius_2)));
		// num_strip = (int)(perimeter / 2.0 * particle_radius / particle_radius_2) + 2;//Total number of parallel strips (including top and bottom points)

		// double hei[num_strip];//height of each strip
		double* hei = 0;  // height of each strip
		hei = new double[num_strip];

		int** nodes_in_hei;  // It tells us at each strip height wich nodes are located (node number). Important: Just for VTK file
		nodes_in_hei = new int*[num_strip];
		for (int nps = 0; nps < num_strip; ++nps) {
			nodes_in_hei[nps] = new int[2000];  // Certainly the number of nodes on one strip is less than 2000
		}

		int* po_per_str = 0;  // Number of points per each strip
		po_per_str = new int[num_strip];

		// Important: In the following configuration, the particle_radius is initially along the X-axis and particle_radius_2 is initially along the Y and Z-axis. You can change it later by adjusting initial euler angles
		kkk = 0;
		particle[0].node[kkk].x = particle[0].center[0].x + particle_radius;
		particle[0].node[kkk].y = particle[0].center[0].y;
		particle[0].node[kkk].z = particle[0].center[0].z;
		kkk = kkk + 1;                   // kkk is the index for number of particles.
		hei[0] = particle[0].node[0].x;  // hei is the elevation of each strip
		po_per_str[0] = 1;
		nodes_in_hei[0][0] = 0;

		for (int hf = 1; hf < num_strip - 1; hf++) {
			int ghk = 0;
			tetak = (double)(hf)*M_PI / (double)(num_strip - 1);

			////////// Method 1
			po_per_str[hf] = int(cir_factor * num_strip * sin(tetak));
			//////// End of Method 1

			////////// Method 2
			// double dist_to_cent = particle_radius_2*sqrt(2)*sin(tetak);
			// po_per_str[hf] = abs(int(2 * M_PI*dist_to_cent*cir_factor));
			//////// End of Method 2

			if (po_per_str[hf] < 2) {
				po_per_str[hf] = 2;
			}
			fik = 0.0;
			for (int hd = 1; hd <= po_per_str[hf]; hd++) {
				fik = fik + 2. * M_PI / po_per_str[hf];
				particle[0].node[kkk].y = particle[0].center[0].y + particle_radius_2 * cos(fik) * sin(tetak);
				particle[0].node[kkk].z = particle[0].center[0].z + particle_radius_2 * sin(fik) * sin(tetak);
				particle[0].node[kkk].x = particle[0].center[0].x + particle_radius * cos(tetak);
				hei[hf] = particle[0].node[kkk].x;
				nodes_in_hei[hf][ghk] = kkk;
				ghk = ghk + 1;
				kkk = kkk + 1;
			}
		}
		particle[0].node[kkk].x = particle[0].center[0].x - particle_radius;  // Last point
		particle[0].node[kkk].y = particle[0].center[0].y;
		particle[0].node[kkk].z = particle[0].center[0].z;

		po_per_str[num_strip - 1] = 1;
		nodes_in_hei[num_strip - 1][0] = kkk;          // Index of point that is located on this strip (last strip)
		hei[num_strip - 1] = particle[0].node[kkk].x;  // hei is the elevation of each strip
		particle_num_nodes = kkk + 1;                  // The actual particle_num_nodes for spheroid
		particle[0].num_nodes = particle_num_nodes;    // Important
		                                               // double area[particle_num_nodes];//If you activate this line you would have wrong results (Just remember not to define a parameter twice)

		mmm = 0;
		double sum, h1, h2, a1, a2;
		sum = 0.0;  // Total area. It is to make a comparison with a Full Spheroid formula
		for (int hf = 0; hf < num_strip - 1; hf++) {
			///////	tetak = (double)(hf)* M_PI / (double)(num_strip - 1);
			for (int hd = 1; hd <= po_per_str[hf]; hd++) {
				if (hf != 0) {
					h1 = (hei[hf] - particle[0].center[0].x + hei[hf - 1] - particle[0].center[0].x) / 2. * hh;
				} else {
					h1 = cc * hh;  //(hei[hf]-particle[0].center[0].x ) * hh;
				}
				h2 = (hei[hf] - particle[0].center[0].x + hei[hf + 1] - particle[0].center[0].x) / 2. * hh;
				if (cc > aa) {
					a1 = M_PI * aa * 2 * (asin(h1) / 2. + (h1 * sqrt(1 - SQ(h1))) / 2.) / hh / po_per_str[hf];
					a2 = M_PI * aa * 2 * (asin(h2) / 2. + (h2 * sqrt(1 - SQ(h2))) / 2.) / hh / po_per_str[hf];
				} else {
					a1 = M_PI * aa * 2 * (asinh(h1) / 2. + (h1 * sqrt(1 + SQ(h1))) / 2.) / hh / po_per_str[hf];
					a2 = M_PI * aa * 2 * (asinh(h2) / 2. + (h2 * sqrt(1 + SQ(h2))) / 2.) / hh / po_per_str[hf];
				}
				area[mmm] = abs(a1 - a2);
				// if (MPI_parallel->processor_id==1)
				//{
				//	cout << mmm << " " << area[mmm] << "  ";
				// }
				sum = sum + area[mmm];
				mmm = mmm + 1;
			}
		}
		area[mmm] = area[0];  // The last point
		sum = sum + area[mmm];
		// cout<<"total area: "<<sum<<endl;

		/////////////////////////////// Below part is just for creating surface triangles for Particle_VTK
		for (int i = 0; i < po_per_str[1]; i++) {
			triangleset1[i] = 0;
			triangleset2[i] = nodes_in_hei[1][i];
			triangleset3[i] = nodes_in_hei[1][(i + 1) % po_per_str[1]];
		}
		// For Particle_vtk we want to create a surface by triangles.
		kkk = 1;
		int tindex = po_per_str[1];
		double dist_node;
		double min_dist;
		int min_node, ipp;
		double max_dist = -2;
		int ik;

		for (int hf = 1; hf < num_strip - 1; hf++) {
			for (int ij = 0; ij < po_per_str[hf]; ij++) {
				ik = hf - 1;
				min_dist = 2 * MAX(particle_radius, particle_radius_2);
				for (int ip = 0; ip < po_per_str[ik]; ip++) {
					dist_node = sqrt(SQ(particle[0].node[nodes_in_hei[hf][ij]].x - particle[0].node[nodes_in_hei[ik][ip]].x) + SQ(particle[0].node[nodes_in_hei[hf][ij]].y - particle[0].node[nodes_in_hei[ik][ip]].y) + SQ(particle[0].node[nodes_in_hei[hf][ij]].z - particle[0].node[nodes_in_hei[ik][ip]].z));
					if (ip == 0) {
						ipp = 0;
					}

					if (dist_node < min_dist) {
						min_node = nodes_in_hei[ik][ip];
						ipp = ip;
						min_dist = dist_node;
					}
				}

				triangleset2[tindex] = nodes_in_hei[ik][ipp];
				triangleset1[tindex] = kkk;
				triangleset3[tindex] = nodes_in_hei[ik][(ipp + 1) % po_per_str[ik]];
				// cout << tindex << " " << triangleset1[tindex] << " " << triangleset2[tindex] << " " << triangleset3[tindex] << endl;
				tindex = tindex + 1;
				triangleset2[tindex] = nodes_in_hei[ik][ipp];
				triangleset1[tindex] = kkk;
				triangleset3[tindex] = nodes_in_hei[ik][(ipp - 1 + po_per_str[ik]) % po_per_str[ik]];
				// cout << tindex << " " << triangleset1[tindex] << " " << triangleset2[tindex] << " " << triangleset3[tindex] << endl;
				tindex = tindex + 1;

				ik = hf + 1;
				min_dist = 2 * MAX(particle_radius, particle_radius_2);
				for (int ip = 0; ip < po_per_str[ik]; ip++) {
					dist_node = sqrt(SQ(particle[0].node[nodes_in_hei[hf][ij]].x - particle[0].node[nodes_in_hei[ik][ip]].x) + SQ(particle[0].node[nodes_in_hei[hf][ij]].y - particle[0].node[nodes_in_hei[ik][ip]].y) + SQ(particle[0].node[nodes_in_hei[hf][ij]].z - particle[0].node[nodes_in_hei[ik][ip]].z));
					if (ip == 0) {
						ipp = 0;
					}

					if (dist_node < min_dist) {
						min_node = nodes_in_hei[ik][ip];
						ipp = ip;
						min_dist = dist_node;
					}
				}

				triangleset2[tindex] = nodes_in_hei[ik][ipp];
				triangleset1[tindex] = kkk;
				triangleset3[tindex] = nodes_in_hei[ik][(ipp + 1) % po_per_str[ik]];
				// cout << tindex << " " << triangleset1[tindex] << " " << triangleset2[tindex] << " " << triangleset3[tindex] << endl;
				tindex = tindex + 1;
				triangleset2[tindex] = nodes_in_hei[ik][ipp];
				triangleset1[tindex] = kkk;
				triangleset3[tindex] = nodes_in_hei[ik][(ipp - 1 + po_per_str[ik]) % po_per_str[ik]];
				// cout << tindex << " " << triangleset1[tindex] << " " << triangleset2[tindex] << " " << triangleset3[tindex] << endl;
				tindex = tindex + 1;
				kkk = kkk + 1;
			}
		}
		number_triangles = tindex;

		for (int i = 0; i < po_per_str[num_strip - 2]; i++) {
			triangleset1[tindex + i] = particle_num_nodes - 1;
			triangleset2[tindex + i] = nodes_in_hei[num_strip - 2][i];
			triangleset3[tindex + i] = nodes_in_hei[num_strip - 2][(i + 1) % po_per_str[1]];
			number_triangles++;
		}
		//////////////////////////////////End of Particle_VTK part
		// if (MPI_parallel->processor_id == 1)
		//{
		//	getch();
		// }
		delete hei;
		delete po_per_str;
#endif  // spheroid

		for (int n = 0; n < particle[0].num_nodes; ++n) {
			if (particle[0].node[n].x > global_parameters.Nx - 0.001 || particle[0].node[n].x < 0.001) {
				printf("ERROR: Particle %d position is out of the domain \n", 1);
				return;
			}
			if (particle[0].node[n].y > global_parameters.Ny - 0.001 || particle[0].node[n].y < 0.001) {
				printf("ERROR: Particle %d position is out of the domain \n", 1);
				return;
			}
			if (global_parameters.Nz > 1) {
				if (particle[0].node[n].z > global_parameters.Nz - 0.001 || particle[0].node[n].z < 0.001) {
					printf("ERROR: Particle %d position is out of the domain \n", 1);
					return;
				}
			}
		}
		/// ***********************************
		/// LOCATION OF 2nd UNTIL Nth PARTICLE
		/// ***********************************
#if defined Random_Arrangement_OFF
		for (int i = 1; i < num_particles; ++i) {
			particle[i].num_nodes = particle[0].num_nodes;               // Important for spheroid
			particle[i].center[0].x = particle_center_x + i * offset_x;  // Location of other particles is adjusted according to location of first one. You can change it.
			particle[i].center[0].y = particle_center_y + i * offset_y;
			particle[i].center[0].z = particle_center_z + i * offset_z;

			// int per_x = 26;//number of particles to be put in x direction
			// int per_y = 5;
			// int per_z = 4;

			// particle[i].center[0].x = (particle_center_x + (i % per_x) * offset_x);//Location of other particles is adjusted according to location of first one. You can change it.
			// particle[i].center[0].y = (particle_center_y + (((int)(i / per_x) % per_y) + 1) * offset_y);
			// particle[i].center[0].z = (particle_center_z + (((int)(i / (per_x*per_y)) % per_z) + 1) * offset_z);
			for (int n = 0; n < particle[i].num_nodes; ++n) {
				particle[i].node[n].x = particle[0].node[n].x - particle[0].center[0].x + particle[i].center[0].x;
				particle[i].node[n].y = particle[0].node[n].y - particle[0].center[0].y + particle[i].center[0].y;
				particle[i].node[n].z = particle[0].node[n].z - particle[0].center[0].z + particle[i].center[0].z;

				if (particle[i].node[n].x > global_parameters.Nx - 1 - 0.1 || particle[i].node[n].x < 0.1) {
					cout << "ERROR: Particle " << i + 1 << " position is out of the domain";
					return;
				}
				if (particle[i].node[n].y > global_parameters.Ny - 1 - 0.1 || particle[i].node[n].y < 0.1) {
					cout << "ERROR: Particle " << i + 1 << " position is out of the domain";
					return;
				}
				if (global_parameters.Nz > 1) {
					if (particle[i].node[n].z > global_parameters.Nz - 1 - 0.1 || particle[i].node[n].z < 0.1) {
						cout << "ERROR: Particle " << i + 1 << " position is out of the domain";
						return;
					}
				}
			}
		}
#endif

#if defined Random_Arrangement_ON
		//	double Ycc=84.0;
		//	double Zcc=84.0;
		//	double PRc=80.0;
		for (int i = 1; i < num_particles; ++i) {
			particle[i].num_nodes = particle[0].num_nodes;  // Important for spheroid
			int part_position;
			srand(5324);
			double max_dist;
#if defined MOVING_SPHERE || defined MOVING_CYLINDER
			max_dist = particle[0].radius;
#endif
#if defined MOVING_SPHEROID
			max_dist = MAX(particle[0].radius_2, particle[0].radius);
			;
#endif
			do {
				part_position = TRUE;
				particle[i].center[0].x = rand() % global_parameters.Nx;
				particle[i].center[0].y = rand() % global_parameters.Ny;
				particle[i].center[0].z = rand() % global_parameters.Nz;
				if (global_parameters.Nz == 1) {
					particle[i].center[0].z = 0;
				}
				if (MPI_parallel->processor_id == (MASTER + 1))
					printf("particle i is %d and x, y, z are %5.3f , %5.3f, %5.3f \n", i + 1, particle[i].center[0].x, particle[i].center[0].y, particle[i].center[0].z);

				for (int j = 0; j < num_particles; ++j) {
					if (j != i) {
						double Dij = sqrt(SQ(particle[i].center[0].x - particle[j].center[0].x) + SQ(particle[i].center[0].y - particle[j].center[0].y) + SQ(particle[i].center[0].z - particle[j].center[0].z));
						if (Dij < 2 * max_dist + 8) {
							part_position = FALSE;
							if (MPI_parallel->processor_id == (MASTER + 1))
								printf("Error, contact with particle %d \n", j);
							//	break;
						}
					}
				}

				if (particle[i].center[0].x > global_parameters.Nx - 1 - max_dist - 1 || particle[i].center[0].x < max_dist + 1) {
					part_position = FALSE;
					if (MPI_parallel->processor_id == (MASTER + 1))
						cout << "Error 1" << endl;
				}
				if (particle[i].center[0].y > global_parameters.Ny - 1 - max_dist - 1 || particle[i].center[0].y < max_dist + 1) {
					part_position = FALSE;
					if (MPI_parallel->processor_id == (MASTER + 1))
						cout << "Error 2" << endl;
				}
				if (global_parameters.Nz > 1) {
					if (particle[i].center[0].z > global_parameters.Nz - 1 - max_dist - 1 || particle[i].center[0].z < max_dist + 1) {
						part_position = FALSE;
						if (MPI_parallel->processor_id == (MASTER + 1))
							cout << "Error 3" << endl;
					}
				}
				if (sqrt((particle[i].center[0].z - Zc) * (particle[i].center[0].z - Zc) + (particle[i].center[0].y - Yc) * (particle[i].center[0].y - Yc)) > PR - 5 - max_dist) {
					part_position = FALSE;
					if (MPI_parallel->processor_id == (MASTER + 1))
						cout << "Error 4" << endl;
				}

			} while (part_position == FALSE);

			for (int n = 0; n < particle_num_nodes; ++n) {
				particle[i].node[n].x = particle[0].node[n].x - particle[0].center[0].x + particle[i].center[0].x;
				particle[i].node[n].y = particle[0].node[n].y - particle[0].center[0].y + particle[i].center[0].y;
				particle[i].node[n].z = particle[0].node[n].z - particle[0].center[0].z + particle[i].center[0].z;
			}
		}
#endif

		/// *****************************
		/// INITIAL ROTATION OF SPHEROID
		/// *****************************
#if defined MOVING_SPHEROID || defined STATIONARY_SPHEROID
		for (int np = 0; np < num_particles; ++np) {
			//// In the next 9 lines we update each Lagragian node position if the initial values of Euler angles are not zero. (It means particle rotation)
			quaternion(particle[np].center[0].Qu0, particle[np].center[0].Qu1, particle[np].center[0].Qu2, particle[np].center[0].Qu3, np, particle[np].center[0].quater);
			particle[np].center[0].deter = determinant(np, particle[np].center[0].quater);  // determinant of Quaternion matrix
			for (int n = 0; n < particle_num_nodes; ++n) {
				particle[np].node[n].x_pos_1 = particle[np].node[n].x - particle[np].center[0].x;
				particle[np].node[n].y_pos_1 = particle[np].node[n].y - particle[np].center[0].y;
				particle[np].node[n].z_pos_1 = particle[np].node[n].z - particle[np].center[0].z;
				update_rot_spheroid(particle[np].center[0].deter, particle[np].node[n].x, particle[np].node[n].y, particle[np].node[n].z, particle[np].center[0].quater, particle[np].node[n].x_pos_1, particle[np].node[n].y_pos_1, particle[np].node[n].z_pos_1, particle[np].center[0].x, particle[np].center[0].y, particle[np].center[0].z);  // Node location at INERTIAL coordinate is calculated based on the Quaternions
			}
		}
#endif
		//***************************************
	}  // End for if (tstart==0)

	All_OUT = new int[num_particles];             // All_OUT=TRUE means that particle has no node in a domain of that processor (including delta function range, e.g. 2)
	Particle_IN = new int[num_particles];         // If CENTER of a particle is located within a range devoted to a processor ==> Particle_IN=TRUE
	current_par_center = new int[num_particles];  // Tag of current particle center holder
	node_in.resize(0);
	number_nodes_in = new int[num_particles];  // It tells us how many nodes of each particle is located within the domain of the processor (before updating by Newton's eq)
	Node_IN = new int*[num_particles];
	for (int nps = 0; nps < num_particles; ++nps) {
		Node_IN[nps] = new int[particle_num_nodes];
	}
	current_node_loc = new int*[num_particles];
	current_par_centerx = new int[num_particles];
	current_par_centery = new int[num_particles];
	current_par_centerz = new int[num_particles];
	for (int nps = 0; nps < num_particles; ++nps) {
		current_node_loc[nps] = new int[particle_num_nodes];
	}

	T_prime_X = new double[num_particles];
	T_prime_Y = new double[num_particles];
	T_prime_Z = new double[num_particles];

	for (int np = 0; np < num_particles; ++np) {
		update_verlet(particle, np);
	}

	return;
}
//=============================================================================================================================
//=============================================================================================================================
//-----------------------------------------------------------------------------------------------------------------------------
// Subroutine: force_calc
//-----------------------------------------------------------------------------------------------------------------------------
// void Particle_sim::force_calc(particle_struct particle, int npn){
void Particle_sim::force_calc(particle_struct particle, int npn) {
	double x_dist, y_dist, z_dist;

	for (int n = 0; n < particle.num_nodes; ++n) {  // Compute Lagrangian Forces
		x_dist = (particle.node[n].x - particle.center[0].x);
		y_dist = (particle.node[n].y - particle.center[0].y);
		z_dist = (particle.node[n].z - particle.center[0].z);

		particle.node[n].force_x = (1.0) * (particle.center[0].previous_vel_x - particle.center[0].omgZ * y_dist + particle.center[0].omgY * z_dist - particle.node[n].vel_x) * area[n];  // Surface force
		particle.node[n].force_y = (1.0) * (particle.center[0].previous_vel_y + particle.center[0].omgZ * x_dist - particle.center[0].omgX * z_dist - particle.node[n].vel_y) * area[n];  // Surface force
		particle.node[n].force_z = (1.0) * (particle.center[0].previous_vel_z + particle.center[0].omgX * y_dist - particle.center[0].omgY * x_dist - particle.node[n].vel_z) * area[n];  // Surface force

		particle.node[n].force_thermal = 1 * (particle.center[0].temp_part - particle.node[n].temperature) * area[n];  // Surface force
	}
	return;
}
//=============================================================================================================================
//-----------------------------------------------------------------------------------------------------------------------------
// Subroutine: Data_Slave_to_Master_rcv
//-----------------------------------------------------------------------------------------------------------------------------
void Particle_sim::Data_Slave_to_Master_rcv(particle_struct particle[], int t, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id == MASTER) {
		int xc[1],yc[1],zc[1];
		for (int npk = 0; npk < num_particles; npk++) {  // Receiving particle data from sub-processor that holds the center
			size_t offset = sizeof(center_struct) / sizeof(double);
			static double* buf_from_cent1 = new double[offset];
			current_par_center[npk] = find_processor_index(particle[npk].center[0].x,particle[npk].center[0].y,particle[npk].center[0].z, MPI_parallel,xc,yc,zc);
			current_par_centerx[npk]=xc[0];
			current_par_centery[npk]=yc[0];
			current_par_centerz[npk]=zc[0];
			//if(t>=1){
			MPI_Recv(buf_from_cent1, offset, MPI_DOUBLE, current_par_center[npk], FROM_WORKER, MPI_COMM_WORLD, &status);
			memcpy(&(particle[npk].center[0]), buf_from_cent1, sizeof(center_struct));
		//	}
		}
	}
	return;
}
//=============================================================================================================================
//=============================================================================================================================
//-----------------------------------------------------------------------------------------------------------------------------
// Subroutine: FIND_PROCESSOR_INDEX
//-----------------------------------------------------------------------------------------------------------------------------
// This function finds relevant processor at which particle center_x is located.

int Particle_sim::find_processor_index(double particle_x_loc,double particle_y_loc,double particle_z_loc, Parallel_MPI* MPI_parallel,int* xc,int* yc,int* zc) {
	int x;
	x = floor(particle_x_loc / MPI_parallel->avg_nod_per_process[0]) + 1;
	if (x > MPI_parallel->Np_X) {
		x = MPI_parallel->Np_X;//This happen when length of last processor is larger than avg_rows_per_process
	}
#if defined X_Periodic
	if (particle_x_loc >= global_parameters.Nx) {
		x = floor((particle_x_loc - global_parameters.Nx) / MPI_parallel->avg_nod_per_process[0]) + 1;
	}

	if (particle_x_loc < 0) {
		x = floor((particle_x_loc + global_parameters.Nx) / MPI_parallel->avg_nod_per_process[0]) + 1;
		if (x > MPI_parallel->Np_X) {
			x = MPI_parallel->Np_X;//This happen when length of last processor is larger than avg_rows_per_process
		}
	}

#endif
//cout<<MPI_parallel->num_processors<<MPI_parallel->Np_X<<MPI_parallel->Np_Y<<MPI_parallel->Np_Z<<endl;
	int y;
	y = floor(particle_y_loc / MPI_parallel->avg_nod_per_process[1]) + 1;
	if (y > MPI_parallel->Np_Y ) {
		y = MPI_parallel->Np_Y ;//This happen when length of last processor is larger than avg_rows_per_process
	}
#if defined Y_Periodic
	if (particle_y_loc >= global_parameters.Ny) {
		y = floor((particle_y_loc - global_parameters.Ny) / MPI_parallel->avg_nod_per_process[1]) + 1;
	}

	if (particle_y_loc < 0) {
		y= floor((particle_y_loc + global_parameters.Ny) / MPI_parallel->avg_nod_per_process[1]) + 1;
		if (y > MPI_parallel->Np_Y ) {
			y = MPI_parallel->Np_Y ;//This happen when length of last processor is larger than avg_rows_per_process
		}
	}

#endif

	int z;
	z = floor(particle_z_loc / MPI_parallel->avg_nod_per_process[2]) + 1;
	if (z > MPI_parallel->Np_Z ) {
		z = MPI_parallel->Np_Z ;//This happen when length of last processor is larger than avg_rows_per_process
	}
#if defined Z_Periodic
	if (particle_z_loc >= global_parameters.Nz) {
		z = floor((particle_z_loc - global_parameters.Nz) / MPI_parallel->avg_nod_per_process[2]) + 1;
	}

	if (particle_z_loc < 0) {
		z= floor((particle_z_loc + global_parameters.Nz) / MPI_parallel->avg_nod_per_process[2]) + 1;
		if (z > MPI_parallel->Np_Z - 1) {
			z = MPI_parallel->Np_Z- 1;//This happen when length of last processor is larger than avg_rows_per_process
		}
	}

#endif
xc[0]=x;
yc[0]=y;
zc[0]=z;
	return MPI_parallel->proc_arrangement[x-1][y-1][z-1];
}

//=============================================================================================================================
//=============================================================================================================================
//-----------------------------------------------------------------------------------------------------------------------------
// Subroutine: Tasks_Before_Newton
//-----------------------------------------------------------------------------------------------------------------------------
void Particle_sim::Tasks_Before_Newton(particle_struct particle[], Thermal_solver* Thermal, Flow_solver* Flow, Parallel_MPI* MPI_parallel, int time) {
	if (MPI_parallel->processor_id != MASTER) {
		double x_dist, y_dist, z_dist;
		double disp_max_1 = 0;  // Max Displacement of particles
		double disp_max_2 = 0;  // Second max of Displacement of particles

		/// ---------------------------------------------------------------------------------------------
		/// ----------------------------- EULERIAN FORCE INITIALIZATION ---------------------------------
		/// ---------------------------------------------------------------------------------------------
		/// Important: Forces at Eulerian points must be set to ZERO at each iteration before SPREAD. We
		/// do this at the end of MOMENTA subroutine.
		/// This is done so just to reduce computational cost & time. (We can set all forces Zero here
		/// but it is time consuming)
		/// ---------------------------------------------------------------------------------------------
		/// ---------------------------------------------------------------------------------------------
for (int X = 0; X < range; X++) {  // Important lines
			for (int Y = 0; Y < MPI_parallel->dev_end[1]; ++Y) {
				for (int Z = 0; Z <MPI_parallel->dev_end[2]; ++Z) {
					for (int mm = 0; mm < 3; mm++) {
						Flow->force[{X, Y, Z, mm}] = 0.;
						Flow->force[{MPI_parallel->end_XYZ2[0] + 1 + X, Y, Z, mm}] = 0.;
						Flow->temp_force[{X, Y, Z, mm}] = 0.;
						Flow->temp_force[{MPI_parallel->end_XYZ2[0] + 1 + X, Y, Z, mm}] = 0.;
				#if defined Flow_With_Thermal_Effect
					Thermal->force_thermal[{X, Y, Z}] = 0.;
					Thermal->force_thermal[{ MPI_parallel->end_XYZ2[0] + 1 +X, Y, Z}] = 0.;
					Thermal->temp_force_thermal[{X, Y, Z}] = 0.;
					Thermal->temp_force_thermal[{ MPI_parallel->end_XYZ2[0] + 1 +X, Y, Z}] = 0.;
				#endif
						
					}
				}
			}
		}

		for (int X = 0; X < MPI_parallel->dev_end[0] ; X++) {  // Important lines
			for (int Y = 0; Y < range; ++Y) {
				for (int Z = 0; Z <MPI_parallel->dev_end[2]; ++Z) {
					for (int mm = 0; mm < 3; mm++) {
						Flow->force[{X, Y, Z, mm}] = 0.;
						Flow->force[{ X, MPI_parallel->end_XYZ2[1] + 1 +Y, Z, mm}] = 0.;
						Flow->temp_force[{X, Y, Z, mm}] = 0.;
						Flow->temp_force[{ X, MPI_parallel->end_XYZ2[1] + 1 +Y, Z, mm}] = 0.;
				#if defined Flow_With_Thermal_Effect
					Thermal->force_thermal[{X, Y, Z}] = 0.;
					Thermal->force_thermal[{ X, MPI_parallel->end_XYZ2[1] + 1 +Y, Z}] = 0.;
					Thermal->temp_force_thermal[{X, Y, Z}] = 0.;
					Thermal->temp_force_thermal[{ X, MPI_parallel->end_XYZ2[1] + 1 +Y, Z}] = 0.;
				#endif
					}
				}
			}
		}
		for (int X = 0; X <MPI_parallel->dev_end[0] ; X++) {  // Important lines
			for (int Y = 0; Y < MPI_parallel->dev_end[1]; ++Y) {
				for (int Z = 0; Z <range; ++Z) {
					for (int mm = 0; mm < 3; mm++) {
						Flow->force[{X, Y, Z, mm}] = 0.;
						Flow->force[{ X, Y, MPI_parallel->end_XYZ2[2] + 1 +Z, mm}] = 0.;
						Flow->temp_force[{X, Y, Z, mm}] = 0.;
						Flow->temp_force[{ X, Y, MPI_parallel->end_XYZ2[2] + 1 +Z, mm}] = 0.;
			#if defined Flow_With_Thermal_Effect
					Thermal->force_thermal[{X, Y, Z}] = 0.;
					Thermal->force_thermal[{ X, Y, MPI_parallel->end_XYZ2[2] + 1 +Z}] = 0.;
					Thermal->temp_force_thermal[{X, Y, Z}] = 0.;
					Thermal->temp_force_thermal[{ X, Y,MPI_parallel->end_XYZ2[2] + 1 + Z}] = 0.;
			#endif
					}
						}
			}
		}


	


		//	//			/// ---------------------------------------------------------------------------------------------
		//	//			/// --------------------------- VERLET LIST UPDATE  -----------------------------
		//	//			/// ---------------------------------------------------------------------------------------------
		//	//
		int check_verlet = FALSE;

		for (int npp = 0; npp < num_particles; ++npp) {
			if (particle[npp].center[0].displacement_verlet > disp_max_1) {
				disp_max_2 = disp_max_1;
				disp_max_1 = particle[npp].center[0].displacement_verlet;
			}
			if ((particle[npp].center[0].displacement_verlet > disp_max_2) && (particle[npp].center[0].displacement_verlet < disp_max_1)) {
				disp_max_2 = particle[npp].center[0].displacement_verlet;
			}
#if defined X_Periodic
			if ((particle[npp].center[0].x - particle[npp].center[0].displacement_verlet) < 0) {
				check_verlet = TRUE;
			}
			if ((particle[npp].center[0].x + particle[npp].center[0].displacement_verlet) > (global_parameters.Nx - 1)) {
				check_verlet = TRUE;
			}
#endif
#if defined Y_Periodic
			if ((particle[npp].center[0].y - particle[npp].center[0].displacement_verlet) < 0) {
				check_verlet = TRUE;
			}
			if ((particle[npp].center[0].y + particle[npp].center[0].displacement_verlet) > (global_parameters.Ny - 1)) {
				check_verlet = TRUE;
			}
#endif
			if (global_parameters.Nz > 1) {
#if defined Z_Periodic
				if ((particle[npp].center[0].z - particle[npp].center[0].displacement_verlet) < 0) {
					check_verlet = TRUE;
				}
				if ((particle[npp].center[0].z + particle[npp].center[0].displacement_verlet) > (global_parameters.Nz - 1)) {
					check_verlet = TRUE;
				}
#endif
			}
		}

		if ((disp_max_1 + disp_max_2) > (r_buffer - r_cutoff)) {
			check_verlet = TRUE;
		}

		//
		//	/// **********************************************
		//	/// LAGRANGIAN FORCES CALCULATION AND SPREAD
		//	/// **********************************************
		for (int np = 0; np < num_particles; ++np) {
			if (check_verlet == TRUE) {
				update_verlet(particle, np);

				disp_max_1 = 0;
				disp_max_2 = 0;
				particle[np].center[0].displacement_verlet = 0;
			}

			All_OUT[np] = TRUE;

			number_nodes_in[np] = 0;
			node_in.resize(0);
int xxc[1],yyc[1],zzc[1];
			for (int nn = 0; nn < particle_num_nodes; ++nn) {
				particle[np].node[nn].force_x = 0;
				particle[np].node[nn].force_y = 0;
				particle[np].node[nn].force_z = 0;

				particle[np].node[nn].force_thermal = 0.0;
				Node_IN[np][nn] = FALSE;

				int node_loc = find_processor_index(particle[np].node[nn].x,particle[np].node[nn].y,particle[np].node[nn].z, MPI_parallel,xxc,yyc,zzc);
				if (node_loc == MPI_parallel->processor_id) {
					// above line is equal to this line:// if (particle[np].node[nn].x >= start_x && particle[np].node[nn].x < start_x + actual_rows[MPI_parallel->processor_id]) {
					Node_IN[np][nn] = TRUE;
					node_in.push_back(nn);
					number_nodes_in[np] += 1;
					All_OUT[np] = FALSE;
				}
			}                         // End of: for (int nn = 0; nn < particle_num_nodes; ++nn)
			Particle_IN[np] = FALSE;  // If the CENTER of a particle is located within the area of a processor the Particle_IN is TRUE
			int parloc = find_processor_index(particle[np].center[0].x,particle[np].center[0].y,particle[np].center[0].z, MPI_parallel,xxc,yyc,zzc);
			current_par_center[np] = parloc;
			current_par_centerx[np]=xxc[0];
			current_par_centery[np]=yyc[0];
			current_par_centerz[np]=zzc[0];
			if (parloc == MPI_parallel->processor_id) {
				Particle_IN[np] = TRUE;
			}

			particle[np].center[0].fx_surf = 0.0;
			particle[np].center[0].fy_surf = 0.0;
			particle[np].center[0].fz_surf = 0.0;
			particle[np].center[0].heat_surf = 0.0;  // Total heat

			particle[np].center[0].t1_x = 0.0;  // Torque due to forces in X direction
			particle[np].center[0].t1_y = 0.0;  // Torque due to forces in Y direction
			particle[np].center[0].t1_z = 0.0;  // Torque due to forces in Z direction

			particle[np].center[0].t2_x = 0.0;  // Torque due to forces in X direction
			particle[np].center[0].t2_y = 0.0;  // Torque due to forces in Y direction
			particle[np].center[0].t2_z = 0.0;  // Torque due to forces in Z direction

			particle[np].center[0].t3_x = 0.0;  // Torque due to collisions with walls and other particles
			particle[np].center[0].t3_y = 0.0;
			particle[np].center[0].t3_z = 0.0;

			/// ---------------------------------------------------------------------------------------------
			/// -------------------------------- END OF PARTICLE INITIALIZATION -----------------------------
			/// ---------------------------------------------------------------------------------------------

			/// ---------------------------------------------------------------------------------------------
			/// --------------------------- LAGRANGIAN FORCES CALCULATION AND SPREAD  -----------------------
			/// ---------------------------------------------------------------------------------------------
			// cout << "UUU" << MPI_parallel->processor_id << " " << node_in.size() << endl;

			if (All_OUT[np] == FALSE) {                                                                                                                       // This line is just for speed up. If I deactivate it, the simulation result does not change.
				interpolate(particle[np], Thermal, Flow, node_in, MPI_parallel->start_XYZ[0], MPI_parallel->start_XYZ[1], MPI_parallel->start_XYZ[2], time);  // interpolate velocity from the Eulerian mesh to Lagrangian node

				particle[np].center[0].f3x = 0.0;  // Total Force due to particle-particle collision
				particle[np].center[0].f3y = 0.0;  // Total Force due to particle-particle collision
				particle[np].center[0].f3z = 0.0;  // Total Force due to particle-particle collision

				particle[np].center[0].f4x = 0.0;  // Forces due to particle-wall collision
				particle[np].center[0].f4y = 0.0;  // Forces due to particle-wall collision
				particle[np].center[0].f4z = 0.0;  // Forces due to particle-wall collision

				if (Particle_IN[np] == TRUE) {  // Even if I deactive this "if" line the program works correctly but it is slower

#if defined MOVING_SPHERE
					collision_sphere(particle, np, Flow->viscosity[{0, 0, 0}]);  // Constant omega and viscosity
#endif
#if defined MOVING_SPHEROID
					collision_spheroid(particle, np, Flow->viscosity[{0, 0, 0}]);
#endif
				}
				force_calc(particle[np], np);  // Force and heat calculation at each node

				spread(particle[np], Thermal, Flow, node_in, MPI_parallel->start_XYZ[0], MPI_parallel->start_XYZ[1], MPI_parallel->start_XYZ[2]);  // spread forces from the Lagrangian to the Eulerian mesh

			}  // End of: if (All_OUT[np] == FALSE)
			   //		  /// ---------------------------------------------------------------------------------------------
			   //		  /// ---------------------- END OF LAGRANGIAN FORCES CALCULATION AND SPREAD  ---------------------
			   //		  /// ---------------------------------------------------------------------------------------------
			   //
			   //
			   //		  /// ---------------------------------------------------------------------------------------------
			   //		  /// ------------------------ SEND PARTICLE DATA TO PROCESSOR OF CENTER  -------------------------
			   //		  /// ---------------------------------------------------------------------------------------------
			   //		  /// Particle forces data are sent from all processors that have at least part of the particle....
			   //		  //  ...is there to the processor that holds the particle center.
			   //		  /// ---------------------------------------------------------------------------------------------
			unsigned int num_d = 4;
			static int* buf_node_to_center = new int[particle_num_nodes];
			static double* buf_force_to_center = new double[particle_num_nodes * num_d];
			static int* buf_node_from_others = new int[particle_num_nodes];
			static double* buf_force_from_others = new double[particle_num_nodes * num_d];

			int stp = 0;
			for (int np1 = 0; np1 < particle_num_nodes; ++np1) {

			buf_node_to_center[stp] = 0.;
			buf_force_to_center[stp * num_d + 0] = 0.;
			buf_force_to_center[stp * num_d + 1] = 0.;
			buf_force_to_center[stp * num_d + 2] =0.;
			buf_force_to_center[stp * num_d + 3] =0.;

				if (Node_IN[np][np1] == TRUE) {
					memcpy(buf_node_to_center + stp, &np1, sizeof(int));
					memcpy(buf_force_to_center + stp * num_d + 0, &particle[np].node[np1].force_x, sizeof(double));
					memcpy(buf_force_to_center + stp * num_d + 1, &particle[np].node[np1].force_y, sizeof(double));
					memcpy(buf_force_to_center + stp * num_d + 2, &particle[np].node[np1].force_z, sizeof(double));
					memcpy(buf_force_to_center + stp * num_d + 3, &particle[np].node[np1].force_thermal, sizeof(double));

					stp++;
				}
			}
			/////////////////////////////////////////////////////////////////

		int dom,dom1,dom2;
		dom =proc_extend [MPI_parallel->processor_id];
		dom1=proc_extend1[MPI_parallel->processor_id];
		dom2=proc_extend2[MPI_parallel->processor_id];


	
		if(MPI_parallel->Np_X==1){dom=0;}
		if(MPI_parallel->Np_Y==1){dom1=0;}
		if(MPI_parallel->Np_Z==1){dom2=0;}

	

	for (int i_pr = current_par_centerx[np] - dom; i_pr <= current_par_centerx[np] + dom; i_pr++) {
			for (int j_pr = current_par_centery[np] - dom1; j_pr <= current_par_centery[np] + dom1; j_pr++) {
				for (int k_pr = current_par_centerz[np] - dom2; k_pr <= current_par_centerz[np] + dom2; k_pr++) {

					int orox = 1;
					int oroy = 1;
					int oroz = 1;
					if (MPI_parallel->Np_X>1){
					 orox = (i_pr - 1 + MPI_parallel->Np_X) % (MPI_parallel->Np_X) + 1;}
					if (MPI_parallel->Np_Y>1){
					oroy = (j_pr - 1 + MPI_parallel->Np_Y ) % (MPI_parallel->Np_Y) + 1;}
					if (MPI_parallel->Np_Z>1){
					oroz = (k_pr - 1 + MPI_parallel->Np_Z ) % (MPI_parallel->Np_Z) + 1;}

								
					if (MPI_parallel->processor_id == MPI_parallel->proc_arrangement[orox-1][oroy-1][oroz-1]) {
			
						if (current_par_center[np] != MPI_parallel->processor_id) {
							int mtype  = MTAG;
							int mtype1 = MTAG+1;
							int mtype2 = MTAG+2;
		
							MPI_Send(&number_nodes_in[np], 1                         , MPI_INT   , current_par_center[np], mtype, MPI_COMM_WORLD);

							MPI_Send(buf_node_to_center  , particle_num_nodes        , MPI_INT   , current_par_center[np], mtype1, MPI_COMM_WORLD);

							MPI_Send(buf_force_to_center , particle_num_nodes * num_d, MPI_DOUBLE, current_par_center[np], mtype2, MPI_COMM_WORLD);
								
								
						}
					}
				}
			}
		}





			if (current_par_center[np] == MPI_parallel->processor_id) {
				int nodes_in_number;
				for (int i_pr = current_par_centerx[np] - dom; i_pr <= current_par_centerx[np] + dom; i_pr++) {
					for (int j_pr = current_par_centery[np] - dom1; j_pr <= current_par_centery[np] + dom1; j_pr++) {
						for (int k_pr = current_par_centerz[np] - dom2; k_pr <= current_par_centerz[np] + dom2; k_pr++) {


							int sou_procx = 1;
							int sou_procy = 1;
							int sou_procz = 1;
							if (MPI_parallel->Np_X>1){
							 sou_procx = (i_pr - 1 + MPI_parallel->Np_X ) % (MPI_parallel->Np_X ) + 1;}
							if (MPI_parallel->Np_Y>1){
							 sou_procy = (j_pr - 1 + MPI_parallel->Np_Y ) % (MPI_parallel->Np_Y ) + 1;}
							if (MPI_parallel->Np_Z>1){
							 sou_procz = (k_pr - 1 + MPI_parallel->Np_Z ) % (MPI_parallel->Np_Z ) + 1;}

							if (MPI_parallel->proc_arrangement[sou_procx-1][sou_procy-1][sou_procz-1] != MPI_parallel->processor_id) { 
								 int mtype = MTAG;
								int mtype1 = MTAG+1;
								int mtype2 = MTAG+2;

								MPI_Recv(&nodes_in_number, 1, MPI_INT,MPI_parallel->proc_arrangement[sou_procx-1][sou_procy-1][sou_procz-1] , mtype, MPI_COMM_WORLD, &status);
								MPI_Recv(buf_node_from_others, particle_num_nodes, MPI_INT, MPI_parallel->proc_arrangement[sou_procx-1][sou_procy-1][sou_procz-1], mtype1, MPI_COMM_WORLD, &status);
								MPI_Recv(buf_force_from_others, num_d * particle_num_nodes, MPI_DOUBLE, MPI_parallel->proc_arrangement[sou_procx-1][sou_procy-1][sou_procz-1], mtype2, MPI_COMM_WORLD, &status);

								for (int np1 = 0; np1 < nodes_in_number; ++np1) {  // Important: only up to nodes_in_number
								int snp;
									memcpy(&snp, buf_node_from_others + np1 * 1 + 0, sizeof(int));
									memcpy(&particle[np].node[snp].force_x, buf_force_from_others + np1 * num_d + 0, sizeof(double));
									memcpy(&particle[np].node[snp].force_y, buf_force_from_others + np1 * num_d + 1, sizeof(double));
									memcpy(&particle[np].node[snp].force_z, buf_force_from_others + np1 * num_d + 2, sizeof(double));
									memcpy(&particle[np].node[snp].force_thermal, buf_force_from_others + np1 * num_d + 3, sizeof(double));
								}
							}
						}
					}
				}
			}



			//		/// ---------------------------------------------------------------------------------------------
			//		/// ------------------------ END OF SEND PARTICLE DATA TO PROCESSOR OF CENTER  ------------------
			//		/// ---------------------------------------------------------------------------------------------
			//
			//		/// ---------------------------------------------------------------------------------------------
			//		/// -------------------------------------- TOTAL FORCE AND HEAT  --------------------------------
			//		/// ---------------------------------------------------------------------------------------------
			//		// Total force and heat and torque on particle center is summed up
			//		/// ---------------------------------------------------------------------------------------------
			////
			if (Particle_IN[np] == TRUE) {
				for (int np1 = 0; np1 < particle_num_nodes; ++np1) {
					particle[np].center[0].fx_surf += particle[np].node[np1].force_x;
					particle[np].center[0].fy_surf += particle[np].node[np1].force_y;
					particle[np].center[0].fz_surf += particle[np].node[np1].force_z;

					x_dist = (particle[np].node[np1].x - particle[np].center[0].x);
					y_dist = (particle[np].node[np1].y - particle[np].center[0].y);
					z_dist = (particle[np].node[np1].z - particle[np].center[0].z);

					particle[np].center[0].t1_x += particle[np].node[np1].force_z * y_dist;
					particle[np].center[0].t2_x += particle[np].node[np1].force_y * z_dist;
					particle[np].center[0].t1_y += particle[np].node[np1].force_z * x_dist;
					particle[np].center[0].t2_y += particle[np].node[np1].force_x * z_dist;
					particle[np].center[0].t1_z += particle[np].node[np1].force_x * y_dist;
					particle[np].center[0].t2_z += particle[np].node[np1].force_y * x_dist;

					particle[np].center[0].heat_surf += particle[np].node[np1].force_thermal;
			}
			}
			/// ---------------------------------------------------------------------------------------------
			/// ------------------------------- END OF TOTAL FORCE AND HEAT  --------------------------------
			/// ---------------------------------------------------------------------------------------------

		}  // End of: for (int np = 0; np < num_particles; ++np)

		/// ---------------------------------------------------------------------------------------------
		//	  /// --------------------- EXCHANGING FORCE AND HEAT WITH OTHER PROCESSORS  ----------------------
		//	  /// ---------------------------------------------------------------------------------------------
		//	  /// Eulerian force that is spread to eulerian nodes is first received from neighbor processors ..
		//	  /// ...and then is summed up at the boundaries. Becuase in boundary part of the force may come...
		//	  /// ... uring spread process from neighboring subdomains
		//	  /// ---------------------------------------------------------------------------------------------
		//	  ///// Exchanging Force data

		unsigned int num_data_exc = 3;

		int NPX=MPI_parallel->dev_end[0]-0*MPI_parallel->buffer_size;
		int NPY=MPI_parallel->dev_end[1]-0*MPI_parallel->buffer_size;
		int NPZ=MPI_parallel->dev_end[2]-0*MPI_parallel->buffer_size;

		static double * buf_toleft_f    = new double[range * NPY * NPZ * num_data_exc];
		static double * buf_toright_f   = new double[range * NPY * NPZ * num_data_exc];
		static double * buf_fromleft_f  = new double[range * NPY * NPZ * num_data_exc];
		static double * buf_fromright_f = new double[range * NPY * NPZ * num_data_exc];

		static double * buf_tofront_f   = new double[range * NPX * NPY * num_data_exc];
		static double * buf_torear_f    = new double[range * NPX * NPY * num_data_exc];
		static double * buf_fromfront_f = new double[range * NPX * NPY * num_data_exc];
		static double * buf_fromrear_f  = new double[range * NPX * NPY * num_data_exc];

		static double * buf_totop_f     = new double[range * NPX * NPZ * num_data_exc];
		static double * buf_tobottom_f  = new double[range * NPX * NPZ * num_data_exc];
		static double * buf_fromtop_f   = new double[range * NPX * NPZ * num_data_exc];
		static double * buf_frombottom_f= new double[range * NPX * NPZ * num_data_exc];

		int Left_neighbour, Right_neighbour;
		Right_neighbour = MPI_parallel->proc_arrangement[(MPI_parallel->proc_position[0] + 1) % MPI_parallel->Np_X][MPI_parallel->proc_position[1]][MPI_parallel->proc_position[2]];
		Left_neighbour = MPI_parallel->proc_arrangement[(MPI_parallel->proc_position[0] + MPI_parallel->Np_X - 1) % MPI_parallel->Np_X][MPI_parallel->proc_position[1]][MPI_parallel->proc_position[2]];
		int Bottom_neighbour, Top_neighbour;
        Top_neighbour    = MPI_parallel->proc_arrangement[MPI_parallel->proc_position[0]][(MPI_parallel->proc_position[1]+1)%MPI_parallel->Np_Y][MPI_parallel->proc_position[2]];
        Bottom_neighbour = MPI_parallel->proc_arrangement[MPI_parallel->proc_position[0]][(MPI_parallel->proc_position[1]+MPI_parallel->Np_Y-1)%MPI_parallel->Np_Y][MPI_parallel->proc_position[2]];
		int Rear_neighbour, Front_neighbour;
    	Front_neighbour  = MPI_parallel->proc_arrangement[MPI_parallel->proc_position[0]][MPI_parallel->proc_position[1]][(MPI_parallel->proc_position[2]+1)%MPI_parallel->Np_Z];
        Rear_neighbour   = MPI_parallel->proc_arrangement[MPI_parallel->proc_position[0]][MPI_parallel->proc_position[1]][(MPI_parallel->proc_position[2]+MPI_parallel->Np_Z-1)%MPI_parallel->Np_Z];



		for (int i = 0; i < 3; i++) {
			for (int X = 0; X < NPX; X++) {
				for (int Y = 0; Y < range; Y++) {
					for (int Z = 0; Z < NPZ; Z++) {


						// future: replace global_parameters.Nz with MPI_parallel->dev_end[2]
						memcpy(buf_totop_f     + i * range * NPX *NPZ + Y * NPX * NPZ + X * NPZ + Z, &Flow->force[{X,MPI_parallel->end_XYZ2[1] + 1 +Y,Z,i}], sizeof(double));
						memcpy(buf_tobottom_f  + i * range * NPX *NPZ + Y * NPX * NPZ + X * NPZ + Z, &Flow->force[{X,Y,Z,i}], sizeof(double));
					
					}
				}
			}
		}

		MPI_Sendrecv(buf_totop_f   , 3 * range * NPX * NPZ, MPI_DOUBLE, Top_neighbour   , LTAG, buf_frombottom_f, 3 * range * NPX * NPZ, MPI_DOUBLE, Bottom_neighbour, LTAG, MPI_COMM_WORLD, &status);
		MPI_Sendrecv(buf_tobottom_f, 3 * range * NPX * NPZ, MPI_DOUBLE, Bottom_neighbour, RTAG, buf_fromtop_f   , 3 * range * NPX * NPZ, MPI_DOUBLE, Top_neighbour   , RTAG, MPI_COMM_WORLD, &status);


		for (int i = 0; i < 3; i++) {
			for (int X = 0; X < NPX; X++) {
				for (int Y = 0; Y < range; Y++) {
					for (int Z = 0; Z < NPZ; Z++) {

						memcpy(&Flow->temp_force[{ X, 2 +Y, Z, i}]                            , buf_frombottom_f + i * range * NPX * NPZ + Y * NPX * NPZ + X * NPZ + Z, sizeof(double));
						memcpy(&Flow->temp_force[{ X, MPI_parallel->end_XYZ2[1] - 1 +Y, Z, i}], buf_fromtop_f    + i * range * NPX * NPZ + Y * NPX * NPZ + X * NPZ + Z, sizeof(double));

					}
				}
			}
		}


		for (int X = 0; X < NPX; ++X) {
			for (int Y = 0; Y < range; ++Y) {
				for (int Z = 0; Z < NPZ; ++Z) {
					int Y1 = 2 + Y;
					int Y2 = MPI_parallel->actual_rows_XYZ[1] + 1 - Y;
	
					Flow->force[{X, Y1, Z, 0}] += Flow->temp_force[{X, Y1, Z, 0}];
					Flow->force[{X, Y1, Z, 1}] += Flow->temp_force[{X, Y1, Z, 1}];
					Flow->force[{X, Y1, Z, 2}] += Flow->temp_force[{X, Y1, Z, 2}];
					Flow->force[{X, Y2, Z, 0}] += Flow->temp_force[{X, Y2, Z, 0}];
					Flow->force[{X, Y2, Z, 1}] += Flow->temp_force[{X, Y2, Z, 1}];
					Flow->force[{X, Y2, Z, 2}] += Flow->temp_force[{X, Y2, Z, 2}];

				}
			}
		}
		for (int i = 0; i < 3; i++) {
			for (int X = 0; X < range; X++) {
				for (int Y = 0; Y < NPY; Y++) {
					for (int Z = 0; Z < NPZ; Z++) {

						memcpy(buf_toright_f + i * range * NPY * NPZ + X * NPY * NPZ + Y * NPZ + Z, &Flow->force[{MPI_parallel->end_XYZ2[0] + 1 + X,Y, Z, i}], sizeof(double));
						memcpy(buf_toleft_f  + i * range * NPY * NPZ + X * NPY * NPZ + Y * NPZ + Z, &Flow->force[{X, Y, Z, i}], sizeof(double));
					}
				}
			}
		}

		MPI_Sendrecv(buf_toright_f , 3 * range * NPY * NPZ, MPI_DOUBLE, Right_neighbour, LTAG , buf_fromleft_f  , 3 * range * NPY * NPZ, MPI_DOUBLE, Left_neighbour  , LTAG, MPI_COMM_WORLD, &status);
		MPI_Sendrecv(buf_toleft_f  , 3 * range * NPY * NPZ, MPI_DOUBLE, Left_neighbour, RTAG  , buf_fromright_f , 3 * range * NPY * NPZ, MPI_DOUBLE, Right_neighbour , RTAG, MPI_COMM_WORLD, &status);

		for (int i = 0; i < 3; i++) {
			for (int X = 0; X < range; X++) {
				for (int Y = 0; Y < NPY; Y++) {
					for (int Z = 0; Z < NPZ; Z++) {

						memcpy(&Flow->temp_force[{2 + X, Y, Z, i}]                             , buf_fromleft_f + i * range * NPY * NPZ + X * NPY * NPZ + Y * NPZ + Z, sizeof(double));
						memcpy(&Flow->temp_force[{MPI_parallel->end_XYZ2[0] - 1 + X, Y, Z, i}], buf_fromright_f + i * range * NPY * NPZ + X * NPY * NPZ + Y * NPZ + Z, sizeof(double));
					}
				}
			}
		}





		for (int X = 0; X < range; ++X) {
			for (int Y = 0; Y < NPY; ++Y) {
				for (int Z = 0; Z < NPZ; ++Z) {
					int X1 = 2 + X;
					int X2 = MPI_parallel->actual_rows_XYZ[0] + 1 - X;

					Flow->force[{X1, Y, Z, 0}] += Flow->temp_force[{X1, Y, Z, 0}];
					Flow->force[{X1, Y, Z, 1}] += Flow->temp_force[{X1, Y, Z, 1}];
					Flow->force[{X1, Y, Z, 2}] += Flow->temp_force[{X1, Y, Z, 2}];
					Flow->force[{X2, Y, Z, 0}] += Flow->temp_force[{X2, Y, Z, 0}];
					Flow->force[{X2, Y, Z, 1}] += Flow->temp_force[{X2, Y, Z, 1}];
					Flow->force[{X2, Y, Z, 2}] += Flow->temp_force[{X2, Y, Z, 2}];


				}
			}
		}





		for (int i = 0; i < 3; i++) {
			for (int X = 0; X < NPX; X++) {
				for (int Y = 0; Y < NPY; Y++) {
					for (int Z = 0; Z < range; Z++) {

						memcpy(buf_tofront_f + i * range * NPX * NPY + Z * NPX * NPY + X * NPY + Y, &Flow->force[{ X,Y,MPI_parallel->end_XYZ2[2] + 1 +Z,i}], sizeof(double));
						memcpy(buf_torear_f  + i * range * NPX * NPY + Z * NPX * NPY + X * NPY + Y, &Flow->force[{ X,Y,Z,i}], sizeof(double));
					
					}
				}
			}
		}



		MPI_Sendrecv(buf_tofront_f , 3 * range * NPX * NPY, MPI_DOUBLE, Front_neighbour , LTAG, buf_fromrear_f  , 3 * range * NPX * NPY, MPI_DOUBLE, Rear_neighbour  , LTAG, MPI_COMM_WORLD, &status);
		MPI_Sendrecv(buf_torear_f  , 3 * range * NPX * NPY, MPI_DOUBLE, Rear_neighbour  , RTAG, buf_fromfront_f , 3 * range * NPX * NPY, MPI_DOUBLE, Front_neighbour , RTAG, MPI_COMM_WORLD, &status);






			for (int i = 0; i < 3; i++) {
			for (int X = 0; X < NPX; X++) {
				for (int Y = 0; Y < NPY; Y++) {
					for (int Z = 0; Z < range; Z++) {

						memcpy(&Flow->temp_force[{ X, Y, 2 +Z, i}]                       , buf_fromrear_f  + i * range * NPX * NPY + Z * NPX * NPY + X * NPY + Y, sizeof(double));
						memcpy(&Flow->temp_force[{ X, Y,MPI_parallel->end_XYZ2[2]-1+Z,i}], buf_fromfront_f + i * range * NPX * NPY + Z * NPX * NPY + X * NPY + Y, sizeof(double));

					}
				}
			}
		}





		
		for (int X = 0; X < NPX; ++X) {
			for (int Y = 0; Y < NPY; ++Y) {
				for (int Z = 0; Z < range; ++Z) {
					int Z1 = 2 + Z;
					int Z2 = MPI_parallel->actual_rows_XYZ[2] + 1 - Z;

					Flow->force[{X, Y, Z1, 0}] += Flow->temp_force[{X, Y, Z1, 0}];
					Flow->force[{X, Y, Z1, 1}] += Flow->temp_force[{X, Y, Z1, 1}];
					Flow->force[{X, Y, Z1, 2}] += Flow->temp_force[{X, Y, Z1, 2}];
					Flow->force[{X, Y, Z2, 0}] += Flow->temp_force[{X, Y, Z2, 0}];
					Flow->force[{X, Y, Z2, 1}] += Flow->temp_force[{X, Y, Z2, 1}];
					Flow->force[{X, Y, Z2, 2}] += Flow->temp_force[{X, Y, Z2, 2}];


				}
			}
		}







		//	//	//////////////////////////////////////////////////
		//	/////////// Force_thermal ////////////////////////
#if defined Flow_With_Thermal_Effect

		static double * buf_toleft_ft    = new double[range * NPY * NPZ ];
		static double * buf_toright_ft   = new double[range * NPY * NPZ ];
		static double * buf_fromleft_ft  = new double[range * NPY * NPZ ];
		static double * buf_fromright_ft = new double[range * NPY * NPZ ];

		static double * buf_tofront_ft   = new double[range * NPX * NPY ];
		static double * buf_torear_ft    = new double[range * NPX * NPY ];
		static double * buf_fromfront_ft = new double[range * NPX * NPY ];
		static double * buf_fromrear_ft  = new double[range * NPX * NPY ];

		static double * buf_totop_ft     = new double[range * NPX * NPZ ];
		static double * buf_tobottom_ft  = new double[range * NPX * NPZ ];
		static double * buf_fromtop_ft   = new double[range * NPX * NPZ ];
		static double * buf_frombottom_ft= new double[range * NPX * NPZ ];


		// Prepare messages to be sent


			for (int X = 0; X < NPX; X++) {
				for (int Y = 0; Y < range; Y++) {
					for (int Z = 0; Z < NPZ; Z++) {


						// future: replace global_parameters.Nz with MPI_parallel->dev_end[2]
						memcpy(buf_totop_ft     + Y * NPX * NPZ + X * NPZ + Z, &Thermal->temp_force_thermal[{X,MPI_parallel->end_XYZ2[1] + 1 +Y,Z}], sizeof(double));
						memcpy(buf_tobottom_ft  + Y * NPX * NPZ + X * NPZ + Z, &Thermal->temp_force_thermal[{X,Y,Z}], sizeof(double));
					
					}
				}
			}


		MPI_Sendrecv(buf_totop_ft   , 1 * range * NPX * NPZ, MPI_DOUBLE, Top_neighbour   , LTAG, buf_frombottom_ft, 1 * range * NPX * NPZ, MPI_DOUBLE, Bottom_neighbour, LTAG, MPI_COMM_WORLD, &status);
		MPI_Sendrecv(buf_tobottom_ft, 1 * range * NPX * NPZ, MPI_DOUBLE, Bottom_neighbour, RTAG, buf_fromtop_ft   , 1 * range * NPX * NPZ, MPI_DOUBLE, Top_neighbour   , RTAG, MPI_COMM_WORLD, &status);



			for (int X = 0; X < NPX; X++) {
				for (int Y = 0; Y < range; Y++) {
					for (int Z = 0; Z < NPZ; Z++) {

						memcpy(&Thermal->temp_force_thermal[{ X, 2 +Y, Z}]                            , buf_frombottom_ft +  Y * NPX * NPZ + X * NPZ + Z, sizeof(double));
						memcpy(&Thermal->temp_force_thermal[{ X, MPI_parallel->end_XYZ2[1] - 1 +Y, Z}], buf_fromtop_ft    +  Y * NPX * NPZ + X * NPZ + Z, sizeof(double));

					}
				}
			}


		for (int X = 0; X < NPX; ++X) {
			for (int Y = 0; Y < range; ++Y) {
				for (int Z = 0; Z < NPZ; ++Z) {
					int Y1 = 2 + Y;
					int Y2 = MPI_parallel->actual_rows_XYZ[1] + 1 - Y;
					
					Thermal->force_thermal[{X, Y1, Z}] += Thermal->temp_force_thermal[{X, Y1, Z}];
					Thermal->force_thermal[{X, Y2, Z}] += Thermal->temp_force_thermal[{X, Y2, Z}];

				}
			}
		}




for (int i = 0; i < 3; i++) {
			for (int X = 0; X < range; X++) {
				for (int Y = 0; Y < NPY; Y++) {
					for (int Z = 0; Z < NPZ; Z++) {

						memcpy(buf_toright_ft + X * NPY * NPZ + Y * NPZ + Z, &Thermal->temp_force_thermal[{MPI_parallel->end_XYZ2[0] + 1 + X,Y, Z}], sizeof(double));
						memcpy(buf_toleft_ft  + X * NPY * NPZ + Y * NPZ + Z, &Thermal->temp_force_thermal[{X, Y, Z}], sizeof(double));
					}
				}
			}
		}

		MPI_Sendrecv(buf_toright_ft , 1 * range * NPY * NPZ, MPI_DOUBLE, Right_neighbour, LTAG , buf_fromleft_ft  , 1 * range * NPY * NPZ, MPI_DOUBLE, Left_neighbour  , LTAG, MPI_COMM_WORLD, &status);
		MPI_Sendrecv(buf_toleft_ft  , 1 * range * NPY * NPZ, MPI_DOUBLE, Left_neighbour, RTAG  , buf_fromright_ft , 1 * range * NPY * NPZ, MPI_DOUBLE, Right_neighbour , RTAG, MPI_COMM_WORLD, &status);

		for (int i = 0; i < 3; i++) {
			for (int X = 0; X < range; X++) {
				for (int Y = 0; Y < NPY; Y++) {
					for (int Z = 0; Z < NPZ; Z++) {

						memcpy(&Thermal->temp_force_thermal[{2 + X, Y, Z}]                            , buf_fromleft_ft +  X * NPY * NPZ + Y * NPZ + Z, sizeof(double));
						memcpy(&Thermal->temp_force_thermal[{MPI_parallel->end_XYZ2[0] - 1 + X, Y, Z}], buf_fromright_ft+  X * NPY * NPZ + Y * NPZ + Z, sizeof(double));
					}
				}
			}
		}



		for (int X = 0; X < range; ++X) {
			for (int Y = 0; Y < NPY; ++Y) {
				for (int Z = 0; Z < NPZ; ++Z) {
					int X1 = 2 + X;
					int X2 = MPI_parallel->actual_rows_XYZ[0] + 1 - X;

					Thermal->force_thermal[{X1, Y, Z}] += Thermal->temp_force_thermal[{X1, Y, Z}];
					Thermal->force_thermal[{X2, Y, Z}] += Thermal->temp_force_thermal[{X2, Y, Z}];

				}
			}
		}


			for (int X = 0; X < NPX; X++) {
				for (int Y = 0; Y < NPY; Y++) {
					for (int Z = 0; Z < range; Z++) {

						memcpy(buf_tofront_ft + Z * NPX * NPY + X * NPY + Y, &Thermal->temp_force_thermal[{ X,Y,MPI_parallel->end_XYZ2[2] + 1 +Z}], sizeof(double));
						memcpy(buf_torear_ft  + Z * NPX * NPY + X * NPY + Y, &Thermal->temp_force_thermal[{ X,Y,Z}], sizeof(double));
					
					}
				}
			}




		MPI_Sendrecv(buf_tofront_ft , 1 * range * NPX * NPY, MPI_DOUBLE, Front_neighbour , LTAG, buf_fromrear_ft  , 1 * range * NPX * NPY, MPI_DOUBLE, Rear_neighbour  , LTAG, MPI_COMM_WORLD, &status);
		MPI_Sendrecv(buf_torear_ft  , 1 * range * NPX * NPY, MPI_DOUBLE, Rear_neighbour  , RTAG, buf_fromfront_ft , 1 * range * NPX * NPY, MPI_DOUBLE, Front_neighbour , RTAG, MPI_COMM_WORLD, &status);


			for (int X = 0; X < NPX; X++) {
				for (int Y = 0; Y < NPY; Y++) {
					for (int Z = 0; Z < range; Z++) {

						memcpy(&Thermal->temp_force_thermal[{ X, Y, 2 +Z}]                        , buf_fromrear_ft  + Z * NPX * NPY + X * NPY + Y, sizeof(double));
						memcpy(&Thermal->temp_force_thermal[{ X, Y,MPI_parallel->end_XYZ2[2]-1+Z}], buf_fromfront_ft + Z * NPX * NPY + X * NPY + Y, sizeof(double));

					}
				}
			}



		for (int X = 0; X < NPX; ++X) {
			for (int Y = 0; Y < NPY; ++Y) {
				for (int Z = 0; Z < range; ++Z) {
					int Z1 = 2 + Z;
					int Z2 = MPI_parallel->actual_rows_XYZ[2] + 1 - Z;

					Thermal->force_thermal[{X, Y, Z1}] += Thermal->temp_force_thermal[{X, Y, Z1}];
					Thermal->force_thermal[{X, Y, Z2}] += Thermal->temp_force_thermal[{X, Y, Z2}];

				}
			}
		}


#endif




		//	/// ---------------------------------------------------------------------------------------------
		//	/// ----------------- END OF EXCHANGING FORCE AND HEAT WITH OTHER PROCESSORS  -------------------
		//	/// ---------------------------------------------------------------------------------------------
		//

		return;
	}
}
//=============================================================================================================================
//=============================================================================================================================
//-----------------------------------------------------------------------------------------------------------------------------
// Subroutine: CHECK_ZERO
//-----------------------------------------------------------------------------------------------------------------------------
// This function checks if a variable is less than epsilon it is set to zero
void Particle_sim::check_zero(double& variab, double epsilon) {
	if (abs(variab) < epsilon) {
		variab = 0.0;
	}
	return;
}
//=============================================================================================================================
//=============================================================================================================================
//-----------------------------------------------------------------------------------------------------------------------------
// Subroutine: update_location
//-----------------------------------------------------------------------------------------------------------------------------
void Particle_sim::update_location(particle_struct particle[], int t, int Particle_IN[], Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
#if defined Flow_With_Particle
		double disp_x, disp_y, disp_z;

		for (int npn = 0; npn < num_particles; ++npn) {
			if (Particle_IN[npn] == TRUE) {  // This line is very important for correct operation of MPI. Never deactive this line.
				check_zero(particle[npn].center[0].t1_x, 1.0e-6);
				check_zero(particle[npn].center[0].t1_y, 1.0e-6);
				check_zero(particle[npn].center[0].t1_z, 1.0e-6);

				check_zero(particle[npn].center[0].t2_x, 1.0e-6);
				check_zero(particle[npn].center[0].t2_y, 1.0e-6);
				check_zero(particle[npn].center[0].t2_z, 1.0e-6);

				////////// Updating Velocity and center position ///////
				if (t > 1) {
#if defined With_added_mass
					particle[npn].center[0].vel_x = particle[npn].center[0].vel_x - particle[npn].center[0].fx_surf / particle_mass + particle_gravity_x * (1.0 - 1.0 / den_ratio) + particle[npn].center[0].f3x / particle_mass + particle[npn].center[0].f4x / particle_mass + 1.0 / den_ratio * (particle[npn].center[0].vel_x - particle[npn].center[0].tempx[0]);  // Newton law
					particle[npn].center[0].vel_y = particle[npn].center[0].vel_y - particle[npn].center[0].fy_surf / particle_mass + particle_gravity_y * (1.0 - 1.0 / den_ratio) + particle[npn].center[0].f3y / particle_mass + particle[npn].center[0].f4y / particle_mass + 1.0 / den_ratio * (particle[npn].center[0].vel_y - particle[npn].center[0].tempy[0]);  // Newton law
					particle[npn].center[0].vel_z = particle[npn].center[0].vel_z - particle[npn].center[0].fz_surf / particle_mass + particle_gravity_z * (1.0 - 1.0 / den_ratio) + particle[npn].center[0].f3z / particle_mass + particle[npn].center[0].f4z / particle_mass + 1.0 / den_ratio * (particle[npn].center[0].vel_z - particle[npn].center[0].tempz[0]);  // Newton law

#endif
#if defined Without_added_mass
					particle[npn].center[0].vel_x = particle[npn].center[0].vel_x - particle[npn].center[0].fx_surf / particle_mass + particle_gravity_x * (1.0 - 1.0 / den_ratio) + particle[npn].center[0].f3x / particle_mass + particle[npn].center[0].f4x / particle_mass;  // Newton law
					particle[npn].center[0].vel_y = particle[npn].center[0].vel_y - particle[npn].center[0].fy_surf / particle_mass + particle_gravity_y * (1.0 - 1.0 / den_ratio) + particle[npn].center[0].f3y / particle_mass + particle[npn].center[0].f4y / particle_mass;  // Newton law
					particle[npn].center[0].vel_z = particle[npn].center[0].vel_z - particle[npn].center[0].fz_surf / particle_mass + particle_gravity_z * (1.0 - 1.0 / den_ratio) + particle[npn].center[0].f3z / particle_mass + particle[npn].center[0].f4z / particle_mass;  // Newton law
#endif
				} else {
					particle[npn].center[0].vel_x = particle[npn].center[0].vel_x - particle[npn].center[0].fx_surf / particle_mass + particle_gravity_x * (1.0 - 1.0 / den_ratio) + particle[npn].center[0].f3x / particle_mass + particle[npn].center[0].f4x / particle_mass;  // Newton law
					particle[npn].center[0].vel_y = particle[npn].center[0].vel_y - particle[npn].center[0].fy_surf / particle_mass + particle_gravity_y * (1.0 - 1.0 / den_ratio) + particle[npn].center[0].f3y / particle_mass + particle[npn].center[0].f4y / particle_mass;  // Newton law
					particle[npn].center[0].vel_z = particle[npn].center[0].vel_z - particle[npn].center[0].fz_surf / particle_mass + particle_gravity_z * (1.0 - 1.0 / den_ratio) + particle[npn].center[0].f3z / particle_mass + particle[npn].center[0].f4z / particle_mass;  // Newton law
				}
				particle[npn].center[0].tempx[0] = particle[npn].center[0].tempx[1];
				particle[npn].center[0].tempx[1] = particle[npn].center[0].vel_x;
				particle[npn].center[0].tempy[0] = particle[npn].center[0].tempy[1];
				particle[npn].center[0].tempy[1] = particle[npn].center[0].vel_y;
				particle[npn].center[0].tempz[0] = particle[npn].center[0].tempz[1];
				particle[npn].center[0].tempz[1] = particle[npn].center[0].vel_z;

				center_old_x = particle[npn].center[0].x;
				center_old_y = particle[npn].center[0].y;
				center_old_z = particle[npn].center[0].z;

				disp_x = 0.5 * (particle[npn].center[0].vel_x + particle[npn].center[0].tempx[0]);
				disp_y = 0.5 * (particle[npn].center[0].vel_y + particle[npn].center[0].tempy[0]);
				disp_z = 0.5 * (particle[npn].center[0].vel_z + particle[npn].center[0].tempz[0]);


				particle[npn].center[0].x = particle[npn].center[0].x + disp_x;
				particle[npn].center[0].y = particle[npn].center[0].y + disp_y;
				particle[npn].center[0].z = particle[npn].center[0].z + disp_z;

				particle[npn].center[0].displacement_verlet += sqrt(SQ(disp_x) + SQ(disp_y) + SQ(disp_z));  // Displacement between two Verlet list update steps

				//////////////////////////////////////////

				//////// Updating Angular Velocity ///////
#if defined MOVING_SPHERE
				if (t > 1) {
#if defined With_added_mass
					particle[npn].center[0].omgX = particle[npn].center[0].omgX + 1.0 / mom_inertia_x * (particle[npn].center[0].t2_x - particle[npn].center[0].t1_x) + 1.0 / den_ratio * (particle[npn].center[0].omgX - particle[npn].center[0].temp_omgX[0]);  // Eq. 100 Kang Thesis (804)
#endif
#if defined Without_added_mass
					particle[npn].center[0].omgX = particle[npn].center[0].omgX + 1.0 / mom_inertia_x * (particle[npn].center[0].t2_x - particle[npn].center[0].t1_x);  // Eq. 100 Kang Thesis (804)
#endif
				} else {
					particle[npn].center[0].omgX = particle[npn].center[0].omgX + 1.0 / mom_inertia_x * (particle[npn].center[0].t2_x - particle[npn].center[0].t1_x);
				}

				check_zero(particle[npn].center[0].omgX, 1.0e-12);

				particle[npn].center[0].temp_omgX[0] = particle[npn].center[0].temp_omgX[1];
				particle[npn].center[0].temp_omgX[1] = particle[npn].center[0].omgX;
				//			/////////// For Y rotation
				if (t > 1) {
#if defined With_added_mass
					particle[npn].center[0].omgY = particle[npn].center[0].omgY + 1.0 / mom_inertia_y * (particle[npn].center[0].t1_y - particle[npn].center[0].t2_y) + 1.0 / den_ratio * (particle[npn].center[0].omgY - particle[npn].center[0].temp_omgY[0]);  // Eq. 100 Kang Thesis (804)
#endif
#if defined Without_added_mass
					particle[npn].center[0].omgY = particle[npn].center[0].omgY + 1.0 / mom_inertia_y * (particle[npn].center[0].t1_y - particle[npn].center[0].t2_y);  // Eq. 100 Kang Thesis (804)
#endif
				} else {
					particle[npn].center[0].omgY = particle[npn].center[0].omgY + 1.0 / mom_inertia_y * (particle[npn].center[0].t1_y - particle[npn].center[0].t2_y);
				}

				check_zero(particle[npn].center[0].omgY, 1.0e-12);
				particle[npn].center[0].temp_omgY[0] = particle[npn].center[0].temp_omgY[1];
				particle[npn].center[0].temp_omgY[1] = particle[npn].center[0].omgY;
				/////////// For Z rotation
				if (t > 1) {
#if defined With_added_mass
					particle[npn].center[0].omgZ = particle[npn].center[0].omgZ + 1.0 / mom_inertia_z * (particle[npn].center[0].t1_z - particle[npn].center[0].t2_z) + 1.0 / den_ratio * (particle[npn].center[0].omgZ - particle[npn].center[0].temp_omgZ[0]);  // Eq. 100 Kang Thesis (804)
#endif
#if defined Without_added_mass
					particle[npn].center[0].omgZ = particle[npn].center[0].omgZ + 1.0 / mom_inertia_z * (particle[npn].center[0].t1_z - particle[npn].center[0].t2_z);  // Eq. 100 Kang Thesis (804)
#endif
				} else {
					particle[npn].center[0].omgZ = particle[npn].center[0].omgZ + 1.0 / mom_inertia_z * (particle[npn].center[0].t1_z - particle[npn].center[0].t2_z);
				}

				check_zero(particle[npn].center[0].omgZ, 1.0e-12);
				particle[npn].center[0].temp_omgZ[0] = particle[npn].center[0].temp_omgZ[1];
				particle[npn].center[0].temp_omgZ[1] = particle[npn].center[0].omgZ;
				//////////////////

				particle[npn].center[0].delta_teta_X = 0.5 * (particle[npn].center[0].omgX + particle[npn].center[0].temp_omgX[0]);
				particle[npn].center[0].delta_teta_Y = 0.5 * (particle[npn].center[0].omgY + particle[npn].center[0].temp_omgY[0]);
				particle[npn].center[0].delta_teta_Z = 0.5 * (particle[npn].center[0].omgZ + particle[npn].center[0].temp_omgZ[0]);
				particle[npn].center[0].tetaX = particle[npn].center[0].tetaX + 0.5 * (particle[npn].center[0].omgX + particle[npn].center[0].temp_omgX[0]);
				particle[npn].center[0].tetaY = particle[npn].center[0].tetaY + 0.5 * (particle[npn].center[0].omgY + particle[npn].center[0].temp_omgY[0]);
				particle[npn].center[0].tetaZ = particle[npn].center[0].tetaZ + 0.5 * (particle[npn].center[0].omgZ + particle[npn].center[0].temp_omgZ[0]);

				double x_pos, y_pos, z_pos;

				////////// Updating nodes position ///////
				for (int n = 0; n < particle[npn].num_nodes; ++n) {
					x_pos = particle[npn].node[n].x - center_old_x;
					y_pos = particle[npn].node[n].y - center_old_y;
					z_pos = particle[npn].node[n].z - center_old_z;

					// Around Z-Axis rotation
					particle[npn].node[n].x = particle[npn].center[0].x + x_pos * cos(particle[npn].center[0].delta_teta_Z) - y_pos * sin(particle[npn].center[0].delta_teta_Z);
					particle[npn].node[n].y = particle[npn].center[0].y + x_pos * sin(particle[npn].center[0].delta_teta_Z) + y_pos * cos(particle[npn].center[0].delta_teta_Z);
					particle[npn].node[n].z = particle[npn].center[0].z + z_pos;

#if defined MOVING_SPHERE
					// Around X-Axis rotation
					x_pos = particle[npn].node[n].x - particle[npn].center[0].x;
					y_pos = particle[npn].node[n].y - particle[npn].center[0].y;
					z_pos = particle[npn].node[n].z - particle[npn].center[0].z;
					particle[npn].node[n].x = particle[npn].center[0].x + x_pos;
					particle[npn].node[n].y = particle[npn].center[0].y + y_pos * cos(particle[npn].center[0].delta_teta_X) - z_pos * sin(particle[npn].center[0].delta_teta_X);
					particle[npn].node[n].z = particle[npn].center[0].z + y_pos * sin(particle[npn].center[0].delta_teta_X) + z_pos * cos(particle[npn].center[0].delta_teta_X);

					////Around Y-Axis rotation
					x_pos = particle[npn].node[n].x - particle[npn].center[0].x;
					y_pos = particle[npn].node[n].y - particle[npn].center[0].y;
					z_pos = particle[npn].node[n].z - particle[npn].center[0].z;
					particle[npn].node[n].x = particle[npn].center[0].x + x_pos * cos(particle[npn].center[0].delta_teta_Y) + z_pos * sin(particle[npn].center[0].delta_teta_Y);
					;
					particle[npn].node[n].y = particle[npn].center[0].y + y_pos;
					particle[npn].node[n].z = particle[npn].center[0].z - x_pos * sin(particle[npn].center[0].delta_teta_Y) + z_pos * cos(particle[npn].center[0].delta_teta_Y);

					////
					particle[npn].node[n].vel_x = particle[npn].center[0].vel_x;  // I think these 3 lines can be removed because node velocity is determined by INTERPOLATION
					particle[npn].node[n].vel_y = particle[npn].center[0].vel_y;
					particle[npn].node[n].vel_z = particle[npn].center[0].vel_z;
#endif
				}
#endif                       // End of rotation procedure for Cylinder and ellipse and sphere
                             ////////////////////////////////////////////
#if defined MOVING_SPHEROID  // Calculating rotation speed for spheroid. Using quaternion concept. See: H. Huang et al, J. Fluid Mech. (2012), vol. 692, pp. 369-394.
				quaternion(particle[npn].center[0].Qu0, particle[npn].center[0].Qu1, particle[npn].center[0].Qu2, particle[npn].center[0].Qu3, npn, particle[npn].center[0].quater);
				double torque_1 = (particle[npn].center[0].t2_x - particle[npn].center[0].t1_x - particle[npn].center[0].t3_x);
				double torque_2 = (particle[npn].center[0].t1_y - particle[npn].center[0].t2_y - particle[npn].center[0].t3_y);
				double torque_3 = (particle[npn].center[0].t1_z - particle[npn].center[0].t2_z - particle[npn].center[0].t3_z);
				// Torque transformation from Inertial coordinate to body-fixed coordinate
				T_prime_X[npn] = particle[npn].center[0].quater[1] * torque_1 + particle[npn].center[0].quater[2] * torque_2 + particle[npn].center[0].quater[3] * torque_3;
				T_prime_Y[npn] = particle[npn].center[0].quater[4] * torque_1 + particle[npn].center[0].quater[5] * torque_2 + particle[npn].center[0].quater[6] * torque_3;
				T_prime_Z[npn] = particle[npn].center[0].quater[7] * torque_1 + particle[npn].center[0].quater[8] * torque_2 + particle[npn].center[0].quater[9] * torque_3;
				rungekutta(particle[npn].center[0].Omg_prime_X, particle[npn].center[0].Omg_prime_Y, particle[npn].center[0].Omg_prime_Z, T_prime_X[npn], T_prime_Y[npn], T_prime_Z[npn], mom_inertia_x, mom_inertia_y, mom_inertia_z, particle[npn].center[0].Qu0, particle[npn].center[0].Qu1, particle[npn].center[0].Qu2, particle[npn].center[0].Qu3);

				quaternion(particle[npn].center[0].Qu0, particle[npn].center[0].Qu1, particle[npn].center[0].Qu2, particle[npn].center[0].Qu3, npn, particle[npn].center[0].quater);

				sumq = sqrt(SQ(particle[npn].center[0].Qu0) + SQ(particle[npn].center[0].Qu1) + SQ(particle[npn].center[0].Qu2) + SQ(particle[npn].center[0].Qu3));
				particle[npn].center[0].Qu0 = particle[npn].center[0].Qu0 / sumq;
				particle[npn].center[0].Qu1 = particle[npn].center[0].Qu1 / sumq;
				particle[npn].center[0].Qu2 = particle[npn].center[0].Qu2 / sumq;
				particle[npn].center[0].Qu3 = particle[npn].center[0].Qu3 / sumq;
				particle[npn].center[0].deter = determinant(npn, particle[npn].center[0].quater);  // determinant of Quaternion matrix

				comp_rot_speed(particle[npn].center[0].deter, particle[npn].center[0].omgX, particle[npn].center[0].omgY, particle[npn].center[0].omgZ, npn, particle[npn].center[0].quater, particle[npn].center[0].Omg_prime_X, particle[npn].center[0].Omg_prime_Y, particle[npn].center[0].Omg_prime_Z);  // Angular velocity transformation from body-fixed coordinate to Inertial coordinate.

				if (t > 1) {
#if defined With_added_mass
					particle[npn].center[0].omgX = particle[npn].center[0].omgX + 1.0 / den_ratio * (particle[npn].center[0].omgX - particle[npn].center[0].temp_omgX[0]);  // Eq. 100 Kang Thesis (804)
					particle[npn].center[0].omgY = particle[npn].center[0].omgY + 1.0 / den_ratio * (particle[npn].center[0].omgY - particle[npn].center[0].temp_omgY[0]);  // Eq. 100 Kang Thesis (804)
					particle[npn].center[0].omgZ = particle[npn].center[0].omgZ + 1.0 / den_ratio * (particle[npn].center[0].omgZ - particle[npn].center[0].temp_omgZ[0]);  // Eq. 100 Kang Thesis (804)
					particle[npn].center[0].Omg_prime_X = particle[npn].center[0].quater[1] * (particle[npn].center[0].omgX) + particle[npn].center[0].quater[2] * (particle[npn].center[0].omgY) + particle[npn].center[0].quater[3] * (particle[npn].center[0].omgZ);
					particle[npn].center[0].Omg_prime_Y = particle[npn].center[0].quater[4] * (particle[npn].center[0].omgX) + particle[npn].center[0].quater[5] * (particle[npn].center[0].omgY) + particle[npn].center[0].quater[6] * (particle[npn].center[0].omgZ);
					particle[npn].center[0].Omg_prime_Z = particle[npn].center[0].quater[7] * (particle[npn].center[0].omgX) + particle[npn].center[0].quater[8] * (particle[npn].center[0].omgY) + particle[npn].center[0].quater[9] * (particle[npn].center[0].omgZ);
#endif
				}

				particle[npn].center[0].temp_omgX[0] = particle[npn].center[0].temp_omgX[1];
				particle[npn].center[0].temp_omgX[1] = particle[npn].center[0].omgX;
				particle[npn].center[0].temp_omgY[0] = particle[npn].center[0].temp_omgY[1];
				particle[npn].center[0].temp_omgY[1] = particle[npn].center[0].omgY;
				particle[npn].center[0].temp_omgZ[0] = particle[npn].center[0].temp_omgZ[1];
				particle[npn].center[0].temp_omgZ[1] = particle[npn].center[0].omgZ;

				particle[npn].center[0].delta_teta_X = 0.5 * (particle[npn].center[0].omgX + particle[npn].center[0].temp_omgX[0]);  // For spheroid they are not important because node location is updated by quaternions
				particle[npn].center[0].delta_teta_Y = 0.5 * (particle[npn].center[0].omgY + particle[npn].center[0].temp_omgY[0]);
				particle[npn].center[0].delta_teta_Z = 0.5 * (particle[npn].center[0].omgZ + particle[npn].center[0].temp_omgZ[0]);
				particle[npn].center[0].tetaX = particle[npn].center[0].tetaX + 0.5 * (particle[npn].center[0].omgX + particle[npn].center[0].temp_omgX[0]);
				particle[npn].center[0].tetaY = particle[npn].center[0].tetaY + 0.5 * (particle[npn].center[0].omgY + particle[npn].center[0].temp_omgY[0]);
				particle[npn].center[0].tetaZ = particle[npn].center[0].tetaZ + 0.5 * (particle[npn].center[0].omgZ + particle[npn].center[0].temp_omgZ[0]);

				// Applying the rotation matrix
				for (int n = 0; n < particle_num_nodes; ++n) {
					// Updating nodes location
					update_rot_spheroid(particle[npn].center[0].deter, particle[npn].node[n].x, particle[npn].node[n].y, particle[npn].node[n].z, particle[npn].center[0].quater, particle[npn].node[n].x_pos_1, particle[npn].node[n].y_pos_1, particle[npn].node[n].z_pos_1, particle[npn].center[0].x, particle[npn].center[0].y, particle[npn].center[0].z);  // Node location at INERTIAL coordinate is calculated based on the Quaternions																																																																					   /////////////////////////
					particle[npn].node[n].vel_x = particle[npn].center[0].vel_x;                                                                                                                                                                                                                                                                                  // I think these 3 lines can be removed because node velocity is determined by INTERPOLATION
					particle[npn].node[n].vel_y = particle[npn].center[0].vel_y;
					particle[npn].node[n].vel_z = particle[npn].center[0].vel_z;
				}

#endif  // End of rotation for spheroid

				////////////////////////////////////////////
				////////// Updating Temperature ///////
#if defined Particle_Varying_Temperature
				if (t > 1) {
					particle[npn].center[0].temperature = particle[npn].center[0].temperature - particle[npn].center[0].heat_surf / (particle_mass * specific_heat_ratio) + 1.0 / (den_ratio * specific_heat_ratio) * (particle[npn].center[0].temperature - particle[npn].center[0].temp_temperat[0]) + particle_heat_source / (particle_mass * specific_heat_ratio);
				} else {
					particle[npn].center[0].temperature = particle[npn].center[0].temperature - particle[npn].center[0].heat_surf / (particle_mass * specific_heat_ratio) + particle_heat_source / (particle_mass * specific_heat_ratio);
				}
#endif
				particle[npn].center[0].temp_temperat[0] = particle[npn].center[0].temp_temperat[1];
				particle[npn].center[0].temp_temperat[1] = particle[npn].center[0].temperature;

				/////////////////////////////////////////

#if defined X_Periodic  // In periodic condition, if particle exits from one side it should enter from other side
				if (particle[npn].center[0].x < 0) {
					particle[npn].center[0].x += global_parameters.Nx;
					for (int np1 = 0; np1 < particle[npn].num_nodes; ++np1) {
						particle[npn].node[np1].x += global_parameters.Nx;
					}
				} else if (particle[npn].center[0].x >= global_parameters.Nx) {
					particle[npn].center[0].x -= global_parameters.Nx;
					for (int np1 = 0; np1 < particle[npn].num_nodes; ++np1) {
						particle[npn].node[np1].x -= global_parameters.Nx;
					}
				}
#endif
#if defined Y_Periodic
				if (particle[npn].center[0].y < 0) {
					particle[npn].center[0].y += global_parameters.Ny;
					for (int np1 = 0; np1 < particle[npn].num_nodes; ++np1) {
						particle[npn].node[np1].y += global_parameters.Ny;
					}
				} else if (particle[npn].center[0].y >= global_parameters.Ny) {
					particle[npn].center[0].y -= global_parameters.Ny;
					for (int np1 = 0; np1 < particle[npn].num_nodes; ++np1) {
						particle[npn].node[np1].y -= global_parameters.Ny;
					}
				}
#endif
				if (global_parameters.Nz > 1) {
#if defined Z_Periodic
					if (particle[npn].center[0].z < 0) {
						particle[npn].center[0].z += global_parameters.Nz;
						for (int np1 = 0; np1 < particle[npn].num_nodes; ++np1) {
							particle[npn].node[np1].z += global_parameters.Nz;
						}
					} else if (particle[npn].center[0].z >= global_parameters.Nz) {
						particle[npn].center[0].z -= global_parameters.Nz;
						for (int np1 = 0; np1 < particle[npn].num_nodes; ++np1) {
							particle[npn].node[np1].z -= global_parameters.Nz;
						}
					}
#endif
				}
				particle[npn].center[0].previous_vel_x = particle[npn].center[0].vel_x;
				particle[npn].center[0].previous_vel_y = particle[npn].center[0].vel_y;
				particle[npn].center[0].previous_vel_z = particle[npn].center[0].vel_z;

#if defined Particle_Varying_Temperature
				particle[npn].center[0].temp_part = particle[npn].center[0].temperature;
#endif
			}  // End: if (Particle_IN[npn] == TRUE)
		}      // End:  for (int npn = 0; npn < num_particles; ++npn)
		return;
#endif
	}
}
//=============================================================================================================================
//=============================================================================================================================
//-----------------------------------------------------------------------------------------------------------------------------
// Subroutine: Tasks_After_Newton
//-----------------------------------------------------------------------------------------------------------------------------
void Particle_sim::Tasks_After_Newton(particle_struct particle[], int t, Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id != MASTER) {
		/// ---------------------------------------------------------------------------------------------
		/// ------------------------ SEND PARTICLE DATA TO OTHER PROCESSORS  ----------------------------
		/// ---------------------------------------------------------------------------------------------
		/// Particle data are sent from the processor that holds the center to other processors
		/// ---------------------------------------------------------------------------------------------
		for (int ij = 0; ij < num_particles; ij++) {
			size_t offset = sizeof(center_struct) / sizeof(double);

			static double* buf_to_all = new double[offset];
			static double* buf_to_others2 = new double[particle_num_nodes * 3];
			static double* buf_from_cent1 = new double[offset];
			static double* buf_from_cent2 = new double[particle_num_nodes * 3];

			static int* send_node_data = new int[num_particles];  // Node destination at next time step

			if (Particle_IN[ij] == TRUE) {
				memcpy(buf_to_all, &(particle[ij].center[0]), sizeof(center_struct));
				for (int np1 = 0; np1 < particle_num_nodes; ++np1) {
					send_node_data[ij] = 0;  // 0 means does not send
					memcpy(buf_to_others2 + np1 * 3 + 0, &particle[ij].node[np1].x, sizeof(double));
					memcpy(buf_to_others2 + np1 * 3 + 1, &particle[ij].node[np1].y, sizeof(double));
					memcpy(buf_to_others2 + np1 * 3 + 2, &particle[ij].node[np1].z, sizeof(double));
				}
			}

			/////////////////////////////////////////////////////////////////
			//////Send the location of particle center to ALL processors
			if (current_par_center[ij] == MPI_parallel->processor_id) {
				MPI_Send(buf_to_all, offset, MPI_DOUBLE, MASTER, FROM_WORKER, MPI_COMM_WORLD);  // Send particle data to master
				for (int i_pr = 1; i_pr <= MPI_parallel->num_processors - 1; i_pr++) {
					if (i_pr != MPI_parallel->processor_id) {  // This line is just for speed up. You can deactivate it.
						int mtype = PTAG;
						MPI_Send(buf_to_all, offset, MPI_DOUBLE, i_pr, mtype, MPI_COMM_WORLD);
					}
				}
			}

			if (current_par_center[ij] != MPI_parallel->processor_id) {  // This line is just for speed up. You can deactivate it
				int source = current_par_center[ij];
				int mtype = PTAG;
				MPI_Recv(buf_from_cent1, offset, MPI_DOUBLE, source, mtype, MPI_COMM_WORLD, &status);
				memcpy(&(particle[ij].center[0]), buf_from_cent1, sizeof(center_struct));
			}

			///////////////////////////////////////////////////////////
			///////////////////////////////////////////////////////////
			///////////////////////////////////////////////////////////
		////Send node data and other particle data


		int dom,dom1,dom2;
		dom =proc_extend [MPI_parallel->processor_id];
		dom1=proc_extend1[MPI_parallel->processor_id];
		dom2=proc_extend2[MPI_parallel->processor_id];



		if(MPI_parallel->Np_X==1){dom=0;}
		if(MPI_parallel->Np_Y==1){dom1=0;}
		if(MPI_parallel->Np_Z==1){dom2=0;}
		if (current_par_center[ij] == MPI_parallel->processor_id) {
		for (int i_pr = current_par_centerx[ij] - dom; i_pr <= current_par_centerx[ij] + dom; i_pr++) {
			for (int j_pr = current_par_centery[ij] - dom1; j_pr <= current_par_centery[ij] + dom1; j_pr++) {
				for (int k_pr = current_par_centerz[ij] - dom2; k_pr <= current_par_centerz[ij] + dom2; k_pr++) {

					int destx = 1;
					int desty = 1;
					int destz = 1;
					if (MPI_parallel->Np_X>1){
					 destx = (i_pr - 1 + MPI_parallel->Np_X ) % (MPI_parallel->Np_X ) + 1;}
					if (MPI_parallel->Np_Y>1){
						 desty = (j_pr - 1 + MPI_parallel->Np_Y) % (MPI_parallel->Np_Y ) + 1;}
					if (MPI_parallel->Np_Z>1){
						 destz = (k_pr - 1 + MPI_parallel->Np_Z) % (MPI_parallel->Np_Z ) + 1;}
	
					if (MPI_parallel->processor_id != MPI_parallel->proc_arrangement[destx-1][desty-1][destz-1]) {
							
						int mtype  = RTAG;
						MPI_Send(buf_to_others2, particle_num_nodes * 3, MPI_DOUBLE, MPI_parallel->proc_arrangement[destx-1][desty-1][destz-1], mtype, MPI_COMM_WORLD);	
					

					}
				}
			}
		}

		}


for (int i_pr = current_par_centerx[ij] - dom; i_pr <= current_par_centerx[ij] + dom; i_pr++) {
			for (int j_pr = current_par_centery[ij] - dom1; j_pr <= current_par_centery[ij] + dom1; j_pr++) {
				for (int k_pr = current_par_centerz[ij] - dom2; k_pr <= current_par_centerz[ij] + dom2; k_pr++) {

						int orox = 1;
						int oroy = 1;
						int oroz = 1;
						if (MPI_parallel->Np_X>1){
						 orox = (i_pr - 1 + MPI_parallel->Np_X ) % (MPI_parallel->Np_X ) + 1;}
						if (MPI_parallel->Np_Y>1){
						 oroy = (j_pr - 1 + MPI_parallel->Np_Y ) % (MPI_parallel->Np_Y) + 1;}
						if (MPI_parallel->Np_Z>1){
						 oroz = (k_pr - 1 + MPI_parallel->Np_Z ) % (MPI_parallel->Np_Z) + 1;}
	
						if (MPI_parallel->processor_id == MPI_parallel->proc_arrangement[orox-1][oroy-1][oroz-1]) {
						
						if (current_par_center[ij] != MPI_parallel->processor_id) {

							int source = current_par_center[ij];
							int mtype = RTAG;
							MPI_Recv(buf_from_cent2, particle_num_nodes * 3, MPI_DOUBLE, source, mtype, MPI_COMM_WORLD, &status);
							for (int np1 = 0; np1 < particle_num_nodes; ++np1) {
								memcpy(&particle[ij].node[np1].x, buf_from_cent2 + np1 * 3 + 0, sizeof(double));
								memcpy(&particle[ij].node[np1].y, buf_from_cent2 + np1 * 3 + 1, sizeof(double));
								memcpy(&particle[ij].node[np1].z, buf_from_cent2 + np1 * 3 + 2, sizeof(double));
							}
								
						}
					}
				}
			}
		}

	}

		return;
	}
}
//=============================================================================================================================
//=============================================================================================================================
//-----------------------------------------------------------------------------------------------------------------------------
// Subroutine: INTERPOLATE
//-----------------------------------------------------------------------------------------------------------------------------
//
// The node velocities are interpolated from the fluid nodes via IBM.
// The two-point interpolation stencil (bi-linear interpolation) and other interpolation schemes are included in the present code.
void Particle_sim::interpolate(particle_struct& particle, Thermal_solver* Thermal, Flow_solver* Flow, const vector<int>& node_i, int xstart, int ystart, int zstart, int time) {
	int xmin = 0;
	int xmax = 0;
	int ymin = 0;
	int ymax = 0;
	int zmin = 0;
	int zmax = 0;
	double weight_x = 0.0;
	double weight_y = 0.0;
	double weight_z = 0.0;

	int n;
	// Run over all object nodes.
	for (int nnn = 0; nnn < node_i.size(); ++nnn) {
		n = node_i[nnn];

		// Reset node velocity first since '+=' is used.
		particle.node[n].vel_x = 0;
		particle.node[n].vel_y = 0;
		particle.node[n].vel_z = 0;

		particle.node[n].den = 0;
		particle.node[n].temperature = 0;

		// Identify the lowest fluid lattice node in interpolation range (see spreading).
#if defined Kernel_Delta || defined Two_Point_Delta || defined Four_Point_Delta
		const int x_int = floor(fmod((particle.node[n].x + global_parameters.Nx), global_parameters.Nx) - xstart + range);
		const int y_int = floor(fmod((particle.node[n].y + global_parameters.Ny), global_parameters.Ny) - ystart + range);
		const int z_int = floor(fmod((particle.node[n].z + global_parameters.Nz), global_parameters.Nz) - zstart + range);
		////y_int = floor(particle.node[n].y);
		////z_int = floor(particle.node[n].z);

		// Run over all neighboring fluid nodes.
		// In the case of the two-point interpolation, it is 2x2 fluid nodes.
#if defined Kernel_Delta || defined Four_Point_Delta
		xmin = x_int - 2;
		xmax = x_int + 2;
		ymin = y_int - 2;
		ymax = y_int + 2;
		zmin = z_int - 2;
		zmax = z_int + 2;
#endif

#if defined Two_Point_Delta
		xmin = x_int - 1;
		xmax = x_int + 1;
		ymin = y_int - 1;
		ymax = y_int + 1;
		zmin = z_int - 1;
		zmax = z_int + 1;
#endif
#if defined X_NOT_Periodic
		if (xmin <= 0) {
			xmin = 0;
		}
		if (xmax >= global_parameters.Nx - 1) {
			xmax = global_parameters.Nx - 1;
		}
#endif
#if defined Y_NOT_Periodic
		if (ymin <= 0) {
			ymin = 0;
		}
		if (ymax >= global_parameters.Ny - 1) {
			ymax = global_parameters.Ny - 1;
		}
#endif
#if defined Z_NOT_Periodic
		if (zmin <= 0) {
			zmin = 0;
		}
		if (zmax >= global_parameters.Nz - 1) {
			zmax = global_parameters.Nz - 1;
		}
#endif
#endif

		// Run over all neighboring fluid nodes.
		// In the case of the two-point interpolation, it is 2x2 fluid nodes.
		for (int X = xmin; X <= xmax; ++X) {
			for (int Y = ymin; Y <= ymax; ++Y) {
				for (int Z = zmin; Z <= zmax; ++Z) {
#if defined Kernel_Delta || defined Two_Point_Delta || defined Four_Point_Delta
					// Compute distance between object node and fluid lattice node.
					const double dist_x = (fmod((particle.node[n].x + global_parameters.Nx), global_parameters.Nx) - xstart + 2) - X;
					const double dist_y = (fmod((particle.node[n].y + global_parameters.Ny), global_parameters.Ny) - ystart + 2) - Y;
					const double dist_z = (fmod((particle.node[n].z + global_parameters.Nz), global_parameters.Nz) - zstart + 2) - Z;

					////dist_y = particle.node[n].y - Y;
					////dist_z = particle.node[n].z - Z;

					// Compute interpolation weights for x- and y-direction based on the distance.
#if defined Kernel_Delta
					if (abs(dist_x) <= 2) {
						weight_x = 1.0 / 4.0 * (1. + cos(M_PI * abs(dist_x) / 2.0));
					} else
						weight_x = 0.0;
					if (abs(dist_y) <= 2) {
						weight_y = 1.0 / 4.0 * (1.0 + cos(M_PI * abs(dist_y) / 2.0));
					} else
						weight_y = 0.0;
					if (abs(dist_z) <= 2) {
						weight_z = 1. / 4.0 * (1. + cos(M_PI * abs(dist_z) / 2));
					} else
						weight_z = 0.0;
#endif

#if defined Four_Point_Delta
					if (abs(dist_x) >= 0 && abs(dist_x) < 1) {
						weight_x = 1. / 8.0 * (3. - 2. * abs(dist_x) + sqrt(1 + 4. * abs(dist_x) - 4. * SQ(dist_x)));
					} else if (abs(dist_x) >= 1 && abs(dist_x) < 2) {
						weight_x = 1. / 8.0 * (5. - 2. * abs(dist_x) - sqrt(-7 + 12. * abs(dist_x) - 4. * SQ(dist_x)));
					} else
						weight_x = 0.0;

					if (abs(dist_y) >= 0 && abs(dist_y) < 1) {
						weight_y = 1. / 8.0 * (3. - 2. * abs(dist_y) + sqrt(1 + 4. * abs(dist_y) - 4. * SQ(dist_y)));
					} else if (abs(dist_y) >= 1 && abs(dist_y) < 2) {
						weight_y = 1. / 8.0 * (5. - 2. * abs(dist_y) - sqrt(-7 + 12. * abs(dist_y) - 4. * SQ(dist_y)));
					} else
						weight_y = 0.0;

					if (abs(dist_z) >= 0 && abs(dist_z) < 1) {
						weight_z = 1. / 8.0 * (3. - 2. * abs(dist_z) + sqrt(1 + 4. * abs(dist_z) - 4. * SQ(dist_z)));
					} else if (abs(dist_z) >= 1 && abs(dist_z) < 2) {
						weight_z = 1. / 8.0 * (5. - 2. * abs(dist_z) - sqrt(-7 + 12. * abs(dist_z) - 4. * SQ(dist_z)));
					} else
						weight_z = 0.0;
#endif

#if defined Two_Point_Delta
					if (abs(dist_x) <= 1) {
						weight_x = 1. - abs(dist_x);
					} else
						weight_x = 0.0;
					if (abs(dist_y) <= 1) {
						weight_y = 1. - abs(dist_y);
					} else
						weight_y = 0.0;

					if (abs(dist_z) <= 1) {
						weight_z = 1. - abs(dist_z);
					} else
						weight_z = 0.0;
#endif
#endif

					particle.node[n].vel_x += ((Flow->velocity[{X, Y, Z, 0}]) * weight_x * weight_y * weight_z);
					particle.node[n].vel_y += ((Flow->velocity[{X, Y, Z, 1}]) * weight_x * weight_y * weight_z);
					particle.node[n].vel_z += ((Flow->velocity[{X, Y, Z, 2}]) * weight_x * weight_y * weight_z);
					particle.node[n].den += ((Flow->density[{X, Y, Z}]) * weight_x * weight_y * weight_z);

#if defined Flow_With_Thermal_Effect
					particle.node[n].temperature += ((Thermal->temperature[{X, (Y + global_parameters.Ny) % global_parameters.Ny, (Z + global_parameters.Nz) % global_parameters.Nz}]) * weight_x * weight_y * weight_z);
#endif
				}
			}
		}
	}
	return;
}

//=============================================================================================================================
//=============================================================================================================================
//-----------------------------------------------------------------------------------------------------------------------------
// Subroutine: SREAD
//-----------------------------------------------------------------------------------------------------------------------------
//
// The node forces are spread to the fluid nodes via IBM.
// The two-point interpolation stencil (bi-linear interpolation) is used in the present code.
// It may be replaced by a higher-order interpolation.
void Particle_sim::spread(particle_struct& particle, Thermal_solver* Thermal, Flow_solver* Flow, vector<int> node_i, int xstart, int ystart, int zstart) {
	/// Reset forces
	// This is necessary since '+=' is used afterwards.
	/*for(int X = 0; X < global_parameters.Nx; ++X) {
	for(int Y = 1; Y < global_parameters.Ny - 1; ++Y) {
	force_x[X][Y] = 0;
	force_y[X][Y] = 0;
	}
	}*/

	int xmin = 0;
	int xmax = 0;
	int ymin = 0;
	int ymax = 0;
	int zmin = 0;
	int zmax = 0;
	double weight_x = 0.0;
	double weight_y = 0.0;
	double weight_z = 0.0;

	int n;
	// Run over all object nodes.
	for (int nnn = 0; nnn < node_i.size(); ++nnn) {
		n = node_i[nnn];
		// Identify the lowest fluid lattice node in interpolation range (see spreading).
#if defined Kernel_Delta || defined Two_Point_Delta || defined Four_Point_Delta
		const int x_int = floor(fmod((particle.node[n].x + global_parameters.Nx), global_parameters.Nx) - xstart + range);
		const int y_int = floor(fmod((particle.node[n].y + global_parameters.Ny), global_parameters.Ny) - ystart + range);
		const int z_int = floor(fmod((particle.node[n].z + global_parameters.Nz), global_parameters.Nz) - zstart + range);

		////y_int = floor(particle.node[n].y);
		////z_int = floor(particle.node[n].z);

		// Run over all neighboring fluid nodes.
		// In the case of the two-point interpolation, it is 2x2 fluid nodes.
#if defined Kernel_Delta || defined Four_Point_Delta
		xmin = x_int - 2;
		xmax = x_int + 2;
		ymin = y_int - 2;
		ymax = y_int + 2;
		zmin = z_int - 2;
		zmax = z_int + 2;
#endif

#if defined Two_Point_Delta
		xmin = x_int - 1;
		xmax = x_int + 1;
		ymin = y_int - 1;
		ymax = y_int + 1;
		zmin = z_int - 1;
		zmax = z_int + 1;
#endif
#if defined X_NOT_Periodic
		if (xmin <= 0) {
			xmin = 0;
		}
		if (xmax >= global_parameters.Nx - 1) {
			xmax = global_parameters.Nx - 1;
		}
#endif
#if defined Y_NOT_Periodic
		if (ymin <= 0) {
			ymin = 0;
		}
		if (ymax >= global_parameters.Ny - 1) {
			ymax = global_parameters.Ny - 1;
		}
#endif
#if defined Z_NOT_Periodic
		if (zmin <= 0) {
			zmin = 0;
		}
		if (zmax >= global_parameters.Nz - 1) {
			zmax = global_parameters.Nz - 1;
		}
#endif
#endif

		// Run over all neighboring fluid nodes.
		// In the case of the two-point interpolation, it is 2x2 fluid nodes.
		for (int X = xmin; X <= xmax; ++X) {
			for (int Y = ymin; Y <= ymax; ++Y) {
				for (int Z = zmin; Z <= zmax; ++Z) {
#if defined Kernel_Delta || defined Two_Point_Delta || defined Four_Point_Delta
					// Compute distance between object node and fluid lattice node.
					const double dist_x = (fmod((particle.node[n].x + global_parameters.Nx), global_parameters.Nx) - xstart + 2) - X;
					const double dist_y = (fmod((particle.node[n].y + global_parameters.Ny), global_parameters.Ny) - ystart + 2) - Y;
					const double dist_z = (fmod((particle.node[n].z + global_parameters.Nz), global_parameters.Nz) - zstart + 2) - Z;

					////dist_y = particle.node[n].y - Y;
					////dist_z = particle.node[n].z - Z;

					// Compute interpolation weights for x- and y-direction based on the distance.
#if defined Kernel_Delta
					if (abs(dist_x) <= 2) {
						weight_x = 1.0 / 4.0 * (1. + cos(M_PI * abs(dist_x) / 2.0));
					} else
						weight_x = 0.0;
					if (abs(dist_y) <= 2) {
						weight_y = 1.0 / 4.0 * (1. + cos(M_PI * abs(dist_y) / 2.0));
					} else
						weight_y = 0.0;
					if (abs(dist_z) <= 2) {
						weight_z = 1. / 4.0 * (1. + cos(M_PI * abs(dist_z) / 2));
					} else
						weight_z = 0.0;
#endif

#if defined Four_Point_Delta
					if (abs(dist_x) >= 0 && abs(dist_x) < 1) {
						weight_x = 1. / 8.0 * (3. - 2. * abs(dist_x) + sqrt(1 + 4. * abs(dist_x) - 4. * SQ(dist_x)));
					} else if (abs(dist_x) >= 1 && abs(dist_x) < 2) {
						weight_x = 1. / 8.0 * (5. - 2. * abs(dist_x) - sqrt(-7 + 12. * abs(dist_x) - 4. * SQ(dist_x)));
					} else
						weight_x = 0.0;

					if (abs(dist_y) >= 0 && abs(dist_y) < 1) {
						weight_y = 1. / 8.0 * (3. - 2. * abs(dist_y) + sqrt(1 + 4. * abs(dist_y) - 4. * SQ(dist_y)));
					} else if (abs(dist_y) >= 1 && abs(dist_y) < 2) {
						weight_y = 1. / 8.0 * (5. - 2. * abs(dist_y) - sqrt(-7 + 12. * abs(dist_y) - 4. * SQ(dist_y)));
					} else
						weight_y = 0.0;

					if (abs(dist_z) >= 0 && abs(dist_z) < 1) {
						weight_z = 1. / 8.0 * (3. - 2. * abs(dist_z) + sqrt(1 + 4. * abs(dist_z) - 4. * SQ(dist_z)));
					} else if (abs(dist_z) >= 1 && abs(dist_z) < 2) {
						weight_z = 1. / 8.0 * (5. - 2. * abs(dist_z) - sqrt(-7 + 12. * abs(dist_z) - 4. * SQ(dist_z)));
					} else
						weight_z = 0.0;
#endif

#if defined Two_Point_Delta
					if (abs(dist_x) <= 1) {
						weight_x = 1. - abs(dist_x);
					} else
						weight_x = 0.0;
					if (abs(dist_y) <= 1) {
						weight_y = 1. - abs(dist_y);
					} else
						weight_y = 0.0;
					if (abs(dist_z) <= 1) {
						weight_z = 1. - abs(dist_z);
					} else
						weight_z = 0.0;
#endif
#endif
					// if (global_parameters.Nz == 1) {
					//	weight_z = 1.0;
					// }
					Flow->force[{X, Y, Z, 0}] += (particle.node[n].force_x * weight_x * weight_y * weight_z);
					Flow->force[{X, Y, Z, 1}] += (particle.node[n].force_y * weight_x * weight_y * weight_z);
					Flow->force[{X, Y, Z, 2}] += (particle.node[n].force_z * weight_x * weight_y * weight_z);

#if defined Flow_With_Thermal_Effect
					Thermal->force_thermal[{X, (Y + global_parameters.Ny) % global_parameters.Ny, (Z + global_parameters.Nz) % global_parameters.Nz}] += (particle.node[n].force_thermal * weight_x * weight_y * weight_z);
#endif
				}
			}
		}
	}
}
//=============================================================================================================================
//=============================================================================================================================
//-----------------------------------------------------------------------------------------------------------------------------
// Subroutine: UPDATE_VERLET
//-----------------------------------------------------------------------------------------------------------------------------
void Particle_sim::update_verlet(particle_struct particle[], int iip) {
	//	for (size_t i = 0; i < num_particles; i++)	{
	verlet_list[iip].resize(0);
	for (size_t j = 0; j < num_particles; j++) {
		if (j != iip) {
			double Dij = sqrt(SQ(particle[iip].center[0].x - particle[j].center[0].x) + SQ(particle[iip].center[0].y - particle[j].center[0].y) + SQ(particle[iip].center[0].z - particle[j].center[0].z));
			if (Dij < r_buffer) {
				verlet_list[iip].push_back(j);
			}
		}
	}
	//	}
}
//=============================================================================================================================
//=============================================================================================================================
//-----------------------------------------------------------------------------------------------------------------------------
// Subroutine: COLLISON_SPHERE
//-----------------------------------------------------------------------------------------------------------------------------
// calculation the collision forces for spherical objects
void Particle_sim::collision_sphere(particle_struct particle[], int np, double omega) {
#if defined Sphere_Collision_1_Feng
	// Collision Forces. JUST for Sphere shape

	double Dij;
	for (int np2 = 0; np2 < num_particles; ++np2) {  // Particle-Particle Collision
		if (np2 != np) {
			Dij = sqrt(SQ(particle[np].center[0].x - particle[np2].center[0].x) + SQ(particle[np].center[0].y - particle[np2].center[0].y) + SQ(particle[np].center[0].z - particle[np2].center[0].z));

			if (Dij <= (2.0 * particle[np].radius + th) && Dij > (2.0 * particle[np].radius)) {
				particle[np].center[0].f3x += cij / ep * ((particle[np].center[0].x - particle[np2].center[0].x)) / Dij * sqr((Dij - 2.0 * particle[np].radius - th) / th);
				particle[np].center[0].f3y += cij / ep * ((particle[np].center[0].y - particle[np2].center[0].y)) / Dij * sqr((Dij - 2.0 * particle[np].radius - th) / th);
				particle[np].center[0].f3z += cij / ep * ((particle[np].center[0].z - particle[np2].center[0].z)) / Dij * sqr((Dij - 2.0 * particle[np].radius - th) / th);
			}
			if (Dij <= (2.0 * particle[np].radius)) {
				particle[np].center[0].f3x += (particle[np].center[0].x - particle[np2].center[0].x) / Dij * (cij / ep * sqr((Dij - 2.0 * particle[np].radius - th) / th) + cij / Ep / th * (2.0 * particle[np].radius - Dij));
				particle[np].center[0].f3y += (particle[np].center[0].y - particle[np2].center[0].y) / Dij * (cij / ep * sqr((Dij - 2.0 * particle[np].radius - th) / th) + cij / Ep / th * (2.0 * particle[np].radius - Dij));
				particle[np].center[0].f3z += (particle[np].center[0].z - particle[np2].center[0].z) / Dij * (cij / ep * sqr((Dij - 2.0 * particle[np].radius - th) / th) + cij / Ep / th * (2.0 * particle[np].radius - Dij));
			}
		}
	}  // END of Particle-Particle Collision

	double Rij;
	/////Important: All non-periodic boundaries are treated as wall and thereofore collision takes place
#if defined Y_NOT_Periodic

	if (particle[np].center[0].y <= (global_parameters.Ny - 2) / 2) {  // Particle-Wall Collision section
		Rij = particle[np].center[0].y + particle[np].radius;
	} else {
		Rij = -1.0 * ((global_parameters.Ny - 1) - particle[np].center[0].y + particle[np].radius);
	}

	if (abs(Rij) <= (2.0 * particle[np].radius + thw) && abs(Rij) > (2.0 * particle[np].radius)) {
		particle[np].center[0].f4x += 0.0;
		particle[np].center[0].f4y += cij / epw * sqr((2.0 * particle[np].radius - abs(Rij) + thw) / thw) * Rij / abs(Rij);
		particle[np].center[0].f4z += 0.0;
	}
	if (abs(Rij) <= (2.0 * particle[np].radius)) {
		particle[np].center[0].f4x += 0.0;
		particle[np].center[0].f4y += (cij / epw * sqr((abs(Rij) - 2.0 * particle[np].radius - thw) / thw) + cij / Epw / thw * (2.0 * particle[np].radius - abs(Rij))) * Rij / abs(Rij);
		particle[np].center[0].f4z += 0.0;
	}
#endif
	/////Important: All non-periodic boundaries are treated as wall and thereofore collision takes place
#if defined X_NOT_Periodic
	if (particle[np].center[0].x <= (global_parameters.Nx - 2) / 2) {  // Particle-Wall Collision section
		Rij = particle[np].center[0].x + particle[np].radius - 3.0;
	} else {
		Rij = -1.0 * ((global_parameters.Nx - 3.0) - particle[np].center[0].x + particle[np].radius);
	}

	if (abs(Rij) <= (2.0 * particle[np].radius + thw) && abs(Rij) > (2.0 * particle[np].radius)) {
		particle[np].center[0].f4x += cij / epw * sqr((2.0 * particle[np].radius - abs(Rij) + thw) / thw) * Rij / abs(Rij);
		particle[np].center[0].f4y += 0.0;
		particle[np].center[0].f4z += 0.0;
	}
	if (abs(Rij) <= (2.0 * particle[np].radius)) {
		particle[np].center[0].f4x += (cij / epw * sqr((abs(Rij) - 2.0 * particle[np].radius - thw) / thw) + cij / Epw / thw * (2.0 * particle[np].radius - abs(Rij))) * Rij / abs(Rij);
		particle[np].center[0].f4y += 0.0;
		particle[np].center[0].f4z += 0.0;
	}
#endif
	/////Important: All non-periodic boundaries are treated as wall and thereofore collision takes place
#if defined Z_NOT_Periodic
	if (particle[np].center[0].z <= (global_parameters.Nz - 2) / 2) {  // Particle-Wall Collision section
		Rij = particle[np].center[0].z + particle[np].radius;
	} else {
		Rij = -1.0 * ((global_parameters.Nz - 1) - particle[np].center[0].z + particle[np].radius);
	}

	if (abs(Rij) <= (2.0 * particle[np].radius + thw) && abs(Rij) > (2.0 * particle[np].radius)) {
		particle[np].center[0].f4x += 0.0;
		particle[np].center[0].f4y += 0.0;
		particle[np].center[0].f4z += cij / epw * sqr((2.0 * particle[np].radius - abs(Rij) + thw) / thw) * Rij / abs(Rij);
	}
	if (abs(Rij) <= (2.0 * particle[np].radius)) {
		particle[np].center[0].f4x += 0.0;
		particle[np].center[0].f4y += 0.0;
		particle[np].center[0].f4z += (cij / epw * sqr((abs(Rij) - 2.0 * particle[np].radius - thw) / thw) + cij / Epw / thw * (2.0 * particle[np].radius - abs(Rij))) * Rij / abs(Rij);
	}
#endif
	// END of Particle-Wall Collision
#endif  // end of Sphere_Collision_1_Feng

#if defined Sphere_Collision_2_Feng_Verlet
	// Collision Forces. JUST for Sphere shape

	double Dij;
	for (int np2 = 0; np2 < verlet_list[np].size(); ++np2) {  // Particle-Particle Collision
		int index_p = verlet_list[np][np2];

		if (index_p != np) {
			Dij = sqrt(SQ(particle[np].center[0].x - particle[index_p].center[0].x) + SQ(particle[np].center[0].y - particle[index_p].center[0].y) + SQ(particle[np].center[0].z - particle[index_p].center[0].z));

			if (Dij <= (2.0 * particle[np].radius + th) && Dij > (2.0 * particle[np].radius)) {
				particle[np].center[0].f3x += cij / ep * ((particle[np].center[0].x - particle[index_p].center[0].x)) / Dij * sqr((Dij - 2.0 * particle[np].radius - th) / th);
				particle[np].center[0].f3y += cij / ep * ((particle[np].center[0].y - particle[index_p].center[0].y)) / Dij * sqr((Dij - 2.0 * particle[np].radius - th) / th);
				particle[np].center[0].f3z += cij / ep * ((particle[np].center[0].z - particle[index_p].center[0].z)) / Dij * sqr((Dij - 2.0 * particle[np].radius - th) / th);
			}
			if (Dij <= (2.0 * particle[np].radius)) {
				particle[np].center[0].f3x += (particle[np].center[0].x - particle[index_p].center[0].x) / Dij * (cij / ep * sqr((Dij - 2.0 * particle[np].radius - th) / th) + cij / Ep / th * (2.0 * particle[np].radius - Dij));
				particle[np].center[0].f3y += (particle[np].center[0].y - particle[index_p].center[0].y) / Dij * (cij / ep * sqr((Dij - 2.0 * particle[np].radius - th) / th) + cij / Ep / th * (2.0 * particle[np].radius - Dij));
				particle[np].center[0].f3z += (particle[np].center[0].z - particle[index_p].center[0].z) / Dij * (cij / ep * sqr((Dij - 2.0 * particle[np].radius - th) / th) + cij / Ep / th * (2.0 * particle[np].radius - Dij));
			}
		}
	}  // END of Particle-Particle Collision

	double Rij;
	/////Important: All non-periodic boundaries are treated as wall and thereofore collision takes place
#if defined Y_NOT_Periodic

	if (particle[np].center[0].y <= (global_parameters.Ny - 2) / 2) {  // Particle-Wall Collision section
		Rij = particle[np].center[0].y + particle[np].radius;
	} else {
		Rij = -1.0 * ((global_parameters.Ny - 1) - particle[np].center[0].y + particle[np].radius);
	}

	if (abs(Rij) <= (2.0 * particle[np].radius + thw) && abs(Rij) > (2.0 * particle[np].radius)) {
		particle[np].center[0].f4x += 0.0;
		particle[np].center[0].f4y += cij / epw * sqr((2.0 * particle[np].radius - abs(Rij) + thw) / thw) * Rij / abs(Rij);
		particle[np].center[0].f4z += 0.0;
	}
	if (abs(Rij) <= (2.0 * particle[np].radius)) {
		particle[np].center[0].f4x += 0.0;
		particle[np].center[0].f4y += (cij / epw * sqr((abs(Rij) - 2.0 * particle[np].radius - thw) / thw) + cij / Epw / thw * (2.0 * particle[np].radius - abs(Rij))) * Rij / abs(Rij);
		particle[np].center[0].f4z += 0.0;
	}
#endif
	/////Important: All non-periodic boundaries are treated as wall and thereofore collision takes place
#if defined X_NOT_Periodic
	if (particle[np].center[0].x <= (global_parameters.Nx - 2) / 2) {  // Particle-Wall Collision section
		Rij = particle[np].center[0].x + particle[np].radius;
	} else {
		Rij = -1.0 * ((global_parameters.Nx - 1) - particle[np].center[0].x + particle[np].radius);
	}

	if (abs(Rij) <= (2.0 * particle[np].radius + thw) && abs(Rij) > (2.0 * particle[np].radius)) {
		particle[np].center[0].f4x += cij / epw * sqr((2.0 * particle[np].radius - abs(Rij) + thw) / thw) * Rij / abs(Rij);
		particle[np].center[0].f4y += 0.0;
		particle[np].center[0].f4z += 0.0;
	}
	if (abs(Rij) <= (2.0 * particle[np].radius)) {
		particle[np].center[0].f4x += (cij / epw * sqr((abs(Rij) - 2.0 * particle[np].radius - thw) / thw) + cij / Epw / thw * (2.0 * particle[np].radius - abs(Rij))) * Rij / abs(Rij);
		particle[np].center[0].f4y += 0.0;
		particle[np].center[0].f4z += 0.0;
	}
#endif
	/////Important: All non-periodic boundaries are treated as wall and thereofore collision takes place
#if defined Z_NOT_Periodic
	if (particle[np].center[0].z <= (global_parameters.Nz - 2) / 2) {  // Particle-Wall Collision section
		Rij = particle[np].center[0].z + particle[np].radius;
	} else {
		Rij = -1.0 * ((global_parameters.Nz - 1) - particle[np].center[0].z + particle[np].radius);
	}

	if (abs(Rij) <= (2.0 * particle[np].radius + thw) && abs(Rij) > (2.0 * particle[np].radius)) {
		particle[np].center[0].f4x += 0.0;
		particle[np].center[0].f4y += 0.0;
		particle[np].center[0].f4z += cij / epw * sqr((2.0 * particle[np].radius - abs(Rij) + thw) / thw) * Rij / abs(Rij);
	}
	if (abs(Rij) <= (2.0 * particle[np].radius)) {
		particle[np].center[0].f4x += 0.0;
		particle[np].center[0].f4y += 0.0;
		particle[np].center[0].f4z += (cij / epw * sqr((abs(Rij) - 2.0 * particle[np].radius - thw) / thw) + cij / Epw / thw * (2.0 * particle[np].radius - abs(Rij))) * Rij / abs(Rij);
	}
#endif
	// END of Particle-Wall Collision
#endif  // end of Sphere_Collision_2_Feng_Verlet

#if defined Sphere_Collision_3_Lub_Verlet
	// double tau = 1. / omega;
	const double tau = 3.0 * omega + 0.5;
	// Collision Forces. JUST for Sphere shape
	double Dij;
	for (int np2 = 0; np2 < verlet_list[np].size(); ++np2) {  // Particle-Particle Collision
		int index_p = verlet_list[np][np2];
		if (index_p != np) {
			Dij = sqrt(SQ(particle[np].center[0].x - particle[index_p].center[0].x) + SQ(particle[np].center[0].y - particle[index_p].center[0].y) + SQ(particle[np].center[0].z - particle[index_p].center[0].z));

			if (Dij <= (2.0 * particle[np].radius + th_lub) && Dij > (2.0 * particle[np].radius + th)) {
				double dot_pro;
				dot_pro = (particle[np].center[0].vel_x - particle[index_p].center[0].vel_x) * (particle[np].center[0].x - particle[index_p].center[0].x);
				particle[np].center[0].f3x += -6 * M_PI * (1.0) * ((tau - 0.5) / 3.0) * SQ(particle_radius * particle_radius / (particle_radius + particle_radius)) * (1. / (Dij - 2 * particle_radius) - 1. / th_lub) * dot_pro / Dij;
				dot_pro = (particle[np].center[0].vel_y - particle[index_p].center[0].vel_y) * (particle[np].center[0].y - particle[index_p].center[0].y);
				particle[np].center[0].f3y += -6 * M_PI * (1.0) * ((tau - 0.5) / 3.0) * SQ(particle_radius * particle_radius / (particle_radius + particle_radius)) * (1. / (Dij - 2 * particle_radius) - 1. / th_lub) * dot_pro / Dij;
				dot_pro = (particle[np].center[0].vel_z - particle[index_p].center[0].vel_z) * (particle[np].center[0].z - particle[index_p].center[0].z);
				particle[np].center[0].f3z += -6 * M_PI * (1.0) * ((tau - 0.5) / 3.0) * SQ(particle_radius * particle_radius / (particle_radius + particle_radius)) * (1. / (Dij - 2 * particle_radius) - 1. / th_lub) * dot_pro / Dij;
			}

			if (Dij <= (2.0 * particle[np].radius + th) && Dij > (2.0 * particle[np].radius)) {
				particle[np].center[0].f3x += cij / ep * ((particle[np].center[0].x - particle[index_p].center[0].x)) / Dij * sqr((Dij - 2.0 * particle[np].radius - th) / th);
				particle[np].center[0].f3y += cij / ep * ((particle[np].center[0].y - particle[index_p].center[0].y)) / Dij * sqr((Dij - 2.0 * particle[np].radius - th) / th);
				particle[np].center[0].f3z += cij / ep * ((particle[np].center[0].z - particle[index_p].center[0].z)) / Dij * sqr((Dij - 2.0 * particle[np].radius - th) / th);
			}
			if (Dij <= (2.0 * particle[np].radius)) {
				particle[np].center[0].f3x += (particle[np].center[0].x - particle[index_p].center[0].x) / Dij * (cij / ep * sqr((Dij - 2.0 * particle[np].radius - th) / th) + cij / Ep / th * (2.0 * particle[np].radius - Dij));
				particle[np].center[0].f3y += (particle[np].center[0].y - particle[index_p].center[0].y) / Dij * (cij / ep * sqr((Dij - 2.0 * particle[np].radius - th) / th) + cij / Ep / th * (2.0 * particle[np].radius - Dij));
				particle[np].center[0].f3z += (particle[np].center[0].z - particle[index_p].center[0].z) / Dij * (cij / ep * sqr((Dij - 2.0 * particle[np].radius - th) / th) + cij / Ep / th * (2.0 * particle[np].radius - Dij));
			}
		}
	}  // END of Particle-Particle Collision

/////Important: All non-periodic boundaries are treated as wall and therefore collision takes place
#if defined X_Pipe
	{
		const double CDij = sqrt(((Yc - particle[np].center[0].y) * (Yc - particle[np].center[0].y)) + ((Zc - particle[np].center[0].z) * (Zc - particle[np].center[0].z)));
		const double Rij = PR - CDij + particle[np].radius;
		const double sinn = abs(Zc - particle[np].center[0].z) / (CDij + 0.00000001);
		const double coss = abs(Yc - particle[np].center[0].y) / (CDij + 0.00000001);
		// if(sinn<0.001)
		//{sinn=0.0;}
		// if(coss<0.001)
		//{coss=0.0;}
		// cout << sinn<< "\t"   <<coss << "\t"   <<sinn*sinn+coss*coss<<endl;
		double Rjj;
		if (particle[np].center[0].y <= Yc) {  // Particle-Y-Wall Collision section
			Rjj = abs(Rij);
		} else {
			Rjj = -1.0 * abs(Rij);
		}
		if (abs(Rij) <= (2.0 * particle[np].radius + thw_lub) && abs(Rij) > (2.0 * particle[np].radius + thw)) {
			// cout <<  "near to wall"   <<endl;
			double dot_pro = (particle[np].center[0].vel_y - 0) * Rjj;
			particle[np].center[0].f4x += 0.0;
			particle[np].center[0].f4y += coss * (-6 * M_PI * (1.0) * ((tau - 0.5) / 3.0) * SQ(particle_radius * particle_radius / (particle_radius + particle_radius)) * (1. / (abs(Rij) - 2 * particle_radius) - 1. / thw_lub) * dot_pro / abs(Rij));
			particle[np].center[0].f4z += 0.0;
		}

		if (abs(Rij) <= (2.0 * particle[np].radius + thw) && abs(Rij) > (2.0 * particle[np].radius)) {
			// cout <<  "very near to wall y"   <<endl;
			particle[np].center[0].f4x += 0.0;
			particle[np].center[0].f4y += coss * (cij / epw * sqr((2.0 * particle[np].radius - abs(Rij) + thw) / thw) * Rjj / abs(Rij));
			particle[np].center[0].f4z += 0.0;
		}
		if (abs(Rij) <= (2.0 * particle[np].radius)) {
			// cout <<  "in wall"   <<endl;
			particle[np].center[0].f4x += 0.0;
			particle[np].center[0].f4y += coss * ((cij / epw * sqr((abs(Rij) - 2.0 * particle[np].radius - thw) / thw) + cij / Epw / thw * (2.0 * particle[np].radius - abs(Rij))) * Rjj / abs(Rij));
			particle[np].center[0].f4z += 0.0;
		}

		// Particle-Y-Wall Collision section
		const double Rww = particle[np].center[0].z <= Zc ? Rij : -Rij;

		if (abs(Rij) <= (2.0 * particle[np].radius + thw_lub) && abs(Rij) > (2.0 * particle[np].radius + thw)) {
			double dot_pro = (particle[np].center[0].vel_z - 0) * Rww;
			particle[np].center[0].f4x += 0.0;
			particle[np].center[0].f4y += 0.0;
			particle[np].center[0].f4z += sinn * (-6 * M_PI * (1.0) * ((tau - 0.5) / 3.0) * SQ(particle_radius * particle_radius / (particle_radius + particle_radius)) * (1. / (abs(Rij) - 2 * particle_radius) - 1. / thw_lub) * dot_pro / abs(Rij));
		}

		if (abs(Rij) <= (2.0 * particle[np].radius + thw) && abs(Rij) > (2.0 * particle[np].radius)) {
			particle[np].center[0].f4x += 0.0;
			particle[np].center[0].f4y += 0.0;
			particle[np].center[0].f4z += sinn * (cij / epw * sqr((2.0 * particle[np].radius - abs(Rij) + thw) / thw) * Rww / abs(Rij));
		}
		if (abs(Rij) <= (2.0 * particle[np].radius)) {
			particle[np].center[0].f4x += 0.0;
			particle[np].center[0].f4y += 0.0;
			particle[np].center[0].f4z += sinn * ((cij / epw * sqr((abs(Rij) - 2.0 * particle[np].radius - th) / th) + cij / Epw / th * (2.0 * particle[np].radius - abs(Rij))) * Rww / abs(Rij));
		}
	}
#endif

#if defined Y_NOT_Periodic
	{
		double Rij;
		if (particle[np].center[0].y <= (global_parameters.Ny - 2) / 2) {  // Particle-Y-Wall Collision section
			Rij = particle[np].center[0].y + particle[np].radius;
		} else {
			Rij = -1.0 * ((global_parameters.Ny - 1) - particle[np].center[0].y + particle[np].radius);
		}

		if (abs(Rij) <= (2.0 * particle[np].radius + thw_lub) && abs(Rij) > (2.0 * particle[np].radius + thw)) {
			double dot_pro = (particle[np].center[0].vel_y - 0) * Rij;
			particle[np].center[0].f4x += 0.0;
			particle[np].center[0].f4y += (-6 * M_PI * (1.0) * ((tau - 0.5) / 3.0) * SQ(particle_radius * particle_radius / (particle_radius + particle_radius)) * (1. / (abs(Rij) - 2 * particle_radius) - 1. / thw_lub) * dot_pro / abs(Rij));
			particle[np].center[0].f4z += 0.0;
		}

		if (abs(Rij) <= (2.0 * particle[np].radius + thw) && abs(Rij) > (2.0 * particle[np].radius)) {
			particle[np].center[0].f4x += 0.0;
			particle[np].center[0].f4y += (cij / epw * sqr((2.0 * particle[np].radius - abs(Rij) + thw) / thw) * Rij / abs(Rij));
			particle[np].center[0].f4z += 0.0;
		}
		if (abs(Rij) <= (2.0 * particle[np].radius)) {
			particle[np].center[0].f4x += 0.0;
			particle[np].center[0].f4y += ((cij / epw * sqr((abs(Rij) - 2.0 * particle[np].radius - thw) / thw) + cij / Epw / thw * (2.0 * particle[np].radius - abs(Rij))) * Rij / abs(Rij));
			particle[np].center[0].f4z += 0.0;
		}
	}
#endif

	/////Important: All non-periodic boundaries are treated as wall and thereofore collision takes place
#if defined X_NOT_Periodic
	{
		double Rij;
		if (particle[np].center[0].x <= (global_parameters.Nx - 2) / 2) {  // Particle-Wall Collision section
			Rij = particle[np].center[0].x + particle[np].radius;
		} else {
			Rij = -1.0 * ((global_parameters.Nx - 1) - particle[np].center[0].x + particle[np].radius);
		}

		if (abs(Rij) <= (2.0 * particle[np].radius + thw_lub) && abs(Rij) > (2.0 * particle[np].radius + thw)) {
			double dot_pro = (particle[np].center[0].vel_x - 0) * Rij;
			particle[np].center[0].f4x += -6 * M_PI * (1.0) * ((tau - 0.5) / 3.0) * SQ(particle_radius * particle_radius / (particle_radius + particle_radius)) * (1. / (abs(Rij) - 2 * particle_radius) - 1. / thw_lub) * dot_pro / abs(Rij);
			particle[np].center[0].f4y += 0.0;
			particle[np].center[0].f4z += 0.0;
		}

		if (abs(Rij) <= (2.0 * particle[np].radius + thw) && abs(Rij) > (2.0 * particle[np].radius)) {
			particle[np].center[0].f4x += cij / epw * sqr((2.0 * particle[np].radius - abs(Rij) + thw) / thw) * Rij / abs(Rij);
			particle[np].center[0].f4y += 0.0;
			particle[np].center[0].f4z += 0.0;
		}
		if (abs(Rij) <= (2.0 * particle[np].radius)) {
			particle[np].center[0].f4x += (cij / epw * sqr((abs(Rij) - 2.0 * particle[np].radius - thw) / thw) + cij / Epw / thw * (2.0 * particle[np].radius - abs(Rij))) * Rij / abs(Rij);
			particle[np].center[0].f4y += 0.0;
			particle[np].center[0].f4z += 0.0;
		}
	}
#endif
	/////Important: All non-periodic boundaries are treated as wall and thereofore collision takes place
#if defined Z_NOT_Periodic
	{
		double Rij;
		if (particle[np].center[0].z <= (global_parameters.Nz - 2) / 2) {  // Particle-Wall Collision section
			Rij = particle[np].center[0].z + particle[np].radius;
		} else {
			Rij = -1.0 * ((global_parameters.Nz - 1) - particle[np].center[0].z + particle[np].radius);
		}

		if (abs(Rij) <= (2.0 * particle[np].radius + thw_lub) && abs(Rij) > (2.0 * particle[np].radius + thw)) {
			double dot_pro = (particle[np].center[0].vel_z - 0) * Rij;
			particle[np].center[0].f4x += 0.0;
			particle[np].center[0].f4y += 0.0;
			particle[np].center[0].f4z += -6 * M_PI * (1.0) * ((tau - 0.5) / 3.0) * SQ(particle_radius * particle_radius / (particle_radius + particle_radius)) * (1. / (abs(Rij) - 2 * particle_radius) - 1. / thw_lub) * dot_pro / abs(Rij);
		}

		if (abs(Rij) <= (2.0 * particle[np].radius + thw) && abs(Rij) > (2.0 * particle[np].radius)) {
			particle[np].center[0].f4x += 0.0;
			particle[np].center[0].f4y += 0.0;
			particle[np].center[0].f4z += cij / epw * sqr((2.0 * particle[np].radius - abs(Rij) + thw) / thw) * Rij / abs(Rij);
		}
		if (abs(Rij) <= (2.0 * particle[np].radius)) {
			particle[np].center[0].f4x += 0.0;
			particle[np].center[0].f4y += 0.0;
			particle[np].center[0].f4z += (cij / epw * sqr((abs(Rij) - 2.0 * particle[np].radius - th) / th) + cij / Epw / th * (2.0 * particle[np].radius - abs(Rij))) * Rij / abs(Rij);
		}
	}
#endif
	// END of Particle-Wall Collision
#endif  // end of Sphere_Collision_3_Lub_Verlet
	return;
}
//=============================================================================================================================
//=============================================================================================================================
//-----------------------------------------------------------------------------------------------------------------------------
// Subroutine: COLLISON_SPHEROID
//-----------------------------------------------------------------------------------------------------------------------------
#if defined MOVING_SPHEROID

void Particle_sim::collision_spheroid(particle_struct particle[], int np, double omega) {
	// Collision Forces. for Spheroid shape
	// double tau = 1. / omega;
	double tau = 3.0 * omega + 0.5;
	double Dij, Dist, ffx, ffy, ffz, ffx4, ffy4, ffz4, norvec;
	double xnor_init, ynor_init, znor_init, xnor_slope, ynor_slope, znor_slope;
	double x_out, y_out, z_out;
	double next_point_x, next_point_y, next_point_z;
	double m_slope, n_slope, p_slope;
	double x_intersect1, y_intersect1, z_intersect1;
	double x_intersect2, y_intersect2, z_intersect2;
	double x_intersect_real, y_intersect_real, z_intersect_real;
	double tt_part1, tt_part2, tt_part3;
	double tt_intersect_1, tt_intersect_2;
	double x_0, y_0, z_0, slope_th, xp1, yp1, zp1;
	double a, b, radius_check;
	double x_0_inertial, y_0_inertial, z_0_inertial;
	double xnor_slope_old, ynor_slope_old, znor_slope_old;
	double Rij;
	double dist_to_wall;
	double M_curvature, N_curvature, R_curvature, fi_curvature;
	double x_intersect_real_inertial, y_intersect_real_inertial, z_intersect_real_inertial;
	double check_Y;
	double u_approach_x, u_approach_y, u_approach_z;
	double x_center_curve, y_center_curve, z_center_curve;
	double x_center_curve_inertial, y_center_curve_inertial, z_center_curve_inertial;
	double nor_slope_length;
	int mmm;

	slope_th = 0.001;  // thereshold of slope. If less than this value, computation stops

	a = particle_radius;
	b = particle_radius_2;
	radius_check = MIN(SQ(a) / b, SQ(b) / a);
	std::vector<int> check_walls;

	//////// Particle-Wall collision  //////////////////
	//----------------------------------------------------//
	/////Important: All non-periodic boundaries are treated as wall and thereofore collision takes place
#if defined X_NOT_Periodic || defined Y_NOT_Periodic || defined Z_NOT_Periodic

	double thw_control;
#if defined Spheroid_Collision_1_Feng || defined Spheroid_Collision_2_Feng_Verlet
	thw_control = thw;
#endif
#if defined Spheroid_Collision_3_Feng_Lub_Verlet
	thw_control = thw_lub;
#endif

#if defined Y_NOT_Periodic
	if ((particle[np].center[0].y <= MAX(particle_radius, particle_radius_2) + thw_control)) {
		check_walls.push_back(1);  // 1 means lower Y-wall
	}
	if (particle[np].center[0].y >= global_parameters.Ny - 1 - (MAX(particle_radius, particle_radius_2) + thw_control)) {
		check_walls.push_back(2);  // 2 means upper Y-wall
	}
#endif
#if defined X_NOT_Periodic
	if ((particle[np].center[0].x <= MAX(particle_radius, particle_radius_2) + thw_control)) {
		check_walls.push_back(3);  // 3 means lower X-wall
	}
	if (particle[np].center[0].x >= global_parameters.Nx - 1 - (MAX(particle_radius, particle_radius_2) + thw_control)) {
		check_walls.push_back(4);  // 4 means upper X-wall
	}
#endif
#if defined Z_NOT_Periodic
	if ((particle[np].center[0].z <= MAX(particle_radius, particle_radius_2) + thw_control)) {
		check_walls.push_back(5);  // 5 means lower Z-wall
	}
	if (particle[np].center[0].z >= global_parameters.Nz - 1 - (MAX(particle_radius, particle_radius_2) + thw_control)) {
		check_walls.push_back(6);  // 6 means upper Z-wall
	}
#endif
	//////////////////////// Finding nearest point to wall and calculate the collision force  /////////////////////
	for (size_t i_ch = 0; i_ch < check_walls.size(); i_ch++) {
		dist_to_wall = 1000;

		switch (check_walls[i_ch]) {
			case 1: {       // Lower Y-wall
				xp1 = 0.0;  // wall point in inertial coordinate minus particle center position
				yp1 = 0 - particle[np].center[0].y;
				zp1 = 0.0;

				xnor_slope = 0.0;
				ynor_slope = 1.0;
				znor_slope = 0.0;
				break;
			}
			case 2: {       // Upper Y-wall
				xp1 = 0.0;  // wall point in inertial coordinate minus particle center position
				yp1 = global_parameters.Ny - 1 - particle[np].center[0].y;
				zp1 = 0.0;

				xnor_slope = 0.0;
				ynor_slope = 1.0;
				znor_slope = 0.0;
				break;
			}
			case 3: {                                // Lower X-wall
				xp1 = 0 - particle[np].center[0].x;  // wall point in inertial coordinate minus particle center position
				yp1 = 0.0;
				zp1 = 0.0;

				xnor_slope = 1.0;
				ynor_slope = 0.0;
				znor_slope = 0.0;
				break;
			}
			case 4: {                                                       // Upper X-wall
				xp1 = global_parameters.Nx - 1 - particle[np].center[0].x;  // wall point in inertial coordinate minus particle center position
				yp1 = 0.0;
				zp1 = 0.0;

				xnor_slope = 1.0;
				ynor_slope = 0.0;
				znor_slope = 0.0;
				break;
			}
			case 5: {       // Lower Z-wall
				xp1 = 0.0;  // wall point in inertial coordinate minus particle center position
				yp1 = 0.0;
				zp1 = 0 - particle[np].center[0].z;

				xnor_slope = 0.0;
				ynor_slope = 0.0;
				znor_slope = 1.0;
				break;
			}
			case 6: {       // Lower Z-wall
				xp1 = 0.0;  // wall point in inertial coordinate minus particle center position
				yp1 = 0.0;
				zp1 = global_parameters.Nz - 1 - particle[np].center[0].z;

				xnor_slope = 0.0;
				ynor_slope = 0.0;
				znor_slope = 1.0;
				break;
			}
			default: std::cout << "Kein Antwort";
		}

		x_0 = 0;  // means particle center position in body-fixed frame
		y_0 = 0;  // means particle center position in body-fixed frame
		z_0 = 0;  // means particle center position in body-fixed frame

		mmm = 0;

		do {
			x_out = particle[np].center[0].quater[1] * (xp1) + particle[np].center[0].quater[2] * (yp1) + particle[np].center[0].quater[3] * (zp1);  // location of wall point in body-fixed frame
			y_out = particle[np].center[0].quater[4] * (xp1) + particle[np].center[0].quater[5] * (yp1) + particle[np].center[0].quater[6] * (zp1);
			z_out = particle[np].center[0].quater[7] * (xp1) + particle[np].center[0].quater[8] * (yp1) + particle[np].center[0].quater[9] * (zp1);
			m_slope = x_out - x_0;  // slope of the line conecting the wall and the center of imaginary circle
			n_slope = y_out - y_0;
			p_slope = z_out - z_0;

			/// Intersection of ellipsoid and line, Obtained by MATLAB  ///
			tt_part1 = a * sqrt(SQ(a) * SQ(b) * SQ(n_slope) + SQ(a) * SQ(b) * SQ(p_slope) - SQ(a) * SQ(n_slope) * SQ(z_0) + 2 * SQ(a) * n_slope * p_slope * y_0 * z_0 - SQ(a * p_slope * y_0) + SQ(SQ(b)) * SQ(m_slope) - SQ(b * m_slope * y_0) - SQ(b * m_slope * z_0) + 2 * SQ(b) * m_slope * n_slope * x_0 * y_0 + 2 * SQ(b) * m_slope * p_slope * x_0 * z_0 - SQ(b * n_slope * x_0) - SQ(b * p_slope * x_0));
			tt_part2 = SQ(b) * m_slope * x_0 + SQ(a) * n_slope * y_0 + SQ(a) * p_slope * z_0;
			tt_part3 = (SQ(a * n_slope) + SQ(a * p_slope) + SQ(b * m_slope));
			tt_intersect_1 = -(tt_part1 + tt_part2) / tt_part3;  // Answers of equation
			tt_intersect_2 = -(-tt_part1 + tt_part2) / tt_part3;

			x_intersect1 = x_0 + m_slope * tt_intersect_1;  // in body fixed frame
			y_intersect1 = y_0 + n_slope * tt_intersect_1;
			z_intersect1 = z_0 + p_slope * tt_intersect_1;

			x_intersect2 = x_0 + m_slope * tt_intersect_2;
			y_intersect2 = y_0 + n_slope * tt_intersect_2;
			z_intersect2 = z_0 + p_slope * tt_intersect_2;

			/// Find the nearst point
			if ((SQ(x_intersect1 - x_out) + SQ(y_intersect1 - y_out) + SQ(z_intersect1 - z_out)) < (SQ(x_intersect2 - x_out) + SQ(y_intersect2 - y_out) + SQ(z_intersect2 - z_out))) {
				x_intersect_real = x_intersect1;  // in body fixed frame
				y_intersect_real = y_intersect1;  // in body fixed frame
				z_intersect_real = z_intersect1;  // in body fixed frame
			} else {
				x_intersect_real = x_intersect2;  // in body fixed frame
				y_intersect_real = y_intersect2;  // in body fixed frame
				z_intersect_real = z_intersect2;  // in body fixed frame
			}
			xnor_slope_old = xnor_slope;
			ynor_slope_old = ynor_slope;
			znor_slope_old = znor_slope;

			xnor_slope = -2. * x_intersect_real / particle_radius;  // slope of normal vector on spheroid in body_fixed frame. INWARD direction
			ynor_slope = -2. * y_intersect_real / particle_radius_2;
			znor_slope = -2. * z_intersect_real / particle_radius_2;

			nor_slope_length = (sqrt(SQ(xnor_slope) + SQ(ynor_slope) + SQ(znor_slope)));
			xnor_slope /= nor_slope_length;
			ynor_slope /= nor_slope_length;
			znor_slope /= nor_slope_length;

			x_0 = x_intersect_real + xnor_slope * radius_check;  // center of an imaginary circle for next check. (body-fixed frame)
			y_0 = y_intersect_real + ynor_slope * radius_check;
			z_0 = z_intersect_real + znor_slope * radius_check;

			// finding the center of imaginary circle in INERTIAL frame (x_0_inertial, y_0_inertial, z_0_inertial)
			update_rot_spheroid(particle[np].center[0].deter, x_0_inertial, y_0_inertial, z_0_inertial, particle[np].center[0].quater, x_0, y_0, z_0, particle[np].center[0].x, particle[np].center[0].y, particle[np].center[0].z);  // Find x_0 position in inertial frame

			switch (check_walls[i_ch]) {
				case 1: {
					xp1 = x_0_inertial - particle[np].center[0].x;  // wall point in inertial coordinate minus particle center position
					yp1 = 0 - particle[np].center[0].y;
					zp1 = z_0_inertial - particle[np].center[0].z;
					break;
				}
				case 2: {
					xp1 = x_0_inertial - particle[np].center[0].x;  // wall point in inertial coordinate minus particle center position
					yp1 = global_parameters.Ny - 1 - particle[np].center[0].y;
					zp1 = z_0_inertial - particle[np].center[0].z;
					break;
				}
				case 3: {
					xp1 = 0 - particle[np].center[0].x;  // wall point in inertial coordinate minus particle center position
					yp1 = y_0_inertial - particle[np].center[0].y;
					zp1 = z_0_inertial - particle[np].center[0].z;
					break;
				}
				case 4: {
					xp1 = global_parameters.Nx - 1 - particle[np].center[0].x;  // wall point in inertial coordinate minus particle center position
					yp1 = y_0_inertial - particle[np].center[0].y;
					zp1 = z_0_inertial - particle[np].center[0].z;
					break;
				}
				case 5: {
					xp1 = x_0_inertial - particle[np].center[0].x;  // wall point in inertial coordinate minus particle center position
					yp1 = y_0_inertial - particle[np].center[0].y;
					zp1 = 0 - particle[np].center[0].z;
					break;
				}
				case 6: {
					xp1 = x_0_inertial - particle[np].center[0].x;  // wall point in inertial coordinate minus particle center position
					yp1 = y_0_inertial - particle[np].center[0].y;
					zp1 = global_parameters.Nz - 1 - particle[np].center[0].z;
					break;
				}
				default: cout << "Keine Antwort";
			}

			mmm++;
		} while (abs(xnor_slope - xnor_slope_old) >= slope_th || abs(ynor_slope - ynor_slope_old) >= slope_th || abs(znor_slope - znor_slope_old) >= slope_th);

		update_rot_spheroid(particle[np].center[0].deter, x_intersect_real_inertial, y_intersect_real_inertial, z_intersect_real_inertial, particle[np].center[0].quater, x_intersect_real, y_intersect_real, z_intersect_real, particle[np].center[0].x, particle[np].center[0].y, particle[np].center[0].z);  // Find x_0 position in inertial frame

		switch (check_walls[i_ch]) {
			case 1: {
				dist_to_wall = y_intersect_real_inertial - 0;
				break;
			}
			case 2: {
				dist_to_wall = global_parameters.Ny - 1 - y_intersect_real_inertial;
				break;
			}
			case 3: {
				dist_to_wall = x_intersect_real_inertial - 0;
				break;
			}
			case 4: {
				dist_to_wall = global_parameters.Nx - 1 - x_intersect_real_inertial;
				break;
			}
			case 5: {
				dist_to_wall = z_intersect_real_inertial - 0;
				break;
			}
			case 6: {
				dist_to_wall = global_parameters.Nz - 1 - z_intersect_real_inertial;
				break;
			}
			default: cout << "Keine Antwort";
		}

		if (dist_to_wall < thw_control) {
			// cout << time << endl;
			fi_curvature = asin(x_intersect_real / (sqrt(SQ(x_intersect_real) + SQ(y_intersect_real) + SQ(z_intersect_real))));
			// cout << fi_curvature * 180.0 / M_PI<<endl;
			R_curvature = SQ(a) * b / (SQ(a * sin(fi_curvature)) + SQ(b * cos(fi_curvature)));  // Paper 1366

			x_center_curve = x_intersect_real + xnor_slope * R_curvature;  // center of circle of curvature. (body-fixed frame)
			y_center_curve = y_intersect_real + ynor_slope * R_curvature;
			z_center_curve = z_intersect_real + znor_slope * R_curvature;

			// finding the center of curvature circle in INERTIAL frame (x_0_inertial, y_0_inertial, z_0_inertial)
			update_rot_spheroid(particle[np].center[0].deter, x_center_curve_inertial, y_center_curve_inertial, z_center_curve_inertial, particle[np].center[0].quater, x_center_curve, y_center_curve, z_center_curve, particle[np].center[0].x, particle[np].center[0].y, particle[np].center[0].z);  // Find  position of circle of curvature in inertial frame

			switch (check_walls[i_ch]) {
				case 1: {
					Rij = y_center_curve_inertial + R_curvature;
					break;
				}
				case 2: {
					Rij = -1.0 * ((global_parameters.Ny - 1) - y_center_curve_inertial + R_curvature);
					break;
				}
				case 3: {
					Rij = x_center_curve_inertial + R_curvature;

					break;
				}
				case 4: {
					// double ooo = -1.0 * ((global_parameters.Nx - 1) - x_intersect_real_inertial + 2 * R_curvature);
					Rij = -1.0 * ((global_parameters.Nx - 1) - x_center_curve_inertial + R_curvature);
					break;
				}
				case 5: {
					Rij = z_center_curve_inertial + R_curvature;

					break;
				}
				case 6: {
					Rij = -1.0 * ((global_parameters.Nz - 1) - z_center_curve_inertial + R_curvature);

					break;
				}
				default: std::cout << "Keine Antwort";
			}

			////////////// Force and torque calculation
			switch (check_walls[i_ch]) {
				case 1: {  // Lower Y-Wall
#if defined Spheroid_Collision_3_Feng_Lub_Verlet
					if (abs(Rij) <= (2.0 * R_curvature + thw_lub) && abs(Rij) > (2.0 * R_curvature + thw)) {
						u_approach_y = particle[np].center[0].vel_y - particle[np].center[0].omgZ * (x_center_curve_inertial - particle[np].center[0].x) + particle[np].center[0].omgX * (z_center_curve_inertial - particle[np].center[0].z);
						double dot_pro = (u_approach_y - 0) * Rij;
						ffx4 = 0.0;
						ffy4 = -6 * M_PI * (1.0) * ((tau - 0.5) / 3.0) * SQ(R_curvature * R_curvature / (R_curvature + R_curvature)) * (1. / (abs(Rij) - 2 * R_curvature) - 1. / thw_lub) * dot_pro / abs(Rij);
						ffz4 = 0.0;
					}
#endif
					if (abs(Rij) <= (2.0 * R_curvature + thw) && abs(Rij) > (2.0 * R_curvature)) {
						ffx4 = 0.0;
						ffy4 = cij / epw * sqr((2.0 * R_curvature - abs(Rij) + thw) / thw) * Rij / abs(Rij);
						ffz4 = 0.0;
					}
					if (abs(Rij) <= (2.0 * R_curvature)) {
						ffx4 = 0.0;
						ffy4 = (cij / epw * sqr((abs(Rij) - 2.0 * R_curvature - thw) / thw) + cij / Epw / thw * (2.0 * R_curvature - abs(Rij))) * Rij / abs(Rij);
						ffz4 = 0.0;
					}
					break;
				}
				case 2: {  // Upper Y-Wall
#if defined Spheroid_Collision_3_Feng_Lub_Verlet
					if (abs(Rij) <= (2.0 * R_curvature + thw_lub) && abs(Rij) > (2.0 * R_curvature + thw)) {
						u_approach_y = particle[np].center[0].vel_y - particle[np].center[0].omgZ * (x_center_curve_inertial - particle[np].center[0].x) + particle[np].center[0].omgX * (z_center_curve_inertial - particle[np].center[0].z);
						double dot_pro = (u_approach_y - 0) * Rij;
						ffx4 = 0.0;
						ffy4 = -6 * M_PI * (1.0) * ((tau - 0.5) / 3.0) * SQ(R_curvature * R_curvature / (R_curvature + R_curvature)) * (1. / (abs(Rij) - 2 * R_curvature) - 1. / thw_lub) * dot_pro / abs(Rij);
						ffz4 = 0.0;
					}
#endif
					if (abs(Rij) <= (2.0 * R_curvature + thw) && abs(Rij) > (2.0 * R_curvature)) {
						ffx4 = 0.0;
						ffy4 = cij / epw * sqr((2.0 * R_curvature - abs(Rij) + thw) / thw) * Rij / abs(Rij);
						ffz4 = 0.0;
					}
					if (abs(Rij) <= (2.0 * R_curvature)) {
						ffx4 = 0.0;
						ffy4 = (cij / epw * sqr((abs(Rij) - 2.0 * R_curvature - thw) / thw) + cij / Epw / thw * (2.0 * R_curvature - abs(Rij))) * Rij / abs(Rij);
						ffz4 = 0.0;
					}
					break;
				}
				case 3: {  // Lower X-Wall
#if defined Spheroid_Collision_3_Feng_Lub_Verlet
					if (abs(Rij) <= (2.0 * R_curvature + thw_lub) && abs(Rij) > (2.0 * R_curvature + thw)) {
						u_approach_x = particle[np].center[0].vel_x + particle[np].center[0].omgZ * (y_center_curve_inertial - particle[np].center[0].y) - particle[np].center[0].omgY * (z_center_curve_inertial - particle[np].center[0].z);
						double dot_pro = (u_approach_x - 0) * Rij;
						ffx4 = -6 * M_PI * (1.0) * ((tau - 0.5) / 3.0) * SQ(R_curvature * R_curvature / (R_curvature + R_curvature)) * (1. / (abs(Rij) - 2 * R_curvature) - 1. / thw_lub) * dot_pro / abs(Rij);
						ffy4 = 0.0;
						ffz4 = 0.0;
					}
#endif
					if (abs(Rij) <= (2.0 * R_curvature + thw) && abs(Rij) > (2.0 * R_curvature)) {
						ffx4 = cij / epw * sqr((2.0 * R_curvature - abs(Rij) + thw) / thw) * Rij / abs(Rij);
						ffy4 = 0.0;
						ffz4 = 0.0;
					}
					if (abs(Rij) <= (2.0 * R_curvature)) {
						ffx4 = (cij / epw * sqr((abs(Rij) - 2.0 * R_curvature - thw) / thw) + cij / Epw / thw * (2.0 * R_curvature - abs(Rij))) * Rij / abs(Rij);
						ffy4 = 0.0;
						ffz4 = 0.0;
					}
					break;
				}
				case 4: {  // Upper X-Wall
#if defined Spheroid_Collision_3_Feng_Lub_Verlet
					if (abs(Rij) <= (2.0 * R_curvature + thw_lub) && abs(Rij) > (2.0 * R_curvature + thw)) {
						u_approach_x = particle[np].center[0].vel_x + particle[np].center[0].omgZ * (y_center_curve_inertial - particle[np].center[0].y) - particle[np].center[0].omgY * (z_center_curve_inertial - particle[np].center[0].z);
						// cout << "salam A " << x_intersect_real_inertial << " " << dist_to_wall << " " << " " << Rij << " " << R_curvature << " " << x_0_inertial<<endl;
						double dot_pro = (u_approach_x - 0) * Rij;
						ffx4 = -6 * M_PI * (1.0) * ((tau - 0.5) / 3.0) * SQ(R_curvature * R_curvature / (R_curvature + R_curvature)) * (1. / (abs(Rij) - 2 * R_curvature) - 1. / thw_lub) * dot_pro / abs(Rij);
						ffy4 = 0.0;
						ffz4 = 0.0;
					}
#endif
					if (abs(Rij) <= (2.0 * R_curvature + thw) && abs(Rij) > (2.0 * R_curvature)) {
						ffx4 = cij / epw * sqr((2.0 * R_curvature - abs(Rij) + thw) / thw) * Rij / abs(Rij);
						ffy4 = 0.0;
						ffz4 = 0.0;
					}
					if (abs(Rij) <= (2.0 * R_curvature)) {
						ffx4 = (cij / epw * sqr((abs(Rij) - 2.0 * R_curvature - thw) / thw) + cij / Epw / thw * (2.0 * R_curvature - abs(Rij))) * Rij / abs(Rij);
						ffy4 = 0.0;
						ffz4 = 0.0;
					}
					break;
				}
				case 5: {  // Lower Z-Wall
#if defined Spheroid_Collision_3_Feng_Lub_Verlet
					if (abs(Rij) <= (2.0 * R_curvature + thw_lub) && abs(Rij) > (2.0 * R_curvature + thw)) {
						u_approach_z = particle[np].center[0].vel_z - particle[np].center[0].omgX * (y_center_curve_inertial - particle[np].center[0].y) + particle[np].center[0].omgY * (x_center_curve_inertial - particle[np].center[0].x);
						double dot_pro = (u_approach_z - 0) * Rij;
						ffx4 = 0.0;
						ffy4 = 0.0;
						ffz4 = -6 * M_PI * (1.0) * ((tau - 0.5) / 3.0) * SQ(R_curvature * R_curvature / (R_curvature + R_curvature)) * (1. / (abs(Rij) - 2 * R_curvature) - 1. / thw_lub) * dot_pro / abs(Rij);
					}
#endif
					if (abs(Rij) <= (2.0 * R_curvature + thw) && abs(Rij) > (2.0 * R_curvature)) {
						ffx4 = 0.0;
						ffy4 = 0.0;
						ffz4 = cij / epw * sqr((2.0 * R_curvature - abs(Rij) + thw) / thw) * Rij / abs(Rij);
					}
					if (abs(Rij) <= (2.0 * R_curvature)) {
						ffx4 = 0.0;
						ffy4 = 0.0;
						ffz4 = (cij / epw * sqr((abs(Rij) - 2.0 * R_curvature - thw) / thw) + cij / Epw / thw * (2.0 * R_curvature - abs(Rij))) * Rij / abs(Rij);
					}
					break;
				}
				case 6: {  // Upper Z-Wall
#if defined Spheroid_Collision_3_Feng_Lub_Verlet
					if (abs(Rij) <= (2.0 * R_curvature + thw_lub) && abs(Rij) > (2.0 * R_curvature + thw)) {
						u_approach_z = particle[np].center[0].vel_z - particle[np].center[0].omgX * (y_center_curve_inertial - particle[np].center[0].y) + particle[np].center[0].omgY * (x_center_curve_inertial - particle[np].center[0].x);
						double dot_pro = (u_approach_z - 0) * Rij;
						ffx4 = 0.0;
						ffy4 = 0.0;
						ffz4 = -6 * M_PI * (1.0) * ((tau - 0.5) / 3.0) * SQ(R_curvature * R_curvature / (R_curvature + R_curvature)) * (1. / (abs(Rij) - 2 * R_curvature) - 1. / thw_lub) * dot_pro / abs(Rij);
					}
#endif
					if (abs(Rij) <= (2.0 * R_curvature + thw) && abs(Rij) > (2.0 * R_curvature)) {
						ffx4 = 0.0;
						ffy4 = 0.0;
						ffz4 = cij / epw * sqr((2.0 * R_curvature - abs(Rij) + thw) / thw) * Rij / abs(Rij);
					}
					if (abs(Rij) <= (2.0 * R_curvature)) {
						ffx4 = 0.0;
						ffy4 = 0.0;
						ffz4 = (cij / epw * sqr((abs(Rij) - 2.0 * R_curvature - thw) / thw) + cij / Epw / thw * (2.0 * R_curvature - abs(Rij))) * Rij / abs(Rij);
					}
					break;
				}
				default: std::cout << "Keine Antwort";
			}

			particle[np].center[0].f4x += ffx4;
			particle[np].center[0].f4y += ffy4;
			particle[np].center[0].f4z += ffz4;

			particle[np].center[0].t3_x += ffy4 * (z_intersect_real_inertial - particle[np].center[0].z) - ffz4 * (y_intersect_real_inertial - particle[np].center[0].y);
			particle[np].center[0].t3_y += ffz4 * (x_intersect_real_inertial - particle[np].center[0].x) - ffx4 * (z_intersect_real_inertial - particle[np].center[0].z);
			particle[np].center[0].t3_z += ffx4 * (y_intersect_real_inertial - particle[np].center[0].y) - ffy4 * (x_intersect_real_inertial - particle[np].center[0].x);
			// cout << dist_to_wall << " " << particle[np].center[0].f4x << endl;

		}  // if (dist_to_wall<thw_control)
	}      /// for (size_t i_ch = 0; i_ch < check_walls.size(); i_ch++)
#endif     // Walls collision

	//------------------------------------------------------------------
	//------------------------------------------------------------------
	//------------------------------------------------------------------
	//-------------- Particle_Particle collision -----------------/////
	int max_par_index;
	int index_p;
	double par_par_dist;
	double xp1_on_p2, yp1_on_p2, zp1_on_p2;
	double x_0_of_p1, y_0_of_p1, z_0_of_p1;
	double x_out_on_p2, y_out_on_p2, z_out_on_p2;
	double m_slope_p1, n_slope_p1, p_slope_p1;
	double x_intersect1_of_p1, y_intersect1_of_p1, z_intersect1_of_p1;
	double x_intersect2_of_p1, y_intersect2_of_p1, z_intersect2_of_p1;
	double x_intersect_real_of_p1, y_intersect_real_of_p1, z_intersect_real_of_p1;
	double xnor_slope_old_of_p1, ynor_slope_old_of_p1, znor_slope_old_of_p1;
	double xnor_slope_of_p1, ynor_slope_of_p1, znor_slope_of_p1;
	double x_0_inertial_of_p1, y_0_inertial_of_p1, z_0_inertial_of_p1;
	double x_intersect_real_inertial_of_p1, y_intersect_real_inertial_of_p1, z_intersect_real_inertial_of_p1;
	double xp1_on_p1, yp1_on_p1, zp1_on_p1;
	double temp_slope_length;
	double xnor_slope_of_p2, ynor_slope_of_p2, znor_slope_of_p2;
	double x_0_of_p2, y_0_of_p2, z_0_of_p2;
	double x_out_on_p1, y_out_on_p1, z_out_on_p1;
	double m_slope_p2, n_slope_p2, p_slope_p2;
	double x_intersect1_of_p2, y_intersect1_of_p2, z_intersect1_of_p2;
	double x_intersect2_of_p2, y_intersect2_of_p2, z_intersect2_of_p2;
	double x_intersect_real_of_p2, y_intersect_real_of_p2, z_intersect_real_of_p2;
	double xnor_slope_old_of_p2, ynor_slope_old_of_p2, znor_slope_old_of_p2;
	double x_0_inertial_of_p2, y_0_inertial_of_p2, z_0_inertial_of_p2;
	double x_intersect_real_inertial_of_p2, y_intersect_real_inertial_of_p2, z_intersect_real_inertial_of_p2;
	double fi_curvature_p1, fi_curvature_p2;
	double R_curvature_p1, R_curvature_p2;
	double thp_control;
	double x_center_curve_p1, y_center_curve_p1, z_center_curve_p1;
	double x_center_curve_inertial_p1, y_center_curve_inertial_p1, z_center_curve_inertial_p1;
	double x_center_curve_inertial_p2, y_center_curve_inertial_p2, z_center_curve_inertial_p2;
	double x_center_curve_p2, y_center_curve_p2, z_center_curve_p2;
	double u_approach_x_p1, u_approach_y_p1, u_approach_z_p1;
	double u_approach_x_p2, u_approach_y_p2, u_approach_z_p2;

#if defined Spheroid_Collision_1_Feng || defined Spheroid_Collision_2_Feng_Verlet
	thp_control = th;
#endif
#if defined Spheroid_Collision_3_Feng_Lub_Verlet
	thp_control = th_lub;
#endif

	ffx = 0.0;
	ffy = 0.0;
	ffz = 0.0;

#if defined Spheroid_Collision_1_Feng
	max_par_index = num_particles;
#endif
#if defined Spheroid_Collision_2_Feng_Verlet || defined Spheroid_Collision_3_Feng_Lub_Verlet
	max_par_index = verlet_list[np].size();
#endif

	for (int np2 = 0; np2 < max_par_index; ++np2) {  // Particle-Particle Collision
#if defined Spheroid_Collision_1_Feng
		index_p = np2;
#endif
#if defined Spheroid_Collision_2_Feng_Verlet || defined Spheroid_Collision_3_Feng_Lub_Verlet
		index_p = verlet_list[np][np2];
#endif
		if (index_p != np) {
			par_par_dist = 1000;

			xp1_on_p2 = particle[index_p].center[0].x - particle[np].center[0].x;  // point on P2 in inertial coordinate minus particle P1 center position
			yp1_on_p2 = particle[index_p].center[0].y - particle[np].center[0].y;
			zp1_on_p2 = particle[index_p].center[0].z - particle[np].center[0].z;

			xnor_slope_of_p1 = particle[np].center[0].x - particle[index_p].center[0].x;  // slope of line connecting nearest points
			ynor_slope_of_p1 = particle[np].center[0].y - particle[index_p].center[0].y;
			znor_slope_of_p1 = particle[np].center[0].z - particle[index_p].center[0].z;

			temp_slope_length = sqrt(SQ(particle[np].center[0].x - particle[index_p].center[0].x) + SQ(particle[np].center[0].y - particle[index_p].center[0].y) + SQ(particle[np].center[0].z - particle[index_p].center[0].z));
			xnor_slope_of_p1 /= temp_slope_length;
			ynor_slope_of_p1 /= temp_slope_length;
			znor_slope_of_p1 /= temp_slope_length;

			x_0_of_p1 = 0;  // means particle center position in body-fixed frame
			y_0_of_p1 = 0;  // means particle center position in body-fixed frame
			z_0_of_p1 = 0;  // means particle center position in body-fixed frame

			x_0_of_p2 = 0;  // means particle center position in body-fixed frame
			y_0_of_p2 = 0;  // means particle center position in body-fixed frame
			z_0_of_p2 = 0;  // means particle center position in body-fixed frame

			xnor_slope_of_p2 = particle[index_p].center[0].x - particle[np].center[0].x;  // slope of line connecting nearest points
			ynor_slope_of_p2 = particle[index_p].center[0].y - particle[np].center[0].y;
			znor_slope_of_p2 = particle[index_p].center[0].z - particle[np].center[0].z;

			// initial set. Changes later
			temp_slope_length = sqrt(SQ(particle[index_p].center[0].x - particle[np].center[0].x) + SQ(particle[index_p].center[0].y - particle[np].center[0].y) + SQ(particle[index_p].center[0].z - particle[np].center[0].z));
			xnor_slope_of_p2 /= temp_slope_length;
			ynor_slope_of_p2 /= temp_slope_length;
			znor_slope_of_p2 /= temp_slope_length;

			mmm = 0;

			do {
				x_out_on_p2 = particle[np].center[0].quater[1] * (xp1_on_p2) + particle[np].center[0].quater[2] * (yp1_on_p2) + particle[np].center[0].quater[3] * (zp1_on_p2);  // location of P2 point in body-fixed frame of P1
				y_out_on_p2 = particle[np].center[0].quater[4] * (xp1_on_p2) + particle[np].center[0].quater[5] * (yp1_on_p2) + particle[np].center[0].quater[6] * (zp1_on_p2);
				z_out_on_p2 = particle[np].center[0].quater[7] * (xp1_on_p2) + particle[np].center[0].quater[8] * (yp1_on_p2) + particle[np].center[0].quater[9] * (zp1_on_p2);
				m_slope_p1 = x_out_on_p2 - x_0_of_p1;  // slope of the line conecting the point on P2 and the center of imaginary circle of P1
				n_slope_p1 = y_out_on_p2 - y_0_of_p1;
				p_slope_p1 = z_out_on_p2 - z_0_of_p1;

				/// Intersection of ellipsoid and line, Obtained by MATLAB  ///
				tt_part1 = a * sqrt(SQ(a) * SQ(b) * SQ(n_slope_p1) + SQ(a) * SQ(b) * SQ(p_slope_p1) - SQ(a) * SQ(n_slope_p1) * SQ(z_0_of_p1) + 2 * SQ(a) * n_slope_p1 * p_slope_p1 * y_0_of_p1 * z_0_of_p1 - SQ(a * p_slope_p1 * y_0_of_p1) + SQ(SQ(b)) * SQ(m_slope_p1) - SQ(b * m_slope_p1 * y_0_of_p1) - SQ(b * m_slope_p1 * z_0_of_p1) + 2 * SQ(b) * m_slope_p1 * n_slope_p1 * x_0_of_p1 * y_0_of_p1 + 2 * SQ(b) * m_slope_p1 * p_slope_p1 * x_0_of_p1 * z_0_of_p1 - SQ(b * n_slope_p1 * x_0_of_p1) - SQ(b * p_slope_p1 * x_0_of_p1));
				tt_part2 = SQ(b) * m_slope_p1 * x_0_of_p1 + SQ(a) * n_slope_p1 * y_0_of_p1 + SQ(a) * p_slope_p1 * z_0_of_p1;
				tt_part3 = (SQ(a * n_slope_p1) + SQ(a * p_slope_p1) + SQ(b * m_slope_p1));
				tt_intersect_1 = -(tt_part1 + tt_part2) / tt_part3;  // Answers of equation
				tt_intersect_2 = -(-tt_part1 + tt_part2) / tt_part3;

				x_intersect1_of_p1 = x_0_of_p1 + m_slope_p1 * tt_intersect_1;  // in body fixed frame of P1
				y_intersect1_of_p1 = y_0_of_p1 + n_slope_p1 * tt_intersect_1;
				z_intersect1_of_p1 = z_0_of_p1 + p_slope_p1 * tt_intersect_1;

				x_intersect2_of_p1 = x_0_of_p1 + m_slope_p1 * tt_intersect_2;
				y_intersect2_of_p1 = y_0_of_p1 + n_slope_p1 * tt_intersect_2;
				z_intersect2_of_p1 = z_0_of_p1 + p_slope_p1 * tt_intersect_2;

				/// Find the nearst point
				if ((SQ(x_intersect1_of_p1 - x_out_on_p2) + SQ(y_intersect1_of_p1 - y_out_on_p2) + SQ(z_intersect1_of_p1 - z_out_on_p2)) < (SQ(x_intersect2_of_p1 - x_out_on_p2) + SQ(y_intersect2_of_p1 - y_out_on_p2) + SQ(z_intersect2_of_p1 - z_out_on_p2))) {
					x_intersect_real_of_p1 = x_intersect1_of_p1;  // in body fixed frame
					y_intersect_real_of_p1 = y_intersect1_of_p1;  // in body fixed frame
					z_intersect_real_of_p1 = z_intersect1_of_p1;  // in body fixed frame
				} else {
					x_intersect_real_of_p1 = x_intersect2_of_p1;  // in body fixed frame
					y_intersect_real_of_p1 = y_intersect2_of_p1;  // in body fixed frame
					z_intersect_real_of_p1 = z_intersect2_of_p1;  // in body fixed frame
				}
				xnor_slope_old_of_p1 = xnor_slope_of_p1;
				ynor_slope_old_of_p1 = ynor_slope_of_p1;
				znor_slope_old_of_p1 = znor_slope_of_p1;

				xnor_slope_of_p1 = -2. * x_intersect_real_of_p1 / particle_radius;  // slope of normal vector on spheroid in body_fixed frame. INWARD direction
				ynor_slope_of_p1 = -2. * y_intersect_real_of_p1 / particle_radius_2;
				znor_slope_of_p1 = -2. * z_intersect_real_of_p1 / particle_radius_2;

				nor_slope_length = (sqrt(SQ(xnor_slope_of_p1) + SQ(ynor_slope_of_p1) + SQ(znor_slope_of_p1)));
				xnor_slope_of_p1 /= nor_slope_length;
				ynor_slope_of_p1 /= nor_slope_length;
				znor_slope_of_p1 /= nor_slope_length;

				x_0_of_p1 = x_intersect_real_of_p1 + xnor_slope_of_p1 * radius_check;  // center of an imaginary circle for next check. (body-fixed frame)
				y_0_of_p1 = y_intersect_real_of_p1 + ynor_slope_of_p1 * radius_check;
				z_0_of_p1 = z_intersect_real_of_p1 + znor_slope_of_p1 * radius_check;

				// finding the center of imaginary circle in INERTIAL frame (x_0_inertial, y_0_inertial, z_0_inertial)
				update_rot_spheroid(particle[np].center[0].deter, x_0_inertial_of_p1, y_0_inertial_of_p1, z_0_inertial_of_p1, particle[np].center[0].quater, x_0_of_p1, y_0_of_p1, z_0_of_p1, particle[np].center[0].x, particle[np].center[0].y, particle[np].center[0].z);  // Find x_0 position in inertial frame

				// finding the intersection point of P1 in INERTIAL frame (x_intersect_real_inertial_of_p1, y_intersect_real_inertial_of_p1, z_intersect_real_inertial_of_p1)
				update_rot_spheroid(particle[np].center[0].deter, x_intersect_real_inertial_of_p1, y_intersect_real_inertial_of_p1, z_intersect_real_inertial_of_p1, particle[np].center[0].quater, x_intersect_real_of_p1, y_intersect_real_of_p1, z_intersect_real_of_p1, particle[np].center[0].x, particle[np].center[0].y, particle[np].center[0].z);  // Find x_0 position in inertial frame
				                                                                                                                                                                                                                                                                                                                                            //////////----------------------------------------------------------------------//////////////////////
				                                                                                                                                                                                                                                                                                                                                            // Now we start the process for P2
				xp1_on_p1 = x_intersect_real_inertial_of_p1 - particle[index_p].center[0].x;                                                                                                                                                                                                                                                                // point on P1 in inertial coordinate minus particle P2 center position
				yp1_on_p1 = y_intersect_real_inertial_of_p1 - particle[index_p].center[0].y;
				zp1_on_p1 = z_intersect_real_inertial_of_p1 - particle[index_p].center[0].z;

				x_out_on_p1 = particle[index_p].center[0].quater[1] * (xp1_on_p1) + particle[index_p].center[0].quater[2] * (yp1_on_p1) + particle[index_p].center[0].quater[3] * (zp1_on_p1);  // location of P1 point in body-fixed frame of P2
				y_out_on_p1 = particle[index_p].center[0].quater[4] * (xp1_on_p1) + particle[index_p].center[0].quater[5] * (yp1_on_p1) + particle[index_p].center[0].quater[6] * (zp1_on_p1);
				z_out_on_p1 = particle[index_p].center[0].quater[7] * (xp1_on_p1) + particle[index_p].center[0].quater[8] * (yp1_on_p1) + particle[index_p].center[0].quater[9] * (zp1_on_p1);

				m_slope_p2 = x_out_on_p1 - x_0_of_p2;  // slope of the line conecting the point on P1 and the center of imaginary circle of P2
				n_slope_p2 = y_out_on_p1 - y_0_of_p2;
				p_slope_p2 = z_out_on_p1 - z_0_of_p2;

				///// Intersection of ellipsoid and line, Obtained by MATLAB  ///
				tt_part1 = a * sqrt(SQ(a) * SQ(b) * SQ(n_slope_p2) + SQ(a) * SQ(b) * SQ(p_slope_p2) - SQ(a) * SQ(n_slope_p2) * SQ(z_0_of_p2) + 2 * SQ(a) * n_slope_p2 * p_slope_p2 * y_0_of_p2 * z_0_of_p2 - SQ(a * p_slope_p2 * y_0_of_p2) + SQ(SQ(b)) * SQ(m_slope_p2) - SQ(b * m_slope_p2 * y_0_of_p2) - SQ(b * m_slope_p2 * z_0_of_p2) + 2 * SQ(b) * m_slope_p2 * n_slope_p2 * x_0_of_p2 * y_0_of_p2 + 2 * SQ(b) * m_slope_p2 * p_slope_p2 * x_0_of_p2 * z_0_of_p2 - SQ(b * n_slope_p2 * x_0_of_p2) - SQ(b * p_slope_p2 * x_0_of_p2));
				tt_part2 = SQ(b) * m_slope_p2 * x_0_of_p2 + SQ(a) * n_slope_p2 * y_0_of_p2 + SQ(a) * p_slope_p2 * z_0_of_p2;
				tt_part3 = (SQ(a * n_slope_p2) + SQ(a * p_slope_p2) + SQ(b * m_slope_p2));
				tt_intersect_1 = -(tt_part1 + tt_part2) / tt_part3;  // Answers of equation
				tt_intersect_2 = -(-tt_part1 + tt_part2) / tt_part3;

				x_intersect1_of_p2 = x_0_of_p2 + m_slope_p2 * tt_intersect_1;  // in body fixed frame of P1
				y_intersect1_of_p2 = y_0_of_p2 + n_slope_p2 * tt_intersect_1;
				z_intersect1_of_p2 = z_0_of_p2 + p_slope_p2 * tt_intersect_1;

				x_intersect2_of_p2 = x_0_of_p2 + m_slope_p2 * tt_intersect_2;
				y_intersect2_of_p2 = y_0_of_p2 + n_slope_p2 * tt_intersect_2;
				z_intersect2_of_p2 = z_0_of_p2 + p_slope_p2 * tt_intersect_2;

				///// Find the nearst point
				if ((SQ(x_intersect1_of_p2 - x_out_on_p1) + SQ(y_intersect1_of_p2 - y_out_on_p1) + SQ(z_intersect1_of_p2 - z_out_on_p1)) < (SQ(x_intersect2_of_p2 - x_out_on_p1) + SQ(y_intersect2_of_p2 - y_out_on_p1) + SQ(z_intersect2_of_p2 - z_out_on_p1))) {
					x_intersect_real_of_p2 = x_intersect1_of_p2;  // in body fixed frame
					y_intersect_real_of_p2 = y_intersect1_of_p2;  // in body fixed frame
					z_intersect_real_of_p2 = z_intersect1_of_p2;  // in body fixed frame
				} else {
					x_intersect_real_of_p2 = x_intersect2_of_p2;  // in body fixed frame
					y_intersect_real_of_p2 = y_intersect2_of_p2;  // in body fixed frame
					z_intersect_real_of_p2 = z_intersect2_of_p2;  // in body fixed frame
				}

				xnor_slope_old_of_p2 = xnor_slope_of_p2;
				ynor_slope_old_of_p2 = ynor_slope_of_p2;
				znor_slope_old_of_p2 = znor_slope_of_p2;

				xnor_slope_of_p2 = -2. * x_intersect_real_of_p2 / particle_radius;  // slope of normal vector on spheroid in body_fixed frame. INWARD direction
				ynor_slope_of_p2 = -2. * y_intersect_real_of_p2 / particle_radius_2;
				znor_slope_of_p2 = -2. * z_intersect_real_of_p2 / particle_radius_2;

				nor_slope_length = (sqrt(SQ(xnor_slope_of_p2) + SQ(ynor_slope_of_p2) + SQ(znor_slope_of_p2)));
				xnor_slope_of_p2 /= nor_slope_length;
				ynor_slope_of_p2 /= nor_slope_length;
				znor_slope_of_p2 /= nor_slope_length;

				x_0_of_p2 = x_intersect_real_of_p2 + xnor_slope_of_p2 * radius_check;  // center of an imaginary circle for next check. (body-fixed frame)
				y_0_of_p2 = y_intersect_real_of_p2 + ynor_slope_of_p2 * radius_check;
				z_0_of_p2 = z_intersect_real_of_p2 + znor_slope_of_p2 * radius_check;

				////finding the center of imaginary circle in INERTIAL frame (x_0_inertial, y_0_inertial, z_0_inertial)
				update_rot_spheroid(particle[index_p].center[0].deter, x_0_inertial_of_p2, y_0_inertial_of_p2, z_0_inertial_of_p2, particle[index_p].center[0].quater, x_0_of_p2, y_0_of_p2, z_0_of_p2, particle[index_p].center[0].x, particle[index_p].center[0].y, particle[index_p].center[0].z);  // Find x_0 position in inertial frame

				// finding the intersection point of P2 in INERTIAL frame (x_intersect_real_inertial_of_p2, y_intersect_real_inertial_of_p2, z_intersect_real_inertial_of_p2)
				update_rot_spheroid(particle[index_p].center[0].deter, x_intersect_real_inertial_of_p2, y_intersect_real_inertial_of_p2, z_intersect_real_inertial_of_p2, particle[index_p].center[0].quater, x_intersect_real_of_p2, y_intersect_real_of_p2, z_intersect_real_of_p2, particle[index_p].center[0].x, particle[index_p].center[0].y, particle[index_p].center[0].z);  // Find x_0 position in inertial frame

				xp1_on_p2 = x_intersect_real_inertial_of_p2 - particle[np].center[0].x;
				yp1_on_p2 = y_intersect_real_inertial_of_p2 - particle[np].center[0].y;
				zp1_on_p2 = z_intersect_real_inertial_of_p2 - particle[np].center[0].z;

				mmm++;
			} while (abs(xnor_slope_of_p1 - xnor_slope_old_of_p1) >= slope_th || abs(ynor_slope_of_p1 - ynor_slope_old_of_p1) >= slope_th || abs(znor_slope_of_p1 - znor_slope_old_of_p1) >= slope_th || abs(xnor_slope_of_p2 - xnor_slope_old_of_p2) >= slope_th || abs(ynor_slope_of_p2 - ynor_slope_old_of_p2) >= slope_th || abs(znor_slope_of_p2 - znor_slope_old_of_p2) >= slope_th);

			par_par_dist = sqrt(SQ(x_intersect_real_inertial_of_p1 - x_intersect_real_inertial_of_p2) + SQ(y_intersect_real_inertial_of_p1 - y_intersect_real_inertial_of_p2) + SQ(z_intersect_real_inertial_of_p1 - z_intersect_real_inertial_of_p2));

			if (par_par_dist < thp_control) {
				fi_curvature_p1 = asin(x_intersect_real_of_p1 / (sqrt(SQ(x_intersect_real_of_p1) + SQ(y_intersect_real_of_p1) + SQ(z_intersect_real_of_p1))));
				fi_curvature_p2 = asin(x_intersect_real_of_p2 / (sqrt(SQ(x_intersect_real_of_p2) + SQ(y_intersect_real_of_p2) + SQ(z_intersect_real_of_p2))));

				R_curvature_p1 = SQ(a) * b / (SQ(a * sin(fi_curvature_p1)) + SQ(b * cos(fi_curvature_p1)));  // Paper 1366
				R_curvature_p2 = SQ(a) * b / (SQ(a * sin(fi_curvature_p2)) + SQ(b * cos(fi_curvature_p2)));  // Paper 1366

				x_center_curve_p1 = x_intersect_real_of_p1 + xnor_slope_of_p1 * R_curvature_p1;  // center of circle of curvature. (body-fixed frame)
				y_center_curve_p1 = y_intersect_real_of_p1 + ynor_slope_of_p1 * R_curvature_p1;
				z_center_curve_p1 = z_intersect_real_of_p1 + znor_slope_of_p1 * R_curvature_p1;

				x_center_curve_p2 = x_intersect_real_of_p2 + xnor_slope_of_p2 * R_curvature_p2;  // center of circle of curvature. (body-fixed frame)
				y_center_curve_p2 = y_intersect_real_of_p2 + ynor_slope_of_p2 * R_curvature_p2;
				z_center_curve_p2 = z_intersect_real_of_p2 + znor_slope_of_p2 * R_curvature_p2;

				// finding the center of curvature circle in INERTIAL frame
				update_rot_spheroid(particle[np].center[0].deter, x_center_curve_inertial_p1, y_center_curve_inertial_p1, z_center_curve_inertial_p1, particle[np].center[0].quater, x_center_curve_p1, y_center_curve_p1, z_center_curve_p1, particle[np].center[0].x, particle[np].center[0].y, particle[np].center[0].z);                           // Find  position of circle of curvature in inertial frame
				update_rot_spheroid(particle[index_p].center[0].deter, x_center_curve_inertial_p2, y_center_curve_inertial_p2, z_center_curve_inertial_p2, particle[index_p].center[0].quater, x_center_curve_p2, y_center_curve_p2, z_center_curve_p2, particle[index_p].center[0].x, particle[index_p].center[0].y, particle[index_p].center[0].z);  // Find  position of circle of curvature in inertial frame

				Dij = sqrt(SQ(x_center_curve_inertial_p1 - x_center_curve_inertial_p2) + SQ(y_center_curve_inertial_p1 - y_center_curve_inertial_p2) + SQ(z_center_curve_inertial_p1 - z_center_curve_inertial_p2));

#if defined Spheroid_Collision_3_Feng_Lub_Verlet
				if (par_par_dist <= th_lub && par_par_dist > th) {
					u_approach_x_p1 = particle[np].center[0].vel_x + particle[np].center[0].omgZ * (y_center_curve_inertial_p1 - particle[np].center[0].y) - particle[np].center[0].omgY * (z_center_curve_inertial_p1 - particle[np].center[0].z);
					u_approach_y_p1 = particle[np].center[0].vel_y - particle[np].center[0].omgZ * (x_center_curve_inertial_p1 - particle[np].center[0].x) + particle[np].center[0].omgX * (z_center_curve_inertial_p1 - particle[np].center[0].z);
					u_approach_z_p1 = particle[np].center[0].vel_z - particle[np].center[0].omgX * (y_center_curve_inertial_p1 - particle[np].center[0].y) + particle[np].center[0].omgY * (x_center_curve_inertial_p1 - particle[np].center[0].x);

					u_approach_x_p2 = particle[index_p].center[0].vel_x + particle[np].center[0].omgZ * (y_center_curve_inertial_p2 - particle[index_p].center[0].y) - particle[np].center[0].omgY * (z_center_curve_inertial_p2 - particle[index_p].center[0].z);
					u_approach_y_p2 = particle[index_p].center[0].vel_y - particle[np].center[0].omgZ * (x_center_curve_inertial_p2 - particle[index_p].center[0].x) + particle[np].center[0].omgX * (z_center_curve_inertial_p2 - particle[index_p].center[0].z);
					u_approach_z_p2 = particle[index_p].center[0].vel_z - particle[np].center[0].omgX * (y_center_curve_inertial_p2 - particle[index_p].center[0].y) + particle[np].center[0].omgY * (x_center_curve_inertial_p2 - particle[index_p].center[0].x);

					double dot_pro = (u_approach_x_p1 - u_approach_x_p2) * (x_center_curve_inertial_p1 - x_center_curve_inertial_p2);
					ffx = -6 * M_PI * (1.0) * ((tau - 0.5) / 3.0) * SQ(R_curvature_p1 * R_curvature_p2 / (R_curvature_p1 + R_curvature_p2)) * (1. / (par_par_dist)-1. / th_lub) * dot_pro / (Dij);
					dot_pro = (u_approach_y_p1 - u_approach_y_p2) * (y_center_curve_inertial_p1 - y_center_curve_inertial_p2);
					ffy = -6 * M_PI * (1.0) * ((tau - 0.5) / 3.0) * SQ(R_curvature_p1 * R_curvature_p2 / (R_curvature_p1 + R_curvature_p2)) * (1. / (par_par_dist)-1. / th_lub) * dot_pro / (Dij);
					dot_pro = (u_approach_z_p1 - u_approach_z_p2) * (z_center_curve_inertial_p1 - z_center_curve_inertial_p2);
					ffz = -6 * M_PI * (1.0) * ((tau - 0.5) / 3.0) * SQ(R_curvature_p1 * R_curvature_p2 / (R_curvature_p1 + R_curvature_p2)) * (1. / (par_par_dist)-1. / th_lub) * dot_pro / (Dij);
				}
#endif
				if (par_par_dist <= th && par_par_dist > 0) {
					ffx = cij / ep * ((x_center_curve_inertial_p1 - x_center_curve_inertial_p2)) / Dij * sqr((par_par_dist - th) / th);
					ffy = cij / ep * ((y_center_curve_inertial_p1 - y_center_curve_inertial_p2)) / Dij * sqr((par_par_dist - th) / th);
					ffz = cij / ep * ((z_center_curve_inertial_p1 - z_center_curve_inertial_p2)) / Dij * sqr((par_par_dist - th) / th);
				}
				if (par_par_dist < 0) {
					ffx = (x_center_curve_inertial_p1 - x_center_curve_inertial_p2) / Dij * (cij / ep * sqr((par_par_dist - th) / th) + cij / Ep / th * (-1 * par_par_dist));
					ffy = (y_center_curve_inertial_p1 - y_center_curve_inertial_p2) / Dij * (cij / ep * sqr((par_par_dist - th) / th) + cij / Ep / th * (-1 * par_par_dist));
					ffz = (z_center_curve_inertial_p1 - z_center_curve_inertial_p2) / Dij * (cij / ep * sqr((par_par_dist - th) / th) + cij / Ep / th * (-1 * par_par_dist));
				}

				particle[np].center[0].f3x += ffx;
				particle[np].center[0].f3y += ffy;
				particle[np].center[0].f3z += ffz;

				particle[np].center[0].t3_x += ffy * (z_intersect_real_inertial_of_p1 - particle[np].center[0].z) - ffz * (y_intersect_real_inertial_of_p1 - particle[np].center[0].y);
				particle[np].center[0].t3_y += ffz * (x_intersect_real_inertial_of_p1 - particle[np].center[0].x) - ffx * (z_intersect_real_inertial_of_p1 - particle[np].center[0].z);
				particle[np].center[0].t3_z += ffx * (y_intersect_real_inertial_of_p1 - particle[np].center[0].y) - ffy * (x_intersect_real_inertial_of_p1 - particle[np].center[0].x);

			}  //// if (par_par_dist<thp_control)
		}      ///// if (index_p != np)
	}          // END of Particle-Particle Collision
	return;
}
#endif

//=============================================================================================================================
//=============================================================================================================================
//-----------------------------------------------------------------------------------------------------------------------------
// Subroutine: RUNGEKUTTA
//-----------------------------------------------------------------------------------------------------------------------------
// this function solves 3+4 equations for "3:rotational speed in body-fixed coordinate" + "4:Quaternions:q0,q1,q2,q3"

void Particle_sim::rungekutta(double& omega_prm_x, double& omega_prm_y, double& omega_prm_z,
                              double Tor_prm_x, double Tor_prm_y, double Tor_prm_z,
                              double mom_inr_x, double mom_inr_y, double mom_inr_z, double& Q0, double& Q1, double& Q2, double& Q3) {
	double dh, f, k1, k2, k3, k4;
	double g, r1, r2, r3, r4;
	double h, s1, s2, s3, s4;
	double m, b1, b2, b3, b4;
	double n, c1, c2, c3, c4;
	double p, j1, j2, j3, j4;
	double q, t1, t2, t3, t4;
	double omg_prm_x_1, omg_prm_y_1, omg_prm_z_1;
	double Q0_1, Q1_1, Q2_1, Q3_1;
	double Tx = Tor_prm_x;
	double Ty = Tor_prm_y;
	double Tz = Tor_prm_z;
	double mom_x = mom_inr_x;
	double mom_y = mom_inr_y;
	double mom_z = mom_inr_z;

#define FF(omg_prm_y, omg_prm_z) Tx / mom_x + omg_prm_y* omg_prm_z*(mom_y - mom_z) / mom_x
#define GG(omg_prm_x, omg_prm_z) Ty / mom_y + omg_prm_x* omg_prm_z*(mom_z - mom_x) / mom_y
#define HH(omg_prm_x, omg_prm_y) Tz / mom_z + omg_prm_x* omg_prm_y*(mom_x - mom_y) / mom_z
#define MM(Q1, Q2, Q3, omg_prm_x, omg_prm_y, omg_prm_z) 0.5 * (-Q1 * omg_prm_x - Q2 * omg_prm_y - Q3 * omg_prm_z)
#define NN(Q0, Q2, Q3, omg_prm_x, omg_prm_y, omg_prm_z) 0.5 * (Q0 * omg_prm_x - Q3 * omg_prm_y + Q2 * omg_prm_z)
#define PP(Q0, Q1, Q3, omg_prm_x, omg_prm_y, omg_prm_z) 0.5 * (Q3 * omg_prm_x + Q0 * omg_prm_y - Q1 * omg_prm_z)
#define QQ(Q0, Q1, Q2, omg_prm_x, omg_prm_y, omg_prm_z) 0.5 * (-Q2 * omg_prm_x + Q1 * omg_prm_y + Q0 * omg_prm_z)

	double omg_prm_x_0 = omega_prm_x;  // Initial value
	double omg_prm_y_0 = omega_prm_y;  // Initial value
	double omg_prm_z_0 = omega_prm_z;  // Initial value
	double Q0_0 = Q0;                  // Initial value
	double Q1_0 = Q1;                  // Initial value
	double Q2_0 = Q2;                  // Initial value
	double Q3_0 = Q3;                  // Initial value

	dh = 1;  // dh here must be time step which is in lattice calculations = 1

	// printf("\nEnter the value of last point: ");
	// scanf("%lf",&nn);
	// for(double t=0; t<nn; t=t+dh)
	//{

	f = FF(omg_prm_y_0, omg_prm_z_0);
	k1 = dh * f;
	g = GG(omg_prm_x_0, omg_prm_z_0);
	r1 = dh * g;
	h = HH(omg_prm_x_0, omg_prm_y_0);
	s1 = dh * h;
	m = MM(Q1_0, Q2_0, Q3_0, omg_prm_x_0, omg_prm_y_0, omg_prm_z_0);
	b1 = dh * m;
	n = NN(Q0_0, Q2_0, Q3_0, omg_prm_x_0, omg_prm_y_0, omg_prm_z_0);
	c1 = dh * n;
	p = PP(Q0_0, Q1_0, Q3_0, omg_prm_x_0, omg_prm_y_0, omg_prm_z_0);
	j1 = dh * p;
	q = QQ(Q0_0, Q1_0, Q2_0, omg_prm_x_0, omg_prm_y_0, omg_prm_z_0);
	t1 = dh * q;

	f = FF((omg_prm_y_0 + r1 / 2), (omg_prm_z_0 + s1 / 2));
	k2 = dh * f;
	g = GG((omg_prm_x_0 + k1 / 2), (omg_prm_z_0 + s1 / 2));
	r2 = dh * g;
	h = HH((omg_prm_x_0 + k1 / 2), (omg_prm_y_0 + r1 / 2));
	s2 = dh * h;
	m = MM((Q1_0 + c1 / 2), (Q2_0 + j1 / 2), (Q3_0 + t1 / 2), (omg_prm_x_0 + k1 / 2), (omg_prm_y_0 + r1 / 2), (omg_prm_z_0 + s1 / 2));
	b2 = dh * m;
	n = NN((Q0_0 + b1 / 2), (Q2_0 + j1 / 2), (Q3_0 + t1 / 2), (omg_prm_x_0 + k1 / 2), (omg_prm_y_0 + r1 / 2), (omg_prm_z_0 + s1 / 2));
	c2 = dh * n;
	p = PP((Q0_0 + b1 / 2), (Q1_0 + c1 / 2), (Q3_0 + t1 / 2), (omg_prm_x_0 + k1 / 2), (omg_prm_y_0 + r1 / 2), (omg_prm_z_0 + s1 / 2));
	j2 = dh * p;
	q = QQ((Q0_0 + b1 / 2), (Q1_0 + c1 / 2), (Q2_0 + j1 / 2), (omg_prm_x_0 + k1 / 2), (omg_prm_y_0 + r1 / 2), (omg_prm_z_0 + s1 / 2));
	t2 = dh * q;

	f = FF((omg_prm_y_0 + r2 / 2), (omg_prm_z_0 + s2 / 2));
	k3 = dh * f;
	g = GG((omg_prm_x_0 + k2 / 2), (omg_prm_z_0 + s2 / 2));
	r3 = dh * g;
	h = HH((omg_prm_x_0 + k2 / 2), (omg_prm_y_0 + r2 / 2));
	s3 = dh * h;
	m = MM((Q1_0 + c2 / 2), (Q2_0 + j2 / 2), (Q3_0 + t2 / 2), (omg_prm_x_0 + k2 / 2), (omg_prm_y_0 + r2 / 2), (omg_prm_z_0 + s2 / 2));
	b3 = dh * m;
	n = NN((Q0_0 + b2 / 2), (Q2_0 + j2 / 2), (Q3_0 + t2 / 2), (omg_prm_x_0 + k2 / 2), (omg_prm_y_0 + r2 / 2), (omg_prm_z_0 + s2 / 2));
	c3 = dh * n;
	p = PP((Q0_0 + b2 / 2), (Q1_0 + c2 / 2), (Q3_0 + t2 / 2), (omg_prm_x_0 + k2 / 2), (omg_prm_y_0 + r2 / 2), (omg_prm_z_0 + s2 / 2));
	j3 = dh * p;
	q = QQ((Q0_0 + b2 / 2), (Q1_0 + c2 / 2), (Q2_0 + j2 / 2), (omg_prm_x_0 + k2 / 2), (omg_prm_y_0 + r2 / 2), (omg_prm_z_0 + s2 / 2));
	t3 = dh * q;

	f = FF((omg_prm_y_0 + r3), (omg_prm_z_0 + s3));
	k4 = dh * f;
	g = GG((omg_prm_x_0 + k3), (omg_prm_z_0 + s3));
	r4 = dh * g;
	h = HH((omg_prm_x_0 + k3), (omg_prm_y_0 + r3));
	s4 = dh * h;
	m = MM((Q1_0 + c3), (Q2_0 + j3), (Q3_0 + t3), (omg_prm_x_0 + k3), (omg_prm_y_0 + r3), (omg_prm_z_0 + s3));
	b4 = dh * m;
	n = NN((Q0_0 + b3), (Q2_0 + j3), (Q3_0 + t3), (omg_prm_x_0 + k3), (omg_prm_y_0 + r3), (omg_prm_z_0 + s3));
	c4 = dh * n;
	p = PP((Q0_0 + b3), (Q1_0 + c3), (Q3_0 + t3), (omg_prm_x_0 + k3), (omg_prm_y_0 + r3), (omg_prm_z_0 + s3));
	j4 = dh * p;
	q = QQ((Q0_0 + b3), (Q1_0 + c3), (Q2_0 + j3), (omg_prm_x_0 + k3), (omg_prm_y_0 + r3), (omg_prm_z_0 + s3));
	t4 = dh * q;

	omg_prm_x_1 = omg_prm_x_0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6.;
	omg_prm_y_1 = omg_prm_y_0 + (r1 + 2 * r2 + 2 * r3 + r4) / 6.;
	omg_prm_z_1 = omg_prm_z_0 + (s1 + 2 * s2 + 2 * s3 + s4) / 6.;
	Q0_1 = Q0_0 + (b1 + 2 * b2 + 2 * b3 + b4) / 6.;
	Q1_1 = Q1_0 + (c1 + 2 * c2 + 2 * c3 + c4) / 6.;
	Q2_1 = Q2_0 + (j1 + 2 * j2 + 2 * j3 + j4) / 6.;
	Q3_1 = Q3_0 + (t1 + 2 * t2 + 2 * t3 + t4) / 6.;

	omg_prm_x_0 = omg_prm_x_1;
	omg_prm_y_0 = omg_prm_y_1;
	omg_prm_z_0 = omg_prm_z_1;
	Q0_0 = Q0_1;
	Q1_0 = Q1_1;
	Q2_0 = Q2_1;
	Q3_0 = Q3_1;
	// }
	omega_prm_x = omg_prm_x_1;
	omega_prm_y = omg_prm_y_1;
	omega_prm_z = omg_prm_z_1;
	Q0 = Q0_1;
	Q1 = Q1_1;
	Q2 = Q2_1;
	Q3 = Q3_1;
	return;
}

//=============================================================================================================================
//=============================================================================================================================
//-----------------------------------------------------------------------------------------------------------------------------
// Subroutine: QUATERNION
//-----------------------------------------------------------------------------------------------------------------------------
//
// This function calculates the elements of quaternion matrix.(Eq.2.13 Article 876: H.Huang et al. J. Fluid Mech. (2012), vol. 692, pp. 369-394.
void Particle_sim::quaternion(double Q0, double Q1, double Q2, double Q3, int np, double quat[]) {
	quat[0] = 0.;  // this value is useless
	quat[1] = SQ(Q0) + SQ(Q1) - SQ(Q2) - SQ(Q3);
	quat[2] = 2 * (Q1 * Q2 + Q0 * Q3);
	quat[3] = 2 * (Q1 * Q3 - Q0 * Q2);
	quat[4] = 2 * (Q1 * Q2 - Q0 * Q3);
	quat[5] = SQ(Q0) - SQ(Q1) + SQ(Q2) - SQ(Q3);
	quat[6] = 2 * (Q2 * Q3 + Q0 * Q1);
	quat[7] = 2 * (Q1 * Q3 + Q0 * Q2);
	quat[8] = 2 * (Q2 * Q3 - Q0 * Q1);
	quat[9] = SQ(Q0) - SQ(Q1) - SQ(Q2) + SQ(Q3);

	return;
}

//=============================================================================================================================
//=============================================================================================================================
//-----------------------------------------------------------------------------------------------------------------------------
// Subroutine: DETERMINANT
//-----------------------------------------------------------------------------------------------------------------------------
// This function calculates the determinant of quaternion matrix (in order to obtain its inverse)

double Particle_sim::determinant(int np, double quat[]) {
	double det;
	det = quat[1] * (quat[5] * quat[9] - quat[6] * quat[8])
	      - quat[2] * (quat[4] * quat[9] - quat[6] * quat[7])
	      + quat[3] * (quat[4] * quat[8] - quat[5] * quat[7]);
	return det;
}
//=============================================================================================================================
//=============================================================================================================================
//-----------------------------------------------------------------------------------------------------------------------------
// Subroutine: COMP_ROT_SPEED
//-----------------------------------------------------------------------------------------------------------------------------
// (This function transforms the rotational speed from body-fixed coordinate to Inertial coordinate
// This function is just for SPHEROID
void Particle_sim::comp_rot_speed(double determi, double& omgX, double& omgY, double& omgZ, int n, double quat[], double omg_pr_x, double omg_pr_y, double omg_pr_z) {
	double RNI[10];
	double om_temp_x = omg_pr_x;
	double om_temp_y = omg_pr_y;
	double om_temp_z = omg_pr_z;

	RNI[0] = 0.;
	RNI[1] = quat[5] * quat[9] - quat[6] * quat[8];
	RNI[2] = -quat[2] * quat[9] + quat[3] * quat[8];
	RNI[3] = quat[2] * quat[6] - quat[3] * quat[5];
	RNI[4] = -quat[4] * quat[9] + quat[6] * quat[7];
	RNI[5] = quat[1] * quat[9] - quat[3] * quat[7];
	RNI[6] = -quat[1] * quat[6] + quat[3] * quat[4];
	RNI[7] = quat[4] * quat[8] - quat[5] * quat[7];
	RNI[8] = -quat[1] * quat[8] + quat[2] * quat[7];
	RNI[9] = quat[1] * quat[5] - quat[2] * quat[4];
	RNI[1] = RNI[1] / determi;
	RNI[2] = RNI[2] / determi;
	RNI[3] = RNI[3] / determi;
	RNI[4] = RNI[4] / determi;
	RNI[5] = RNI[5] / determi;
	RNI[6] = RNI[6] / determi;
	RNI[7] = RNI[7] / determi;
	RNI[8] = RNI[8] / determi;
	RNI[9] = RNI[9] / determi;

	omgX = RNI[1] * om_temp_x + RNI[2] * om_temp_y + RNI[3] * om_temp_z;
	omgY = RNI[4] * om_temp_x + RNI[5] * om_temp_y + RNI[6] * om_temp_z;
	omgZ = RNI[7] * om_temp_x + RNI[8] * om_temp_y + RNI[9] * om_temp_z;

	return;
}
//=============================================================================================================================
//=============================================================================================================================
//-----------------------------------------------------------------------------------------------------------------------------
// Subroutine: UPDATE_ROT_SPHEROID
//-----------------------------------------------------------------------------------------------------------------------------
// If at t=0 Euler angles are not zero then the spheroid must be rotated. This function is part of caculating the Lagrangian node position in INERTIAL coordinate based on Euler angles and node location in BODY coordinate
// This function is just for SPHEROID
void Particle_sim::update_rot_spheroid(double determi, double& X_inertial, double& Y_inertial, double& Z_inertial, double quat[], double X_body, double Y_body, double Z_body, double center_x, double center_y, double center_z) {
	// X_body is node X position in body coordinate minus particle center
	// X_inertial is node X position in INERTIAL coordinate

	double RNI[10];

	RNI[0] = 0.;
	RNI[1] = quat[5] * quat[9] - quat[6] * quat[8];
	RNI[2] = -quat[2] * quat[9] + quat[3] * quat[8];
	RNI[3] = quat[2] * quat[6] - quat[3] * quat[5];
	RNI[4] = -quat[4] * quat[9] + quat[6] * quat[7];
	RNI[5] = quat[1] * quat[9] - quat[3] * quat[7];
	RNI[6] = -quat[1] * quat[6] + quat[3] * quat[4];
	RNI[7] = quat[4] * quat[8] - quat[5] * quat[7];
	RNI[8] = -quat[1] * quat[8] + quat[2] * quat[7];
	RNI[9] = quat[1] * quat[5] - quat[2] * quat[4];
	RNI[1] = RNI[1] / determi;
	RNI[2] = RNI[2] / determi;
	RNI[3] = RNI[3] / determi;
	RNI[4] = RNI[4] / determi;
	RNI[5] = RNI[5] / determi;
	RNI[6] = RNI[6] / determi;
	RNI[7] = RNI[7] / determi;
	RNI[8] = RNI[8] / determi;
	RNI[9] = RNI[9] / determi;

	X_inertial = (RNI[1] * X_body + RNI[2] * Y_body + RNI[3] * Z_body) + center_x;
	Y_inertial = (RNI[4] * X_body + RNI[5] * Y_body + RNI[6] * Z_body) + center_y;
	Z_inertial = (RNI[7] * X_body + RNI[8] * Y_body + RNI[9] * Z_body) + center_z;

	return;
}
//=============================================================================================================================
//=============================================================================================================================
//-----------------------------------------------------------------------------------------------------------------------------
// Subroutine: NORMAL_VEC_SPHEROID
//-----------------------------------------------------------------------------------------------------------------------------
// Calculates NORMAL vector at each node at INERTIAL coordinate based on the body coordinate valuse
void Particle_sim::normal_vec_spheroid(double determi, double& X_inertial, double& Y_inertial, double& Z_inertial, int n, double quat[], double X_body, double Y_body, double Z_body) {
	// X_body is normal vector X index in body coordinate
	// X_inertial normal vector X index in INERTIAL coordinate

	double RNI[10];

	RNI[0] = 0.;
	RNI[1] = quat[5] * quat[9] - quat[6] * quat[8];
	RNI[2] = -quat[2] * quat[9] + quat[3] * quat[8];
	RNI[3] = quat[2] * quat[6] - quat[3] * quat[5];
	RNI[4] = -quat[4] * quat[9] + quat[6] * quat[7];
	RNI[5] = quat[1] * quat[9] - quat[3] * quat[7];
	RNI[6] = -quat[1] * quat[6] + quat[3] * quat[4];
	RNI[7] = quat[4] * quat[8] - quat[5] * quat[7];
	RNI[8] = -quat[1] * quat[8] + quat[2] * quat[7];
	RNI[9] = quat[1] * quat[5] - quat[2] * quat[4];
	RNI[1] = RNI[1] / determi;
	RNI[2] = RNI[2] / determi;
	RNI[3] = RNI[3] / determi;
	RNI[4] = RNI[4] / determi;
	RNI[5] = RNI[5] / determi;
	RNI[6] = RNI[6] / determi;
	RNI[7] = RNI[7] / determi;
	RNI[8] = RNI[8] / determi;
	RNI[9] = RNI[9] / determi;

	X_inertial = (RNI[1] * X_body + RNI[2] * Y_body + RNI[3] * Z_body);
	Y_inertial = (RNI[4] * X_body + RNI[5] * Y_body + RNI[6] * Z_body);
	Z_inertial = (RNI[7] * X_body + RNI[8] * Y_body + RNI[9] * Z_body);

	return;
}

//=============================================================================================================================
//=============================================================================================================================
//-----------------------------------------------------------------------------------------------------------------------------
// Subroutine: WRITE_PARTICLE_VTK
//-----------------------------------------------------------------------------------------------------------------------------
//
// The particle state (node positions) is writen to a VTK file at each t_vtk step and can be read by ParaView

void Particle_sim::write_particle_vtk(const std::string& base_path, int tm, particle_struct particle, int npp) {
#if defined VTK_ASCII || defined VTK_BINARY
	/// Create filename
	stringstream output_filename;
	output_filename << base_path << "_p" << npp + 1 << "_t" << tm << ".vtk";

	/// open file
	ofstream output_file(output_filename.str().c_str());

	/// Write VTK header
	output_file << "# vtk DataFile Version 3.0\n";
	output_file << "particle_state\n";
	output_file << "ASCII\n";
	output_file << "DATASET POLYDATA\n";

	/// Write node positions
	output_file << "POINTS " << particle_num_nodes << " float\n";
	for (int n = 0; n < particle_num_nodes; ++n) {
		output_file << xbegin + particle.node[n].x * global_parameters.D_x << " " << ybegin + particle.node[n].y * global_parameters.D_x << " " << zbegin + particle.node[n].z * global_parameters.D_x << "\n";
	}

#if defined MOVING_SPHERE || defined STATIONARY_SPHERE
	output_file << "VERTICES 1 " << particle_num_nodes + 1 << "\n";
	output_file << particle_num_nodes << "\n";
	for (int n = 0; n < particle_num_nodes; ++n) {
		output_file << n << "\n";
	}

	const string output_filename1 = base_path + "_center_p" + std::to_string(npp + 1) + ".vtk";
	//	string output_filename1 << "Alborz_Results/vtk_particle/particle_center_" << npp + 1 << ".vtk";
	// Below part is useful for SPHERE to create surface using Glyph for canter in Paraview
	// stringstream output_filename1;
	// output_filename1 << "Alborz_Results/vtk_particle/particle_center_" << npp + 1 << "_t" << time << ".vtk";
	// output_filename1 << "Alborz_Results/vtk_particle/particle_center_" << ".vtk";

	/// Open file
	ofstream output_file1(output_filename1.c_str(), fstream::app);
	/// Write VTK header
	// output_file1 << "# vtk DataFile Version 3.0\n";
	// output_file1 << "particle_state\n";
	// output_file1 << "ASCII\n";
	// output_file1 << "DATASET POLYDATA\n";

	//	output_file1 << "POINTS " << "2" << " float\n";
	// output_file1 << particle.center[0].x << "	" << particle.center[0].y << "	" << particle.center[0].z << "\t";
	// output_file1 << particle.center[0].vel_x<< "	" << particle.center[0].vel_y << "	" << particle.center[0].vel_z<< "\n";
	output_file1 << std::left << std::setw(25) << particle.center[0].x << "\t";  // center position (x-component)
	output_file1 << std::left << std::setw(25) << particle.center[0].y << "\t";  // center position (y-component)
	output_file1 << std::left << std::setw(25) << particle.center[0].z << "\t";  // center position (y-component)

	output_file1 << std::left << std::setw(25) << particle.center[0].vel_x << "\t";  // center velocity (x-component)
	output_file1 << std::left << std::setw(25) << particle.center[0].vel_y << "\t";  // center velocity (y-component)
	output_file1 << std::left << std::setw(25) << particle.center[0].vel_z << "\n";  // center velocity (z-component)
	/// Close file
	output_file1.close();
#endif
#if defined MOVING_SPHEROID || defined STATIONARY_SPHEROID
	// Create triangle surfaces
	output_file << "POLYGONS " << number_triangles << " " << 4 * number_triangles << "\n";
	for (int n = 0; n < number_triangles; ++n) {
		output_file << "3 " << triangleset1[n] << " " << triangleset2[n] << " " << triangleset3[n] << "\n";
	}
#endif
	/// Close file
	output_file.close();
#endif
// this feature was not updated and probably needs needs some work
#if defined CASE
	unsigned _oldExponentFormat = _set_output_format(_TWO_DIGIT_EXPONENT);  // number are shown as 2.0e02 instead of 2.0e002 (necessay for geo and data files)
	                                                                        /////////// Writing CASE file (.case) ///////////////////
	                                                                        /////////////////////////////////////////////////////////
	                                                                        /// Create filename
	stringstream output_filename;
	output_filename << "Alborz_Results/vtk_particle/particle_" << npp + 1 << ".case";
	ofstream output_file;

	/// Open file
	output_file.open(output_filename.str().c_str());

	/// Write VTK header
	output_file << "# EnSight Gold Model\n";
	output_file << "# Created by ALBORZ\n\n";
	output_file << "FORMAT\n";
	output_file << "type:                   ensight gold\n\n";
	output_file << "GEOMETRY\n";
	output_file << "model:                 1 particle_" << npp + 1 << "_******.geo\n\n";
	output_file << "TIME\n";
	output_file << "time set:               1\n";
	output_file << "number of steps:        " << t_num / t_vtk << "\n";
	output_file << "filename start number:  " << t_vtk << "\n";
	output_file << "filename increment:     " << t_vtk << "\n";
	output_file << "time values:\n";
	for (int itm = t_vtk; itm <= t_num; itm = itm + t_vtk) {
		output_file << itm << "\n";
	}
	/// Close file
	output_file.close();
	/////////// End of Writing CASE file (.case) ///////////////
	////////////////////////////////////////////////////////////

	/////////// Writing GEOMETRY file (.geo) ///////////////////
	////////////////////////////////////////////////////////////

	double min_node_x, min_node_y, min_node_z;
	double max_node_x, max_node_y, max_node_z;
	min_node_x = particle.node[0].x;
	min_node_y = particle.node[0].y;
	max_node_x = particle.node[0].x;
	max_node_y = particle.node[0].y;
	max_node_y = particle.node[0].y;
	max_node_z = particle.node[0].z;
	for (int n = 0; n < particle_num_nodes; ++n) {
		if (particle.node[n].x < min_node_x) {
			min_node_x = particle.node[n].x;
		}
		if (particle.node[n].y < min_node_y) {
			min_node_y = particle.node[n].y;
		}
		if (particle.node[n].x > max_node_x) {
			max_node_x = particle.node[n].x;
		}
		if (particle.node[n].y > max_node_y) {
			max_node_y = particle.node[n].y;
		}
		if (particle.node[n].y > max_node_y) {
			max_node_y = particle.node[n].y;
		}
		if (particle.node[n].z > max_node_z) {
			max_node_z = particle.node[n].z;
		}
	}
	stringstream output_filename1;
	output_filename1 << "Alborz_Results/vtk_particle/particle_" << npp + 1 << "_" << setfill('0') << setw(6) << time << ".geo";  // Geometry file
	ofstream output_file1;

	/// Open file
	output_file1.open(output_filename1.str().c_str());

	/// Write VTK header
	output_file1 << "EnSight Model Geometry File\n";
	output_file1 << "EnSight 10.1.1\n";
	output_file1 << "node id given\n";
	output_file1 << "element id given\n";
	output_file1 << "extents\n";
	output_file1 << std::scientific << std::setw(12) << std::setprecision(5) << min_node_x << std::scientific << std::setw(12) << std::setprecision(5) << max_node_x << "\n";
	output_file1 << std::scientific << std::setw(12) << std::setprecision(5) << min_node_y << std::scientific << std::setw(12) << std::setprecision(5) << max_node_y << "\n";
	output_file1 << std::scientific << std::setw(12) << std::setprecision(5) << min_node_z << std::scientific << std::setw(12) << std::setprecision(5) << max_node_z << "\n";
	output_file1 << "part\n";
	output_file1 << std::setw(10) << 1 << "\n";
	output_file1 << "Particle" << npp + 1 << "\n";
	output_file1 << "coordinates\n";
	output_file1 << std::setw(10) << particle_num_nodes << "\n";
	for (int itm = 1; itm <= particle_num_nodes; itm++) {
		output_file1 << std::setw(10) << itm << "\n";
	}

	for (int i = 0; i < particle_num_nodes; i++) {
		output_file1 << std::scientific << std::setw(12) << std::setprecision(5) << particle.node[i].x << "\n";
	}
	for (int i = 0; i < particle_num_nodes; i++) {
		output_file1 << std::scientific << std::setw(12) << std::setprecision(5) << particle.node[i].y << "\n";
	}
	for (int i = 0; i < particle_num_nodes; i++) {
		output_file1 << std::scientific << std::setw(12) << std::setprecision(5) << particle.node[i].z << "\n";
	}

#if defined MOVING_SPHEROID || defined STATIONARY_SPHEROID
	output_file1 << "tria3\n";
	output_file1 << std::setw(10) << number_triangles << "\n";
	for (int itm = 1; itm <= number_triangles; itm++) {
		output_file1 << std::setw(10) << itm << "\n";
	}

	// Create triangle surfaces
	for (int n = 0; n < number_triangles; ++n) {
		output_file1 << std::setw(10) << triangleset1[n] + 1 << std::setw(10) << triangleset2[n] + 1 << std::setw(10) << triangleset3[n] + 1 << "\n";
	}
#endif

	/// Close file
	output_file1.close();

	/////////// End of Writing GEOMETRY file (.geo) ///////////////////
	///////////////////////////////////////////////////////////////////

#endif  // End of CASE file for Ensight

	return;
}

//=============================================================================================================================
//=============================================================================================================================
//-----------------------------------------------------------------------------------------------------------------------------
// Subroutine: WRITE_PARTICLE_DATA
//-----------------------------------------------------------------------------------------------------------------------------
//
// The following PARTICLE quantities are written to the disk at each t_data step:
// - drag and lift forces (x- and y-components of the force)
// - object center position (x- and y-components)
// - object center velocity (x- and y-components)
// The data file is readable by gnuplot
void Particle_sim::write_particle_data(const std::string& base_path, int tm, particle_struct particle, int npp) {
	/// Open file
	const string output_filename = base_path + "_p" + std::to_string(npp) + "_t" + std::to_string(tm) + ".dat";
	ofstream output_file(output_filename.c_str(), fstream::app);

	/// Write data
	output_file << std::left << std::setw(25) << tm << "\t";                          // time step
	output_file << std::left << std::setw(25) << particle.center[0].fx_surf << "\t";  // drag force
	output_file << std::left << std::setw(25) << particle.center[0].fy_surf << "\t";  // lift force
	output_file << std::left << std::setw(25) << particle.center[0].fz_surf << "\t";

	output_file << std::left << std::setw(25) << particle.center[0].x << "\t";  // center position (x-component)
	output_file << std::left << std::setw(25) << particle.center[0].y << "\t";  // center position (y-component)
	output_file << std::left << std::setw(25) << particle.center[0].z << "\t";  // center position (y-component)

	output_file << std::left << std::setw(25) << particle.center[0].vel_x << "\t";  // center velocity (x-component)
	output_file << std::left << std::setw(25) << particle.center[0].vel_y << "\t";  // center velocity (y-component)
	output_file << std::left << std::setw(25) << particle.center[0].vel_z << "\t";  // center velocity (z-component)

	output_file << std::left << std::setw(25) << particle.center[0].omgX << "\t";  // Rotational speed
	output_file << std::left << std::setw(25) << particle.center[0].omgY << "\t";  // Rotational speed
	output_file << std::left << std::setw(25) << particle.center[0].omgZ << "\t";  // Rotational speed

	output_file << std::left << std::setw(25) << particle.center[0].tetaX << "\t";
	output_file << std::left << std::setw(25) << particle.center[0].tetaY << "\t";
	output_file << std::left << std::setw(25) << particle.center[0].tetaZ << "\t";
#if defined Flow_Without_Thermal_Effect
	output_file << "\n";
#endif
#if defined Flow_With_Thermal_Effect
	output_file << particle.center[0].temperature << "\n";  // center velocity (y-component)
#endif
	                                                        /// Close file
	output_file.close();

	return;
}

//=============================================================================================================================
//=============================================================================================================================
//-----------------------------------------------------------------------------------------------------------------------------
// Subroutine: WRITE_PARTICLE_RECOVERY
//-----------------------------------------------------------------------------------------------------------------------------
template <typename T>
static void write_binary(std::ofstream& stream, const T& val) {
	stream.write(reinterpret_cast<const char*>(&val), sizeof(T));
}

void Particle_sim::write_particle_recovery(const std::string& base_path, int tm, particle_struct particle, int npn) {
	/// Open file
	const string output_filename = make_recovery_name(base_path, tm, npn);
	ofstream output_file(output_filename.c_str(), ofstream::binary);

	write_binary(output_file, particle_mass);
	write_binary(output_file, mom_inertia_x);
	write_binary(output_file, mom_inertia_y);
	write_binary(output_file, mom_inertia_z);
	write_binary(output_file, particle_num_nodes);
	write_binary(output_file, particle.center[0].x);      // center position (x-component)
	write_binary(output_file, particle.center[0].y);      // center position (y-component)
	write_binary(output_file, particle.center[0].z);      // center position (y-component)
	write_binary(output_file, particle.center[0].vel_x);  // center velocity (x-component)
	write_binary(output_file, particle.center[0].vel_y);  // center velocity (y-component)
	write_binary(output_file, particle.center[0].vel_z);  // center velocity (y-component)
	write_binary(output_file, particle.center[0].omgX);   // Rotational speed
	write_binary(output_file, particle.center[0].omgY);   // Rotational speed
	write_binary(output_file, particle.center[0].omgZ);   // Rotational speed
	write_binary(output_file, particle.center[0].tetaX);  // Angle with respect to Horizontal line
	write_binary(output_file, particle.center[0].tetaY);  // Angle with respect to Horizontal line
	write_binary(output_file, particle.center[0].tetaZ);  // Angle with respect to Horizontal line
	write_binary(output_file, particle.center[0].previous_vel_x);
	write_binary(output_file, particle.center[0].previous_vel_y);
	write_binary(output_file, particle.center[0].previous_vel_z);
	write_binary(output_file, particle.center[0].tempx[0]);
	write_binary(output_file, particle.center[0].tempx[1]);
	write_binary(output_file, particle.center[0].tempy[0]);
	write_binary(output_file, particle.center[0].tempy[1]);
	write_binary(output_file, particle.center[0].tempz[0]);
	write_binary(output_file, particle.center[0].tempz[1]);
	write_binary(output_file, particle.center[0].temp_omgX[0]);
	write_binary(output_file, particle.center[0].temp_omgX[1]);
	write_binary(output_file, particle.center[0].temp_omgY[0]);
	write_binary(output_file, particle.center[0].temp_omgY[1]);
	write_binary(output_file, particle.center[0].temp_omgZ[0]);
	write_binary(output_file, particle.center[0].temp_omgZ[1]);
	write_binary(output_file, particle.center[0].Omg_prime_X);
	write_binary(output_file, particle.center[0].Omg_prime_Y);
	write_binary(output_file, particle.center[0].Omg_prime_Z);
	write_binary(output_file, particle.center[0].Qu0);
	write_binary(output_file, particle.center[0].Qu1);
	write_binary(output_file, particle.center[0].Qu2);
	write_binary(output_file, particle.center[0].Qu3);

	write_binary(output_file, particle.center[0].temperature);  // Temperature
	write_binary(output_file, particle.center[0].temp_temperat[0]);
	write_binary(output_file, particle.center[0].temp_temperat[1]);
	write_binary(output_file, particle.center[0].temp_part);

	for (int np1 = 0; np1 < particle_num_nodes; ++np1) {
		write_binary(output_file, particle.node[np1].x);
		write_binary(output_file, particle.node[np1].y);
		write_binary(output_file, particle.node[np1].z);
#if defined MOVING_SPHEROID || defined STATIONARY_SPHEROID
		write_binary(output_file, particle.node[np1].x_pos_1);
		write_binary(output_file, particle.node[np1].y_pos_1);
		write_binary(output_file, particle.node[np1].z_pos_1);
#endif
		write_binary(output_file, area[np1]);
	}
}

//=============================================================================================================================
//=============================================================================================================================
//-----------------------------------------------------------------------------------------------------------------------------
// Subroutine: READ_PARTICLE_RECOVERY
//-----------------------------------------------------------------------------------------------------------------------------
template <typename T>
static void read_binary(std::ifstream& stream, T& val) {
	stream.read(reinterpret_cast<char*>(&val), sizeof(T));
}

void Particle_sim::read_particle_recovery(const std::string& base_path, int tm, particle_struct particle[]) {
	for (int npn = 0; npn < num_particles; ++npn) {
		const std::string input_filename = make_recovery_name(base_path, tm, npn);
		ifstream input_file(input_filename.c_str());

		/// Read particle data
		read_binary(input_file, particle_mass);
		read_binary(input_file, mom_inertia_x);
		read_binary(input_file, mom_inertia_y);
		read_binary(input_file, mom_inertia_z);
		read_binary(input_file, particle_num_nodes);

		/// Read data
		read_binary(input_file, particle[npn].center[0].x);      // center position (x-component)
		read_binary(input_file, particle[npn].center[0].y);      // center position (y-component)
		read_binary(input_file, particle[npn].center[0].z);      // center position (y-component)
		read_binary(input_file, particle[npn].center[0].vel_x);  // center velocity (x-component)
		read_binary(input_file, particle[npn].center[0].vel_y);  // center velocity (y-component)
		read_binary(input_file, particle[npn].center[0].vel_z);  // center velocity (y-component)
		read_binary(input_file, particle[npn].center[0].omgX);   // Rotational speed
		read_binary(input_file, particle[npn].center[0].omgY);   // Rotational speed
		read_binary(input_file, particle[npn].center[0].omgZ);   // Rotational speed
		read_binary(input_file, particle[npn].center[0].tetaX);  // Angle
		read_binary(input_file, particle[npn].center[0].tetaY);  // Angle
		read_binary(input_file, particle[npn].center[0].tetaZ);  // Angle
		read_binary(input_file, particle[npn].center[0].previous_vel_x);
		read_binary(input_file, particle[npn].center[0].previous_vel_y);
		read_binary(input_file, particle[npn].center[0].previous_vel_z);
		read_binary(input_file, particle[npn].center[0].tempx[0]);
		read_binary(input_file, particle[npn].center[0].tempx[1]);
		read_binary(input_file, particle[npn].center[0].tempy[0]);
		read_binary(input_file, particle[npn].center[0].tempy[1]);
		read_binary(input_file, particle[npn].center[0].tempz[0]);
		read_binary(input_file, particle[npn].center[0].tempz[1]);
		read_binary(input_file, particle[npn].center[0].temp_omgX[0]);
		read_binary(input_file, particle[npn].center[0].temp_omgX[1]);
		read_binary(input_file, particle[npn].center[0].temp_omgY[0]);
		read_binary(input_file, particle[npn].center[0].temp_omgY[1]);
		read_binary(input_file, particle[npn].center[0].temp_omgZ[0]);
		read_binary(input_file, particle[npn].center[0].temp_omgZ[1]);
		read_binary(input_file, particle[npn].center[0].Omg_prime_X);
		read_binary(input_file, particle[npn].center[0].Omg_prime_Y);
		read_binary(input_file, particle[npn].center[0].Omg_prime_Z);
		read_binary(input_file, particle[npn].center[0].Qu0);
		read_binary(input_file, particle[npn].center[0].Qu1);
		read_binary(input_file, particle[npn].center[0].Qu2);
		read_binary(input_file, particle[npn].center[0].Qu3);

		read_binary(input_file, particle[npn].center[0].temperature);  // Angle with respect to Horizontal line
		read_binary(input_file, particle[npn].center[0].temp_temperat[0]);
		read_binary(input_file, particle[npn].center[0].temp_temperat[1]);
		read_binary(input_file, particle[npn].center[0].temp_part);

		for (int np1 = 0; np1 < particle_num_nodes; ++np1) {
			read_binary(input_file, particle[npn].node[np1].x);
			read_binary(input_file, particle[npn].node[np1].y);
			read_binary(input_file, particle[npn].node[np1].z);
#if defined MOVING_SPHEROID || defined STATIONARY_SPHEROID
			read_binary(input_file, particle[npn].node[np1].x_pos_1);
			read_binary(input_file, particle[npn].node[np1].y_pos_1);
			read_binary(input_file, particle[npn].node[np1].z_pos_1);
#endif
			read_binary(input_file, area[np1]);
		}

		/// Close particle recovery data file
		input_file.close();
	}
	return;
}
//=============================================================================================================================
//=============================================================================================================================
//-----------------------------------------------------------------------------------------------------------------------------
// Subroutine: WRITE_PARTICLE_PARAMETERS
//-----------------------------------------------------------------------------------------------------------------------------
void Particle_sim::write_particle_parameters(Parallel_MPI* MPI_parallel) {
	if (MPI_parallel->processor_id == MASTER) {
		std::cout << "\nParticle Data" << endl;
		std::cout << "=====================" << endl;

		/// Create filename
		string output_filename("Alborz_Results/particle_parameters");
		output_filename += ".dat";
		ofstream output_file;

		/// Open file
		output_file.open(output_filename.c_str(), fstream::app);  // write to end of file

		output_file << "--------------------PARTICLES--------------------------" << endl;

#if defined MOVING_CYLINDER
		output_file << "MOVING_CYLINDER " << endl;
#endif

#if defined MOVING_SPHEROID
		output_file << "MOVING_SPHEROID " << endl;
		cout << "MOVING_SPHEROID" << endl;
#endif
		output_file << "Number of particles : " << num_particles << endl;
		cout << "Number of Particles = " << num_particles << endl;

		if (num_particles > 0) {
			output_file << "Particle radius     : " << particle_radius << endl;
			cout << "Particle Diameter   = " << particle_radius * 2 << endl;
#if defined MOVING_SPHEROID
			cout << "Particle Diameter 2 = " << particle_radius_2 * 2 << endl;
#endif

			output_file << "First particle X position : " << particle_center_x << endl;
			output_file << "First particle Y position : " << particle_center_y << endl;
			output_file << "First particle Z position : " << particle_center_z << endl;

#if defined Without_added_mass
			output_file << "Without added mass " << endl;
#endif
#if defined With_added_mass
			output_file << "With added mass " << endl;
#endif
			output_file << "Particle number of nodes  : " << particle_num_nodes << endl;
			cout << "Particle number of Nodes = " << particle_num_nodes << endl;

			output_file << "Particle gravity in X dir  : " << particle_gravity_x << endl;
			output_file << "Particle gravity in Y dir  : " << particle_gravity_y << endl;
			output_file << "Particle gravity in Z dir  : " << particle_gravity_z << endl;

			output_file << "Particle to Fluid density ratio  : " << den_ratio << endl;
#if defined Two_Point_Delta
			output_file << "Two_Point_Delta   " << endl;
#endif
#if defined Original_Delta
			output_file << "Original_Delta   " << endl;
#endif
#if defined Four_Point_Delta
			output_file << "Four_Point_Delta   " << endl;
#endif
#if defined Kernel_Delta
			output_file << "Kernel_Delta   " << endl;
#endif
			output_file << "------------------------------------------------------" << endl;
		}
		cout << endl;

		output_file.close();
	}
	return;
}

void Particle_sim::register_recovery(IO_interface& io_interface, Parallel_MPI& parallel_MPI) {
	io_interface.add_custom_write([this, &parallel_MPI](const std::string& base_path, int tm) {
		if (parallel_MPI.is_master()) {
			return;
		}
		for (int np = 0; np < num_particles; ++np) {
			if (Particle_IN[np] == TRUE) {
				write_particle_recovery(base_path, tm, particle[np], np);
			}
		}
	});

	io_interface.add_custom_read([this](const std::string& base_path, int tm) {
		read_particle_recovery(base_path, tm, particle);

		for (int np = 0; np < num_particles; ++np) {
			update_verlet(particle, np);
		}
	});
}

//=============================================================================================================================
//=============================================================================================================================
Particle_sim::~Particle_sim() {
}

std::string Particle_sim::make_recovery_name(const std::string& base_path, int tm, int npn) {
	return base_path + "_particle" + std::to_string(npn) + '_' + std::to_string(tm) + ".dat";
}