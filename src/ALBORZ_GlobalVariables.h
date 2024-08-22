#ifndef ALBORZ_GLOBALVARIABLES_H_
#define ALBORZ_GLOBALVARIABLES_H_

#include "ALBORZ_SETTINGS.h"
#include "mpi.h"
#include <random>

struct run_parameters {
	double D_x, D_t;
	unsigned int Nx, Ny, Nz;
};
extern run_parameters global_parameters;
const int surface_num_nodes = 500;  // number of surface nodes on a solid boundary
//-------------------------------------------------------
//-------------------------------------------------------
//--------------DO NOT MODIFY----------------------------
//-------------------------------------------------------
//-------------------------------------------------------
extern int recovery_step;

extern int t_num;   // number of time steps (running from 1 to t_num)
extern int t_data;  // disk write time step (data will be written to the disk every t_data step)
extern int t_vtk;   // VTK write time step
extern int t_info;  // info time step (screen message will be printed every t_info step)
extern int t_time;  // Report elapsed time on screen
extern int t_recovery;
extern double physical_time_cal;
extern int t_residual;
extern double residual_flow;
extern double residual_thermal;
extern double residual_species;

extern int range;

extern std::default_random_engine g_random;
///------------------------------------------------------------------------------
///                          MPI
///------------------------------------------------------------------------------

extern MPI_Status status;
extern MPI_Request request;

extern double ini_velocity;
#endif
