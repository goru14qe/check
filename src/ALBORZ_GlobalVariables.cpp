// #include "stdafx.h"
#include "ALBORZ_SETTINGS.h"
#include "ALBORZ_GlobalVariables.h"
#include "stdio.h"
#include "Thermal_solver.h"
#include "mpi.h"

run_parameters global_parameters;

int recovery_step;
int t_num, t_data, t_vtk, t_info, t_time, t_recovery;
double physical_time_cal;
double time_limit;

double residual_flow, residual_thermal, residual_species;
int t_residual;

#if defined Flow_Without_Particle
int range;
#endif
#if defined Two_Point_Delta
int range = 1;
#endif
#if defined Four_Point_Delta || defined Kernel_Delta || defined Modified_Two_Point_Delta
int range = 2;
#endif

std::default_random_engine g_random;
///------------------------------------------------------------------------------
///                          MPI
///------------------------------------------------------------------------------
MPI_Status status;
MPI_Request request;
