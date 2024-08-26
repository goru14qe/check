#ifndef ALBORZ_SETTINGS_H_
#define ALBORZ_SETTINGS_H_
/// *********************
/// SIMULATION PARAMETERS
/// *********************
// These are the relevant simulation parameters.
// They can be changed by the user.
/// ==================================================================================================================================== ///
///                                                           FLOW FIELD                                                                 ///
///===================================================================================================================================== ///
/// #define FD_CENTRAL
/// #define FD_CENTRAL4
/// #define FD_UPWIND
/// #define FD_UPWIND2
#define FD_WENO3
/// #define FD_WENO5
#define CORR_UPWIND

// enables faster code paths that may be incompatible with new features
#define PERFORMANCE_MODE
// #define DEBUG_MODE
// #define compressible
#define LMNA_solver
/// -----------------------------------> Boundaries Definition <----------------------------------------------------------------------- ///
// these values are only used in 
#define X_Periodic
//#define X_NOT_Periodic
#define Y_Periodic
//#define Y_NOT_Periodic
#define Z_Periodic
//#define Z_NOT_Periodic
/// ---------------------------------------------> Forcing scheme <---------------------------------------------------------------------- ///
#define Kupershtokh
/// ---------------------------------------------> Boundary scheme <---------------------------------------------------------------------- ///

// #define VTK_ASCII //Paraview file format in ASCII format
#define VTK_BINARY  // Paraview file format in BINARY format
/// ==================================================================================================================================== ///
///                                                           PARTICLE                                                                   ///
///===================================================================================================================================== ///
// Flow with/without particle is decided by the loop selection
// so this should no cause any issues if it is always active
//#define Flow_With_Particle
#define Flow_Without_Particle
#if defined Flow_With_Particle
#define Flow_With_Thermal_Effect
// #define Flow_Without_Thermal_Effect
#endif  // defined
#if defined Flow_With_Particle
#define MOVING_SPHERE
// #define MOVING_SPHEROID

// #define Without_added_mass
#define With_added_mass
#define Corrected_Radius
#define Four_Point_Delta
// #define Kernel_Delta

#if defined MOVING_SPHERE
// - select collision model Particle-Particle or Particle-Wall collision. (Just for sphere shape and just Direct forcing).
// #define Sphere_Collision_1_Feng  // Feng ZG, Michaelides 10E. Journal of Computational Physics,202 (2005), 20-51.
// #define Sphere_Collision_2_Feng_Verlet   // Use verlet list and avoid checking all particles. Feng ZG, Michaelides 10E. Journal of Computational Physics,202 (2005), 20-51.
#define Sphere_Collision_3_Lub_Verlet    //Lubrication and Feng repulsive models. Lubrication based on paper 952.
#endif

#if defined MOVING_SPHEROID  // Collision model for Spheroid
// - select collision model for more than 1 moving Spheroidal particles or particle-wall collision.
// #define Spheroid_Collision_1_Feng
// #define Spheroid_Collision_2_Feng_Verlet
#define Spheroid_Collision_3_Feng_Lub_Verlet
#endif

#if defined Flow_With_Thermal_Effect
// #define Particle_Fixed_Temperature
#define Particle_Varying_Temperature
#endif

// #define Random_Arrangement_ON
#define Random_Arrangement_OFF

#endif

/// ==================================================================================================================================== ///
///                                                           CHEMICAL SPECIES                                                           ///
///===================================================================================================================================== ///
/// #define POROUS_MEDIA
#define Flow_With_Species
//#define Flow_Without_Species
/// -------------------------------------> Transport properties <----------------------------------------------------------------------- ///
#if defined Flow_With_Species
/// #define IMPLEX
#endif

/// #if defined REGATH_LIB
#define Wilke_viscosity
/// #define Polynomial_viscosity
//#define MixtureAveraged_Species_diffusion
#define ConstLewis_Species_diffusion
//#define UnitLewis_Species_diffusion
//#define Diffusion_model
///#endif
/// ==================================================================================================================================== ///
///                                                       DO NOT TOUCH                                                                   ///
///===================================================================================================================================== ///

// Read_Geometry_From_File::
// Condition of Read_Geometry_From_File is just for Flow_Without_Particle
// In this case you should:
// 1: convert the geometry to a matrix of 0 and 1 by the MATLAB program: image_to_matrix.m .This MATLAB rogram creates geometry.txt
// 2: put the geometry.txt in the same folder as where parameters.dat or fluid_data.dat are created.
// 3: Enter manually global_parameters.Nx and global_parameters.Ny values in ALBORZ program.
// Below condition can be used by different boundary conditions.
// If gravity has a non-zero value then Permeability is calculated and reported and saved to fluid_data.dat
// If you deactive below line you can still define some solid nodes in the geometry by putting is_solid=TRUE (See beginning of Flow_Without_Particle section)

// Force Calculation method.
// Guo: Z. Guo, C. Zheng, B. Shi, Discrete lattice effects on the forcing term in the lattice Boltzmann method, Phys. Rev. E 65 (046308) (2002) 1�6.
// Shan-Chen: Shan, X., Chen, H. (1993), Lattice Boltzmann model for simulating flows with multiple phase and components, Physical Review E, 47, 1815-1819
// Luo: L.-S. Luo, Lattice-Gas Automata and Lattice Boltzmann Equations for Two-Dimensional Hydrodynamics, Ph.D. thesis, Georgia Institute of Technology,1993.
// Guo model is a second order model but has less stability than Shan-Chen in low relaxation times. It is also somewhat slower.

// Select Single Relaxation Time (SRT) or Multi Relaxation Time (MRT)
// -MRT model is JUST adjusted for "Guo" force scheme.

// Boundary types
// Exactly one of the boundary options has to be defined.
// X_Periodic
// - Boundary condition of X-Direction: Periodic inlet and outlet
// X_NOT_Periodic
// - Boundary condition of X-Direction: Not periodic

// Output visualisation file format: VTK or CASE(Ensight)

/// Particle types
// Exactly one of the following options has to be defined.
// STATIONARY_CYLINDER
// - for simulation of, e.g., Karman vortex street
// - the cylinder is kept stationary in space, it does not move
// STATIONARY_ELLIPSE
// DEFORMABLE_CYLINDER
// - for simulation of a moving cylinder.
// - If you choose "Direct_Forcing" then the cylinder CANNOT deform but moves in the fluid.
// DEFORMABLE_ELLIPSE
// - for simulation of a moving ellipse

// SDF: Single-Direct-Forcing.
// MDF: Multi-Direct-Forcing. Do NOT use it when there is NO particle inside the Flow.
// use MDF in case of Flow_Without_Thermal_Effect

// For description of added mass refer to page 88 and 89 of Kang PhD Thesis (2010)

// Delta Function choice
// - You can choose Original delta function which was first developed by Kr�ger.NOT available in this version
// - You can choose 2 point function: Eq. 67 Kang thesis (804)
// - You can choose 4 point function: Eq 68 Kang thesis (804)
// - You can choose Kernel function: Eq. 17 Feng (806) Z.-G. Feng, E.E. Michaelides / Journal of Computational Physics 202 (2005) pp.20�51

// Collision model for CIRCLE
// - select collision model for more than 1 moving CIRCULAR particles or particle-wall collision. (Just for Direct forcing).

// Activate "Recovery_ON" if you want to continue the program from a recovery step (for example if the program has stopped due to electercity failure but you do NOT want to start from 1st iteration)
// In case of "Recovery_ON", initial values for fluid and particles are read from "data_fluid_recovery.dat" (See Initialize subroutine)
// Activate "Recovery_OFF" if you want to start the program from the 1st iteration.
// Only ONE of these options must be active at each run.

// Stop_by_Residual: Stop the program when residual falls below a certain value

// Curved boundary treatment
//  Bouzidi: M. Bouzidi, M. Firdaouss, P. Lallemand, Momentum transfer of a Boltzmann-lattice fluid with boundaries, Physics of Fluids 13, 3452 (2001);
//  Filippova
//  Mei_Luo: R. Mei, L. Luo, Wl Shy, An Accurate Curved Boundary Treatment in the Lattice Boltzmann Method, Journal of Computational Physics 155, 307�330 (1999)

#endif
