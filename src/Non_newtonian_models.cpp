#include "Non_newtonian_models.h"
non_newtonian_fluid::non_newtonian_fluid() {}
void non_newtonian_fluid::initialize(const std::string& filename, Flow_solver* Flow, Parallel_MPI* MPI_parallel) {
	/* Open input file */
	std::string input_filename(filename);
	input_filename += ".dat";
	ifstream input_file;
	input_file.open(input_filename.c_str(), ios::binary);
	/* Find header in input */
	find_line_after_header(input_file, "c\tNon-Newtonian");
	find_line_after_comment(input_file);
	input_file >> keyword >> nu_0 >> nu_inf >> n >> a >> lambda >> Smin >> Smax >> implicit;
	nu_0_unscaled = nu_0;
	Smin = Smin * global_parameters.D_t;
	Smax = Smax * global_parameters.D_t;
	/* Set pointer to viscosity law */
	if (keyword == "PL") {
		point2visclaw = &non_newtonian_fluid::PL;
		nu_0 = nu_0 / (sqr(global_parameters.D_x) * pow(global_parameters.D_t, n - 2.));
		nu_inf = 0;
		a = 0;
		lambda = 0;
	}
	if (keyword == "CY") {
		point2visclaw = &non_newtonian_fluid::CY;
		nu_0 = nu_0 / (sqr(global_parameters.D_x) * pow(global_parameters.D_t, -1.));
		nu_inf = nu_inf / (sqr(global_parameters.D_x) * pow(global_parameters.D_t, -1.));
		lambda = lambda / global_parameters.D_t;
	}
	if (keyword == "Cr") {
		point2visclaw = &non_newtonian_fluid::Cr;
		nu_0 = nu_0 / (sqr(global_parameters.D_x) * pow(global_parameters.D_t, -1.));
		nu_inf = nu_inf / (sqr(global_parameters.D_x) * pow(global_parameters.D_t, -1.));
		lambda = lambda / global_parameters.D_t;
		a = 0;
	}
	if (!implicit) { point2Solver = &non_newtonian_fluid::get_omega_exp; }
	if (implicit) {
		point2Solver = &non_newtonian_fluid::get_omega_imp;
		input_file >> conv_imp;
	}
	input_file.close();
	if (MPI_parallel->processor_id == MASTER) {
		std::cout << "fluid properties = " << keyword << "\n";
		std::cout << "nu0 = " << nu_0 << std::endl;
		std::cout << "nuinf = " << nu_inf << std::endl;
		std::cout << "n = " << n << std::endl;
		std::cout << "a = " << a << std::endl;
		std::cout << "Smin = " << Smin << std::endl;
		std::cout << "Smax = " << Smax << std::endl;
		std::cout << "Implicit = ";
		if (!implicit) std::cout << "no" << std::endl;
		if (implicit) {
			std::cout << "yes" << std::endl;
			std::cout << "criterion =" << conv_imp << std::endl;
		}
	}
	if (MPI_parallel->processor_id != MASTER) {
		double cs2 = 1. / Flow->c_s2;
		omega_previous = new double**[MPI_parallel->dev_end[0]];
		for (int X = 0; X < MPI_parallel->dev_end[0]; ++X) {
			omega_previous[X] = new double*[MPI_parallel->dev_end[1]];
			for (int Y = 0; Y < MPI_parallel->dev_end[1]; Y++) {
				omega_previous[X][Y] = new double[MPI_parallel->dev_end[2]];
				for (int Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
					omega_previous[X][Y][Z] = 1. / (nu_0 / cs2 + .5);
					Flow->viscosity[{X, Y, Z}] = nu_0;
				}
			}
		}
	}
}
void non_newtonian_fluid::set_viscosity(int tm, int t_out, Flow_solver* Flow, Parallel_MPI* MPI_parallel) {
	max_iteration = 0;
	int max_iteation_global = 0;
	if (MPI_parallel->processor_id != MASTER) {
		for (int X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
			for (int Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
				for (int Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
					if (Flow->is_solid[{X, Y, Z}] == FALSE) {
						int temp_iterations;
						/* First get rate-of strain */
						/* It must be a 1-D vector of size 6 */
						/* XX, YY, ZZ, XY, XZ, YZ */
						double cs2 = 1. / Flow->c_s2;
						double U = Flow->velocity[{X, Y, Z, 0}];
						double V = Flow->velocity[{X, Y, Z, 1}];
						double W = Flow->velocity[{X, Y, Z, 2}];
						double rho = Flow->density[{X, Y, Z}];
						double DII = 0.;
						double Sxx = 0, Syy = 0, Szz = 0, Sxy = 0, Sxz = 0, Syz = 0;
						for (int alpha = 1; alpha < Flow->Discrete_Velocity; alpha++) {
							double CX = Flow->c_alpha[alpha][0];
							double CY = Flow->c_alpha[alpha][1];
							double CZ = Flow->c_alpha[alpha][2];
							/* use for the non-equilibrium momentum flux */
							double CXX = CX * CX;
							double CYY = CY * CY;
							double CZZ = CZ * CZ;
							double CXY = CX * CY;
							double CXZ = CX * CZ;
							double CYZ = CY * CZ;
							/* get non-equilibrium momentum flux */
							double delpop = Flow->pop[{X, Y, Z, alpha}];
							Sxx += CXX * delpop;
							Syy += CYY * delpop;
							Szz += CZZ * delpop;
							Sxy += CXY * delpop;
							Sxz += CXZ * delpop;
							Syz += CYZ * delpop;
						}
						Sxx = Sxx - rho * (U * U + cs2);
						Syy = Syy - rho * (V * V + cs2);
						Szz = Szz - rho * (W * W + cs2);
						Sxy = Sxy - rho * U * V;
						Sxz = Sxz - rho * U * W;
						Syz = Syz - rho * V * W;
						if (Flow->Dimension == 2) DII = .5 * sqr(Sxx - Syy) + 2. * Sxy * Sxy;
						if (Flow->Dimension == 3) {
							DII = 2. * (Sxy * Sxy + Sxz * Sxz + Syz * Syz)
							      + (1. / 9.) * (sqr(2. * Sxx - Syy - Szz) + sqr(2. * Syy - Sxx - Szz) + sqr(2. * Szz - Sxx - Syy));
						}
						omega_previous[X][Y][Z] = 1. / (Flow->viscosity[{X, Y, Z}] / cs2 + .5);
						/* write strain to file */
						// Flow->alpha_entropic[{X,Y,Z}] = fabs( (.5/cs2)*omega_previous[X][Y][Z]*sqrt(2.*DII))/D_t;
						double omega_new = (this->*point2Solver)(cs2, DII, omega_previous[X][Y][Z], temp_iterations);
						Flow->viscosity[{X, Y, Z}] = cs2 * (1. / omega_new - .5);
						max_iteration = max(max_iteration, temp_iterations);
					}
				}
			}
		}
	}
	MPI_Allreduce(&max_iteration, &max_iteation_global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
	max_iteration = max_iteation_global;
	if (MPI_parallel->processor_id == MASTER && tm % t_out == 1) {
		constexpr const char* output_filename = "Alborz_Results/debug/NN_max_iterations.dat";
		/* First check if file already exists */
		const bool write_header = !std::ifstream(output_filename).is_open();
		ofstream output_file(output_filename, fstream::app);
		if (write_header) { output_file << setw(32) << "time"
			                            << "\t"
			                            << "min_it" << endl; }   // time step
		output_file << setprecision(30) << fixed << tm << "\t";  // time step
		output_file << setprecision(30) << fixed << max_iteration << endl;
	}
}
non_newtonian_fluid::~non_newtonian_fluid() {}

// power_law_fluid::power_law_fluid(){}
// void power_law_fluid::initialize(std::string filename, Flow_solver* Flow, Parallel_MPI* MPI_parallel){
//     /* Open input file */
//     std::string input_filename(filename);
//     input_filename += ".dat";
//     ifstream input_file;
//     input_file.open(input_filename.c_str(),ios::binary);
//     find_line_after_header(input_file, "c\tNon-Newtonian Power-law");
//     find_line_after_comment(input_file);
//     input_file >> K >> n >> Smin >> Smax >> implicit;
//     K = K / ( sqr(D_x) * pow(D_t, n-2) );
//     //K = K / ( sqr(D_x) * pow(D_t, -1) );
//     Smin = Smin * D_t;
//     Smax = Smax * D_t;
//     if(!implicit){point2Solver = &power_law_fluid::get_omega_exp;}
//     if(implicit){point2Solver = &power_law_fluid::get_omega_imp; input_file >> conv_imp;}
//     input_file.close();
//     if (processor_id == MASTER) {
//         std::cout << "Power-law fluid properties\n";
//         std::cout << "K = " << K << std::endl;
//         std::cout << "Smin = " << Smin << std::endl;
//         std::cout << "Smax = " << Smax << std::endl;
//         std::cout << "n = " << n << std::endl;
//         std::cout << "Implicit = ";
//         if(!implicit) std::cout << "no" << std::endl;
//         if(implicit) {std::cout << "yes" << std::endl; std::cout << "criterion =" << conv_imp << std::endl;}
//         }
//	if (processor_id != MASTER) {
//         double cs2 = 1./Flow->c_s2;
//         omega_previous = new double**[MPI_parallel->dev_end[0]];
//         for (int X = 0; X < MPI_parallel->dev_end[0]; ++X) {
//             omega_previous[X] = new double*[MPI_parallel->dev_end[1]];
//             for (int Y = 0; Y < MPI_parallel->dev_end[1]; Y++){
//                 omega_previous[X][Y] = new double[MPI_parallel->dev_end[2]];
//                 for (int Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
//                     omega_previous[X][Y][Z] = 1./(K/cs2 + .5);
//                     Flow->viscosity[{X,Y,Z}] = K;
//                     }
//                 }
//             }
//         }
//     }
// void power_law_fluid::set_viscosity(int tm, int t_out, Flow_solver* Flow, Parallel_MPI* MPI_parallel){
//     max_iteration = 0;
//     int max_iteation_global = 0;
//     if (processor_id != MASTER) {
//         for (int X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
//             for (int Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
//                 for (int Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
//                     if (Flow->is_solid[{X,Y,Z}] == FALSE){
//                         int temp_iterations;
//                         /* First get rate-of strain */
//                         /* It must be a 1-D vector of size 6 */
//                         /* XX, YY, ZZ, XY, XZ, YZ */
//                         double cs2 = 1./Flow->c_s2;
//                         double U = Flow->velocity[{X,Y,Z,0}];
//                         double V = Flow->velocity[{X,Y,Z,1}];
//                         double W = Flow->velocity[{X,Y,Z,2}];
//                         double rho = Flow->density[{X,Y,Z}];
//                         double DII = 0.;
//                         double Sxx=0, Syy=0, Szz=0, Sxy=0, Sxz=0, Syz=0;
//                         for (int alpha=1; alpha<Flow->Discrete_Velocity; alpha++){
//                             double CX = Flow->c_alpha[alpha][0];
//                             double CY = Flow->c_alpha[alpha][1];
//                             double CZ = Flow->c_alpha[alpha][2];
//                             /* use for the non-equilibrium momentum flux */
//                             double CXX = CX*CX; double CYY = CY*CY; double CZZ = CZ*CZ;
//                             double CXY = CX*CY; double CXZ = CX*CZ; double CYZ = CY*CZ;
//                             /* get non-equilibrium momentum flux */
//                             //double delpop = Flow->pop_old[{X,Y,Z,alpha}]; Feng
//                             double delpop = Flow->pop[{X,Y,Z,alpha}];
//                             Sxx += CXX * delpop; Syy += CYY * delpop; Szz += CZZ * delpop;
//                             Sxy += CXY * delpop; Sxz += CXZ * delpop; Syz += CYZ * delpop;
//                             }
//                         Sxx = Sxx - rho*(U*U + cs2);
//                         Syy = Syy - rho*(V*V + cs2);
//                         Szz = Szz - rho*(W*W + cs2);
//                         Sxy = Sxy - rho*U*V;
//                         Sxz = Sxz - rho*U*W;
//                         Syz = Syz - rho*V*W;
//                         if (Flow->Dimension == 2) DII = .5*sqr(Sxx-Syy) + 2.*Sxy*Sxy;
//                         if (Flow->Dimension == 3) {
//                             DII = 2.*(Sxy*Sxy + Sxz*Sxz + Syz*Syz)
//                                 + (1./9.) * (sqr(2.*Sxx-Syy-Szz) + sqr(2.*Syy-Sxx-Szz) + sqr(2.*Szz-Sxx-Syy) );
//                             }
//                         omega_previous[X][Y][Z] = 1./(Flow->viscosity[{X,Y,Z}]/cs2 + .5);
//                         double omega_new = (this->*point2Solver)(cs2, DII, omega_previous[X][Y][Z], temp_iterations);
//                         Flow->viscosity[{X,Y,Z}] = cs2*(1./omega_new - .5);
//                         max_iteration = max(max_iteration, temp_iterations);
//                         //Flow->alpha_entropic[{X,Y,Z}] = Flow->viscosity[{X,Y,Z}];
//                         }
//                     }
//                 }
//             }
//         }
//
//     MPI_Allreduce(&max_iteration, &max_iteation_global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
//     max_iteration = max_iteation_global;
//
//     if (processor_id == MASTER && tm%t_out == 1){
//         /* Create filename */
//         stringstream output_filename;
//         output_filename << "Alborz_Results/debug/NN_max_iterations.dat";
//         /* First check if file already exists */
//         unsigned int header = 0;
//         std::ifstream fileStream;
//         fileStream.open(output_filename.str().c_str());
//         if (fileStream.fail() == TRUE ) {header = 1;}
//         if (fileStream.fail() == FALSE) {fileStream.close();}
//         ofstream output_file;
//         output_file.open(output_filename.str().c_str(), fstream::app);
//         if (header == 1) {output_file << setw(32) << "time" << "\t" << "min_it" <<endl;} // time step
//         output_file << setprecision(30) << fixed << tm << "\t"; // time step
//         output_file << setprecision(30) << fixed << max_iteration << endl;
//         /* close file */
//         output_file.close();
//         }
//     }
// power_law_fluid::~power_law_fluid(){}
//
//
// carreau_fluid::carreau_fluid(){}
// void carreau_fluid::initialize(std::string filename, Flow_solver* Flow, Parallel_MPI* MPI_parallel){
//     /* Open input file */
//     std::string input_filename(filename);
//     input_filename += ".dat";
//     ifstream input_file;
//     input_file.open(input_filename.c_str(),ios::binary);
//     find_line_after_header(input_file, "c\tNon-Newtonian Carreau");
//     find_line_after_comment(input_file);
//     input_file >> nu0 >> nuinf >> lambda >> a >> n >> implicit;
//     nu0 = nu0 / ( sqr(D_x) * pow(D_t, -1) );
//     nuinf = nuinf / ( sqr(D_x) * pow(D_t, -1) );
//     lambda = lambda/D_t;
//
//     if(!implicit){point2Solver = &carreau_fluid::get_omega_exp;}
//     if(implicit){point2Solver = &carreau_fluid::get_omega_imp; input_file >> conv_imp;}
//     input_file.close();
//     if (processor_id == MASTER) {
//         std::cout << "Carreau fluid properties\n";
//         std::cout << "nu0 = " << nu0 << std::endl;
//         std::cout << "nuinf = " << nuinf << std::endl;
//         std::cout << "lambda = " << lambda << std::endl;
//         std::cout << "a = " << a << std::endl;
//         std::cout << "n = " << n << std::endl;
//         std::cout << "Implicit = ";
//         if(!implicit) std::cout << "no" << std::endl;
//         if(implicit) {std::cout << "yes" << std::endl; std::cout << "criterion =" << conv_imp << std::endl;}
//         }
//         if (processor_id != MASTER) {
//         double cs2 = 1./Flow->c_s2;
//         omega_previous = new double**[MPI_parallel->dev_end[0]];
//         for (int X = 0; X < MPI_parallel->dev_end[0]; ++X) {
//             omega_previous[X] = new double*[MPI_parallel->dev_end[1]];
//             for (int Y = 0; Y < MPI_parallel->dev_end[1]; Y++){
//                 omega_previous[X][Y] = new double[MPI_parallel->dev_end[2]];
//                 for (int Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
//                     omega_previous[X][Y][Z] = 1./(nu0/cs2 + .5);
//                     Flow->viscosity[{X,Y,Z}] = nu0;
//                     }
//                 }
//             }
//         }
//     }
// void carreau_fluid::set_viscosity(int tm, int t_out, Flow_solver* Flow, Parallel_MPI* MPI_parallel){
//     max_iteration = 0;
//     int max_iteation_global = 0;
//     if (processor_id != MASTER) {
//         for (int X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
//             for (int Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
//                 for (int Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
//                     if (Flow->is_solid[{X,Y,Z}] == FALSE){
//                         int temp_iterations;
//                         /* First get rate-of strain */
//                         /* It must be a 1-D vector of size 6 */
//                         /* XX, YY, ZZ, XY, XZ, YZ */
//                         double cs2 = 1./Flow->c_s2;
//                         double U = Flow->velocity[{X,Y,Z,0}];
//                         double V = Flow->velocity[{X,Y,Z,1}];
//                         double W = Flow->velocity[{X,Y,Z,2}];
//                         double rho = Flow->density[{X,Y,Z}];
//                         double DII = 0.;
//                         double Sxx=0, Syy=0, Szz=0, Sxy=0, Sxz=0, Syz=0;
//                         for (int alpha=1; alpha<Flow->Discrete_Velocity; alpha++){
//                             double CX = Flow->c_alpha[alpha][0];
//                             double CY = Flow->c_alpha[alpha][1];
//                             double CZ = Flow->c_alpha[alpha][2];
//                             /* use for the non-equilibrium momentum flux */
//                             double CXX = CX*CX; double CYY = CY*CY; double CZZ = CZ*CZ;
//                             double CXY = CX*CY; double CXZ = CX*CZ; double CYZ = CY*CZ;
//                             /* get non-equilibrium momentum flux */
//                             double delpop = Flow->pop[{X,Y,Z,alpha}];
//                             Sxx += CXX * delpop; Syy += CYY * delpop; Szz += CZZ * delpop;
//                             Sxy += CXY * delpop; Sxz += CXZ * delpop; Syz += CYZ * delpop;
//                             }
//                         Sxx = Sxx - rho*(U*U + cs2);
//                         Syy = Syy - rho*(V*V + cs2);
//                         Szz = Szz - rho*(W*W + cs2);
//                         Sxy = Sxy - rho*U*V;
//                         Sxz = Sxz - rho*U*W;
//                         Syz = Syz - rho*V*W;
//                         if (Flow->Dimension == 2) DII = .5*sqr(Sxx-Syy) + 2.*Sxy*Sxy;
//                         if (Flow->Dimension == 3) {
//                             DII = 2.*(Sxy*Sxy + Sxz*Sxz + Syz*Syz)
//                                 + (1./9.) * (sqr(2.*Sxx-Syy-Szz) + sqr(2.*Syy-Sxx-Szz) + sqr(2.*Szz-Sxx-Syy) );
//                             }
//                         omega_previous[X][Y][Z] = 1./(Flow->viscosity[{X,Y,Z}]/cs2 + .5);
//                         double omega_new = (this->*point2Solver)(cs2, DII, omega_previous[X][Y][Z], temp_iterations);
//                         Flow->viscosity[{X,Y,Z}] = cs2*(1./omega_new - .5);
//                         max_iteration = max(max_iteration, temp_iterations);
//                         }
//                     }
//                 }
//             }
//         }
//     MPI_Allreduce(&max_iteration, &max_iteation_global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
//     max_iteration = max_iteation_global;
//     if (processor_id == MASTER && tm%t_out == 1){
//         /* Create filename */
//         stringstream output_filename;
//         output_filename << "Alborz_Results/debug/NN_max_iterations.dat";
//         /* First check if file already exists */
//         unsigned int header = 0;
//         std::ifstream fileStream;
//         fileStream.open(output_filename.str().c_str());
//         if (fileStream.fail() == TRUE ) {header = 1;}
//         if (fileStream.fail() == FALSE) {fileStream.close();}
//         ofstream output_file;
//         output_file.open(output_filename.str().c_str(), fstream::app);
//         if (header == 1) {output_file << setw(32) << "time" << "\t" << "min_it" <<endl;} // time step
//         output_file << setprecision(30) << fixed << tm << "\t"; // time step
//         output_file << setprecision(30) << fixed << max_iteration << endl;
//         /* close file */
//         output_file.close();
//         }
//     }
// carreau_fluid::~carreau_fluid(){}
//
// kuangluo_fluid::kuangluo_fluid(){}
// void kuangluo_fluid::initialize(std::string filename, Flow_solver* Flow, Parallel_MPI* MPI_parallel){
//
//     /* Open input file */
//     std::string input_filename(filename);
//
//     input_filename += ".dat";
//
//     ifstream input_file;
//
//     input_file.open(input_filename.c_str(),ios::binary);
//
//     find_line_after_header(input_file, "c\tNon-Newtonian KL");
//
//     find_line_after_comment(input_file);
//
//     input_file >> eta1 >> eta2 >> sigmay >> m >> implicit;
//
////    eta1 = eta1 / ( sqr(D_x) * pow(D_t, -0.5) );
////
////    eta2 = eta2 / ( sqr(D_x) * pow(D_t, 0) );
//
//    eta1 = eta1 / ( sqr(D_x) * pow(D_t, -1) );
//    eta2 = eta2 / ( sqr(D_x) * pow(D_t, -0.5) );
//
//    m = m/D_t;
//
//
//    if(!implicit){point2Solver = &kuangluo_fluid::get_omega_exp;}
//
//    if(implicit){point2Solver = &kuangluo_fluid::get_omega_imp; input_file >> conv_imp;}
//
//    input_file.close();
//
//    if (processor_id == MASTER) {
//
//        std::cout << "Kuang-Luo fluid properties\n";
//
//        std::cout << "eta1 = " << eta1 << std::endl;
//
//        std::cout << "eta2 = " << eta2 << std::endl;
//
//        std::cout << "sigmay = " << sigmay << std::endl;
//
//        std::cout << "m = " << m << std::endl;
//
//        std::cout << "Implicit = ";
//
//        if(!implicit) std::cout << "no" << std::endl;
//
//        if(implicit) {std::cout << "yes" << std::endl; std::cout << "criterion =" << conv_imp << std::endl;}
//
//        }
//
//    if (processor_id != MASTER) {
//
//        double cs2 = 1./Flow->c_s2;
//
//        omega_previous = new double**[MPI_parallel->dev_end[0]];
//
//        for (int X = 0; X < MPI_parallel->dev_end[0]; ++X) {
//
//            omega_previous[X] = new double*[MPI_parallel->dev_end[1]];
//
//            for (int Y = 0; Y < MPI_parallel->dev_end[1]; Y++){
//
//                omega_previous[X][Y] = new double[MPI_parallel->dev_end[2]];
//
//                for (int Z = 0; Z < MPI_parallel->dev_end[2]; Z++) {
//
//                    omega_previous[X][Y][Z] = 1./(eta1/cs2 + .5);
//
//                    Flow->viscosity[{X,Y,Z}] = eta1;
//
//                    }
//
//                }
//
//            }
//
//        }
//
//    }
//
// void kuangluo_fluid::set_viscosity(int tm, int t_out, Flow_solver* Flow, Parallel_MPI* MPI_parallel){
//
//    max_iteration = 0;
//
//    int max_iteation_global = 0;
//
//    if (processor_id != MASTER) {
//
//        for (int X = MPI_parallel->start_XYZ2[0]; X <= MPI_parallel->end_XYZ2[0]; ++X) {
//
//            for (int Y = MPI_parallel->start_XYZ2[1]; Y <= MPI_parallel->end_XYZ2[1]; ++Y) {
//
//                for (int Z = MPI_parallel->start_XYZ2[2]; Z <= MPI_parallel->end_XYZ2[2]; ++Z) {
//
//                    if (Flow->is_solid[{X,Y,Z}] == FALSE){
//
//                        int temp_iterations;
//
//                        /* First get rate-of strain */
//
//                        /* It must be a 1-D vector of size 6 */
//
//                        /* XX, YY, ZZ, XY, XZ, YZ */
//
//                        double cs2 = 1./Flow->c_s2;
//
//                        double U = Flow->velocity[{X,Y,Z,0}];
//
//                        double V = Flow->velocity[{X,Y,Z,1}];
//
//                        double W = Flow->velocity[{X,Y,Z,2}];
//
//                        double rho = Flow->density[{X,Y,Z}];
//
//                        double DII = 0.;
//
//                        double Sxx=0, Syy=0, Szz=0, Sxy=0, Sxz=0, Syz=0;
//
//                        for (int alpha=1; alpha<Flow->Discrete_Velocity; alpha++){
//
//                            double CX = Flow->c_alpha[alpha][0];
//
//                            double CY = Flow->c_alpha[alpha][1];
//
//                            double CZ = Flow->c_alpha[alpha][2];
//
//                            /* use for the non-equilibrium momentum flux */
//
//                            double CXX = CX*CX; double CYY = CY*CY; double CZZ = CZ*CZ;
//
//                            double CXY = CX*CY; double CXZ = CX*CZ; double CYZ = CY*CZ;
//
//                            /* get non-equilibrium momentum flux */
//
//                            double delpop = Flow->pop[{X,Y,Z,alpha}];
//
//                            Sxx += CXX * delpop; Syy += CYY * delpop; Szz += CZZ * delpop;
//
//                            Sxy += CXY * delpop; Sxz += CXZ * delpop; Syz += CYZ * delpop;
//
//                            }
//
//                        Sxx = Sxx - rho*(U*U + cs2);
//
//                        Syy = Syy - rho*(V*V + cs2);
//
//                        Szz = Szz - rho*(W*W + cs2);
//
//                        Sxy = Sxy - rho*U*V;
//
//                        Sxz = Sxz - rho*U*W;
//
//                        Syz = Syz - rho*V*W;
//
//                        if (Flow->Dimension == 2) DII = .5*sqr(Sxx-Syy) + 2.*Sxy*Sxy;
//
//                        if (Flow->Dimension == 3) {
//
//                            DII = 2.*(Sxy*Sxy + Sxz*Sxz + Syz*Syz)
//
//                                + (1./9.) * (sqr(2.*Sxx-Syy-Szz) + sqr(2.*Syy-Sxx-Szz) + sqr(2.*Szz-Sxx-Syy) );
//
//                            }
//
//                        omega_previous[X][Y][Z] = 1./(Flow->viscosity[{X,Y,Z}]/cs2 + .5);
//
//                        double omega_new = (this->*point2Solver)(cs2, DII, omega_previous[X][Y][Z], temp_iterations);
//
//                        Flow->viscosity[{X,Y,Z}] = cs2*(1./omega_new - .5);
//
//                        max_iteration = max(max_iteration, temp_iterations);
//
//                        }
//
//                    }
//
//                }
//
//            }
//
//        }
//
//    MPI_Allreduce(&max_iteration, &max_iteation_global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
//
//    max_iteration = max_iteation_global;
//
//    if (processor_id == MASTER && tm%t_out == 1){
//
//        /* Create filename */
//
//        stringstream output_filename;
//
//        output_filename << "Alborz_Results/debug/NN_max_iterations.dat";
//
//        /* First check if file already exists */
//
//        unsigned int header = 0;
//
//        std::ifstream fileStream;
//
//        fileStream.open(output_filename.str().c_str());
//
//        if (fileStream.fail() == TRUE ) {header = 1;}
//
//        if (fileStream.fail() == FALSE) {fileStream.close();}
//
//        ofstream output_file;
//
//        output_file.open(output_filename.str().c_str(), fstream::app);
//
//        if (header == 1) {output_file << setw(32) << "time" << "\t" << "min_it" <<endl;} // time step
//
//        output_file << setprecision(30) << fixed << tm << "\t"; // time step
//
//        output_file << setprecision(30) << fixed << max_iteration << endl;
//
//        /* close file */
//
//        output_file.close();
//
//        }
//
//    }
// kuangluo_fluid::~kuangluo_fluid(){}
