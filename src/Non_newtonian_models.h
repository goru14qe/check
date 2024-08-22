#include <vector>
#include <map>
#include <cmath>
#include <random>
#include <string.h>
#include <sstream>   // string streams
#include <iostream>  // for the use of 'cout'
#include <cstdlib>
#include "ALBORZ_GlobalVariables.h"
#include "ALBORZ_SETTINGS.h"
#include "ALBORZ_Macros.h"
#include "Parallel.h"
#include "Flow_solver.h"
#include "Thermal_solver.h"
#include "Species_solver.h"
#include "Fluid_read_write.h"

#ifndef NONNEWTONIANMODELS_H
#define NONNEWTONIANMODELS_H
class non_newtonian_fluid {
public:
	/* viscosity law keyword: */
	/* power law : PL */
	/* Carreau-Yasuda : CY */
	/* Cross : Cr */
	std::string keyword;
	/* consistency index */
	double nu_0, nu_inf, nu_0_unscaled;
	/* fluid behavior index */
	double n, a, lambda;
	/* Upper strain limite*/
	double Smax;
	/* Lower strain limit */
	double Smin;
	/* implicit (or not) */
	bool implicit;
	/* convergence criterion for implicit solver */
	double conv_imp;
	/* maximum number of iterations */
	int max_iteration = 0;
	/* Pointer to solver function, implicit or explicit */
	double (non_newtonian_fluid::*point2Solver)(double const&, double const&, double const&, int&);
	double get_omega_exp(double const& cs2, double const& DII, double const& omega_old, int& iterations) {
		double DIItemp = min(max(Smin, fabs((.5 / cs2) * omega_old * sqrt(2. * DII))), Smax);
		double nu = (this->*point2visclaw)(DIItemp);
		double omega_new = 1. / (nu / cs2 + 0.5);
		iterations = 0;
		return omega_new;
	}
	double get_omega_imp(double const& cs2, double const& DII, double const& omega_old, int& iterations) {
		double deltaomega;
		double omega_1, omega_2;
		omega_1 = omega_old;
		omega_2 = omega_old;
		iterations = 0;
		do {
			omega_1 = omega_2;
			double DIItemp = min(max(Smin, fabs((.5 / cs2) * omega_1 * sqrt(2. * DII))), Smax);
			double nu_2 = (this->*point2visclaw)(DIItemp);
			omega_2 = 1. / (nu_2 / cs2 + 0.5);
			deltaomega = fabs(omega_2 - omega_1) / (fabs(omega_2) + 1e-16);
			iterations++;
		} while (deltaomega > conv_imp);
		double omega_new = omega_2;
		return omega_new;
	}
	/* Pointer to viscosity law function */
	double (non_newtonian_fluid::*point2visclaw)(double const&);

	double PL(double const& DIItemp) {
		return (nu_0 * pow(DIItemp, n - 1.));
	}
	double CY(double const& DIItemp) {
		return (nu_inf + (nu_0 - nu_inf) * pow(pow(1. + (lambda * DIItemp), a), (n - 1.) / a));
	}
	double Cr(double const& DIItemp) {
		return (nu_inf + (nu_0 - nu_inf) / pow(1. + (lambda * DIItemp), n));
	}
	/* Array holding previous time-step omega */
	double*** omega_previous;
	non_newtonian_fluid();
	void initialize(const std::string&, Flow_solver*, Parallel_MPI*);
	void set_viscosity(int, int, Flow_solver*, Parallel_MPI*);
	~non_newtonian_fluid();

private:
};

#endif

// class power_law_fluid {
//     public:
//         /* consistency index */
//         double K;
//         /* fluid behavior index */
//         double n;
//         /* Upper strain limite*/
//         double Smax;
//         /* Lower strain limit */
//         double Smin;
//         /* implicit (or not) */
//         bool implicit;
//         /* convergence criterion for implicit solver */
//         double conv_imp;
//         /* maximum number of iterations */
//         int max_iteration = 0;
//         /* Pointer to solver function, implicit or explicit */
//         double (power_law_fluid::*point2Solver)(double const&, double const&, double const&, int &);
//         /* Array holding previous time-step omega */
//         double ***omega_previous;
//         power_law_fluid();
//         double get_omega_exp(double const& cs2, double const& DII, double const& omega_old, int &iterations){
//             double DIItemp = min(max(Smin, fabs( (.5/cs2)*omega_old*sqrt(2.*DII))), Smax);
//
//             double omega_new = 1./( (K/cs2) * pow(DIItemp, n-1.) + 0.5);
//             iterations = 0;
//             return omega_new;
//             }
//         double get_omega_imp(double const& cs2, double const& DII, double const& omega_old, int &iterations){
//             double deltaomega;
//             double omega_1, omega_2;
//             omega_1 = omega_old;
//             omega_2 = omega_old;
//             iterations = 0;
//             do{
//                 omega_1 = omega_2;
//                 double DIItemp = min(max(Smin, fabs( (.5/cs2)*omega_1*sqrt(2.*DII))), Smax);
//                 omega_2 = 1./( (K/cs2) * pow(DIItemp, n-1.) + 0.5);
//                 deltaomega = fabs(omega_2-omega_1)/(fabs(omega_2)+1e-16);
//                 iterations++;
//                 } while (deltaomega>conv_imp);
//             double omega_new = omega_2;
//             return omega_new;
//             }
//         void initialize(std::string, Flow_solver*, Parallel_MPI*);
//         void set_viscosity(int, int, Flow_solver*, Parallel_MPI*);
//         ~power_law_fluid();
//         private:
//     };
// #endif
//
// class carreau_fluid {
//     public:
//         /* viscosity at zero and infinit shear */
//         double nu0, nuinf;
//         /* fluid behavior index */
//         double n;
//         /* relaxation rate*/
//         double lambda, a;
//         /* implicit? */
//         bool implicit;
//         /* convergence criterion for implicit solver */
//         double conv_imp;
//         /* maximum number of iterations */
//         int max_iteration = 0;
//         /* Pointer to solver function, implicit or explicit */
//         double (carreau_fluid::*point2Solver)(double const&, double const&, double const&, int &);
//         /* Array holding previous time-step omega */
//         double ***omega_previous;
//         carreau_fluid();
//         double get_omega_exp(double const& cs2, double const& DII, double const& omega_old, int &iterations){
//             double DIItemp = fabs( (.5/cs2)*omega_old*sqrt(2.*DII));
//             double nu = nuinf + (nu0 - nuinf) * pow(1. + pow(lambda*DIItemp,a), (n-1.)/a);
//             double omega_new = 1./( nu/cs2 + 0.5);
//             iterations = 0;
//             return omega_new;
//             }
//         double get_omega_imp(double const& cs2, double const& DII, double const& omega_old, int &iterations){
//             double deltaomega;
//             double omega_1, omega_2;
//             omega_1 = omega_old;
//             omega_2 = omega_old;
//             iterations = 0;
//             do{
//                 omega_1 = omega_2;
//                 double DIItemp = fabs( (.5/cs2)*omega_old*sqrt(2.*DII));
//
//                 double nu_2 = nuinf + (nu0 - nuinf) * pow(1. + pow(lambda*DIItemp,a), (n-1.)/a);
//                 omega_2 = 1./( nu_2/cs2 + 0.5);
//                 deltaomega = fabs(omega_2-omega_1)/(fabs(omega_2)+1e-16);
//                 iterations++;
//                 } while (deltaomega>conv_imp);
//             double omega_new = omega_2;
//             return omega_new;
//             }
//         void initialize(std::string, Flow_solver*, Parallel_MPI*);
//         void set_viscosity(int, int, Flow_solver*, Parallel_MPI*);
//         ~carreau_fluid();
//         private:
//     };
//
// class kuangluo_fluid {
//
//     public:
//
//         /* viscosity at zero and infinit shear */
//
//         double eta1, eta2;
//
//         /* fluid behavior index */
//
//         double m;
//
//         /* relaxation rate*/
//
//         double sigmay;
//
//         /* implicit? */
//
//         bool implicit;
//
//         /* convergence criterion for implicit solver */
//
//         double conv_imp;
//
//         /* maximum number of iterations */
//
//         int max_iteration = 0;
//
//         /* Pointer to solver function, implicit or explicit */
//
//         double (kuangluo_fluid::*point2Solver)(double const&, double const&, double const&, int &);
//
//         /* Array holding previous time-step omega */
//
//         double ***omega_previous;
//
//         kuangluo_fluid();
//
//         double get_omega_exp(double const& cs2, double const& DII, double const& omega_old, int &iterations){
//
//             double DIItemp = fabs( (.5/cs2)*omega_old*sqrt(2.*DII));
//
//             double nu = eta1 + (eta2/sqrt(DIItemp) + sigmay/DIItemp ) * (1. - exp(-m*DIItemp) );
//
//             double omega_new = 1./( nu/cs2 + 0.5);
//
//             iterations = 0;
//
//             return omega_new;
//
//             }
//
//         double get_omega_imp(double const& cs2, double const& DII, double const& omega_old, int &iterations){
//
//             double deltaomega;
//
//             double omega_1, omega_2;
//
//             omega_1 = omega_old;
//
//             omega_2 = omega_old;
//
//             iterations = 0;
//
//             do{
//
//                 omega_1 = omega_2;
//
//                 double DIItemp = fabs( (.5/cs2)*omega_old*sqrt(2.*DII));
//
//                 double nu_2 = eta1 + (eta2/sqrt(DIItemp) + sigmay/DIItemp ) * (1. - exp(-m*DIItemp) );
//
//                 omega_2 = 1./( nu_2/cs2 + 0.5);
//
//                 deltaomega = fabs(omega_2-omega_1)/(fabs(omega_2)+1e-16);
//
//                 iterations++;
//
//                 } while (deltaomega>conv_imp);
//
//             double omega_new = omega_2;
//
//             return omega_new;
//
//             }
//
//         void initialize(std::string, Flow_solver*, Parallel_MPI*);
//
//         void set_viscosity(int, int, Flow_solver*, Parallel_MPI*);
//
//         ~kuangluo_fluid();
//
//         private:
//
//     };
