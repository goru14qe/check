#include "../Flow_solver.h"
#include "../Thermal_solver.h"
#include "../ALBORZ_Macros.h"
#include "../ALBORZ_SETTINGS.h"
#include "../ALBORZ_GlobalVariables.h"
#include "../Geometry.h"

template <>
void Flow_solver::LBM_CM_MRT_impl<3, 27>(const Parallel_MPI& parallel_MPI, const Thermal_solver& Thermal) {
	constexpr int D = 3;
	constexpr int Q = 27;
	const double cs2 = 1. / c_s2;
#if defined compressible
	/* rT in SI units */
	const double conv = 1. / sqr(global_parameters.D_x / global_parameters.D_t);
	const double r = R_GAS / M_av;
#endif
	std::array<double, Q> Mstar_i;

	constexpr double omega_bulk = 1.;

	non_solid_lattice.update([&, this](Flat_index idx) {
#if defined compressible
		/* Temperature in SI units */
		const double theta_temp = Thermal->temperature[idx] * Thermal->T_0;
		const double theta = r * theta_temp * conv;
#else
		constexpr double theta = 1. / 3.;
#endif
		const double omega_shear = 1. / (viscosity[idx] / theta + 0.5);

		/* GET CORRECTION */
		double dPxxx = 0, dPyyy = 0, dPzzz = 0;
		bool S = false;
		for (int alpha = 1; alpha < Q; alpha++) {
			if (is_solid[idx + c_alpha_offsets[alpha]] == TRUE) {
				S = true;
				break;
			}
		}
		if (!S) {
			for (int alpha = 1; alpha < Q; alpha++) {
				double WE = Stencil<D, Q>::w_alpha[alpha];
				const Flat_index alpha_idx = idx + c_alpha_offsets[alpha];
				double Up = velocity(alpha_idx, 0);
				double Vp = velocity(alpha_idx, 1);
				double Wp = velocity(alpha_idx, 2);
				double rhop = density[alpha_idx];
#if defined compressible
				theta_temp = Thermal->temperature[alpha_idx] * Thermal->T_0;
				const double Tp = r * theta_temp * conv;
#else
				constexpr double Tp = 1. / 3.;
#endif
				double Pxxxp = (.5) * WE * Stencil<D, Q>::c_alpha[alpha][0] * rhop * Up * (Up * Up + 3. * (Tp - 1. / 3.));
				double Pyyyp = (.5) * WE * Stencil<D, Q>::c_alpha[alpha][1] * rhop * Vp * (Vp * Vp + 3. * (Tp - 1. / 3.));
				double Pzzzp = (.5) * WE * Stencil<D, Q>::c_alpha[alpha][2] * rhop * Wp * (Wp * Wp + 3. * (Tp - 1. / 3.));
				dPxxx += Pxxxp;
				dPyyy += Pyyyp;
				dPzzz += Pzzzp;
			}
			const double scale = (2. / cs2) * (1. - omega_bulk * omega_shear / (omega_bulk + omega_shear));
			dPxxx *= scale;
			dPyyy *= scale;
			dPzzz *= scale;
		}
		velocity_corrections(idx, 0) = dPxxx;
		velocity_corrections(idx, 1) = dPyyy;
		velocity_corrections(idx, 2) = dPzzz;
	});

	non_solid_lattice.update([&, this](Flat_index idx) {
#if defined compressible
		/* Temperature in SI units */
		const double theta_temp = Thermal->temperature[idx] * Thermal->T_0;
		const double theta = r * theta_temp * conv;
#else
		constexpr double theta = 1. / 3.;
#endif
		const double omega_shear = 1. / (viscosity[idx] / theta + 0.5);
#ifndef PERFORMANCE_MODE
		std::array<double, 27> omega;
		constexpr double omega_ghost = 1.;
		omega[0] = 1.;
		omega[1] = 1.;
		omega[2] = 1.;
		omega[3] = 1.;
		omega[4] = omega_shear;
		omega[5] = omega_shear;
		omega[6] = omega_shear;
		omega[7] = omega_shear;
		omega[8] = omega_shear;
		omega[9] = omega_bulk;
		omega[10] = omega_ghost;
		omega[11] = omega_ghost;
		omega[12] = omega_ghost;
		omega[13] = omega_ghost;
		omega[14] = omega_ghost;
		omega[15] = omega_ghost;
		omega[16] = omega_ghost;
		omega[17] = omega_ghost;
		omega[18] = omega_ghost;
		omega[19] = omega_ghost;
		omega[20] = omega_ghost;
		omega[21] = omega_ghost;
		omega[22] = omega_ghost;
		omega[23] = omega_ghost;
		omega[24] = omega_ghost;
		omega[25] = omega_ghost;
		omega[26] = omega_ghost;
#endif

		const double rho = density[idx];
		const double U = velocity(idx, 0);
		const double V = velocity(idx, 1);
		const double W = velocity(idx, 2);
		const double Fx = force(idx, 0);
		const double Fy = force(idx, 1);
		const double Fz = force(idx, 2);
		const double dPxxx = velocity_corrections(idx, 0) * rho;
		const double dPyyy = velocity_corrections(idx, 1) * rho;
		const double dPzzz = velocity_corrections(idx, 2) * rho;

		const double pop0 = pop_old(idx, 0);
		const double pop1 = pop_old(idx, 1);
		const double pop2 = pop_old(idx, 2);
		const double pop3 = pop_old(idx, 3);
		const double pop4 = pop_old(idx, 4);
		const double pop5 = pop_old(idx, 5);
		const double pop6 = pop_old(idx, 6);
		const double pop7 = pop_old(idx, 7);
		const double pop8 = pop_old(idx, 8);
		const double pop9 = pop_old(idx, 9);
		const double pop10 = pop_old(idx, 10);
		const double pop11 = pop_old(idx, 11);
		const double pop12 = pop_old(idx, 12);
		const double pop13 = pop_old(idx, 13);
		const double pop14 = pop_old(idx, 14);
		const double pop15 = pop_old(idx, 15);
		const double pop16 = pop_old(idx, 16);
		const double pop17 = pop_old(idx, 17);
		const double pop18 = pop_old(idx, 18);
		const double pop19 = pop_old(idx, 19);
		const double pop20 = pop_old(idx, 20);
		const double pop21 = pop_old(idx, 21);
		const double pop22 = pop_old(idx, 22);
		const double pop23 = pop_old(idx, 23);
		const double pop24 = pop_old(idx, 24);
		const double pop25 = pop_old(idx, 25);
		const double pop26 = pop_old(idx, 26);

		const double theta_cs2_sq = sqr(theta - cs2);
		const double rho_2 = rho * rho;
		const double rho_3 = rho_2 * rho;
		const double rho_4 = rho_3 * rho;

		/// *************************************************************************************************** ///
		///                                       EQUILIBRIUM CENTRAL MOMENTS                                   ///
		///                                             MOMENT SPACE :                                          ///
		///                                                  H_0,                                               ///
		///                                             H_x, H_y, H_z                                           ///
		///                                   H_xy, H_xz, H_yz, H_xx, H_yy, H_zz                                ///
		///                               H_xxy, H_xxz, H_xyy, H_xzz, H_yzz, H_yyz, H_xyz                       ///
		///                                H_xxyy, H_xxzz, H_yyzz, H_xyzz, H_xyyz, H_xxyz                       ///
		///                                        H_xxyzz, H_xxyyz, H_xyyzz                                    ///
		///                                                 H_xxyyzz                                            ///
		/// *************************************************************************************************** ///
		/// FIRST STEP: GET CENTRAL HERMITE MOMENTS OF EDF (ASSUMING ALL TERSM SUPPORTED BY THE STENCIL ARE KEPT)
#ifdef PERFORMANCE_MODE
		Mstar_i[0] = rho;
		Mstar_i[4] = -(omega_shear - 1.0) * (pop7 * (U - 1.0) * (V - 1.0) + pop8 * (U + 1.0) * (V - 1.0) + pop9 * (U - 1.0) * (V + 1.0) + pop10 * (U + 1.0) * (V + 1.0) + pop19 * (U - 1.0) * (V - 1.0) + pop20 * (U + 1.0) * (V - 1.0) + pop21 * (U - 1.0) * (V + 1.0) + pop22 * (U + 1.0) * (V + 1.0) + pop23 * (U - 1.0) * (V - 1.0) + pop24 * (U + 1.0) * (V - 1.0) + pop25 * (U - 1.0) * (V + 1.0) + pop26 * (U + 1.0) * (V + 1.0) + U * V * pop0 + U * V * pop5 + U * V * pop6 + V * pop1 * (U - 1.0) + V * pop2 * (U + 1.0) + U * pop3 * (V - 1.0) + U * pop4 * (V + 1.0) + V * pop11 * (U - 1.0) + V * pop12 * (U + 1.0) + V * pop13 * (U - 1.0) + V * pop14 * (U + 1.0) + U * pop15 * (V - 1.0) + U * pop16 * (V + 1.0) + U * pop17 * (V - 1.0) + U * pop18 * (V + 1.0));
		Mstar_i[5] = -(omega_shear - 1.0) * (pop11 * (U - 1.0) * (W - 1.0) + pop12 * (U + 1.0) * (W - 1.0) + pop13 * (U - 1.0) * (W + 1.0) + pop14 * (U + 1.0) * (W + 1.0) + pop19 * (U - 1.0) * (W - 1.0) + pop20 * (U + 1.0) * (W - 1.0) + pop21 * (U - 1.0) * (W - 1.0) + pop22 * (U + 1.0) * (W - 1.0) + pop23 * (U - 1.0) * (W + 1.0) + pop24 * (U + 1.0) * (W + 1.0) + pop25 * (U - 1.0) * (W + 1.0) + pop26 * (U + 1.0) * (W + 1.0) + U * W * pop0 + U * W * pop3 + U * W * pop4 + W * pop1 * (U - 1.0) + W * pop2 * (U + 1.0) + U * pop5 * (W - 1.0) + U * pop6 * (W + 1.0) + W * pop7 * (U - 1.0) + W * pop8 * (U + 1.0) + W * pop9 * (U - 1.0) + W * pop10 * (U + 1.0) + U * pop15 * (W - 1.0) + U * pop16 * (W - 1.0) + U * pop17 * (W + 1.0) + U * pop18 * (W + 1.0));
		Mstar_i[6] = -(omega_shear - 1.0) * (pop15 * (V - 1.0) * (W - 1.0) + pop16 * (V + 1.0) * (W - 1.0) + pop17 * (V - 1.0) * (W + 1.0) + pop18 * (V + 1.0) * (W + 1.0) + pop19 * (V - 1.0) * (W - 1.0) + pop20 * (V - 1.0) * (W - 1.0) + pop21 * (V + 1.0) * (W - 1.0) + pop22 * (V + 1.0) * (W - 1.0) + pop23 * (V - 1.0) * (W + 1.0) + pop24 * (V - 1.0) * (W + 1.0) + pop25 * (V + 1.0) * (W + 1.0) + pop26 * (V + 1.0) * (W + 1.0) + V * W * pop0 + V * W * pop1 + V * W * pop2 + W * pop3 * (V - 1.0) + W * pop4 * (V + 1.0) + V * pop5 * (W - 1.0) + V * pop6 * (W + 1.0) + W * pop7 * (V - 1.0) + W * pop8 * (V - 1.0) + W * pop9 * (V + 1.0) + W * pop10 * (V + 1.0) + V * pop11 * (W - 1.0) + V * pop12 * (W - 1.0) + V * pop13 * (W + 1.0) + V * pop14 * (W + 1.0));
		Mstar_i[7] = -(omega_shear - 1.0) * (pop1 * (sqr(U - 1.0) - V * V) + pop2 * (sqr(U + 1.0) - V * V) - pop3 * (sqr(V - 1.0) - U * U) - pop4 * (sqr(V + 1.0) - U * U) + pop11 * (sqr(U - 1.0) - V * V) + pop12 * (sqr(U + 1.0) - V * V) + pop13 * (sqr(U - 1.0) - V * V) + pop14 * (sqr(U + 1.0) - V * V) - pop15 * (sqr(V - 1.0) - U * U) - pop16 * (sqr(V + 1.0) - U * U) - pop17 * (sqr(V - 1.0) - U * U) - pop18 * (sqr(V + 1.0) - U * U) + pop7 * (sqr(U - 1.0) - sqr(V - 1.0)) + pop8 * (sqr(U + 1.0) - sqr(V - 1.0)) + pop9 * (sqr(U - 1.0) - sqr(V + 1.0)) + pop10 * (sqr(U + 1.0) - sqr(V + 1.0)) + pop19 * (sqr(U - 1.0) - sqr(V - 1.0)) + pop20 * (sqr(U + 1.0) - sqr(V - 1.0)) + pop21 * (sqr(U - 1.0) - sqr(V + 1.0)) + pop22 * (sqr(U + 1.0) - sqr(V + 1.0)) + pop23 * (sqr(U - 1.0) - sqr(V - 1.0)) + pop24 * (sqr(U + 1.0) - sqr(V - 1.0)) + pop25 * (sqr(U - 1.0) - sqr(V + 1.0)) + pop26 * (sqr(U + 1.0) - sqr(V + 1.0)) + pop0 * (U * U - V * V) + pop5 * (U * U - V * V) + pop6 * (U * U - V * V));
		Mstar_i[8] = -(omega_shear - 1.0) * (pop1 * (sqr(U - 1.0) - W * W) + pop2 * (sqr(U + 1.0) - W * W) - pop5 * (sqr(W - 1.0) - U * U) - pop6 * (sqr(W + 1.0) - U * U) + pop7 * (sqr(U - 1.0) - W * W) + pop8 * (sqr(U + 1.0) - W * W) + pop9 * (sqr(U - 1.0) - W * W) + pop10 * (sqr(U + 1.0) - W * W) - pop15 * (sqr(W - 1.0) - U * U) - pop16 * (sqr(W - 1.0) - U * U) - pop17 * (sqr(W + 1.0) - U * U) - pop18 * (sqr(W + 1.0) - U * U) + pop11 * (sqr(U - 1.0) - sqr(W - 1.0)) + pop12 * (sqr(U + 1.0) - sqr(W - 1.0)) + pop13 * (sqr(U - 1.0) - sqr(W + 1.0)) + pop14 * (sqr(U + 1.0) - sqr(W + 1.0)) + pop19 * (sqr(U - 1.0) - sqr(W - 1.0)) + pop20 * (sqr(U + 1.0) - sqr(W - 1.0)) + pop21 * (sqr(U - 1.0) - sqr(W - 1.0)) + pop22 * (sqr(U + 1.0) - sqr(W - 1.0)) + pop23 * (sqr(U - 1.0) - sqr(W + 1.0)) + pop24 * (sqr(U + 1.0) - sqr(W + 1.0)) + pop25 * (sqr(U - 1.0) - sqr(W + 1.0)) + pop26 * (sqr(U + 1.0) - sqr(W + 1.0)) + pop0 * (U * U - W * W) + pop3 * (U * U - W * W) + pop4 * (U * U - W * W));
		Mstar_i[9] = rho * (theta - cs2) * 3.0;
		Mstar_i[17] = rho * theta_cs2_sq;
		Mstar_i[18] = rho * theta_cs2_sq;
		Mstar_i[19] = rho * theta_cs2_sq;
		Mstar_i[26] = rho * pow(theta - cs2, 3.0);

		Mstar_i[1] = Fx;
		Mstar_i[2] = Fy;
		Mstar_i[3] = Fz;
		Mstar_i[4] += (Fx * Fy) / rho;
		Mstar_i[5] += (Fx * Fz) / rho;
		Mstar_i[6] += (Fy * Fz) / rho;
		Mstar_i[7] += (dPxxx - dPyyy + Fx * Fx - Fy * Fy) / rho;
		Mstar_i[8] += (dPxxx - dPzzz + Fx * Fx - Fz * Fz) / rho;
		Mstar_i[9] += (dPxxx + dPyyy + dPzzz + Fx * Fx + Fy * Fy + Fz * Fz) / rho;
		Mstar_i[10] = Fy * 1.0 / (rho_2) * (dPxxx + (rho_2)*theta - (rho_2)*cs2 + Fx * Fx);
		Mstar_i[11] = Fz * 1.0 / (rho_2) * (dPxxx + (rho_2)*theta - (rho_2)*cs2 + Fx * Fx);
		Mstar_i[12] = Fx * 1.0 / (rho_2) * (dPyyy + (rho_2)*theta - (rho_2)*cs2 + Fy * Fy);
		Mstar_i[13] = Fz * 1.0 / (rho_2) * (dPyyy + (rho_2)*theta - (rho_2)*cs2 + Fy * Fy);
		Mstar_i[14] = Fx * 1.0 / (rho_2) * (dPzzz + (rho_2)*theta - (rho_2)*cs2 + Fz * Fz);
		Mstar_i[15] = Fy * 1.0 / (rho_2) * (dPzzz + (rho_2)*theta - (rho_2)*cs2 + Fz * Fz);
		Mstar_i[16] = Fx * Fy * Fz * 1.0 / (rho_2);
		Mstar_i[17] += 0 - rho * theta_cs2_sq + 1.0 / (rho_3) * (dPxxx + (rho_2)*theta - (rho_2)*cs2 + Fx * Fx) * (dPyyy + (rho_2)*theta - (rho_2)*cs2 + Fy * Fy);
		Mstar_i[18] += 0 - rho * theta_cs2_sq + 1.0 / (rho_3) * (dPxxx + (rho_2)*theta - (rho_2)*cs2 + Fx * Fx) * (dPzzz + (rho_2)*theta - (rho_2)*cs2 + Fz * Fz);
		Mstar_i[19] += 0 - rho * theta_cs2_sq + 1.0 / (rho_3) * (dPyyy + (rho_2)*theta - (rho_2)*cs2 + Fy * Fy) * (dPzzz + (rho_2)*theta - (rho_2)*cs2 + Fz * Fz);
		Mstar_i[20] = Fy * Fz * 1.0 / (rho_3) * (dPxxx + (rho_2)*theta - (rho_2)*cs2 + Fx * Fx);
		Mstar_i[21] = Fx * Fz * 1.0 / (rho_3) * (dPyyy + (rho_2)*theta - (rho_2)*cs2 + Fy * Fy);
		Mstar_i[22] = Fx * Fy * 1.0 / (rho_3) * (dPzzz + (rho_2)*theta - (rho_2)*cs2 + Fz * Fz);
		Mstar_i[23] = Fz * 1.0 / (rho_4) * (dPxxx + (rho_2)*theta - (rho_2)*cs2 + Fx * Fx) * (dPyyy + (rho_2)*theta - (rho_2)*cs2 + Fy * Fy);
		Mstar_i[24] = Fy * 1.0 / (rho_4) * (dPxxx + (rho_2)*theta - (rho_2)*cs2 + Fx * Fx) * (dPzzz + (rho_2)*theta - (rho_2)*cs2 + Fz * Fz);
		Mstar_i[25] = Fx * 1.0 / (rho_4) * (dPyyy + (rho_2)*theta - (rho_2)*cs2 + Fy * Fy) * (dPzzz + (rho_2)*theta - (rho_2)*cs2 + Fz * Fz);
		Mstar_i[26] += 0 - rho * theta_cs2_sq * (theta - cs2) + 1.0 / (rho_4 * rho) * (dPxxx + (rho_2)*theta - (rho_2)*cs2 + Fx * Fx) * (dPyyy + (rho_2)*theta - (rho_2)*cs2 + Fy * Fy) * (dPzzz + (rho_2)*theta - (rho_2)*cs2 + Fz * Fz);
#else
		Mstar_i[0] = rho * omega[0] - (omega[0] - 1.0) * (pop0 + pop1 + pop2 + pop3 + pop4 + pop5 + pop6 + pop7 + pop8 + pop9 + pop10 + pop11 + pop12 + pop13 + pop14 + pop15 + pop16 + pop17 + pop18 + pop19 + pop20 + pop21 + pop22 + pop23 + pop24 + pop25 + pop26);
		Mstar_i[1] = (omega[1] - 1.0) * (U * pop0 + U * pop3 + U * pop4 + U * pop5 + U * pop6 + U * pop15 + U * pop16 + U * pop17 + U * pop18 + pop1 * (U - 1.0) + pop2 * (U + 1.0) + pop7 * (U - 1.0) + pop8 * (U + 1.0) + pop9 * (U - 1.0) + pop10 * (U + 1.0) + pop11 * (U - 1.0) + pop12 * (U + 1.0) + pop13 * (U - 1.0) + pop14 * (U + 1.0) + pop19 * (U - 1.0) + pop20 * (U + 1.0) + pop21 * (U - 1.0) + pop22 * (U + 1.0) + pop23 * (U - 1.0) + pop24 * (U + 1.0) + pop25 * (U - 1.0) + pop26 * (U + 1.0));
		Mstar_i[2] = (omega[2] - 1.0) * (V * pop0 + V * pop1 + V * pop2 + V * pop5 + V * pop6 + V * pop11 + V * pop12 + V * pop13 + V * pop14 + pop3 * (V - 1.0) + pop4 * (V + 1.0) + pop7 * (V - 1.0) + pop8 * (V - 1.0) + pop9 * (V + 1.0) + pop10 * (V + 1.0) + pop15 * (V - 1.0) + pop16 * (V + 1.0) + pop17 * (V - 1.0) + pop18 * (V + 1.0) + pop19 * (V - 1.0) + pop20 * (V - 1.0) + pop21 * (V + 1.0) + pop22 * (V + 1.0) + pop23 * (V - 1.0) + pop24 * (V - 1.0) + pop25 * (V + 1.0) + pop26 * (V + 1.0));
		Mstar_i[3] = (omega[3] - 1.0) * (W * pop0 + W * pop1 + W * pop2 + W * pop3 + W * pop4 + W * pop7 + W * pop8 + W * pop9 + W * pop10 + pop5 * (W - 1.0) + pop6 * (W + 1.0) + pop11 * (W - 1.0) + pop12 * (W - 1.0) + pop13 * (W + 1.0) + pop14 * (W + 1.0) + pop15 * (W - 1.0) + pop16 * (W - 1.0) + pop17 * (W + 1.0) + pop18 * (W + 1.0) + pop19 * (W - 1.0) + pop20 * (W - 1.0) + pop21 * (W - 1.0) + pop22 * (W - 1.0) + pop23 * (W + 1.0) + pop24 * (W + 1.0) + pop25 * (W + 1.0) + pop26 * (W + 1.0));
		Mstar_i[4] = -(omega[4] - 1.0) * (pop7 * (U - 1.0) * (V - 1.0) + pop8 * (U + 1.0) * (V - 1.0) + pop9 * (U - 1.0) * (V + 1.0) + pop10 * (U + 1.0) * (V + 1.0) + pop19 * (U - 1.0) * (V - 1.0) + pop20 * (U + 1.0) * (V - 1.0) + pop21 * (U - 1.0) * (V + 1.0) + pop22 * (U + 1.0) * (V + 1.0) + pop23 * (U - 1.0) * (V - 1.0) + pop24 * (U + 1.0) * (V - 1.0) + pop25 * (U - 1.0) * (V + 1.0) + pop26 * (U + 1.0) * (V + 1.0) + U * V * pop0 + U * V * pop5 + U * V * pop6 + V * pop1 * (U - 1.0) + V * pop2 * (U + 1.0) + U * pop3 * (V - 1.0) + U * pop4 * (V + 1.0) + V * pop11 * (U - 1.0) + V * pop12 * (U + 1.0) + V * pop13 * (U - 1.0) + V * pop14 * (U + 1.0) + U * pop15 * (V - 1.0) + U * pop16 * (V + 1.0) + U * pop17 * (V - 1.0) + U * pop18 * (V + 1.0));
		Mstar_i[5] = -(omega[5] - 1.0) * (pop11 * (U - 1.0) * (W - 1.0) + pop12 * (U + 1.0) * (W - 1.0) + pop13 * (U - 1.0) * (W + 1.0) + pop14 * (U + 1.0) * (W + 1.0) + pop19 * (U - 1.0) * (W - 1.0) + pop20 * (U + 1.0) * (W - 1.0) + pop21 * (U - 1.0) * (W - 1.0) + pop22 * (U + 1.0) * (W - 1.0) + pop23 * (U - 1.0) * (W + 1.0) + pop24 * (U + 1.0) * (W + 1.0) + pop25 * (U - 1.0) * (W + 1.0) + pop26 * (U + 1.0) * (W + 1.0) + U * W * pop0 + U * W * pop3 + U * W * pop4 + W * pop1 * (U - 1.0) + W * pop2 * (U + 1.0) + U * pop5 * (W - 1.0) + U * pop6 * (W + 1.0) + W * pop7 * (U - 1.0) + W * pop8 * (U + 1.0) + W * pop9 * (U - 1.0) + W * pop10 * (U + 1.0) + U * pop15 * (W - 1.0) + U * pop16 * (W - 1.0) + U * pop17 * (W + 1.0) + U * pop18 * (W + 1.0));
		Mstar_i[6] = -(omega[6] - 1.0) * (pop15 * (V - 1.0) * (W - 1.0) + pop16 * (V + 1.0) * (W - 1.0) + pop17 * (V - 1.0) * (W + 1.0) + pop18 * (V + 1.0) * (W + 1.0) + pop19 * (V - 1.0) * (W - 1.0) + pop20 * (V - 1.0) * (W - 1.0) + pop21 * (V + 1.0) * (W - 1.0) + pop22 * (V + 1.0) * (W - 1.0) + pop23 * (V - 1.0) * (W + 1.0) + pop24 * (V - 1.0) * (W + 1.0) + pop25 * (V + 1.0) * (W + 1.0) + pop26 * (V + 1.0) * (W + 1.0) + V * W * pop0 + V * W * pop1 + V * W * pop2 + W * pop3 * (V - 1.0) + W * pop4 * (V + 1.0) + V * pop5 * (W - 1.0) + V * pop6 * (W + 1.0) + W * pop7 * (V - 1.0) + W * pop8 * (V - 1.0) + W * pop9 * (V + 1.0) + W * pop10 * (V + 1.0) + V * pop11 * (W - 1.0) + V * pop12 * (W - 1.0) + V * pop13 * (W + 1.0) + V * pop14 * (W + 1.0));
		Mstar_i[7] = -(omega[7] - 1.0) * (pop1 * (sqr(U - 1.0) - V * V) + pop2 * (sqr(U + 1.0) - V * V) - pop3 * (sqr(V - 1.0) - U * U) - pop4 * (sqr(V + 1.0) - U * U) + pop11 * (sqr(U - 1.0) - V * V) + pop12 * (sqr(U + 1.0) - V * V) + pop13 * (sqr(U - 1.0) - V * V) + pop14 * (sqr(U + 1.0) - V * V) - pop15 * (sqr(V - 1.0) - U * U) - pop16 * (sqr(V + 1.0) - U * U) - pop17 * (sqr(V - 1.0) - U * U) - pop18 * (sqr(V + 1.0) - U * U) + pop7 * (sqr(U - 1.0) - sqr(V - 1.0)) + pop8 * (sqr(U + 1.0) - sqr(V - 1.0)) + pop9 * (sqr(U - 1.0) - sqr(V + 1.0)) + pop10 * (sqr(U + 1.0) - sqr(V + 1.0)) + pop19 * (sqr(U - 1.0) - sqr(V - 1.0)) + pop20 * (sqr(U + 1.0) - sqr(V - 1.0)) + pop21 * (sqr(U - 1.0) - sqr(V + 1.0)) + pop22 * (sqr(U + 1.0) - sqr(V + 1.0)) + pop23 * (sqr(U - 1.0) - sqr(V - 1.0)) + pop24 * (sqr(U + 1.0) - sqr(V - 1.0)) + pop25 * (sqr(U - 1.0) - sqr(V + 1.0)) + pop26 * (sqr(U + 1.0) - sqr(V + 1.0)) + pop0 * (U * U - V * V) + pop5 * (U * U - V * V) + pop6 * (U * U - V * V));
		Mstar_i[8] = -(omega[8] - 1.0) * (pop1 * (sqr(U - 1.0) - W * W) + pop2 * (sqr(U + 1.0) - W * W) - pop5 * (sqr(W - 1.0) - U * U) - pop6 * (sqr(W + 1.0) - U * U) + pop7 * (sqr(U - 1.0) - W * W) + pop8 * (sqr(U + 1.0) - W * W) + pop9 * (sqr(U - 1.0) - W * W) + pop10 * (sqr(U + 1.0) - W * W) - pop15 * (sqr(W - 1.0) - U * U) - pop16 * (sqr(W - 1.0) - U * U) - pop17 * (sqr(W + 1.0) - U * U) - pop18 * (sqr(W + 1.0) - U * U) + pop11 * (sqr(U - 1.0) - sqr(W - 1.0)) + pop12 * (sqr(U + 1.0) - sqr(W - 1.0)) + pop13 * (sqr(U - 1.0) - sqr(W + 1.0)) + pop14 * (sqr(U + 1.0) - sqr(W + 1.0)) + pop19 * (sqr(U - 1.0) - sqr(W - 1.0)) + pop20 * (sqr(U + 1.0) - sqr(W - 1.0)) + pop21 * (sqr(U - 1.0) - sqr(W - 1.0)) + pop22 * (sqr(U + 1.0) - sqr(W - 1.0)) + pop23 * (sqr(U - 1.0) - sqr(W + 1.0)) + pop24 * (sqr(U + 1.0) - sqr(W + 1.0)) + pop25 * (sqr(U - 1.0) - sqr(W + 1.0)) + pop26 * (sqr(U + 1.0) - sqr(W + 1.0)) + pop0 * (U * U - W * W) + pop3 * (U * U - W * W) + pop4 * (U * U - W * W));
		Mstar_i[9] = -(omega[9] - 1.0) * (pop0 * (cs2 * -3.0 + U * U + V * V + W * W) + pop19 * (cs2 * -3.0 + sqr(U - 1.0) + sqr(V - 1.0) + sqr(W - 1.0)) + pop20 * (cs2 * -3.0 + sqr(U + 1.0) + sqr(V - 1.0) + sqr(W - 1.0)) + pop21 * (cs2 * -3.0 + sqr(U - 1.0) + sqr(V + 1.0) + sqr(W - 1.0)) + pop22 * (cs2 * -3.0 + sqr(U + 1.0) + sqr(V + 1.0) + sqr(W - 1.0)) + pop23 * (cs2 * -3.0 + sqr(U - 1.0) + sqr(V - 1.0) + sqr(W + 1.0)) + pop24 * (cs2 * -3.0 + sqr(U + 1.0) + sqr(V - 1.0) + sqr(W + 1.0)) + pop25 * (cs2 * -3.0 + sqr(U - 1.0) + sqr(V + 1.0) + sqr(W + 1.0)) + pop26 * (cs2 * -3.0 + sqr(U + 1.0) + sqr(V + 1.0) + sqr(W + 1.0)) + pop1 * (cs2 * -3.0 + sqr(U - 1.0) + V * V + W * W) + pop2 * (cs2 * -3.0 + sqr(U + 1.0) + V * V + W * W) + pop3 * (cs2 * -3.0 + sqr(V - 1.0) + U * U + W * W) + pop4 * (cs2 * -3.0 + sqr(V + 1.0) + U * U + W * W) + pop5 * (cs2 * -3.0 + sqr(W - 1.0) + U * U + V * V) + pop6 * (cs2 * -3.0 + sqr(W + 1.0) + U * U + V * V) + pop7 * (cs2 * -3.0 + sqr(U - 1.0) + sqr(V - 1.0) + W * W) + pop8 * (cs2 * -3.0 + sqr(U + 1.0) + sqr(V - 1.0) + W * W) + pop9 * (cs2 * -3.0 + sqr(U - 1.0) + sqr(V + 1.0) + W * W) + pop10 * (cs2 * -3.0 + sqr(U + 1.0) + sqr(V + 1.0) + W * W) + pop11 * (cs2 * -3.0 + sqr(U - 1.0) + sqr(W - 1.0) + V * V) + pop12 * (cs2 * -3.0 + sqr(U + 1.0) + sqr(W - 1.0) + V * V) + pop13 * (cs2 * -3.0 + sqr(U - 1.0) + sqr(W + 1.0) + V * V) + pop14 * (cs2 * -3.0 + sqr(U + 1.0) + sqr(W + 1.0) + V * V) + pop15 * (cs2 * -3.0 + sqr(V - 1.0) + sqr(W - 1.0) + U * U) + pop16 * (cs2 * -3.0 + sqr(V + 1.0) + sqr(W - 1.0) + U * U) + pop17 * (cs2 * -3.0 + sqr(V - 1.0) + sqr(W + 1.0) + U * U) + pop18 * (cs2 * -3.0 + sqr(V + 1.0) + sqr(W + 1.0) + U * U)) + rho * omega[9] * (theta - cs2) * 3.0;
		Mstar_i[10] = -(omega[10] - 1.0) * (V * pop1 * (cs2 - sqr(U - 1.0)) + V * pop2 * (cs2 - sqr(U + 1.0)) + pop3 * (V - 1.0) * (cs2 - U * U) + pop4 * (V + 1.0) * (cs2 - U * U) + V * pop11 * (cs2 - sqr(U - 1.0)) + V * pop12 * (cs2 - sqr(U + 1.0)) + V * pop13 * (cs2 - sqr(U - 1.0)) + V * pop14 * (cs2 - sqr(U + 1.0)) + pop15 * (V - 1.0) * (cs2 - U * U) + pop16 * (V + 1.0) * (cs2 - U * U) + pop17 * (V - 1.0) * (cs2 - U * U) + pop18 * (V + 1.0) * (cs2 - U * U) + pop7 * (cs2 - sqr(U - 1.0)) * (V - 1.0) + pop8 * (cs2 - sqr(U + 1.0)) * (V - 1.0) + pop9 * (cs2 - sqr(U - 1.0)) * (V + 1.0) + pop10 * (cs2 - sqr(U + 1.0)) * (V + 1.0) + pop19 * (cs2 - sqr(U - 1.0)) * (V - 1.0) + pop20 * (cs2 - sqr(U + 1.0)) * (V - 1.0) + pop21 * (cs2 - sqr(U - 1.0)) * (V + 1.0) + pop22 * (cs2 - sqr(U + 1.0)) * (V + 1.0) + pop23 * (cs2 - sqr(U - 1.0)) * (V - 1.0) + pop24 * (cs2 - sqr(U + 1.0)) * (V - 1.0) + pop25 * (cs2 - sqr(U - 1.0)) * (V + 1.0) + pop26 * (cs2 - sqr(U + 1.0)) * (V + 1.0) + V * pop0 * (cs2 - U * U) + V * pop5 * (cs2 - U * U) + V * pop6 * (cs2 - U * U));
		Mstar_i[11] = -(omega[11] - 1.0) * (W * pop1 * (cs2 - sqr(U - 1.0)) + W * pop2 * (cs2 - sqr(U + 1.0)) + pop5 * (W - 1.0) * (cs2 - U * U) + pop6 * (W + 1.0) * (cs2 - U * U) + W * pop7 * (cs2 - sqr(U - 1.0)) + W * pop8 * (cs2 - sqr(U + 1.0)) + W * pop9 * (cs2 - sqr(U - 1.0)) + W * pop10 * (cs2 - sqr(U + 1.0)) + pop15 * (W - 1.0) * (cs2 - U * U) + pop16 * (W - 1.0) * (cs2 - U * U) + pop17 * (W + 1.0) * (cs2 - U * U) + pop18 * (W + 1.0) * (cs2 - U * U) + pop11 * (cs2 - sqr(U - 1.0)) * (W - 1.0) + pop12 * (cs2 - sqr(U + 1.0)) * (W - 1.0) + pop13 * (cs2 - sqr(U - 1.0)) * (W + 1.0) + pop14 * (cs2 - sqr(U + 1.0)) * (W + 1.0) + pop19 * (cs2 - sqr(U - 1.0)) * (W - 1.0) + pop20 * (cs2 - sqr(U + 1.0)) * (W - 1.0) + pop21 * (cs2 - sqr(U - 1.0)) * (W - 1.0) + pop22 * (cs2 - sqr(U + 1.0)) * (W - 1.0) + pop23 * (cs2 - sqr(U - 1.0)) * (W + 1.0) + pop24 * (cs2 - sqr(U + 1.0)) * (W + 1.0) + pop25 * (cs2 - sqr(U - 1.0)) * (W + 1.0) + pop26 * (cs2 - sqr(U + 1.0)) * (W + 1.0) + W * pop0 * (cs2 - U * U) + W * pop3 * (cs2 - U * U) + W * pop4 * (cs2 - U * U));
		Mstar_i[12] = -(omega[12] - 1.0) * (pop1 * (U - 1.0) * (cs2 - V * V) + pop2 * (U + 1.0) * (cs2 - V * V) + U * pop3 * (cs2 - sqr(V - 1.0)) + U * pop4 * (cs2 - sqr(V + 1.0)) + pop11 * (U - 1.0) * (cs2 - V * V) + pop12 * (U + 1.0) * (cs2 - V * V) + pop13 * (U - 1.0) * (cs2 - V * V) + pop14 * (U + 1.0) * (cs2 - V * V) + U * pop15 * (cs2 - sqr(V - 1.0)) + U * pop16 * (cs2 - sqr(V + 1.0)) + U * pop17 * (cs2 - sqr(V - 1.0)) + U * pop18 * (cs2 - sqr(V + 1.0)) + pop7 * (cs2 - sqr(V - 1.0)) * (U - 1.0) + pop8 * (cs2 - sqr(V - 1.0)) * (U + 1.0) + pop9 * (cs2 - sqr(V + 1.0)) * (U - 1.0) + pop10 * (cs2 - sqr(V + 1.0)) * (U + 1.0) + pop19 * (cs2 - sqr(V - 1.0)) * (U - 1.0) + pop20 * (cs2 - sqr(V - 1.0)) * (U + 1.0) + pop21 * (cs2 - sqr(V + 1.0)) * (U - 1.0) + pop22 * (cs2 - sqr(V + 1.0)) * (U + 1.0) + pop23 * (cs2 - sqr(V - 1.0)) * (U - 1.0) + pop24 * (cs2 - sqr(V - 1.0)) * (U + 1.0) + pop25 * (cs2 - sqr(V + 1.0)) * (U - 1.0) + pop26 * (cs2 - sqr(V + 1.0)) * (U + 1.0) + U * pop0 * (cs2 - V * V) + U * pop5 * (cs2 - V * V) + U * pop6 * (cs2 - V * V));
		Mstar_i[13] = -(omega[13] - 1.0) * (W * pop3 * (cs2 - sqr(V - 1.0)) + W * pop4 * (cs2 - sqr(V + 1.0)) + pop5 * (W - 1.0) * (cs2 - V * V) + pop6 * (W + 1.0) * (cs2 - V * V) + W * pop7 * (cs2 - sqr(V - 1.0)) + W * pop8 * (cs2 - sqr(V - 1.0)) + W * pop9 * (cs2 - sqr(V + 1.0)) + W * pop10 * (cs2 - sqr(V + 1.0)) + pop11 * (W - 1.0) * (cs2 - V * V) + pop12 * (W - 1.0) * (cs2 - V * V) + pop13 * (W + 1.0) * (cs2 - V * V) + pop14 * (W + 1.0) * (cs2 - V * V) + pop15 * (cs2 - sqr(V - 1.0)) * (W - 1.0) + pop16 * (cs2 - sqr(V + 1.0)) * (W - 1.0) + pop17 * (cs2 - sqr(V - 1.0)) * (W + 1.0) + pop18 * (cs2 - sqr(V + 1.0)) * (W + 1.0) + pop19 * (cs2 - sqr(V - 1.0)) * (W - 1.0) + pop20 * (cs2 - sqr(V - 1.0)) * (W - 1.0) + pop21 * (cs2 - sqr(V + 1.0)) * (W - 1.0) + pop22 * (cs2 - sqr(V + 1.0)) * (W - 1.0) + pop23 * (cs2 - sqr(V - 1.0)) * (W + 1.0) + pop24 * (cs2 - sqr(V - 1.0)) * (W + 1.0) + pop25 * (cs2 - sqr(V + 1.0)) * (W + 1.0) + pop26 * (cs2 - sqr(V + 1.0)) * (W + 1.0) + W * pop0 * (cs2 - V * V) + W * pop1 * (cs2 - V * V) + W * pop2 * (cs2 - V * V));
		Mstar_i[14] = -(omega[14] - 1.0) * (pop1 * (U - 1.0) * (cs2 - W * W) + pop2 * (U + 1.0) * (cs2 - W * W) + U * pop5 * (cs2 - sqr(W - 1.0)) + U * pop6 * (cs2 - sqr(W + 1.0)) + pop7 * (U - 1.0) * (cs2 - W * W) + pop8 * (U + 1.0) * (cs2 - W * W) + pop9 * (U - 1.0) * (cs2 - W * W) + pop10 * (U + 1.0) * (cs2 - W * W) + U * pop15 * (cs2 - sqr(W - 1.0)) + U * pop16 * (cs2 - sqr(W - 1.0)) + U * pop17 * (cs2 - sqr(W + 1.0)) + U * pop18 * (cs2 - sqr(W + 1.0)) + pop11 * (cs2 - sqr(W - 1.0)) * (U - 1.0) + pop12 * (cs2 - sqr(W - 1.0)) * (U + 1.0) + pop13 * (cs2 - sqr(W + 1.0)) * (U - 1.0) + pop14 * (cs2 - sqr(W + 1.0)) * (U + 1.0) + pop19 * (cs2 - sqr(W - 1.0)) * (U - 1.0) + pop20 * (cs2 - sqr(W - 1.0)) * (U + 1.0) + pop21 * (cs2 - sqr(W - 1.0)) * (U - 1.0) + pop22 * (cs2 - sqr(W - 1.0)) * (U + 1.0) + pop23 * (cs2 - sqr(W + 1.0)) * (U - 1.0) + pop24 * (cs2 - sqr(W + 1.0)) * (U + 1.0) + pop25 * (cs2 - sqr(W + 1.0)) * (U - 1.0) + pop26 * (cs2 - sqr(W + 1.0)) * (U + 1.0) + U * pop0 * (cs2 - W * W) + U * pop3 * (cs2 - W * W) + U * pop4 * (cs2 - W * W));
		Mstar_i[15] = -(omega[15] - 1.0) * (pop3 * (V - 1.0) * (cs2 - W * W) + pop4 * (V + 1.0) * (cs2 - W * W) + V * pop5 * (cs2 - sqr(W - 1.0)) + V * pop6 * (cs2 - sqr(W + 1.0)) + pop7 * (V - 1.0) * (cs2 - W * W) + pop8 * (V - 1.0) * (cs2 - W * W) + pop9 * (V + 1.0) * (cs2 - W * W) + pop10 * (V + 1.0) * (cs2 - W * W) + V * pop11 * (cs2 - sqr(W - 1.0)) + V * pop12 * (cs2 - sqr(W - 1.0)) + V * pop13 * (cs2 - sqr(W + 1.0)) + V * pop14 * (cs2 - sqr(W + 1.0)) + pop15 * (cs2 - sqr(W - 1.0)) * (V - 1.0) + pop16 * (cs2 - sqr(W - 1.0)) * (V + 1.0) + pop17 * (cs2 - sqr(W + 1.0)) * (V - 1.0) + pop18 * (cs2 - sqr(W + 1.0)) * (V + 1.0) + pop19 * (cs2 - sqr(W - 1.0)) * (V - 1.0) + pop20 * (cs2 - sqr(W - 1.0)) * (V - 1.0) + pop21 * (cs2 - sqr(W - 1.0)) * (V + 1.0) + pop22 * (cs2 - sqr(W - 1.0)) * (V + 1.0) + pop23 * (cs2 - sqr(W + 1.0)) * (V - 1.0) + pop24 * (cs2 - sqr(W + 1.0)) * (V - 1.0) + pop25 * (cs2 - sqr(W + 1.0)) * (V + 1.0) + pop26 * (cs2 - sqr(W + 1.0)) * (V + 1.0) + V * pop0 * (cs2 - W * W) + V * pop1 * (cs2 - W * W) + V * pop2 * (cs2 - W * W));
		Mstar_i[16] = (omega[16] - 1.0) * (V * W * pop1 * (U - 1.0) + V * W * pop2 * (U + 1.0) + U * W * pop3 * (V - 1.0) + U * W * pop4 * (V + 1.0) + U * V * pop5 * (W - 1.0) + U * V * pop6 * (W + 1.0) + W * pop7 * (U - 1.0) * (V - 1.0) + W * pop8 * (U + 1.0) * (V - 1.0) + W * pop9 * (U - 1.0) * (V + 1.0) + W * pop10 * (U + 1.0) * (V + 1.0) + V * pop11 * (U - 1.0) * (W - 1.0) + V * pop12 * (U + 1.0) * (W - 1.0) + V * pop13 * (U - 1.0) * (W + 1.0) + V * pop14 * (U + 1.0) * (W + 1.0) + U * pop15 * (V - 1.0) * (W - 1.0) + U * pop16 * (V + 1.0) * (W - 1.0) + U * pop17 * (V - 1.0) * (W + 1.0) + U * pop18 * (V + 1.0) * (W + 1.0) + U * V * W * pop0 + pop19 * (U - 1.0) * (V - 1.0) * (W - 1.0) + pop20 * (U + 1.0) * (V - 1.0) * (W - 1.0) + pop21 * (U - 1.0) * (V + 1.0) * (W - 1.0) + pop22 * (U + 1.0) * (V + 1.0) * (W - 1.0) + pop23 * (U - 1.0) * (V - 1.0) * (W + 1.0) + pop24 * (U + 1.0) * (V - 1.0) * (W + 1.0) + pop25 * (U - 1.0) * (V + 1.0) * (W + 1.0) + pop26 * (U + 1.0) * (V + 1.0) * (W + 1.0));
		Mstar_i[17] = -(omega[17] - 1.0) * (pop7 * (cs2 - sqr(U - 1.0)) * (cs2 - sqr(V - 1.0)) + pop8 * (cs2 - sqr(U + 1.0)) * (cs2 - sqr(V - 1.0)) + pop9 * (cs2 - sqr(U - 1.0)) * (cs2 - sqr(V + 1.0)) + pop10 * (cs2 - sqr(U + 1.0)) * (cs2 - sqr(V + 1.0)) + pop19 * (cs2 - sqr(U - 1.0)) * (cs2 - sqr(V - 1.0)) + pop20 * (cs2 - sqr(U + 1.0)) * (cs2 - sqr(V - 1.0)) + pop21 * (cs2 - sqr(U - 1.0)) * (cs2 - sqr(V + 1.0)) + pop22 * (cs2 - sqr(U + 1.0)) * (cs2 - sqr(V + 1.0)) + pop23 * (cs2 - sqr(U - 1.0)) * (cs2 - sqr(V - 1.0)) + pop24 * (cs2 - sqr(U + 1.0)) * (cs2 - sqr(V - 1.0)) + pop25 * (cs2 - sqr(U - 1.0)) * (cs2 - sqr(V + 1.0)) + pop26 * (cs2 - sqr(U + 1.0)) * (cs2 - sqr(V + 1.0)) + pop0 * (cs2 - U * U) * (cs2 - V * V) + pop5 * (cs2 - U * U) * (cs2 - V * V) + pop6 * (cs2 - U * U) * (cs2 - V * V) + pop1 * (cs2 - sqr(U - 1.0)) * (cs2 - V * V) + pop2 * (cs2 - sqr(U + 1.0)) * (cs2 - V * V) + pop3 * (cs2 - sqr(V - 1.0)) * (cs2 - U * U) + pop4 * (cs2 - sqr(V + 1.0)) * (cs2 - U * U) + pop11 * (cs2 - sqr(U - 1.0)) * (cs2 - V * V) + pop12 * (cs2 - sqr(U + 1.0)) * (cs2 - V * V) + pop13 * (cs2 - sqr(U - 1.0)) * (cs2 - V * V) + pop14 * (cs2 - sqr(U + 1.0)) * (cs2 - V * V) + pop15 * (cs2 - sqr(V - 1.0)) * (cs2 - U * U) + pop16 * (cs2 - sqr(V + 1.0)) * (cs2 - U * U) + pop17 * (cs2 - sqr(V - 1.0)) * (cs2 - U * U) + pop18 * (cs2 - sqr(V + 1.0)) * (cs2 - U * U)) + rho * omega[17] * theta_cs2_sq;
		Mstar_i[18] = -(omega[18] - 1.0) * (pop11 * (cs2 - sqr(U - 1.0)) * (cs2 - sqr(W - 1.0)) + pop12 * (cs2 - sqr(U + 1.0)) * (cs2 - sqr(W - 1.0)) + pop13 * (cs2 - sqr(U - 1.0)) * (cs2 - sqr(W + 1.0)) + pop14 * (cs2 - sqr(U + 1.0)) * (cs2 - sqr(W + 1.0)) + pop19 * (cs2 - sqr(U - 1.0)) * (cs2 - sqr(W - 1.0)) + pop20 * (cs2 - sqr(U + 1.0)) * (cs2 - sqr(W - 1.0)) + pop21 * (cs2 - sqr(U - 1.0)) * (cs2 - sqr(W - 1.0)) + pop22 * (cs2 - sqr(U + 1.0)) * (cs2 - sqr(W - 1.0)) + pop23 * (cs2 - sqr(U - 1.0)) * (cs2 - sqr(W + 1.0)) + pop24 * (cs2 - sqr(U + 1.0)) * (cs2 - sqr(W + 1.0)) + pop25 * (cs2 - sqr(U - 1.0)) * (cs2 - sqr(W + 1.0)) + pop26 * (cs2 - sqr(U + 1.0)) * (cs2 - sqr(W + 1.0)) + pop0 * (cs2 - U * U) * (cs2 - W * W) + pop3 * (cs2 - U * U) * (cs2 - W * W) + pop4 * (cs2 - U * U) * (cs2 - W * W) + pop1 * (cs2 - sqr(U - 1.0)) * (cs2 - W * W) + pop2 * (cs2 - sqr(U + 1.0)) * (cs2 - W * W) + pop5 * (cs2 - sqr(W - 1.0)) * (cs2 - U * U) + pop6 * (cs2 - sqr(W + 1.0)) * (cs2 - U * U) + pop7 * (cs2 - sqr(U - 1.0)) * (cs2 - W * W) + pop8 * (cs2 - sqr(U + 1.0)) * (cs2 - W * W) + pop9 * (cs2 - sqr(U - 1.0)) * (cs2 - W * W) + pop10 * (cs2 - sqr(U + 1.0)) * (cs2 - W * W) + pop15 * (cs2 - sqr(W - 1.0)) * (cs2 - U * U) + pop16 * (cs2 - sqr(W - 1.0)) * (cs2 - U * U) + pop17 * (cs2 - sqr(W + 1.0)) * (cs2 - U * U) + pop18 * (cs2 - sqr(W + 1.0)) * (cs2 - U * U)) + rho * omega[18] * theta_cs2_sq;
		Mstar_i[19] = -(omega[19] - 1.0) * (pop15 * (cs2 - sqr(V - 1.0)) * (cs2 - sqr(W - 1.0)) + pop16 * (cs2 - sqr(V + 1.0)) * (cs2 - sqr(W - 1.0)) + pop17 * (cs2 - sqr(V - 1.0)) * (cs2 - sqr(W + 1.0)) + pop18 * (cs2 - sqr(V + 1.0)) * (cs2 - sqr(W + 1.0)) + pop19 * (cs2 - sqr(V - 1.0)) * (cs2 - sqr(W - 1.0)) + pop20 * (cs2 - sqr(V - 1.0)) * (cs2 - sqr(W - 1.0)) + pop21 * (cs2 - sqr(V + 1.0)) * (cs2 - sqr(W - 1.0)) + pop22 * (cs2 - sqr(V + 1.0)) * (cs2 - sqr(W - 1.0)) + pop23 * (cs2 - sqr(V - 1.0)) * (cs2 - sqr(W + 1.0)) + pop24 * (cs2 - sqr(V - 1.0)) * (cs2 - sqr(W + 1.0)) + pop25 * (cs2 - sqr(V + 1.0)) * (cs2 - sqr(W + 1.0)) + pop26 * (cs2 - sqr(V + 1.0)) * (cs2 - sqr(W + 1.0)) + pop0 * (cs2 - V * V) * (cs2 - W * W) + pop1 * (cs2 - V * V) * (cs2 - W * W) + pop2 * (cs2 - V * V) * (cs2 - W * W) + pop3 * (cs2 - sqr(V - 1.0)) * (cs2 - W * W) + pop4 * (cs2 - sqr(V + 1.0)) * (cs2 - W * W) + pop5 * (cs2 - sqr(W - 1.0)) * (cs2 - V * V) + pop6 * (cs2 - sqr(W + 1.0)) * (cs2 - V * V) + pop7 * (cs2 - sqr(V - 1.0)) * (cs2 - W * W) + pop8 * (cs2 - sqr(V - 1.0)) * (cs2 - W * W) + pop9 * (cs2 - sqr(V + 1.0)) * (cs2 - W * W) + pop10 * (cs2 - sqr(V + 1.0)) * (cs2 - W * W) + pop11 * (cs2 - sqr(W - 1.0)) * (cs2 - V * V) + pop12 * (cs2 - sqr(W - 1.0)) * (cs2 - V * V) + pop13 * (cs2 - sqr(W + 1.0)) * (cs2 - V * V) + pop14 * (cs2 - sqr(W + 1.0)) * (cs2 - V * V)) + rho * omega[19] * theta_cs2_sq;
		Mstar_i[20] = (omega[20] - 1.0) * (W * pop7 * (cs2 - sqr(U - 1.0)) * (V - 1.0) + W * pop8 * (cs2 - sqr(U + 1.0)) * (V - 1.0) + W * pop9 * (cs2 - sqr(U - 1.0)) * (V + 1.0) + W * pop10 * (cs2 - sqr(U + 1.0)) * (V + 1.0) + V * pop11 * (cs2 - sqr(U - 1.0)) * (W - 1.0) + V * pop12 * (cs2 - sqr(U + 1.0)) * (W - 1.0) + V * pop13 * (cs2 - sqr(U - 1.0)) * (W + 1.0) + V * pop14 * (cs2 - sqr(U + 1.0)) * (W + 1.0) + pop15 * (V - 1.0) * (W - 1.0) * (cs2 - U * U) + pop16 * (V + 1.0) * (W - 1.0) * (cs2 - U * U) + pop17 * (V - 1.0) * (W + 1.0) * (cs2 - U * U) + pop18 * (V + 1.0) * (W + 1.0) * (cs2 - U * U) + V * W * pop0 * (cs2 - U * U) + pop19 * (cs2 - sqr(U - 1.0)) * (V - 1.0) * (W - 1.0) + pop20 * (cs2 - sqr(U + 1.0)) * (V - 1.0) * (W - 1.0) + pop21 * (cs2 - sqr(U - 1.0)) * (V + 1.0) * (W - 1.0) + pop22 * (cs2 - sqr(U + 1.0)) * (V + 1.0) * (W - 1.0) + pop23 * (cs2 - sqr(U - 1.0)) * (V - 1.0) * (W + 1.0) + pop24 * (cs2 - sqr(U + 1.0)) * (V - 1.0) * (W + 1.0) + pop25 * (cs2 - sqr(U - 1.0)) * (V + 1.0) * (W + 1.0) + pop26 * (cs2 - sqr(U + 1.0)) * (V + 1.0) * (W + 1.0) + V * W * pop1 * (cs2 - sqr(U - 1.0)) + V * W * pop2 * (cs2 - sqr(U + 1.0)) + W * pop3 * (V - 1.0) * (cs2 - U * U) + W * pop4 * (V + 1.0) * (cs2 - U * U) + V * pop5 * (W - 1.0) * (cs2 - U * U) + V * pop6 * (W + 1.0) * (cs2 - U * U));
		Mstar_i[21] = (omega[21] - 1.0) * (W * pop7 * (cs2 - sqr(V - 1.0)) * (U - 1.0) + W * pop8 * (cs2 - sqr(V - 1.0)) * (U + 1.0) + W * pop9 * (cs2 - sqr(V + 1.0)) * (U - 1.0) + W * pop10 * (cs2 - sqr(V + 1.0)) * (U + 1.0) + pop11 * (U - 1.0) * (W - 1.0) * (cs2 - V * V) + pop12 * (U + 1.0) * (W - 1.0) * (cs2 - V * V) + pop13 * (U - 1.0) * (W + 1.0) * (cs2 - V * V) + pop14 * (U + 1.0) * (W + 1.0) * (cs2 - V * V) + U * pop15 * (cs2 - sqr(V - 1.0)) * (W - 1.0) + U * pop16 * (cs2 - sqr(V + 1.0)) * (W - 1.0) + U * pop17 * (cs2 - sqr(V - 1.0)) * (W + 1.0) + U * pop18 * (cs2 - sqr(V + 1.0)) * (W + 1.0) + U * W * pop0 * (cs2 - V * V) + pop19 * (cs2 - sqr(V - 1.0)) * (U - 1.0) * (W - 1.0) + pop20 * (cs2 - sqr(V - 1.0)) * (U + 1.0) * (W - 1.0) + pop21 * (cs2 - sqr(V + 1.0)) * (U - 1.0) * (W - 1.0) + pop22 * (cs2 - sqr(V + 1.0)) * (U + 1.0) * (W - 1.0) + pop23 * (cs2 - sqr(V - 1.0)) * (U - 1.0) * (W + 1.0) + pop24 * (cs2 - sqr(V - 1.0)) * (U + 1.0) * (W + 1.0) + pop25 * (cs2 - sqr(V + 1.0)) * (U - 1.0) * (W + 1.0) + pop26 * (cs2 - sqr(V + 1.0)) * (U + 1.0) * (W + 1.0) + W * pop1 * (U - 1.0) * (cs2 - V * V) + W * pop2 * (U + 1.0) * (cs2 - V * V) + U * W * pop3 * (cs2 - sqr(V - 1.0)) + U * W * pop4 * (cs2 - sqr(V + 1.0)) + U * pop5 * (W - 1.0) * (cs2 - V * V) + U * pop6 * (W + 1.0) * (cs2 - V * V));
		Mstar_i[22] = (omega[22] - 1.0) * (pop7 * (U - 1.0) * (V - 1.0) * (cs2 - W * W) + pop8 * (U + 1.0) * (V - 1.0) * (cs2 - W * W) + pop9 * (U - 1.0) * (V + 1.0) * (cs2 - W * W) + pop10 * (U + 1.0) * (V + 1.0) * (cs2 - W * W) + V * pop11 * (cs2 - sqr(W - 1.0)) * (U - 1.0) + V * pop12 * (cs2 - sqr(W - 1.0)) * (U + 1.0) + V * pop13 * (cs2 - sqr(W + 1.0)) * (U - 1.0) + V * pop14 * (cs2 - sqr(W + 1.0)) * (U + 1.0) + U * pop15 * (cs2 - sqr(W - 1.0)) * (V - 1.0) + U * pop16 * (cs2 - sqr(W - 1.0)) * (V + 1.0) + U * pop17 * (cs2 - sqr(W + 1.0)) * (V - 1.0) + U * pop18 * (cs2 - sqr(W + 1.0)) * (V + 1.0) + U * V * pop0 * (cs2 - W * W) + pop19 * (cs2 - sqr(W - 1.0)) * (U - 1.0) * (V - 1.0) + pop20 * (cs2 - sqr(W - 1.0)) * (U + 1.0) * (V - 1.0) + pop21 * (cs2 - sqr(W - 1.0)) * (U - 1.0) * (V + 1.0) + pop22 * (cs2 - sqr(W - 1.0)) * (U + 1.0) * (V + 1.0) + pop23 * (cs2 - sqr(W + 1.0)) * (U - 1.0) * (V - 1.0) + pop24 * (cs2 - sqr(W + 1.0)) * (U + 1.0) * (V - 1.0) + pop25 * (cs2 - sqr(W + 1.0)) * (U - 1.0) * (V + 1.0) + pop26 * (cs2 - sqr(W + 1.0)) * (U + 1.0) * (V + 1.0) + V * pop1 * (U - 1.0) * (cs2 - W * W) + V * pop2 * (U + 1.0) * (cs2 - W * W) + U * pop3 * (V - 1.0) * (cs2 - W * W) + U * pop4 * (V + 1.0) * (cs2 - W * W) + U * V * pop5 * (cs2 - sqr(W - 1.0)) + U * V * pop6 * (cs2 - sqr(W + 1.0)));
		Mstar_i[23] = (omega[23] - 1.0) * (W * pop1 * (cs2 - sqr(U - 1.0)) * (cs2 - V * V) + W * pop2 * (cs2 - sqr(U + 1.0)) * (cs2 - V * V) + W * pop3 * (cs2 - sqr(V - 1.0)) * (cs2 - U * U) + W * pop4 * (cs2 - sqr(V + 1.0)) * (cs2 - U * U) + pop5 * (W - 1.0) * (cs2 - U * U) * (cs2 - V * V) + pop6 * (W + 1.0) * (cs2 - U * U) * (cs2 - V * V) + W * pop7 * (cs2 - sqr(U - 1.0)) * (cs2 - sqr(V - 1.0)) + W * pop8 * (cs2 - sqr(U + 1.0)) * (cs2 - sqr(V - 1.0)) + W * pop9 * (cs2 - sqr(U - 1.0)) * (cs2 - sqr(V + 1.0)) + W * pop10 * (cs2 - sqr(U + 1.0)) * (cs2 - sqr(V + 1.0)) + pop11 * (cs2 - sqr(U - 1.0)) * (W - 1.0) * (cs2 - V * V) + pop12 * (cs2 - sqr(U + 1.0)) * (W - 1.0) * (cs2 - V * V) + pop13 * (cs2 - sqr(U - 1.0)) * (W + 1.0) * (cs2 - V * V) + pop14 * (cs2 - sqr(U + 1.0)) * (W + 1.0) * (cs2 - V * V) + pop15 * (cs2 - sqr(V - 1.0)) * (W - 1.0) * (cs2 - U * U) + pop16 * (cs2 - sqr(V + 1.0)) * (W - 1.0) * (cs2 - U * U) + pop17 * (cs2 - sqr(V - 1.0)) * (W + 1.0) * (cs2 - U * U) + pop18 * (cs2 - sqr(V + 1.0)) * (W + 1.0) * (cs2 - U * U) + W * pop0 * (cs2 - U * U) * (cs2 - V * V) + pop19 * (cs2 - sqr(U - 1.0)) * (cs2 - sqr(V - 1.0)) * (W - 1.0) + pop20 * (cs2 - sqr(U + 1.0)) * (cs2 - sqr(V - 1.0)) * (W - 1.0) + pop21 * (cs2 - sqr(U - 1.0)) * (cs2 - sqr(V + 1.0)) * (W - 1.0) + pop22 * (cs2 - sqr(U + 1.0)) * (cs2 - sqr(V + 1.0)) * (W - 1.0) + pop23 * (cs2 - sqr(U - 1.0)) * (cs2 - sqr(V - 1.0)) * (W + 1.0) + pop24 * (cs2 - sqr(U + 1.0)) * (cs2 - sqr(V - 1.0)) * (W + 1.0) + pop25 * (cs2 - sqr(U - 1.0)) * (cs2 - sqr(V + 1.0)) * (W + 1.0) + pop26 * (cs2 - sqr(U + 1.0)) * (cs2 - sqr(V + 1.0)) * (W + 1.0));
		Mstar_i[24] = (omega[24] - 1.0) * (V * pop1 * (cs2 - sqr(U - 1.0)) * (cs2 - W * W) + V * pop2 * (cs2 - sqr(U + 1.0)) * (cs2 - W * W) + pop3 * (V - 1.0) * (cs2 - U * U) * (cs2 - W * W) + pop4 * (V + 1.0) * (cs2 - U * U) * (cs2 - W * W) + V * pop5 * (cs2 - sqr(W - 1.0)) * (cs2 - U * U) + V * pop6 * (cs2 - sqr(W + 1.0)) * (cs2 - U * U) + pop7 * (cs2 - sqr(U - 1.0)) * (V - 1.0) * (cs2 - W * W) + pop8 * (cs2 - sqr(U + 1.0)) * (V - 1.0) * (cs2 - W * W) + pop9 * (cs2 - sqr(U - 1.0)) * (V + 1.0) * (cs2 - W * W) + pop10 * (cs2 - sqr(U + 1.0)) * (V + 1.0) * (cs2 - W * W) + V * pop11 * (cs2 - sqr(U - 1.0)) * (cs2 - sqr(W - 1.0)) + V * pop12 * (cs2 - sqr(U + 1.0)) * (cs2 - sqr(W - 1.0)) + V * pop13 * (cs2 - sqr(U - 1.0)) * (cs2 - sqr(W + 1.0)) + V * pop14 * (cs2 - sqr(U + 1.0)) * (cs2 - sqr(W + 1.0)) + pop15 * (cs2 - sqr(W - 1.0)) * (V - 1.0) * (cs2 - U * U) + pop16 * (cs2 - sqr(W - 1.0)) * (V + 1.0) * (cs2 - U * U) + pop17 * (cs2 - sqr(W + 1.0)) * (V - 1.0) * (cs2 - U * U) + pop18 * (cs2 - sqr(W + 1.0)) * (V + 1.0) * (cs2 - U * U) + V * pop0 * (cs2 - U * U) * (cs2 - W * W) + pop19 * (cs2 - sqr(U - 1.0)) * (cs2 - sqr(W - 1.0)) * (V - 1.0) + pop20 * (cs2 - sqr(U + 1.0)) * (cs2 - sqr(W - 1.0)) * (V - 1.0) + pop21 * (cs2 - sqr(U - 1.0)) * (cs2 - sqr(W - 1.0)) * (V + 1.0) + pop22 * (cs2 - sqr(U + 1.0)) * (cs2 - sqr(W - 1.0)) * (V + 1.0) + pop23 * (cs2 - sqr(U - 1.0)) * (cs2 - sqr(W + 1.0)) * (V - 1.0) + pop24 * (cs2 - sqr(U + 1.0)) * (cs2 - sqr(W + 1.0)) * (V - 1.0) + pop25 * (cs2 - sqr(U - 1.0)) * (cs2 - sqr(W + 1.0)) * (V + 1.0) + pop26 * (cs2 - sqr(U + 1.0)) * (cs2 - sqr(W + 1.0)) * (V + 1.0));
		Mstar_i[25] = (omega[25] - 1.0) * (pop1 * (U - 1.0) * (cs2 - V * V) * (cs2 - W * W) + pop2 * (U + 1.0) * (cs2 - V * V) * (cs2 - W * W) + U * pop3 * (cs2 - sqr(V - 1.0)) * (cs2 - W * W) + U * pop4 * (cs2 - sqr(V + 1.0)) * (cs2 - W * W) + U * pop5 * (cs2 - sqr(W - 1.0)) * (cs2 - V * V) + U * pop6 * (cs2 - sqr(W + 1.0)) * (cs2 - V * V) + pop7 * (cs2 - sqr(V - 1.0)) * (U - 1.0) * (cs2 - W * W) + pop8 * (cs2 - sqr(V - 1.0)) * (U + 1.0) * (cs2 - W * W) + pop9 * (cs2 - sqr(V + 1.0)) * (U - 1.0) * (cs2 - W * W) + pop10 * (cs2 - sqr(V + 1.0)) * (U + 1.0) * (cs2 - W * W) + pop11 * (cs2 - sqr(W - 1.0)) * (U - 1.0) * (cs2 - V * V) + pop12 * (cs2 - sqr(W - 1.0)) * (U + 1.0) * (cs2 - V * V) + pop13 * (cs2 - sqr(W + 1.0)) * (U - 1.0) * (cs2 - V * V) + pop14 * (cs2 - sqr(W + 1.0)) * (U + 1.0) * (cs2 - V * V) + U * pop15 * (cs2 - sqr(V - 1.0)) * (cs2 - sqr(W - 1.0)) + U * pop16 * (cs2 - sqr(V + 1.0)) * (cs2 - sqr(W - 1.0)) + U * pop17 * (cs2 - sqr(V - 1.0)) * (cs2 - sqr(W + 1.0)) + U * pop18 * (cs2 - sqr(V + 1.0)) * (cs2 - sqr(W + 1.0)) + U * pop0 * (cs2 - V * V) * (cs2 - W * W) + pop19 * (cs2 - sqr(V - 1.0)) * (cs2 - sqr(W - 1.0)) * (U - 1.0) + pop20 * (cs2 - sqr(V - 1.0)) * (cs2 - sqr(W - 1.0)) * (U + 1.0) + pop21 * (cs2 - sqr(V + 1.0)) * (cs2 - sqr(W - 1.0)) * (U - 1.0) + pop22 * (cs2 - sqr(V + 1.0)) * (cs2 - sqr(W - 1.0)) * (U + 1.0) + pop23 * (cs2 - sqr(V - 1.0)) * (cs2 - sqr(W + 1.0)) * (U - 1.0) + pop24 * (cs2 - sqr(V - 1.0)) * (cs2 - sqr(W + 1.0)) * (U + 1.0) + pop25 * (cs2 - sqr(V + 1.0)) * (cs2 - sqr(W + 1.0)) * (U - 1.0) + pop26 * (cs2 - sqr(V + 1.0)) * (cs2 - sqr(W + 1.0)) * (U + 1.0));
		Mstar_i[26] = (omega[26] - 1.0) * (pop7 * (cs2 - sqr(U - 1.0)) * (cs2 - sqr(V - 1.0)) * (cs2 - W * W) + pop8 * (cs2 - sqr(U + 1.0)) * (cs2 - sqr(V - 1.0)) * (cs2 - W * W) + pop9 * (cs2 - sqr(U - 1.0)) * (cs2 - sqr(V + 1.0)) * (cs2 - W * W) + pop10 * (cs2 - sqr(U + 1.0)) * (cs2 - sqr(V + 1.0)) * (cs2 - W * W) + pop11 * (cs2 - sqr(U - 1.0)) * (cs2 - sqr(W - 1.0)) * (cs2 - V * V) + pop12 * (cs2 - sqr(U + 1.0)) * (cs2 - sqr(W - 1.0)) * (cs2 - V * V) + pop13 * (cs2 - sqr(U - 1.0)) * (cs2 - sqr(W + 1.0)) * (cs2 - V * V) + pop14 * (cs2 - sqr(U + 1.0)) * (cs2 - sqr(W + 1.0)) * (cs2 - V * V) + pop15 * (cs2 - sqr(V - 1.0)) * (cs2 - sqr(W - 1.0)) * (cs2 - U * U) + pop16 * (cs2 - sqr(V + 1.0)) * (cs2 - sqr(W - 1.0)) * (cs2 - U * U) + pop17 * (cs2 - sqr(V - 1.0)) * (cs2 - sqr(W + 1.0)) * (cs2 - U * U) + pop18 * (cs2 - sqr(V + 1.0)) * (cs2 - sqr(W + 1.0)) * (cs2 - U * U) + pop0 * (cs2 - U * U) * (cs2 - V * V) * (cs2 - W * W) + pop19 * (cs2 - sqr(U - 1.0)) * (cs2 - sqr(V - 1.0)) * (cs2 - sqr(W - 1.0)) + pop20 * (cs2 - sqr(U + 1.0)) * (cs2 - sqr(V - 1.0)) * (cs2 - sqr(W - 1.0)) + pop21 * (cs2 - sqr(U - 1.0)) * (cs2 - sqr(V + 1.0)) * (cs2 - sqr(W - 1.0)) + pop22 * (cs2 - sqr(U + 1.0)) * (cs2 - sqr(V + 1.0)) * (cs2 - sqr(W - 1.0)) + pop23 * (cs2 - sqr(U - 1.0)) * (cs2 - sqr(V - 1.0)) * (cs2 - sqr(W + 1.0)) + pop24 * (cs2 - sqr(U + 1.0)) * (cs2 - sqr(V - 1.0)) * (cs2 - sqr(W + 1.0)) + pop25 * (cs2 - sqr(U - 1.0)) * (cs2 - sqr(V + 1.0)) * (cs2 - sqr(W + 1.0)) + pop26 * (cs2 - sqr(U + 1.0)) * (cs2 - sqr(V + 1.0)) * (cs2 - sqr(W + 1.0)) + pop1 * (cs2 - sqr(U - 1.0)) * (cs2 - V * V) * (cs2 - W * W) + pop2 * (cs2 - sqr(U + 1.0)) * (cs2 - V * V) * (cs2 - W * W) + pop3 * (cs2 - sqr(V - 1.0)) * (cs2 - U * U) * (cs2 - W * W) + pop4 * (cs2 - sqr(V + 1.0)) * (cs2 - U * U) * (cs2 - W * W) + pop5 * (cs2 - sqr(W - 1.0)) * (cs2 - U * U) * (cs2 - V * V) + pop6 * (cs2 - sqr(W + 1.0)) * (cs2 - U * U) * (cs2 - V * V)) + rho * omega[26] * pow(theta - cs2, 3.0);

		Mstar_i[1] += Fx;
		Mstar_i[2] += Fy;
		Mstar_i[3] += Fz;
		Mstar_i[4] += (Fx * Fy) / rho;
		Mstar_i[5] += (Fx * Fz) / rho;
		Mstar_i[6] += (Fy * Fz) / rho;
		Mstar_i[7] += (dPxxx * rho - dPyyy * rho + Fx * Fx - Fy * Fy) / rho;
		Mstar_i[8] += (dPxxx * rho - dPzzz * rho + Fx * Fx - Fz * Fz) / rho;
		Mstar_i[9] += (dPxxx * rho + dPyyy * rho + dPzzz * rho + Fx * Fx + Fy * Fy + Fz * Fz) / rho;
		Mstar_i[10] += Fy * 1.0 / (rho_2) * (dPxxx * rho + (rho_2)*theta - (rho_2)*cs2 + Fx * Fx);
		Mstar_i[11] += Fz * 1.0 / (rho_2) * (dPxxx * rho + (rho_2)*theta - (rho_2)*cs2 + Fx * Fx);
		Mstar_i[12] += Fx * 1.0 / (rho_2) * (dPyyy * rho + (rho_2)*theta - (rho_2)*cs2 + Fy * Fy);
		Mstar_i[13] += Fz * 1.0 / (rho_2) * (dPyyy * rho + (rho_2)*theta - (rho_2)*cs2 + Fy * Fy);
		Mstar_i[14] += Fx * 1.0 / (rho_2) * (dPzzz * rho + (rho_2)*theta - (rho_2)*cs2 + Fz * Fz);
		Mstar_i[15] += Fy * 1.0 / (rho_2) * (dPzzz * rho + (rho_2)*theta - (rho_2)*cs2 + Fz * Fz);
		Mstar_i[16] += Fx * Fy * Fz * 1.0 / (rho_2);
		Mstar_i[17] += 0 - rho * theta_cs2_sq + 1.0 / (rho_3) * (dPxxx * rho + (rho_2)*theta - (rho_2)*cs2 + Fx * Fx) * (dPyyy * rho + (rho_2)*theta - (rho_2)*cs2 + Fy * Fy);
		Mstar_i[18] += 0 - rho * theta_cs2_sq + 1.0 / (rho_3) * (dPxxx * rho + (rho_2)*theta - (rho_2)*cs2 + Fx * Fx) * (dPzzz * rho + (rho_2)*theta - (rho_2)*cs2 + Fz * Fz);
		Mstar_i[19] += 0 - rho * theta_cs2_sq + 1.0 / (rho_3) * (dPyyy * rho + (rho_2)*theta - (rho_2)*cs2 + Fy * Fy) * (dPzzz * rho + (rho_2)*theta - (rho_2)*cs2 + Fz * Fz);
		Mstar_i[20] += Fy * Fz * 1.0 / (rho_3) * (dPxxx * rho + (rho_2)*theta - (rho_2)*cs2 + Fx * Fx);
		Mstar_i[21] += Fx * Fz * 1.0 / (rho_3) * (dPyyy * rho + (rho_2)*theta - (rho_2)*cs2 + Fy * Fy);
		Mstar_i[22] += Fx * Fy * 1.0 / (rho_3) * (dPzzz * rho + (rho_2)*theta - (rho_2)*cs2 + Fz * Fz);
		Mstar_i[23] += Fz * 1.0 / (rho_4) * (dPxxx * rho + (rho_2)*theta - (rho_2)*cs2 + Fx * Fx) * (dPyyy * rho + (rho_2)*theta - (rho_2)*cs2 + Fy * Fy);
		Mstar_i[24] += Fy * 1.0 / (rho_4) * (dPxxx * rho + (rho_2)*theta - (rho_2)*cs2 + Fx * Fx) * (dPzzz * rho + (rho_2)*theta - (rho_2)*cs2 + Fz * Fz);
		Mstar_i[25] += Fx * 1.0 / (rho_4) * (dPyyy * rho + (rho_2)*theta - (rho_2)*cs2 + Fy * Fy) * (dPzzz * rho + (rho_2)*theta - (rho_2)*cs2 + Fz * Fz);
		Mstar_i[26] += 0 - rho * theta_cs2_sq * (theta - cs2) + 1.0 / (rho_4 * rho) * (dPxxx * rho + (rho_2)*theta - (rho_2)*cs2 + Fx * Fx) * (dPyyy * rho + (rho_2)*theta - (rho_2)*cs2 + Fy * Fy) * (dPzzz * rho + (rho_2)*theta - (rho_2)*cs2 + Fz * Fz);
#endif
		const double x0 = Mstar_i[23] * W;
		const double x1 = Mstar_i[24] * V;
		const double x2 = 2.0 * U;
		const double x3 = Mstar_i[20] * W;
		const double x4 = 4.0 * V;
		const double x5 = Mstar_i[21] * W;
		const double x6 = 4.0 * U;
		const double x7 = Mstar_i[22] * V;
		const double x8 = Mstar_i[16] * W;
		const double x9 = sqr(W);
		const double x10 = cs2 - 1.0;
		const double x11 = x10 + x9;
		const double x12 = sqr(V);
		const double x13 = x10 + x12;
		const double x14 = sqr(U);
		const double x15 = x10 + x14;
		const double x16 = 2.0 * V;
		const double x17 = Mstar_i[10] * x11;
		const double x18 = 2.0 * W;
		const double x19 = Mstar_i[11] * x13;
		const double x20 = Mstar_i[12] * x11;
		const double x21 = Mstar_i[13] * x15;
		const double x22 = Mstar_i[14] * x13;
		const double x23 = Mstar_i[15] * x15;
		const double x24 = U * x11;
		const double x25 = Mstar_i[4] * x24;
		const double x26 = W * x13;
		const double x27 = Mstar_i[5] * x26;
		const double x28 = W * x15;
		const double x29 = Mstar_i[6] * x28;
		const double x30 = x11 * x13;
		const double x31 = Mstar_i[1] * x30;
		const double x32 = Mstar_i[2] * x15;
		const double x33 = x11 * x32;
		const double x34 = Mstar_i[3] * x15;
		const double x35 = x13 * x34;
		const double x36 = Mstar_i[0] * x15;
		const double x37 = 0.66666666666666663 * x9;
		const double x38 = 0.33333333333333331 * x14;
		const double x39 = x12 * x38;
		const double x40 = 0.33333333333333331 * x9;
		const double x41 = cs2 * x40;
		const double x42 = x39 - x41;
		const double x43 = x38 + x40;
		const double x44 = x12 * x40;
		const double x45 = cs2 * x38;
		const double x46 = x44 - x45;
		const double x47 = 0.66666666666666663 * x12;
		const double x48 = cs2 * x47 - x47;
		const double x49 = x38 * x9;
		const double x50 = 0.33333333333333331 * x12;
		const double x51 = cs2 * x50;
		const double x52 = x49 - x51;
		const double x53 = x38 + x50;
		const double x54 = cs2 * x37 - x37;
		const double x55 = sqr(cs2);
		const double x56 = 0.66666666666666663 * x14;
		const double x57 = 0.16666666666666666 * x12;
		const double x58 = U * x57;
		const double x59 = 0.16666666666666666 * x9;
		const double x60 = U * x59;
		const double x61 = 0.33333333333333331 * U;
		const double x62 = cs2 * x61;
		const double x64 = x14 * x57;
		const double x66 = 0.5 * x55;
		const double x67 = -0.66666666666666663 * cs2;
		const double x68 = x14 * x59;
		const double x69 = x41 + x68;
		const double x70 = x45 + x64 - x59 + x66 + x67 + x69 + 0.16666666666666666;
		const double x72 = x57 * x9;
		const double x73 = x51 + x72;
		const double x74 = -x57 + x73;
		const double x75 = -x38 + x70 + x74;
		const double x76 = U * x40;
		const double x77 = 0.16666666666666666 * cs2;
		const double x79 = x77 - x58;
		const double x80 = cs2 * x59;
		const double x81 = -x64 + x80;
		const double x82 = x57 + x81;
		const double x83 = x52 + x82;
		const double x84 = U * x77;
		const double x85 = 0.16666666666666666 * U;
		const double x87 = 0.16666666666666666 * x14;
		const double x89 = -x85 - x87;
		const double x90 = x14 * x77;
		const double x91 = -x72 + x90;
		const double x92 = x59 + x91 - 0.16666666666666666;
		const double x93 = x84 + x89 + x92;
		const double x94 = U * x50;
		const double x96 = x77 - x60;
		const double x97 = cs2 * x57;
		const double x98 = -x68 + x97;
		const double x99 = x42 + x98;
		const double x100 = x57 + x99;
		const double x101 = cs2 + x14;
		const double x102 = U + x101;
		const double x103 = 0.5 * Mstar_i[19];
		const double x104 = Mstar_i[13] * W;
		const double x105 = Mstar_i[15] * V;
		const double x106 = x2 + 1.0;
		const double x107 = 0.5 * x106;
		const double x108 = V * x11;
		const double x109 = Mstar_i[2] * x102;
		const double x110 = Mstar_i[3] * x102;
		const double x111 = Mstar_i[4] * x108;
		const double x112 = Mstar_i[0] * x102;
		const double x113 = 0.5 * x30;
		const double x114 = x16 * x8;
		const double x115 = Mstar_i[6] * x102;
		const double x116 = W * x16;
		const double x117 = Mstar_i[10] * V;
		const double x118 = Mstar_i[11] * W;
		const double x119 = 0.5 * Mstar_i[26];
		const double x120 = cs2 - 1;
		const double x121 = x120 + x9;
		const double x122 = 0.5 * Mstar_i[17];
		const double x123 = x0 + x119 + x121 * x122;
		const double x124 = x12 + x120;
		const double x125 = 0.5 * Mstar_i[18];
		const double x126 = x1 + x124 * x125;
		const double x127 = x11 * x117 + x118 * x13 + x123 + x126 + x16 * x3;
		const double x129 = x77 + x92;
		const double x130 = x85 - x87;
		const double x131 = -x84 + x129 + x130;
		const double x132 = -U + x101;
		const double x133 = x2 - 1.0;
		const double x134 = 0.5 * x133;
		const double x135 = Mstar_i[2] * x132;
		const double x136 = Mstar_i[3] * x132;
		const double x137 = Mstar_i[0] * x132;
		const double x138 = Mstar_i[6] * x132;
		const double x139 = V * x87;
		const double x140 = x64 - x80;
		const double x141 = x139 + x140;
		const double x142 = 0.33333333333333331 * cs2;
		const double x143 = x142 - x90 - 0.33333333333333331;
		const double x144 = 0.33333333333333331 * V;
		const double x145 = V * x59;
		const double x146 = V * x142;
		const double x147 = -x144 + x145 + x146;
		const double x149 = -x50 + x73;
		const double x150 = x147 + x149;
		const double x151 = x70 - x87;
		const double x152 = V * x38;
		const double x154 = V * x77;
		const double x155 = x154 + x87;
		const double x156 = 0.16666666666666666 * V;
		const double x158 = -x156 - x57;
		const double x159 = x129 + x99;
		const double x160 = cs2 + x12;
		const double x161 = V + x160;
		const double x162 = Mstar_i[14] * U;
		const double x163 = x16 + 1.0;
		const double x164 = Mstar_i[22] * U;
		const double x165 = 0.5 * x163;
		const double x166 = Mstar_i[1] * x161;
		const double x167 = Mstar_i[3] * x28;
		const double x168 = 0.5 * x36;
		const double x169 = x11 * x161;
		const double x170 = x2 * x8;
		const double x171 = Mstar_i[5] * x161;
		const double x172 = W * x2;
		const double x173 = Mstar_i[12] * U;
		const double x174 = Mstar_i[25] * U;
		const double x175 = x120 + x14;
		const double x176 = x103 * x175 + x174;
		const double x177 = x104 * x15 + x11 * x173 + x123 + x176 + x2 * x5;
		const double x180 = -x142 - x38 + x91 + 0.33333333333333331;
		const double x182 = -x154 + x87;
		const double x183 = x156 - x57;
		const double x184 = -V + x160;
		const double x185 = x16 - 1.0;
		const double x186 = 0.5 * x185;
		const double x187 = Mstar_i[1] * x184;
		const double x188 = x11 * x184;
		const double x189 = Mstar_i[5] * x184;
		const double x190 = W * x87;
		const double x192 = x190 - x97;
		const double x193 = -x40 + x69;
		const double x194 = 0.33333333333333331 * W;
		const double x195 = W * x57;
		const double x196 = W * x142;
		const double x197 = -x194 + x195 + x196;
		const double x198 = x193 + x197;
		const double x199 = x45 + x64 + x66 + x67 + x74 - x87 + 0.16666666666666666;
		const double x200 = W * x38;
		const double x202 = W * x77;
		const double x203 = x202 + x87;
		const double x204 = 0.16666666666666666 * W;
		const double x206 = -x204 - x59;
		const double x207 = x77 + x83 + x91 - 0.16666666666666666;
		const double x208 = cs2 + x9;
		const double x209 = W + x208;
		const double x210 = x18 + 1.0;
		const double x211 = Mstar_i[20] * V;
		const double x212 = Mstar_i[21] * U;
		const double x213 = 0.5 * x210;
		const double x214 = U * x13;
		const double x215 = Mstar_i[1] * x214;
		const double x216 = V * x15;
		const double x217 = Mstar_i[2] * x216;
		const double x218 = Mstar_i[5] * x214;
		const double x219 = Mstar_i[6] * x216;
		const double x220 = x13 * x168;
		const double x221 = V * x2;
		const double x222 = Mstar_i[16] * x210;
		const double x223 = Mstar_i[4] * x221;
		const double x224 = x105 * x15 + x119 + x126 + x13 * x162 + x176 + x2 * x7;
		const double x227 = -W + x208;
		const double x228 = x18 - 1.0;
		const double x229 = 0.5 * x228;
		const double x230 = Mstar_i[16] * x228;
		const double x231 = 0.083333333333333329 * U;
		const double x232 = x12 * x231;
		const double x233 = cs2 * x231;
		const double x235 = x232 - x233;
		const double x236 = 0.083333333333333329 * V;
		const double x237 = x236 * x9;
		const double x238 = -x236 + x237;
		const double x239 = V * x231;
		const double x240 = x14 * x236;
		const double x241 = x239 + x240;
		const double x242 = x238 + x241;
		const double x243 = 0.083333333333333329 * cs2;
		const double x244 = 0.083333333333333329 * x14;
		const double x245 = cs2 * x244;
		const double x247 = x243 - x245;
		const double x248 = x247 + x85;
		const double x249 = 0.083333333333333329 * x12;
		const double x250 = x12 * x244;
		const double x251 = x249 * x9;
		const double x252 = x250 + x251;
		const double x253 = -x249 + x252;
		const double x254 = x243 * x9;
		const double x256 = -x254 + x98;
		const double x257 = x253 + x256;
		const double x258 = x155 + x257;
		const double x259 = x154 + x242;
		const double x261 = -x244 - x77;
		const double x262 = x244 * x9;
		const double x263 = x262 + 0.25 * x55 + x80 + x90 + x97;
		const double x264 = x253 + x263;
		const double x265 = x261 + x264;
		const double x266 = x231 * x9;
		const double x268 = x232 - x231 + x84;
		const double x269 = x266 + x268;
		const double x270 = x265 + x269;
		const double x271 = V * x85;
		const double x272 = cs2 * x249;
		const double x273 = -x262 + x272;
		const double x274 = x249 + x271 + x273;
		const double x275 = cs2 * x236;
		const double x277 = x275 - x237;
		const double x278 = x141 + x277;
		const double x279 = x236 + x278;
		const double x281 = -x266 + x58;
		const double x282 = x231 + x233;
		const double x284 = x244 + x245 - x251;
		const double x285 = x282 + x284 + x77;
		const double x286 = x281 + x285;
		const double x287 = 0.25 * x106;
		const double x288 = Mstar_i[14] * x161;
		const double x289 = 0.25 * x163;
		const double x290 = Mstar_i[15] * x289;
		const double x291 = Mstar_i[22] * x106;
		const double x292 = 0.25 * x169;
		const double x293 = x11 * x166;
		const double x294 = x11 * x289;
		const double x295 = Mstar_i[4] * x106;
		const double x296 = 0.5 * W;
		const double x297 = x106 * x296;
		const double x298 = Mstar_i[16] * x163;
		const double x299 = x161 * x296;
		const double x300 = x163 * x296;
		const double x301 = Mstar_i[11] * x296;
		const double x302 = Mstar_i[20] * x296;
		const double x303 = 0.25 * Mstar_i[26];
		const double x304 = 0.25 * Mstar_i[17];
		const double x305 = 0.5 * x0 + x121 * x304 + x303;
		const double x306 = 0.5 * V;
		const double x307 = 0.25 * Mstar_i[18];
		const double x308 = Mstar_i[24] * (x306 + 0.25) + x161 * x307;
		const double x309 = x161 * x301 + x163 * x302 + x17 * x289 + x305 + x308;
		const double x310 = Mstar_i[13] * x296;
		const double x311 = Mstar_i[21] * x296;
		const double x312 = 0.5 * U;
		const double x313 = 0.25 * Mstar_i[19];
		const double x314 = Mstar_i[25] * (x312 + 0.25) + x102 * x313;
		const double x315 = x102 * x310 + x106 * x311 + x20 * x287 + x314;
		const double x317 = x238 + x240;
		const double x318 = -x239 + x317;
		const double x320 = -x232 + x60;
		const double x321 = x233 + x320;
		const double x322 = x247 - x85;
		const double x323 = x321 + x322;
		const double x324 = -x84 + x231 - x232;
		const double x325 = -x266 + x324;
		const double x326 = x265 + x325;
		const double x327 = x249 - x271 + x273;
		const double x328 = -x233 + x266;
		const double x329 = -x231 + x284;
		const double x330 = x328 + x329 + x79;
		const double x331 = 0.25 * x133;
		const double x332 = Mstar_i[22] * x331;
		const double x333 = x11 * x331;
		const double x334 = Mstar_i[4] * x163;
		const double x335 = x133 * x296;
		const double x336 = Mstar_i[25] * x331 + x132 * x313;
		const double x337 = x132 * x310 + x133 * x311 + x20 * x331 + x336;
		const double x338 = x236 - x237;
		const double x340 = -x154 - x240;
		const double x341 = x338 + x340;
		const double x342 = -x250 + x254;
		const double x343 = x342 + x68;
		const double x344 = -x97 + x343;
		const double x345 = -x243 + x245 - x251;
		const double x346 = x345 + x89;
		const double x348 = -x139 - x275;
		const double x349 = x140 + x348;
		const double x350 = x238 + x349;
		const double x351 = Mstar_i[14] * x184;
		const double x352 = 0.25 * x185;
		const double x353 = Mstar_i[15] * x352;
		const double x354 = 0.25 * x188;
		const double x355 = x11 * x352;
		const double x356 = Mstar_i[16] * x185;
		const double x357 = x184 * x296;
		const double x358 = x185 * x296;
		const double x359 = Mstar_i[24] * x352 + x184 * x307;
		const double x360 = x17 * x352 + x184 * x301 + x185 * x302 + x305 + x359;
		const double x361 = x239 - x240;
		const double x362 = x257 + x338;
		const double x363 = Mstar_i[4] * x185;
		const double x364 = x328 - x58;
		const double x365 = 0.083333333333333329 * x9;
		const double x367 = 0.083333333333333329 * W;
		const double x368 = W * x249;
		const double x369 = -x367 + x368;
		const double x370 = W * x231;
		const double x371 = W * x244;
		const double x372 = x370 + x371;
		const double x373 = x369 + x372;
		const double x374 = -x365 + x373;
		const double x375 = x262 - x272;
		const double x376 = x251 + x375;
		const double x377 = x203 + x376 + x81;
		const double x378 = x252 + x261;
		const double x379 = x263 + x378;
		const double x380 = x202 + x379;
		const double x381 = W * x85;
		const double x382 = x285 + x320 + x365;
		const double x383 = W * x243;
		const double x385 = x383 - x368;
		const double x386 = x367 + x385;
		const double x387 = x192 + x343;
		const double x388 = x386 + x387;
		const double x389 = Mstar_i[12] * x209;
		const double x390 = 0.25 * x210;
		const double x391 = Mstar_i[13] * x390;
		const double x392 = Mstar_i[21] * x210;
		const double x393 = x13 * x209;
		const double x394 = 0.25 * x393;
		const double x395 = Mstar_i[1] * x393;
		const double x396 = x13 * x390;
		const double x397 = x13 * x287;
		const double x398 = Mstar_i[5] * x210;
		const double x399 = x222 * x306;
		const double x400 = x209 * x306;
		const double x401 = x210 * x306;
		const double x402 = Mstar_i[15] * x306;
		const double x403 = 0.5 * x1 + x124 * x307 + x303;
		const double x404 = x102 * x402 + x22 * x287 + x291 * x306 + x314 + x403;
		const double x405 = Mstar_i[10] * x306;
		const double x406 = Mstar_i[20] * x306;
		const double x407 = Mstar_i[23] * (x296 + 0.25) + x209 * x304;
		const double x408 = x19 * x390 + x209 * x405 + x210 * x406 + x407;
		const double x409 = x233 + x281;
		const double x410 = x369 + x371;
		const double x412 = -x365 - x370;
		const double x413 = x410 + x412;
		const double x415 = x13 * x331;
		const double x416 = Mstar_i[4] * x133;
		const double x417 = Mstar_i[22] * x133;
		const double x418 = x132 * x402 + x22 * x331 + x306 * x417 + x336 + x403;
		const double x420 = x266 - x371;
		const double x421 = -x202 + x367 - x368;
		const double x423 = -x190 - x383;
		const double x424 = x202 + x273 + x365;
		const double x425 = x140 + x424;
		const double x426 = Mstar_i[12] * x227;
		const double x427 = 0.25 * x228;
		const double x428 = Mstar_i[13] * x427;
		const double x429 = Mstar_i[21] * x106;
		const double x430 = 0.25 * x227;
		const double x431 = x13 * x430;
		const double x432 = Mstar_i[1] * x227;
		const double x433 = x13 * x427;
		const double x434 = x230 * x306;
		const double x435 = x227 * x306;
		const double x436 = x228 * x306;
		const double x437 = Mstar_i[23] * x427 + x227 * x304;
		const double x438 = x19 * x427 + x227 * x405 + x228 * x406 + x437;
		const double x439 = -x365 + x378;
		const double x440 = -x266 - x371;
		const double x441 = x345 + x410;
		const double x442 = Mstar_i[21] * x228;
		const double x443 = W * x236;
		const double x444 = x247 + x443;
		const double x445 = x444 + x59;
		const double x446 = x154 + x317;
		const double x447 = x257 + x446;
		const double x448 = x202 - x365 + x410;
		const double x449 = x376 + x448 + x82;
		const double x450 = x443 + x446;
		const double x451 = -x77 + x264;
		const double x452 = x448 + x451;
		const double x453 = Mstar_i[10] * x209;
		const double x454 = Mstar_i[11] * x390;
		const double x455 = Mstar_i[20] * x210;
		const double x456 = x161 * x36;
		const double x457 = 0.25 * x209;
		const double x458 = x209 * x32;
		const double x459 = x34 * x390;
		const double x460 = Mstar_i[6] * x15;
		const double x461 = x210 * x460;
		const double x462 = x209 * x312;
		const double x463 = x222 * x312;
		const double x464 = x210 * x312;
		const double x465 = 0.5 * x174 + x175 * x313 + x303;
		const double x466 = x21 * x390 + x312 * x389 + x312 * x392 + x407 + x465;
		const double x467 = Mstar_i[22] * x312;
		const double x468 = x163 * x467 + x23 * x289 + x288 * x312 + x308;
		const double x470 = x247 - x443;
		const double x471 = x184 * x36;
		const double x472 = x185 * x467 + x23 * x352 + x312 * x351 + x359;
		const double x473 = x190 - x204 + x385;
		const double x474 = -x365 - x371 + x421 + x451;
		const double x475 = x424 + x441;
		const double x476 = Mstar_i[10] * x227;
		const double x477 = Mstar_i[11] * x427;
		const double x478 = Mstar_i[20] * x163;
		const double x479 = x227 * x32;
		const double x480 = x34 * x427;
		const double x481 = x227 * x312;
		const double x482 = x230 * x312;
		const double x483 = x228 * x312;
		const double x484 = x21 * x427 + x312 * x426 + x312 * x442 + x437 + x465;
		const double x485 = Mstar_i[20] * x228;
		const double x486 = 0.041666666666666664 * U;
		const double x487 = cs2 * x486;
		const double x489 = 0.041666666666666664 * V;
		const double x490 = W * x489;
		const double x491 = 0.041666666666666664 * x12;
		const double x492 = W * x491;
		const double x493 = x490 + x492;
		const double x494 = -x487 + x493;
		const double x495 = x489 * x9;
		const double x496 = x491 * x9;
		const double x497 = 0.041666666666666664 * cs2;
		const double x498 = x14 * x497;
		const double x499 = x496 - x498;
		const double x500 = x495 + x499;
		const double x501 = x494 + x500;
		const double x502 = W * x497;
		const double x504 = -x370 - x502;
		const double x505 = x12 * x486;
		const double x506 = x14 * x491;
		const double x507 = x505 + x506;
		const double x508 = V * x486;
		const double x509 = x14 * x489;
		const double x510 = x275 + x508 + x509;
		const double x511 = x507 + x510;
		const double x512 = x497 * x9;
		const double x513 = x273 - x512;
		const double x514 = x511 + x513;
		const double x515 = -x239 - x232;
		const double x516 = x486 * x9;
		const double x517 = 0.041666666666666664 * x14;
		const double x518 = x517 * x9;
		const double x519 = x516 + x518;
		const double x520 = W * x486;
		const double x521 = W * x517;
		const double x522 = x383 + x520 + x521;
		const double x523 = x519 + x522;
		const double x524 = cs2 * x489;
		const double x526 = cs2 * x491;
		const double x528 = x342 - x524 - x526;
		const double x529 = x493 + x495;
		const double x530 = x245 + x254 + x272 + x496 + 0.125 * x55;
		const double x531 = x233 + x530;
		const double x532 = x523 + x531;
		const double x533 = 0.125 * x209;
		const double x534 = x161 * x533;
		const double x535 = 0.125 * x106;
		const double x536 = x166 * x209;
		const double x537 = 0.125 * x163;
		const double x538 = x222 * x537;
		const double x539 = x209 * x537;
		const double x540 = 0.125 * x210;
		const double x541 = x161 * x540;
		const double x542 = x171 * x540;
		const double x543 = x210 * x537;
		const double x544 = Mstar_i[11] * x540;
		const double x545 = 0.125 * Mstar_i[26];
		const double x546 = 0.25 * W;
		const double x547 = 0.125 * Mstar_i[17];
		const double x548 = Mstar_i[23] * (x546 + 0.125) + x209 * x547 + x545;
		const double x549 = 0.25 * V;
		const double x550 = 0.125 * Mstar_i[18];
		const double x551 = Mstar_i[24] * (x549 + 0.125) + x161 * x550;
		const double x552 = x161 * x544 + x453 * x537 + x455 * x537 + x548 + x551;
		const double x553 = Mstar_i[13] * x540;
		const double x554 = 0.25 * U;
		const double x555 = 0.125 * Mstar_i[19];
		const double x556 = Mstar_i[25] * (x554 + 0.125) + x102 * x555;
		const double x557 = x102 * x553 + x389 * x535 + x392 * x535 + x556;
		const double x558 = Mstar_i[15] * x537;
		const double x559 = x102 * x558 + x288 * x535 + x291 * x537;
		const double x560 = x487 + x493;
		const double x561 = x500 + x560;
		const double x563 = x275 + x509 - x508;
		const double x565 = x506 - x505;
		const double x566 = x513 + x565;
		const double x567 = x563 + x566;
		const double x569 = x383 + x521 - x520;
		const double x570 = x361 + x569;
		const double x572 = x232 - x516;
		const double x573 = x518 + x572;
		const double x574 = -x233 + x518 + x530 + x565 - x516;
		const double x575 = x569 + x574;
		const double x576 = 0.125 * x133;
		const double x577 = Mstar_i[25] * (x554 - 0.125) + x132 * x555;
		const double x578 = x132 * x553 + x389 * x576 + x392 * x576 + x577;
		const double x579 = x132 * x558 + x288 * x576 + x417 * x537;
		const double x581 = x524 - x495;
		const double x582 = x241 + x581;
		const double x584 = -x487 - x490;
		const double x585 = -x232 + x492 + x584;
		const double x586 = x342 + x499 - x526;
		const double x587 = x492 - x490;
		const double x588 = -x275 - x509 - x495;
		const double x589 = x507 - x508 + x588;
		const double x590 = x372 + x502;
		const double x591 = x266 + x487;
		const double x592 = x490 + x591;
		const double x594 = -x496 + x498;
		const double x596 = x495 - x492;
		const double x597 = x375 + x512 - x506 + x594 + x596;
		const double x598 = x510 - x505;
		const double x599 = x184 * x533;
		const double x600 = x209 * x535;
		const double x601 = x106 * x540;
		const double x602 = 0.125 * x185;
		const double x603 = x209 * x602;
		const double x604 = x184 * x540;
		const double x605 = x185 * x540;
		const double x606 = Mstar_i[15] * x602;
		const double x607 = Mstar_i[24] * (x549 - 0.125) + x184 * x550;
		const double x608 = x102 * x606 + x291 * x602 + x351 * x535 + x607;
		const double x609 = x184 * x544 + x453 * x602 + x455 * x602 + x548;
		const double x610 = x240 + x569 + x581;
		const double x611 = x487 + x587;
		const double x612 = -x239 + x611;
		const double x613 = x508 + x588;
		const double x614 = x371 - x370 + x502;
		const double x615 = x209 * x576;
		const double x616 = x133 * x540;
		const double x617 = x132 * x606 + x351 * x576 + x417 * x602 + x607;
		const double x618 = x500 - x492;
		const double x619 = -x383 - x521;
		const double x620 = -x490 + x596 + x619;
		const double x621 = x519 + x531 - x520;
		const double x622 = x250 - x254 - x518 + x526 + x594;
		const double x623 = x522 + x572 + x622;
		const double x624 = 0.125 * x227;
		const double x625 = x161 * x624;
		const double x626 = x166 * x227;
		const double x627 = x230 * x537;
		const double x628 = x227 * x537;
		const double x629 = 0.125 * x228;
		const double x630 = x161 * x629;
		const double x631 = x171 * x228;
		const double x632 = x228 * x537;
		const double x633 = Mstar_i[11] * x629;
		const double x634 = Mstar_i[23] * (x546 - 0.125) + x227 * x547 + x545;
		const double x635 = x161 * x633 + x476 * x537 + x478 * x629 + x551 + x634;
		const double x636 = Mstar_i[13] * x629;
		const double x637 = x102 * x636 + x426 * x535 + x429 * x629 + x556;
		const double x638 = x520 + x574;
		const double x639 = x516 + x622;
		const double x640 = x132 * x636 + x426 * x576 + x442 * x576 + x577;
		const double x641 = x490 + x619;
		const double x642 = x495 - x524;
		const double x643 = x184 * x624;
		const double x644 = x227 * x535;
		const double x645 = x185 * x230;
		const double x646 = x227 * x602;
		const double x647 = x184 * x629;
		const double x648 = x189 * x228;
		const double x649 = x228 * x602;
		const double x650 = x184 * x633 + x476 * x602 + x485 * x602 + x634;
		const double x651 = -x492 + x613;
		const double x652 = x227 * x576;
		pop(idx + c_alpha_offsets[0], 0) = -Mstar_i[17] * x11 - Mstar_i[18] * x13 - Mstar_i[19] * x15 - Mstar_i[25] * x2 - Mstar_i[26] - Mstar_i[7] * (-x14 * x37 + x42 + x43 + x46 + x48) - Mstar_i[8] * (-x14 * x47 + x46 + x52 + x53 + x54) - Mstar_i[9] * (cs2 * x56 - 2.0 * cs2 + x39 + x44 + x48 + x49 + x54 + x55 - x56 + 1.0) - 8.0 * U * V * x8 - 2.0 * x0 - 2.0 * x1 - x16 * x17 - x16 * x23 - x16 * x33 - x18 * x19 - x18 * x21 - x18 * x35 - x2 * x20 - x2 * x22 - x2 * x31 - x25 * x4 - x27 * x6 - x29 * x4 - x3 * x4 - x30 * x36 - x5 * x6 - x6 * x7;
		pop(idx + c_alpha_offsets[1], 1) = Mstar_i[25] * (U + 0.5) - Mstar_i[7] * (x76 + x79 + x83 + x93) - Mstar_i[8] * (x100 + x93 + x94 + x96) + Mstar_i[9] * (x58 + x60 - x61 + x62 + x75) + x102 * x103 + x102 * x104 + x102 * x105 + x106 * x111 + x106 * x114 + x106 * x27 + x106 * x5 + x106 * x7 + x107 * x20 + x107 * x22 + x107 * x31 + x108 * x109 + x110 * x26 + x112 * x113 + x115 * x116 + x127;
		pop(idx + c_alpha_offsets[2], 2) = Mstar_i[25] * (U - 0.5) - Mstar_i[7] * (x131 + x58 - x76 + x83) - Mstar_i[8] * (x100 + x131 + x60 - x94) + Mstar_i[9] * (x61 - x62 + x75 - x58 - x60) + x103 * x132 + x104 * x132 + x105 * x132 + x108 * x135 + x111 * x133 + x113 * x137 + x114 * x133 + x116 * x138 + x127 + x133 * x27 + x133 * x5 + x133 * x7 + x134 * x20 + x134 * x22 + x134 * x31 + x136 * x26;
		pop(idx + c_alpha_offsets[3], 3) = Mstar_i[24] * (V + 0.5) + Mstar_i[7] * (x141 + x143 + x150 + x43 - x49) - Mstar_i[8] * (x152 - x145 + x155 + x158 + x159) + Mstar_i[9] * (x139 + x150 + x151) + x118 * x161 + x125 * x161 + x161 * x162 + x161 * x167 + x163 * x164 + x163 * x170 + x163 * x25 + x163 * x29 + x163 * x3 + x165 * x17 + x165 * x23 + x165 * x33 + x166 * x24 + x168 * x169 + x171 * x172 + x177;
		pop(idx + c_alpha_offsets[4], 4) = Mstar_i[24] * (V - 0.5) - Mstar_i[7] * (x139 + x147 - x40 + x180 + x50 + x52 + x81) - Mstar_i[8] * (x145 - x152 + x159 + x182 + x183) + Mstar_i[9] * (x144 - x146 + x149 + x151 - x145 - x139) + x118 * x184 + x125 * x184 + x162 * x184 + x164 * x185 + x167 * x184 + x168 * x188 + x17 * x186 + x170 * x185 + x172 * x189 + x177 + x185 * x25 + x185 * x29 + x185 * x3 + x186 * x23 + x186 * x33 + x187 * x24;
		pop(idx + c_alpha_offsets[5], 5) = Mstar_i[23] * (W + 0.5) - Mstar_i[7] * (x200 - x195 + x203 + x206 + x207) + Mstar_i[8] * (x143 + x192 + x198 - x39 + x53 + x72) + Mstar_i[9] * (x190 + x198 + x199) + x117 * x209 + x122 * x209 + x173 * x209 + x19 * x213 + x209 * x215 + x209 * x217 + x209 * x220 + x209 * x223 + x21 * x213 + x210 * x211 + x210 * x212 + x210 * x218 + x210 * x219 + x213 * x35 + x221 * x222 + x224;
		pop(idx + c_alpha_offsets[6], 6) = Mstar_i[23] * (W - 0.5) - Mstar_i[7] * (x195 - x200 + x204 + x207 - x202 - x59 + x87) - Mstar_i[8] * (-x50 + x180 + x190 + x197 + x40 + x99) + Mstar_i[9] * (x193 + x194 - x196 + x199 - x195 - x190) + x117 * x227 + x122 * x227 + x173 * x227 + x19 * x229 + x21 * x229 + x211 * x228 + x212 * x228 + x215 * x227 + x217 * x227 + x218 * x228 + x219 * x228 + x220 * x227 + x221 * x230 + x223 * x227 + x224 + x229 * x35;
		pop(idx + c_alpha_offsets[7], 7) = -Mstar_i[7] * (x235 + x242 + x248 + x258 - x60) + Mstar_i[8] * (x274 + x279 + x286) - Mstar_i[9] * (x259 + x270) - x102 * x290 - x109 * x294 - x110 * x299 - x112 * x292 - x115 * x300 - x171 * x297 - x287 * x288 - x287 * x293 - x289 * x291 - x294 * x295 - x297 * x298 - x309 - x315;
		pop(idx + c_alpha_offsets[8], 8) = -Mstar_i[7] * (x258 + x318 + x323) + Mstar_i[8] * (x279 + x327 + x330) - Mstar_i[9] * (x154 + x318 + x326) - x132 * x290 - x135 * x294 - x136 * x299 - x137 * x292 - x138 * x300 - x163 * x332 - x171 * x335 - x288 * x331 - x293 * x331 - x298 * x335 - x309 - x333 * x334 - x337;
		pop(idx + c_alpha_offsets[9], 9) = Mstar_i[7] * (x249 + x259 + x321 + x344 + x346) + Mstar_i[8] * (x286 + x327 + x350) - Mstar_i[9] * (x270 - x239 + x341) - x102 * x353 - x109 * x355 - x11 * x187 * x287 - x110 * x357 - x112 * x354 - x115 * x358 - x189 * x297 - x287 * x351 - x291 * x352 - x295 * x355 - x297 * x356 - x315 - x360;
		pop(idx + c_alpha_offsets[10], 10) = -Mstar_i[7] * (x182 + x323 + x361 + x362) + Mstar_i[8] * (x274 + x330 + x350) - Mstar_i[9] * (-x154 + x326 + x338 + x361) - x132 * x353 - x135 * x355 - x136 * x357 - x137 * x354 - x138 * x358 - x185 * x332 - x187 * x333 - x189 * x335 - x331 * x351 - x333 * x363 - x335 * x356 - x337 - x360;
		pop(idx + c_alpha_offsets[11], 11) = Mstar_i[7] * (x381 + x382 + x388) - Mstar_i[8] * (x248 + x364 + x374 + x377) - Mstar_i[9] * (x269 + x374 + x380) - x102 * x391 - x106 * x399 - x109 * x400 - x110 * x396 - x112 * x394 - x115 * x401 - x287 * x389 - x287 * x392 - x287 * x395 - x295 * x400 - x397 * x398 - x404 - x408;
		pop(idx + c_alpha_offsets[12], 12) = Mstar_i[7] * (x235 + x329 + x365 + x388 - x381 + x96) - Mstar_i[8] * (x322 + x377 + x409 + x413) - Mstar_i[9] * (x325 + x380 + x413) - x132 * x391 - x133 * x399 - x135 * x400 - x136 * x396 - x137 * x394 - x138 * x401 - x331 * x389 - x331 * x392 - x331 * x395 - x398 * x415 - x400 * x416 - x408 - x418;
		pop(idx + c_alpha_offsets[13], 13) = -Mstar_i[5] * x106 * x433 + Mstar_i[7] * (x344 + x369 + x382 - x381 + x423) + Mstar_i[8] * (x346 + x373 + x409 + x425) - Mstar_i[9] * (x268 + x379 + x412 + x420 + x421) - x102 * x428 - x106 * x434 - x109 * x435 - x110 * x433 - x112 * x431 - x115 * x436 - x287 * x426 - x295 * x435 - x397 * x432 - x404 - x427 * x429 - x438;
		pop(idx + c_alpha_offsets[14], 14) = -Mstar_i[5] * x228 * x415 - Mstar_i[7] * (x190 - x245 + x256 + x282 + x320 + x386 - x381 + x439) + Mstar_i[8] * (x130 + x364 - x370 + x425 + x441) - Mstar_i[9] * (x263 + x324 + x370 + x421 + x439 + x440) - x132 * x428 - x133 * x434 - x135 * x435 - x136 * x433 - x137 * x431 - x138 * x436 - x331 * x426 - x331 * x442 - x415 * x432 - x416 * x435 - x418 - x438;
		pop(idx + c_alpha_offsets[15], 15) = -Mstar_i[7] * (x204 + x368 + x423 + x445 + x447) - Mstar_i[8] * (x156 + x237 + x348 + x444 + x449) - Mstar_i[9] * (x450 + x452) - x161 * x454 - x161 * x459 - x163 * x463 - x166 * x462 - x171 * x464 - x289 * x453 - x289 * x455 - x289 * x458 - x289 * x461 - x334 * x462 - x456 * x457 - x466 - x468;
		pop(idx + c_alpha_offsets[16], 16) = Mstar_i[7] * (x206 + x249 + x345 + x385 + x387 + x450) - Mstar_i[8] * (x139 - x156 + x277 + x449 + x470) - Mstar_i[9] * (x341 + x452 - x443) - x184 * x454 - x184 * x459 - x185 * x463 - x187 * x462 - x189 * x464 - x352 * x453 - x352 * x455 - x352 * x458 - x352 * x461 - x363 * x462 - x457 * x471 - x466 - x472;
		pop(idx + c_alpha_offsets[17], 17) = -Mstar_i[7] * (x447 + x470 + x473 + x59) + Mstar_i[8] * (x158 + x278 + x443 + x475) - Mstar_i[9] * (x446 - x443 + x474) - x161 * x477 - x161 * x480 - x163 * x427 * x460 - x163 * x482 - x166 * x481 - x171 * x483 - x289 * x476 - x289 * x479 - x334 * x481 - x427 * x478 - x430 * x456 - x468 - x484;
		pop(idx + c_alpha_offsets[18], 18) = -Mstar_i[7] * (x340 + x362 + x445 + x473) + Mstar_i[8] * (x183 + x237 + x349 - x443 + x475) - Mstar_i[9] * (x341 + x443 + x474) - x184 * x477 - x184 * x480 - x185 * x482 - x187 * x481 - x189 * x483 - x228 * x352 * x460 - x352 * x476 - x352 * x479 - x352 * x485 - x363 * x481 - x430 * x471 - x472 - x484;
		pop(idx + c_alpha_offsets[19], 19) = Mstar_i[7] * (x440 + x501 + x504 + x514) + Mstar_i[8] * (-x240 + x501 + x515 + x523 + x528) + Mstar_i[9] * (x511 + x529 + x532) + x106 * x538 + x106 * x542 + x109 * x539 + x110 * x541 + x112 * x534 + x115 * x543 + x295 * x539 + x535 * x536 + x552 + x557 + x559;
		pop(idx + c_alpha_offsets[20], 20) = Mstar_i[7] * (x370 + x420 - x502 + x561 + x567) + Mstar_i[8] * (x528 + x561 + x570 + x573) + Mstar_i[9] * (x529 + x563 + x575) + x133 * x538 + x133 * x542 + x135 * x539 + x136 * x541 + x137 * x534 + x138 * x543 + x416 * x539 + x536 * x576 + x552 + x578 + x579;
		pop(idx + c_alpha_offsets[21], 21) = -Mstar_i[7] * (x590 + x592 + x597 + x598) + Mstar_i[8] * (x523 + x582 + x585 + x586) + Mstar_i[9] * (x532 + x587 + x589) + x109 * x603 + x110 * x604 + x112 * x599 + x115 * x605 + x187 * x600 + x189 * x601 + x356 * x601 + x363 * x600 + x557 + x608 + x609;
		pop(idx + c_alpha_offsets[22], 22) = -Mstar_i[7] * (-x266 - x487 + x490 + x505 + x563 + x597 + x614) + Mstar_i[8] * (x573 + x586 + x610 + x612) + Mstar_i[9] * (x575 + x587 + x613) + x135 * x603 + x136 * x604 + x137 * x599 + x138 * x605 + x187 * x615 + x189 * x616 + x356 * x616 + x363 * x615 + x578 + x609 + x617;
		pop(idx + c_alpha_offsets[23], 23) = Mstar_i[7] * (-x266 + x514 + x584 + x590 + x618) - Mstar_i[8] * (x560 + x582 + x623) + Mstar_i[9] * (x511 + x620 + x621) + x106 * x627 + x109 * x628 + x110 * x630 + x112 * x625 + x115 * x632 + x295 * x628 + x535 * x626 + x535 * x631 + x559 + x635 + x637;
		pop(idx + c_alpha_offsets[24], 24) = Mstar_i[7] * (x567 - x490 + x591 + x614 + x618) - Mstar_i[8] * (x494 + x515 + x610 + x639) + Mstar_i[9] * (x563 + x620 + x638) + x133 * x627 + x135 * x628 + x136 * x630 + x137 * x625 + x138 * x632 + x416 * x628 + x576 * x626 + x576 * x631 + x579 + x635 + x640;
		pop(idx + c_alpha_offsets[25], 25) = -Mstar_i[7] * (x375 + x420 + x495 + x504 + x512 - x506 + x594 + x598 + x611) - Mstar_i[8] * (-x240 + x612 + x623 + x642) + Mstar_i[9] * (x589 - x492 + x621 + x641) + x109 * x646 + x110 * x647 + x112 * x643 + x115 * x649 + x187 * x644 + x363 * x644 + x535 * x645 + x535 * x648 + x608 + x637 + x650;
		pop(idx + c_alpha_offsets[26], 26) = Mstar_i[7] * (x499 + x566 + x592 + x614 + x651) - Mstar_i[8] * (x570 + x585 + x639 + x642) + Mstar_i[9] * (x638 + x641 + x651) + x135 * x646 + x136 * x647 + x137 * x643 + x138 * x649 + x187 * x652 + x363 * x652 + x576 * x645 + x576 * x648 + x617 + x640 + x650;
	});
}

template <>
void Flow_solver::LBM_CM_MRT_impl<2, 9>(const Parallel_MPI& parallel_MPI, const Thermal_solver& Thermal) {
	constexpr int D = 2;
	constexpr int Q = 9;
	const double cs2 = 1. / c_s2;
#if defined compressible
	/* rT in SI units */
	const double conv = 1. / sqr(global_parameters.D_x / global_parameters.D_t);
	const double r = R_GAS / M_av;
#endif
	std::array<double, 9> omega;
	std::array<double, 9> Mstar_i;
	non_solid_lattice.update_no_simd([&, this](Flat_index idx) {
#if defined compressible
		/* Temperature in SI units */
		double theta_temp = Thermal->temperature[idx] * Thermal->T_0;
		const double theta = r * theta_temp * conv;
#else
		constexpr double theta = 1. / 3.;
#endif
		/* We have three types of relaxations */
		const double omega_shear = 1. / (viscosity[idx] / theta + 0.5);
		constexpr double omega_bulk = 1.;

		/* GET CORRECTION */
		double dPxxx = 0, dPyyy = 0;
		bool S = false;
		for (int alpha = 1; alpha < Q; alpha++) {
			if (is_solid[idx + c_alpha_offsets[alpha]] == TRUE) {
				S = true;
				break;
			}
		}
		if (!S) {
			for (int alpha = 1; alpha < Q; alpha++) {
				const Flat_index alpha_idx = idx + c_alpha_offsets[alpha];
				const double WE = Stencil<D, Q>::w_alpha[alpha];
				double Up = velocity(alpha_idx, 0);
				double Vp = velocity(alpha_idx, 1);
				double rhop = density[alpha_idx];
#if defined compressible
				theta_temp = Thermal->temperature[alpha_idx] * Thermal->T_0;
				const double Tp = r * theta_temp * conv;
#else
				constexpr double Tp = 1. / 3.;
#endif
				double Pxxxp = (.5) * WE * Stencil<D, Q>::c_alpha[alpha][0] * rhop * Up * (Up * Up + 3. * (Tp - 1. / 3.));
				double Pyyyp = (.5) * WE * Stencil<D, Q>::c_alpha[alpha][1] * rhop * Vp * (Vp * Vp + 3. * (Tp - 1. / 3.));
				dPxxx += Pxxxp;
				dPyyy += Pyyyp;
			}
			const double scale = (2. / cs2) * (1. - omega_bulk * omega_shear / (omega_bulk + omega_shear));
			dPxxx *= scale;
			dPyyy *= scale;
		}
		velocity_corrections(idx, 0) = dPxxx;
		velocity_corrections(idx, 1) = dPyyy;
	});
	non_solid_lattice.update_no_simd([&](Flat_index idx) {
#if defined compressible
		/* Temperature in SI units */
		double theta_temp = Thermal->temperature[idx] * Thermal->T_0;
		const double theta = r * theta_temp * conv;
#else
		constexpr double theta = 1. / 3.;
#endif
		/* We have three types of relaxations */
		const double omega_shear = 1. / (viscosity[idx] / theta + 0.5);
		constexpr double omega_bulk = 1.;
		constexpr double omega_ghost = 1.;
		omega[0] = 1.;
		omega[1] = 1.;
		omega[2] = 1.;
		omega[3] = omega_shear;
		omega[4] = omega_shear;
		omega[5] = omega_bulk;
		omega[6] = omega_ghost;
		omega[7] = omega_ghost;
		omega[8] = omega_ghost;

		double rho = density[idx];
		double U = velocity(idx, 0);
		double V = velocity(idx, 1);
		double Fx = force(idx, 0);
		double Fy = force(idx, 1);

		const double dPxxx = velocity_corrections(idx, 0);
		const double dPyyy = velocity_corrections(idx, 1);
/// *************************************************************************************************** ///
///                                       EQUILIBRIUM CENTRAL MOMENTS                                   ///
///                                             MOMENT SPACE :                                          ///
///                                                  H_0,                                               ///
///                                                H_x, H_y,                                            ///
///                                     H_xy, H_xx - H_yy, H_yy + H_xx                                  ///
///                                              H_xxy, H_xyy                                           ///
///                                                  H_xxyy                                             ///
/// *************************************************************************************************** ///
/// FIRST STEP: GET CENTRAL HERMITE MOMENTS OF EDF (ASSUMING ALL TERMS SUPPORTED BY THE STENCIL ARE KEPT)
#ifndef PERFORMANCE_MODE
		Mstar_i[0] = rho * omega[0] - (omega[0] - 1.0) * (pop_old(idx, 0) + pop_old(idx, 1) + pop_old(idx, 2) + pop_old(idx, 3) + pop_old(idx, 4) + pop_old(idx, 5) + pop_old(idx, 6) + pop_old(idx, 7) + pop_old(idx, 8));
		Mstar_i[1] = (omega[1] - 1.0) * (U * pop_old(idx, 0) + U * pop_old(idx, 2) + U * pop_old(idx, 4) + pop_old(idx, 1) * (U - 1.0) + pop_old(idx, 3) * (U + 1.0) + pop_old(idx, 5) * (U - 1.0) + pop_old(idx, 6) * (U + 1.0) + pop_old(idx, 7) * (U + 1.0) + pop_old(idx, 8) * (U - 1.0));
		Mstar_i[2] = (omega[2] - 1.0) * (V * pop_old(idx, 0) + V * pop_old(idx, 1) + V * pop_old(idx, 3) + pop_old(idx, 2) * (V - 1.0) + pop_old(idx, 4) * (V + 1.0) + pop_old(idx, 5) * (V - 1.0) + pop_old(idx, 6) * (V - 1.0) + pop_old(idx, 7) * (V + 1.0) + pop_old(idx, 8) * (V + 1.0));
		Mstar_i[3] = -(omega[3] - 1.0) * (pop_old(idx, 5) * (U - 1.0) * (V - 1.0) + pop_old(idx, 6) * (U + 1.0) * (V - 1.0) + pop_old(idx, 7) * (U + 1.0) * (V + 1.0) + pop_old(idx, 8) * (U - 1.0) * (V + 1.0) + U * V * pop_old(idx, 0) + V * pop_old(idx, 1) * (U - 1.0) + U * pop_old(idx, 2) * (V - 1.0) + V * pop_old(idx, 3) * (U + 1.0) + U * pop_old(idx, 4) * (V + 1.0));
		Mstar_i[4] = -(omega[4] - 1.0) * (pop_old(idx, 1) * (sqr(U - 1.0) - V * V) - pop_old(idx, 2) * (sqr(V - 1.0) - U * U) + pop_old(idx, 3) * (sqr(U + 1.0) - V * V) - pop_old(idx, 4) * (sqr(V + 1.0) - U * U) + pop_old(idx, 5) * (sqr(U - 1.0) - sqr(V - 1.0)) + pop_old(idx, 6) * (sqr(U + 1.0) - sqr(V - 1.0)) + pop_old(idx, 7) * (sqr(U + 1.0) - sqr(V + 1.0)) + pop_old(idx, 8) * (sqr(U - 1.0) - sqr(V + 1.0)) + pop_old(idx, 0) * (U * U - V * V));
		Mstar_i[5] = -(omega[5] - 1.0) * (pop_old(idx, 5) * (cs2 * -2.0 + sqr(U - 1.0) + sqr(V - 1.0)) + pop_old(idx, 6) * (cs2 * -2.0 + sqr(U + 1.0) + sqr(V - 1.0)) + pop_old(idx, 7) * (cs2 * -2.0 + sqr(U + 1.0) + sqr(V + 1.0)) + pop_old(idx, 8) * (cs2 * -2.0 + sqr(U - 1.0) + sqr(V + 1.0)) + pop_old(idx, 0) * (cs2 * -2.0 + U * U + V * V) + pop_old(idx, 1) * (cs2 * -2.0 + sqr(U - 1.0) + V * V) + pop_old(idx, 2) * (cs2 * -2.0 + sqr(V - 1.0) + U * U) + pop_old(idx, 3) * (cs2 * -2.0 + sqr(U + 1.0) + V * V) + pop_old(idx, 4) * (cs2 * -2.0 + sqr(V + 1.0) + U * U)) + rho * omega[5] * (theta - cs2) * 2.0;
		Mstar_i[6] = -(omega[6] - 1.0) * (V * pop_old(idx, 1) * (cs2 - sqr(U - 1.0)) + pop_old(idx, 2) * (V - 1.0) * (cs2 - U * U) + V * pop_old(idx, 3) * (cs2 - sqr(U + 1.0)) + pop_old(idx, 4) * (V + 1.0) * (cs2 - U * U) + pop_old(idx, 5) * (cs2 - sqr(U - 1.0)) * (V - 1.0) + pop_old(idx, 6) * (cs2 - sqr(U + 1.0)) * (V - 1.0) + pop_old(idx, 7) * (cs2 - sqr(U + 1.0)) * (V + 1.0) + pop_old(idx, 8) * (cs2 - sqr(U - 1.0)) * (V + 1.0) + V * pop_old(idx, 0) * (cs2 - U * U));
		Mstar_i[7] = -(omega[7] - 1.0) * (pop_old(idx, 1) * (U - 1.0) * (cs2 - V * V) + U * pop_old(idx, 2) * (cs2 - sqr(V - 1.0)) + pop_old(idx, 3) * (U + 1.0) * (cs2 - V * V) + U * pop_old(idx, 4) * (cs2 - sqr(V + 1.0)) + pop_old(idx, 5) * (cs2 - sqr(V - 1.0)) * (U - 1.0) + pop_old(idx, 6) * (cs2 - sqr(V - 1.0)) * (U + 1.0) + pop_old(idx, 7) * (cs2 - sqr(V + 1.0)) * (U + 1.0) + pop_old(idx, 8) * (cs2 - sqr(V + 1.0)) * (U - 1.0) + U * pop_old(idx, 0) * (cs2 - V * V));
		Mstar_i[8] = -(omega[8] - 1.0) * (pop_old(idx, 5) * (cs2 - sqr(U - 1.0)) * (cs2 - sqr(V - 1.0)) + pop_old(idx, 6) * (cs2 - sqr(U + 1.0)) * (cs2 - sqr(V - 1.0)) + pop_old(idx, 7) * (cs2 - sqr(U + 1.0)) * (cs2 - sqr(V + 1.0)) + pop_old(idx, 8) * (cs2 - sqr(U - 1.0)) * (cs2 - sqr(V + 1.0)) + pop_old(idx, 0) * (cs2 - U * U) * (cs2 - V * V) + pop_old(idx, 1) * (cs2 - sqr(U - 1.0)) * (cs2 - V * V) + pop_old(idx, 2) * (cs2 - sqr(V - 1.0)) * (cs2 - U * U) + pop_old(idx, 3) * (cs2 - sqr(U + 1.0)) * (cs2 - V * V) + pop_old(idx, 4) * (cs2 - sqr(V + 1.0)) * (cs2 - U * U)) + rho * omega[8] * sqr(theta - cs2);
#endif
#ifdef PERFORMANCE_MODE
		Mstar_i[0] = rho;
		Mstar_i[1] = 0;
		Mstar_i[2] = 0;
		Mstar_i[3] = -(omega[3] - 1.0) * (pop_old(idx, 5) * (U - 1.0) * (V - 1.0) + pop_old(idx, 6) * (U + 1.0) * (V - 1.0) + pop_old(idx, 7) * (U + 1.0) * (V + 1.0) + pop_old(idx, 8) * (U - 1.0) * (V + 1.0) + U * V * pop_old(idx, 0) + V * pop_old(idx, 1) * (U - 1.0) + U * pop_old(idx, 2) * (V - 1.0) + V * pop_old(idx, 3) * (U + 1.0) + U * pop_old(idx, 4) * (V + 1.0));
		Mstar_i[4] = -(omega[4] - 1.0) * (pop_old(idx, 1) * (sqr(U - 1.0) - V * V) - pop_old(idx, 2) * (sqr(V - 1.0) - U * U) + pop_old(idx, 3) * (sqr(U + 1.0) - V * V) - pop_old(idx, 4) * (sqr(V + 1.0) - U * U) + pop_old(idx, 5) * (sqr(U - 1.0) - sqr(V - 1.0)) + pop_old(idx, 6) * (sqr(U + 1.0) - sqr(V - 1.0)) + pop_old(idx, 7) * (sqr(U + 1.0) - sqr(V + 1.0)) + pop_old(idx, 8) * (sqr(U - 1.0) - sqr(V + 1.0)) + pop_old(idx, 0) * (U * U - V * V));
		Mstar_i[5] = -(omega[5] - 1.0) * (pop_old(idx, 5) * (cs2 * -2.0 + sqr(U - 1.0) + sqr(V - 1.0)) + pop_old(idx, 6) * (cs2 * -2.0 + sqr(U + 1.0) + sqr(V - 1.0)) + pop_old(idx, 7) * (cs2 * -2.0 + sqr(U + 1.0) + sqr(V + 1.0)) + pop_old(idx, 8) * (cs2 * -2.0 + sqr(U - 1.0) + sqr(V + 1.0)) + pop_old(idx, 0) * (cs2 * -2.0 + U * U + V * V) + pop_old(idx, 1) * (cs2 * -2.0 + sqr(U - 1.0) + V * V) + pop_old(idx, 2) * (cs2 * -2.0 + sqr(V - 1.0) + U * U) + pop_old(idx, 3) * (cs2 * -2.0 + sqr(U + 1.0) + V * V) + pop_old(idx, 4) * (cs2 * -2.0 + sqr(V + 1.0) + U * U)) + rho * omega[5] * (theta - cs2) * 2.0;
		Mstar_i[6] = 0;
		Mstar_i[7] = 0;
		Mstar_i[8] = rho * sqr(theta - cs2);
#endif
		// THIRD STEP: ADD EXTERNAL FORCES
		Mstar_i[1] += Fx;
		Mstar_i[2] += Fy;
		Mstar_i[3] += (Fx * Fy) / rho;
		Mstar_i[4] += (rho * (dPxxx - dPyyy) + Fx * Fx - Fy * Fy) / rho;
		Mstar_i[5] += (rho * (dPxxx + dPyyy) + Fx * Fx + Fy * Fy) / rho;
		Mstar_i[6] += Fy * 1.0 / (rho * rho) * (dPxxx * rho + (rho * rho) * (theta - cs2) + Fx * Fx);
		Mstar_i[7] += Fx * 1.0 / (rho * rho) * (dPyyy * rho + (rho * rho) * (theta - cs2) + Fy * Fy);
		Mstar_i[8] += 0. - rho * sqr(theta - cs2) + 1.0 / (rho * rho * rho) * (dPxxx * rho + (rho * rho) * (theta - cs2) + Fx * Fx) * (dPyyy * rho + (rho * rho) * (theta - cs2) + Fy * Fy);

		pop(idx + c_alpha_offsets[0], 0) = Mstar_i[8] + U * Mstar_i[7] * 2.0 + V * Mstar_i[6] * 2.0 + Mstar_i[5] * (cs2 + (U * U) / 2.0 + (V * V) / 2.0 - 1.0) - (Mstar_i[4] * (U * U - V * V)) / 2.0 + U * V * Mstar_i[3] * 4.0 + U * Mstar_i[1] * (cs2 + V * V - 1.0) * 2.0 + V * Mstar_i[2] * (cs2 + U * U - 1.0) * 2.0 + Mstar_i[0] * (cs2 + U * U - 1.0) * (cs2 + V * V - 1.0);
		pop(idx + c_alpha_offsets[1], 1) = Mstar_i[8] * (-1.0 / 2.0) + Mstar_i[4] * (U / 4.0 + (U * U) / 4.0 - (V * V) / 4.0 + 1.0 / 4.0) - V * Mstar_i[6] - Mstar_i[7] * (U + 1.0 / 2.0) - Mstar_i[5] * (U / 4.0 + cs2 / 2.0 + (U * U) / 4.0 + (V * V) / 4.0 - 1.0 / 4.0) - (Mstar_i[0] * (cs2 + V * V - 1.0) * (U + cs2 + U * U)) / 2.0 - V * Mstar_i[3] * (U * 2.0 + 1.0) - (Mstar_i[1] * (U * 2.0 + 1.0) * (cs2 + V * V - 1.0)) / 2.0 - V * Mstar_i[2] * (U + cs2 + U * U);
		pop(idx + c_alpha_offsets[2], 2) = Mstar_i[8] * (-1.0 / 2.0) - Mstar_i[4] * (V / 4.0 - (U * U) / 4.0 + (V * V) / 4.0 + 1.0 / 4.0) - U * Mstar_i[7] - Mstar_i[6] * (V + 1.0 / 2.0) - Mstar_i[5] * (V / 4.0 + cs2 / 2.0 + (U * U) / 4.0 + (V * V) / 4.0 - 1.0 / 4.0) - (Mstar_i[0] * (cs2 + U * U - 1.0) * (V + cs2 + V * V)) / 2.0 - U * Mstar_i[3] * (V * 2.0 + 1.0) - (Mstar_i[2] * (V * 2.0 + 1.0) * (cs2 + U * U - 1.0)) / 2.0 - U * Mstar_i[1] * (V + cs2 + V * V);
		pop(idx + c_alpha_offsets[3], 3) = Mstar_i[8] * (-1.0 / 2.0) - Mstar_i[4] * (U / 4.0 - (U * U) / 4.0 + (V * V) / 4.0 - 1.0 / 4.0) - V * Mstar_i[6] - Mstar_i[7] * (U - 1.0 / 2.0) - Mstar_i[5] * (U * (-1.0 / 4.0) + cs2 / 2.0 + (U * U) / 4.0 + (V * V) / 4.0 - 1.0 / 4.0) - V * Mstar_i[3] * (U * 2.0 - 1.0) - V * Mstar_i[2] * (-U + cs2 + U * U) - (Mstar_i[1] * (U * 2.0 - 1.0) * (cs2 + V * V - 1.0)) / 2.0 - (Mstar_i[0] * (-U + cs2 + U * U) * (cs2 + V * V - 1.0)) / 2.0;
		pop(idx + c_alpha_offsets[4], 4) = Mstar_i[8] * (-1.0 / 2.0) + Mstar_i[4] * (V / 4.0 + (U * U) / 4.0 - (V * V) / 4.0 - 1.0 / 4.0) - U * Mstar_i[7] - Mstar_i[6] * (V - 1.0 / 2.0) - Mstar_i[5] * (V * (-1.0 / 4.0) + cs2 / 2.0 + (U * U) / 4.0 + (V * V) / 4.0 - 1.0 / 4.0) - U * Mstar_i[3] * (V * 2.0 - 1.0) - U * Mstar_i[1] * (-V + cs2 + V * V) - (Mstar_i[2] * (V * 2.0 - 1.0) * (cs2 + U * U - 1.0)) / 2.0 - (Mstar_i[0] * (-V + cs2 + V * V) * (cs2 + U * U - 1.0)) / 2.0;
		pop(idx + c_alpha_offsets[5], 5) = Mstar_i[8] / 4.0 + Mstar_i[5] * (U / 8.0 + V / 8.0 + cs2 / 4.0 + (U * U) / 8.0 + (V * V) / 8.0) - Mstar_i[4] * (U / 8.0 - V / 8.0 + (U * U) / 8.0 - (V * V) / 8.0) + Mstar_i[7] * (U / 2.0 + 1.0 / 4.0) + Mstar_i[6] * (V / 2.0 + 1.0 / 4.0) + (Mstar_i[0] * (U + cs2 + U * U) * (V + cs2 + V * V)) / 4.0 + (Mstar_i[2] * (V * 2.0 + 1.0) * (U + cs2 + U * U)) / 4.0 + (Mstar_i[1] * (U * 2.0 + 1.0) * (V + cs2 + V * V)) / 4.0 + (Mstar_i[3] * (U * 2.0 + 1.0) * (V * 2.0 + 1.0)) / 4.0;
		pop(idx + c_alpha_offsets[6], 6) = Mstar_i[8] / 4.0 + Mstar_i[5] * (U * (-1.0 / 8.0) + V / 8.0 + cs2 / 4.0 + (U * U) / 8.0 + (V * V) / 8.0) + Mstar_i[4] * (U / 8.0 + V / 8.0 - (U * U) / 8.0 + (V * V) / 8.0) + Mstar_i[7] * (U / 2.0 - 1.0 / 4.0) + Mstar_i[6] * (V / 2.0 + 1.0 / 4.0) + (Mstar_i[2] * (V * 2.0 + 1.0) * (-U + cs2 + U * U)) / 4.0 + (Mstar_i[1] * (U * 2.0 - 1.0) * (V + cs2 + V * V)) / 4.0 + (Mstar_i[0] * (-U + cs2 + U * U) * (V + cs2 + V * V)) / 4.0 + (Mstar_i[3] * (U * 2.0 - 1.0) * (V * 2.0 + 1.0)) / 4.0;
		pop(idx + c_alpha_offsets[7], 7) = Mstar_i[8] / 4.0 + Mstar_i[5] * (U * (-1.0 / 8.0) - V / 8.0 + cs2 / 4.0 + (U * U) / 8.0 + (V * V) / 8.0) + Mstar_i[4] * (U / 8.0 - V / 8.0 - (U * U) / 8.0 + (V * V) / 8.0) + Mstar_i[7] * (U / 2.0 - 1.0 / 4.0) + Mstar_i[6] * (V / 2.0 - 1.0 / 4.0) + (Mstar_i[2] * (V * 2.0 - 1.0) * (-U + cs2 + U * U)) / 4.0 + (Mstar_i[1] * (U * 2.0 - 1.0) * (-V + cs2 + V * V)) / 4.0 + (Mstar_i[0] * (-U + cs2 + U * U) * (-V + cs2 + V * V)) / 4.0 + (Mstar_i[3] * (U * 2.0 - 1.0) * (V * 2.0 - 1.0)) / 4.0;
		pop(idx + c_alpha_offsets[8], 8) = Mstar_i[8] / 4.0 + Mstar_i[5] * (U / 8.0 - V / 8.0 + cs2 / 4.0 + (U * U) / 8.0 + (V * V) / 8.0) - Mstar_i[4] * (U / 8.0 + V / 8.0 + (U * U) / 8.0 - (V * V) / 8.0) + Mstar_i[7] * (U / 2.0 + 1.0 / 4.0) + Mstar_i[6] * (V / 2.0 - 1.0 / 4.0) + (Mstar_i[1] * (U * 2.0 + 1.0) * (-V + cs2 + V * V)) / 4.0 + (Mstar_i[2] * (V * 2.0 - 1.0) * (U + cs2 + U * U)) / 4.0 + (Mstar_i[0] * (-V + cs2 + V * V) * (U + cs2 + U * U)) / 4.0 + (Mstar_i[3] * (U * 2.0 + 1.0) * (V * 2.0 - 1.0)) / 4.0;
	});
}
