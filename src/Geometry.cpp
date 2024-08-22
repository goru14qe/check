#include <iostream>
#include <cstdio>
#include <fstream>
#include <vector>
#include "Geometry.h"
#include <mpi.h>
#include <sstream>
#include "ALBORZ_Macros.h"
#include "ALBORZ_SETTINGS.h"
#include "ALBORZ_GlobalVariables.h"
#include "Flow_solver.h"
#include "Thermal_solver.h"
#include "Species_solver.h"

/// ***************************************************** ///
/// FINITE-DIFFERENCE DISCRETIZATION SCHEMES              ///
/// ***************************************************** ///
namespace FD {
/// ***************************************************** ///
/// 5TH ORDER WENO                                        ///
/// ***************************************************** ///
double WENO5NONCONS(double u, double f1, double f2, double f3, double f4, double f5, double f6, double f7) {
	/// fp is flux at x+1/2 and fn at x-1/2
	/// f1 is the value from cell x-2 ...
	double f, fp1, fp2, fp3, fn1, fn2, fn3;
	double fp, fn;
	double w1, w2, w3, wtot;
	double beta1, beta2, beta3;
	double epsilon = 1e-50;
	/// ****************************************************************************************************** ///
	///        f1(x-3)      f2(x-2)      f3(x-1)  x-1/2  f4(x)  x-1/2 f5(x+1)      f6(x+2)      f7(x+3)        ///
	///         |            |              |      |      |       |     |             |            |           ///
	///                     |______________________________|                                                   ///
	///                                    f1p                                                                 ///
	///                                    |_____________________________|                                     ///
	///                                                   f2p                                                  ///
	///                                                 |______________________________|                       ///
	///                                                                  f3p                                   ///
	///        |____________________________|                                                                  ///
	///                      f1n                                                                               ///
	///                     |____________________________|                                                     ///
	///                                    f2n                                                                 ///
	///                                    |____________________________|                                      ///
	///                                                   f3n                                                  ///
	/// ****************************************************************************************************** ///
	if (u >= 0) {
		fp1 = (2. / 6.) * f2 - (7. / 6.) * f3 + (11. / 6.) * f4;
		fp2 = -(1. / 6.) * f3 + (5. / 6.) * f4 + (2. / 6.) * f5;
		fp3 = (2. / 6.) * f4 + (5. / 6.) * f5 - (1. / 6.) * f6;
		beta1 = (13. / 12.) * sqr(1. * f2 - 2 * f3 + 1. * f4) + (1. / 4.) * sqr(1. * f2 - 4. * f3 + 3. * f4);
		beta2 = (13. / 12.) * sqr(1. * f3 - 2 * f4 + 1. * f5) + (1. / 4.) * sqr(1. * f3 - 0. * f4 - 1. * f5);
		beta3 = (13. / 12.) * sqr(1. * f4 - 2 * f5 + 1. * f6) + (1. / 4.) * sqr(1. * f4 - 4. * f5 + 3. * f6);
		w1 = (1. / 10.) / sqr(epsilon + beta1);
		w2 = (6. / 10.) / sqr(epsilon + beta2);
		// todo: checkme!
		// this value was assigned to w2 before, leaving w3 uninitialized
		// the same was true for repeats of this line
		w3 = (3. / 10.) / sqr(epsilon + beta3);
		wtot = w1 + w2 + w3;
		w1 = w1 / wtot;
		w2 = w2 / wtot;
		w3 = w3 / wtot;
		fp = w1 * fp1 + w2 * fp2 + w3 * fp3;

		fn1 = (2. / 6.) * f1 - (7. / 6.) * f2 + (11. / 6.) * f3;
		fn2 = -(1. / 6.) * f2 + (5. / 6.) * f3 + (2. / 6.) * f4;
		fn3 = (2. / 6.) * f3 + (5. / 6.) * f4 - (1. / 6.) * f5;
		beta1 = (13. / 12.) * sqr(1. * f1 - 2 * f2 + 1. * f3) + (1. / 4.) * sqr(1. * f1 - 4. * f2 + 3. * f5);
		beta2 = (13. / 12.) * sqr(1. * f2 - 2 * f3 + 1. * f4) + (1. / 4.) * sqr(1. * f2 - 0. * f3 - 1. * f6);
		beta3 = (13. / 12.) * sqr(1. * f3 - 2 * f4 + 1. * f5) + (1. / 4.) * sqr(1. * f3 - 4. * f4 + 3. * f7);
		w1 = (1. / 10.) / sqr(epsilon + beta1);
		w2 = (6. / 10.) / sqr(epsilon + beta2);
		w3 = (3. / 10.) / sqr(epsilon + beta3);
		wtot = w1 + w2 + w3;
		w1 = w1 / wtot;
		w2 = w2 / wtot;
		w3 = w3 / wtot;
		// todo: checkme!
		// fp was assigned here, leaving fn uninitialized
		fn = w1 * fn1 + w2 * fn2 + w3 * fn3;
	}
	/// ****************************************************************************************************** ///
	///        f1(x-3)      f2(x-2)      f3(x-1)  x-1/2  f4(x)  x-1/2 f5(x+1)      f6(x+2)      f7(x+3)        ///
	///         |            |              |      |      |       |     |             |            |           ///
	///                                                                |____________________________|          ///
	///                                                                               f1p                      ///
	///                                                  |_____________________________|                       ///
	///                                                                 f2p                                    ///
	///                                    |______________________________|                                    ///
	///                                                    f3p                                                 ///
	///                                                   |____________________________|                       ///
	///                                                               f1n                                      ///
	///                                    |____________________________|                                      ///
	///                                                   f2n                                                  ///
	///              |______________________|                                                                  ///
	///                        f3n                                                                             ///
	/// ****************************************************************************************************** ///
	if (u < 0) {
		fp1 = (2. / 6.) * f5 - (7. / 6.) * f6 + (11. / 6.) * f7;
		fp2 = -(1. / 6.) * f4 + (5. / 6.) * f5 + (2. / 6.) * f6;
		fp3 = (2. / 6.) * f3 + (5. / 6.) * f4 - (1. / 6.) * f5;
		beta1 = (13. / 12.) * sqr(1. * f5 - 2 * f6 + 1. * f7) + (1. / 4.) * sqr(1. * f5 - 4. * f6 + 3. * f7);
		beta2 = (13. / 12.) * sqr(1. * f4 - 2 * f5 + 1. * f6) + (1. / 4.) * sqr(1. * f4 - 0. * f5 - 1. * f6);
		beta3 = (13. / 12.) * sqr(1. * f3 - 2 * f4 + 1. * f5) + (1. / 4.) * sqr(1. * f3 - 4. * f4 + 3. * f5);
		w1 = (1. / 10.) / sqr(epsilon + beta1);
		w2 = (6. / 10.) / sqr(epsilon + beta2);
		w3 = (3. / 10.) / sqr(epsilon + beta3);
		wtot = w1 + w2 + w3;
		w1 = w1 / wtot;
		w2 = w2 / wtot;
		w3 = w3 / wtot;
		fp = w1 * fp1 + w2 * fp2 + w3 * fp3;

		fn1 = (2. / 6.) * f4 - (7. / 6.) * f5 + (11. / 6.) * f6;
		fn2 = -(1. / 6.) * f3 + (5. / 6.) * f4 + (2. / 6.) * f5;
		fn3 = (2. / 6.) * f2 + (5. / 6.) * f3 - (1. / 6.) * f4;
		beta1 = (13. / 12.) * sqr(1. * f4 - 2 * f5 + 1. * f6) + (1. / 4.) * sqr(1. * f4 - 4. * f5 + 3. * f6);
		beta2 = (13. / 12.) * sqr(1. * f3 - 2 * f4 + 1. * f5) + (1. / 4.) * sqr(1. * f3 - 0. * f4 - 1. * f5);
		beta3 = (13. / 12.) * sqr(1. * f2 - 2 * f3 + 1. * f4) + (1. / 4.) * sqr(1. * f2 - 4. * f3 + 3. * f4);
		w1 = (1. / 10.) / sqr(epsilon + beta1);
		w2 = (6. / 10.) / sqr(epsilon + beta2);
		w3 = (3. / 10.) / sqr(epsilon + beta3);
		wtot = w1 + w2 + w3;
		w1 = w1 / wtot;
		w2 = w2 / wtot;
		w3 = w3 / wtot;
		fn = w1 * fn1 + w2 * fn2 + w3 * fn3;
	}
	f = u * (fp - fn);
	return f;
}
/// ***************************************************** ///
/// 3RD ORDER WENO                                        ///
/// ***************************************************** ///
/*
Here f1, f2, f3, f4, f5 are the values of the function at x-2, x-1, x, x+1, x+2
u is the velocity at the cell interface
it basically calculates ∂(u*f)/∂x at the cell interface.
*/
double WENO3NONCONS(double u, double f1, double f2, double f3, double f4, double f5) {
	/// fp is flux at x+1/2 and fn at x-1/2
	/// f1 is the value from cell x-2 ...
	double f, fp1, fp2, fn1, fn2;
	double fp, fn;
	double w1, w2, wtot;
	double beta1, beta2;
	double epsilon = 1e-50;
	/// ********************************************************************************** ///
	///        f1(x-2)      f2(x-1) x-1/2  f3(x)  x+1/2  f4(x+1)      f5(x+2)              ///
	///         |            |       |      |      |      |            |                   ///
	///                     |________________|                                             ///
	///                            f1p                                                     ///
	///                                    |________________|                              ///
	///                                            f2p                                     ///
	///        |______________|                                                            ///
	///              f2n                                                                   ///
	///                     |_______________|                                              ///
	///                            f1n                                                     ///
	/// ********************************************************************************** ///
	if (u >= 0) {
		fp1 = -0.5 * f2 + 1.5 * f3;
		fp2 = 0.5 * f3 + 0.5 * f4;
		beta1 = sqr(f3 - f2);
		beta2 = sqr(f4 - f3);
		w1 = (1. / 3.) / sqr(epsilon + beta1);
		w2 = (2. / 3.) / sqr(epsilon + beta2);
		wtot = w1 + w2;
		w1 = w1 / wtot;
		w2 = w2 / wtot;
		fp = w1 * fp1 + w2 * fp2;

		fn1 = 1.5 * f2 - 0.5 * f1;
		fn2 = 0.5 * f2 + 0.5 * f3;
		beta1 = sqr(f2 - f1);
		beta2 = sqr(f3 - f2);
		w1 = (1. / 3.) / sqr(epsilon + beta1);
		w2 = (2. / 3.) / sqr(epsilon + beta2);
		wtot = w1 + w2;
		w1 = w1 / wtot;
		w2 = w2 / wtot;
		fn = w1 * fn1 + w2 * fn2;
	}
	/// ********************************************************************************** ///
	///        f1(x-2)      f2(x-1) x-1/2  f3(x)  x+1/2  f4(x+1)      f5(x+2)              ///
	///         |            |       |      |      |      |            |                   ///
	///                                    |________________|                              ///
	///                                           f1p                                      ///
	///                                                  |________________|                ///
	///                                                          f2p                       ///
	///                     |______________|                                               ///
	///                            f2n                                                     ///
	///                                   |_______________|                                ///
	///                                          f1n                                       ///
	/// ********************************************************************************** ///
	if (u < 0) {
		fp1 = 1.5 * f4 - 0.5 * f5;
		fp2 = 0.5 * f3 + 0.5 * f4;
		beta1 = sqr(f4 - f5);
		beta2 = sqr(f3 - f4);
		w1 = ((1. / 3.) / sqr(epsilon + beta1));
		w2 = ((2. / 3.) / sqr(epsilon + beta2));
		wtot = w1 + w2;
		w1 = w1 / wtot;
		w2 = w2 / wtot;
		fp = w1 * fp1 + w2 * fp2;

		fn1 = 1.5 * f3 - 0.5 * f4;
		fn2 = 0.5 * f2 + 0.5 * f3;
		beta1 = sqr(f3 - f4);
		beta2 = sqr(f2 - f3);
		w1 = ((1. / 3.) / sqr(epsilon + beta1));
		w2 = ((2. / 3.) / sqr(epsilon + beta2));
		wtot = w1 + w2;
		w1 = w1 / wtot;
		w2 = w2 / wtot;
		fn = w1 * fn1 + w2 * fn2;
	}
	f = u * (fp - fn);
	return f;
}
/// ***************************************************** ///
/// 1ST ORDER UPWIND                                      ///
/// ***************************************************** ///
double UPWIND1NONCONS(double u, double f1, double f2, double f3) {
	double f;
	if (u >= 0) {
		f = u * (f2 - f1);
	}
	if (u < 0) {
		f = u * (f3 - f2);
	}
	return f;
}
/// ***************************************************** ///
/// 2ND ORDER UPWIND                                      ///
/// ***************************************************** ///
double UPWIND2NONCONS(double u, double f1, double f2, double f3, double f4, double f5) {
	double f;
	if (u >= 0) {
		f = .5 * u * (3 * f3 - 4 * f2 + f1);
	}
	if (u < 0) {
		f = .5 * u * (4 * f4 - f5 - 3 * f3);
	}
	return f;
}
/// ***************************************************** ///
/// 3RD ORDER UPWIND                                      ///
/// ***************************************************** ///
double UPWIND3NONCONS(double u, double f1, double f2, double f3, double f4, double f5) {
	double f;
	if (u >= 0) {
		f = (1. / 6.) * u * (2 * f4 + f3 - 6 * f2 + f1);
	}
	if (u < 0) {
		f = (1. / 6.) * u * (6 * f4 - f5 - 3 * f3 - 2 * f2);
	}
	return f;
}
/// ***************************************************** ///
/// 2ND ORDER CENTRAL                                     ///
/// ***************************************************** ///
double CENTRALNONCONS(double u, double f1, double f2) {
	/// fp is flux at x+1/2 and fn at x-1/2
	/// f1 is the value from cell x-2 ...
	double f;
	f = 0.5 * u * (f2 - f1);
	return f;
}
/// ***************************************************** ///
/// 4TH ORDER CENTRAL                                     ///
/// ***************************************************** ///
double CENTRAL4NONCONS(double u, double f1, double f2, double f3, double f4, double f5) {
	/// fp is flux at x+1/2 and fn at x-1/2
	/// f1 is the value from cell x-2 ...
	double f;
	f = (1. / 12.) * u * (f1 - 8. * f2 + 8. * f4 - f5);
	return f;
}
/// ***************************************************** ///
/// 2ND ORDER CENTRAL FOR 2ND ORDER DERIVATIVE            ///
/// ***************************************************** ///
/* here f1, f2, f3, f4 are the values of the function at x-1, x, x+1, x+2 */
double CENTRAL2FLUX(double f1, double f2, double D1, double D2) {
	double f;
	f = 0.5 * (D2 + D1) * (f2 - f1);
	return f;
}
/// ***************************************************** ///
/// 4TH ORDER CENTRAL FOR 2ND ORDER DERIVATIVE            ///
/// ***************************************************** ///
/*
The CENTRAL4FLUX function is a central difference scheme that uses four points (two on each side of the current point) to compute a second-order derivative. The formula inside the function combines the values and diffusion coefficients at these points to provide a weighted average approximation of the second derivative.
here f1, f2, f3, f4 are the values of the function at x-1, x, x+1, x+2
and D1, D2, D3, D4 are the values of the diffusion coefficient at x-1/2, x, x+1/2, x+3/2
*/
double CENTRAL4FLUX(double f1, double f2, double f3, double f4, double D1, double D2, double D3, double D4) {
	double f;
	f = D1 * (f1 / 8. - f2 / 6. + f3 / 24.)
	    + D2 * (-f1 / 6. - 3. * f2 / 8. + 2. * f3 / 3. - f4 / 8.)
	    + D3 * (f1 / 8. - 2. * f2 / 3. + 3. * f3 / 8. + f4 / 6.)
	    + D4 * (-1. * f2 / 24. + f3 / 6. - f4 / 8.);
	return f;
}
/// ***************************************************** ///
/// SMOOTHEN SPECIES AND TEMPERATURE FIELDS               ///
/// ***************************************************** ///
void smoothen_fields(Flow_solver* Flow, Thermal_solver* Thermal, Species_solver* Species, Parallel_MPI* MPI_parallel, unsigned int Nt, double D) {
	double temp_rho_0 = Flow->rho_0;
	if (MPI_parallel->processor_id != MASTER) {
		for (int X = 0; X < MPI_parallel->dev_end[0]; ++X) {
			for (int Y = 0; Y < MPI_parallel->dev_end[1]; ++Y) {
				for (int Z = 0; Z < MPI_parallel->dev_end[2]; ++Z) {
					Thermal->thermal_diffusion_coefficient[{X, Y, Z}] = D * Thermal->c_p[{X, Y, Z}];
					/// Thermal->c_p[{X,Y,Z}] = 1;
					//Flow->velocity[{X, Y, Z, 0}] = 0;
					//Flow->velocity[{X, Y, Z, 1}] = 0;
					//Flow->velocity[{X, Y, Z, 2}] = 0;
					Flow->density[{X, Y, Z}] = 1;
					Flow->rho_0 = 1;
					for (int k = 0; k < Species->Nb_spec; ++k) {
						Species->diffusion_coefficient[{X, Y, Z, k}] = D;
						Species->Production[{X, Y, Z, k}] = 0;
					}
				}
			}
		}
	}
	Species->BC(0, Flow, Thermal, MPI_parallel);
	Thermal->BC(0, Flow, MPI_parallel);
	for (unsigned int t = 0; t < Nt; t++) {
		/// ***************************************
		///  LBM ALGORITHM : (a) TEMPERATURE FD
		/// ***************************************
		Thermal->FD_Euler(t, Flow, Species, MPI_parallel);
		Thermal->FD_Euler_diffusion(t, Flow, Species, MPI_parallel);
		/// ***************************************
		///  LBM ALGORITHM : (b) SPECIES FD
		/// ***************************************
		Species->FD_Euler(t, Flow, Thermal, MPI_parallel);
		Species->FD_HC_Euler(t, Flow, Thermal, MPI_parallel, 1);
		/// *****************************************
		///  BOUNDARY CONDITIONS : (b) TEMPERATURE
		/// *****************************************
		Thermal->BC(t, Flow, MPI_parallel);
		/// *****************************************
		///  BOUNDARY CONDITIONS : (c) SPECIES
		/// *****************************************
		Species->BC(t, Flow, Thermal, MPI_parallel);
		/// ********************************************************************
		///  MACRO DATA EXCHANGE BETWEEN CORES : NEEDED FOR FD-TYPE APPROACHES
		/// ********************************************************************
		Flow->Data_Exchange_Macroscopic(MPI_parallel);
		Thermal->Data_Exchange_Macroscopic(MPI_parallel);
		Species->Data_Exchange_Macroscopic(MPI_parallel);
	}
	Flow->rho_0 = temp_rho_0;
}

}  // namespace FD
namespace stl {
std::ostream& operator<<(std::ostream& out, const point p) {
	out << "(" << p.x << ", " << p.y << ", " << p.z << ")" << std::endl;
	return out;
}
std::ostream& operator<<(std::ostream& out, const triangle& t) {
	out << "---- TRIANGLE ----" << std::endl;
	out << t.normal << std::endl;
	out << t.v1 << std::endl;
	out << t.v2 << std::endl;
	out << t.v3 << std::endl;
	return out;
}
float parse_float(std::ifstream& s) {
	char f_buf[sizeof(float)];
	s.read(f_buf, 4);
	float* fptr = (float*)f_buf;
	return *fptr;
}
point parse_point(std::ifstream& s) {
	float x = parse_float(s);
	float y = parse_float(s);
	float z = parse_float(s);
	return point(x, y, z);
}
stl_data parse_stl(const std::string& stl_path) {
	std::ifstream stl_file(stl_path.c_str(), std::ios::in | std::ios::binary);
	if (!stl_file) {
		std::cerr << "ERROR: COULD NOT READ FILE '" << stl_path << "'" << std::endl;
		std::abort();
	}

	char header_info[80] = "";  // It reads 80 bytes for the header information and 4 bytes for the number of triangles.
	char n_triangles[4];
	stl_file.read(header_info, 80);
	stl_file.read(n_triangles, 4);
	std::string h(header_info);  // The header information is converted to a string and used to construct an stl_data object.
	stl_data info(h);
	unsigned int* r = (unsigned int*)n_triangles;
	unsigned int num_triangles = *r;
	for (unsigned int i = 0; i < num_triangles; i++) {  // For each triangle, it calls the parse_point function four times to read the normal vector and three vertices of the triangle.
		point normal = parse_point(stl_file);
		point v1 = parse_point(stl_file);
		point v2 = parse_point(stl_file);
		point v3 = parse_point(stl_file);
		info.triangles.push_back(triangle(normal, v1, v2, v3));  // A triangle object is constructed using this information and added to the info.triangles vector.
		char dummy[2];
		stl_file.read(dummy, 2);
	}
	return info;  // The function returns the stl_data object containing the parsed information.
}
double DOT_stl(const point& A, const point& B) {
	double C;
	C = A.x * B.x + A.y * B.y + A.z * B.z;
	return C;
}
void SUB_stl(point& C, const point& A, const point& B) {
	C.x = A.x - B.x;
	C.y = A.y - B.y;
	C.z = A.z - B.z;
}
void CROSS_stl(point& C, const point& A, const point& B) {
	C.x = A.y * B.z - A.z * B.y;
	C.y = A.z * B.x - A.x * B.z;
	C.z = A.x * B.y - A.y * B.x;
}

double DISTANCE(const point& A, const point& B) {
	return std::sqrt(sqr(A.x - B.x) + sqr(A.y - B.y) + sqr(A.z - B.z));
}
}  // namespace stl

Geometry::Geometry() {
}

void Geometry::Get_matrix_from_bmp(const std::string& filename) {
	FILE* f = fopen(filename.c_str(), "r");

	constexpr size_t BMP_HEADER_SIZE = 54;
	unsigned char head[BMP_HEADER_SIZE];
	const size_t bytesRead = fread(head, 1, BMP_HEADER_SIZE, f);
	if (bytesRead < BMP_HEADER_SIZE) {
		std::cerr << "File \"" << filename << "\" is not a valid bmp.\n";
		std::abort();
	}
	w = head[18] + (((int)head[19]) << 8) + (((int)head[20]) << 16) + (((int)head[21]) << 24);
	h = head[22] + (((int)head[23]) << 8) + (((int)head[24]) << 16) + (((int)head[25]) << 24);
	std::cout << "Geometry read from *.bmp file"
			  << "\n";
	std::cout << "-------------------------"
			  << "\n";
	std::cout << "Image size (XxY) : " << w << " " << h << "\n";
	// lines are aligned on 4-byte boundary
	int lineSize = (w / 32) * 4;
	if (w % 32)
		lineSize += 4;
	int fileSize = lineSize * h;

	unsigned char* img_temp = new unsigned char[w * h];
	img = new int*[w];
	for (int i = 0; i < w; i++) {
		img[i] = new int[h];
	}
	unsigned char* data = new unsigned char[fileSize];

	// skip the header
	fseek(f, 54, SEEK_SET);

	// skip palette - two rgb quads, 8 bytes
	fseek(f, 8, SEEK_CUR);

	// read data
	const size_t dataBytesRead = fread(data, 1, fileSize, f);
	if (dataBytesRead != static_cast<size_t>(fileSize)) {
		std::cerr << "Unexpected end of file in  \"" << filename << "\".\n";
		std::abort();
	}

	// decode bits
	int i, j, k, rev_j;
	for (j = 0, rev_j = h - 1; j < h; j++, rev_j--) {
		for (i = 0; i <= w / 8; i++) {
			int fpos = j * lineSize + i, pos = rev_j * w + i * 8;
			for (k = 0; k < 8; k++) {
				if (i < w / 8 || k >= 8 - (w % 8)) {
					img_temp[pos + (7 - k)] = (data[fpos] >> k) & 1;
				}
			}
		}
	}
	for (j = 0; j < h; j++) {
		for (i = 0; i < w; i++) {
			img[i][h - j - 1] = (int)(img_temp[j * w + i]);
		}
	}
}
namespace interpolation {
double Lin_Int_1d(double x1, double x2, double f1, double f2, double x) {
	double f;
	///  The interpolation is directly based on a Lagrange polynomial formulation
	//                 First interpolation in X direction
	f = (x - x2) * f1 / (x1 - x2) + (x - x1) * f2 / (x2 - x1);
	return f;
}
double Lin_Int(double x1, double x2, double y1, double y2, double f1, double f2, double f3, double f4, double x, double y) {
	double f;
	///  The interpolation is directly based on a Lagrange polynomial formulation
	//                 First interpolation in X direction
	f = ((x - x2) * f1 / (x1 - x2) + (x - x1) * f2 / (x2 - x1)) * (y - y2) / (y1 - y2) + ((x - x2) * f3 / (x1 - x2) + (x - x1) * f4 / (x2 - x1)) * (y - y1) / (y2 - y1);
	return f;
}
double Quad_Int(double x0, double x1, double x2, double y0, double y1, double y2, double x) {
	double y;
	///  The interpolation is directly based on a Lagrange polynomial formulation
	y = ((x - x1) * (x - x2)) * y0 / ((x0 - x1) * (x0 - x2)) + ((x - x0) * (x - x2)) * y1 / ((x1 - x0) * (x1 - x2)) + ((x - x0) * (x - x1)) * y2 / ((x2 - x0) * (x2 - x1));
	return y;
}
}  // namespace interpolation

void Initial_field_slice::read_flame_properties(const std::string& filename, const Parallel_MPI& MPI_parallel) {
	std::ifstream input_file(filename + ".dat", std::ios::binary);
	find_line_after_header(input_file, "c\tFlame properties");
	find_line_after_comment(input_file);
	input_file >> is_2D >> xflamepos >> flow_dir >> profile_type;
	if (profile_type == "tanh") {
		input_file >> stiffness;
	} else if (profile_type == "parabolic" || profile_type == "sphere") {
		input_file >> radius;
	} else if (profile_type == "box") {
		input_file >> box_length >> box_width;
	} else if (profile_type == "gaussian") {
		input_file >> width;
	} else if (profile_type == "linear_gradient") {
		input_file >> linear_start >> linear_end;
	} else if (profile_type == "exponential_decay") {
		input_file >> decay_rate;
	} else if (profile_type == "step_function") {
		input_file >> num_steps;
		step_positions.resize(num_steps);
		step_values.resize(num_steps);
		for (int i = 0; i < num_steps; ++i) {
			input_file >> step_positions[i] >> step_values[i];
		}
	} else {
		throw std::invalid_argument("Invalid profile type");
	}
	if (MPI_parallel.processor_id == (MASTER+1)) {
		std::cout << "Flame Properties From File \n";
		std::cout << "================================ \n";
		std::cout << "2D: " << (is_2D ? "Yes" : "No") << "\n";
		std::cout << "Flame Position: " << xflamepos << "\n";
		std::cout << "Flow Direction: " << flow_dir << "\n";
		std::cout << "Profile Type: " << profile_type << "\n";
		if (profile_type == "tanh") {
			std::cout << "Stiffness: " << stiffness << "\n";
		} else if (profile_type == "parabolic" || profile_type == "sphere") {
			std::cout << "Radius: " << radius << "\n";
		} else if (profile_type == "box") {
			std::cout << "Box Length: " << box_length << "\n";
			std::cout << "Box Width: " << box_width << "\n";
		} else if (profile_type == "gaussian") {
			std::cout << "Width: " << width << "\n";
		} else if (profile_type == "linear_gradient") {
			std::cout << "Start: " << linear_start << "\n";
			std::cout << "End: " << linear_end << "\n";
		} else if (profile_type == "exponential_decay") {
			std::cout << "Decay Rate: " << decay_rate << "\n";
		} else if (profile_type == "step_function") {
			std::cout << "Number of Steps: " << num_steps << "\n";
			for (int i = 0; i < num_steps; ++i) {
				std::cout << "Position: " << step_positions[i] << " Value: " << step_values[i] << "\n";
			}
		}
		std::cout << "================================ \n";
	}
	input_file.close();
}

// Function to calculate profile value 's'
double Initial_field_slice::calculate_profile(const std::string& profile_type, Solid_field& solid, const std::string& filename, const Parallel_MPI& MPI_parallel, int X, int Y, int Z) {
	if (MPI_parallel.is_master()) {
		return 0.0;
	}
	int index, global_param, perp_global_param1 = 0, perp_global_param2 = 0;
	if (flow_dir == "X") {
		index = 0;
		global_param = global_parameters.Nx;
		perp_global_param1 = global_parameters.Ny;  // Y dimension in 2D
		perp_global_param2 = global_parameters.Nz;  // Z dimension in 3D
	} else if (flow_dir == "Y") {
		index = 1;
		global_param = global_parameters.Ny;
		perp_global_param1 = global_parameters.Nx;  // X dimension in 2D
		perp_global_param2 = global_parameters.Nz;  // Z dimension in 3D
	} else if (flow_dir == "Z") {
		index = 2;
		global_param = global_parameters.Nz;
		perp_global_param1 = global_parameters.Nx;  // X dimension in 3D
		perp_global_param2 = global_parameters.Ny;  // Y dimension in 3D
	} else {
		throw std::invalid_argument("Invalid flow_dir");
	}
	double s = 0.0;
	const double flamepos = global_param * xflamepos;
	const double xx = fmod(X - MPI_parallel.start_XYZ2[0] + MPI_parallel.start_XYZ[0] + global_parameters.Nx, global_parameters.Nx);
	const double yy = fmod(Y - MPI_parallel.start_XYZ2[1] + MPI_parallel.start_XYZ[1] + global_parameters.Ny, global_parameters.Ny);
	const double zz = fmod(Z - MPI_parallel.start_XYZ2[2] + MPI_parallel.start_XYZ[2] + global_parameters.Nz, global_parameters.Nz);

	if (solid[{X, Y, Z}] == FALSE) {
		const double coord = (index == 0) ? xx : (index == 1) ? yy
		                                                      : zz;
		const double perp_coord1 = (index == 0) ? yy : (index == 1) ? zz
		                                                            : xx;
		const double perp_coord2 = (index == 0) ? zz : (index == 1) ? xx
		                                                            : yy;
		if (profile_type == "tanh") {
			// Tanh profile: s = 0.5*(1+tanh(stiff*(x - flamepos)))
			double stiffness2 = 100 * stiffness / global_param;
			double fac = stiffness2 * (coord - flamepos);
			s = 0.5 * (1.0 + std::tanh(fac));
		} else if (profile_type == "parabolic") {
			// Parabolic profile: s = 1 - (x-flamepos)^2/radius^2
			double radius2 = radius * (perp_global_param1 / 2.0);
			double distance = is_2D ? std::sqrt(std::pow(coord - flamepos, 2) + std::pow(perp_coord1 - (perp_global_param1 / 2.0), 2))
			                        : std::sqrt(std::pow(coord - flamepos, 2) + std::pow(perp_coord1 - (perp_global_param1 / 2.0), 2) + std::pow(perp_coord2 - (perp_global_param2 / 2.0), 2));
			s = (distance <= radius2) ? (1.0 - std::pow(distance / radius2, 2)) : 0.0;
		} else if (profile_type == "box") {
			// Box profile: s = 1 if x in [flamepos - box_length/2, flamepos + box_length/2], 0 otherwise
			double half_box_length = box_length / 2.0;
			double half_box_width = box_width / 2.0;
			double flame_min = flamepos - half_box_length;
			double flame_max = flamepos + half_box_length;
			double domain_mid1 = perp_global_param1 / 2.0;
			double perp_min1 = domain_mid1 - half_box_width;
			double perp_max1 = domain_mid1 + half_box_width;
			if (!is_2D) {
				double domain_mid2 = perp_global_param2 / 2.0;
				double perp_min2 = domain_mid2 - half_box_width;
				double perp_max2 = domain_mid2 + half_box_width;
				s = (coord >= flame_min && coord <= flame_max) && (perp_coord1 >= perp_min1 && perp_coord1 <= perp_max1) && (perp_coord2 >= perp_min2 && perp_coord2 <= perp_max2) ? 1.0 : 0.0;
			} else {
				s = (coord >= flame_min && coord <= flame_max) && (perp_coord1 >= perp_min1 && perp_coord1 <= perp_max1) ? 1.0 : 0.0;
			}
		} else if (profile_type == "sphere") {
			// Sphere profile: s = 1 if (x-flamepos)^2 + (y-flamepos)^2 + (z-flamepos)^2 <= radius^2, 0 otherwise
			double radius2 = radius * (perp_global_param1 / 2.0);
			double distance = is_2D ? std::sqrt(std::pow(coord - flamepos, 2) + std::pow(perp_coord1 - (perp_global_param1 / 2.0), 2))
			                        : std::sqrt(std::pow(coord - flamepos, 2) + std::pow(perp_coord1 - (perp_global_param1 / 2.0), 2) + std::pow(perp_coord2 - (perp_global_param2 / 2.0), 2));
			s = (distance <= radius2) ? 1.0 : 0.0;
		} else if (profile_type == "gaussian") {
			// Gaussian profile: s = exp(-(x-flamepos)^2/(2*sigma^2))
			double sigma = width * (perp_global_param1 / 2);
			double exponent = -0.5 * std::pow((coord - flamepos) / sigma, 2);
			s = std::exp(exponent);
		} else if (profile_type == "linear_gradient") {
			// Linear gradient profile: s = (x - start)/(end - start)
			double linear_start_global = linear_start * perp_global_param1;
			double linear_end_global = linear_end * perp_global_param1;
			if (linear_end_global <= linear_start_global) {
				throw std::invalid_argument("linear_end must be greater than linear_start");
			}
			if (is_2D) {
				if (perp_coord1 >= linear_start_global && perp_coord1 <= linear_end_global) {
					s = (perp_coord1 - linear_start_global) / (linear_end_global - linear_start_global);
				} else if (perp_coord1 < linear_start_global) {
					s = 0.0;
				} else {
					s = 1.0;
				}
			} else {
				double s1 = (perp_coord1 >= linear_start_global && perp_coord1 <= linear_end_global) ? (perp_coord1 - linear_start_global) / (linear_end_global - linear_start_global) : (perp_coord1 < linear_start_global ? 0.0 : 1.0);
				double s2 = (perp_coord2 >= linear_start_global && perp_coord2 <= linear_end_global) ? (perp_coord2 - linear_start_global) / (linear_end_global - linear_start_global) : (perp_coord2 < linear_start_global ? 0.0 : 1.0);
				s = std::max(s1, s2);
			}
		} else if (profile_type == "exponential_decay") {
			if (decay_rate <= 0) {
				throw std::invalid_argument("decay_rate must be a positive value");
			}
			double distance = is_2D ? std::abs(perp_coord1 - flamepos) : std::sqrt(std::pow(std::abs(perp_coord1 - flamepos), 2) + std::pow(std::abs(perp_coord2 - flamepos), 2));
			s = std::exp(-decay_rate * distance / 100);
		} else if (profile_type == "step_function") {
			int num_steps = step_positions.size();  // Moved num_steps initialization here
			s = 0.0;
			for (int i = 0; i < num_steps; ++i) {
				if (perp_coord1 <= (step_positions[i] * perp_global_param1)) {
					s = step_values[i];
					break;
				}
			}
		} else {
			throw std::invalid_argument("Invalid profile name");
		}
	}
	return s;  // Return s at the end of the function
}

// Common initialization function for scalar field
void Initial_field_slice::initialize_profile_scalar(Scalar_field& field, std::vector<double>& ini_val, Solid_field& solid, const std::string& filename, const Parallel_MPI& MPI_parallel) {
    if (MPI_parallel.is_master()) {
        return;
    }
    read_flame_properties(filename, MPI_parallel);
    for (int X = 0; X < MPI_parallel.dev_end[0]; X++) {
        for (int Y = 0; Y < MPI_parallel.dev_end[1]; Y++) {
            for (int Z = 0; Z < MPI_parallel.dev_end[2]; Z++) {
                double s = calculate_profile(profile_type, solid, filename, MPI_parallel, X, Y, Z);
                field[{X, Y, Z}] = s * ini_val.back() + (1.0 - s) * field[{X, Y, Z}];
            }
        }
    }
}

// Common initialization function for vector field
void Initial_field_slice::initialize_profile_vector(Vector_field& field, std::vector<std::vector<double>>& ini_val, Solid_field& solid, const std::string& filename, const Parallel_MPI& MPI_parallel) {
	if (MPI_parallel.is_master()) {
		return;
	}
	read_flame_properties(filename, MPI_parallel);
	for (int X = 0; X < MPI_parallel.dev_end[0]; X++) {
		for (int Y = 0; Y < MPI_parallel.dev_end[1]; Y++) {
			for (int Z = 0; Z < MPI_parallel.dev_end[2]; Z++) {
				double s = calculate_profile(profile_type, solid, filename, MPI_parallel, X, Y, Z);
				for (int i = 0; i < ini_val.size(); i++) {
					field[{X, Y, Z, i}] = s * ini_val.back()[i] + (1.0 - s) * field[{X, Y, Z, i}];
				}
			}
		}
	}
}

/*double signed_distance(const Box& box, const Vec3& point) {
    double sqr_dist = 0.0;
    double min_dist = std::numeric_limits<double>::max();
    bool inside = true;
    for (int d = 0; d < 3; ++d) {
        if (point[d] > box.start[d] && point[d] < box.end[d]) {
            const double dist = std::min(std::abs(point[d] - box.start[d]), std::abs(point[d] - box.end[d]));
            if (dist < min_dist) {
                min_dist = dist;
            }
        } else {
            inside = false;
            sqr_dist += std::min(sqr(point[d] - box.start[d]), sqr(point[d] - box.end[d]));
        }
    }

    return inside ? -min_dist : std::sqrt(sqr_dist);
}

double Initial_field_slice::get_scale(const Vec3& pos) const {
    const double s = -signed_distance(*this, pos) / boundary_thickness;
    switch (type) {
        case Type::POISEULLE:
            return sqr(0.5 + std::max(std::min(s, 0.5), -0.5));
        case Type::TANH:
            return 0.5 * (1.0 + std::tanh(s));
        default:
            ERROR_ABORT("Interpolation type " << TYPE_NAMES[static_cast<int>(type)] << " is not implemented.");
    }
    return 0.0;
}*/
/*
Initial_field_slice::Type Initial_field_slice::str_to_type(const std::string& type_str) {
    auto it = std::find_if(TYPE_NAMES.begin(), TYPE_NAMES.end(), [&](const char* name) {
        return type_str == name;
    });
    if (it == TYPE_NAMES.end()) {
        ERROR_ABORT("Unknown initial field interpolation type \"" << type_str << "\".");
    }

    return static_cast<Type>(std::distance(TYPE_NAMES.begin(), it));
}

std::istream& operator>>(std::istream& istream, Initial_field_slice& slice) {
    std::string type_str;
    istream >> type_str;
    slice.type = Initial_field_slice::str_to_type(type_str);

    for (int d = 0; d < 3; ++d) {
        istream >> slice.start[d];
        istream >> slice.end[d];
    }
    istream >> slice.boundary_thickness;

    return istream;
}

std::ostream& operator<<(std::ostream& ostream, const Initial_field_slice& slice) {
    ostream << "Slice extend: ";
    for (int d = 0; d < 3; ++d) {
        std::cout << "[" << slice.start[d] << ", " << slice.end[d] << "] ";
    }
    std::cout << "boundary thickness: " << slice.boundary_thickness << "\n";

    return ostream;
}
*/
/// ***************************************************** ///
/// STENCILS INITIALIZATION                               ///
/// ***************************************************** ///
void D3Q27(unsigned int& Dimension, unsigned int& Discrete_Velocity, std::vector<double>& w_alpha,
           std::vector<std::vector<int>>& c_alpha, std::vector<unsigned int>& alpha_bar, double& c_s2) {
	w_alpha.resize(27);
	alpha_bar.resize(27);
	c_alpha.resize(27);
	for (int alpha = 0; alpha < 27; alpha++) {
		c_alpha[alpha].resize(3);
	}

	c_s2 = 3;
	w_alpha = {8. / 27., 2. / 27., 2. / 27., 2. / 27., 2. / 27., 2. / 27., 2. / 27., 1. / 54., 1. / 54.,
	           1. / 54., 1. / 54., 1. / 54., 1. / 54., 1. / 54., 1. / 54., 1. / 54., 1. / 54., 1. / 54., 1. / 54.,
	           1. / 216., 1. / 216., 1. / 216., 1. / 216., 1. / 216., 1. / 216., 1. / 216., 1. / 216.};
	///   0  1  2  3  4  5  6  7  8  9 10  11  12  13  14  15  16  17  18
	alpha_bar = {0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15, 26, 25, 24, 23, 22, 21, 20, 19};
	c_alpha = {{0, 0, 0},      // 0
	           {1, 0, 0},      // 1
	           {-1, 0, 0},     // 2
	           {0, 1, 0},      // 3
	           {0, -1, 0},     // 4
	           {0, 0, 1},      // 5
	           {0, 0, -1},     // 6
	           {1, 1, 0},      // 7
	           {-1, 1, 0},     // 8
	           {1, -1, 0},     // 9
	           {-1, -1, 0},    // 10
	           {1, 0, 1},      // 11
	           {-1, 0, 1},     // 12
	           {1, 0, -1},     // 13
	           {-1, 0, -1},    // 14
	           {0, 1, 1},      // 15
	           {0, -1, 1},     // 16
	           {0, 1, -1},     // 17
	           {0, -1, -1},    // 18
	           {1, 1, 1},      // 19
	           {-1, 1, 1},     // 20
	           {1, -1, 1},     // 21
	           {-1, -1, 1},    // 22
	           {1, 1, -1},     // 23
	           {-1, 1, -1},    // 24
	           {1, -1, -1},    // 25
	           {-1, -1, -1}};  // 26
}
void D3Q19(unsigned int& Dimension, unsigned int& Discrete_Velocity, std::vector<double>& w_alpha,
           std::vector<std::vector<int>>& c_alpha, std::vector<unsigned int>& alpha_bar, double& c_s2) {
	w_alpha.resize(19);
	alpha_bar.resize(19);
	c_alpha.resize(19);
	for (int alpha = 0; alpha < 19; alpha++) {
		c_alpha[alpha].resize(3);
	}

	c_s2 = 3;
	w_alpha = {1. / 3., 1. / 18., 1. / 18., 1. / 18., 1. / 18., 1. / 18., 1. / 18., 1. / 36., 1. / 36.,
	           1. / 36., 1. / 36., 1. / 36., 1. / 36., 1. / 36., 1. / 36., 1. / 36., 1. / 36., 1. / 36., 1. / 36.};
	///   0  1  2  3  4  5  6  7  8  9 10  11  12  13  14  15  16  17  18
	alpha_bar = {0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15};
	c_alpha = {{0, 0, 0},     // 0
	           {1, 0, 0},     // 1
	           {-1, 0, 0},    // 2
	           {0, 1, 0},     // 3
	           {0, -1, 0},    // 4
	           {0, 0, 1},     // 5
	           {0, 0, -1},    // 6
	           {1, 1, 0},     // 7
	           {-1, 1, 0},    // 8
	           {1, -1, 0},    // 9
	           {-1, -1, 0},   // 10
	           {1, 0, 1},     // 11
	           {-1, 0, 1},    // 12
	           {1, 0, -1},    // 13
	           {-1, 0, -1},   // 14
	           {0, 1, 1},     // 15
	           {0, -1, 1},    // 16
	           {0, 1, -1},    // 17
	           {0, -1, -1}};  // 18
}
void D3Q15(unsigned int& Dimension, unsigned int& Discrete_Velocity, std::vector<double>& w_alpha,
           std::vector<std::vector<int>>& c_alpha, std::vector<unsigned int>& alpha_bar, double& c_s2) {
	w_alpha.resize(15);
	alpha_bar.resize(15);
	c_alpha.resize(15);
	for (int alpha = 0; alpha < 15; alpha++) {
		c_alpha[alpha].resize(3);
	}

	c_s2 = 3;
	w_alpha = {2. / 9., 1. / 9., 1. / 9., 1. / 9., 1. / 9., 1. / 9., 1. / 9., 1. / 72., 1. / 72.,
	           1. / 72., 1. / 72., 1. / 72., 1. / 72., 1. / 72., 1. / 72.};
	///   0  1  2  3  4  5  6  7  8  9 10  11  12  13  14
	alpha_bar = {0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13};
	c_alpha = {{0, 0, 0},     // 0
	           {1, 0, 0},     // 1
	           {-1, 0, 0},    // 2
	           {0, 1, 0},     // 3
	           {0, -1, 0},    // 4
	           {0, 0, 1},     // 5
	           {0, 0, -1},    // 6
	           {1, 1, 1},     // 7
	           {-1, -1, -1},  // 8
	           {1, 1, -1},    // 9
	           {-1, -1, 1},   // 10
	           {1, -1, 1},    // 11
	           {-1, 1, -1},   // 12
	           {-1, 1, 1},    // 13
	           {1, -1, -1}};  // 14
}
void D3Q7(unsigned int& Dimension, unsigned int& Discrete_Velocity, std::vector<double>& w_alpha,
          std::vector<std::vector<int>>& c_alpha, std::vector<unsigned int>& alpha_bar, double& c_s2) {
	w_alpha.resize(7);
	alpha_bar.resize(7);
	c_alpha.resize(7);

	for (int alpha = 0; alpha < 7; alpha++) {
		c_alpha[alpha].resize(3);
	}

	c_s2 = 4;
	w_alpha = {1. / 4., 1. / 8., 1. / 8., 1. / 8., 1. / 8., 1. / 8., 1. / 8.};
	alpha_bar = {0, 2, 1, 4, 3, 6, 5};
	c_alpha = {
		{0, 0, 0},   // 0
		{1, 0, 0},   // 1
		{-1, 0, 0},  // 2
		{0, 1, 0},   // 3
		{0, -1, 0},  // 4
		{0, 0, 1},   // 5
		{0, 0, -1}   // 6 };
	};
}
void D2Q9(unsigned int& Dimension, unsigned int& Discrete_Velocity, std::vector<double>& w_alpha,
          std::vector<std::vector<int>>& c_alpha, std::vector<unsigned int>& alpha_bar, double& c_s2) {
	w_alpha.resize(9);
	alpha_bar.resize(9);
	c_alpha.resize(9);
	for (int alpha = 0; alpha < 9; alpha++) {
		c_alpha[alpha].resize(3);
	}

	c_s2 = 3;
	w_alpha = {4. / 9., 1. / 9., 1. / 9., 1. / 9., 1. / 9., 1. / 36., 1. / 36., 1. / 36., 1. / 36.};
	alpha_bar = {0, 3, 4, 1, 2, 7, 8, 5, 6};
	c_alpha = {{0, 0, 0},
	           {1, 0, 0},
	           {0, 1, 0},
	           {-1, 0, 0},
	           {0, -1, 0},
	           {1, 1, 0},
	           {-1, 1, 0},
	           {-1, -1, 0},
	           {1, -1, 0}};
}
void D2Q5(unsigned int& Dimension, unsigned int& Discrete_Velocity, std::vector<double>& w_alpha,
          std::vector<std::vector<int>>& c_alpha, std::vector<unsigned int>& alpha_bar, double& c_s2) {
	w_alpha.resize(5);
	alpha_bar.resize(5);
	c_alpha.resize(5);
	for (int alpha = 0; alpha < 5; alpha++) {
		c_alpha[alpha].resize(3);
	}

	c_s2 = 3;
	w_alpha = {1. / 3., 1. / 6., 1. / 6., 1. / 6., 1. / 6.};
	alpha_bar = {0, 3, 4, 1, 2};
	c_alpha = {{0, 0, 0},
	           {1, 0, 0},
	           {0, 1, 0},
	           {-1, 0, 0},
	           {0, -1, 0}};
}
void D2Q4(unsigned int& Dimension, unsigned int& Discrete_Velocity, std::vector<double>& w_alpha,
          std::vector<std::vector<int>>& c_alpha, std::vector<unsigned int>& alpha_bar, double& c_s2) {
	w_alpha.resize(4);
	alpha_bar.resize(4);
	c_alpha.resize(4);
	for (int alpha = 0; alpha < 4; alpha++) {
		c_alpha[alpha].resize(4);
	}

	c_s2 = 2;
	w_alpha = {1. / 4., 1. / 4., 1. / 4., 1. / 4.};
	alpha_bar = {0, 3, 4, 1, 2};
	c_alpha = {{1, 0},
	           {0, 1},
	           {-1, 0},
	           {0, -1}};
}
Geometry::~Geometry() {
}

// ******************************************************************* //
DdQq::DdQq(){};
DdQq::~DdQq(){};

void DdQq::Initialize() {
	StencilFunctions["D2Q9"] = &D2Q9;
	StencilFunctions["D2Q4"] = &D2Q4;
	StencilFunctions["D2Q5"] = &D2Q5;
	StencilFunctions["D3Q19"] = &D3Q19;
	StencilFunctions["D3Q15"] = &D3Q15;
	StencilFunctions["D3Q27"] = &D3Q27;
	StencilFunctions["D3Q7"] = &D3Q7;
}
Stencil_Definition DdQq::get(unsigned dim, unsigned q) const {
	std::stringstream DQ;
	DQ << "D" << dim << "Q" << q;
	const std::string stencil = DQ.str();
	const StencilMap::const_iterator x = StencilFunctions.find(stencil);
	if (x == StencilFunctions.end()) {
		std::cerr << "[Error] The requested stencil " << stencil << " is not available.\n";
		return nullptr;
	}
	return x->second;
}

/// ***************************************************** ///
/// READ GEOMETRY FROM A SERIES OF STL FILES              ///
/// ***************************************************** ///
/// for each node inside a processor, this function defines which node belong to which stl.
/// check for which node lies in which stl.
void stl_import::Initialize_geometry(int N_x, int N_y, int N_z, Parallel_MPI* MPI_parallel) {
	MPI_parallel->center = Vec3{x_center, y_center, z_center};
	if (MPI_parallel->processor_id != MASTER) {
		int intersection_counter;
		stl::point O, D, temp, center, minimum, maximum;
		std::vector<stl::point> out;
		int temp1;
		center.x = x_center;
		center.y = y_center;
		center.z = z_center;
		std::vector<int> TList;
		double max_temp, min_temp;

		domain = Tensor<int, 3>::zeros({
			static_cast<Index>(MPI_parallel->dev_end[0]),
			static_cast<Index>(MPI_parallel->dev_end[1]),
			static_cast<Index>(MPI_parallel->dev_end[2]),
		});

		for (int file_index = 0; file_index < Source_count; file_index++) {
			/// *********************************************************************************** ///
			///          READ STL FILE, PUT IT INTO INFO                                            ///
			/// *********************************************************************************** ///
			m_stl_data.emplace_back(stl::parse_stl(Geo_filename[file_index]));
			const std::vector<stl::triangle>& triangles = m_stl_data.back().triangles;
			/// *********************************************************************************** ///
			///          GET DOMAIN MAX                                                             ///
			/// *********************************************************************************** ///
			maximum.x = std::max(std::max(triangles[0].v1.x, triangles[0].v2.x), triangles[0].v3.x);
			maximum.y = std::max(std::max(triangles[0].v1.y, triangles[0].v2.y), triangles[0].v3.y);
			maximum.z = std::max(std::max(triangles[0].v1.z, triangles[0].v2.z), triangles[0].v3.z);
			/// *********************************************************************************** ///
			///          GET DOMAIN MIN                                                             ///
			/// *********************************************************************************** ///
			minimum.x = std::min(std::min(triangles[0].v1.x, triangles[0].v2.x), triangles[0].v3.x);
			minimum.y = std::min(std::min(triangles[0].v1.y, triangles[0].v2.y), triangles[0].v3.y);
			minimum.z = std::min(std::min(triangles[0].v1.z, triangles[0].v2.z), triangles[0].v3.z);

			for (auto l = 0; l < triangles.size(); l++) {
				min_temp = std::min(std::min(triangles[l].v1.x, triangles[l].v2.x), triangles[l].v3.x);
				if (min_temp < minimum.x) {
					minimum.x = min_temp;
				}
				min_temp = std::min(std::min(triangles[l].v1.y, triangles[l].v2.y), triangles[l].v3.y);
				if (min_temp < minimum.y) {
					minimum.y = min_temp;
				}
				min_temp = std::min(std::min(triangles[l].v1.z, triangles[l].v2.z), triangles[l].v3.z);
				if (min_temp < minimum.z) {
					minimum.z = min_temp;
				}
				max_temp = std::max(std::max(triangles[l].v1.x, triangles[l].v2.x), triangles[l].v3.x);
				if (maximum.x < max_temp) {
					maximum.x = max_temp;
				}
				max_temp = std::max(std::max(triangles[l].v1.y, triangles[l].v2.y), triangles[l].v3.y);
				if (maximum.y < max_temp) {
					maximum.y = max_temp;
				}
				max_temp = std::max(std::max(triangles[l].v1.z, triangles[l].v2.z), triangles[l].v3.z);
				if (maximum.z < max_temp) {
					maximum.z = max_temp;
				}
			}

			for (int i = 0; i < MPI_parallel->dev_end[0]; i++) {
				for (int j = 0; j < MPI_parallel->dev_end[1]; j++) {
					for (int l = 0; l < triangles.size(); l++) {
						O.x = global_parameters.D_x * ((i - MPI_parallel->start_XYZ2[0] + MPI_parallel->start_XYZ[0] + global_parameters.Nx) % global_parameters.Nx) + center.x;
						O.y = global_parameters.D_x * ((j - MPI_parallel->start_XYZ2[1] + MPI_parallel->start_XYZ[1] + global_parameters.Ny) % global_parameters.Ny) + center.y;
						O.z = maximum.z + global_parameters.D_x;
						D.x = 0;
						D.y = 0;
						D.z = (minimum.z - global_parameters.D_x - O.z);
						intersection_counter = triangle_intersection(triangles[l].v1, triangles[l].v2, triangles[l].v3, O, D, &temp);
						if (intersection_counter != 0 && i > -1 && j > -1) {
							TList.push_back(l);
						}
					}
					for (int k = 0; k < MPI_parallel->dev_end[2]; k++) {
						out.resize(0);
						temp1 = TList.size();
						for (int l = 0; l < temp1; l++) {
							O.x = global_parameters.D_x * ((i - MPI_parallel->start_XYZ2[0] + MPI_parallel->start_XYZ[0] + global_parameters.Nx) % global_parameters.Nx) + center.x;
							O.y = global_parameters.D_x * ((j - MPI_parallel->start_XYZ2[1] + MPI_parallel->start_XYZ[1] + global_parameters.Ny) % global_parameters.Ny) + center.y;
							O.z = global_parameters.D_x * ((k - MPI_parallel->start_XYZ2[2] + MPI_parallel->start_XYZ[2] + global_parameters.Nz) % global_parameters.Nz) + center.z;
							D.x = 0;  //(center.x - dx - O.x);
							D.y = 0;  //(center.y - dx - O.y);
							D.z = (minimum.z - global_parameters.D_x - O.z);
							intersection_counter = triangle_intersection(triangles[TList[l]].v1, triangles[TList[l]].v2, triangles[TList[l]].v3, O, D, &temp);
							if (intersection_counter == 1) {
								out.push_back(temp);
							}
						}
						check_for_duplicat(out);
						if ((out.size() % 2) == 1) {
							domain[{i, j, k}] = file_index + 1;
						}
					}
					TList.clear();
				}
			}
		}
	}
}

/// ***************************************************** ///
/// CHECK INTERSECTION BETWEEN RAY AND TRIANGLE           ///
/// ***************************************************** ///
int stl_import::triangle_intersection(const stl::point& V1,  // Triangle vertices
                                      const stl::point& V2,
                                      const stl::point& V3,
                                      const stl::point& O,  // Ray origin
                                      const stl::point& D,  // Ray direction
                                      stl::point* out) {
	stl::point e1, e2;  // Edge1, Edge2
	stl::point P, Q, T;
	double det, inv_det, u, v;
	double t;
	// Find vectors for two edges sharing V1
	stl::SUB_stl(e1, V2, V1);
	stl::SUB_stl(e2, V3, V1);
	// Begin calculating determinant - also used to calculate u parameter
	stl::CROSS_stl(P, D, e2);
	// if determinant is near zero, ray lies in plane of triangle or ray is parallel to plane of triangle
	det = stl::DOT_stl(e1, P);
	// NOT CULLING
	if (det > -EPSILON && det < EPSILON) return 0;
	inv_det = 1.f / det;

	// calculate distance from V1 to ray origin
	stl::SUB_stl(T, O, V1);

	// Calculate u parameter and test bound
	u = stl::DOT_stl(T, P) * inv_det;
	// The intersection lies outside of the triangle
	if (u < 0.f || u > 1.f) return 0;

	// Prepare to test v parameter
	stl::CROSS_stl(Q, T, e1);

	// Calculate V parameter and test bound
	v = stl::DOT_stl(D, Q) * inv_det;
	// The intersection lies outside of the triangle
	if (v < 0.f || u + v > 1.f) return 0;

	t = stl::DOT_stl(e2, Q) * inv_det;
	if (t > EPSILON) {  // ray intersection
		out->x = V1.x + v * e2.x + u * e1.x;
		out->y = V1.y + v * e2.y + u * e1.y;
		out->z = V1.z + v * e2.z + u * e1.z;
		return 1;
	}

	return 0;
}

double stl_import::triangle_shortest_distance(const stl::point& V1, const stl::point& V2, const stl::point& V3,
                                              const stl::point& I,
                                              const stl::point& n,
                                              stl::point* out) {
	double distance;
	double distance_temp;
	stl::point out_temp, temp;
	/* check distance from triangle sides */
	stl::point L, W1;
	double t;
	/* vector from V1->V2 */
	stl::SUB_stl(L, V2, V1);
	/* vector from V1->I */
	stl::SUB_stl(W1, I, V1);
	t = stl::DOT_stl(W1, L) / stl::DOT_stl(L, L);
	t = MIN(MAX(t, 0.), 1.);
	/* If 0<t<1, the perpendicular line from I to V1-V2 falls between V1 and V2 */
	/* If t=0, the perpendicular falls on V1 */
	/* If t=1, the perpendicular falls on V2 */
	temp.x = V1.x + L.x * t;
	temp.y = V1.y + L.y * t;
	temp.z = V1.z + L.z * t;
	distance_temp = DISTANCE(temp, I);
	out_temp.x = temp.x;
	out_temp.y = temp.y;
	out_temp.z = temp.z;
	distance = distance_temp;
	/* check second side */
	/* vector from V1-V2 */
	stl::SUB_stl(L, V3, V1);
	/* vector from V1-I */
	stl::SUB_stl(W1, I, V1);
	t = stl::DOT_stl(W1, L) / stl::DOT_stl(L, L);
	t = MIN(MAX(t, 0.), 1.);
	temp.x = V1.x + L.x * t;
	temp.y = V1.y + L.y * t;
	temp.z = V1.z + L.z * t;
	distance_temp = DISTANCE(temp, I);
	if (distance_temp < distance) {
		out_temp.x = temp.x;
		out_temp.y = temp.y;
		out_temp.z = temp.z;
		distance = distance_temp;
	}
	/* check third side */
	/* vector from V1-V2 */
	stl::SUB_stl(L, V3, V2);
	/* vector from V1-I */
	stl::SUB_stl(W1, I, V2);
	t = stl::DOT_stl(W1, L) / stl::DOT_stl(L, L);
	t = MIN(MAX(t, 0.), 1.);
	temp.x = V2.x + L.x * t;
	temp.y = V2.y + L.y * t;
	temp.z = V2.z + L.z * t;
	distance_temp = DISTANCE(temp, I);
	if (distance_temp < distance) {
		out_temp.x = temp.x;
		out_temp.y = temp.y;
		out_temp.z = temp.z;
		distance = distance_temp;
	}
	/* check if point can intersect triangle in normal direction */
	stl::point stl_normal;
	stl_normal.x = n.x;
	stl_normal.y = n.y;
	stl_normal.z = n.z;
	int intersection = stl_import::triangle_intersection(V1, V2, V3, I, stl_normal, &temp);
	if (intersection == 0) {
		stl_normal.x = -n.x;
		stl_normal.y = -n.y;
		stl_normal.z = -n.z;
		intersection = stl_import::triangle_intersection(V1, V2, V3, I, stl_normal, &temp);
	}
	if (intersection == 1) {
		distance_temp = DISTANCE(temp, I);
		out_temp.x = temp.x;
		out_temp.y = temp.y;
		out_temp.z = temp.z;
		distance = distance_temp;
	}
	out->x = out_temp.x;
	out->y = out_temp.y;
	out->z = out_temp.z;

	return distance;
}

/// ***************************************************** ///
/// CHECK FOR DUPLICATES IN LIST OF INTERSECTIONS         ///
/// ***************************************************** ///
void stl_import::check_for_duplicat(std::vector<stl::point>& intersections) {
	double distance;
	for (int i = 0; i < intersections.size(); i++) {
		for (int j = i + 1; j < intersections.size(); j++) {
			distance = sqrt(sqr(intersections[j].x - intersections[i].x) + sqr(intersections[j].y - intersections[i].y) + sqr(intersections[j].z - intersections[i].z));
			if (distance < EPSILON) {
				intersections.erase(intersections.begin() + j);
				/// j--;
			}
		}
	}
}

Index_vec3 stl_import::compute_simple_normal(const Index_vec3& idx, const Solid_field& is_solid, int out_zone) const {
	std::array<int, 3> start;
	std::array<int, 3> end;
	for (int d = 0; d < 3; ++d) {
		start[d] = std::max(0, idx[d] - 1);
		end[d] = std::min(static_cast<int>(domain.sizes()[d]), idx[d] + 2);
	}

	Index_vec3 normal = {};

	const int solid_origin = is_solid[idx];
	assert(solid_origin == 1 || solid_origin == -1);

	if (out_zone >= 0) {
		// for corners we take the diagonal to get away from the boundary
		for (int x = -1; x < 2; ++x) {
			for (int y = -1; y < 2; ++y) {
				for (int z = -1; z < 2; ++z) {
					if (domain[{idx[0] - x, idx[1] - y, idx[2] - z}] == out_zone
					    && is_solid[{idx[0] + x, idx[1] + y, idx[2] + z}] == solid_origin) {
						normal[0] += x;
						normal[1] += y;
						normal[2] += z;
					}
				}
			}
		}
	} else {
		for (int x = -1; x < 2; ++x) {
			for (int y = -1; y < 2; ++y) {
				for (int z = -1; z < 2; ++z) {
					if (is_solid[{idx[0] - x, idx[1] - y, idx[2] - z}] != solid_origin
					    && is_solid[{idx[0] + x, idx[1] + y, idx[2] + z}] == solid_origin) {
						normal[0] += x;
						normal[1] += y;
						normal[2] += z;
					}
				}
			}
		}
	}

	// rescale to just one step on the lattice
	for (int d = 0; d < 3; ++d) {
		normal[d] = (normal[d] > 0) ? 1 : ((normal[d] < 0) ? -1 : 0);
	}

	return normal;
}

Vec3 stl_import::compute_normal(const Index_vec3& idx, const Parallel_MPI& MPI_parallel, int in_zone, int out_zone) const {
	Vec3 normal = {};
	stl::point O, intersection_coordinate, direction;
	// get coordinates of boundary node in stl space
	MPI_parallel.get_coordinates(idx[0], idx[1], idx[2],
	                             x_center, y_center, z_center,
	                             O.x, O.y, O.z);
	double minimum_distance = std::numeric_limits<double>::max();

	auto check_stl = [&](int zone) {
		for (const stl::triangle& triangle : get_stl_data()[zone - 1].triangles) {
			direction.x = triangle.normal.x;
			direction.y = triangle.normal.y;
			direction.z = triangle.normal.z;
			const double distance = triangle_shortest_distance(triangle.v1, triangle.v2, triangle.v3, O, direction, &intersection_coordinate);
			if (distance < minimum_distance /*&& (distance / global_parameters.D_x) <= sqrt(global_parameters.D)*/) {
				const double normX = (intersection_coordinate.x - O.x) / global_parameters.D_x;
				const double normY = (intersection_coordinate.y - O.y) / global_parameters.D_x;
				const double normZ = (intersection_coordinate.z - O.z) / global_parameters.D_x;
				minimum_distance = distance;
				normal = {normX, normY, normZ};
			}
		}
	};
	// Go through stl file of the solid zone
	check_stl(out_zone);
	check_stl(in_zone);

	return normal;
}

const std::vector<Boundary_node>& stl_import::get_boundary_nodes(int in_zone, int out_zone) const {
	auto it = m_boundary_nodes.find({in_zone - 1, out_zone - 1});
	if (it != m_boundary_nodes.end()) {
		return it->second;
	}
	return m_dummy_boundary;
}

const std::vector<stl::stl_data>& stl_import::get_stl_data() const {
	return m_stl_data;
}

/// ***************************************************** ///
/// DETECT SOLID BOUNDARY NODES                           ///
/// ***************************************************** ///
void stl_import::initialize_boundary_nodes(const Solid_field& is_solid, const Parallel_MPI& MPI_parallel) {
	if (MPI_parallel.is_master()) {
		return;
	}

	auto insert_as_boundary_node = [&](const Index_vec3& idx) {
		// we are looking for the solid side of the boundary
		// zero indicates that the domain is undefined
		if (is_solid[idx] == FALSE || domain[idx] <= 0) {
			return;
		}

		// non solid node in 1-ring neighborhood?
		std::array<int, 3> start;
		std::array<int, 3> end;
		for (int d = 0; d < 3; ++d) {
			start[d] = std::max(0, idx[d] - 1);
			end[d] = std::min(static_cast<int>(MPI_parallel.dev_end[d]), idx[d] + 2);
		}

		for (int x = start[0]; x < end[0]; ++x) {
			for (int y = start[1]; y < end[1]; ++y) {
				for (int z = start[2]; z < end[2]; ++z) {
					if (is_solid[{x, y, z}] == FALSE && domain[{x, y, z}] > 0) {
						m_boundary_nodes[{domain[{x, y, z}] - 1, domain[idx] - 1}].push_back({idx, is_solid.flat_index(idx)});
						return;
					}
				}
			}
		}
	};

	for (int i = 0; i < MPI_parallel.dev_end[0]; i++) {
		for (int j = 0; j < MPI_parallel.dev_end[1]; j++) {
			for (int k = 0; k < MPI_parallel.dev_end[2]; k++) {
				insert_as_boundary_node({i, j, k});
			}
		}
	}
	initialize_boundary_normals(MPI_parallel);
}

void stl_import::initialize_boundary_normals(const Parallel_MPI& MPI_parallel) {
	for (auto& boundary_zones : m_boundary_nodes) {
		// values in m_boundary_nodes are already 0-based but compute_normal expects the index
		const size_t in_zone = boundary_zones.first[0] + 1;
		const size_t out_zone = boundary_zones.first[1] + 1;

		for (auto& node : boundary_zones.second) {
			node.normal = compute_normal(node.idx, MPI_parallel, in_zone, out_zone);
		}
	}
}

namespace debug {
bool undefined_number(double x) {
	switch (std::fpclassify(x)) {
		case FP_INFINITE: return true;
		case FP_NAN: return true;
		case FP_NORMAL: return false;
		case FP_SUBNORMAL: return false;
		case FP_ZERO: return false;
		default: return false;
	}
}
}  // namespace debug

// before C++17 static constexpr variables still need to be defined
constexpr int Stencil<3, 27>::dim;
constexpr int Stencil<3, 27>::discrete_velocity;
constexpr std::array<double, 27> Stencil<3, 27>::w_alpha;
constexpr std::array<std::array<int, 3>, 27> Stencil<3, 27>::c_alpha;

constexpr int Stencil<2, 9>::dim;
constexpr int Stencil<2, 9>::discrete_velocity;
constexpr std::array<double, 9> Stencil<2, 9>::w_alpha;
constexpr std::array<std::array<int, 3>, 9> Stencil<2, 9>::c_alpha;