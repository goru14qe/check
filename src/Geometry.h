#ifndef GEOMETRY_H
#define GEOMETRY_H

#include "ALBORZ_GlobalVariables.h"
#include "ALBORZ_SETTINGS.h"
#include "Parallel.h"
#include <cassert>
#include <vector>
#include <functional>
#include <map>
#include <cmath>
#include <unordered_map>
#include "Thermal_solver.h"
#include "Species_solver.h"
#include "Phase_Field.h"
#include "Fluid_read_write.h"
#include "Vec.h"
#define EPSILON 1e-20

typedef double (*grid_gen)(int X, int Nx);
typedef void (*Stencil_Definition)(unsigned int& Dimension, unsigned int& Discrete_Velocity, std::vector<double>& w_alpha,
                                   std::vector<std::vector<int>>& c_alpha, std::vector<unsigned int>& alpha_bar, double& c_s2);
typedef std::map<std::string, Stencil_Definition> StencilMap;

class Flow_solver;
class Temperature_Field;

namespace debug {
bool undefined_number(double);
}

namespace stl {
struct point {
	double x;
	double y;
	double z;

	point()
		: x(0), y(0), z(0) {}
	point(double xp, double yp, double zp)
		: x(xp), y(yp), z(zp) {}
};
struct triangle {
	point normal;
	point v1;
	point v2;
	point v3;
	triangle(point normalp, point v1p, point v2p, point v3p)
		: normal(normalp), v1(v1p), v2(v2p), v3(v3p) {}
};
std::ostream& operator<<(std::ostream& out, const triangle& t);
struct stl_data {
	std::string name;
	std::vector<triangle> triangles;

	stl_data(std::string namep)
		: name(std::move(namep)) {}
};
stl_data parse_stl(const std::string& stl_path);
double DOT_stl(const point& A, const point& B);
void SUB_stl(point& C, const point& A, const point& B);
void CROSS_stl(point& C, const point& A, const point& B);
double DISTANCE(const point& A, const point& B);
}  // namespace stl

namespace FD {
double WENO5NONCONS(double u, double f1, double f2, double f3, double f4, double f5, double f6, double f7);
double WENO3NONCONS(double u, double f1, double f2, double f3, double f4, double f5);
double UPWIND1NONCONS(double u, double f1, double f2, double f3);
double UPWIND2NONCONS(double u, double f1, double f2, double f3, double f4, double f5);
double UPWIND3NONCONS(double u, double f1, double f2, double f3, double f4, double f5);
double CENTRALNONCONS(double u, double f1, double f2);
double CENTRAL4NONCONS(double u, double f1, double f2, double f3, double f4, double f5);
double CENTRAL2FLUX(double f1, double f2, double D1, double D2);
double CENTRAL4FLUX(double f1, double f2, double f3, double f4, double D1, double D2, double D3, double D4);
void smoothen_fields(Flow_solver* Flow, Thermal_solver* Thermal, Species_solver* Species, Parallel_MPI* MPI_parallel, unsigned int Nt, double D);
}  // namespace FD
///   Bitmap file reader
class Geometry {
public:
	int** img;
	int w, h;
	double *Grid_X, *Grid_Y;
	bool flag = 0;
	void Get_matrix_from_bmp(const std::string&);
	void Init_grid(int, int, grid_gen, grid_gen);
	Geometry();
	~Geometry();

private:
};
///   Mapping Stencil initialization functions
class DdQq {
public:
	DdQq();
	~DdQq();

	Stencil_Definition get(unsigned dim, unsigned q) const;
	StencilMap StencilFunctions;
	void Initialize();

private:
};

struct Boundary_node {
	Index_vec3 idx;
	Flat_index flat_idx;
	Vec3 normal;
};

///   stl file reader
class stl_import {
public:
	void Initialize_geometry(int, int, int, Parallel_MPI*);
	void initialize_boundary_nodes(const Solid_field& is_solid, const Parallel_MPI& MPI_parallel);
	static int triangle_intersection(const stl::point& V1,  // Triangle vertices
	                                 const stl::point& V2,
	                                 const stl::point& V3,
	                                 const stl::point& O,  // Ray origin
	                                 const stl::point& D,  // Ray direction
	                                 stl::point* out);
	static double triangle_shortest_distance(const stl::point& V1,  // Triangle vertices
	                                         const stl::point& V2,
	                                         const stl::point& V3,
	                                         const stl::point& I,  // Origin
	                                         const stl::point& n,  // Normal
	                                         stl::point* out);
	static void check_for_duplicat(std::vector<stl::point>& intersections);

	// Compute a normal pointing away from out_zone assuming straight boundaries.
	// If out_zone < 0, then all nodes across a boundary are considered instead.
	Index_vec3 compute_simple_normal(const Index_vec3& idx, const Solid_field& is_solid, int out_zone) const;

	// Compute a normal pointing towards the surface of the stl out_zone or in_zone, which ever is closer.
	Vec3 compute_normal(const Index_vec3& idx, const Parallel_MPI& MPI_parallel, int in_zone, int out_zone) const;

	// Get boundary nodes belonging to a specific zone, defined by the Source_count geometry files.
	const std::vector<Boundary_node>& get_boundary_nodes(int in_zone, int out_zone) const;
	const std::vector<stl::stl_data>& get_stl_data() const;

	bool flag = 0;
	unsigned int Source_count;
	std::vector<std::string> Geo_filename;
	double x_center, y_center, z_center;
	double units;
	Tensor<int, 3> domain;

private:
	void initialize_boundary_normals(const Parallel_MPI& MPI_parallel);
	std::unordered_map<Base_index_vec<2>, std::vector<Boundary_node>> m_boundary_nodes;
	std::vector<stl::stl_data> m_stl_data;
	std::vector<Boundary_node> m_dummy_boundary;
};

///   Miscellaneous functions
double Lin_Int_1d(double x1, double x2, double f1, double f2, double x);
double Lin_Int(double x1, double x2, double y1, double y2, double f1, double f2, double f3, double f4, double x, double y);
double Quad_Int(double x0, double x1, double x2, double y0, double y1, double y2, double x);

class Initial_field_slice {
private:
public:
	int index;
	bool is_2D;
	double xflamepos;
	std::string flow_dir, profile_type;
	double stiffness, radius, box_length, box_width, width, linear_start, linear_end, decay_rate;
	int num_steps;
	std::vector<double> step_positions;
	std::vector<double> step_values;
	void read_flame_properties(const std::string& filename, const Parallel_MPI& MPI_parallel);
	double calculate_profile(const std::string& profile_type, Solid_field& solid, const std::string&, const Parallel_MPI&, int X, int Y, int Z);
	void initialize_profile_scalar(Scalar_field& field, std::vector<double>& ini_val, Solid_field& solid, const std::string& filename, const Parallel_MPI& MPI_parallel);
	void initialize_profile_vector(Vector_field& field, std::vector<std::vector<double>>& ini_val, Solid_field& solid, const std::string& filename, const Parallel_MPI& MPI_parallel);
};
/*struct Box {
    Vec3 start;
    Vec3 end;
    };
    : public Box {
    enum struct Type {
        POISEULLE,
        TANH,
        COUNT
    };
    Type type;
    int index;
    double boundary_thickness;
    // get scale from [0,1] for interpolation based on the distance of pos to the box
    double get_scale(const Vec3& pos) const;
    static Type str_to_type(const std::string& str);
*/
/*
// reads the size of the Initial_field_slice from the config
std::istream& operator>>(std::istream& istream, Initial_field_slice& slice);
std::ostream& operator<<(std::ostream& ostream, const Initial_field_slice& slice);
double signed_distance(const Box& box, const Vec3& point);
*/

///   Stencil Initializers
void D2Q9(unsigned int& Dimension, unsigned int& Discrete_Velocity, std::vector<double>& w_alpha, std::vector<std::vector<int>>& c_alpha,
          std::vector<unsigned int>& alpha_bar, double& c_s2);
void D2Q5(unsigned int& Dimension, unsigned int& Discrete_Velocity, std::vector<double>& w_alpha, std::vector<std::vector<int>>& c_alpha,
          std::vector<unsigned int>& alpha_bar, double& c_s2);
void D2Q4(unsigned int& Dimension, unsigned int& Discrete_Velocity, std::vector<double>& w_alpha, std::vector<std::vector<int>>& c_alpha,
          std::vector<unsigned int>& alpha_bar, double& c_s2);
void D3Q19(unsigned int& Dimension, unsigned int& Discrete_Velocity, std::vector<double>& w_alpha, std::vector<std::vector<int>>& c_alpha,
           std::vector<unsigned int>& alpha_bar, double& c_s2);
void D3Q15(unsigned int& Dimension, unsigned int& Discrete_Velocity, std::vector<double>& w_alpha, std::vector<std::vector<int>>& c_alpha,
           std::vector<unsigned int>& alpha_bar, double& c_s2);
void D3Q27(unsigned int& Dimension, unsigned int& Discrete_Velocity, std::vector<double>& w_alpha, std::vector<std::vector<int>>& c_alpha,
           std::vector<unsigned int>& alpha_bar, double& c_s2);
void D3Q7(unsigned int& Dimension, unsigned int& Discrete_Velocity, std::vector<double>& w_alpha, std::vector<std::vector<int>>& c_alpha,
          std::vector<unsigned int>& alpha_bar, double& c_s2);

/// compile time stencils
template <int D, int Q>
struct Stencil;

template <>
struct Stencil<3, 27> {
	static constexpr int dim = 3;
	static constexpr int discrete_velocity = 27;
	static constexpr std::array<double, 27> w_alpha = {8. / 27., 2. / 27., 2. / 27., 2. / 27., 2. / 27., 2. / 27., 2. / 27., 1. / 54., 1. / 54.,
	                                                   1. / 54., 1. / 54., 1. / 54., 1. / 54., 1. / 54., 1. / 54., 1. / 54., 1. / 54., 1. / 54., 1. / 54.,
	                                                   1. / 216., 1. / 216., 1. / 216., 1. / 216., 1. / 216., 1. / 216., 1. / 216., 1. / 216.};
	static constexpr std::array<std::array<int, 3>, 27> c_alpha = {{{0, 0, 0},       // 0
	                                                                {1, 0, 0},       // 1
	                                                                {-1, 0, 0},      // 2
	                                                                {0, 1, 0},       // 3
	                                                                {0, -1, 0},      // 4
	                                                                {0, 0, 1},       // 5
	                                                                {0, 0, -1},      // 6
	                                                                {1, 1, 0},       // 7
	                                                                {-1, 1, 0},      // 8
	                                                                {1, -1, 0},      // 9
	                                                                {-1, -1, 0},     // 10
	                                                                {1, 0, 1},       // 11
	                                                                {-1, 0, 1},      // 12
	                                                                {1, 0, -1},      // 13
	                                                                {-1, 0, -1},     // 14
	                                                                {0, 1, 1},       // 15
	                                                                {0, -1, 1},      // 16
	                                                                {0, 1, -1},      // 17
	                                                                {0, -1, -1},     // 18
	                                                                {1, 1, 1},       // 19
	                                                                {-1, 1, 1},      // 20
	                                                                {1, -1, 1},      // 21
	                                                                {-1, -1, 1},     // 22
	                                                                {1, 1, -1},      // 23
	                                                                {-1, 1, -1},     // 24
	                                                                {1, -1, -1},     // 25
	                                                                {-1, -1, -1}}};  // 26
};

template <>
struct Stencil<2, 9> {
	static constexpr int dim = 2;
	static constexpr int discrete_velocity = 9;
	static constexpr std::array<double, 9> w_alpha = {4. / 9., 1. / 9., 1. / 9., 1. / 9., 1. / 9., 1. / 36., 1. / 36., 1. / 36., 1. / 36.};
	static constexpr std::array<std::array<int, 3>, 9> c_alpha = {{{0, 0, 0},
	                                                               {1, 0, 0},
	                                                               {0, 1, 0},
	                                                               {-1, 0, 0},
	                                                               {0, -1, 0},
	                                                               {1, 1, 0},
	                                                               {-1, 1, 0},
	                                                               {-1, -1, 0},
	                                                               {1, -1, 0}}};
};

#endif  // GEOMETRY_H