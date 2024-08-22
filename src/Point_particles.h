#include <vector>
#include <string>

class Parallel_MPI;
class stl_import;
class Flow_solver;

struct Tracer_point_particle {
	Tracer_point_particle(double ID, double xx, double yy, double zz, double ux, double uy, double uz);

	void update_position();

	double dummy_index;
	double X, Y, Z;
	double U, V, W;
};

class Point_particles {
public:
	std::vector<Tracer_point_particle> particles;

	void initialize(const std::string& filename, const stl_import& geo_stl, const Parallel_MPI& parallel_MPI);
	void update_positions(const Flow_solver& Flow, const Parallel_MPI& parallel_MPI);
	void data_exchange(const stl_import& geo_stl, const Parallel_MPI& parallel_MPI);
	void write_vtk(int time, int t_vtk, const stl_import& geo_stl, const Parallel_MPI& parallel_MPI);
	void filter_points(double, const Flow_solver& Flow, const Parallel_MPI& parallel_MPI);
};