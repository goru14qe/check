#include "Parallel.h"
#include "Geometry.h"
#include "Flow_solver.h"
#include "Point_particles.h"

using namespace std;

namespace interpolation{
	template<typename Interval>
	void order_1(const Interval&, const Interval&, double, double& out){
		out = 0.0;
	}
}

Tracer_point_particle::Tracer_point_particle(double ID, double xx, double yy, double zz, double ux, double uy, double uz)
	: dummy_index{ID}, X{xx}, Y{yy}, Z{zz}, U{ux}, V{uy}, W{uz} {
}

void Tracer_point_particle::update_position() {
	X += U;
	Y += V;
	Z += W;
}

void Point_particles::initialize(const std::string& filename, const stl_import& geo_stl, const Parallel_MPI& parallel_MPI) {
	constexpr int column_width = 40;
	if (parallel_MPI.processor_id != MASTER) {
		/* Open input file */
		string input_filename(filename);
		ifstream input_file(filename + ".dat", ios::binary);
		/* find header of the Flow field BC paragraph */
		find_line_after_header(input_file, "c\tPoint Particles");
		/* skip comment lines (starting with #) */
		find_line_after_comment(input_file);
		/* read number of volumes to be seeded with tracers */
		int number_of_zones;
		input_file >> number_of_zones;
		vector<int> index(number_of_zones, 0), zone(number_of_zones, 0);
		vector<double> NPP(number_of_zones, 0);
		/* read indexes, volumes and density */
		for (int i = 0; i < number_of_zones; i++) {
			/* skip comment lines (starting with #) */
			find_line_after_comment(input_file);
			input_file >> index[i] >> zone[i] >> NPP[i];
		}
		if (parallel_MPI.processor_id == MASTER + 1) {
			std::cout << "Point Particles\n";
			std::cout << "=====================\n";
			std::cout << setw(column_width) << left << "Number of zones : " << number_of_zones << "\n";
			for (int i = 0; i < number_of_zones; i++) {
				std::cout << setw(column_width) << left << "Index = " << index[i] << "\n";
				std::cout << setw(column_width) << left << "Volume = " << zone[i] << "\n";
				std::cout << setw(column_width) << left << "distance = " << NPP[i] << std::endl;
			}
		}
		for (int i = 0; i < number_of_zones; i++) {
			/* The seeding starts at (0,0,0) based on absolute coordinates */
			double Xstart = ceil(parallel_MPI.start_XYZ[0] / (double)NPP[i]) * NPP[i] - parallel_MPI.start_XYZ[0] + parallel_MPI.start_XYZ2[0];
			double Ystart = ceil(parallel_MPI.start_XYZ[1] / (double)NPP[i]) * NPP[i] - parallel_MPI.start_XYZ[1] + parallel_MPI.start_XYZ2[1];
			double Zstart = ceil(parallel_MPI.start_XYZ[2] / (double)NPP[i]) * NPP[i] - parallel_MPI.start_XYZ[2] + parallel_MPI.start_XYZ2[2];
			for (double X = Xstart; X <= parallel_MPI.end_XYZ2[0]; X += NPP[i]) {
				for (double Y = Ystart; Y <= parallel_MPI.end_XYZ2[1]; Y += NPP[i]) {
					for (double Z = Zstart; Z <= parallel_MPI.end_XYZ2[2]; Z += NPP[i]) {
						if (geo_stl.domain[{int(X),int(Y),int(Z)}] == zone[i]) {
							Tracer_point_particle temp_particle{double(index[i]), double(X), double(Y), double(Z), 0, 0, 0};
							particles.push_back(temp_particle);
						}
					}
				}
			}
		}
	}
}
void Point_particles::update_positions(const Flow_solver& Flow, const Parallel_MPI& parallel_MPI) {
	if (parallel_MPI.processor_id != MASTER) {
		for (int i = 0; i < particles.size(); i++) {
			/* Get the fluid velocity at the location of the point particle */
			const std::array<int, 2> X = {floor(particles[i].X), ceil(particles[i].X)};
			const std::array<int, 2> Y = {floor(particles[i].Y), ceil(particles[i].Y)};
			const std::array<int, 2> Z = {floor(particles[i].Z), ceil(particles[i].Z)};
			const std::array<double, 2> fX = {X[0], X[1]};
			const std::array<double, 2> fY = {Y[0], Y[1]};
			const std::array<double, 2> fZ = {Z[0], Z[1]};
			
			/* In x-direction */
			/* Y0Z0 */
			double Uy0z0, Vy0z0, Wy0z0;
			std::array<double, 2> U = {Flow.velocity[{X[0],Y[0],Z[0],0}], Flow.velocity[{X[1],Y[0],Z[0],0}]};
			std::array<double, 2> V = {Flow.velocity[{X[0],Y[0],Z[0],1}], Flow.velocity[{X[1],Y[0],Z[0],1}]};
			std::array<double, 2> W = {Flow.velocity[{X[0],Y[0],Z[0],2}], Flow.velocity[{X[1],Y[0],Z[0],2}]};
#if defined Shan_Chen || defined Kupershtokh
			U[0] += 0.5 * Flow.force[{X[0],Y[0],Z[0],0}] / Flow.density[{X[0],Y[0],Z[0]}];
			U[1] += 0.5 * Flow.force[{X[1],Y[0],Z[0],0}] / Flow.density[{X[1],Y[0],Z[0]}];
			V[0] += 0.5 * Flow.force[{X[0],Y[0],Z[0],1}] / Flow.density[{X[0],Y[0],Z[0]}];
			V[1] += 0.5 * Flow.force[{X[1],Y[0],Z[0],1}] / Flow.density[{X[1],Y[0],Z[0]}];
			W[0] += 0.5 * Flow.force[{X[0],Y[0],Z[0],2}] / Flow.density[{X[0],Y[0],Z[0]}];
			W[1] += 0.5 * Flow.force[{X[1],Y[0],Z[0],2}] / Flow.density[{X[1],Y[0],Z[0]}];
#endif
			interpolation::order_1(fX, U, particles[i].X, Uy0z0);
			interpolation::order_1(fX, V, particles[i].X, Vy0z0);
			interpolation::order_1(fX, W, particles[i].X, Wy0z0);
			/* Y1Z0 */
			double Uy1z0, Vy1z0, Wy1z0;
			U[0] = Flow.velocity[{X[0],Y[1],Z[0],0}];
			U[1] = Flow.velocity[{X[1],Y[1],Z[0],0}];
			V[0] = Flow.velocity[{X[0],Y[1],Z[0],1}];
			V[1] = Flow.velocity[{X[1],Y[1],Z[0],1}];
			W[0] = Flow.velocity[{X[0],Y[1],Z[0],2}];
			W[1] = Flow.velocity[{X[1],Y[1],Z[0],2}];
#if defined Shan_Chen || defined Kupershtokh
			U[0] += 0.5 * Flow.force[{X[0],Y[1],Z[0],0}] / Flow.density[{X[0],Y[1],Z[0]}];
			U[1] += 0.5 * Flow.force[{X[1],Y[1],Z[0],0}] / Flow.density[{X[1],Y[1],Z[0]}];
			V[0] += 0.5 * Flow.force[{X[0],Y[1],Z[0],1}] / Flow.density[{X[0],Y[1],Z[0]}];
			V[1] += 0.5 * Flow.force[{X[1],Y[1],Z[0],1}] / Flow.density[{X[1],Y[1],Z[0]}];
			W[0] += 0.5 * Flow.force[{X[0],Y[1],Z[0],2}] / Flow.density[{X[0],Y[1],Z[0]}];
			W[1] += 0.5 * Flow.force[{X[1],Y[1],Z[0],2}] / Flow.density[{X[1],Y[1],Z[0]}];
#endif
			interpolation::order_1(fX, U, particles[i].X, Uy1z0);
			interpolation::order_1(fX, V, particles[i].X, Vy1z0);
			interpolation::order_1(fX, W, particles[i].X, Wy1z0);
			/* Y0Z1 */
			double Uy0z1, Vy0z1, Wy0z1;
			U[0] = Flow.velocity[{X[0],Y[0],Z[1],0}];
			U[1] = Flow.velocity[{X[1],Y[0],Z[1],0}];
			V[0] = Flow.velocity[{X[0],Y[0],Z[1],1}];
			V[1] = Flow.velocity[{X[1],Y[0],Z[1],1}];
			W[0] = Flow.velocity[{X[0],Y[0],Z[1],2}];
			W[1] = Flow.velocity[{X[1],Y[0],Z[1],2}];
#if defined Shan_Chen || defined Kupershtokh
			U[0] += 0.5 * Flow.force[{X[0],Y[0],Z[1],0}] / Flow.density[{X[0],Y[0],Z[1]}];
			U[1] += 0.5 * Flow.force[{X[1],Y[0],Z[1],0}] / Flow.density[{X[1],Y[0],Z[1]}];
			V[0] += 0.5 * Flow.force[{X[0],Y[0],Z[1],1}] / Flow.density[{X[0],Y[0],Z[1]}];
			V[1] += 0.5 * Flow.force[{X[1],Y[0],Z[1],1}] / Flow.density[{X[1],Y[0],Z[1]}];
			W[0] += 0.5 * Flow.force[{X[0],Y[0],Z[1],2}] / Flow.density[{X[0],Y[0],Z[1]}];
			W[1] += 0.5 * Flow.force[{X[1],Y[0],Z[1],2}] / Flow.density[{X[1],Y[0],Z[1]}];
#endif
			interpolation::order_1(fX, U, particles[i].X, Uy0z1);
			interpolation::order_1(fX, V, particles[i].X, Vy0z1);
			interpolation::order_1(fX, W, particles[i].X, Wy0z1);
			/* Y1Z1 */
			double Uy1z1, Vy1z1, Wy1z1;
			U[0] = Flow.velocity[{X[0],Y[1],Z[1],0}];
			U[1] = Flow.velocity[{X[1],Y[1],Z[1],0}];
			V[0] = Flow.velocity[{X[0],Y[1],Z[1],1}];
			V[1] = Flow.velocity[{X[1],Y[1],Z[1],1}];
			W[0] = Flow.velocity[{X[0],Y[1],Z[1],2}];
			W[1] = Flow.velocity[{X[1],Y[1],Z[1],2}];
#if defined Shan_Chen || defined Kupershtokh
			U[0] += 0.5 * Flow.force[{X[0],Y[1],Z[1],0}] / Flow.density[{X[0],Y[1],Z[1]}];
			U[1] += 0.5 * Flow.force[{X[1],Y[1],Z[1],0}] / Flow.density[{X[1],Y[1],Z[1]}];
			V[0] += 0.5 * Flow.force[{X[0],Y[1],Z[1],1}] / Flow.density[{X[0],Y[1],Z[1]}];
			V[1] += 0.5 * Flow.force[{X[1],Y[1],Z[1],1}] / Flow.density[{X[1],Y[1],Z[1]}];
			W[0] += 0.5 * Flow.force[{X[0],Y[1],Z[1],2}] / Flow.density[{X[0],Y[1],Z[1]}];
			W[1] += 0.5 * Flow.force[{X[1],Y[1],Z[1],2}] / Flow.density[{X[1],Y[1],Z[1]}];
#endif
			interpolation::order_1(fX, U, particles[i].X, Uy1z1);
			interpolation::order_1(fX, V, particles[i].X, Vy1z1);
			interpolation::order_1(fX, W, particles[i].X, Wy1z1);
			/* In y-direction */
			/* Z0 */
			double Uz0, Vz0, Wz0;
			U[0] = Uy0z0;
			U[1] = Uy1z0;
			V[0] = Vy0z0;
			V[1] = Vy1z0;
			W[0] = Wy0z0;
			W[1] = Wy1z0;
			interpolation::order_1(fY, U, particles[i].Y, Uz0);
			interpolation::order_1(fY, V, particles[i].Y, Vz0);
			interpolation::order_1(fY, W, particles[i].Y, Wz0);
			/* Z1 */
			double Uz1, Vz1, Wz1;
			U[0] = Uy0z1;
			U[1] = Uy1z1;
			V[0] = Vy0z1;
			V[1] = Vy1z1;
			W[0] = Wy0z1;
			W[1] = Wy1z1;
			interpolation::order_1(fY, U, particles[i].Y, Uz1);
			interpolation::order_1(fY, V, particles[i].Y, Vz1);
			interpolation::order_1(fY, W, particles[i].Y, Wz1);
			/* In z-direction */
			double Utemp, Vtemp, Wtemp;
			U[0] = Uz0;
			U[1] = Uz1;
			V[0] = Vz0;
			V[1] = Vz1;
			W[0] = Wz0;
			W[1] = Wz1;
			interpolation::order_1(fZ, U, particles[i].Z, Utemp);
			interpolation::order_1(fZ, V, particles[i].Z, Vtemp);
			interpolation::order_1(fZ, W, particles[i].Z, Wtemp);
			particles[i].U = MIN(Utemp, 0.3);
			particles[i].V = MIN(Vtemp, 0.3);
			particles[i].W = MIN(Wtemp, 0.3);
			particles[i].update_position();
		}
	}
}

void Point_particles::write_vtk(int time, int t_vtk, const stl_import& geo_stl, const Parallel_MPI& parallel_MPI) {
	/* check whether it is time to write output */
	if (time % t_vtk == 0) {
		stringstream output_filename;
		/* Slave processors will write *.vtp files */
		if (parallel_MPI.processor_id != MASTER) {
			output_filename << "Alborz_Results/vtk_particle/particle_t" << time << "_" << parallel_MPI.processor_id << ".vtp";
			ofstream output_file;
			/* Create filename */
			output_file.open(output_filename.str().c_str(), ios::out);
			/* Write VTK header */
			output_file << "<?xml version=\"1.0\"?>" << endl;
			output_file << "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">" << endl;
			output_file << "<PolyData>" << endl;
			output_file << "<Piece NumberOfPoints=\"" << particles.size() << "\" NumberOfVerts=\"0\" NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">" << endl;
			output_file << "<Points>" << endl;
			output_file << "<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">" << endl;
			for (int i = 0; i < particles.size(); i++) {
				double xc, yc, zc;
				parallel_MPI.get_coordinates(particles[i].X, particles[i].Y, particles[i].Z,
				                              geo_stl.x_center, geo_stl.y_center, geo_stl.z_center,
				                              xc, yc, zc);
				output_file << xc << " " << yc << " " << zc << endl;
			}
			output_file << "</DataArray>" << endl;
			output_file << "</Points>" << endl;
			output_file << "<PointData Scalars=\"index\">" << endl;
			output_file << "<DataArray type=\"Float32\" Name=\"index\" format=\"ascii\">" << endl;
			for (int i = 0; i < particles.size(); i++) {
				output_file << particles[i].dummy_index << endl;
			}
			output_file << "</DataArray>" << endl;
			output_file << "</PointData>" << endl;
			output_file << "</Piece>" << endl;
			output_file << "</PolyData>" << endl;
			output_file << "</VTKFile>" << endl;
			output_file.close();
		}
		/* Slave processors will write *.pvtp files */
		if (parallel_MPI.processor_id == MASTER) {
			output_filename << "Alborz_Results/vtk_particle/particle_t" << time << ".pvtp";
			ofstream output_file;
			/// Open file
			output_file.open(output_filename.str().c_str());
			output_file << "<?xml version=\"1.0\"?>" << endl;
			output_file << "<VTKFile type=\"PPolyData\" version=\"0.1\" byte_order=\"LittleEndian\">" << endl;
			output_file << "<PPolyData GhostLevel=\"0\">" << endl;
			output_file << "<PPointData Scalars=\"index\">" << endl;
			output_file << "<PDataArray type=\"Float32\" Name=\"Type\"/>" << endl;
			output_file << "</PPointData>" << endl;
			output_file << "<PPoints>" << endl;
			output_file << "<PDataArray type=\"Float32\" NumberOfComponents=\"3\"/>" << endl;
			output_file << "</PPoints>" << endl;
			for (int i = 1; i < parallel_MPI.num_processors; i++) {
				output_file << "<Piece Source=\"particle_t" << time << "_" << i << ".vtp\"/>" << endl;
			}
			output_file << "</PPolyData>" << endl;
			output_file << "</VTKFile>" << endl;
		}
	}
	return;
}
void Point_particles::data_exchange(const stl_import& geo_stl, const Parallel_MPI& parallel_MPI) {
	/* IN X-DIRECTION */
	if (parallel_MPI.processor_id != MASTER) {
		int direction = 0;
		/* IN X-DIRECTION */
		if (direction == 0) {
			/* first detect particles that have gone out, store in temp holder, and remove from cell list */
			vector<Tracer_point_particle> to_left_temp, to_right_temp;
			for (int i = 0; i < particles.size(); i++) {
				bool remove_element = false;
				if (particles[i].X < parallel_MPI.start_XYZ2[0]) {
					remove_element = true;
					to_left_temp.push_back(particles[i]);
				}
				if (particles[i].X > parallel_MPI.end_XYZ2[0] + 1) {
					remove_element = true;
					to_right_temp.push_back(particles[i]);
				}
				if (remove_element) {
					particles.erase(particles.begin() + i);
					i--;
				}
			}
			/* store double corresponding to these particles in a buffer */
			vector<double> to_left(4 * to_left_temp.size(), 0), to_right(4 * to_right_temp.size(), 0);
			for (int i = 0; i < to_left_temp.size(); i++) {
				to_left[4 * i] = fmod(to_left_temp[i].X - parallel_MPI.start_XYZ2[0] + parallel_MPI.start_XYZ[0] + global_parameters.Nx, global_parameters.Nx);
				to_left[4 * i + 1] = fmod(to_left_temp[i].Y - parallel_MPI.start_XYZ2[1] + parallel_MPI.start_XYZ[1] + global_parameters.Ny, global_parameters.Ny);
				to_left[4 * i + 2] = fmod(to_left_temp[i].Z - parallel_MPI.start_XYZ2[2] + parallel_MPI.start_XYZ[2] + global_parameters.Nz, global_parameters.Nz);
				to_left[4 * i + 3] = to_left_temp[i].dummy_index;
			}
			for (int i = 0; i < to_right_temp.size(); i++) {
				to_right[4 * i] = fmod(to_right_temp[i].X - parallel_MPI.start_XYZ2[0] + parallel_MPI.start_XYZ[0] + global_parameters.Nx, global_parameters.Nx);
				to_right[4 * i + 1] = fmod(to_right_temp[i].Y - parallel_MPI.start_XYZ2[1] + parallel_MPI.start_XYZ[1] + global_parameters.Ny, global_parameters.Ny);
				to_right[4 * i + 2] = fmod(to_right_temp[i].Z - parallel_MPI.start_XYZ2[2] + parallel_MPI.start_XYZ[2] + global_parameters.Nz, global_parameters.Nz);
				to_right[4 * i + 3] = to_right_temp[i].dummy_index;
			}
			/* determine left and right neighbours */
			int right_neighbour, left_neighbour;
			right_neighbour = parallel_MPI.proc_arrangement[(parallel_MPI.proc_position[0] + 1) % parallel_MPI.Np_X][parallel_MPI.proc_position[1]][parallel_MPI.proc_position[2]];
			left_neighbour = parallel_MPI.proc_arrangement[(parallel_MPI.proc_position[0] + parallel_MPI.Np_X - 1) % parallel_MPI.Np_X][parallel_MPI.proc_position[1]][parallel_MPI.proc_position[2]];
			/* send and receive sizes first : double to LEFT first*/
			int to_left_size = 0, from_right_size = 0;
			to_left_size = to_left_temp.size();
			MPI_Isend(&(to_left_size), 1, MPI_INT, left_neighbour, LTAG, MPI_COMM_WORLD, &request);
			MPI_Irecv(&(from_right_size), 1, MPI_INT, right_neighbour, LTAG, MPI_COMM_WORLD, &request);
			MPI_Wait(&request, &status);
			if (to_left_size > 0) {
				/* send and receive particle double */
				MPI_Isend(&(to_left[0]), 4 * to_left_size, MPI_DOUBLE, left_neighbour, LTAG, MPI_COMM_WORLD, &request);
			}
			if (from_right_size > 0) {
				/* Now allocate memory to buffer */
				vector<double> from_right(4 * from_right_size, 0);
				MPI_Irecv(&(from_right[0]), 4 * from_right_size, MPI_DOUBLE, right_neighbour, LTAG, MPI_COMM_WORLD, &request);
				MPI_Wait(&request, &status);
				/* put received particles into cell list */
				for (int i = 0; i < from_right_size; i++) {
					Tracer_point_particle temp_particle{0, 0, 0, 0, 0, 0, 0};
					temp_particle.X = from_right[4 * i] - parallel_MPI.start_XYZ[0] + parallel_MPI.start_XYZ2[0];
					temp_particle.Y = from_right[4 * i + 1] - parallel_MPI.start_XYZ[1] + parallel_MPI.start_XYZ2[1];
					temp_particle.Z = from_right[4 * i + 2] - parallel_MPI.start_XYZ[2] + parallel_MPI.start_XYZ2[2];
					temp_particle.dummy_index = from_right[4 * i + 3];
					temp_particle.U = 0;
					temp_particle.V = 0;
					temp_particle.W = 0;
					particles.push_back(temp_particle);
				}
			}
			/* send and receive sizes first : double to RIGHT second*/
			int to_right_size, from_left_size;
			to_right_size = to_right_temp.size();
			MPI_Isend(&(to_right_size), 1, MPI_INT, right_neighbour, LTAG, MPI_COMM_WORLD, &request);
			MPI_Irecv(&(from_left_size), 1, MPI_INT, left_neighbour, LTAG, MPI_COMM_WORLD, &request);
			MPI_Wait(&request, &status);
			if (to_right_size > 0) {
				/* send and receive particle double */
				MPI_Isend(&(to_right[0]), 4 * to_right_size, MPI_DOUBLE, right_neighbour, LTAG, MPI_COMM_WORLD, &request);
			}
			if (from_left_size > 0) {
				/* Now allocate memory to buffer */
				vector<double> from_left(4 * from_left_size, 0);
				MPI_Irecv(&(from_left[0]), 4 * from_left_size, MPI_DOUBLE, left_neighbour, LTAG, MPI_COMM_WORLD, &request);
				MPI_Wait(&request, &status);
				for (int i = 0; i < from_left_size; i++) {
					Tracer_point_particle temp_particle{0, 0, 0, 0, 0, 0, 0};
					temp_particle.X = from_left[4 * i] - parallel_MPI.start_XYZ[0] + parallel_MPI.start_XYZ2[0];
					temp_particle.Y = from_left[4 * i + 1] - parallel_MPI.start_XYZ[1] + parallel_MPI.start_XYZ2[1];
					temp_particle.Z = from_left[4 * i + 2] - parallel_MPI.start_XYZ[2] + parallel_MPI.start_XYZ2[2];
					temp_particle.dummy_index = from_left[4 * i + 3];
					particles.push_back(temp_particle);
				}
			}
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	/* IN Y-DIRECTION */
	if (parallel_MPI.processor_id != MASTER) {
		int direction = 1;
		if (direction == 1) {
			/* first detect particles that have gone out, store in temp holder, and remove from cell list */
			vector<Tracer_point_particle> to_left_temp, to_right_temp;
			for (int i = 0; i < particles.size(); i++) {
				bool remove_element = false;
				if (particles[i].Y < parallel_MPI.start_XYZ2[1]) {
					remove_element = true;
					to_left_temp.push_back(particles[i]);
				}
				if (particles[i].Y > parallel_MPI.end_XYZ2[1] + 1) {
					remove_element = true;
					to_right_temp.push_back(particles[i]);
				}
				if (remove_element) {
					particles.erase(particles.begin() + i);
					i--;
				}
			}
			/* store double corresponding to these particles in a buffer */
			vector<double> to_left(4 * to_left_temp.size(), 0), to_right(4 * to_right_temp.size(), 0);
			for (int i = 0; i < to_left_temp.size(); i++) {
				to_left[4 * i] = fmod(to_left_temp[i].X - parallel_MPI.start_XYZ2[0] + parallel_MPI.start_XYZ[0] + global_parameters.Nx, global_parameters.Nx);
				to_left[4 * i + 1] = fmod(to_left_temp[i].Y - parallel_MPI.start_XYZ2[1] + parallel_MPI.start_XYZ[1] + global_parameters.Ny, global_parameters.Ny);
				to_left[4 * i + 2] = fmod(to_left_temp[i].Z - parallel_MPI.start_XYZ2[2] + parallel_MPI.start_XYZ[2] + global_parameters.Nz, global_parameters.Nz);
				to_left[4 * i + 3] = to_left_temp[i].dummy_index;
			}
			for (int i = 0; i < to_right_temp.size(); i++) {
				to_right[4 * i] = fmod(to_right_temp[i].X - parallel_MPI.start_XYZ2[0] + parallel_MPI.start_XYZ[0] + global_parameters.Nx, global_parameters.Nx);
				to_right[4 * i + 1] = fmod(to_right_temp[i].Y - parallel_MPI.start_XYZ2[1] + parallel_MPI.start_XYZ[1] + global_parameters.Ny, global_parameters.Ny);
				to_right[4 * i + 2] = fmod(to_right_temp[i].Z - parallel_MPI.start_XYZ2[2] + parallel_MPI.start_XYZ[2] + global_parameters.Nz, global_parameters.Nz);
				to_right[4 * i + 3] = to_right_temp[i].dummy_index;
			}
			/* determine left and right neighbours */
			int right_neighbour, left_neighbour;
			right_neighbour = parallel_MPI.proc_arrangement[parallel_MPI.proc_position[0]][(parallel_MPI.proc_position[1] + 1) % parallel_MPI.Np_Y][parallel_MPI.proc_position[2]];
			left_neighbour = parallel_MPI.proc_arrangement[parallel_MPI.proc_position[0]][(parallel_MPI.proc_position[1] + parallel_MPI.Np_Y - 1) % parallel_MPI.Np_Y][parallel_MPI.proc_position[2]];
			/* send and receive sizes first : double to LEFT first*/
			int to_left_size = 0, from_right_size = 0;
			to_left_size = to_left_temp.size();
			MPI_Isend(&(to_left_size), 1, MPI_INT, left_neighbour, LTAG, MPI_COMM_WORLD, &request);
			MPI_Irecv(&(from_right_size), 1, MPI_INT, right_neighbour, LTAG, MPI_COMM_WORLD, &request);
			MPI_Wait(&request, &status);
			if (to_left_size > 0) {
				/* send and receive particle double */
				MPI_Isend(&(to_left[0]), 4 * to_left_size, MPI_DOUBLE, left_neighbour, LTAG, MPI_COMM_WORLD, &request);
			}
			if (from_right_size > 0) {
				/* Now allocate memory to buffer */
				vector<double> from_right(4 * from_right_size, 0);
				MPI_Irecv(&(from_right[0]), 4 * from_right_size, MPI_DOUBLE, right_neighbour, LTAG, MPI_COMM_WORLD, &request);
				MPI_Wait(&request, &status);
				/* put received particles into cell list */
				for (int i = 0; i < from_right_size; i++) {
					Tracer_point_particle temp_particle{0, 0, 0, 0, 0, 0, 0};
					temp_particle.X = from_right[4 * i] - parallel_MPI.start_XYZ[0] + parallel_MPI.start_XYZ2[0];
					temp_particle.Y = from_right[4 * i + 1] - parallel_MPI.start_XYZ[1] + parallel_MPI.start_XYZ2[1];
					temp_particle.Z = from_right[4 * i + 2] - parallel_MPI.start_XYZ[2] + parallel_MPI.start_XYZ2[2];
					temp_particle.dummy_index = from_right[4 * i + 3];
					temp_particle.U = 0;
					temp_particle.V = 0;
					temp_particle.W = 0;
					particles.push_back(temp_particle);
				}
			}
			/* send and receive sizes first : double to RIGHT second*/
			int to_right_size, from_left_size;
			to_right_size = to_right_temp.size();
			MPI_Isend(&(to_right_size), 1, MPI_INT, right_neighbour, LTAG, MPI_COMM_WORLD, &request);
			MPI_Irecv(&(from_left_size), 1, MPI_INT, left_neighbour, LTAG, MPI_COMM_WORLD, &request);
			MPI_Wait(&request, &status);
			if (to_right_size > 0) {
				/* send and receive particle double */
				MPI_Isend(&(to_right[0]), 4 * to_right_size, MPI_DOUBLE, right_neighbour, LTAG, MPI_COMM_WORLD, &request);
			}
			if (from_left_size > 0) {
				/* Now allocate memory to buffer */
				vector<double> from_left(4 * from_left_size, 0);
				MPI_Irecv(&(from_left[0]), 4 * from_left_size, MPI_DOUBLE, left_neighbour, LTAG, MPI_COMM_WORLD, &request);
				MPI_Wait(&request, &status);
				for (int i = 0; i < from_left_size; i++) {
					Tracer_point_particle temp_particle{0, 0, 0, 0, 0, 0, 0};
					temp_particle.X = from_left[4 * i] - parallel_MPI.start_XYZ[0] + parallel_MPI.start_XYZ2[0];
					temp_particle.Y = from_left[4 * i + 1] - parallel_MPI.start_XYZ[1] + parallel_MPI.start_XYZ2[1];
					temp_particle.Z = from_left[4 * i + 2] - parallel_MPI.start_XYZ[2] + parallel_MPI.start_XYZ2[2];
					temp_particle.dummy_index = from_left[4 * i + 3];
					particles.push_back(temp_particle);
				}
			}
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	/* IN Z-DIRECTION */
	if (parallel_MPI.processor_id != MASTER) {
		int direction = 2;
		if (direction == 2) {
			/* first detect particles that have gone out, store in temp holder, and remove from cell list */
			vector<Tracer_point_particle> to_left_temp, to_right_temp;
			for (int i = 0; i < particles.size(); i++) {
				bool remove_element = false;
				if (particles[i].Z < parallel_MPI.start_XYZ2[2]) {
					remove_element = true;
					to_left_temp.push_back(particles[i]);
				}
				if (particles[i].Z > parallel_MPI.end_XYZ2[2] + 1) {
					remove_element = true;
					to_right_temp.push_back(particles[i]);
				}
				if (remove_element) {
					particles.erase(particles.begin() + i);
					i--;
				}
			}
			/* store double corresponding to these particles in a buffer */
			vector<double> to_left(4 * to_left_temp.size(), 0), to_right(4 * to_right_temp.size(), 0);
			for (int i = 0; i < to_left_temp.size(); i++) {
				to_left[4 * i] = fmod(to_left_temp[i].X - parallel_MPI.start_XYZ2[0] + parallel_MPI.start_XYZ[0] + global_parameters.Nx, global_parameters.Nx);
				to_left[4 * i + 1] = fmod(to_left_temp[i].Y - parallel_MPI.start_XYZ2[1] + parallel_MPI.start_XYZ[1] + global_parameters.Ny, global_parameters.Ny);
				to_left[4 * i + 2] = fmod(to_left_temp[i].Z - parallel_MPI.start_XYZ2[2] + parallel_MPI.start_XYZ[2] + global_parameters.Nz, global_parameters.Nz);
				to_left[4 * i + 3] = to_left_temp[i].dummy_index;
			}
			for (int i = 0; i < to_right_temp.size(); i++) {
				to_right[4 * i] = fmod(to_right_temp[i].X - parallel_MPI.start_XYZ2[0] + parallel_MPI.start_XYZ[0] + global_parameters.Nx, global_parameters.Nx);
				to_right[4 * i + 1] = fmod(to_right_temp[i].Y - parallel_MPI.start_XYZ2[1] + parallel_MPI.start_XYZ[1] + global_parameters.Ny, global_parameters.Ny);
				to_right[4 * i + 2] = fmod(to_right_temp[i].Z - parallel_MPI.start_XYZ2[2] + parallel_MPI.start_XYZ[2] + global_parameters.Nz, global_parameters.Nz);
				to_right[4 * i + 3] = to_right_temp[i].dummy_index;
			}
			/* determine left and right neighbours */
			int right_neighbour, left_neighbour;
			right_neighbour = parallel_MPI.proc_arrangement[parallel_MPI.proc_position[0]][parallel_MPI.proc_position[1]][(parallel_MPI.proc_position[2] + 1) % parallel_MPI.Np_Z];
			left_neighbour = parallel_MPI.proc_arrangement[parallel_MPI.proc_position[0]][parallel_MPI.proc_position[1]][(parallel_MPI.proc_position[2] + parallel_MPI.Np_Z - 1) % parallel_MPI.Np_Z];
			/* send and receive sizes first : double to LEFT first*/
			int to_left_size, from_right_size;
			to_left_size = to_left_temp.size();
			MPI_Isend(&(to_left_size), 1, MPI_INT, left_neighbour, LTAG, MPI_COMM_WORLD, &request);
			MPI_Irecv(&(from_right_size), 1, MPI_INT, right_neighbour, LTAG, MPI_COMM_WORLD, &request);
			MPI_Wait(&request, &status);
			if (to_left_size > 0) {
				/* send and receive particle double */
				MPI_Isend(&(to_left[0]), 4 * to_left_size, MPI_DOUBLE, left_neighbour, LTAG, MPI_COMM_WORLD, &request);
			}
			if (from_right_size > 0) {
				/* Now allocate memory to buffer */
				vector<double> from_right(4 * from_right_size, 0);
				MPI_Irecv(&(from_right[0]), 4 * from_right_size, MPI_DOUBLE, right_neighbour, LTAG, MPI_COMM_WORLD, &request);
				MPI_Wait(&request, &status);
				/* put received particles into cell list */
				for (int i = 0; i < from_right_size; i++) {
					Tracer_point_particle temp_particle{0, 0, 0, 0, 0, 0, 0};
					temp_particle.X = from_right[4 * i] - parallel_MPI.start_XYZ[0] + parallel_MPI.start_XYZ2[0];
					temp_particle.Y = from_right[4 * i + 1] - parallel_MPI.start_XYZ[1] + parallel_MPI.start_XYZ2[1];
					temp_particle.Z = from_right[4 * i + 2] - parallel_MPI.start_XYZ[2] + parallel_MPI.start_XYZ2[2];
					temp_particle.dummy_index = from_right[4 * i + 3];
					temp_particle.U = 0;
					temp_particle.V = 0;
					temp_particle.W = 0;
					particles.push_back(temp_particle);
				}
			}
			/* send and receive sizes first : double to RIGHT second*/
			int to_right_size, from_left_size;
			to_right_size = to_right_temp.size();
			MPI_Isend(&(to_right_size), 1, MPI_INT, right_neighbour, LTAG, MPI_COMM_WORLD, &request);
			MPI_Irecv(&(from_left_size), 1, MPI_INT, left_neighbour, LTAG, MPI_COMM_WORLD, &request);
			MPI_Wait(&request, &status);
			if (to_right_size > 0) {
				/* send and receive particle double */
				MPI_Isend(&(to_right[0]), 4 * to_right_size, MPI_DOUBLE, right_neighbour, LTAG, MPI_COMM_WORLD, &request);
			}
			if (from_left_size > 0) {
				/* Now allocate memory to buffer */
				vector<double> from_left(4 * from_left_size, 0);
				MPI_Irecv(&(from_left[0]), 4 * from_left_size, MPI_DOUBLE, left_neighbour, LTAG, MPI_COMM_WORLD, &request);
				MPI_Wait(&request, &status);
				for (int i = 0; i < from_left_size; i++) {
					Tracer_point_particle temp_particle{0, 0, 0, 0, 0, 0, 0};
					temp_particle.X = from_left[4 * i] - parallel_MPI.start_XYZ[0] + parallel_MPI.start_XYZ2[0];
					temp_particle.Y = from_left[4 * i + 1] - parallel_MPI.start_XYZ[1] + parallel_MPI.start_XYZ2[1];
					temp_particle.Z = from_left[4 * i + 2] - parallel_MPI.start_XYZ[2] + parallel_MPI.start_XYZ2[2];
					temp_particle.dummy_index = from_left[4 * i + 3];
					particles.push_back(temp_particle);
				}
			}
		}
	}
}
/* temp function */
void Point_particles::filter_points(double low_rho, const Flow_solver& Flow, const Parallel_MPI& parallel_MPI) {
	if (parallel_MPI.processor_id != MASTER) {
		std::array<double, 2> rho_temp;
		for (int i = 0; i < particles.size(); i++) {
			const std::array<double, 2> fXX = {floor(particles[i].X), ceil(particles[i].X)};
			const std::array<double, 2> fYY = {floor(particles[i].Y), ceil(particles[i].Y)};
			const std::array<double, 2> fZZ = {floor(particles[i].Z), ceil(particles[i].Z)};
			const std::array<double, 2> XX = {fXX[0], fXX[1]};
			const std::array<double, 2> YY = {fYY[0], fYY[1]};
			const std::array<double, 2> ZZ = {fZZ[0], fZZ[1]};


			rho_temp[0] = Flow.density[{XX[0],YY[0],ZZ[0]}];
			rho_temp[1] = Flow.density[{XX[0],YY[0],ZZ[1]}];
			double rho_XnYn;
			interpolation::order_1(fZZ, rho_temp, particles[i].Z, rho_XnYn);

			rho_temp[0] = Flow.density[{XX[1],YY[0],ZZ[0]}];
			rho_temp[1] = Flow.density[{XX[1],YY[0],ZZ[1]}];
			double rho_XpYn;
			interpolation::order_1(fZZ, rho_temp, particles[i].Z, rho_XpYn);
			rho_temp[0] = Flow.density[{XX[0],YY[1],ZZ[0]}];
			rho_temp[1] = Flow.density[{XX[0],YY[1],ZZ[1]}];
			double rho_XnYp;
			interpolation::order_1(fZZ, rho_temp, particles[i].Z, rho_XnYp);
			rho_temp[0] = Flow.density[{XX[1],YY[1],ZZ[0]}];
			rho_temp[1] = Flow.density[{XX[1],YY[1],ZZ[1]}];
			double rho_XpYp;
			interpolation::order_1(fZZ, rho_temp, particles[i].Z, rho_XpYp);

			rho_temp[0] = rho_XnYn;
			rho_temp[1] = rho_XnYp;
			double rho_Xn;
			interpolation::order_1(fYY, rho_temp, particles[i].Y, rho_Xn);
			rho_temp[0] = rho_XpYn;
			rho_temp[1] = rho_XpYp;
			double rho_Xp;
			interpolation::order_1(fYY, rho_temp, particles[i].Y, rho_Xp);

			rho_temp[0] = rho_Xn;
			rho_temp[1] = rho_Xp;
			double rho;
			interpolation::order_1(fXX, rho_temp, particles[i].X, rho);
			if (rho < low_rho) {
				particles.erase(particles.begin() + i);
				i--;
			}
		}
	}
}