#ifndef AVERAGE_FIELD_H
#define AVERAGE_FIELD_H

#include "Tensor.h"
#include "utils/Meta_prog_utils.h"
#include "Parallel.h"
#include "io/IO_interface.h"

#include <functional>
#include <vector>


// Helper class to compute the average of (derived) fields over time-steps.
class Average_Field {
public:
	unsigned int start_sampling;
	unsigned int counter_sampling;
	unsigned int sampling_frequency;

	double start_physical_time;
	double sampling_frequency_physical;

	Average_Field();
	~Average_Field();

	// Add a field to be averaged over time.
	// @param field The field to be averaged.
	// @param name The name given to the field in the output file.
	// @param io_interface The interface used to write this field.
	// @param mask If provided, every point where mask[{X,Y,Z}] != -1 will be set to 0 in the output.
	// @param scaling Scalar scaling applied to the outputs.
	template <typename T, int N>
	const Tensor<T,N>& add_field(const Tensor<T, N>& field, const std::string& name, IO_interface& io_interface, T scaling, const Solid_field* mask) {
		auto update_fn = [&field](Tensor<T, N>& sum_field, const Parallel_MPI& parallel_MPI) {
			sum_field += field;
		};
		auto& sum_fields = get_element_by_type<Sum_fields<T, N>>(m_fields);
		sum_fields.emplace_back(new Data<T, N>{Tensor<T, N>::zeros(field.sizes()), name, update_fn});

		auto avg = [this](T v, const Base_index_vec<N>& idx) {
			return v / this->counter_sampling;
		};
		io_interface.add_field(filter_view(sum_fields.back()->field, avg), name, scaling, mask);
		return sum_fields.back()->field;
	}

	// Add an average field computed by a custom function.
	// See add_field() above for additional parameters.
	// @param compute_fn Function of signature void(Tensor<T,N>& avg_field, int X, int Y, int Z)
	//                   that is called on every grid point to update the average_field.
	// @param sizes Target size of the average field that is written to in compute_fn.
	template <typename Compute_fn, size_t N, typename T = double>
	const Tensor<T,N>& add_field(Compute_fn compute_fn, const Base_index_vec<N>& sizes, const std::string& name, IO_interface& io_interface, T scaling, const Solid_field* mask) {
		auto update_fn = [compute_fn](Tensor<T, N>& sum_field, const Parallel_MPI& parallel_MPI) {
			for (int X = 0; X < parallel_MPI.dev_end[0]; ++X) {
				for (int Y = 0; Y < parallel_MPI.dev_end[1]; ++Y) {
					for (int Z = 0; Z < parallel_MPI.dev_end[2]; ++Z) {
						compute_fn(sum_field, X, Y, Z);
					}
				}
			}
		};
		auto& sum_fields = get_element_by_type<Sum_fields<T, N>>(m_fields);
		sum_fields.emplace_back(new Data<T, N>{Tensor<T, N>::zeros(sizes), name, update_fn});

		auto avg = [this](T v, const Base_index_vec<N>& idx) {
			return v / this->counter_sampling;
		};
		io_interface.add_field(filter_view(sum_fields.back()->field, avg), name, scaling, mask);
		return sum_fields.back()->field;
	}

	// Add an average field computed by a custom function.
	// See add_field() above for additional parameters.
	// @param Output_fn Function of signature void(T val, const Base_index_vec<N>& idx)
	//                   that is called on every grid point and returns the value for the output field.
	template <typename Compute_fn, typename Output_fn, size_t N, typename T = double>
	const Tensor<T,N>& add_field(Compute_fn compute_fn, const Base_index_vec<N>& sizes, const std::string& name, IO_interface& io_interface, T scaling, const Solid_field* mask, Output_fn output_fn) {
		auto update_fn = [compute_fn](Tensor<T, N>& sum_field, const Parallel_MPI& parallel_MPI) {
			for (int X = 0; X < parallel_MPI.dev_end[0]; ++X) {
				for (int Y = 0; Y < parallel_MPI.dev_end[1]; ++Y) {
					for (int Z = 0; Z < parallel_MPI.dev_end[2]; ++Z) {
						compute_fn(sum_field, X, Y, Z);
					}
				}
			}
		};
		auto& sum_fields = get_element_by_type<Sum_fields<T, N>>(m_fields);
		sum_fields.emplace_back(new Data<T, N>{Tensor<T, N>::zeros(sizes), name, update_fn});

		io_interface.add_field(filter_view(sum_fields.back()->field, output_fn), name, scaling, mask);
		return sum_fields.back()->field;
	}

	void Memory_allocation(const std::string& config_file_name, const Parallel_MPI& parallel_MPI);
	// Updates all fields. Should be called last in the simulation step.
	void add_data(const Parallel_MPI& parallel_MPI, unsigned t);
	void register_recovery(IO_interface& io_interface);

private:
	template <typename T, int N>
	using Update_fn = std::function<void(Tensor<T, N>&, const Parallel_MPI&)>;
	template <typename T, int N>
	struct Data {
		Tensor<T, N> field;
		std::string name;
		Update_fn<T, N> update_fn;
	};
	template <typename T, int N>
	using Sum_fields = std::vector<std::unique_ptr<Data<T, N>>>;
	std::tuple<Sum_fields<double, 3>, Sum_fields<double, 4>> m_fields;
};

#endif