#include "Average_field.h"
#include "ALBORZ_GlobalVariables.h"
#include "utils/Config_utils.h"
#include <fstream>

Average_Field::Average_Field(){};
Average_Field::~Average_Field(){};
/// ***************************************************** ///
/// ALLOCATE MEMORY FOR AVERAGE FIELD                     ///
/// ***************************************************** ///
void Average_Field::Memory_allocation(const std::string& filename, const Parallel_MPI& parallel_MPI) {
	/// Open input file
	const std::string input_filename = filename + ".dat";
	std::ifstream input_file(input_filename, std::ios::binary);

	find_line_after_header(input_file, "c\tFlow Averaging");
	find_line_after_comment(input_file);
	/// read physical time to start sampling
	input_file >> start_physical_time >> sampling_frequency_physical;
	start_sampling = floor(start_physical_time / global_parameters.D_t);
	sampling_frequency = std::max(sampling_frequency_physical, global_parameters.D_t) / global_parameters.D_t;
	input_file.close();
	if (parallel_MPI.is_master()) {
		std::cout << "Data averaging \n";
		std::cout << "start sampling : " << start_sampling * global_parameters.D_t << " [s]" << std::endl;
		std::cout << "sampling frequency : " << sampling_frequency * global_parameters.D_t << " [s]" << std::endl;
	}
}
/// ***************************************************** ///
/// ADD OUTPUT AVERAGE FIELD                              ///
/// ***************************************************** ///
/*template <typename T, int N>
void Average_Field::add_output(const Tensor<T, N>& avg_field,
                              const std::string& name,
                              IO_interface& io_interface,
							  T scaling,
                              const Solid_field* mask) {

	auto avg = [this](T v, const Base_index_vec<N>& idx) {
		return v / this->counter_sampling;
	};

	io_interface.add_field(filter_view(avg_field, avg), name, scaling, mask);
}

// explicit specializations to reduce compile times
template void Average_Field::add_output<double, 3>(const Tensor<double, 3>&, const std::string&, IO_interface&, double, const Solid_field*);
template void Average_Field::add_output<double, 4>(const Tensor<double, 4>&, const std::string&, IO_interface&, double, const Solid_field*);
*/
/// ***************************************************** ///
/// ADD DATA AVERAGE FIELD                                ///
/// ***************************************************** ///

void Average_Field::add_data(const Parallel_MPI& parallel_MPI, unsigned t) {
	if (t % sampling_frequency != 0) {
		return;
	}
	counter_sampling += 1;
	if (!parallel_MPI.is_master()) {
		for (auto& data : std::get<0>(m_fields)) {
			data->update_fn(data->field, parallel_MPI);
		}
		for (auto& data : std::get<1>(m_fields)) {
			data->update_fn(data->field, parallel_MPI);
		}
	}
}

/// ***************************************************** ///
/// Recovery read and write                               ///
/// ***************************************************** ///
void Average_Field::register_recovery(IO_interface& io) {
	for (auto& data : std::get<0>(m_fields)) {
		io.add_field(data->field, "average_field_" + data->name);
	}
	for (auto& data : std::get<1>(m_fields)) {
		io.add_field(data->field, "average_field_" + data->name);
	}
	io.add_scalar(counter_sampling, "average_field_counter_sampling");
}