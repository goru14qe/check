#ifndef LOOP_FLOW_REACTIVE_H
#define LOOP_FLOW_REACTIVE_H

//#ifdef REGATH_LIB

#include "../Loop.h"
#include "../utils/Timer.hpp"
#include "../Average_field.h"
#include "../CANTERA_INTERFACE.h"
#include "../Geometry.h"
#include "../Thermal_solver.h"
//#include "../REGATH_INTERFACE.h"

class Loop_flow_reactive : public Loop {
public:
	using Loop::Loop;
	//~Loop_flow_reactive();
//#if defined REGATH_LIB
//	ThermoChemistry ThermoChemSchem;
//#endif
	Thermo_chemistry_cantera thermo_chemistry_cantera;
	Non_uniform_boundary non_uniform_boundary;
	Initial_field_slice initial_field_slice;
	Average_Field average;
	DataSampling Sampler;

protected:
	void initialize(const std::string& config_file_path) override;
	void step(int tm) override;
	void register_outputs() override;
	void register_recovery(IO_interface& io_interface) override;

private:
	// pointers to simplify initialization
	std::unique_ptr<Timer> lbm_timer;
	std::unique_ptr<Timer> data_exchange_timer;
	std::unique_ptr<Timer> moments_timer;
	std::unique_ptr<Timer> bc_timer;
};
#endif // LOOP_FLOW_REACTIVE_H