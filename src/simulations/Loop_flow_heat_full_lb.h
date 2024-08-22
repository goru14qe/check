#ifndef LOOP_FLOW_HEAT_FULL_LB_H
#define LOOP_FLOW_HEAT_FULL_LB_H

#include "../Loop.h"
#include "../utils/Timer.hpp"
#include "../CANTERA_INTERFACE.h"

class Loop_flow_heat_full_lb : public Loop {
public:
	using Loop::Loop;
	Thermo_chemistry_cantera thermo_chemistry_cantera;
	~Loop_flow_heat_full_lb();

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

#endif