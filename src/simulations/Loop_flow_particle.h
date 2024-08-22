#ifndef LOOP_FLOW_PARTICLE_H
#define LOOP_FLOW_PARTICLE_H

#include "../Loop.h"
#include "../Average_field.h"
#include "../Particle_sim.h"
#include "../CANTERA_INTERFACE.h"

class Loop_flow_particle : public Loop {
public:
	using Loop::Loop;

protected:
	void initialize(const std::string& config_file_path) override;
	void step(int tm) override;
	void register_outputs() override;
	void register_recovery(IO_interface& io_interface) override;

	Particle_sim particle_sim;
	Average_Field average;
	Thermo_chemistry_cantera thermo_chemistry_cantera;
};

#endif