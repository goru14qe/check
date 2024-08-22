#ifndef LOOP_FLOW_CRYSTAL_H
#define LOOP_FLOW_CRYSTAL_H

#include "../Loop.h"
#include "../Species_solver.h"
#include "../Phase_Field.h"
#include "../CANTERA_INTERFACE.h"

class Loop_flow_crystal_LB : public Loop {
public:
	using Loop::Loop;
	Phase_Field phase_field;
	Species_solver species_field;
	Thermo_chemistry_cantera thermo_chemistry_cantera;
protected:
	void initialize(const std::string& config_file_path) override;
	void step(int tm) override;
	void register_outputs() override;
	void register_recovery(IO_interface& io_interface) override;

};

// inherit from LB loop because outputs and recovery are the same
class Loop_flow_crystal_FD : public Loop_flow_crystal_LB {
public:
	using Loop_flow_crystal_LB::Loop_flow_crystal_LB;
protected:
	void initialize(const std::string& config_file_path) override;
	void step(int tm) override;
};

#endif