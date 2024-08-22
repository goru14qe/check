# Alborz
## Build
You need a C++11 compiler (tested on g++10,11,12, clang++14) and cmake (>=3.9). The only dependency is a compatible MPI library(>=2.0?). Then just run
```bash
git clone https://code.ovgu.de/sehossei/alborz.git
mkdir build
cd build
cmake ..
make -j4
```
Optionally you can enable additional features by calling cmake once with additional arguments. To build with [hdf5](https://www.hdfgroup.org/solutions/hdf5) support, which is required to make recovery saves, run
```
cmake .. -DENABLE_HDF5=ON
```
Make sure that the required library is installed or that the module is loaded. For hdf5 use the parallel version, e.g.:
```
sudo apt install libhdf5-openmpi-dev
```
On some platforms additional care needs to be taken when selecting compatible versions of MPI and hdf5. See below for a list of configurations that are known to work.

| system | MPI | hdf |
|-----|-------------|---------------|
|ubuntu 21| openmpi-4.1 | libhdf5-mpi-dev-1.10 |
|sofja | openmpi-4.1 | hdf-1.12.2-mpi|

---
Integration of the library [Cantera](https://cantera.org/) can be enabled by calling cmake with
```
cmake .. -DENABLE_CANTERA=ON
```
It can be installed on ubuntu via
```
sudo apt-add-repository ppa:cantera-team/cantera
sudo apt install cantera-dev
```
or build from [source](https://www.cantera.org/install/compiling-install.html#sec-compiling). Be aware that newer versions of Cantera require C++17.
## Tests
A number of tests are provided to check both specific features as well some to validate the whole simulation. To use tests you first need to enable them with
```
cmake .. -DBUILD_TESTS=ON
```
Afterwards the required executable will automatically be build together with the main code when running make. After a successful build you can simply run all tests with the command
```
ctest
```
from within the build directory, i.e. the same one where cmake is called from. If an individual test fails or if you are interested in the accuracy of a validation simulation, you can also run the tests individually to get additional information. Each test has its own executable, found in `build/tests`. They should be called from within this directory and, for the tests using MPI, with the proper number of cores. For example, to run `test_2d_poiseuille`, use
```
mpirun -n 6 --oversubscribe test_2d_poiseuille
```
The proper arguments for each test can also be seen in `tests/CMakeLists.txt`. The number of processes for tests using MPI are chosen for a 6-core CPU. If you have fewer cores, the tests will still run but take significantly longer. Also notice that `test_particle_sed2` can take upwards of 40 mins, so if available, more cores should be used for this test.

There are also two cases available to validate crystallization. Since they do not come with a custom executable, results have to be checked manually. The cases can be run with the regular alborz executable directly from the `build`  directory with
```
mpirun -n 5 alborz ../tests/data/PT_Younsi_hexagonal/Input_Config flow_crystal_lb
mpirun -n 5 alborz ../tests/data/Mandelic_Convection/Input_Config flow_crystal_fd
```
Finally, there is a special test that just compares results of two different simulation runs. This test is intended for regression testing, i.e. to validate that changes to the code do not alter simulation results unexpectedly. To build the executable it has to be enabled separately by
```
cmake .. -DBUILD_RESULTS_TEST=ON
```
Afterwards, the test can be run from the `build`  directory with
```
mpirun -n <num_proc> tests/test_simulation_results <path/to/Input_config> --generate_ref
```
to generate reference data, e.g. from a previous code version.
Then switch to the new version and rebuild the test. When running the simulation again with
```
mpirun -n <num_proc> tests/test_simulation_results <path/to/Input_config>
```
results will be compared to those of the recorded run and differences reported.
## Performance
The choice of parameters can have an significant impact on performance. Below some reference values are listed for what be expected with the current version of Alborz. Use these as comparison before running a large simulation to see whether some tweaks should be done.

| X | Y | Z | MLUPS |
|-|-|-|-|
| 3 | 3 | 7 | 1.40 |
| 3 | 7 | 3 | 1.41 |
| 63 | 1 | 1 | 3.29 |
| 9 | 7 | 1 | 3.42 |

Measurements where taken on Sofja with 2 nodes x 32 cores (Ice Lake Xeon 6326) with `Loop_flow_heat_full_lb` on a domain of size 308 x 308 x 308 without obstacles. 

To make the lattice updates more efficient, the orientation of the domain and the decomposition should be chosen such that chunks have a large number of nodes in Z direction. If the domain is too small in Z-direction, the speedup seen above may not be impossible. A secondary objective while selecting the domain decomposition is to minimize the necessary synchronization. To quickly generate possible decompositions with their surface area, the script `scripts/generate_decompositions.py` can be used.
```bash
python generate_decompositions.py --num_proc 63 --domain 308 308 308
```
By default, the script only shows a selection of promising candidates, which can be tested in short runs to determine the sweet spot between surface area and number of Z nodes. Additional options of the script are displayed by the `--help` argument.
## Troubleshooting
### Could NOT find MPI: MPI component 'Fortran' was request, but language 'Fortran' is not enabled
This error is somewhat misleading as the issue is not related to Fortran but instead to the MPI installation. Try updating OpenMPI as indicated above.