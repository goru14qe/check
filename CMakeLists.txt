cmake_minimum_required (VERSION 3.9) # required for imported target MPI

project(ALBORZ VERSION 2.0
	DESCRIPTION "ALBORZ multi-physics code by S.A.Hosseini"
	HOMEPAGE_URL "https://hosseiniabhari.com/codes")

OPTION(ENABLE_HDF5 "Build with support for hdf5 files" OFF)
OPTION(ENABLE_CANTERA "Build with support for cantera" OFF)
OPTION(ENABLE_REGATH "Build with support for regath" OFF)
OPTION(BUILD_TESTS "Build unit and regression tests" OFF)
# not included by default in BUILD_TESTS because data has to be generated 
# before it can be used
OPTION(BUILD_RESULTS_TEST "Compare results from different runs" OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "Setting build type to 'Release' as none was specified.")
	set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel ..." FORCE)
	# Set the possible values of build type for cmake-gui
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# look for source and header files
file(GLOB_RECURSE SOURCE_FILES "src/*.cpp" "src/*.h")
list(REMOVE_ITEM SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/src/main.cpp")

if (NOT ENABLE_HDF5)
	list(REMOVE_ITEM SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/src/io/H5_ALB.cpp")
endif()

if (NOT ENABLE_CANTERA)
	list(REMOVE_ITEM SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/src/CANTERA_INTERFACE.cpp")
	list(REMOVE_ITEM SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/src/simulations/Loop_flow_reactive.cpp")
endif()

# main code as a library
add_library(alborzlib ${SOURCE_FILES})
target_include_directories(alborzlib INTERFACE "${CMAKE_CURRENT_LIST_DIR}/src/")

# standard program
add_executable(alborz "src/main.cpp")
target_link_libraries(alborz PRIVATE alborzlib)

# cantera headers need C++17 so we might as well enable it for the whole application
if (ENABLE_CANTERA)
	set(CXX_STANDARD_REQ 17)
else()
	set(CXX_STANDARD_REQ 11)
endif()

set_target_properties(alborzlib PROPERTIES
	CXX_STANDARD ${CXX_STANDARD_REQ}
	CXX_STANDARD_REQUIRED YES
)

#set(CMAKE_CXX_COMPILER /lustre/hpe/ws10/ws10.2/ws/xuatanya-alborz/alborz-master_2/build/scorep-mpicxx-filter)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
	# Use gcc- prefixed binutils for LTO
	set(CMAKE_AR gcc-ar)
	set(CMAKE_RANLIB gcc-ranlib)
	set(CMAKE_NM gcc-nm)
	# options for any build type
	# sign-compare is also useful in principle but a proper fix is complicated
	target_compile_options(alborzlib PUBLIC -Wall;-Wno-sign-compare;-Wno-narrowing)
	# optimization flags
	# some of these can produce unexpected results;
	# to exclude such issues switch to the line below
	target_compile_options(alborzlib PUBLIC "$<$<CONFIG:RELEASE>:-g;-O3;-march=native;-flto;-fassociative-math;-fno-trapping-math;-fno-signed-zeros;-fno-math-errno;-freciprocal-math;-DNDEBUG>") # -fopt-info-vec-missed=missedopt.log
	#target_compile_options(alborzlib PUBLIC "$<$<CONFIG:RELEASE>:-O2;-DNDEBUG>")
	target_link_options(alborzlib PUBLIC "$<$<CONFIG:RELEASE>:-flto=auto>")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
	# Use llvm- prefixed binutils for LTO
	set(CMAKE_AR llvm-ar)
	set(CMAKE_RANLIB llvm-ranlib)
	set(CMAKE_NM llvm-nm)
	# options for any build type
	# sign-compare is also useful in principle but a proper fix is complicated
	target_compile_options(alborzlib PUBLIC -Wall;-Wno-sign-compare;-Wno-narrowing)
	# optimization flags
	# some of these can produce unexpected results;
	# to exclude such issues switch to the line below

	# !!!!! -flto breaks the application for cpe it seems (some value out of range -> abort) !!!!!
	target_compile_options(alborzlib PUBLIC "$<$<CONFIG:RELEASE>:-O3;-march=native;-fassociative-math;-fno-trapping-math;-fno-signed-zeros;-fno-math-errno;-freciprocal-math;-DNDEBUG>") # -fopt-info-vec-missed=missedopt.log
	#target_compile_options(alborzlib PUBLIC "$<$<CONFIG:RELEASE>:-O2;-DNDEBUG>")

	# -flto breaks the application for cpe it seems (some value out of range -> abort)
	target_link_options(alborzlib PUBLIC "$<$<CONFIG:RELEASE>:-flto>")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
	# options for any build type
	# sign-compare is also useful in principle but a proper fix is complicated
	target_compile_options(alborzlib PUBLIC -Wall;-Wno-sign-compare;-Wno-narrowing)
	# optimization flags
	# some of these can produce unexpected results;
	# to exclude such issues switch to the line below
	target_compile_options(alborzlib PUBLIC "$<$<CONFIG:RELEASE>:-O3;-march=haswell;-ipo;-fassociative-math;-fno-trapping-math;-fno-signed-zeros;-fno-math-errno;-freciprocal-math;-DNDEBUG>") # -fopt-info-vec-missed=missedopt.log
	#target_compile_options(alborzlib PUBLIC "$<$<CONFIG:RELEASE>:-O2;-DNDEBUG>")
	target_link_options(alborzlib PUBLIC "$<$<CONFIG:RELEASE>:-ipo>")
	
	elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
	# disable all warnings for now since they are overwhelming
	target_compile_options(alborzlib PUBLIC /w ) # /wd4018 /wd4267
	set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT alborz)
endif()

#----------------------------------------------------------------------------
# libraries
#----------------------------------------------------------------------------

# MPI
find_package(MPI REQUIRED)
target_link_libraries(alborzlib PUBLIC MPI::MPI_CXX)

# HDF5
if (ENABLE_HDF5)
	SET(HDF5_PREFER_PARALLEL TRUE)
	# c interface and high-level c interface
	find_package (HDF5 REQUIRED COMPONENTS C HL) 

	# older cmake versions do not define these targets yet
	if (NOT TARGET hdf5::hdf5)
		add_library(hdf5::hdf5 INTERFACE IMPORTED)
		set_target_properties(hdf5::hdf5 PROPERTIES
			INTERFACE_INCLUDE_DIRECTORIES "${HDF5_C_INCLUDE_DIRS}"
			INTERFACE_COMPILE_DEFINITIONS "${HDF5_C_DEFINITIONS}")
		target_link_libraries(hdf5::hdf5 INTERFACE ${HDF5_C_LIBRARIES})
	endif()

	if (NOT TARGET hdf5::hdf5_hl)
		add_library(hdf5::hdf5_hl INTERFACE IMPORTED)
		set_target_properties(hdf5::hdf5_hl PROPERTIES
			INTERFACE_INCLUDE_DIRECTORIES "${HDF5_C_HL_INCLUDE_DIRS}"
			INTERFACE_COMPILE_DEFINITIONS "${HDF5_C_HL_DEFINITIONS}")
		target_link_libraries(hdf5::hdf5_hl INTERFACE ${HDF5_C_HL_LIBRARIES})
	endif()

	target_link_libraries(alborzlib PUBLIC hdf5::hdf5 hdf5::hdf5_hl)
	target_compile_definitions(alborzlib PUBLIC "WITH_HDF5")
endif ()

# Cantera
if (ENABLE_CANTERA)
	set(CANTERA_LIB "/home/tneeraj/cantera/cantera/build/lib/libcantera.a" CACHE PATH "path of the cantera library")
	set(CANTERA_INCLUDE_PATH "/home/tneeraj/cantera/cantera/include/" CACHE PATH "path of the cantera headers")
	if (EXISTS "${CANTERA_LIB}")
		target_link_libraries(alborzlib PUBLIC "${CANTERA_LIB}")
		# Cantera dependencies should be installed together with the library
		# in which case this should not be necessary. 
		find_package(fmt QUIET)
		find_package(yaml-cpp QUIET)
		if (fmt_FOUND AND yaml-cpp_FOUND)
			target_link_libraries(alborzlib PUBLIC sundials_cvodes sundials_ida sundials_nvecserial fmt blas lapack yaml-cpp)
		endif()
		target_include_directories(alborzlib PUBLIC ${CANTERA_INCLUDE_PATH})
		target_compile_definitions(alborzlib PUBLIC "WITH_CANTERA")
	else()
		message(SEND_ERROR "Could not find the cantera library at \"${CANTERA_LIB}\". \
			You can manually set the variables CANTERA_LIB and CANTERA_INCLUDE_PATH.")
	endif ()
endif()

# Regath
if (ENABLE_REGATH)
	set(REGATH_LIB "${CMAKE_CURRENT_LIST_DIR}/dependencies/regath/regath_library.a" CACHE PATH "path of the regath library")
	if (EXISTS ${REGATH_LIB})
		# we also need to link againsst the fortran runtime
		enable_language(Fortran)
		# find required libraries through the compiler
		if(${CMAKE_Fortran_FLAGS} MATCHES ".*-m64.*")
			EXECUTE_PROCESS( COMMAND ${CMAKE_Fortran_COMPILER} -m64 -print-file-name=libgfortran.a
				OUTPUT_VARIABLE RES OUTPUT_STRIP_TRAILING_WHITESPACE )
			EXECUTE_PROCESS( COMMAND ${CMAKE_Fortran_COMPILER} -m64 -print-file-name=libgcc.a
				OUTPUT_VARIABLE RESGCC OUTPUT_STRIP_TRAILING_WHITESPACE )
			EXECUTE_PROCESS( COMMAND ${CMAKE_Fortran_COMPILER} -m64 -print-file-name=libquadmath.a
				OUTPUT_VARIABLE RESQUAD OUTPUT_STRIP_TRAILING_WHITESPACE )
		else()
			EXECUTE_PROCESS( COMMAND ${CMAKE_Fortran_COMPILER} -print-file-name=libgfortran.a
				OUTPUT_VARIABLE RES OUTPUT_STRIP_TRAILING_WHITESPACE )
			EXECUTE_PROCESS( COMMAND ${CMAKE_Fortran_COMPILER} -print-file-name=libgcc.a
				OUTPUT_VARIABLE RESGCC OUTPUT_STRIP_TRAILING_WHITESPACE )
			EXECUTE_PROCESS( COMMAND ${CMAKE_Fortran_COMPILER} -print-file-name=libquadmath.a
				OUTPUT_VARIABLE RESQUAD OUTPUT_STRIP_TRAILING_WHITESPACE )
		endif(${CMAKE_Fortran_FLAGS} MATCHES ".*-m64.*")
		SET( FORTRAN_RUNTIME ${RES} ${RESGCC} ${RESQUAD} CACHE INTERNAL "The fortran runtime library" )
		
		target_compile_definitions(alborzlib PUBLIC "REGATH_LIB")
		target_link_libraries(alborzlib PRIVATE ${REGATH_LIB} ${FORTRAN_RUNTIME})
	else()
		message(SEND_ERROR "Could not find the regath library at \"${REGATH_LIB}\". \
			You can manually set the variables REGATH_LIB to provide a hint.")
	endif()
endif()

find_package(OpenMP REQUIRED)
target_link_libraries(alborzlib PUBLIC OpenMP::OpenMP_CXX)

#add_executable(tensor_benchmark "tests/Benchmark_tensor.cpp")
#target_link_libraries(tensor_benchmark PRIVATE alborzlib)

#----------------------------------------------------------------------------
# test
#----------------------------------------------------------------------------
if (BUILD_TESTS)
	enable_testing()
	add_subdirectory(tests)
endif()
