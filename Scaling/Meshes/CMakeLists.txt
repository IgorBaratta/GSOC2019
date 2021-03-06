cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME test)
project(${PROJECT_NAME})

# Get DOLFINX configuration data (DOLFINXConfig.cmake must be in
# DOLFINX_CMAKE_CONFIG_PATH)
if (NOT TARGET dolfinx)
  find_package(DOLFINX REQUIRED)
endif()

# Executable
add_executable(${PROJECT_NAME} meshes.cpp)

# Set C++17 standard
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

# Target libraries
target_link_libraries(${PROJECT_NAME} dolfinx)

# Do not throw error for 'multi-line comments' (these are typical in
# rst which includes LaTeX)
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-Wno-comment" HAVE_NO_MULTLINE)
target_compile_options(${PROJECT_NAME} PRIVATE $<$<BOOL:${HAVE_NO_MULTLINE}>:-Wno-comment>)

# Test targets
set(TEST_PARAMETERS -np 3 "./${PROJECT_NAME}")
add_test(NAME ${PROJECT_NAME}_mpi COMMAND "mpirun" ${TEST_PARAMETERS})
add_test(NAME ${PROJECT_NAME}_serial COMMAND ${PROJECT_NAME})
