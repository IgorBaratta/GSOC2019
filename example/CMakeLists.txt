cmake_minimum_required(VERSION 3.10)

project(example)

#------------------------------------------------------------------------------
# Use C++14
set(CMAKE_CXX_STANDARD 14)

# Require C++14
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Do not enable compler-specific extensions
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "-DMODE_KAFFPAE")
#------------------------------------------------------------------------------
# Require MPI for this project:
find_package(MPI REQUIRED)
find_package(OpenMP)
if (OPENMP_FOUND)
  add_definitions(-fopenmp -DHAVE_OPENMP)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  if (CMAKE_COMPILER_IS_GNUCC)
    set(EXTERNAL_LIBS ${EXTERNAL_LIBS} gomp)
    message("-- OpenMP found, parallel computer enabled")
  endif(CMAKE_COMPILER_IS_GNUCC)
endif(OPENMP_FOUND)

set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})
include_directories(MPI_INCLUDE_PATH)


list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
find_package(KaHIP REQUIRED)

add_executable(ex ex.cpp)

target_include_directories(ex
  PRIVATE
    ${MPI_CXX_INCLUDE_PATH}
    ${KAHIP_INCLUDE_DIRS}
  )

target_link_libraries(ex
  PRIVATE
    ${MPI_CXX_LIBRARIES}
    ${KAHIP_LIBRARIES}
  )
