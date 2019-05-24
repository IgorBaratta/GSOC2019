cmake_minimum_required(VERSION 3.10)

project(example)

#------------------------------------------------------------------------------
# Use C++14
set(CMAKE_CXX_STANDARD 14)

# Require C++14
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Do not enable compler-specific extensions
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "-DMODE_KAFFPA")
#------------------------------------------------------------------------------
find_package(MPI REQUIRED)


list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
find_package(KaHIP REQUIRED)

add_executable(ex ex.cpp)

target_include_directories(ex
  PRIVATE
    ${KAHIP_INCLUDE_DIRS}
    ${MPI_CXX_INCLUDE_PATH}
  )

target_link_libraries(ex
  PRIVATE
    ${KAHIP_LIBRARIES}
    ${MPI_CXX_LIBRARIES}
  )
