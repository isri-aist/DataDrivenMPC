cmake_minimum_required(VERSION 3.1)
project(data_driven_mpc)

add_compile_options(-std=c++17)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  roscpp
  nmpc_ddp
  )

# Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

catkin_package(
  CATKIN_DEPENDS
  roscpp
  nmpc_ddp
  DEPENDS EIGEN3
  INCLUDE_DIRS include
  LIBRARIES DDMPC
  )

add_subdirectory(src)
