cmake_minimum_required(VERSION 3.1)
project(data_driven_mpc)

add_compile_options(-std=c++17)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  roscpp
  rospy
  nmpc_ddp
  )

# Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Torch
message("-- LIBTORCH_PATH: ${LIBTORCH_PATH}")
list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_PATH})
find_package(Torch REQUIRED)

add_message_files(
  FILES
  Dataset.msg
  )

add_service_files(
  FILES
  RunSimOnce.srv
  GenerateDataset.srv
  )

generate_messages(
  DEPENDENCIES
  )

catkin_package(
  CATKIN_DEPENDS
  roscpp
  rospy
  nmpc_ddp
  DEPENDS EIGEN3
  INCLUDE_DIRS include
  LIBRARIES DDMPC
  )

add_subdirectory(src)

add_subdirectory(samples)

if(CATKIN_ENABLE_TESTING)
  add_subdirectory(tests)
endif()

OPTION(INSTALL_DOCUMENTATION "Generate and install the documentation" OFF)
if(INSTALL_DOCUMENTATION)
  add_subdirectory(doc)
endif()
