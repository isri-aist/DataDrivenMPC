# [DataDrivenMPC](https://github.com/isri-aist/DataDrivenMPC)
Model predictive control based on data-driven model

[![CI](https://github.com/isri-aist/DataDrivenMPC/actions/workflows/ci.yaml/badge.svg)](https://github.com/isri-aist/DataDrivenMPC/actions/workflows/ci.yaml)
[![Documentation](https://img.shields.io/badge/doxygen-online-brightgreen?logo=read-the-docs&style=flat)](https://isri-aist.github.io/DataDrivenMPC/)

## Install

### Requirements
- Compiler supporting C++17
- Tested with Ubuntu 18.04 / ROS Melodic

### Dependencies
This package depends on
- [libtorch (PyTorch C++ Frontend)](https://pytorch.org/cppdocs/installing.html)
- [NMPC](https://github.com/isri-aist/NMPC)

Some tests depend on
- [PyBullet](https://pybullet.org)

### Installation procedure
It is assumed that ROS is installed.

1. Follow [the official instructions](https://pytorch.org/cppdocs/installing.html) to download and extract the zip file of libtorch.

2. Setup catkin workspace.
```bash
$ mkdir -p ~/ros/ws_ddmpc/src
$ cd ~/ros/ws_ddmpc
$ wstool init src
$ wstool set -t src isri-aist/NMPC git@github.com:isri-aist/NMPC.git --git -y
$ wstool set -t src isri-aist/DataDrivenMPC git@github.com:isri-aist/DataDrivenMPC.git --git -y
$ wstool update -t src
```

3. Install dependent packages.
```bash
$ source /opt/ros/${ROS_DISTRO}/setup.bash
$ rosdep install -y -r --from-paths src --ignore-src
```

4. Build a package.
```bash
$ catkin build data_driven_mpc -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLIBTORCH_PATH=<absolute path to libtorch> --catkin-make-args all tests
```
`<absolute path to libtorch>` is the path to the directory named libtorch that was extracted in step 1.

## Examples
Make sure that it is built with `--catkin-make-args tests` option.

### [MPC for Van der Pol oscillator](tests/src/TestMpcOscillator.cpp)
Control the [Van der Pol oscillator](https://web.casadi.org/docs/#a-simple-test-problem) by the learned state equation.
```bash
$ rosrun data_driven_mpc TestMpcOscillator
```

### [MPC for walking with pushing](tests/src/TestMpcPushWalk.cpp)
Control the CoM motion of robot and object by combining the known CoM-ZMP model and the learned object dynamics model.
```bash
$ rosrun data_driven_mpc TestMpcPushWalk --gtest_filter=*.RunMPC
$ rosrun data_driven_mpc plotTestMpcPushWalk.py
```

### [MPC for cart pushing](tests/src/TestMpcCart.cpp)
Control the position and angle of the one wheel cart on the PyBullet dynamics simulator.
```bash
# 3-second simulation
$ rostest data_driven_mpc TestMpcCart.test enable_gui:=true --text
# Endless simulation
$ rostest data_driven_mpc TestMpcCart.test no_exit:=true enable_gui:=true --text
```

### [MPC for walking with cart pushing](tests/src/TestMpcCartWalk.cpp)
Control the robot CoM and the position and angle of the one wheel cart on the PyBullet dynamics simulator.
The robot CoM-ZMP model is known. The object dynamics model is learned.
```bash
$ rostest data_driven_mpc TestMpcCartWalk.test enable_gui:=true --text
```
