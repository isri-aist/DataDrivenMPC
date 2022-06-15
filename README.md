# [DataDrivenMPC](https://github.com/isri-aist/DataDrivenMPC)
Model predictive control based on data-driven model

## Install

### Requirements
- Compiler supporting C++17
- Tested with Ubuntu 18.04 / ROS Melodic

### Dependencies
This package depends on
- [libtorch (PyTorch C++ Frontend)](https://pytorch.org/cppdocs/installing.html)
- [NMPC](https://github.com/isri-aist/NMPC)

### Installation procedure
It is assumed that ROS is installed.

```bash
$ catkin build -DLIBTORCH_PATH=<absolute path to libtorch>
```
