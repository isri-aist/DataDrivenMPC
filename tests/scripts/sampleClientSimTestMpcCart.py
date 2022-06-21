#! /usr/bin/env python

import time
import numpy as np
import matplotlib.pyplot as plt

import rospy
from data_driven_mpc.srv import *


# Setup ROS
rospy.init_node("sample_client_sim_test_mpc_cart")
run_sim_once_cli = rospy.ServiceProxy("/run_sim_once", RunSimOnce)
generate_dataset_cli = rospy.ServiceProxy("/generate_dataset", GenerateDataset)

# Generate dataset
rospy.wait_for_service("/generate_dataset")

req = GenerateDatasetRequest()
req.filename = "/tmp/DataDrivenMPCDataset.bag"
req.dataset_size = 1000
req.dt = 0.05 # [sec]
req.state_max = np.array([1.0, 1.0, np.deg2rad(30), np.deg2rad(60)])
req.state_min = -1 * req.state_max
req.input_max = np.array([100.0, 100.0])
req.input_min = -1 * req.input_max
generate_dataset_cli(req)

# Set initial state
rospy.wait_for_service("/run_sim_once")

req = RunSimOnceRequest()
req.dt = 0.0
req.state = [0.0, 1.0, np.deg2rad(-10.0), 0.0]
req.input = [0.0, 0.0]
res = run_sim_once_cli(req)
state = np.array(res.state)

# Setup variables
time_list = []
state_list = []
force_list = []

t = 0.0 # [sec]
end_t = 3.0 # [sec]
dt = 0.05 # [sec]
while True:
    # Calculate manipulation force
    _, _, theta, theta_dot = state
    manip_force_z = -100.0 * theta -20.0 * theta_dot # [N]

    # Save
    time_list.append(t)
    state_list.append(state.tolist())
    force_list.append(manip_force_z)

    # Check terminal condition
    if t >= end_t:
        break

    # Run simulation step
    req = RunSimOnceRequest()
    req.dt = dt
    req.state = []
    req.input = [0.0, manip_force_z]
    res = run_sim_once_cli(req)
    state = np.array(res.state)

    # Sleep and increment time
    time.sleep(dt)
    t += dt

# Plot result
time_list = np.array(time_list)
state_list = np.array(state_list)
force_list = np.array(force_list)

fig = plt.figure()

data_list = [state_list[:, 0], np.rad2deg(state_list[:, 2]), force_list]
label_list = ["pos [m]", "angle [deg]", "force [N]"]
for i in range(len(data_list)):
    ax = fig.add_subplot(3, 1, i + 1)
    ax.plot(time_list, data_list[i], linestyle="-", marker='o')
    ax.set_xlabel("time [s]")
    ax.set_ylabel(label_list[i])
    ax.grid()

plt.show()
