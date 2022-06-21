#! /usr/bin/env python

import time
import numpy as np
import matplotlib.pyplot as plt

import rospy
from data_driven_mpc.srv import *


# Setup ROS
rospy.init_node("sample_client_sim_test_mpc_cart")

rospy.wait_for_service("/run_sim_once")
run_sim_once_srv  = rospy.ServiceProxy("/run_sim_once", RunSimOnce)

# Set initial state
req = RunSimOnceRequest()
req.duration = 0.0
req.state = [0.0, 1.0, np.deg2rad(-10.0), 0.0]
req.input = [0.0, 0.0]
res = run_sim_once_srv(req)
state = np.array(res.state)

# Setup variables
time_list = []
state_list = []
force_list = []

t = 0.0 # [sec]
dt = 0.1 # [sec]
while t < 3.0:
    # Calculate manipulation force
    _, _, theta, theta_dot = state
    manip_force_z = -100.0 * theta -20.0 * theta_dot # [N]

    # Run simulation step
    req = RunSimOnceRequest()
    req.duration = dt
    req.state = []
    req.input = [0.0, manip_force_z]
    res = run_sim_once_srv(req)
    state = np.array(res.state)

    # Save
    time_list.append(t)
    state_list.append(state.tolist())
    force_list.append(manip_force_z)

    # Sleep and increment time
    time.sleep(dt)
    t += dt

# Plot result
time_list = np.array(time_list)
state_list = np.array(state_list)
force_list = np.array(force_list)

fig = plt.figure()

data_list = [state_list[:, 0], np.rad2deg(state_list[:, 2]), force_list]
label_list = ["p [m]", "theta [deg]", "force [N]"]
for i in range(len(data_list)):
    ax = fig.add_subplot(3, 1, i + 1)
    ax.plot(time_list, data_list[i], linestyle="-", marker='o')
    ax.set_xlabel("time [s]")
    ax.set_ylabel(label_list[i])
    ax.grid()

plt.show()
