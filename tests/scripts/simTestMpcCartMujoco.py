#! /usr/bin/env python

import time
import numpy as np
import mujoco
import mujoco.viewer

import rospy
import rosbag
from data_driven_mpc.msg import *
from data_driven_mpc.srv import *


class SimTestMpcCart(object):
    def __init__(self, enable_gui):
        # TODO: Support enable_gui=False

        # Setup xml
        self.box_half_scale = np.array([0.35, 0.25, 0.15]) # [m]
        box_mass = rospy.get_param("~box_mass", 8.0) # [kg]
        box_com_offset = np.array([-0.02, 0.0, -0.1]) # [m]
        self.cylinder_radius = 0.1 # [m]
        cylinder_height = 0.1 # [m]
        cylinder_mass = 2.0 # [kg]
        lateral_friction = rospy.get_param("~lateral_friction", 0.05)

        xml_str = """
<mujoco model="sim_test_mpc_cart">
  <option integrator="RK4"/>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.1 0.2 0.3" rgb2="0.2 0.3 0.4"
             width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="2 2" texuniform="true"/>
  </asset>
  <worldbody>
    <light directional="true" castshadow="false" pos="0.0 0.0 2.0"/>
    <geom size="0.0 0.0 0.01" type="plane" material="grid"/>
    <body name="cart" pos="0.0 0.0 {cart_pos_z}">
      <freejoint/>
      <geom name="box" type="box" rgba="0.0 1.0 0.0 0.8"
            size="{box_half_scale[0]} {box_half_scale[1]} {box_half_scale[2]}" mass="1e-3"/>
      <geom name="box_mass" type="box" contype="0" conaffinity="0" group="3"
            size="{box_half_scale[0]} {box_half_scale[1]} {box_half_scale[2]}" mass="{box_mass}"
            pos="{box_com_offset[0]} {box_com_offset[1]} {box_com_offset[2]}"/>
      <geom name="cylinder" type="cylinder" rgba="0.1 0.1 0.1 0.8"
            size="{cylinder_radius} {cylinder_half_height}" mass="{cylinder_mass}"
            priority="1" friction="{sliding_friction} 0.005 0.0001"
            pos="0.0 0.0 {cylinder_pos_z}" euler="90.0 0.0 0.0"/>
    </body>
  </worldbody>
</mujoco>
""".format(cart_pos_z=2.0*self.cylinder_radius+self.box_half_scale[2],
           box_half_scale=self.box_half_scale,
           box_mass=box_mass,
           box_com_offset=box_com_offset,
           cylinder_radius=self.cylinder_radius,
           cylinder_half_height=0.5*cylinder_height,
           cylinder_mass=cylinder_mass,
           sliding_friction=lateral_friction,
           cylinder_pos_z=-1*(self.cylinder_radius+self.box_half_scale[2]))
        # with open("/tmp/simTestMpcCartMujoco.xml", "w") as f:
        #     f.write(xml_str)

        # Instantiate simulator
        self.model = mujoco.MjModel.from_xml_string(xml_str)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.cam.azimuth = 120.0
        self.viewer.cam.distance = 10.0

        # Set simulation parameters
        self.dt = 0.005 # [sec]
        self.model.opt.timestep = self.dt

        # Setup ROS
        run_sim_once_srv = rospy.Service("/run_sim_once", RunSimOnce, self.runSimOnceCallback)
        generate_dataset_srv = rospy.Service("/generate_dataset", GenerateDataset, self.generateDatasetCallback)

    def runOnce(self, manip_force=None):
        """"Run simulation step once.

        Args:
            manip_force manipulation force in world frame
        """
        if manip_force is not None:
            # Apply manipulation force
            box_link_pos = self.data.geom("box").xpos
            box_link_quat = np.zeros(4)
            mujoco.mju_mat2Quat(box_link_quat, self.data.geom("box").xmat)
            manip_pos_local = np.array([-1 * self.box_half_scale[0], 0.0, self.box_half_scale[2]])
            manip_pos = np.zeros(3)
            mujoco.mju_trnVecPose(manip_pos, box_link_pos, box_link_quat, manip_pos_local)
            manip_moment = np.cross(manip_pos - self.data.body("cart").xipos, manip_force)
            self.data.body("cart").xfrc_applied += np.concatenate([manip_force, manip_moment])

            # TODO: Visualize manipulation force

        # Process simulation step
        mujoco.mj_step(self.model, self.data)
        self.viewer.sync()

    def getState(self):
        """"Get state [p, p_dot, theta, theta_dot]."""
        p = self.data.geom("cylinder").xpos[0] # [m]
        box_quat = np.zeros(4)
        mujoco.mju_mat2Quat(box_quat, self.data.geom("box").xmat)
        theta = np.arcsin(2 * (box_quat[0] * box_quat[2] - box_quat[1] * box_quat[3])) # [rad]
        local_pos_from_cylinder_to_box = np.array([0.0, 0.0, self.cylinder_radius + self.box_half_scale[2]])
        global_pos_from_cylinder_to_box = np.zeros(3)
        mujoco.mju_rotVecQuat(global_pos_from_cylinder_to_box, local_pos_from_cylinder_to_box, box_quat)
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_XBODY, self.model.body("cart").id, vel, 0)
        vel[3:6] += np.cross(vel[0:3], -1 * global_pos_from_cylinder_to_box)
        p_dot = vel[3] # [m/s]
        theta_dot = vel[1] # [rad/s]
        return np.array([p, p_dot, theta, theta_dot])

    def setState(self, state):
        """Set state [p, p_dot, theta, theta_dot]."""
        p, p_dot, theta, theta_dot = state
        box_quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(box_quat, np.array([0.0, 1.0, 0.0]), theta)
        local_pos_from_cylinder_to_box = np.array([0.0, 0.0, self.cylinder_radius + self.box_half_scale[2]])
        global_pos_from_cylinder_to_box = np.zeros(3)
        mujoco.mju_rotVecQuat(global_pos_from_cylinder_to_box, local_pos_from_cylinder_to_box, box_quat)
        box_pos = np.array([p, 0.0, self.cylinder_radius]) + global_pos_from_cylinder_to_box
        self.data.qpos = np.concatenate([box_pos, box_quat])
        linear_vel = np.array([p_dot, 0.0, 0.0]) + np.cross(np.array([0.0, theta_dot, 0.0]), global_pos_from_cylinder_to_box)
        angular_vel = np.array([0.0, theta_dot, 0.0])
        self.data.qvel = np.concatenate([linear_vel, angular_vel])

    def runSimOnceCallback(self, req):
        """ROS service callback to run simulation step once."""
        assert len(req.state) == 0 or len(req.state) == 4, \
            "req.state dimension is invalid {} != 0 or 4".format(len(req.state))
        assert len(req.input) == 2, \
            "req.input dimension is invalid {} != 2".format(len(req.input))
        assert len(req.additional_data) == 0, \
            "req.additional_data dimension is invalid {} != 0".format(len(req.additional_data))

        if len(req.state) > 0:
            self.setState(np.array(req.state))

        manip_force = np.array([req.input[0], 0.0, req.input[1]])
        for i in range(int(req.dt / self.dt)):
            self.runOnce(manip_force)

        res = RunSimOnceResponse()
        res.state = self.getState()
        return res

    def generateDatasetCallback(self, req):
        """ROS service callback to generate dataset."""
        state_min = np.array(req.state_min)
        state_max = np.array(req.state_max)
        input_min = np.array(req.input_min)
        input_max = np.array(req.input_max)

        state_all = []
        input_all = []
        next_state_all = []
        for i in range(req.dataset_size):
            state = state_min + np.random.rand(len(state_min)) * (state_max - state_min)
            input = input_min + np.random.rand(len(input_min)) * (input_max - input_min)
            manip_force = np.array([input[0], 0.0, input[1]])
            self.setState(state)
            for i in range(int(req.dt / self.dt)):
                self.runOnce(manip_force)
            next_state = self.getState()
            state_all.append(state)
            input_all.append(input)
            next_state_all.append(next_state)

        msg = Dataset()
        msg.dataset_size = req.dataset_size
        msg.dt = req.dt
        msg.state_dim = len(state_min)
        msg.input_dim = len(input_min)
        msg.state_all = np.array(state_all).flatten()
        msg.input_all = np.array(input_all).flatten()
        msg.next_state_all = np.array(next_state_all).flatten()
        bag = rosbag.Bag(req.filename, "w")
        bag.write("/dataset", msg)
        bag.close()
        print("[SimTestMpcCart] Save dataset to {}".format(req.filename))

        res = GenerateDatasetResponse()
        return res


def demo():
    sim = SimTestMpcCart(True)
    sim.setState([0.3, 1.0, np.deg2rad(-10.0), 0.0])

    t = 0.0 # [sec]
    while sim.viewer.is_running():
        # Calculate manipulation force
        _, _, theta, theta_dot = sim.getState()
        manip_force_z = -500.0 * theta -100.0 * theta_dot # [N]
        manip_force = np.array([0.0, 0.0, manip_force_z])

        # Run simulation step
        sim.runOnce(manip_force)

        # Sleep and increment time
        time.sleep(sim.dt)
        t += sim.dt


if __name__ == "__main__":
    # demo()

    rospy.init_node("sim_test_mpc_cart")
    sim = SimTestMpcCart(enable_gui=rospy.get_param("~enable_gui", True))
    rospy.spin()
