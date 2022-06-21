#! /usr/bin/env python

import time
import numpy as np
import eigen as e
import pybullet
import pybullet_data
import matplotlib.pyplot as plt

import rospy
import rosbag
from data_driven_mpc.msg import *
from data_driven_mpc.srv import *


class SimTestMpcCart(object):
    def __init__(self):
        # Instantiate simulator
        pybullet.connect(pybullet.GUI)
        # pybullet.connect(pybullet.DIRECT)

        # Set simulation parameters
        self.dt = 0.005 # [sec]
        pybullet.setTimeStep(self.dt)
        pybullet.setGravity(0, 0, -9.8) # [m/s^2]

        # Set debug parameters
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)

        # Setup models
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())

        ## Setup floor
        pybullet.loadURDF("plane100.urdf")

        ## Setup cart
        self.box_half_scale = np.array([0.35, 0.25, 0.15]) # [m]
        box_col_shape_idx = pybullet.createCollisionShape(pybullet.GEOM_BOX,
                                                          halfExtents=self.box_half_scale)
        self.cylinder_radius = 0.1 # [m]
        cylinder_height = 0.1 # [m]
        cylinder_col_shape_idx = pybullet.createCollisionShape(pybullet.GEOM_CYLINDER,
                                                               radius=self.cylinder_radius,
                                                               height=cylinder_height)
        box_mass = 10.0 # [kg]
        self.box_com_offset = np.array([0.0, 0.0, -0.1]) # [m]
        cylinder_mass = 1.0 # [kg]
        self.cart_body_uid = pybullet.createMultiBody(baseMass=box_mass,
                                                      baseCollisionShapeIndex=box_col_shape_idx,
                                                      baseVisualShapeIndex=-1,
                                                      basePosition=[0.0, 0.0, 2 * self.cylinder_radius + self.box_half_scale[2]], # [m]
                                                      baseOrientation=[0.0, 0.0, 0.0, 1.0],
                                                      baseInertialFramePosition=self.box_com_offset,
                                                      baseInertialFrameOrientation=[0.0, 0.0, 0.0, 1.0],
                                                      linkMasses=[cylinder_mass],
                                                      linkCollisionShapeIndices=[cylinder_col_shape_idx],
                                                      linkVisualShapeIndices=[-1],
                                                      linkPositions=[[0.0, 0.0, -1 * (self.cylinder_radius + self.box_half_scale[2])]], # [m]
                                                      linkOrientations=[pybullet.getQuaternionFromEuler([np.pi/2, 0.0, 0.0])],
                                                      linkInertialFramePositions=[[0.0, 0.0, 0.0]], # [m]
                                                      linkInertialFrameOrientations=[[0.0, 0.0, 0.0, 1.0]],
                                                      linkParentIndices=[0],
                                                      linkJointTypes=[pybullet.JOINT_FIXED],
                                                      linkJointAxis=[[0.0, 0.0, 1.0]])
        pybullet.changeVisualShape(objectUniqueId=self.cart_body_uid,
                                   linkIndex=-1,
                                   rgbaColor=[0.0, 1.0, 0.0, 0.8])
        pybullet.changeVisualShape(objectUniqueId=self.cart_body_uid,
                                   linkIndex=0,
                                   rgbaColor=[0.1, 0.1, 0.1, 0.8])

        # Set dynamics parameters
        pybullet.changeDynamics(bodyUniqueId=self.cart_body_uid, linkIndex=0, lateralFriction=0.05)

        # Setup variables
        self.force_line_uid = -1

        # Setup ROS
        run_sim_once_srv = rospy.Service("/run_sim_once", RunSimOnce, self.runSimOnceCallback)
        generate_dataset_srv = rospy.Service("/generate_dataset", GenerateDataset, self.generateDatasetCallback)

    def runOnce(self, manip_force=None):
        """"Run simulation step once.

        Args:
            manip_force manipulation force in world frame
        """
        # Process simulation step
        pybullet.stepSimulation()

        if manip_force is not None:
            # Apply manipulation force
            box_link_pos, box_link_rot = pybullet.getBasePositionAndOrientation(bodyUniqueId=self.cart_body_uid)
            box_link_pos = np.array(box_link_pos)
            box_link_rot = np.array(pybullet.getMatrixFromQuaternion(box_link_rot)).reshape((3, 3))
            manip_pos_local = np.array([-1 * self.box_half_scale[0], 0.0, self.box_half_scale[2]]) - self.box_com_offset
            manip_pos = box_link_pos + box_link_rot.dot(manip_pos_local)
            pybullet.applyExternalForce(objectUniqueId=self.cart_body_uid,
                                        linkIndex=0,
                                        forceObj=manip_force,
                                        posObj=manip_pos,
                                        flags=pybullet.WORLD_FRAME)

            # Visualize external force
            force_scale = 0.01
            self.force_line_uid = pybullet.addUserDebugLine(lineFromXYZ=manip_pos,
                                                            lineToXYZ=manip_pos + force_scale * manip_force,
                                                            lineColorRGB=[1, 0, 0],
                                                            lineWidth=5.0,
                                                            replaceItemUniqueId=self.force_line_uid)
        else:
            # Delete external force
            if self.force_line_uid != -1:
                pybullet.removeUserDebugItem(self.force_line_uid)
                self.force_line_uid = -1

    def getState(self):
        """"Get state [p, p_dot, theta, theta_dot]."""
        cylinder_link_state = pybullet.getLinkState(bodyUniqueId=self.cart_body_uid, linkIndex=0, computeLinkVelocity=True)
        p = cylinder_link_state[4][0] # [m]
        p_dot = cylinder_link_state[6][0] # [m/s]
        theta = pybullet.getEulerFromQuaternion(
            pybullet.getBasePositionAndOrientation(bodyUniqueId=self.cart_body_uid)[1])[1] # [rad]
        theta_dot = pybullet.getBaseVelocity(bodyUniqueId=self.cart_body_uid)[1][1] # [rad/s]
        return np.array([p, p_dot, theta, theta_dot])

    def setState(self, state):
        """Set state [p, p_dot, theta, theta_dot]."""
        p, p_dot, theta, theta_dot = state
        local_pos_from_cylinder_to_box = np.array(
            [self.box_com_offset[0], 0.0, self.cylinder_radius + self.box_half_scale[2] + self.box_com_offset[2]])
        global_pos_from_cylinder_to_box = np.array(
            e.AngleAxisd(theta, e.Vector3d.UnitY()).toRotationMatrix()).dot(local_pos_from_cylinder_to_box)
        box_pos = np.array([p, 0.0, self.cylinder_radius]) + global_pos_from_cylinder_to_box
        box_rot = pybullet.getQuaternionFromEuler([0.0, theta, 0.0])
        pybullet.resetBasePositionAndOrientation(bodyUniqueId=self.cart_body_uid,
                                                 posObj=box_pos,
                                                 ornObj=box_rot)
        linear_vel = np.array([p_dot, 0.0, 0.0]) + \
                     theta_dot * np.array(e.Vector3d.UnitY().cross(e.Vector3d(global_pos_from_cylinder_to_box))).flatten()
        angular_vel = np.array([0.0, theta_dot, 0.0])
        pybullet.resetBaseVelocity(objectUniqueId=self.cart_body_uid,
                                   linearVelocity=linear_vel,
                                   angularVelocity=angular_vel)

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
    sim = SimTestMpcCart()
    sim.setState([0.3, 1.0, np.deg2rad(-10.0), 0.0])

    t = 0.0 # [sec]
    while pybullet.isConnected():
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
    sim = SimTestMpcCart()
    rospy.spin()
