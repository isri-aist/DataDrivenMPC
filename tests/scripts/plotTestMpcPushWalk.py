#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


class PlotTestMpcPushWalk(object):
    def __init__(self, result_file_path):
        self.result_data_list = np.genfromtxt(result_file_path, dtype=None, delimiter=None, names=True)
        print("[PlotTestMpcPushWalk] Load {}".format(result_file_path))

        fig = plt.figure()
        plt.rcParams["font.size"] = 16

        ax = fig.add_subplot(311)
        ax.plot(self.result_data_list["time"], self.result_data_list["robot_com_pos"],
                color="green", label="planned robot pos")
        ax.plot(self.result_data_list["time"], self.result_data_list["obj_com_pos"],
                color="coral", label="planned obj pos")
        ax.plot(self.result_data_list["time"], self.result_data_list["robot_zmp"],
                color="red", label="planned robot zmp")
        ax.plot(self.result_data_list["time"], self.result_data_list["ref_obj_com_pos"],
                color="cyan", linestyle="dashed", label="ref obj pos", zorder=-1)
        ax.plot(self.result_data_list["time"], self.result_data_list["ref_robot_zmp"],
                color="blue", linestyle="dashed", label="ref robot zmp", zorder=-1)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("pos [m]")
        ax.grid()
        ax.legend(loc="upper left")

        ax = fig.add_subplot(312)
        ax.plot(self.result_data_list["time"], self.result_data_list["obj_force"],
                color="green", label="obj force")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("force [N]")
        ax.grid()
        ax.legend(loc="upper right")

        mass = 50.0 # [kg]
        damper_forces = mass * (self.result_data_list["obj_com_vel"][1:] - self.result_data_list["obj_com_vel"][:-1]) / \
                        (self.result_data_list["time"][1:] - self.result_data_list["time"][:-1]) \
                        - self.result_data_list["obj_force"][:-1]
        ax = fig.add_subplot(313)
        ax.scatter(self.result_data_list["obj_com_vel"][:-1], damper_forces,
                   color="green", label="damper force")
        ax.set_xlabel("vel [m/s]")
        ax.set_ylabel("force [N]")
        ax.grid()
        ax.legend(loc="upper right")

        plt.show()


if __name__ == "__main__":
    result_file_path = "/tmp/TestMpcPushWalkResult-Linear.txt"

    import sys
    if len(sys.argv) >= 2:
        result_file_path = sys.argv[1]

    plot = PlotTestMpcPushWalk(result_file_path)
