#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


class PlotTestMpcLocomanipPush(object):
    def __init__(self, result_file_path):
        self.result_data_list = np.genfromtxt(result_file_path, dtype=None, delimiter=None, names=True)
        print("[PlotTestMpcLocomanipPush] Load {}".format(result_file_path))

        fig = plt.figure()
        plt.rcParams["font.size"] = 16

        ax = fig.add_subplot(211)
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

        ax = fig.add_subplot(212)
        ax.plot(self.result_data_list["time"], self.result_data_list["obj_force"],
                color="green", label="obj force")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("force [N]")
        ax.grid()
        ax.legend(loc="upper left")

        plt.show()


if __name__ == "__main__":
    result_file_path = "/tmp/TestMpcLocomanipPushResult.txt"

    import sys
    if len(sys.argv) >= 2:
        result_file_path = sys.argv[1]

    plot = PlotTestMpcLocomanipPush(result_file_path)
