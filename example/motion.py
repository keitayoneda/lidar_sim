import numpy as np
from typing import List
import matplotlib.pyplot as plt


class AbstMotion:
    def __init__(self, init_pos: np.ndarray):
        self.pos = init_pos


##
# @brief 時刻tを与えるとそれに応じた位置をかえす
class PosMotion(AbstMotion):
    def __init__(self, init_pos: np.ndarray, *pos_funcs, **kwarg):
        super().__init__(init_pos)
        if (len(pos_funcs) != 3):
            raise RuntimeError(
                "pos_funcs must include 3 functions. (x, y, th)")
        self.x_func = pos_funcs[0]
        self.y_func = pos_funcs[1]
        self.th_func = pos_funcs[2]
        self.pos_stdev: List[float] = [0, 0, 0]
        if "pos_stdev" in kwarg:
            # add noise
            if type(kwarg["pos_stdev"]) == tuple or type(kwarg["pos_stdev"]) == list:
                if len(kwarg) == 2:
                    self.pos_stdev = [kwarg["pos_stdev"][0],
                                      kwarg["pos_stdev"][0], kwarg["pos_stdev"][1]]
                elif len(kwarg["pos_stdev"]) == 3:
                    self.pos_stdev = kwarg["pos_stdev"]
            else:
                raise RuntimeError(
                    "pos_stdev must have 2 or 3 elements. (x, y) or (x, y, th)")

    def __call__(self, time: float):
        noise = np.array([np.random.normal(0.0, pos_stdev)
                          for pos_stdev in self.pos_stdev])
        pos = np.array(
            [self.x_func(time), self.y_func(time), self.th_func(time)]) + noise
        return pos


##
# @brief 時刻tを与えると位置と速度をかえす
class PosVelMotion(PosMotion):
    def __init__(self, init_pos: np.ndarray, *pos_vel_funcs, **kwarg):
        super().__init__(init_pos, *pos_vel_funcs[-3], **kwarg)
        if (len(pos_vel_funcs[3:]) != 3):
            raise RuntimeError(
                "vel_funcs must include 3 functions. (vx, vy, vth)")
        self.vx_func = pos_vel_funcs[4]
        self.vy_func = pos_vel_funcs[5]
        self.vth_func = pos_vel_funcs[6]
        if "vel_stdev" in kwarg:
            # add noise
            if type(kwarg["vel_stdev"]) == tuple or type(kwarg["vel_stdev"]) == list:
                if len(kwarg) == 2:
                    self.vel_stdev = [kwarg["vel_stdev"][0],
                                      kwarg["vel_stdev"][0], kwarg["vel_stdev"][1]]
                elif len(kwarg["vel_stdev"]) == 3:
                    self.vel_stdev = kwarg["vel_stdev"]
            else:
                raise RuntimeError(
                    "vel_stdev must have 2 or 3 elements. (x, y) or (x, y, th)")

    def __call__(self, time: float):
        pos = super().__call__(time)
        vel_noise = np.array([np.random.normal(0.0, vel_stdev)
                              for vel_stdev in self.vel_stdev])
        vel = np.array(
            [self.vx_func(time), self.vy_func(time), self.vth_func(time)]) + vel_noise
        return pos, vel


if __name__ == "__main__":
    def x_func(t): return np.sin(3*np.pi*t)
    def y_func(t): return np.cos(2*np.pi*t)
    def th_func(t): return np.pi*t
    funcs = [x_func, y_func, th_func]
    m = PosMotion(np.array([0, 0, 0]),   *funcs, pos_stdev=(0.1, 0.1, 0.1))
    t = np.linspace(0, 2, 100)
    pos_vel = np.array([m(ti) for ti in t])
    x, y, th = pos_vel[:, 0], pos_vel[:, 1], pos_vel[:, 2]
    plt.plot(x, y)
    plt.show()
