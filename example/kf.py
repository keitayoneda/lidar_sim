#!/usr/bin/env python3
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pygame
import core
core.appendParendPath()
from simulator import simulator  # noqa


def dotKC(K, c):
    K0 = K.shape[0]
    c0 = c.shape[0]
    if (K0 == c0) and (K.shape == (K0,)) and (c.shape == (c0,)):
        ret = np.zeros((K0, c0))
        for i in range(K0):
            for j in range(c0):
                ret[i][j] = K[i]*c[j]
        return ret
    else:
        print("shape error")


np.random.seed(0)


def main():
    walls = simulator.Walls()
    walls.append([-200, -200], [200, -200])
    walls.append([200, -200], [200, 200])
    walls.append([200, 200], [-200, 200])
    walls.append([-200, 200], [-200, -200])
    wall_arr = walls.get()
    init_pos = np.array([0, 0, 0])
    num_lazer = 2
    u_limit = np.pi/2
    l_limit = 0

    sim = simulator.Simulator(wall_arr, init_pos, num_lazer, u_limit, l_limit)

    dt = 0.01
    p_stdv = 10
    r1, r2 = np.random.normal(0, p_stdv, 2)
    robot_pos = init_pos+np.array([r1, r2, 0])
    P = np.diag([p_stdv**2, p_stdv**2, 0])
    q_stdv = 0.5
    Q = np.diag([q_stdv**2, q_stdv**2, 0])
    r_stdv = 5
    C = [[-1, 0, 0], [0, -1, 0]]

    robot_pos_arr = []
    real_pos_arr = []
    count = 0
    max_frame = 1000
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break
        if count >= max_frame:
            break
        # 予測
        vx, vy, omega = 0, 10, 0  # 制御入力
        r1, r2 = np.random.normal(0, q_stdv, 2)
        sim.move(vx*dt+r1, vy*dt+r2, omega*dt)
        robot_pos += np.array([vx*dt, vy*dt, omega*dt])
        P = P+Q

        # 補正
        sensor_result = sim.sensor_measure()
        if count % 10 == 0:
            predict_result = sim.predict_measure(robot_pos)
            res = 0
            for i in range(num_lazer):
                r = np.random.normal(0, r_stdv, 1)
                if predict_result[i][0] and sensor_result[i][0]:
                    c = np.array(C[i])
                    s_yy = r_stdv**2 + np.dot(np.dot(c, P), c.T)
                    s_xy = np.dot(P, c.T)
                    K = s_xy / s_yy
                    res = sensor_result[i][3] + r - predict_result[i][3]
                    robot_pos = robot_pos + K*res
                    P = np.dot((np.eye(3) - dotKC(K, c)), P)

        robot_pos_arr.append(robot_pos)
        real_pos_arr.append(sim.getPos())
        sim.draw_init()
        sim.draw_wall()
        sim.draw_robot()
        sim.draw_lazer()
        x, y, _ = [np.sqrt(x) for x in np.diag(P)]
        sim.draw_predicted_pos(robot_pos, x, y)
        sim.draw_finish()
        count += 1

    t = [x for x in range(1000)]
    robot_x = [x[0] for x in robot_pos_arr]
    real_x = [x[0] for x in real_pos_arr]
    plt.plot(t, real_x, label="sim")
    plt.plot(t, robot_x, label="robot")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
