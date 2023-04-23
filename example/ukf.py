#!/usr/bin/env python3

import numpy as np
import pygame
import matplotlib.pyplot as plt
import core
core.appendParendPath()
from simulator import simulator  # noqa

np.random.seed(0)


def round_pi(th):
    while abs(th) > np.pi:
        if th > np.pi:
            th -= 2*np.pi
        else:
            th += 2*np.pi
    return th


def dot_to_mat(K, c):
    K0 = K.shape[0]
    c0 = c.shape[0]
    ret = np.zeros((K0, c0))
    for i in range(K0):
        for j in range(c0):
            ret[i][j] = K[i]*c[j]
    return ret


def genSigmaPoints(pos, n, Lambda, P):
    cP = np.linalg.cholesky(P)
    pos = pos.copy()
    ret = [pos]
    for i in range(cP.shape[0]):
        ret.append(pos + np.sqrt(n+Lambda)*cP[:, i])
        ret.append(pos - np.sqrt(n+Lambda)*cP[:, i])
    return ret


def preceed(x, u):
    return x + u


def genRandomNoiseNormal(stdv_arr):
    noise = [np.random.normal(0, stdv, 1)[0] for stdv in stdv_arr]
    return noise


def print_sigma(sigma):
    for point in sigma:
        print("s:", point)


def predict(x, P, u, Q, Lambda, w_m, w_c):
    """
    予測を行う
    """
    cP = np.linalg.cholesky(P)
    dim_x = x.shape[0]
    sigma_points = genSigmaPoints(x, dim_x, Lambda, cP)
    pred_x = [preceed(points, u) for points in sigma_points]

    expected_x = np.zeros(dim_x)
    for i in range(2*dim_x+1):
        expected_x = expected_x + w_m(i)*pred_x[i]

    P = Q.copy()
    for i in range(2*dim_x+1):
        vec = pred_x[i] - expected_x
        P += w_c(i)*dot_to_mat(vec, vec)

    next_x = expected_x.copy()
    next_x[2] = round_pi(next_x[2])
    return next_x, P


def discardUnusedData(all_prediction, is_usable):
    """
    すべてのデータにおいて条件の良いセンサーだけを使うようにする。
    絞り込んだセンサーのみの予測データを取り出して返す
    all_prediction:すべての予測値を入れたリスト
    usable_data:どのセンサーが使えるかの真偽値のリスト
    """
    observed_element_num = len(all_prediction[0])
    usable_data = []
    for one_pred in all_prediction:
        for i in range(observed_element_num):
            if not one_pred[0]:
                is_usable[i] = False

    usable_data_index = []  # iが要素であるとき、i番目のセンサーが有効であることを示す
    for i in range(observed_element_num):
        if is_usable[i]:
            usable_data_index.append(i)
    for one_pred in all_prediction:
        usable_result = []
        for i in usable_data_index:
            usable_result.append(one_pred[i][3])
        usable_data.append(np.array(usable_result))
    return usable_data, usable_data_index


def formatData(sensor_result, sigma_points, sim):
    is_usable = [result[0] for result in sensor_result]
    all_prediction = []

    for point in sigma_points:
        all_prediction.append(sim.predict_measure(point))
    usable_pred_data, usable_pred_data_index = discardUnusedData(
        all_prediction, is_usable)
    return usable_pred_data, usable_pred_data_index


def calcMeanObservedVec(usable_pred_data, w_m):
    usable_odata_num = len(usable_pred_data[0])
    mean_lazer = np.zeros(usable_odata_num)
    for i, pred_data in enumerate(usable_pred_data):
        for j, pred_sensor_data in enumerate(pred_data):
            mean_lazer[j] += w_m(i) * pred_sensor_data
    return mean_lazer


def calcRealObservedVec(sensor_result, usable_pred_data_index, r_stdv):
    usable_odata_num = len(usable_pred_data_index)
    observed_vec = np.zeros(usable_odata_num, dtype="float64")
    for i in range(usable_odata_num):
        r = np.random.normal(0, r_stdv, 1)
        observed_vec[i] = sensor_result[usable_pred_data_index[i]][3]+r
    return observed_vec


def calcSyy(R, dim_x, usable_pred_data, mean_lazer, w_c):
    s_yy = R.copy()
    for i in range(2*dim_x+1):
        diff_in_prediction = usable_pred_data[i] - mean_lazer
        s_yy = s_yy + w_c(i)*dot_to_mat(diff_in_prediction, diff_in_prediction)
    return s_yy


def calcSxy(robot_pos, sigma_points, usable_pred_data, mean_lazer, w_c):
    dim_x = robot_pos.shape[0]
    usable_odata_num = len(usable_pred_data[0])
    s_xy = np.zeros((dim_x, usable_odata_num))
    for i in range(2*dim_x+1):
        diff_in_prediction = usable_pred_data[i] - mean_lazer
        s_xy = s_xy + \
            w_c(i)*dot_to_mat((sigma_points[i]-robot_pos), diff_in_prediction)
    return s_xy


def correct(robot_pos, P, sim, r_stdv, Lambda, w_m, w_c):
    """
    補正を行う
    """
    dim_x = robot_pos.shape[0]
    sigma_points = genSigmaPoints(robot_pos, robot_pos.shape[0], Lambda, P)
    sensor_result = sim.sensor_measure()
    usable_pred_data, usable_pred_data_index = formatData(
        sensor_result, sigma_points, sim)

    usable_odata_num = len(usable_pred_data_index)
    R = np.diag([r_stdv**2 for _ in range(usable_odata_num)])  # 観測ノイズの共分散行列

    mean_lazer = calcMeanObservedVec(usable_pred_data, w_m)
    observed_vec = calcRealObservedVec(
        sensor_result, usable_pred_data_index, r_stdv)

    s_yy = calcSyy(R, dim_x, usable_pred_data, mean_lazer, w_c)
    s_xy = calcSxy(robot_pos, sigma_points, usable_pred_data, mean_lazer, w_c)

    K = np.dot(s_xy, np.linalg.inv(s_yy))
    robot_pos = robot_pos + np.dot(K, (observed_vec - mean_lazer))
    P = P - np.dot(np.dot(K, s_yy), K.T)

    return robot_pos, P


def init_wall():
    walls = simulator.Walls()
    walls.append([-200, -200], [200, -200])
    walls.append([200, -200], [200, 200])
    walls.append([200, 200], [-200, 200])
    walls.append([-200, -200], [-200, 200])
    return walls.get()


def main():
    wall_arr = init_wall()

    init_pos = np.array([0, 0, 0], dtype="float64")
    num_lazer = 4
    u_limit = np.pi/2
    l_limit = 0

    sim = simulator.Simulator(wall_arr, init_pos, num_lazer, u_limit, l_limit)

    dim_x = init_pos.shape[0]
    alpha, beta, kappa = 1, 2, 0
    Lambda = alpha**2*(dim_x+kappa) - dim_x

    dt = 0.01
    p_stdv1, p_stdv2 = 10, 0.01
    q_stdv1, q_stdv2 = 0.5, 0.05
    r_stdv = 5
    P = np.diag([p_stdv1**2, p_stdv1**2, p_stdv2**2])
    Q = np.diag([q_stdv1**2, q_stdv1**2, q_stdv2**2])

    r1, r2, r3 = genRandomNoiseNormal([p_stdv1, p_stdv1, p_stdv2])
    robot_pos = init_pos+np.array([r1, r2, r3], dtype="float64")

    def w_m(i): return Lambda/(dim_x+Lambda) if i == 0 else 1/(2*dim_x+Lambda)

    def w_c(i): return Lambda/(dim_x+Lambda) + \
        (1-alpha**2+beta) if i == 0 else 1/(2*(dim_x+Lambda))

    robot_pos_arr = []
    real_pos_arr = []
    count, max_frame = 0, 1000
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break
        if count >= max_frame:
            break

        vx, vy, omega = -5, 5, 0.1  # 制御入力
        vel = np.array([vx, vy, omega])  # 制御入力
        r1, r2, r3 = genRandomNoiseNormal(
            [q_stdv1, q_stdv1, q_stdv2])  # 状態変数に乗るノイズ
        sim.move(vx*dt+r1, vy*dt+r2, omega*dt+r3)

        # 予測
        robot_pos, P = predict(robot_pos, P, vel*dt, Q, Lambda, w_m, w_c)

        # 補正
        robot_pos, P = correct(robot_pos, P, sim, r_stdv, Lambda, w_m, w_c)

        robot_pos_arr.append(robot_pos)
        real_pos_arr.append(sim.getPos())
        sim.draw_init()
        sim.draw_wall()
        sim.draw_robot()
        sim.draw_lazer()
        x, y, _ = [np.sqrt(abs(x)) for x in np.diag(P)]
        sim.draw_predicted_pos(robot_pos, x, y)
        sim.draw_finish()
        count += 1

    pygame.quit()

    t = [x for x in range(max_frame)]
    robot_pos_arr = np.array(robot_pos_arr)
    real_pos_arr = np.array(real_pos_arr)
    plot_raw = 0
    fig = plt.figure(figsize=(10, 4))
    ax_x = fig.add_subplot(1, 3, 1)
    ax_x.set_title("x")
    ax_y = fig.add_subplot(1, 3, 2)
    ax_y.set_title("y")
    ax_th = fig.add_subplot(1, 3, 3)
    ax_th.set_title("th")

    axis_list = [ax_x, ax_y, ax_th]
    for plot_raw, ax in enumerate(axis_list):
        ax.plot(t, real_pos_arr[:, plot_raw], label="sim")
        ax.plot(t, robot_pos_arr[:, plot_raw], label="robot")
        ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
