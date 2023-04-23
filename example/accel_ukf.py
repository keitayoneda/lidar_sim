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
    """
    (n,1) * (1,m) -> (n,m)という行列の掛け算を行う
    np.matrixを使いたくなかったので
    """
    K0 = K.shape[0]
    c0 = c.shape[0]
    ret = np.zeros((K0, c0))
    for i in range(K0):
        for j in range(c0):
            ret[i][j] = K[i]*c[j]
    return ret


def genSigmaPoints(x, Lambda, P):
    """
    シグマ点を生成する
    """
    cP = np.linalg.cholesky(P)
    n = x.shape[0]
    x0 = x.copy()
    ret = [x0]
    for i in range(n):
        ret.append(x0 + np.sqrt(n+Lambda)*cP[:, i])
        ret.append(x0 - np.sqrt(n+Lambda)*cP[:, i])
    return ret


def preceed(x, u, dt):
    """
    状態を遷移させる
    """
    pos, vel = x[0:3], x[3:6]
    # pos = pos + vel*dt + u*dt**2/2
    pos = pos + vel*dt
    vel = vel + u*dt
    next_x = np.concatenate([pos, vel])
    return next_x


def genRandomNoiseNormal(stdv_arr):
    """
    平均0,標準偏差stdvの正規分布に従う乱数を生成する(リストでうけとる)
    """
    noise = [np.random.normal(0, stdv, 1)[0] for stdv in stdv_arr]
    return noise


def print_sigma(sigma):
    """
    シグマ点を出力する.デバッグ用
    """
    for point in sigma:
        print("s:", point)


def predict(x, P, u, dt, Q, Lambda, w_m, w_c):
    """
    予測を行う
    """
    cP = np.linalg.cholesky(P)
    dim_x = x.shape[0]
    sigma_points = genSigmaPoints(x, Lambda, cP)
    pred_x = [preceed(points, u, dt) for points in sigma_points]

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
    すべてのデータにおいて距離を読み取れているセンサーだけを使うようにする。
    絞り込んだセンサーのみの予測データを取り出して返す
    """
    observed_element_num = len(all_prediction[0])
    usable_lazer_data = []
    for one_pred in all_prediction:
        for i in range(observed_element_num):
            if not one_pred[0]:
                is_usable[i] = False

    usable_lazer_data_index = []  # iが要素であるとき、i番目のセンサーが有効であることを示す
    for i in range(observed_element_num):
        if is_usable[i]:
            usable_lazer_data_index.append(i)
    for one_pred in all_prediction:
        usable_result = []
        for i in usable_lazer_data_index:
            usable_result.append(one_pred[i][3])
        usable_lazer_data.append(np.array(usable_result))
    return usable_lazer_data, usable_lazer_data_index


def formatLazerData(sensor_result, sigma_points, sim):
    """
    シグマ点から予測されるセンサの値と実際の観測値から使えるセンサのデータだけを取り出す
    """
    is_usable = [result[0] for result in sensor_result]
    all_prediction = []

    for point in sigma_points:
        all_prediction.append(sim.predict_measure(point[0:3]))
    usable_pred_data, usable_pred_data_index = discardUnusedData(
        all_prediction, is_usable)
    return usable_pred_data, usable_pred_data_index


def integratePredData(usable_pred_lazer, pred_vel):
    """
    レーザの予測データとエンコーダから推定される速度のデータをまとめて一つの観測ベクトルにする
    """
    n = len(usable_pred_lazer)
    pred_observed_list = np.zeros(
        (n, usable_pred_lazer[0].shape[0] + pred_vel[0].shape[0]))
    for i in range(n):
        pred_observed_list[i] = np.concatenate(
            (usable_pred_lazer[i], pred_vel[i]))
    return pred_observed_list


def calcMeanObservedVec(pred_observed_list, w_m):
    """
    観測ベクトルの重み付き平均値を求める
    """
    pred_observed_mean = np.zeros(pred_observed_list[0].shape[0])
    for i, pred_data_each in enumerate(pred_observed_list):
        pred_observed_mean = pred_observed_mean + w_m(i) * pred_data_each
    return pred_observed_mean


def calcRealObservedVec(real_lazer_result, usable_pred_lazer_index, real_vel_result, r_stdv):
    """
    実際の観測データから使えないセンサのデータをのぞいた観測ベクトルを作る
    """
    usable_odata_num = len(usable_pred_lazer_index)
    observed_vec = np.zeros(usable_odata_num+3, dtype="float64")
    for i in range(usable_odata_num):
        r = np.random.normal(0, r_stdv, 1)
        observed_vec[i] = real_lazer_result[usable_pred_lazer_index[i]][3]+r
    for i in range(3):
        observed_vec[i+usable_odata_num] = real_vel_result[i]
    return observed_vec


def calcSyy(R, pred_observed_list, pred_observed_mean, w_c):
    """
    観測量の共分散行列s_yyを求める
    """
    s_yy = R.copy()
    for i in range(len(pred_observed_list)):
        diff_in_prediction = pred_observed_list[i] - pred_observed_mean
        s_yy = s_yy + w_c(i)*dot_to_mat(diff_in_prediction, diff_in_prediction)
    return s_yy


def calcSxy(robot_pos, sigma_points, usable_pred_data, mean_lazer, w_c):
    """
    状態量と観測量の相関行列を求める
    """
    dim_x = robot_pos.shape[0]
    usable_odata_num = len(usable_pred_data[0])
    s_xy = np.zeros((dim_x, usable_odata_num))
    for i in range(2*dim_x+1):
        diff_in_prediction = usable_pred_data[i] - mean_lazer
        s_xy = s_xy + \
            w_c(i)*dot_to_mat((sigma_points[i]-robot_pos), diff_in_prediction)
    return s_xy


def correct(robot_state, P, sim, r_stdv, Lambda, w_m, w_c):
    """
    補正を行う
    """
    sigma_points = genSigmaPoints(robot_state, Lambda, P)
    real_lazer_result = sim.sensor_measure()
    real_vel_result = sim.getVel()

    usable_pred_lazer, usable_pred_lazer_index = formatLazerData(
        real_lazer_result, sigma_points, sim)
    pred_vel = [point[3:6] for point in sigma_points]

    pred_observed_list = integratePredData(usable_pred_lazer, pred_vel)

    usable_odata_num = len(usable_pred_lazer_index)
    R = np.diag([r_stdv**2 for _ in range(usable_odata_num)] +
                [5, 5, 1])  # 観測ノイズの共分散行列

    pred_observed_mean = calcMeanObservedVec(pred_observed_list, w_m)
    real_observed_vec = calcRealObservedVec(
        real_lazer_result, usable_pred_lazer_index, real_vel_result, r_stdv)

    s_yy = calcSyy(R, pred_observed_list, pred_observed_mean, w_c)
    s_xy = calcSxy(robot_state, sigma_points,
                   pred_observed_list, pred_observed_mean, w_c)

    K = np.dot(s_xy, np.linalg.inv(s_yy))

    robot_state = robot_state + \
        np.dot(K, (real_observed_vec - pred_observed_mean))
    P = P - np.dot(np.dot(K, s_yy), K.T)

    return robot_state, P


def init_wall():
    """
    壁の配置を設定する
    """
    walls = simulator.Walls()
    walls.append([-200, -200], [200, -200])
    walls.append([200, -200], [200, 200])
    walls.append([200, 200], [-200, 200])
    walls.append([-200, -200], [-200, 200])
    return walls.get()


def main():
    wall_arr = init_wall()

    """
    シミュレータの設定
    """
    init_state = np.array([0, 0, 0, 0, 0, 0], dtype="float64")
    num_lazer = 3
    u_limit = np.pi/2
    l_limit = 0

    sim = simulator.accelSimulator(
        wall_arr, init_state, num_lazer, u_limit, l_limit)

    """
    UKFに必要なパラメータの設定
    """
    dim_x = init_state.shape[0]
    alpha, beta, kappa = 0.9, 2, -3
    Lambda = alpha**2*(dim_x+kappa) - dim_x

    dt = 0.01
    p_stdv_xy, p_stdv_th, p_stdv_vxy, p_stdv_omega = 5, 0.01, 0.01, 0.01  # 初期座標の分散
    q_stdv_xy, q_stdv_th, q_stdv_vxy, q_stdv_omega = 0.01, 0.01, 0.05001, 0.01  # 制御誤差の分散
    r_stdv = 20  # 観測誤差の分散
    P = np.diag([p_stdv_xy**2, p_stdv_xy**2, p_stdv_th**2,
                p_stdv_vxy**2, p_stdv_vxy**2, p_stdv_omega**2])
    Q = np.diag([q_stdv_xy**2, q_stdv_xy**2, q_stdv_th**2,
                q_stdv_vxy**2, q_stdv_vxy**2, q_stdv_omega**2])

    rx, ry, rth, rvx, rvy, romega = genRandomNoiseNormal(
        [p_stdv_xy, p_stdv_xy, p_stdv_th, p_stdv_vxy, p_stdv_vxy, p_stdv_omega])
    robot_state = init_state + \
        np.array([rx, ry, rth, rvx, rvy, romega], dtype="float64")

    def w_m(i): return Lambda/(dim_x+Lambda) if i == 0 else 1/(2*dim_x+Lambda)

    def w_c(i): return Lambda/(dim_x+Lambda) + \
        (1-alpha**2+beta) if i == 0 else 1/(2*(dim_x+Lambda))

    robot_state_arr = []
    real_state_arr = []
    count, max_frame = 0, 300

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break
        if count >= max_frame:
            break

        ax, ay, ath = -1.0, 0.0, 0.0  # 制御入力
        u = np.array([ax, ay, ath])  # 制御入力
        q_noise = genRandomNoiseNormal(
            [q_stdv_xy, q_stdv_xy, q_stdv_th, q_stdv_vxy, q_stdv_vxy, q_stdv_omega])  # 状態変数に乗るノイズ
        sim.accel(ax, ay, ath, dt, q_noise)

        # 予測
        P_cp = P.copy()
        try:
            robot_state, P = predict(
                robot_state, P, u, dt, Q, Lambda, w_m, w_c)
        except np.linalg.LinAlgError:
            print("P:", P_cp)
            raise RuntimeError('count', count, '予測段')

        # 補正
        P_cp = P.copy()
        try:
            robot_state, P = correct(
                robot_state, P, sim, r_stdv, Lambda, w_m, w_c)
        except np.linalg.LinAlgError:
            print("P:", P_cp)
            raise RuntimeError('count', count, "補正段")

        robot_state_arr.append(robot_state)
        real_state_arr.append(sim.getState())
        sim.draw_init()
        sim.draw_wall()
        sim.draw_robot()
        sim.draw_lazer()
        x, y, _, _, _, _ = [np.sqrt(abs(x)) for x in np.diag(P)]
        sim.draw_predicted_pos(robot_state[0:3], x, y)
        sim.draw_finish()
        count += 1

    pygame.quit()

    """
    結果の描画
    """
    t = [x for x in range(max_frame)]
    robot_state_arr = np.array(robot_state_arr)
    real_state_arr = np.array(real_state_arr)
    plot_raw = 0
    fig = plt.figure(figsize=(10, 6))
    ax_x = fig.add_subplot(2, 3, 1)
    ax_x.set_title("x")
    ax_y = fig.add_subplot(2, 3, 2)
    ax_y.set_title("y")
    ax_th = fig.add_subplot(2, 3, 3)
    ax_th.set_title("th")
    ax_vx = fig.add_subplot(2, 3, 4)
    ax_vx.set_title("vx")
    ax_vy = fig.add_subplot(2, 3, 5)
    ax_vy.set_title("vy")
    ax_vth = fig.add_subplot(2, 3, 6)
    ax_vth.set_title("vth")

    axis_list = [ax_x, ax_y, ax_th, ax_vx, ax_vy, ax_vth]
    for plot_raw, ax in enumerate(axis_list):
        ax.plot(t, real_state_arr[:, plot_raw], label="sim")
        ax.plot(t, robot_state_arr[:, plot_raw], label="robot")
        ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
