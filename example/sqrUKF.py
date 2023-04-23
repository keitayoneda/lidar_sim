#!/usr/bin/env python3

import numpy as np
import pygame
import matplotlib.pyplot as plt
import core
core.appendParendPath()
from simulator import simulator  # noqa

np.random.seed(0)


def genZeroVec(n, m=1):
    """
    要素数nの0ベクトル(縦)を返す
    """
    return np.matrix(np.zeros((n, m)))


def cholupdate(R, x, coef=1):
    """
    cholupdateを実装する
    """
    if type(x) != np.matrix:
        raise TypeError("x must be np.matrix")
    elif type(R) != np.matrix:
        raise TypeError("R must be np.matrix")
    elif R.shape[0] != R.shape[1]:
        raise RuntimeError("shape error: R must be square matirx")
    elif R.shape[0] != x.shape[0]:
        raise RuntimeError("shape error: unmatch shape")

    return np.linalg.cholesky(R+coef*x*x.T)


def dot(x, y):
    """
    np.matrixの掛け算を行う
    """
    if x.shape[1] != y.shape[0]:
        raise RuntimeError("shape error")
    else:
        return x*y


class sqrUKF:
    def __init__(self, init_state, init_covar, updateFunc, observeFunc):
        if type(init_state) != np.matrix:
            raise TypeError("expected:np.matrix, given", type(init_state))
        elif type(init_covar) != np.matrix:
            raise TypeError("expected:np.matrix, given", type(init_covar))
        elif init_state.shape[1] != 1:
            raise RuntimeError("state col must be 1, given",
                               init_state.shape[1])
        elif init_state.shape[0] != init_covar.shape[0]:
            raise RuntimeError(
                "shape error: init_state.shape[0] != init_covar.shape[0]")
        elif init_covar.shape[0] != init_covar.shape[1]:
            raise RuntimeError("shape error: init_cavar is not square matrix")

        self.dim_state = init_state.shape[0]
        self.state = init_state
        try:
            self.state_sr_covar = np.linalg.cholesky(init_covar)
        except np.linalg.LinAlgError:
            raise RuntimeError("正定値じゃないよ")
        self.updateFunc = updateFunc
        self.observeFunc = observeFunc
        self.alpha = 1
        self.beta = 2
        self.kappa = self.dim_state*(self.alpha**2-1)
        self.w_m0 = self.kappa / (self.dim_state + self.kappa)
        self.w_c0 = self.kappa / \
            (self.dim_state + self.kappa) + (1-self.alpha**2 + self.beta)
        self.w_m1 = 1 / (2*(self.dim_state + self.kappa))

    def genSigmaPoints(self):
        """
        シグマ点を生成する
        """
        sigmaPoints = [self.state.copy()]
        for i in range(self.dim_state):
            sigmaPoints.append(self.state.copy(
            ) + np.sqrt(self.dim_state+self.kappa) * self.state_sr_covar[:, i])
            sigmaPoints.append(self.state.copy(
            ) - np.sqrt(self.dim_state+self.kappa) * self.state_sr_covar[:, i])
        return sigmaPoints

    def unscentedTransform(self, sigma_points, u, sr_Q):
        """
        updateFuncによって遷移した後の状態とその平均、共分散行列の平方根を求める
        """
        updated_points = [self.updateFunc(sigma_points[0], u)]
        updated_mean = self.w_m0 * updated_points[0]
        for point in sigma_points[1:]:
            updated_point = self.updateFunc(point, u)
            updated_points.append(updated_point)
            updated_mean = updated_mean + self.w_m1 * updated_point

        pre_qr_matrix = np.matrix(np.zeros((3*self.dim_state, self.dim_state)))
        for i in range(2*self.dim_state):
            pre_qr_matrix[i] = np.sqrt(
                self.w_m1)*(updated_points[i+1] - updated_mean).T
        pre_qr_matrix[2*self.dim_state:] = sr_Q

        updated_sr_covar = np.linalg.qr(pre_qr_matrix)[1][0:self.dim_state].T
        updated_sr_covar = cholupdate(
            updated_sr_covar*updated_sr_covar.T, updated_points[0]-updated_mean, self.w_c0)

        self.state = updated_mean
        self.state_sr_covar = updated_sr_covar

        return updated_points, updated_mean, updated_sr_covar

    def predict(self, u, sr_Q):
        """
        予測段を回す
        """
        sigma_points = self.genSigmaPoints()
        return self.unscentedTransform(sigma_points, u, sr_Q)

    def observe(self, real_observed_vec, updated_points, sr_R, trim_func=None):
        """
        観測を行い、2N+1個の測定値と平均,共分散行列の平方根を返す
        """
        observed_points = []
        for point in updated_points:
            observed_point = self.observeFunc(point, sr_R*sr_R.T)
            observed_points.append(observed_point)

        if type(observed_points[0]) != np.matrix or observed_points[0].shape[1] != 1:
            raise TypeError("observed vec must be (m,1) matrix")

        if trim_func != None:
            # 観測データに含まれる無効なデータを除去する
            real_observed_vec, observed_points, sr_R = trim_func(
                real_observed_vec, observed_points, sr_R)

        dim_observe = observed_points[0].shape[0]

        # 平均を求める
        observed_mean = self.w_m0*observed_points[0]
        for point in observed_points[1:]:
            observed_mean = observed_mean + self.w_m1 * point

        # 観測共分散行列の平方根を計算する
        pre_qr_matrix = np.matrix(
            np.zeros((2*self.dim_state + dim_observe, dim_observe)))
        for i in range(2*self.dim_state):
            pre_qr_matrix[i] = np.sqrt(
                self.w_m1) * (observed_points[i+1] - observed_mean).T
        pre_qr_matrix[2*self.dim_state:] = sr_R

        observed_sr_covar = np.linalg.qr(pre_qr_matrix)[1][0:dim_observe].T
        observed_sr_covar = cholupdate(
            observed_sr_covar*observed_sr_covar.T, observed_points[0]-observed_mean, self.w_c0)
        return observed_points, observed_mean, observed_sr_covar

    def correct(self, real_observed_vec, updated_points, updated_mean, updated_sr_covar, sr_R):
        """
        補正段をまわす
        """
        observed_points, observed_mean, sr_s_yy = self.observe(
            real_observed_vec, updated_points, sr_R)

        s_xy = self.w_c0 * \
            (updated_points[0] - updated_mean) * \
            (observed_points[0] - observed_mean).T
        for i in range(2*self.dim_state):
            s_xy = s_xy + self.w_m1 * \
                (updated_points[i+1] - updated_mean) * \
                (observed_points[i+1] - observed_mean).T

        inv_sr_s_yy = np.linalg.inv(sr_s_yy)
        K2 = s_xy * inv_sr_s_yy.T
        K = K2 * inv_sr_s_yy

        self.state = self.state + K * (real_observed_vec - observed_mean)
        updated_sr_covar = cholupdate(
            updated_sr_covar*updated_sr_covar.T, K2, -1)
        self.state_sr_covar = updated_sr_covar
        return self.state, self.state_sr_covar


def genRandomNoiseNormal(stdv_arr):
    noise = [np.random.normal(0, stdv, 1)[0] for stdv in stdv_arr]
    return noise


def main():
    walls = simulator.Walls()
    walls.append((200, 200), (-200, 200))
    walls.append((-200, 200), (-200, -200))
    walls.append((-200, -200), (200, -200))
    walls.append((200, -200), (200, 200))

    wall_arr = walls.get()
    num_lazer = 3
    u_range, l_range = np.pi/2, 0

    p_list = [0.1, 0.1, 0.01, 0.1, 0.1, 0.01]
    init_state_noise = np.matrix(genRandomNoiseNormal(p_list)).T
    init_state = np.matrix(
        np.array([150, 150, 0, 0, 0, 0], dtype="float64")).T + init_state_noise

    sim = simulator.accelSimulator(wall_arr, np.array(
        init_state.T)[0], num_lazer, u_range, l_range)
    dt = 0.01

    lazer_stdv = 5.0
    enc_gyro_stdv = [5, 5, 0.01]
    q_list = [0.01, 0.01, 0.001, 0.001, 0.001, 0.0001]
    r_list = [lazer_stdv for _ in range(num_lazer)] + enc_gyro_stdv
    P = np.matrix(np.diag(p_list))
    Q = np.matrix(np.diag(q_list))
    R = np.matrix(np.diag(r_list))

    sr_Q = np.linalg.cholesky(Q)
    sr_R = np.linalg.cholesky(R)

    def trimData(lazer_num, real_observed_vec, observed_points, R):
        dim_observe = real_observed_vec.shape[0]
        is_valid = [(real_observed_vec[i] > 0) for i in range(
            num_lazer)] + [True for _ in range(dim_observe-lazer_num)]
        for point in observed_points:
            for i in range(num_lazer):
                if point[i] < 0:
                    is_valid[i] = False

        num_is_valid = 0
        for b in is_valid:
            if b:
                num_is_valid += 1

        ret_real_observed_vec = []
        for i, data in enumerate(real_observed_vec):
            if is_valid[i]:
                ret_real_observed_vec.append(data)

        ret_observed_points = [
            np.matrix(np.zeros((num_is_valid, 1))) for _ in range(len(observed_points))]
        count, i = 0, 0
        while count < num_is_valid:
            if is_valid[i]:
                for j in range(len(observed_points)):
                    ret_observed_points[j][count] = observed_points[j][i]
                count += 1
            i += 1

        ret_R = R[dim_observe-num_is_valid:, dim_observe-num_is_valid]
        return ret_real_observed_vec, ret_observed_points, ret_R

    def updateFunc(state, u):
        state[0:3] = state[0:3] + state[3:6]*dt + u*dt**2/2
        state[3:6] = state[3:6] + u*dt
        return state

    def observeFunc(state, R):
        lazer_vec = sim.measure(np.array(state.T)[0][0:3])
        ret = np.matrix(np.zeros((len(lazer_vec)+3, 1)))
        for i, each_data in enumerate(lazer_vec):
            if not each_data[0]:
                ret[i] = -1
            else:
                ret[i] = each_data[3] + np.random.normal(0, R[i, i], 1)
        for i in range(3):
            ret[len(lazer_vec)+i] = state[i+3] + \
                np.random.normal(R[len(lazer_vec)+i, len(lazer_vec)+i])
        return ret

    ukf = sqrUKF(init_state, P, updateFunc, observeFunc)

    corrected_state = init_state.copy()
    updated_mean = init_state.copy()
    real_state_vec = []
    ukf_state_vec = []
    frame, max_frame = 0, 1000
    correct = True
    while frame < max_frame:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break

        if correct:
            ax, ay, ath = -corrected_state[0, 0] * \
                2, -corrected_state[1, 0]*1, 0.0
        else:
            ax, ay, ath = -updated_mean[1, 0]*0.2, updated_mean[0, 0]*0.2, 0.0
        u = np.matrix(np.array([[ax], [ay], [ath]]))
        sim.accel(ax, ay, ath, dt, q_list)

        updated_points, updated_mean, updated_sr_covar = ukf.predict(u, sr_Q)

        lazer_vec = []
        for data in sim.sensor_measure():
            if data[0]:
                lazer_vec.append(data[3])
            else:
                lazer_vec.append(-1)
        lazer_vec = np.array(lazer_vec)

        vel_vec = sim.getVel() + genRandomNoiseNormal(enc_gyro_stdv)
        observed_vec = np.matrix(np.concatenate((lazer_vec, vel_vec))).T
        if correct:
            corrected_state, corrected_sr_covar = ukf.correct(
                observed_vec, updated_points, updated_mean, updated_sr_covar, sr_R)
            corrected_covar = corrected_sr_covar * corrected_sr_covar.T
            ukf_state_vec.append(np.array(corrected_state.T)[0])
        else:
            ukf_state_vec.append(np.array(updated_mean.T)[0])
        real_state_vec.append(sim.getState())

        sim.draw_init()
        sim.draw_wall()
        sim.draw_robot()
        sim.draw_lazer()
        if correct:
            sim.draw_predicted_pos(np.array(corrected_state.T)[0][0:3], np.sqrt(
                corrected_covar[0, 0])*3, np.sqrt(corrected_covar[1, 1]))
        else:
            sim.draw_predicted_pos(np.array(updated_mean.T)[0][0:3], 10, 10)
        sim.draw_finish()

        frame += 1

    t = [x for x in range(max_frame)]
    robot_state_arr = np.array(ukf_state_vec)
    real_state_arr = np.array(real_state_vec)
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
