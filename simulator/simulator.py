import pygame
import numpy as np
import copy

def calc_d1(pos0, pos1, pos2):
    """
    壁(線分AB)にレーザが点Pで当たったときの、d1=(線分APの長さ)/(線分ABの長さ)を返す
    -inf < d1 < inf
    """
    x0, y0, th = pos0
    x1, y1 = pos1
    x2, y2 = pos2
    A1 = (y1-y0)*np.cos(th) - (x1-x0)*np.sin(th)
    A2 = (x2-x1)*np.sin(th) - (y2-y1)*np.cos(th)
    if A1*A2>0 and (abs(A1)<abs(A2)):
        return True, A1/A2
    else:
        return False, 0

def calc_d0(pos0, pos1, pos2 , d1):
    """
    点Oから出たレーザが壁(線分AB)に点Pで当たったときの、d0=(線分OPの長さ)を返す
    """
    x0, y0, th = pos0
    x1, y1 = pos1
    x2, y2 = pos2
    A1, A2 = 0,0
    if (abs(th) < np.pi/4) or (abs(th) > np.pi*3/4):
        A1 = (x1-x0)+d1*(x2-x1)
        A2 = np.cos(th)
    else:
        A1=(y1-y0)+d1*(y2-y1)
        A2 = np.sin(th)
    if A1*A2 > 0:
        return True, A1/A2
    else:
        return False, 0


def calc_intersection(pos0, pos1, pos2):
    """
    壁(線分)と光線の交点を求める
    交わるかどうかの判定もここで行う
    """
    th = pos0[2]
    isCross1, d1 = calc_d1(pos0, pos1, pos2)
    isCross2, d0 = calc_d0(pos0, pos1, pos2, d1)
    if isCross1 and isCross2:
        pos = pos0[0:2] + d0*np.array([np.cos(th), np.sin(th)])
        return True, pos, d0 
    else:
        return False, [0,0], 0

def calc_jacobian(pos, omega, wall_pos1, wall_pos2):
    x,y,th = pos
    x1, y1 = wall_pos1
    x2, y2 = wall_pos2
    c = np.cos(th)
    s = np.sin(th)

    A = (x2-x1)*s - (y2-y1)*c
    dx0 = x1- x
    dx1 = x2-x1
    dy0 = y1-y
    dy1 = y2-y1

    j = np.array([0, 0, 0])
    j[0] = dy1/A
    j[1] = -dx1/A
    j[2] = omega/c**2*(
            dx0*s +
            dx1/A*(dy0*c - dx0*s)*s-
            dx1/A**2*(dy0*dx1 + dy1*dx0)*(2*c**2-1)+
            2*(dy0*dy1+dx0*dx1)*s*c)
    return j

class Walls:
    """
    壁オブジェクトをまとめて扱うクラス
    appendで追加、getでリストを返す
    """
    def __init__(self):
        self.wall_num = 0
        self.wall_arr = []

    def append(self, start, end):
        self.wall_num += 1
        self.wall_arr.append([np.array(start), np.array(end)])

    def get(self):
        return self.wall_arr

class Simulator:
    def __init__(self, wall_arr, init_pos, num_lazer, u_range=np.pi/2, l_range=-np.pi/2):
        #シミュレーション環境の初期化
        self.wall_arr = wall_arr
        self.pos = init_pos
        #センサーの初期化
        self.u_range = u_range #upper limit
        self.l_range = l_range  #lower limit
        self.num_lazer = num_lazer 
        if num_lazer > 1:
            self.lazer_angle = [(self.u_range-self.l_range)/(self.num_lazer-1)*i+self.l_range for i in range(self.num_lazer)] #各レーザの角度
        elif num_lazer == 1:
            self.lazer_angle = [(self.u_range + self.l_range)/2]
        self.measured_pos = [[False, 0, 0, 0] for i in range(self.num_lazer)] #各レーザで測定されたデータを入れるリスト

        #pygame関係の初期化
        pygame.init()
        self.win_size = (600, 600)
        self.screen = pygame.display.set_mode(self.win_size)
        self.clock = pygame.time.Clock()
        #色の設定
        self.background_color = (255,255,255)
        self.wall_color = (0,0,0)
        self.robot_color = (0,0,0)
        self.lazer_color = (255, 0, 0)
        self.estimate_color = (0, 255, 0)
    
    def move(self, dx, dy, dth):
        """
        シミュレータ上の機体の座標を動かす
        """
        self.pos = self.pos + np.array([dx, dy, dth])
    
    def getPos(self):
        return self.pos
    def measure(self, pos, jacobian=False, omega=0):
        """
        レーザで壁との距離を図る。壁に当たったかどうかがmeasured_pos[i]の第一要素にboolとして記録される 
        """
        if not jacobian:
            measured_pos = [[False, 0,0, 0] for i in range(self.num_lazer)]
            for i, angle in enumerate(self.lazer_angle):
                lazer_pos = copy.copy(pos)
                lazer_pos[2] += angle
                lazer_pos = np.array(lazer_pos)
                for wall in self.wall_arr:
                    isCross, wall_pos, d0 = calc_intersection(lazer_pos, wall[0], wall[1])
                    if isCross:
                        if (not measured_pos[i][0]) or (measured_pos[i][0] and (measured_pos[i][3] > d0)):
                            measured_pos[i] = [True, wall_pos[0], wall_pos[1], d0]
            return measured_pos
        else:
            measured_pos = [[False, 0,0, 0, np.array([0, 0, 0])] for i in range(self.num_lazer)]
            for i, angle in enumerate(self.lazer_angle):
                lazer_pos = copy.copy(pos)
                lazer_pos[2] += angle
                lazer_pos = np.array(lazer_pos)
                for wall in self.wall_arr:
                    isCross, wall_pos, d0 = calc_intersection(lazer_pos, wall[0], wall[1])
                    if isCross:
                        if (not measured_pos[i][0]) or (measured_pos[i][0] and (measured_pos[i][3] > d0)):
                            j = calc_jacobian(lazer_pos, omega, wall[0], wall[1])
                            measured_pos[i] = [True, wall_pos[0], wall_pos[1], d0, j]
            return measured_pos

    def predict_measure(self, pos):
        """
        posから図った壁までの距離を返す
        """
        return self.measure(pos)

    def predict_measure_with_jacobian(self, pos, omega):
        return self.measure(pos, True, omega)

    def sensor_measure(self):
        """
        実際の座標から図った壁までの距離を返す
        """
        self.measured_pos = self.measure(self.pos)
        return self.measured_pos

    def toScreen(self, pos):
        """
        シミュレータ内の座標をscreen座標に変換する
        screen座標系は左上が原点の左手系
         ・→ x
         ↓
         y
        """
        x, y = pos
        return np.array([x+self.win_size[0]/2, self.win_size[1]/2 -y])

    def draw_init(self):
        """
        背景を塗りつぶす
        """
        self.screen.fill(self.background_color)

    def draw_wall(self):
        """
        壁を描画する
        """
        for wall in self.wall_arr:
            posA = self.toScreen(wall[0])
            posB = self.toScreen(wall[1])
            pygame.draw.line(self.screen, self.wall_color, posA, posB, 3)

    def draw_predicted_pos(self, pos, r1, r2):
        x, y = copy.copy(pos[0:2])
        x -= r1/2
        y += r2/2
        rect_info = [x for x in self.toScreen([x,y])] + [2*r1, 2*r2]
        pygame.draw.ellipse(self.screen, self.estimate_color, rect_info) 

    def draw_robot(self):
        """
        位置と向きをそれぞれ点と棒で表現する
        """
        th = self.pos[2]
        pos1 = self.toScreen(self.pos[0:2] + (40*np.array([np.cos(th), np.sin(th)])))
        pos0 = [int(x) for x in self.toScreen(self.pos[0:2])]
        pygame.draw.circle(self.screen, self.robot_color, pos0, 10)
        pygame.draw.line(self.screen, self.robot_color, pos0, pos1, 1)

    def draw_dummy_robot(self, pos ,color):
        th = pos[2]
        pos1 = self.toScreen(pos[0:2] + np.array(np.cos(th), np.sin(th)) * 20)
        pos0 = [int(x) for x in self.toScreen(pos[0:2])]
        pygame.draw.circle(self.screen, color, pos0, 5)
        pygame.draw.line(self.screen, color, pos0, pos1, 1)


    def draw_lazer(self):
        """
        レーザを描画する
        壁に当たらなかったものは描画しない
        """
        measured_pos = self.measure(self.pos)
        for intersection in measured_pos:
            if intersection[0]:
                pos = np.array([intersection[1], intersection[2]])
                pygame.draw.line(self.screen, self.lazer_color, self.toScreen(self.pos[0:2]), self.toScreen(pos), 1)

    def draw_finish(self):
        """
        今までに描画したものを画面状に反映する
        更新のスピード(フレームレート)を下で変えられる
        """
        pygame.display.update()
        t = self.clock.tick(100)#100fps, tは前回の更新からの秒数(ms)を返す
        #print(t)

    def draw(self):
        """
        描画系をまとめて実行
        """
        self.draw_init()
        self.draw_wall()
        self.draw_robot()
        self.draw_lazer()
        self.draw_finish()

class accelSimulator(Simulator):
    def __init__(self, wall_arr, init_state, num_lazer, u_range=np.pi/2, l_range=-np.pi/2):
        init_pos = init_state[0:3]
        init_vel = init_state[3:6]
        super().__init__(wall_arr, init_pos, num_lazer, u_range, l_range)
        self.vel = init_vel

    def getVel(self):
        return self.vel

    def getState(self):
        return np.concatenate([self.pos, self.vel])
    
    def accel(self, ax, ay, ath, dt, q_noise):
        pos_noise, vel_noise = q_noise[0:3], q_noise[3:6]
        self.pos = self.pos + self.vel * dt + np.array([ax, ay, ath])*dt**2/2 + pos_noise 
        self.vel = self.vel + np.array([ax, ay, ath])*dt + vel_noise
