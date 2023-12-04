import numpy as np
import gym
from gym import spaces
import reco
import guidance
import json
from assit import assit
from clamp_angle import clamp_angle
from PyQt5.QtCore import QObject, pyqtSignal

class SharedData(QObject):
    azimuthChanged = pyqtSignal(int)
    elevationChanged = pyqtSignal(int)
    focalLengthChanged = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.azimuth = 0  # 方位角
        self.elevation = 45  # 仰角
        self.focal_length = 0.0043
        self.gud_a = 0  # 新增：gud_a，初始值为0
        self.gud_e = 0  # 新增：gud_e，初始值为0

    def setAzimuth(self, value):
        # 限制方位角在 -180 到 180 度之间
        self.azimuth = max(-180, min(value, 180))
        self.azimuthChanged.emit(self.azimuth)

    def setElevation(self, value):
        # 限制仰角在 -120 到 90 度之间
        self.elevation = max(-120, min(value, 90))
        self.elevationChanged.emit(self.elevation)

    def setFocalLength(self, value):
        # 限制焦距在 0.0043 到 0.129 之间
        self.focal_length = max(0.0043, min(value, 0.129))
        self.focalLengthChanged.emit(self.focal_length)

    # 新增：设置gud_a和gud_e的方法
    def setGudA(self, value):
        self.gud_a = value

    def setGudE(self, value):
        self.gud_e = value


    # 新增：设置速度和加速度的方法
    def setVelocity(self, value):
        self.velocity = value

    def setAcceleration(self, value):
        self.acceleration = value

class CameraControlEnv(gym.Env):
    def __init__(self):
        super(CameraControlEnv, self).__init__()

        # 定义连续动作空间，例如控制相机方位角、俯仰角和焦距的连续值
        self.state_dim = 9  # 新增：状态空间为11维
        self.observation_dim = 5
        self.action_dim = 2  # 动作维度为2维，包括方位角、俯仰角
        self.total_reward = 0

        # 状态空间的定义，包括方位角、俯仰角、焦距、gud_a、gud_e、速度和加速度
        self.state_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float32
        )

        # 新增：定义包含连续动作的连续动作空间，每个维度有相应的范围
        self.action_space = spaces.Box(
            low=np.array([-6, -6], dtype=np.float32),
            high=np.array([6, 6], dtype=np.float32),
            dtype=np.float32
        )

        self.camera_pos = np.array([0, 0, 0])
        self.d = 0.00818
        self.aspect_ratio = 16/9
        self.blta = 0.000625
        self.start_point = np.array([900, 1200, 200])
        self.target_point = np.array([- 900, - 1200, 200])

        # 构建文件的绝对路径
        file_path = 'tmpkw.json'

        # 读取 JSON 文件
        with open(file_path, "r") as file:
            data = json.load(file)
        
        # 打印输出 UAV 的数据
        uav_data = data.get("UAV", [])

        self.trajectory = uav_data
        self.uav_var_count = len(uav_data)
        self.shared_data = SharedData()

    def reset(self):
        self.shared_data.azimuth = 0
        self.shared_data.elevation = 45
        self.shared_data.focal_length = 0.0043
        self.shared_data.gud_a = 0  # 新增：重置gud_a为0
        self.shared_data.gud_e = 0  # 新增：重置gud_e为0
        self.x, self.y, self.z = self.trajectory[0] - self.camera_pos
        self.done = False
        self.current_t = 0
        state = self.get_state()
        self.total_reward = 0  # 重置累计奖励为0
        return state

    def step(self, action):
        azimuth_1, elevation_1 = action

        # 获取上一个状态的azimuth和elevation
        prev_azimuth = self.shared_data.azimuth
        prev_elevation = self.shared_data.elevation

        # 计算azimuth和elevation相对于上一个状态的变化
        azimuth = prev_azimuth + azimuth_1
        elevation = prev_elevation + elevation_1           
        if elevation > 90:
            elevation = 90
        if elevation < -120:
            elevation = - 120

        azimuth = clamp_angle(azimuth)

        # 将动作应用于相机的方位角、俯仰角和焦距
        self.shared_data.azimuth = azimuth
        self.shared_data.elevation = elevation
        
        self.x, self.y, self.z = self.trajectory[self.current_t] - self.camera_pos

        prev_focal_length = self.shared_data.focal_length
        _, _, R3 = assit(prev_focal_length)
        target_vector = np.array([self.x, self.y, self.z])
        distance = np.linalg.norm(target_vector)

        if distance >= R3:
            focal_action = -0.006
        else:
            focal_action = 0.006
        
        focal_length = prev_focal_length + focal_action

        if focal_length > 0.129:
            focal_length = 0.129           
        if focal_length < 0.0043:
            focal_length = 0.0043
        
        self.shared_data.focal_length = focal_length

        self.current_t += 1
        state = self.get_state()
        reward, reward1 = self.calculate_reward()
        done = self.current_t >= self.uav_var_count or reward1 == 1
        self.total_reward += reward
        info = {}
        return state, reward, done, info

    def get_state(self):
        # print(f"{self.x, self.y, self.z}")
        state = np.array([
            self.shared_data.azimuth,
            self.shared_data.elevation,
            self.shared_data.focal_length,
            self.shared_data.gud_a,  # 新增：gud_a
            self.shared_data.gud_e,  # 新增：gud_e
            self.x,
            self.y,
            self.z,
            self.current_t
        ])
        return state

    def calculate_reward(self):
        camera_pos = self.camera_pos  # 相机的位置
        azimuth = self.shared_data.azimuth
        elevation = self.shared_data.elevation
        focal_length = self.shared_data.focal_length
        x = self.x
        y = self.y
        z = self.z
        # print(f"{x, y, z}")       

        # 计算目标相对于相机的位置矢量
        target_vector = np.array([x, y, z]) - camera_pos

        # 计算俯角和仰角
        distance = np.linalg.norm(target_vector)
        elevation_rad = np.arcsin(target_vector[2] / distance)
        azimuth_rad = np.arctan2(target_vector[1], target_vector[0])

        elevation_deg = np.rad2deg(elevation_rad)
        elevation_deg = round(elevation_deg, 2)

        azimuth_deg = np.rad2deg(azimuth_rad)
        azimuth_deg = round(azimuth_deg, 2)

        # 相机的水平视场角度
        fov_adjusted1 = np.rad2deg(2 * np.arctan(0.5 * self.d / focal_length))
        fov_adjusted1 = round(fov_adjusted1, 2)
        fov_adjusted1 = fov_adjusted1 / 2

        # 相机的上下俯仰视场角度
        fov_adjusted2 = fov_adjusted1 / self.aspect_ratio
        fov_adjusted2 = round(fov_adjusted2, 2)
        fov_adjusted2 = fov_adjusted2 / 2

        gap_a = np.abs(azimuth_deg - azimuth)
        gap_e = np.abs(elevation_deg - elevation)

        reward_a = (fov_adjusted1 - gap_a) / fov_adjusted1
        reward_e = (fov_adjusted2 - gap_e) / fov_adjusted2

        reward_a1 = max(reward_a, -1)
        reward_e1 = max(reward_e, -1)

        reward0 = (reward_a1 + reward_e1) / 2

        # 对真实方位信息模糊后获得指引信息
        self.gud_e, self.gud_a = guidance.gd(elevation_deg, azimuth_deg)
        self.shared_data.gud_e = self.gud_e
        self.shared_data.gud_a = self.gud_a
        # print(f"{self.gud_a, self.gud_e, azimuth_deg, elevation_deg}")
        
        # print(f"{reward2}")
        
        a1 = azimuth - fov_adjusted1
        a2 = azimuth + fov_adjusted1
        e1 = elevation - fov_adjusted2
        e2 = elevation + fov_adjusted2
        # print(f"{a1, e1}")        

        p1 = 1 if a1 <= azimuth_deg <= a2 else 0
        p2 = 1 if e1 <= elevation_deg <= e2 else 0

        pg = 1 if p1 == 1 and p2 == 1 else 0

        g = reco.recog(distance, focal_length)          

        # 辨认奖励的计算
        if (pg == 1):
            if g == 3:
                # 辨认目标
                # print('indentification')
                # print(f"{a1, a2, azimuth_deg}")
                reward1 = 1
                reward = self.blta * reward0 + reward1
                return reward, reward1
            else:
                if g == 2:
                    # 识别目标
                    # print('recognition')
                    # print(f"{a1, a2, azimuth_deg}")
                    reward1 = 0
                    reward = self.blta * reward0 + reward1
                    return reward, reward1
                else:
                    if g == 1:
                        # 探测目标
                        # print('detection')
                        # print(f"{a1, a2, azimuth_deg}")
                        reward1 = 0
                        reward = (self.blta * reward0 + reward1)
                        return reward, reward1
                    else:
                        # 未识别目标
                        # print('undetection')
                        reward1 = 0
                        reward = self.blta * reward0 + reward1
                        return reward, reward1
        else:
            # print('none')
            reward1 = 0
            reward = self.blta * reward0 + reward1
            return reward, reward1