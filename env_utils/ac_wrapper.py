'''
@Author: WANG Maonan
@Date: 2023-09-08 15:49:30
@Description: 处理 ACEnvironment
+ state wrapper: 获得每个 aircraft 在覆盖范围内车辆的信息, 只有 drone 与车辆进行通信
+ reward wrapper: aircraft 覆盖车辆个数
@LastEditTime: 2023-09-25 14:03:14
'''
import random

import numpy as np
import gymnasium as gym
import math
from gymnasium.core import Env
from typing import Any, SupportsFloat, Tuple, Dict
from typing import List
from collections import defaultdict, deque

from sympy.integrals.intpoly import distance_to_side
from tshub.aircraft.aircraft_action_type import aircraft_action_type


class ACEnvWrapper(gym.Wrapper):
    """Aircraft Env Wrapper for single junction with tls_id
    """
    def __init__(self, env: Env, aircraft_inits, max_states: int = 3) -> None:
        super().__init__(env)
        # TODO: ADD ROAD DENSITY HEATMAP
        self._pos_set = deque([self._get_initial_state()] * max_states, maxlen=max_states)  # max state : 3
        self.speed = aircraft_inits["drone_1"]["speed"]
        self.x_range, self.y_range, self.h_min, self.x_max, self.y_max, self.h_max, self.side_length = (
            None, None, None, None, None, None, None)

        self.initial_points = {
            ac_id: ac_value["position"] for ac_id, ac_value in aircraft_inits.items()
        }

        self.grid_size = None

        self.latest_ac_pos = {}
        self.latest_veh_pos = {}
        self._veh_traj = defaultdict(lambda: deque(maxlen=10))
        self.latest_cover_radius = {}
        self.cover_time = defaultdict(int)
        self.frame_rate = getattr(env, "metadata", {}).get("render.fps", 10)
        speed = self.speed
        self.air_actions = {
            0: (speed, 0),  # -> 右
            1: (speed, 1),  # ↗ 右上
            2: (speed, 2),  # ↑ 正上
            3: (speed, 3),  # ↖ 左上
            4: (speed, 4),  # ← 左
            5: (speed, 5),  # ↙ 左下
            6: (speed, 6),  # ↓ 正下
            7: (speed, 7),  # ↘ 右下
        }
        self.end_point = [-345,-40]
        self.last_action = {}

    def get_relative_pos(self, aircraft_id, pos) -> List:
        _init_points = self.initial_points[aircraft_id]
        pos_new = [pos[0] - _init_points[0], pos[1] - _init_points[1], pos[2] - _init_points[2]]
        return pos_new

    def get_mid_point(self, ac_id, veh_id) -> float:
        mid_x = (self.latest_veh_pos[veh_id][0] + self.end_point[0]) / 2
        mid_y = (self.latest_veh_pos[veh_id][1] + self.end_point[1]) / 2
        dist = np.linalg.norm(np.array([mid_x,mid_y]) - np.array(self.latest_ac_pos[ac_id][:2]))
        return dist


    @property
    def action_space(self):
        return gym.spaces.Discrete(8)
    
    @property
    def observation_space(self):

        spaces = {
            "ac_attr": gym.spaces.Box(low=np.zeros((9,)), high=np.ones((9,)), shape=(9,)),
            "relative_vecs": gym.spaces.Box(low=np.zeros((30,)), high=np.ones((30,)), shape=(30,)),
            "cover_counts": gym.spaces.Box(low=0, high=np.inf, shape=(1,)),
            "bound_dist": gym.spaces.Box(low=np.zeros((2,)), high=np.ones((2,)), shape=(2,)),
            "action_dir": gym.spaces.Box(low=np.zeros((8,)), high=np.ones((8,)), shape=(8,)),
        }
        dict_space = gym.spaces.Dict(spaces)
        return dict_space

    def _get_initial_state(self) -> List[int]:
        return [0, 0, 0]  # x, y, z

    # Wrapper
    def state_wrapper(self, state):
        """自定义 state 的处理, 只找出与 aircraft 通信范围内的 vehicle
        """
        new_state = dict()
        veh = state['vehicle']
        aircraft = state['aircraft']
        dist_to_bound = []
        relative_vecs = []
        cover_counts = []
        action_onehot = []

        for aircraft_id, aircraft_info in aircraft.items():
            if aircraft_info['aircraft_type'] != 'drone':
                continue

            drone_init = self.initial_points[aircraft_id]
            cover_radius = aircraft_info['cover_radius']
            aircraft_pos = aircraft_info['position']
            ac_pos = self.get_relative_pos(aircraft_id, aircraft_pos)
            self.latest_cover_radius[aircraft_id] = cover_radius
            self.latest_ac_pos[aircraft_id] = ac_pos
            self._pos_set.append(ac_pos)
            # print("ac_pos",ac_pos)
            vehicle_state = {}

            action_id = self.last_action.get(aircraft_id, 0)
            action_onehot = np.eye(8)[action_id]
            if 0 <= action_id < 8:
                action_onehot[action_id] = 1

            for vehicle_id, vehicle_info in veh.items():

                vehicle_pos = vehicle_info['position']
                veh_pos = [vehicle_pos[0] - drone_init[0], vehicle_pos[1] - drone_init[1]]
                self.latest_veh_pos[vehicle_id] = veh_pos
                self._veh_traj[vehicle_id].append(veh_pos)
                # print('veh_pos',veh_pos)

                dx = ac_pos[0] - veh_pos[0]
                dy = ac_pos[1] - veh_pos[1]
                dist = math.hypot(dx, dy)
                dist_to_bound = [self.x_range - abs(ac_pos[0]), self.y_range - abs(ac_pos[1])]
                dist_to_bound = np.array(dist_to_bound)

                if dist <= cover_radius:
                    vehicle_state[vehicle_id] = vehicle_info.copy()
                    # veh_in_range = 1 - dist / cover_radius
                    # vehicle_state[vehicle_id]['veh_in_range'] = veh_in_range
                    # veh_feature.append([veh_in_range])
                relative_vecs.append([dx, dy])

            cover_counts.append(len(vehicle_state))
            new_state[aircraft_id] = vehicle_state

        if len(relative_vecs) == 0:
            relative_vecs = np.zeros((15,2))
        else:
            relative_vecs = np.array(relative_vecs[:15])
            if relative_vecs.shape[0] < 15:
                pad = np.zeros((15 - relative_vecs.shape[0], 2))
                relative_vecs = np.vstack((relative_vecs, pad))

        if len(dist_to_bound) == 0:
            dist_to_bound = np.zeros((2,))
        # if len(veh_feature) == 0:
        #     veh_feature = np.zeros((20,1))
        # else:
        #     veh_feature = np.array(veh_feature[:20])
        #     if veh_feature.shape[0] < 20:
        #         pad = np.zeros((20 - veh_feature.shape[0], 1))
        #         veh_feature = np.vstack((veh_feature, pad))

        feature_set = {
            "ac_attr": np.array(self._pos_set).reshape(-1),
            "relative_vecs": np.array(relative_vecs).reshape(-1),
            "cover_counts": np.array([cover_counts[0]]).reshape(-1),
            "bound_dist": dist_to_bound.reshape(1,-1).squeeze(),
            "action_dir": action_onehot
            # "veh_traj":
            # "road_density":
        }
        # print("dist_to_bound",dist_to_bound)
        return feature_set, new_state

    def reward_wrapper(self, states, dones) -> float:
        """自定义 reward 的计算
        """
        # TODO: ADD MORE REWARD METHOD
        reward = 0
        frame_threshold = 3 * self.frame_rate
        for aircraft_id, vehicle_info in states.items():
            aircraft_pos = self.latest_ac_pos[aircraft_id]
            cover_radius = self.latest_cover_radius[aircraft_id]
            _x, _y, _h = aircraft_pos

            reward += len(vehicle_info) * 3

            proximity_bonus = 0
            midpoint_bonus = 0
            d_max = 200
            veh_keys = list(self.latest_veh_pos.keys())
            if len(veh_keys) > 2:
                first_veh_id = veh_keys[0]
                last_veh_id = veh_keys[-1]
                f_dist = self.get_mid_point(aircraft_id, first_veh_id)
                if f_dist < d_max:
                    midpoint_bonus += 1 - f_dist / d_max
                l_dist = self.get_mid_point(aircraft_id, last_veh_id)
                if l_dist < d_max:
                    midpoint_bonus += 1 - l_dist / d_max
            else:
                pass
            # for vehicle_id, vehicle_pos in self.latest_veh_pos.items():
            #     dist = self.get_mid_point(aircraft_id, vehicle_id)
            #     if dist < d_max:
            #         midpoint_bonus += 1 - dist / d_max
                # veh_traj = self._veh_traj[vehicle_id]
                # dist = np.linalg.norm(np.array(vehicle_pos[:2]) - np.array(aircraft_pos[:2]))
                #
                # if dist <= d_max:
                #     proximity_bonus += 1 - dist / d_max
                #     reward += proximity_bonus

                # mid_x = (vehicle_pos[0] + self.end_point[0]) / 2
                # mid_y = (vehicle_pos[1] + self.end_point[1]) / 2
                # dist_to_mid = np.linalg.norm(np.array([mid_x, mid_y]) - np.array(aircraft_pos[:2]))
                # if dist_to_mid <= d_max:
                #     midpoint_bonus += 1 - dist_to_mid / d_max
                #     reward += midpoint_bonus


                # if dist <= cover_radius:
                #     self.cover_time[vehicle_id] += 1
                #     if self.cover_time[vehicle_id] % frame_threshold == 0:
                #         persistent_bonus += 5
                # else:
                #     proximity_bonus += (cover_radius/dist)
                # elif dist > d_max:
                #     proximity_bonus -= (dist - d_max) / cover_radius
                # print(self.cover_time[vehicle_id], frame_threshold)
                # if len(veh_traj) >= 2:
                #     prev_pos = veh_traj[-2]
                #     cur_pos = veh_traj[-1]
                #     dx = cur_pos[0] - prev_pos[0]
                #     dy = cur_pos[1] - prev_pos[1]
                #     N = 3
                #     pred_pos = [cur_pos[0] + dx*N, cur_pos[1] + dy*N]
                #     veh_dist = np.linalg.norm(np.array(aircraft_pos[:2]) - np.array(pred_pos[:2]))
                #     if veh_dist < d_max:
                #         proximity_bonus = 1-veh_dist/d_max

            reward += midpoint_bonus

            bound_penalty = 0
            if abs(_x) <= self.x_range and abs(_y) <= self.y_range:
                reward += 0.01
            else:

                if abs(_x) > self.x_range:
                    dones = True
                    bound_penalty = -50
                    reward += bound_penalty
                    return reward, dones
                elif abs(_x) > self.x_range - 30:
                    bound_penalty = -10 # -= abs(_x) - self.x_range
                    reward += bound_penalty

                if abs(_x) > self.x_range:
                    dones = True
                    bound_penalty = -50
                    reward += bound_penalty
                    return reward, dones
                elif abs(_y) > self.y_range -30:
                    bound_penalty = -10 # -= abs(_y) - self.y_range
                    reward += bound_penalty

            # else: bound_penalty += 0.1
            # if _y < 0 or _y > self.y_max:
            #     bound_penalty += -30
            # if _h > self.h_max:
            #     bound_penalty += -30
            # height_penalty = 0.0005 * _h
            # reward -= height_penalty
            # print("drone position:",_x, _y, reward)
        return reward, dones

    def reset(self, seed=1) -> Tuple[Any, Dict[str, Any]]:
        """reset 时初始化 (1) 静态信息; (2) 动态信息
        """
        state =  self.env.reset()

        self.side_length = 2000
        # self.side_length = min(state['grid']['100'].x_max - state['grid']['100'].x_min, state['grid']['100'].y_max - state['grid']['100'].y_min)
        self.x_max = 1000 # self.side_length  # side_length = min(x_max - x_min, y_max - y_min)
        # x_max = env.get_attr("x_max")[0],
        self.y_max = 1000 # self.side_length
        # calculate the max-height (to limit the height)
        #self.h_max = (self.side_length / 10) * math.tan(math.radians(75 / 2))
        self.x_range = 450
        self.y_range = 450

        state, _ = self.state_wrapper(state=state)
        return state, {'step_time':0}

    def step(self, action: Dict[str, int]) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        new_actions = {}
        self.last_action = {}
        # old_pos = self.latest_ac_pos['drone_1']
        if isinstance(action, np.int64):
            new_actions["drone_1"] = self.air_actions[action]
            self.last_action = {"drone_1": action}
        # else:
        #     new_actions = {}
        #     for key, value in action.items():
        #         new_actions[key] = self.air_actions[value]
        elif isinstance(action, dict):
            new_actions = {}
            for key, value in action.items():
                if isinstance(value, np.int64):
                    new_actions[key] = self.air_actions[value]
                elif isinstance(value, tuple):
                    new_actions[key] = value
                else:
                    raise TypeError(f"Unrecognized action type: {value} (type {type(value)})")
        else:
            raise TypeError(f"Action format not recognized: {action} (type {type(action)})")

        states, rewards, truncated, dones, infos = super().step(new_actions) # 与环境交互
        feature_set, veh_states = self.state_wrapper(state=states) # 处理 state
        rewards, dones = self.reward_wrapper(states=veh_states,dones=dones) # 处理 reward

        # print('new action',new_actions['drone_1'])
        # new_pos = self.latest_ac_pos['drone_1']
        # dalta = ((new_pos[0] - old_pos[0]), (new_pos[1] - old_pos[1]))
        # print(f"x: {dalta[0]}, y: {dalta[1]}")

        return feature_set, rewards, truncated, dones, infos
    
    def close(self) -> None:
        return super().close()

