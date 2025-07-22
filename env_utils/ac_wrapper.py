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

from numpy import floating
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
        self.x_range, self.y_range, self.break_spot = (None, None, None)
        self.initial_points = {
            ac_id: ac_value["position"] for ac_id, ac_value in aircraft_inits.items()
        }

        self.grid_size = None

        self.latest_ac_pos = {}
        self.latest_veh_pos = {}
        self.latest_cover_radius = {}
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
        self.lane_veh_pos = {}
        # self.init_break_spot = None

    def  get_relative_ac_pos(self, aircraft_id, pos) -> List:
        _init_points = self.initial_points[aircraft_id]
        pos_new = [pos[0] - _init_points[0], pos[1] - _init_points[1], pos[2] - _init_points[2]]
        return pos_new

    def  get_relative_pos(self, aircraft_id, pos) -> List:
        _init_points = self.initial_points[aircraft_id]
        pos_new = [pos[0] - _init_points[0], pos[1] - _init_points[1]]
        return pos_new

    def get_veh_dist(self, veh_pos) -> float:
        if isinstance(veh_pos, np.ndarray) and veh_pos.ndim == 1 and veh_pos.shape[0] == 2:
            mid_x, mid_y = veh_pos
        else:
            x, y = zip(*veh_pos)
            mid_x = (min(x) + max(x)) / 2
            mid_y = (min(y) + max(y)) / 2
        veh_dist = np.linalg.norm(np.array([mid_x,mid_y]))
        return veh_dist

    @staticmethod
    def distance_penalty(dist, cover_radius, p=10):
        ratio = dist / cover_radius
        penalty = -np.log1p(ratio) / np.log1p(p)
        return penalty

    def prune_old_vehicles(self, current_veh_ids):
        self.latest_veh_pos = {vid: pos for vid, pos in self.latest_veh_pos.items() if vid in current_veh_ids}

    @property
    def action_space(self):
        return gym.spaces.Discrete(8)
    
    @property
    def observation_space(self):

        spaces = {
            "ac_attr": gym.spaces.Box(low=np.zeros((9,)), high=np.ones((9,)), shape=(9,)),
            "relative_vecs": gym.spaces.Box(low=np.zeros((40,)), high=np.ones((40,)), shape=(40,)),
            "cover_counts": gym.spaces.Box(low=0, high=np.inf, shape=(1,)),
            "bound_dist": gym.spaces.Box(low=np.zeros((2,)), high=np.ones((2,)), shape=(2,)),
            "break_spot": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            "no_vehicles": gym.spaces.Box(low=0, high=1, shape=(1,)),
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
        self.prune_old_vehicles(set(veh.keys()))

        dist_to_bound = []
        relative_vecs = []
        cover_counts = [0]
        no_veh = 1

        self.lane_veh_pos = defaultdict(list)

        for aircraft_id, aircraft_info in aircraft.items():
            if aircraft_info['aircraft_type'] != 'drone':
                continue
            cover_radius = aircraft_info['cover_radius']
            aircraft_pos = aircraft_info['position']
            ac_pos = self.get_relative_ac_pos(aircraft_id, aircraft_pos)

            self.latest_cover_radius[aircraft_id] = cover_radius
            self.latest_ac_pos[aircraft_id] = ac_pos
            self._pos_set.append(ac_pos)
            vehicle_state = {}

            break_spot_vec = -np.array(ac_pos[:2])

            for vehicle_id, vehicle_info in veh.items():

                vehicle_pos = vehicle_info['position']
                road_id = vehicle_info['road_id']
                veh_pos = self.get_relative_pos(aircraft_id, vehicle_pos)
                self.latest_veh_pos[vehicle_id] = veh_pos

                dx = veh_pos[0] - ac_pos[0]
                dy = veh_pos[1] - ac_pos[1]

                self.lane_veh_pos[road_id].append([dx, dy])
                relative_vecs.append([dx, dy])

                dist = math.hypot(dx, dy)
                dist_to_bound = [self.x_range - abs(ac_pos[0]), self.y_range - abs(ac_pos[1])]
                dist_to_bound = np.array(dist_to_bound)

                if dist <= cover_radius:
                    vehicle_state[vehicle_id] = vehicle_info.copy()

            cover_counts = np.array([len(vehicle_state)])
            new_state[aircraft_id] = vehicle_state

            has_veh = any(len(veh_list) > 0 for veh_list in self.lane_veh_pos.values())
            no_veh = 1 if not has_veh else 0
            # print("----------",no_veh)
            self.break_spot = break_spot_vec * no_veh
            # print("++++++++++", self.break_spot, "\n")
        if len(relative_vecs) == 0:
            relative_vecs = np.zeros((20,2))
        else:
            relative_vecs = np.array(relative_vecs[:20])
            if relative_vecs.shape[0] < 20:
                pad = np.zeros((20 - relative_vecs.shape[0], 2))
                relative_vecs = np.vstack((relative_vecs, pad))

        if len(dist_to_bound) == 0:
            dist_to_bound = np.zeros((2,))

        # print(self.break_spot,"\n")
        feature_set = {
            "ac_attr": np.array(self._pos_set).reshape(-1),
            "relative_vecs": np.array(relative_vecs).reshape(-1),
            "cover_counts": cover_counts,
            "bound_dist": dist_to_bound.reshape(1,-1).squeeze(),
            "break_spot": np.array(self.break_spot).reshape(-1),
            "no_vehicles": np.array(no_veh).reshape(-1)
        }
        # print("----------", no_veh)
        #  print("relative_vecs: ",relative_vecs)
        return feature_set, new_state

    def reward_wrapper(self, states, dones) -> float:
        """自定义 reward 的计算
        """
        reward = 0
        for aircraft_id, vehicle_info in states.items():
            aircraft_pos = self.latest_ac_pos[aircraft_id]
            cover_radius = self.latest_cover_radius[aircraft_id]
            _x, _y, _h = aircraft_pos

            # reward += len(vehicle_info)
            # proximity_bonus = 0
            # midpoint_bonus = 0
            # veh_keys = list(self.latest_veh_pos.keys())
            # if len(veh_keys) >= 2:
                # first_veh_id = veh_keys[0]
                # last_veh_id = veh_keys[-1]
            if self.lane_veh_pos:
                max_road_id = max(self.lane_veh_pos, key=lambda x: len(self.lane_veh_pos[x]))
                max_road_veh_pos = self.lane_veh_pos[max_road_id]
                # if len(max_road_veh_pos)>=2:
                m_dist = self.get_veh_dist(max_road_veh_pos)
                if m_dist <= cover_radius+50:
                    if len(vehicle_info) != 0:
                        reward += len(vehicle_info)
                else:
                    penalty = self.distance_penalty(m_dist-cover_radius, cover_radius, p=25)
                    reward += penalty*0.45
            else:
                spot_dist = np.linalg.norm(self.break_spot)
                # print("------------------------------------",spot_dist)
                if spot_dist <= cover_radius:
                    reward += 2 # 1， 2, 3
                else:
                    penalty = self.distance_penalty(spot_dist - cover_radius, cover_radius, p=25)
                    reward += penalty*0.45

            bound_penalty = 0
            if abs(_y) > (self.y_range - 100):
                bound_penalty = -5  # -= abs(_y) - self.y_range
                reward += bound_penalty
            if abs(_x) > (self.x_range - 100):
                bound_penalty = -5  # -= abs(_x) - self.x_range
                reward += bound_penalty

            if abs(_x) > self.x_range:
                dones = True
                bound_penalty = -100
                reward += bound_penalty
                return reward, dones
            if abs(_y) > self.y_range:
                dones = True
                bound_penalty = -100
                reward += bound_penalty
                return reward, dones

            # print("drone position:",_x, _y, reward)
        return reward, dones

    def reset(self, seed=1) -> Tuple[Any, Dict[str, Any]]:
        """reset 时初始化 (1) 静态信息; (2) 动态信息
        """
        state =  self.env.reset()
        self.x_range = 650
        self.y_range = 650
        # self.init_break_spot = [-400,350]

        state, _ = self.state_wrapper(state=state)
        return state, {'step_time':0}

    def step(self, action: Dict[str, int]) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        new_actions = {}
        #self.last_action = {}
        # old_pos = self.latest_ac_pos['drone_1']
        if isinstance(action, np.int64):
            new_actions["drone_1"] = self.air_actions[action]
            #self.last_action = {"drone_1": action}
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

        return feature_set, rewards, truncated, dones, infos
    
    def close(self) -> None:
        return super().close()

