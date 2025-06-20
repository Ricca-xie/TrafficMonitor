'''
@Author: WANG Maonan
@Date: 2023-09-08 15:49:30
@Description: 处理 ACEnvironment
+ state wrapper: 获得每个 aircraft 在覆盖范围内车辆的信息, 只有 drone 与车辆进行通信
+ reward wrapper: aircraft 覆盖车辆个数
@LastEditTime: 2023-09-25 14:03:14
'''
import numpy as np
import gymnasium as gym
from gymnasium.core import Env
from typing import Any, SupportsFloat, Tuple, Dict
from typing import List
from collections import deque

class ACEnvWrapper(gym.Wrapper):
    """Aircraft Env Wrapper for single junction with tls_id
    """
    def __init__(self, env: Env, max_states: int = 3) -> None:
        super().__init__(env)
        self._pos_set = deque([self._get_initial_state()] * max_states, maxlen=max_states)  # max state : 3
        speed = 10
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
    @property
    def action_space(self):
        return gym.spaces.Discrete(8)
    
    @property
    def observation_space(self):

        spaces = {
            "ac_attr": gym.spaces.Box(low=np.zeros((9,)), high=np.ones((9,)), shape=(9,)),

        }
        dict_space = gym.spaces.Dict(spaces)
        return dict_space

    def _get_initial_state(self) -> List[int]:
        return [0, 0, 0]  # x,y, z
    
    # Wrapper
    def state_wrapper(self, state):
        """自定义 state 的处理, 只找出与 aircraft 通信范围内的 vehicle
        """
        new_state = dict()
        veh = state['vehicle']
        aircraft = state['aircraft']

        for aircraft_id, aircraft_info in aircraft.items():
            vehicle_state = {}
            cover_radius = aircraft_info['cover_radius']
            aircraft_type = aircraft_info['aircraft_type']
            aircraft_position = aircraft_info['position']
            self._pos_set.append(aircraft_position)

            if aircraft_type == 'drone': # 只统计 drone 类型
                for vehicle_id, vehicle_info in veh.items():
                    vehicle_position = vehicle_info['position']
                    distance = ((vehicle_position[0] - aircraft_position[0]) ** 2 +
                                (vehicle_position[1] - aircraft_position[1]) ** 2) ** 0.5

                    if distance <= cover_radius:
                        vehicle_state[vehicle_id] = vehicle_info
                        spacial_con = -distance / cover_radius + 1
                        vehicle_state[vehicle_id].update({'spacial_con': spacial_con})
                
                new_state[aircraft_id] = vehicle_state

        feature_set = {
                "ac_attr" : np.array(self._pos_set).reshape(-1),
            }

        return feature_set
    
    def reward_wrapper(self, states) -> float:
        """自定义 reward 的计算
        """
        total_cover_vehicles = 0 # 覆盖车的数量
        for _, aircraft_info in states.items():
            total_cover_vehicles += len(aircraft_info)
        return total_cover_vehicles

    def reset(self, seed=1) -> Tuple[Any, Dict[str, Any]]:
        """reset 时初始化 (1) 静态信息; (2) 动态信息
        """
        state =  self.env.reset()
        state = self.state_wrapper(state=state)
        return state, {'step_time':0}
    

    def step(self, action: int) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        new_actions = {}
        if isinstance(action, np.int64):
            new_actions["drone_1"] = self.air_actions[action]
        else:
            new_actions = {}
            for key, value in action.items():
                new_actions[key] = self.air_actions[value]
        states, rewards, truncated, dones, infos = super().step(new_actions) # 与环境交互
        states = self.state_wrapper(state=states) # 处理 state
        rewards = self.reward_wrapper(states=states) # 处理 reward

        return states, rewards, truncated, dones, infos
    
    def close(self) -> None:
        return super().close()