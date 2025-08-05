'''
@Author: WANG Maonan
@Date: 2023-09-08 15:49:30
@Description: 处理 ACEnvironment
+ state wrapper: 获得每个 aircraft 在覆盖范围内车辆的信息, 只有 drone 与车辆进行通信
+ reward wrapper: aircraft 覆盖车辆个数
@LastEditTime: 2023-09-25 14:03:14
'''
import random
import os

from tensorflow.python.ops.special_math_ops import bessel_y0

os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import gymnasium as gym
import math
from gymnasium.core import Env
from typing import Any, SupportsFloat, Tuple, Dict
from typing import List
from collections import defaultdict, deque, Counter

from numpy import floating
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans

class ACEnvWrapper(gym.Wrapper):
    """Aircraft Env Wrapper for single junction with tls_id
    """
    def __init__(self, env: Env, aircraft_inits, max_states: int = 3) -> None:
        super().__init__(env)
        self._pos_set = deque([self._get_initial_state()] * max_states, maxlen=max_states)  # max state : 3
        self.speed = aircraft_inits["drone_1"]["speed"]
        self.x_range, self.y_range, self.break_spot = (None, None, None)
        self.initial_points = {
            ac_id: ac_value["position"] for ac_id, ac_value in aircraft_inits.items()
        }
        self.latest_ac_pos = {}
        self.latest_veh_pos = {}
        self.latest_cover_radius = {}
        self.veh_trajectories = defaultdict(list)
        self.ac_trajectories = defaultdict(list)
        self.cluster_point = []
        self.total_covered = 0
        self.cover_efficiency = None
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

    def  get_relative_ac_pos(self, aircraft_id, pos) -> List:
        _init_points = self.initial_points[aircraft_id]
        pos_new = [pos[0] - _init_points[0], pos[1] - _init_points[1], pos[2] - _init_points[2]]
        return pos_new

    def  get_relative_pos(self, aircraft_id, pos) -> List:
        _init_points = self.initial_points[aircraft_id]
        pos_new = [pos[0] - _init_points[0], pos[1] - _init_points[1]]
        return pos_new

    def compute_cover_center(self, veh_pos, radius: float) -> tuple[floating[Any], int]:
        tree = KDTree(veh_pos, leaf_size=len(veh_pos) * 2)
        max_count = 0
        best_center = None
        top = 5 #len(veh_pos) 5, 3
        counts = tree.query_radius(veh_pos, r=radius, count_only=True)
        idx = np.argsort(counts)[-top:]
        for i in idx:
            p = veh_pos[i]
            cnt = counts[i]
            if cnt > max_count:
                max_count = cnt
                best_center = p.copy()

        for i in idx:
            p1 = veh_pos[i]
            neigh = tree.query_radius([p1], r=2 * radius)[0]
            for j in neigh:
                if j <= i:
                    continue
                p2 = veh_pos[j]
                d = np.linalg.norm(p2 - p1)
                if d == 0 or d > 2 * radius:
                    continue
                mid = (p1 + p2) / 2
                h = np.sqrt(radius ** 2 - (d/2) ** 2)
                diff = p2 - p1
                perp = np.array([-diff[1], diff[0]]) * (h / d)
                for c in (mid + perp, mid - perp):
                    cnt = len(tree.query_radius([c], r=radius)[0])
                    if cnt > max_count:
                        max_count = cnt
                        best_center = c.copy()
        return best_center

    # TODO: compare with density method and K-Mean method
    def k_mean_find_cluster(self, veh_pos, n_clusters=2):
        from threadpoolctl import threadpool_limits
        ###########################################################
        # 在loop外面先定义一次
        # mbkm = MiniBatchKMeans(n_clusters=len(veh_pos),
        #                        batch_size=100,
        #                        max_iter=1,
        #                        init='k-means++',)
        # mbkm.fit(veh_pos) #最初的k-mean
        ############################################################
        kmeans = KMeans(
            n_clusters=n_clusters,
            init='k-means++',
            n_init=1,  # 单次初始化加快速度
            max_iter=1  # 单次迭代
        )
        kmeans.fit(veh_pos)

        center = kmeans.cluster_centers_
        labels = kmeans.labels_
        count = np.bincount(labels)
        largest_cluster = np.argmax(count)
        return center[largest_cluster]

    def density_peaks_clustering(self, veh_pos, radius, n_centers=1):
        N = len(veh_pos)
        dist = np.linalg.norm(veh_pos[:, None, :] - veh_pos[None, :, :], axis=2)
        rho = np.sum(dist < radius, axis=1) - 1
        delta = np.zeros(N)
        for i in range(N):
            higher = np.where(rho > rho[i])[0]
            if higher.size > 0:
                delta[i] = np.min(dist[i, higher])
            else:
                delta[i] = np.max(dist[i])
        gamma = rho * delta
        centers = np.argsort(-gamma)[:n_centers]
        center_point = veh_pos[centers]
        center_point = center_point.reshape(-1)
        return center_point

    @staticmethod
    def distance_penalty(dist, cover_radius, p=10):
        ratio = dist / cover_radius
        penalty = -np.log1p(ratio) / np.log1p(p)
        return penalty

    def prune_old_vehicles(self, current_veh_ids):
        # 删掉消失的车辆
        self.latest_veh_pos = {vid: pos for vid, pos in self.latest_veh_pos.items() if vid in current_veh_ids}

    @property
    def action_space(self):
        return gym.spaces.Discrete(8)
    
    @property
    def observation_space(self):
        spaces = {
            "ac_attr": gym.spaces.Box(low=np.zeros((9,)), high=np.ones((9,)), shape=(9,)), # (1,9)
            "relative_vecs": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(20,2), dtype=np.float32), # (1,20,2)
            # "relative_vecs": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(40, ), dtype=np.float32),  # (1,20,2)
            "cover_counts": gym.spaces.Box(low=0, high=np.inf, shape=(1,)), # (1,1)
            # "bound_dist": gym.spaces.Box(low=np.zeros((2,)), high=np.ones((2,)), shape=(2,)), # (1,2)
            "break_spot": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,)), # (1,2)
            "no_vehicles": gym.spaces.Box(low=0, high=1, shape=(1,)), # (1,1)
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

        all_vehicle_ids = list(veh.keys())
        if all_vehicle_ids:
            prefixes = [vid.split('#')[0] for vid in all_vehicle_ids]
            cnt = Counter(prefixes)
            if cnt:
                _, max_count = cnt.most_common(1)[0]
            else:
                max_count = 0
        else:
            max_count = 0

        dist_to_bound = []
        relative_vecs = []
        cover_counts = [0]
        no_veh = 1

        for aircraft_id, aircraft_info in aircraft.items():
            if aircraft_info['aircraft_type'] != 'drone':
                continue
            cover_radius = aircraft_info['cover_radius']
            aircraft_pos = aircraft_info['position']
            ac_pos = self.get_relative_ac_pos(aircraft_id, aircraft_pos)

            self.latest_cover_radius[aircraft_id] = cover_radius
            self.latest_ac_pos[aircraft_id] = ac_pos
            self._pos_set.append(ac_pos)
            self.ac_trajectories[aircraft_id].append(ac_pos)
            vehicle_state = {}

            break_spot_vec = -np.array(ac_pos[:2])

            for vehicle_id, vehicle_info in veh.items():

                vehicle_pos = vehicle_info['position']
                veh_pos = self.get_relative_pos(aircraft_id, vehicle_pos)
                self.veh_trajectories[vehicle_id].append(veh_pos)
                dx = veh_pos[0] - ac_pos[0]
                dy = veh_pos[1] - ac_pos[1]

                self.latest_veh_pos[vehicle_id] = [dx,dy]
                relative_vecs.append([dx, dy])
                # relative_vecs.append(veh_pos)

                dist = math.hypot(dx, dy)
                if dist <= cover_radius:
                    vehicle_state[vehicle_id] = vehicle_info.copy()

            cover_counts = np.array([len(vehicle_state)])
            new_state[aircraft_id] = vehicle_state

            has_veh = any(len(veh_list) > 0 for veh_list in self.latest_veh_pos.values())
            no_veh = 1 if not has_veh else 0
            self.break_spot = break_spot_vec * no_veh
            # self.break_spot = [0,0]

        if max_count > 0:
            self.cover_efficiency = cover_counts / max_count
        else: self.cover_efficiency = None

        if len(relative_vecs) == 0:
            relative_vecs = np.zeros((20, 2))
        else:
            relative_vecs = np.array(relative_vecs[:20])
            if relative_vecs.shape[0] < 20:
                pad = np.zeros((20 - relative_vecs.shape[0], 2))
                relative_vecs = np.vstack((relative_vecs, pad))

        feature_set = {
            "ac_attr": np.array(self._pos_set).reshape(-1), # 无人机历史坐标，对无人机起点的相对坐标
            "relative_vecs": np.array(relative_vecs), # 车辆实时位置，对无人机起点的相对坐标
            # "relative_vecs": np.array(relative_vecs).reshape(-1),
            "cover_counts": cover_counts,
            # "bound_dist": dist_to_bound.reshape(1,-1).squeeze(),
            "break_spot": np.array(self.break_spot).reshape(-1), # 休息点，对无人机起点的相对坐标
            "no_vehicles": np.array(no_veh).reshape(-1) # bool，作为relative_vecs的mask使用
        }

        return feature_set, new_state

    def reward_wrapper(self, states, dones) -> float:
        """自定义 reward 的计算
        """
        reward = 0
        for aircraft_id, vehicle_info in states.items():
            aircraft_pos = self.latest_ac_pos[aircraft_id]
            cover_radius = self.latest_cover_radius[aircraft_id]
            _x, _y, _ = aircraft_pos

            if self.latest_veh_pos:  # 如果路网当中有车辆便存在驱使无人机接近密度最大的车流
                veh_poses = np.array(list(self.latest_veh_pos.values()))
                best_center = self.compute_cover_center(veh_pos=veh_poses, radius=cover_radius) # KD-Tree Method
                # best_center = self.density_peaks_clustering(veh_poses, cover_radius) # Density Map Method
                # if len(veh_poses) > 2: # Vehicle amount must greater than cluster amount
                #     best_center = self.k_mean_find_cluster(veh_poses) # KMeans Method
                if best_center is not None:
                    self.cluster_point.append(best_center + np.array(aircraft_pos[:2]))
                m_dist = np.linalg.norm(best_center)
                if m_dist <= cover_radius + 10:
                    if len(vehicle_info) != 0:
                        reward += len(vehicle_info)
                else:
                    penalty = self.distance_penalty(m_dist - cover_radius, cover_radius, p=25)
                    reward += penalty * 0.45

            else:  # 无车环境让无人机返回休息点
                spot_dist = np.linalg.norm([_x, _y])
                if spot_dist <= cover_radius:
                    reward += 2  # 1, 2, 3
                else:
                    penalty = self.distance_penalty(spot_dist - cover_radius, cover_radius, p=25)
                    reward += penalty * 0.45

            bound_penalty = 0
            # 靠近边界100米每步扣5分
            if abs(_y) > (self.y_range - 100):
                bound_penalty = -5
                reward += bound_penalty
            if abs(_x) > (self.x_range - 100):
                bound_penalty = -5
                reward += bound_penalty
            # 超过边界扣100分并结束episode
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
