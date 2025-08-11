'''
@Author: WANG Maonan
@Date: 2023-09-14 13:47:34
@Description: Check aircraft and vehicle ENV
+ Two types of aircraft, custom image
@LastEditTime: 2023-09-25 14:20:32
'''
import math
import numpy as np
from loguru import logger
from typing import List
from tshub.utils.init_log import set_logger
from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.format_dict import dict_to_str
from env_utils.ac_env import ACEnvironment
from env_utils.ac_wrapper import ACEnvWrapper
from env_utils.vis_snir import render_map

path_convert = get_abs_path(__file__)
set_logger(path_convert('./'))

def custom_update_cover_radius(position:List[float], communication_range:float) -> float:
    """自定义的更新地面覆盖半径的方法, 在这里实现您的自定义逻辑

    Args:
        position (List[float]): 飞行器的坐标, (x,y,z)
        communication_range (float): 飞行器的通行范围
    """
    height = position[2]
    cover_radius = height / np.tan(math.radians(75/2))
    return cover_radius

def make_env(
        num_seconds:int,sumo_cfg:str,use_gui:bool,
        log_file:str, aircraft_inits:dict,
        ):

    ac_env = ACEnvironment(
        sumo_cfg=sumo_cfg,
        num_seconds=500,
        aircraft_inits=aircraft_inits,
        use_gui=True
    )
    ac_wrapper = ACEnvWrapper(env=ac_env, aircraft_inits=aircraft_inits)
    # ac_env = Monitor(ac_wrapper, filename=f'{log_file}/{env_index}')
    return ac_wrapper


if __name__ == '__main__':
    # sumo_cfg = path_convert("./sumo_envs/LONG_GANG/env/osm.sumocfg")
    sumo_cfg = path_convert("./sumo_envs/Nguyen_Dupuis/ND_env/resize_rectangle.sumocfg")

    aircraft_inits = {
        'drone_1': {
            "aircraft_type": "drone",
            "action_type": "horizontal_movement",
            "position": (0, 0, 20), "speed": 10, "heading": (1, 1, 0), "communication_range": 10,
            # "position": (1750, 1000, 50), "speed": 10, "heading": (1, 1, 0), "communication_range": 50,
            "if_sumo_visualization": True, "img_file": path_convert('./asset/drone.png'),
            "custom_update_cover_radius":custom_update_cover_radius # 使用自定义覆盖范围的计算
        },
    }

    ac_env = make_env(
        sumo_cfg=sumo_cfg,
        num_seconds=100,
        aircraft_inits=aircraft_inits,
        log_file="./check_log",
        use_gui=False
    )
    # ac_env_wrapper = ACEnvWrapper(env=ac_env, aircraft_inits=aircraft_inits)

    done = False
    ac_env.reset()
    # import random
    while not done:
        action = {
            "drone_1": (0, 0),
        }
        states, rewards, truncated, done, infos = ac_env.step(action=action)
        # logger.info(f'SIM: State: \n{dict_to_str(states)} \nReward:\n {rewards}')

    # render_map(
    #     trajectories=ac_env.veh_trajectories,
    #     veh_trajectories=ac_env.ac_trajectories,
    #     cluster_point = ac_env.cluster_point,
    #     img_path=path_convert("./test_trajectories.jpg")
    # )
    ac_env.close()