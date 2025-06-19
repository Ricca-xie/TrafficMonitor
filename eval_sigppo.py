'''
@Author: Ricca
@Date: 2024-07-16
@Description: 使用训练好的 RL Agent 进行测试
@LastEditTime:
'''
import argparse
import torch
import os
import numpy as np
import math

from loguru import logger
from tshub.utils.get_abs_path import get_abs_path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from env_utils.make_tsc_env import make_env
# from env_utils.vis_snir import render_map
from typing import List

path_convert = get_abs_path(__file__)
logger.remove()


def custom_update_cover_radius(position:List[float], communication_range:float) -> float:
    """自定义的更新地面覆盖半径的方法, 在这里实现您的自定义逻辑

    Args:
        position (List[float]): 飞行器的坐标, (x,y,z)
        communication_range (float): 飞行器的通行范围
    """
    height = position[2]
    cover_radius = height / np.tan(math.radians(75/2))
    return cover_radius

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters.')
    parser.add_argument('--env_name', type=str, default="LONG_GANG", help='The name of environment')
    parser.add_argument('--speed', type=int, default=160, help="100,160,320") # speed决定了地图的scale
    parser.add_argument('--num_envs', type=int, default=1, help='The number of environments')
    parser.add_argument('--policy_model', type=str, default="baseline", help='policy network: baseline_models or fusion_models_0')
    parser.add_argument('--features_dim', type=int, default=512, help='The dimension of output features 64')
    parser.add_argument('--num_seconds', type=int, default=300, help='exploration steps')
    parser.add_argument('--n_steps', type=int, default=512, help='The number of steps in each environment') #500
    parser.add_argument('--lr', type=float, default=5e-4, help='The learning rate of PPO') #5e-5
    parser.add_argument('--batch_size', type=int, default=32, help='The batch size of PPO') # 350
    # parser.add_argument('--ent_coef', type=float, default=0.05, help='entropy coefficient')
    parser.add_argument('--cuda_id', type=int, default=0, help='The id of cuda device')
    args = parser.parse_args()  # Parse the arguments
    device = f'cuda:{args.cuda_id}' if torch.cuda.is_available() else 'cpu'
    # #########
    # Init Env
    # #########

    log_path = path_convert('./eval_log/')
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    sumo_cfg = path_convert(f"./sumo_envs/{args.env_name}/env/osm.sumocfg")
    # net_file = path_convert(f"./sumo_envs/{args.env_name}/{args.env_name}.net.xml")


    aircraft_inits = {
        'drone_1': {
            "aircraft_type": "drone",
            "action_type": "horizontal_movement", # combined_movement
            "position": (1750, 1000, 50), "speed": 15, "heading": (1, 1, 0), "communication_range": 50,
            "if_sumo_visualization": True, "img_file": path_convert('./asset/drone.png'),
            "custom_update_cover_radius": custom_update_cover_radius  # 使用自定义覆盖范围的计算
        },
    }

    params = {
        'num_seconds': args.num_seconds,
        'sumo_cfg': sumo_cfg,
        'use_gui': True,
        # "net_file": net_file,
        'log_file': log_path,
        'aircraft_inits': aircraft_inits,
    }
    param_name = f'explore_{args.num_seconds}_n_steps_{args.n_steps}_lr_{str(args.lr)}_batch_size_{args.batch_size}'

    env = SubprocVecEnv([make_env(env_index=f'{i}', **params) for i in range(args.num_envs)])  # multiprocess
    env = VecNormalize(env, norm_obs=False, norm_reward=True)

    env.training = False  # 测试的时候不要更新
    env.norm_reward = False

    #model_path = path_convert(f'./{args.passenger_type}/{args.env_name}/P{args.passenger_len}/speed_{args.speed}/snir_{args.snir_min}/{args.policy_model}/{param_name}/models/best_model.zip')
    model_path = path_convert(f'Result/{args.env_name}/speed_{args.speed}/{args.policy_model}/{param_name}/models/best_model.zip')
    model = PPO.load(model_path, env=env, device=device)

    # 使用模型进行测试
    obs = env.reset()
    dones = False  # 默认是 False
    total_reward = 0.0
    total_steps = 0
    # trajectory = None  # 轨迹

    while not dones:
        action, _state = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        total_reward += rewards
        total_steps += 1
        print(rewards)

    # x_min = env.get_attr("x_min")[0]
    # y_min = env.get_attr("y_min")[0]

    # noise grid
    # grid_z = env.get_attr("grid_z")[0]
    # 进行可视化
    # render_map(
    #     x_min=env.get_attr("x_min")[0],
    #     y_min=env.get_attr("y_min")[0],
    #     x_max=env.get_attr("x_max")[0],
    #     y_max=env.get_attr("y_max")[0],
    #     resolution=env.get_attr("resolution")[0],
    #     grid_z=env.get_attr("noise_grid_z")[0][0], # env.get_attr("grid_z")[0], env.get_attr("noise_grid_z")[0],
    #     trajectories=trajectory,
    #     goal_points=env.get_attr("goal_seq")[0],
    #     speed=aircraft_inits["drone_1"]["speed"],  # 60: 50*50, 100: 30*30 aircraft_inits["drone_1"]["speed"]
    #     snir_threshold=args.snir_min, # args.snir_min
    #     img_path=path_convert(f'./{args.env_name}_{args.passenger_type}_P{args.passenger_len}_S{args.snir_min}_{args.policy_model}_{param_name}_snir.png')
    # )

    env.close()
    # print(f'trajectory, {trajectory}.')
    print(f'累积奖励为, {total_reward}.')
    print(f"total steps:{total_steps}.")
    # print(f"total distance:{total_steps * aircraft_inits['drone_1']['speed']}.")
    # print(f"empty loaded rate:{ac_seat_flag_list.count(0) / len(ac_seat_flag_list)}")
    # print(f"average waiting time:{wait_time}, {wait_time.mean()}")
    # print(f"average fly time: {fly_time}, {fly_time.mean()}")
    # print(f'average total time consumed:{(wait_time + fly_time).mean()}')


