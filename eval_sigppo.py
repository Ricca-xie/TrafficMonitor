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
from loguru import logger
from tshub.utils.get_abs_path import get_abs_path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from env_utils.make_tsc_env import make_env
from env_utils.vis_snir import render_map

path_convert = get_abs_path(__file__)
logger.remove()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters.')
    parser.add_argument('--env_name', type=str, default='detroit_UAM', help='The name of environment: detroit_UAM,detroit_UAM_noi, berlin_UAM')
    parser.add_argument('--passenger_len', type=int, default=5, help='The number of passengers:4,6')
    parser.add_argument('--passenger_type', type=str, default="real_time", help='fix or real time')
    parser.add_argument('--speed', type=int, default=160, help="100,160,320")  # speed决定了地图的scale
    parser.add_argument('--snir_min', type=int, default=-17, help='The threshold of SNIR') # 最小SNIR值，小于这个值的乘客不参与训练
    parser.add_argument('--policy_model', type=str, default="fusion_noi_models_4",
                        help='policy network: baseline_models or fusion_models') # fusion_models_4
    parser.add_argument('--features_dim', type=int, default=8192, help='The dimension of output features')
    parser.add_argument('--num_seconds', type=int, default=2500, help='exploration steps')
    parser.add_argument('--n_steps', type=int, default=2000, help='The number of steps in each environment')
    parser.add_argument('--lr', type=float, default=5e-5, help='The learning rate of PPO')
    parser.add_argument('--batch_size', type=int, default=64, help='The batch size of PPO')
    parser.add_argument('--cuda_id', type=int, default=0, help='The id of cuda device')
    args = parser.parse_args()  # Parse the arguments
    device = f'cuda:{args.cuda_id}' if torch.cuda.is_available() else 'cpu'
    # #########
    # Init Env
    # #########

    log_path = path_convert('./eval_log/')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    sumo_cfg = path_convert(f"./sumo_envs/{args.env_name}/{args.env_name}.sumocfg")
    net_file = path_convert(f"./sumo_envs/{args.env_name}/{args.env_name}.net.xml")

    snir_files = {
        '100': path_convert(f"./sumo_envs/{args.env_name}/{args.env_name}_SNIR_100.txt"),
        # xxx, 这里可以添加不同高度的文件
    }

    if args.env_name == "berlin_UAM":
        aircraft_inits = {
            'drone_1': {
                "aircraft_type": "drone",
                "action_type": "horizontal_multi_movement",
                "position": (2600, 1500, 100), "speed": 100, "heading": (1, 1, 0), "communication_range": 120,
                "if_sumo_visualization": False, "img_file": path_convert('./asset/drone.png'),
                # "custom_update_cover_radius":custom_update_cover_radius # 使用自定义覆盖范围的计算
            },
        }
        passenger_seq = {
            'num_seconds': args.num_seconds + 12,
            "passenger_seq": {
                # "step_0": [[(2220, 1400), (2000, 1100)], [(1400, 1680), (1270, 450)], [(650, 1000), (1700, 620)]], # models_4
                "step_0": [[(2220, 1400), (1850, 1100)], [(1450, 2000), (1270, 450)], [(650, 1000), (1700, 620)]], # models_3
                "step_5": [[(1010, 2220), (400, 1200)]], # "step_5": [[(1130, 1700), (400, 1200)]],"step_10": [[(1130, 1700), (400, 2300)]],
            }
        }


    else:
        aircraft_inits = {
            'drone_1': {
                "aircraft_type": "drone",
                "action_type": "horizontal_multi_movement",
                "position": (6460, 9750, 100), "speed": args.speed, "heading": (1, 1, 0), "communication_range": 120,
                "if_sumo_visualization": False, "img_file": path_convert('./asset/drone.png'),
            },
        }  # (6200, 9000, 100)
        passenger_seq = {
            'num_seconds': args.num_seconds + 10,
            # "passenger_seq": {  # (9570, 7000)
            #     # "step_0": [[(2220, 1400), (2000, 1100)], [(1400, 1680), (1270, 450)], [(650, 1000), (1700, 620)]], # models_4
            #     "step_0": [[(6594, 7563), (9570, 7210)], [(7494, 5855), (2175, 6906)]],
            #     "step_5": [[(3400, 5520), (5100, 2500)]],
            #     "step_8": [[(1500, 6530), (2234, 3850)]],
            #     "step_15": [[(3781, 3438), (7900, 2430)]],
            # }  # 新的坐标
            "passenger_seq": {  # 5 users
                # "step_0": [[(2220, 1400), (2000, 1100)], [(1400, 1680), (1270, 450)], [(650, 1000), (1700, 620)]], # models_4
                "step_0": [[(9710, 8100), (6594, 7563)], [(8894, 7055), (3400, 5720)]],
                # (6594, 7563), (9570, 7210)], [(7494, 5855), (3400, 5520)
                "step_5": [[(4951, 6238), (934, 4850)]],  # (4951, 6238), (934, 4850)
                "step_8": [[(2175, 6906), (5100, 2500)]],  # (2175, 6906), (5100, 2500)
                "step_15": [[(2234, 3850), (7900, 2430)]],  # (2234, 3850), (7900, 2430)
                # "step_93": [[(9570, 6210), (6900, 4430)]], # (2234, 3850), (7900, 2430)
            }  # 新的坐标
        }
        # passenger_seq = {
        #     'num_seconds': 10,
        #     "passenger_seq": { #(9570, 7000)
        #         # "step_0": [[(2220, 1400), (2000, 1100)], [(1400, 1680), (1270, 450)], [(650, 1000), (1700, 620)]], # models_4
        #         "step_0": [[(6594, 7563), (9570, 6400)], [(9200, 7350), (5700, 4580)], [(7494, 5855), (1375, 5420)]],
        #         "step_5": [[(3500, 5270), (5100, 2500)]],
        #         "step_8": [[(2175, 6906), (2234, 3850)]],
        #         "step_15": [[(3781, 3438), (7830, 2430)]],
        #     }
        # }


    params = {
        'num_seconds': args.num_seconds,
        'sumo_cfg': sumo_cfg,
        'use_gui': False,
        "net_file": net_file,
        "snir_files": snir_files,
        'log_file': log_path,
        'aircraft_inits': aircraft_inits,
        'passenger_seq': passenger_seq,
        "snir_min": args.snir_min,
    }
    param_name = f'explore_{args.num_seconds}_n_steps_{args.n_steps}_lr_{str(args.lr)}_batch_size_{args.batch_size}'
    env = SubprocVecEnv([make_env(env_index=f'{i}', **params) for i in range(1)])
    env = VecNormalize.load(load_path=path_convert(f'./{args.passenger_type}/{args.env_name}/P{args.passenger_len}/speed_{args.speed}/snir_{args.snir_min}/{args.policy_model}/{param_name}/models/best_vec_normalize.pkl'), venv=env)
    # env = VecNormalize.load(load_path=path_convert(f'./{args.passenger_type}/{args.env_name}/P{args.passenger_len}/speed_{args.speed}/snir_{args.snir_min}/{args.policy_model}/{param_name}/models/last_vec_normalize.pkl'), venv=env)
    # env = VecNormalize.load(load_path=path_convert(f'./{args.passenger_type}/{args.env_name}/P{args.passenger_len}/speed_{args.speed}/snir_{args.snir_min}/{args.policy_model}/{param_name}/models/rl_model_vecnormalize_960000_steps.pkl'), venv=env)

    env.training = False  # 测试的时候不要更新
    env.norm_reward = False

    model_path = path_convert(f'./{args.passenger_type}/{args.env_name}/P{args.passenger_len}/speed_{args.speed}/snir_{args.snir_min}/{args.policy_model}/{param_name}/models/best_model.zip')
    # model_path = path_convert(f'./{args.passenger_type}/{args.env_name}/P{args.passenger_len}/speed_{args.speed}/snir_{args.snir_min}/{args.policy_model}/{param_name}/models/last_rl_model.zip')
    # model_path = path_convert(f'./{args.passenger_type}/{args.env_name}/P{args.passenger_len}/speed_{args.speed}/snir_{args.snir_min}/{args.policy_model}/{param_name}/models/rl_model_1680000_steps.zip')
    model = PPO.load(model_path, env=env, device=device)

    # 使用模型进行测试
    passenger_len = args.passenger_len
    obs = env.reset()
    dones = False  # 默认是 False
    total_reward = 0.0
    total_steps = 0
    trajectory = None  # 轨迹
    ac_seat_flag_list = [env.get_attr("ac_seat_flag")[0]]
    wait_time = np.zeros(passenger_len)
    fly_time = np.zeros(passenger_len)
    x_min = env.get_attr("x_min")[0]
    y_min = env.get_attr("y_min")[0]
    # import time;time.sleep(5)
    while not dones:
        action, _state = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        ac_seat_flag = env.get_attr("ac_seat_flag")[0]
        ac_seat_flag_list.append(ac_seat_flag)
        total_reward += rewards
        total_steps += 1

        print(rewards)
        if rewards < -60:
            print("error")
        # 计算等待时间
        for i in range(passenger_len):
            on_ac_flag = env.get_attr("padded_passen_attr")[0][i, -2] # 是否在飞行器上
            is_served_flag = env.get_attr("padded_passen_attr")[0][i, -1] # 是否被服务过
            if not on_ac_flag and not is_served_flag:
                wait_time[i] += 1

            # 计算每个用户飞行时间
            if on_ac_flag and not is_served_flag:  # 如果在飞机上但没有被服务完成
                fly_time[i] += 1
    before_time = int(list(passenger_seq['passenger_seq'].keys())[-1].split("_")[-1])
    wait_time[-1] -= before_time # 减去还没有到达的时间
    if env.get_attr("actru_trajectory")[0]:
        trajectory = env.get_attr("actru_trajectory")[0]

    # noise grid
    grid_z = env.get_attr("grid_z")[0]
    #
    # grid_z1 = grid_z[625:, :645]
    # grid_z2 = grid_z[:205, 350:700]
    # grid_z3 = grid_z[:550, 1000:]
    # grid_z4 = grid_z[900:, 1020:]
    #
    # noise_z = np.random.normal(-1, np.sqrt(32), grid_z.shape)
    # noise_z[625:, :645] = np.random.normal(-0.5, 2, grid_z1.shape)
    # noise_z[:205, 350:700] = np.random.normal(-0.5, 2, grid_z2.shape)
    # noise_z[:550, 1000:] = np.random.normal(-0.5, 2, grid_z3.shape)
    # noise_z[900:, 1020:] = np.random.normal(-0.5, 2, grid_z4.shape)
    # 进行可视化
    render_map(
        x_min=env.get_attr("x_min")[0],
        y_min=env.get_attr("y_min")[0],
        x_max=env.get_attr("x_max")[0],
        y_max=env.get_attr("y_max")[0],
        resolution=env.get_attr("resolution")[0],
        grid_z=env.get_attr("noise_grid_z")[0][0], # env.get_attr("grid_z")[0], env.get_attr("noise_grid_z")[0],
        trajectories=trajectory,
        goal_points=env.get_attr("goal_seq")[0],
        speed=aircraft_inits["drone_1"]["speed"],  # 60: 50*50, 100: 30*30 aircraft_inits["drone_1"]["speed"]
        snir_threshold=args.snir_min, # args.snir_min
        img_path=path_convert(f'./{args.env_name}_{args.passenger_type}_P{args.passenger_len}_S{args.snir_min}_{args.policy_model}_{param_name}_snir.png')
    )

    env.close()
    # print(f'trajectory, {trajectory}.')
    print(f'累积奖励为, {total_reward}.')
    print(f"total steps:{total_steps}.")
    print(f"total distance:{total_steps * aircraft_inits['drone_1']['speed']}.")
    print(f"empty loaded rate:{ac_seat_flag_list.count(0) / len(ac_seat_flag_list)}")
    print(f"average waiting time:{wait_time}, {wait_time.mean()}")
    print(f"average fly time: {fly_time}, {fly_time.mean()}")
    print(f'average total time consumed:{(wait_time + fly_time).mean()}')


