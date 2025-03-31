import numpy as np
import os
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import pickle

save_path = "/home/jha/HPO-for-DRL/analysis/"
os.makedirs(save_path, exist_ok=True)

envs = ["CartPole-v1"]#, "MountainCar-v0", "LunarLander-v2"]#
DRL_algos = ["DQN", "PPO"]
HPO_algos = ["RS", "GS"]#, "DE", "CMAES"]

for env in envs:
    for DRL_algo in DRL_algos:
        HPO_algo_to_reward_info = {}
        for HPO_algo in HPO_algos:
            identifier = f"{DRL_algo}_{env}_{HPO_algo}"
            reward_path = f"/home/jha/HPO-for-DRL/results/{env}/{identifier}/reward_over_runs.npy"
            

            reward_over_runs = np.load(reward_path)
            max_reward = np.max(reward_over_runs)
            min_reward = np.min(reward_over_runs)
            mean_reward = np.mean(reward_over_runs)
            std_reward = np.std(reward_over_runs)
            
            HPO_algo_to_reward_info[HPO_algo] = {
                "max": max_reward,
                "min": min_reward,
                "mean": mean_reward,
                "std": std_reward
            }
        
        # plot the results
        plt.figure(figsize=(10, 6))
        sns.set(style="whitegrid")
        sns.set_palette("husl")
        keys = list(HPO_algo_to_reward_info.keys())
        values = [HPO_algo_to_reward_info[k]["mean"] for k in keys]
        plt.bar(keys, values,  capsize=5)
        
        plt.xlabel("HPO Algorithm")
        plt.ylabel("Mean Reward")
        plt.title(f"Mean Reward for {env} with {DRL_algo}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        save_path_specific = f"{save_path}/{env}_{DRL_algo}"
        os.makedirs(save_path_specific, exist_ok=True)
        plt.savefig(f"{save_path_specific}/mean_reward.png")
        
        # max reward
        plt.figure(figsize=(10, 6))
        sns.set(style="whitegrid")
        sns.set_palette("husl")
        keys = list(HPO_algo_to_reward_info.keys())
        values = [HPO_algo_to_reward_info[k]["max"] for k in keys]
        plt.bar(keys, values, capsize=5)
        plt.xlabel("HPO Algorithm")
        plt.ylabel("Max Reward")
        plt.title(f"Max Reward for {env} with {DRL_algo}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{save_path_specific}/max_reward.png")
        
        