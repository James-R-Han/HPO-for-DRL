import numpy as np
import os
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import pickle

save_path = "/home/jha/HPO-for-DRL/analysis/"
os.makedirs(save_path, exist_ok=True)

envs = ["CartPole-v1"]#, "LunarLander-v3"]#, "MountainCar-v0"]#
DRL_algos = ["DQN", "PPO"]
HPO_algos = ["RS", "GS", "DE", "CMAES"]

def normalize_reward(target, base):
    return (target-base)/base

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

            reward_of_best_hparams_path = f"/home/jha/HPO-for-DRL/results/{env}/{identifier}/reward_of_best_hparams.txt"
            with open(reward_of_best_hparams_path, 'r') as file:
                reward_of_best_hparams = float(file.read().strip())
            
            HPO_algo_to_reward_info[HPO_algo] = {
                "max": max_reward,
                "min": min_reward,
                "mean": mean_reward,
                "std": std_reward,
                "max_best_hparams_testing": reward_of_best_hparams,
                "reward_over_search": reward_over_runs,
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
        plt.savefig(f"{save_path_specific}/mean_reward_search.png")
        
        # max reward during search
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
        plt.savefig(f"{save_path_specific}/max_reward_search.png")

        # max reward of best hparams
        plt.figure(figsize=(10, 6))
        sns.set(style="whitegrid")
        sns.set_palette("husl")
        keys = list(HPO_algo_to_reward_info.keys())
        values = [HPO_algo_to_reward_info[k]["max_best_hparams_testing"] for k in keys]
        plt.bar(keys, values, capsize=5)
        plt.xlabel("HPO Algorithm")
        plt.ylabel("Reward of Best Hparams (Test)")
        plt.title(f"Reward of Best Hparams on Test Env for {env} with {DRL_algo}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{save_path_specific}/best_hparams_test_reward.png")
        

        # plot reward over the HPO search
        plt.figure(figsize=(10, 6))
        sns.set(style="whitegrid")
        sns.set_palette("husl")

        window_size = 8
        for HPO_algo in HPO_algos:
            rewards = HPO_algo_to_reward_info[HPO_algo]["reward_over_search"]
            smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            # Add padding to make the lengths match for plotting
            padding = [smoothed_rewards[0]] * (window_size - 1)
            smoothed_rewards = np.concatenate([padding, smoothed_rewards])
            plt.plot(smoothed_rewards, label=HPO_algo)

        plt.xlabel("HPO Iteration")
        plt.ylabel("Reward")
        plt.title(f"Reward Progression During HPO for {env} with {DRL_algo}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_path_specific}/reward_progression.png")
        plt.close()
        


        rs_best_test = HPO_algo_to_reward_info["RS"]["max_best_hparams_testing"]
        
        # 1. Normalized Mean Reward (vs. best RS)
        plt.figure(figsize=(10, 6))
        sns.set(style="whitegrid")
        sns.set_palette("husl")

        keys = list(HPO_algo_to_reward_info.keys())
        norm_values = [
            normalize_reward(HPO_algo_to_reward_info[k]["mean"], rs_best_test)
            for k in keys
        ]
        plt.bar(keys, norm_values)
        plt.xlabel("HPO Algorithm")
        plt.ylabel("Normalized Mean Reward\n(w.r.t. Best RS)")
        plt.title(f"{env}-{DRL_algo}: Mean Reward vs. Best RS")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{save_path_specific}/norm_mean_reward_vs_RSbest.png")
        plt.close()

        # 2. Normalized Max Reward (vs. best RS)
        plt.figure(figsize=(10, 6))
        sns.set(style="whitegrid")
        sns.set_palette("husl")

        norm_values = [
            normalize_reward(HPO_algo_to_reward_info[k]["max"], rs_best_test)
            for k in keys
        ]
        plt.bar(keys, norm_values)
        plt.xlabel("HPO Algorithm")
        plt.ylabel("Normalized Max Reward\n(w.r.t. Best RS)")
        plt.title(f"{env}-{DRL_algo}: Max Reward vs. Best RS")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{save_path_specific}/norm_max_reward_vs_RSbest.png")
        plt.close()

        # 3. Normalized Best Hparams Reward (vs. best RS)
        plt.figure(figsize=(10, 6))
        sns.set(style="whitegrid")
        sns.set_palette("husl")

        norm_values = [
            normalize_reward(HPO_algo_to_reward_info[k]["max_best_hparams_testing"], rs_best_test)
            for k in keys
        ]
        plt.bar(keys, norm_values)
        plt.xlabel("HPO Algorithm")
        plt.ylabel("Normalized Best Hparams Reward\n(w.r.t. Best RS)")
        plt.title(f"{env}-{DRL_algo}: Best Hparams Reward vs. Best RS")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{save_path_specific}/norm_best_hparams_test_reward_vs_RSbest.png")
        plt.close()

        # 4. Normalized Progression Over HPO Search
        #    Option A: Compare each iteration to the single best RS (all iters).
        plt.figure(figsize=(10, 6))
        sns.set(style="whitegrid")
        sns.set_palette("husl")

        for HPO_algo in HPO_algos:
            rewards = HPO_algo_to_reward_info[HPO_algo]["reward_over_search"]
            # Compare each iteration to best RS hparams
            normed_rewards = [normalize_reward(r, rs_best_test) for r in rewards]
            # Smooth
            smoothed = np.convolve(normed_rewards, np.ones(window_size)/window_size, mode='valid')
            padding = [smoothed[0]]*(window_size-1)
            smoothed = np.concatenate([padding, smoothed])
            plt.plot(smoothed, label=HPO_algo)

        plt.xlabel("HPO Iteration")
        plt.ylabel("Normalized Reward\n(w.r.t. Best RS)")
        plt.title(f"{env}-{DRL_algo}: Reward Progression vs. Best RS")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_path_specific}/norm_reward_progression_vs_RSbest.png")
        plt.close()

