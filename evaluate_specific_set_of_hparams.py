import os
import shutil
import pickle
from itertools import product
from copy import deepcopy
import random

import gymnasium as gym
import numpy as np
from tqdm import tqdm
import torch
from cmaes import CMA

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.env_util import make_vec_env

testing_seeds = [2,3,4,5,6] 
num_eval_episodes = 100
device = 'cuda'
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def evaluate_best_hparams(best_hparams, best_params_idx):
        """
        Evaluate the best hyperparameters on the test set.
        """
        test_env = gym.make(env_id)
        test_env = DummyVecEnv([lambda: test_env])
        
        reward_over_seeds = []
        for seed in testing_seeds:
            set_seed(seed)
            # Train the model with the best hyperparameters
            model = train_model(test_env, DRL_algo, best_hparams)
            
            # Evaluate the model (this will also record a video for run i)
            mean_reward = evaluate_model(model, num_eval_episodes, run_idx=f"test_{seed}")
            reward_over_seeds.append(mean_reward)
        
        reward_of_best_hparams = np.mean(np.array(reward_over_seeds))
        # Save the results
        with open(os.path.join(save_path, "reward_of_best_hparams.txt"), "w") as f:
            f.write(f"{reward_of_best_hparams}")
        with open(os.path.join(save_path, "best_hparams_idx.txt"), "w") as f:
            f.write(f"Best hyperparameters index: {best_params_idx}\n")
            f.write(f"Best hyperparameters: {best_hparams}\n")



def train_model(env, DRL_algo, hparams):
    """
    Train the model with the given hyperparameters.
    """
    
    lr_in_logscale = hparams["learning_rate"]
    hparams_copy = deepcopy(hparams)
    hparams_copy["learning_rate"] = 10 ** lr_in_logscale
        
    if DRL_algo == "PPO":
        # Note SB3 throws me a warning that we should use cpu for PPO
        model = PPO("MlpPolicy", env, verbose=1, device='cpu', tensorboard_log=os.path.join(save_path, "tb_logs"), **hparams_copy)
    elif DRL_algo == "DQN":
        model = DQN("MlpPolicy", env, verbose=1, device=device, tensorboard_log=os.path.join(save_path, "tb_logs"), **hparams_copy)
    
    # Train the model
    model.learn(total_timesteps=training_timesteps)
    
    return model

def evaluate_model(model, num_eval_episodes, run_idx) -> float:
    # Create a fresh environment just for evaluation & video recording
    eval_env = DummyVecEnv([lambda: gym.make(env_id, render_mode="rgb_array")])
    
    # Where to save the video
    video_folder = os.path.join(save_path, "videos")
    os.makedirs(video_folder, exist_ok=True)

    # Wrap the environment with VecVideoRecorder
    eval_env = VecVideoRecorder(
        eval_env,
        video_folder,
        record_video_trigger=lambda step: step == 0,
        video_length=1000,
        name_prefix=f"run_{run_idx}"
    )

    mean_reward, _ = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=num_eval_episodes,
        deterministic=True
    )
    
    eval_env.close()
    return mean_reward

env_id = "CartPole-v1"
training_timesteps = 100000
DRL_algo = "PPO"
save_path = "/home/jha/HPO-for-DRL/results/CartPole-v1/PPO_CartPole-v1_DE"
# DRL_algo = "DQN"
# save_path = "/home/jha/HPO-for-DRL/results/CartPole-v1/DQN_CartPole-v1_RS"
reward_over_runs = np.load(f"{save_path}/reward_over_runs.npy")
best_hparams_idx = np.argmax(reward_over_runs)
best_hparams = pickle.load(open(os.path.join(save_path, f"hyperparams/hyperparams_{best_hparams_idx}.pkl"), "rb"))

print(best_hparams)

evaluate_best_hparams(best_hparams, best_hparams_idx)