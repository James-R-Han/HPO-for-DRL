import os
import shutil
import pickle
from itertools import product

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.env_util import make_vec_env

from configs.config_base import ConfigBase
from configs.config_HPO import ConfigHPO
from configs.config_training import ConfigTraining


class HPO_for_DRL:
    
    def __init__(self, config_HPO, config_training):
        self.config_HPO = config_HPO
        self.config_training = config_training
        
        self.device = config_training.device
        self.HPO_algo = config_training.HPO_algo
        self.DRL_algo = config_training.DRL_algo
        self.DRL_algo_params_bounds = config_HPO.bounds[self.DRL_algo]
        self.env_id = config_training.env_id

        self.training_timesteps = config_training.training_timesteps
        self.num_eval_episodes = config_training.num_eval_episodes

        save_path = config_training.save_path
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        # Copy the config folder to the save path
        config_folder = "/home/jha/HPO-for-DRL/configs"
        shutil.copytree(config_folder, os.path.join(save_path, "configs"), dirs_exist_ok=True)
        
        # Main training env (use a DummyVecEnv for stable_baselines)
        if self.DRL_algo == "PPO":
            self.env = make_vec_env(self.env_id, n_envs=4)
        else:
            env = gym.make(self.env_id)
            self.env = DummyVecEnv([lambda: env])

        # Execute the HPO algorithm
        if self.HPO_algo == "RS":
            self.RS()
        elif self.HPO_algo == "GS":
            self.GS()
        elif self.HPO_algo == "DE":
            self.DE()
        elif self.HPO_algo == "CMAES":
            self.CMAES()
            
    def RS(self):
        num_runs = self.config_HPO.RS_config["num_runs"]
        
        reward_over_runs = []
        for i in tqdm(range(num_runs)):
            hparams = {}
            for hparam, (low, high) in self.DRL_algo_params_bounds.items():
                if hparam in ["n_steps", "batch_size"]:
                    hparams[hparam] = int(np.random.uniform(low, high))
                else:
                    hparams[hparam] = np.random.uniform(low, high)
                    
            # Save sampled hyperparams
            with open(os.path.join(self.save_path, f"hyperparams_{i}.pkl"), "wb") as f:
                pickle.dump(hparams, f)
            
            # Train the model with the sampled hyperparameters
            model = self.train_model(self.env, self.DRL_algo, hparams)
            # Evaluate the model (this will also record a video for run i)
            mean_reward = self.evaluate_model(model, self.num_eval_episodes, run_idx=i)
            reward_over_runs.append(mean_reward)
    
        # Save the results
        reward_over_runs = np.array(reward_over_runs)
        np.save(os.path.join(self.save_path, "reward_over_runs.npy"), reward_over_runs)


    def GS(self):
        # Get the grid search configuration for the selected algorithm
        GS_config = self.config_HPO.GS_config[self.DRL_algo]
        
        # Generate all combinations of hyperparameters
        param_grid = {}
        for hparam, splits in GS_config.items():
            low, high = self.DRL_algo_params_bounds[hparam]
            # we want to take the middle splits of the range. Ie. there is even gap between every elements and wrt the bounds
            # Exclude the first and last values to avoid extremes
            gridded_values = np.linspace(low, high, splits+2)[1:-1]
            if hparam in ["n_steps", "batch_size"]:
                gridded_values = np.round(gridded_values).astype(int)
                # I'm writing this for correctness but all hparam bounds are convex sets
                gridded_values = np.clip(gridded_values, low, high) 
            param_grid[hparam] = gridded_values
        
        
        # Create a list of all combinations of hyperparameters
        all_combinations = list(product(*param_grid.values()))
        
        reward_over_runs = []
        for i, combination in tqdm(enumerate(all_combinations)):
            hparams = dict(zip(param_grid.keys(), combination))
            # Save sampled hyperparams
            with open(os.path.join(self.save_path, f"hyperparams_{i}.pkl"), "wb") as f:
                pickle.dump(hparams, f)
            
            # Train the model with the sampled hyperparameters
            model = self.train_model(self.env, self.DRL_algo, hparams)
            # Evaluate the model (this will also record a video for run i)
            mean_reward = self.evaluate_model(model, self.num_eval_episodes, run_idx=i)
            reward_over_runs.append(mean_reward)

        # Save the results
        reward_over_runs = np.array(reward_over_runs)
        np.save(os.path.join(self.save_path, "reward_over_runs.npy"), reward_over_runs)

    def train_model(self, env, DRL_algo, hparams):
        """
        Train the model with the given hyperparameters.
        """
        if DRL_algo == "PPO":
            # Note SB3 throws me a warning that we should use cpu for PPO
            model = PPO("MlpPolicy", env, verbose=1, device='cpu', tensorboard_log=os.path.join(self.save_path, "tb_logs"), **hparams)
        elif DRL_algo == "DQN":
            model = DQN("MlpPolicy", env, verbose=1, device=self.device, tensorboard_log=os.path.join(self.save_path, "tb_logs"), **hparams)
        
        # Train the model
        model.learn(total_timesteps=self.training_timesteps)
        
        return model

    def evaluate_model(self, model, num_eval_episodes, run_idx) -> float:
        # Create a fresh environment just for evaluation & video recording
        eval_env = DummyVecEnv([lambda: gym.make(self.env_id, render_mode="rgb_array")])
        
        # Where to save the video
        video_folder = os.path.join(self.save_path, "videos")
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



if __name__ == "__main__":
    config_HPO = ConfigHPO()
    config_training = ConfigTraining()

    HPO_for_DRL(config_HPO, config_training)
