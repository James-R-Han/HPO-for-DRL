import os
import shutil
import pickle
from itertools import product
from copy import deepcopy

import gymnasium as gym
import numpy as np
from tqdm import tqdm
import torch
import random

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.env_util import make_vec_env

from configs.config_base import ConfigBase
from configs.config_HPO import ConfigHPO
from configs.config_training import ConfigTraining

from differential_evolution import create_population, mutate_and_crossover


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
        os.makedirs(os.path.join(save_path, "hyperparams"), exist_ok=True)
        
        # Main training env (use a DummyVecEnv for stable_baselines)
        if self.DRL_algo == "PPO":
            self.env = make_vec_env(self.env_id, n_envs=4)
        else:
            env = gym.make(self.env_id)
            self.env = DummyVecEnv([lambda: env])

        # Execute the HPO algorithm
        if self.HPO_algo == "RS":
            best_hparams, best_params_idx = self.RS()
        elif self.HPO_algo == "GS":
            best_hparams, best_params_idx = self.GS()
        elif self.HPO_algo == "DE":
            best_hparams, best_params_idx = self.DE()
        elif self.HPO_algo == "CMAES":
            best_hparams, best_params_idx = self.CMAES()
            
        self.evaluate_best_hparams(best_hparams, best_params_idx)
            
    def set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
    
    def evaluate_best_hparams(self, best_hparams, best_params_idx):
        """
        Evaluate the best hyperparameters on the test set.
        """
        test_env = gym.make(self.env_id)
        test_env = DummyVecEnv([lambda: test_env])
        
        reward_over_seeds = []
        for seed in self.config_training.testing_seeds:
            self.set_seed(seed)
            # Train the model with the best hyperparameters
            model = self.train_model(test_env, self.DRL_algo, best_hparams)
            
            # Evaluate the model (this will also record a video for run i)
            mean_reward = self.evaluate_model(model, self.num_eval_episodes, run_idx=f"test_{seed}")
            reward_over_seeds.append(mean_reward)
        
        reward_of_best_hparams = np.mean(np.array(reward_over_seeds))
        # Save the results
        with open(os.path.join(self.save_path, "reward_of_best_hparams.txt"), "w") as f:
            f.write(f"{reward_of_best_hparams}")
        with open(os.path.join(self.save_path, "best_hparams_idx.txt"), "w") as f:
            f.write(f"Best hyperparameters index: {best_params_idx}\n")
            f.write(f"Best hyperparameters: {best_hparams}\n")
    
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
            with open(os.path.join(self.save_path, f"hyperparams/hyperparams_{i}.pkl"), "wb") as f:
                pickle.dump(hparams, f)
            
            reward_over_seeds = []
            for seed in self.config_training.training_seeds:
                self.set_seed(seed)
                # Train the model with the sampled hyperparameters
            
                # Train the model with the sampled hyperparameters
                model = self.train_model(self.env, self.DRL_algo, hparams)
                # Evaluate the model (this will also record a video for run i)
                mean_reward_on_this_seed = self.evaluate_model(model, self.num_eval_episodes, run_idx=i)
                reward_over_seeds.append(mean_reward_on_this_seed)
                
            mean_reward = np.mean(np.array(reward_over_seeds))
            reward_over_runs.append(mean_reward)
    
            # Save the results every epoch
            np.save(os.path.join(self.save_path, "reward_over_runs.npy"), np.array(reward_over_runs))
            
            with open(os.path.join(self.save_path, "reward_over_runs_num_runs.txt"), "w") as f:
                f.write(f"Number of runs: {len(reward_over_runs)}\n")
                f.write(f"Mean reward over runs: {np.mean(np.array(reward_over_runs))}\n")

        reward_over_runs = np.array(reward_over_runs)
        # Save the best hyperparameters
        best_hparams_idx = np.argmax(reward_over_runs)
        best_hparams = pickle.load(open(os.path.join(self.save_path, f"hyperparams/hyperparams_{best_hparams_idx}.pkl"), "rb"))

        return best_hparams, best_hparams_idx

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
            with open(os.path.join(self.save_path, f"hyperparams/hyperparams_{i}.pkl"), "wb") as f:
                pickle.dump(hparams, f)
            
            reward_over_seeds = []
            for seed in self.config_training.training_seeds:
                self.set_seed(seed)
                # Train the model with the sampled hyperparameters
                model = self.train_model(self.env, self.DRL_algo, hparams)
                # Evaluate the model (this will also record a video for run i)
                mean_reward_on_this_seed = self.evaluate_model(model, self.num_eval_episodes, run_idx=i)
                reward_over_seeds.append(mean_reward_on_this_seed)
                
            mean_reward = np.mean(np.array(reward_over_seeds))
            reward_over_runs.append(mean_reward)

            # Save the results
            np.save(os.path.join(self.save_path, "reward_over_runs.npy"), np.array(reward_over_runs))
            with open(os.path.join(self.save_path, "reward_over_runs_num_runs.txt"), "w") as f:
                f.write(f"Number of runs: {len(reward_over_runs)}\n")
                f.write(f"Mean reward over runs: {np.mean(np.array(reward_over_runs))}\n")
                
        reward_over_runs = np.array(reward_over_runs)
        # Save the best hyperparameters
        best_hparams_idx = np.argmax(reward_over_runs)
        best_hparams = pickle.load(open(os.path.join(self.save_path, f"hyperparams/hyperparams_{best_hparams_idx}.pkl"), "rb"))
        return best_hparams, best_hparams_idx
                

    def DE(self):
        population_size = self.config_HPO.DE_config["pop_size"]
        hparam_order, current_population = create_population(population_size, self.DRL_algo_params_bounds)
        generations = self.config_HPO.DE_config["max_gens"]
        reward_over_runs = [] # takes only best individuals (ie. children that are evaluated may not enter here)
        reward_over_runs_with_children = []
        
        # Evaluate the fitness of the initial population
        fitness_of_population = []
        for idx, (individual_hparam, individual_vector) in enumerate(current_population):
            # save hparams
            with open(os.path.join(self.save_path, f"hyperparams/hyperparams_0_{idx}.pkl"), "wb") as f:
                pickle.dump(individual_hparam, f)
            
            reward_over_seeds = []
            for seed in self.config_training.training_seeds:
                self.set_seed(seed)
                model = self.train_model(self.env, self.DRL_algo, individual_hparam)
                # Evaluate the model (this will also record a video for run i)
                mean_reward_for_this_seed = self.evaluate_model(model, self.num_eval_episodes, run_idx=idx)
                reward_over_seeds.append(mean_reward_for_this_seed)
                
            mean_reward = np.mean(np.array(reward_over_seeds))
            fitness_of_population.append(mean_reward)
            reward_over_runs.append(mean_reward)
            reward_over_runs_with_children.append(mean_reward)
            
            np.save(os.path.join(self.save_path, "reward_over_runs.npy"), np.array(reward_over_runs))
            with open(os.path.join(self.save_path, "reward_over_runs_num_runs.txt"), "w") as f:
                f.write(f"Number of runs: {len(reward_over_runs)}\n")
                f.write(f"Mean reward over runs: {np.mean(np.array(reward_over_runs))}\n")
        
        for generation in tqdm(range(generations)):
            clone_of_current_population = deepcopy(current_population)
            for child in range(population_size):
                # Select three random individuals from the population
                indices = list(range(population_size))
                a, b, c = np.random.choice(indices, 3, replace=False)
                
                F = np.random.uniform(0.5, 1.0) # scale factor
                donor_vector = clone_of_current_population[a][1] + F * (clone_of_current_population[b][1] - clone_of_current_population[c][1])
                
                # Perform crossover and mutation
                mutant_hparams, mutant_vector = mutate_and_crossover(hparam_order, current_population[child][1], donor_vector, self.DRL_algo_params_bounds, self.config_HPO.DE_config["crossover_probability"])
                
                # Evaluate the new individual
                reward_over_seeds = []
                for seed in self.config_training.training_seeds:
                    self.set_seed(seed)
                    model = self.train_model(self.env, self.DRL_algo, mutant_hparams)
                    mean_reward_for_this_seed = self.evaluate_model(model, self.num_eval_episodes, run_idx=(generation+1)*len(current_population)+child)
                    reward_over_seeds.append(mean_reward_for_this_seed)
                mean_reward = np.mean(np.array(reward_over_seeds))
                reward_over_runs_with_children.append(mean_reward) 
                
                
                with open(os.path.join(self.save_path, f"hyperparams/hyperparams_{generation+1}_{child}_mutant.pkl"), "wb") as f:
                        pickle.dump(mutant_hparams, f)
                
                # Select the better of the two individuals
                if mean_reward > fitness_of_population[child]:
                    fitness_of_population[child] = mean_reward
                    current_population[child] = (mutant_hparams, mutant_vector)
                    
                
                reward_over_runs.append(fitness_of_population[child])
                
                with open(os.path.join(self.save_path, f"hyperparams/hyperparams_{generation+1}_{child}.pkl"), "wb") as f:
                    pickle.dump(current_population[child][0], f)
            
            # save per generation
            np.save(os.path.join(self.save_path, "reward_over_runs.npy"), np.array(reward_over_runs))
            with open(os.path.join(self.save_path, "reward_over_runs_num_runs.txt"), "w") as f:
                f.write(f"Number of runs: {len(reward_over_runs)}\n")
                f.write(f"Mean reward over runs: {np.mean(np.array(reward_over_runs))}\n")
    
            np.save(os.path.join(self.save_path, "reward_over_runs_with_children.npy"), np.array(reward_over_runs_with_children))
            with open(os.path.join(self.save_path, "reward_over_runs_with_children_num_runs.txt"), "w") as f:
                f.write(f"Number of runs: {len(reward_over_runs_with_children)}\n")
                f.write(f"Mean reward over runs: {np.mean(np.array(reward_over_runs_with_children))}\n") # should be less than the reward_over_runs
            
        # Save the best hyperparameters
        best_hparams_idx = np.argmax(fitness_of_population)
        best_hparams = pickle.load(open(os.path.join(self.save_path, f"hyperparams/hyperparams_{best_hparams_idx}.pkl"), "rb"))
        return best_hparams, best_hparams_idx
    
    def CMAES(self):
        pass

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
