import sys
from pathlib import Path

configs_path = Path(__file__).parent
sys.path.insert(0, str(configs_path.resolve()))
from config_base import ConfigBase

class ConfigTraining:
    VALID_ALGOS = ["DQN", "PPO"]
    VALID_ENVS = ["CartPole-v1", "MountainCar-v0", "LunarLander-v3"]
    VALID_HPO_ALGOS = ["RS", "GS", "DE", "CMAES"]
    TIMESTEPS = {"CartPole-v1": 100000 , "MountainCar-v0": 500000, "LunarLander-v3": 1000000}
    #100000
    def __init__(self):
        self.device = 'cuda'
        
        self.DRL_algo = "PPO"  # or "DQN"
        self.HPO_algo = "CMAES"
        self.env_id = "MountainCar-v0"
        self.training_seeds = [0,1]
        self.testing_seeds = [2,3,4,5,6]
        
        self.training_timesteps = self.TIMESTEPS[self.env_id]
        self.num_eval_episodes = 100
        
        self.identifier = f"{self.DRL_algo}_{self.env_id}_{self.HPO_algo}"
        self.save_path = f"/home/jha/HPO-for-DRL/results/{self.env_id}/{self.identifier}"
        
        assert self.DRL_algo in self.VALID_ALGOS, f"Invalid DRL algorithm: {self.DRL_algo}. Valid options are: {self.VALID_ALGOS}"
        assert self.HPO_algo in self.VALID_HPO_ALGOS, f"Invalid HPO algorithm: {self.HPO_algo}. Valid options are: {self.VALID_HPO_ALGOS}"
        assert self.env_id in self.VALID_ENVS, f"Invalid environment ID: {self.env_id}. Valid options are: {self.VALID_ENVS}"