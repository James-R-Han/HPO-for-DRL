import sys
from pathlib import Path

configs_path = Path(__file__).parent
sys.path.insert(0, str(configs_path.resolve()))
from config_base import ConfigBase


PPO_BOUNDS = {
    "learning_rate": (-6, -1), # This is in log scale
    "n_steps": (64, 4096),      # though n_steps should be integer
    "batch_size": (8, 2048),   # also integer
    "gamma": (0.75, 0.9999),
    "gae_lambda": (0.5, 0.99),
}

DQN_BOUNDS = {
    "learning_rate": (-6, -1), # This is in log scale
    "batch_size": (8, 2048),
    "exploration_fraction": (0.01, 0.75),
    "gamma": (0.75, 0.999),
}

class ConfigHPO:
    def __init__(self):
        super().__init__()
        
        self.HPO_budget = 48
        self.bounds = {}
        self.bounds["PPO"] = PPO_BOUNDS
        self.bounds["DQN"] = DQN_BOUNDS
        ###########################################
        # Random Search
        RS_config = {}
        RS_config["num_runs"] = self.HPO_budget
        self.RS_config = RS_config
        
        assert RS_config["num_runs"] <= self.HPO_budget, f"RS budget {RS_config['num_runs']} exceeds HPO budget {self.HPO_budget}"
        ###########################################
        # Grid Search
        GS_config = {}
        
        PPO_GS_config = {}
        PPO_GS_config["learning_rate"] = 2
        PPO_GS_config["n_steps"] = 3
        PPO_GS_config["batch_size"] = 2
        PPO_GS_config["gamma"] = 2
        PPO_GS_config["gae_lambda"] = 2
        
        GS_config["PPO"] = PPO_GS_config
        
        DQN_GS_config = {}
        DQN_GS_config["learning_rate"] = 2
        DQN_GS_config["batch_size"] = 2
        DQN_GS_config["exploration_fraction"] = 3
        DQN_GS_config["gamma"] = 4
        GS_config["DQN"] = DQN_GS_config
        self.GS_config = GS_config
        
        GS_budget_PPO = PPO_GS_config["learning_rate"] * PPO_GS_config["n_steps"] * PPO_GS_config["batch_size"] * PPO_GS_config["gamma"] * PPO_GS_config["gae_lambda"]
        GS_budget_DQN = DQN_GS_config["learning_rate"] * DQN_GS_config["batch_size"] * DQN_GS_config["exploration_fraction"] * DQN_GS_config["gamma"]
        assert GS_budget_PPO <= self.HPO_budget, f"PPO GS budget {GS_budget_PPO} exceeds HPO budget {self.HPO_budget}"
        assert GS_budget_DQN <= self.HPO_budget, f"DQN GS budget {GS_budget_DQN} exceeds HPO budget {self.HPO_budget}"
        ###########################################
        # Differential Evolution (DE)
        DE_config = {}
        
        # note F will be randomly sampled following Appendix A of https://arxiv.org/pdf/2105.09821
        DE_config["pop_size"] = 6
        DE_config["max_gens"] = 7
        DE_config["crossover_probability"] = 0.75
        
        DE_budget = DE_config["pop_size"] * (DE_config["max_gens"] + 1)
        self.DE_config = DE_config
        
        assert DE_budget <= self.HPO_budget, f"DE budget {DE_budget} exceeds HPO budget {self.HPO_budget}"
        ###########################################
        # Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
        CMAES_config = {}
        
        CMAES_config["pop_size"] = 6
        CMAES_config["max_gens"] = 7
        