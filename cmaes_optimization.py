import numpy as np
from cmaes import CMA
from train_drl import HPO_for_DRL



def clamp_and_round_params(params_dict, bounds):
    """
    Clamp parameter values to the specified min/max from `bounds`.
    Round them if they are originally meant to be integer.
    """
    clamped = {}
    for k, (low, high) in bounds.items():
        val = params_dict[k]
        if k in ["n_steps", "batch_size"]:
            clamped_val = int(np.clip(round(val), low, high))
        else:
            clamped_val = float(np.clip(val, low, high))
        clamped[k] = clamped_val
    return clamped

def cmaes_optimize(agent_name="PPO", bounds=PPO_BOUNDS, total_timesteps=10000, max_iter=10):
    """
    Basic CMA-ES optimization for PPO hyperparameters. 
    We treat the reward as something to *maximize*, so the CMA-ES objective will minimize negative reward.
    """
    # Prepare an initial solution in the middle of the ranges
    initial_solution = []
    for (low, high) in bounds.values():
        initial_solution.append((low + high) / 2.0)
    initial_solution = np.array(initial_solution)

    # Prepare sigma (spread) as 30% of each dimension's range, just as a guess
    sigma_init = 0.3 * np.array([high - low for (low, high) in bounds.values()])

    # Initialize CMA
    # (You can tune population_size, bounds, etc. Check cmaes docs for more advanced usage)
    cma = CMA(mean=initial_solution, sigma=sigma_init.mean(), population_size=8)

    # For referencing each hyperparam dimension
    param_keys = list(bounds.keys())

    best_solution = None
    best_score = -np.inf

    for generation in range(max_iter):
        solutions = []
        for _ in range(cma.population_size):
            x = cma.ask()
            # Convert vector x to dictionary of named hyperparams
            trial_params_dict = {k: v for k, v in zip(param_keys, x)}
            # Clamp or round
            clamped_params = clamp_and_round_params(trial_params_dict, bounds)

            # Evaluate reward
            reward = HPO_for_DRL(
                agent_name,
                clamped_params,
                total_timesteps=total_timesteps
            )
            # Minimizing negative reward => negative sign
            value = -reward
            solutions.append((x, value))

            if reward > best_score:
                best_score = reward
                best_solution = clamped_params

        # Update CMA with the solutions
        cma.tell(solutions)

        print(f"Generation {generation+1}/{max_iter} => Best reward so far: {best_score}")

    return best_solution, best_score

if __name__ == "__main__":
    best_params, best_perf = cmaes_optimize(
        agent_name="PPO",
        bounds=PPO_BOUNDS,
        total_timesteps=5000,
        max_iter=3,
    )
    print("\n=== CMA-ES Results ===")
    print(f"Best hyperparameters: {best_params}")
    print(f"Best score: {best_perf}")
