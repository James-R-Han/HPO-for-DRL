import numpy as np
from train_drl import HPO_for_DRL


def create_population(pop_size, bounds):
    """
    Create an initial population of candidate solutions within the given bounds.
    Each individual is a dict of hyperparams.
    """
    population = []
    for _ in range(pop_size):
        individual = {}
        for k, (low, high) in bounds.items():
            # Discrete or continuous?
            if k in ["batch_size", "buffer_size", "target_update_interval"]:
                # Round for discrete hyperparams
                val = np.random.randint(low, high)
            else:
                # float
                val = np.random.uniform(low, high)
            individual[k] = val
        population.append(individual)
    return population

def mutate_and_crossover(target, donor, bounds, crossover_rate=0.7):
    """
    Perform crossover/mutation of a single target individual with a donor vector.
    """
    new_individual = {}
    for k, (low, high) in bounds.items():
        if np.random.rand() < crossover_rate:
            # Crossover from donor
            new_val = donor[k]
        else:
            # Retain from target
            new_val = target[k]

        # If discrete, clamp and round
        if k in ["batch_size", "buffer_size", "target_update_interval"]:
            new_val = int(np.clip(round(new_val), low, high))
        else:
            new_val = float(np.clip(new_val, low, high))

        new_individual[k] = new_val
    return new_individual

def differential_evolution(
    agent_name="DQN",
    bounds=DQN_BOUNDS,
    pop_size=10,
    max_gens=5,
    F=0.8,  # scale factor
    crossover_rate=0.7,
    total_timesteps=10000,
):
    """
    Basic Differential Evolution for optimizing hyperparameters.
    """
    # Create initial population
    population = create_population(pop_size, bounds)

    # Evaluate fitness of initial population
    fitness = []
    for ind in population:
        reward = HPO_for_DRL(agent_name, ind, total_timesteps=total_timesteps)
        fitness.append(reward)
    fitness = np.array(fitness)

    best_idx = np.argmax(fitness)
    best_sol = population[best_idx]
    best_score = fitness[best_idx]

    print(f"Initial best solution => Score: {best_score}  Hyperparams: {best_sol}")

    # Evolution loop
    for gen in range(max_gens):
        for i in range(pop_size):
            # Choose 3 random distinct individuals (excluding i)
            idxs = np.random.choice([idx for idx in range(pop_size) if idx != i], 3, replace=False)
            a, b, c = population[idxs[0]], population[idxs[1]], population[idxs[2]]

            # Create donor by a + F*(b - c) for each param
            donor = {}
            for k in bounds.keys():
                # If discrete, we do the float math but will clamp/round later
                donor[k] = a[k] + F * (b[k] - c[k])

            # Crossover
            trial = mutate_and_crossover(population[i], donor, bounds, crossover_rate=crossover_rate)

            # Evaluate trial individual
            trial_reward = HPO_for_DRL(agent_name, trial, total_timesteps=total_timesteps)

            # Selection
            if trial_reward > fitness[i]:
                population[i] = trial
                fitness[i] = trial_reward

                # Update best solution
                if trial_reward > best_score:
                    best_sol = trial
                    best_score = trial_reward

        print(f"Generation {gen+1}/{max_gens} => Best Score so far: {best_score}")

    return best_sol, best_score

if __name__ == "__main__":
    best_params, best_perf = differential_evolution(
        agent_name="DQN",
        bounds=DQN_BOUNDS,
        pop_size=5,
        max_gens=2,
        total_timesteps=5000,
    )
    print("\n=== Differential Evolution Results ===")
    print(f"Best hyperparameters: {best_params}")
    print(f"Best score: {best_perf}")
