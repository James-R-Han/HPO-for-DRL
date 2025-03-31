import numpy as np


def create_population(pop_size, bounds):
    """
    Create an initial population of candidate solutions within the given bounds.
    Each individual is a dict of hyperparams.
    """
    param_order = []
    for hparam, (low, high) in bounds.items():
        param_order.append(hparam)
    
    population = []
    for i in range(pop_size):
        hparams = {}
        vector_of_params = []
        for hparam_name in param_order:
        # for hparam, (low, high) in bounds.items():
            hparam, (low, high) = hparam_name, bounds[hparam_name]
            if hparam in ["n_steps", "batch_size"]:
                hparams[hparam] = int(np.random.uniform(low, high))
            else:
                hparams[hparam] = np.random.uniform(low, high)
            vector_of_params.append(hparams[hparam])
        
        # we need vector_of_params to be a numpy array for the DE algorithm
        vector_of_params = np.array(vector_of_params)
        population.append((hparams, vector_of_params))
        
    return param_order, population

def mutate_and_crossover(hparam_order, target, donor, bounds, crossover_rate):
    """
    Perform crossover/mutation of a single target individual with a donor vector.
    """
    new_individual = {}
    new_individual_vector = []
    at_least_one_crossover = False
    for k, hparam_name in enumerate(hparam_order):
        hparam, (low, high) = hparam_name, bounds[hparam_name]
        if np.random.rand() < crossover_rate:
            # Crossover from donor
            new_val = donor[k]
            at_least_one_crossover = True
        else:
            # Retain from target
            new_val = target[k]
            
        # If discrete, clamp and round
        if hparam in ["n_steps", "batch_size"]:
            new_val = int(np.clip(round(new_val), low, high))
        else:
            new_val = float(np.clip(new_val, low, high))

        new_individual[hparam] = new_val
        new_individual_vector.append(new_val)
        
    # If no crossover happened, randomly replace one index at random
    if not at_least_one_crossover:
        rand_hparam = np.random.randint(0, len(hparam_order))
        hparam_name = hparam_order[rand_hparam]
        low, high = bounds[hparam_name]
        if hparam_name in ["n_steps", "batch_size"]:
            new_individual[hparam_name] = int(np.clip(round(donor[rand_hparam]), low, high))
        else:
            new_individual[hparam_name] = float(np.clip(donor[rand_hparam], low, high))
        new_individual_vector[rand_hparam] = new_individual[hparam_name]
        
    new_individual_vector = np.array(new_individual_vector)
    return new_individual, new_individual_vector
