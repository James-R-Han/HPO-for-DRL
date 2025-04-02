import os
import pickle


path = "/home/jha/HPO-for-DRL/results/CartPole-v1/DQN_CartPole-v1_GS/hyperparams/hyperparams_36.pkl"

a = pickle.load(open(path, "rb"))

with open(os.path.join("/home/jha/HPO-for-DRL", "best_hparams_idx.txt"), "w") as f:
    f.write(f"Best hyperparameters: {a}\n")