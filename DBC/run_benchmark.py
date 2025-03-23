"""Run benchmarking tests on a given model with a given set of reference actions"""

import AnalysisEnv
from benchmarking import get_benchmark_scores, print_avg_scores
# from stable_baselines3.ppo import PPO 
import torch
import random
import pickle
import argparse
from dbc import PolicyNetwork


def run_benchmark(model_path, dataset_type, dataset_num, num_trajs, max_steps=None, ref_acts_path=None, num_refs=None):

    max_steps = max_steps if max_steps is not None else 12

    if dataset_type == "NETWORKING":
        dataset_path = f'Datasets/Networks/{dataset_num}.tsv'
    elif dataset_type == "FLIGHTS":
        dataset_path = f'Datasets/Flights/{dataset_num}.tsv'
    env = AnalysisEnv.AnalysisEnv(dataset_path)

    # load model from torch
    min_action, max_action = env.get_action_space()
    model = PolicyNetwork(165, 6, min_action, max_action)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    if ref_acts_path is not None:
        try:
            with open(ref_acts_path, 'rb') as f:
                ref_acts = pickle.load(f)
            if num_refs is not None:
                random.seed(0)
                ref_acts = random.sample(ref_acts, num_refs)
        except:
            print('Failed to load reference actions.')
            return

    scores = get_benchmark_scores(model, env, int(num_trajs), dataset_type, int(dataset_num), (ref_acts_path and ref_acts), max_steps)

    print_avg_scores(scores)
    return scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run ATENA benchmark to get scores for policy performance",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--model_path", required=True, help="Path to the model") # Models/Networks/policy.pt
    parser.add_argument("--dataset_num", required=True, help="Dataset number (1,2,3,4)") # 1
    parser.add_argument("--num_trajs", required=True, help="Number of trajectories over which scores are averaged") # 1
    parser.add_argument("--max_steps", required=False, help="Max steps to take in env") # 12
    parser.add_argument("--ref_acts_path", required=False, help="Path to reference trajectories") # atena/evaluation/references/cyber/dataset1.py
    parser.add_argument("--num_refs", required=False, help="Number of reference trajectories to use")

    kwargs = vars(parser.parse_args())
    
    run_benchmark(dataset_type="NETWORKING", **kwargs)