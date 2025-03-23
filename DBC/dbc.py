import torch
import torch.nn as nn
import AnalysisEnv
from benchmarking import get_benchmark_scores, print_avg_scores

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, action_low, action_high):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.action_low = torch.tensor(action_low)
        self.action_high = torch.tensor(action_high)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # Output range [-1, 1]
        # Rescale to the desired action range
        action = self.action_low + (x + 1.0) * 0.5 * (self.action_high - self.action_low)
        return action

def main(model_path, dataset_path, dataset_num, num_trajs, max_steps):
    # Load the model
    state_vector_dim = 165
    action_vector_dim = 6
    min_action, max_action = [0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]

    policy_net = PolicyNetwork(state_vector_dim, action_vector_dim, min_action, max_action)
    policy_net.load_state_dict(torch.load(model_path))
    policy_net.eval()  # Set the model to evaluation mode

    # Initialize the environment
    Env = AnalysisEnv.AnalysisEnv(dataset_path)

    # Run the benchmark
    dataset_type = "NETWORKING"
    ref_acts_path = None
    ref_acts = None

    scores = get_benchmark_scores(policy_net, Env, num_trajs, dataset_type, dataset_num, ref_acts, max_steps)
    print_avg_scores(scores)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run benchmark")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model file")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset file")
    parser.add_argument("--dataset_num", type=int, required=True, help="Dataset number")
    parser.add_argument("--num_trajs", type=int, required=True, help="Number of trajectories")
    parser.add_argument("--max_steps", type=int, required=True, help="Maximum number of steps")

    args = parser.parse_args()

    main(args.model_path, args.dataset_path, args.dataset_num, args.num_trajs, args.max_steps)