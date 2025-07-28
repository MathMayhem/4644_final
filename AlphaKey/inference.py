# inference.py
#
# HOW TO RUN:
#
# Example 1: Run the RL model on QWERTY with default weights.
#   python inference.py --model rl --layout qwerty
#
# Example 2: Run the Greedy model on a random layout with custom weights.
#   python inference.py --model greedy --layout random --max_swaps 20 --sfb 1.5 --alt -0.5

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange

# --- Import from the original project files ---
from config import Config
from utils import (
    set_seed, generate_random_layout, prepare_graph_data_v4,
    apply_action, ACTION_TO_SWAP
)
from api import KeyboardEnvironment
from model import PolicyValueNetV4
from torch_geometric.data import Batch

# To avoid importing from an executable script, we'll redefine the GreedyPolicyNet
# class here. It's identical to the one in `greedy_policy_trainer.py`.
class GreedyPolicyNet(PolicyValueNetV4):
    """Modified network without the value head."""
    def __init__(self):
        super(GreedyPolicyNet, self).__init__()
        del self.value_head

    def forward(self, batch):
        x, edge_index, batch_map = batch.x, batch.edge_index, batch.batch
        x = F.relu(self.initial_embedding(x))
        positions = torch.arange(Config.NUM_KEYS, device=x.device).repeat(batch.num_graphs)
        pos_enc = self.positional_embeddings(positions)
        x = x + pos_enc
        for i, layer in enumerate(self.gat_layers):
            x = x + F.relu(layer(x, edge_index))
            x = self.norm_layers[i](x)
        graph_embedding = self.attention_pooling(x, batch_map)
        policy_logits = self.policy_head(graph_embedding)
        return policy_logits, None


def load_model(model_type, device):
    """Loads the specified pre-trained model."""
    if model_type == 'greedy':
        model_path = 'greedy_optimizer_project/greedy_policy_model.pth'
        model = GreedyPolicyNet().to(device)
    elif model_type == 'rl':
        model_path = 'checkpoints/final_rl_model.pth'
        model = PolicyValueNetV4().to(device)
    else:
        raise ValueError("Invalid model type specified. Use 'greedy' or 'rl'.")

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Successfully loaded model from: {model_path}")
        return model
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {model_path}")
        print("Please ensure you have trained the required model first.")
        exit(1)


def get_total_score(score_components, weights_dict):
    """Calculates the final weighted score from the score components."""
    total = 0
    for key, value in score_components.items():
        total += value * weights_dict.get(key, 0.0)
    return total


def plot_inference_run(score_history, model_type, layout_type, weights_dict):
    """Generates and saves a plot of the score over swaps."""
    output_dir = "inference_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    weights_str = "_".join([f"{k}{v}" for k, v in weights_dict.items()]).replace('.','p')
    filename = f"{model_type}_on_{layout_type}_{weights_str}.png"
    filepath = os.path.join(output_dir, filename)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7))

    ax = sns.lineplot(x=range(len(score_history)), y=score_history, marker='o')
    ax.set_title(f'"{model_type.upper()}" Model Optimization on "{layout_type.upper()}" Layout', fontsize=16)
    ax.set_xlabel('Swap Number', fontsize=12)
    ax.set_ylabel('Total Weighted Score', fontsize=12)
    ax.set_xticks(range(len(score_history)))

    plt.tight_layout()
    plt.savefig(filepath)
    print(f"\nSaved visualization to: {filepath}")
    plt.show()


def run_inference(model, model_type, start_layout, weights_dict, max_swaps, device):
    """Runs the inference loop with stateful anti-looping logic."""
    print("\n" + "="*60)
    print(f"Running Stateful Inference with '{model_type.upper()}' model.")
    print("Anti-looping engaged: will not return to a previously seen layout.")
    print(f"Starting Layout: {start_layout}")
    print(f"Preference Weights: {weights_dict}")
    print("="*60 + "\n")

    env = KeyboardEnvironment()
    current_layout = start_layout
    score_history = []
    seen_layouts = {start_layout}

    initial_components = env.get_score_components(current_layout, weights_dict)
    initial_score = get_total_score(initial_components, weights_dict)
    score_history.append(initial_score)

    print(f"Step 0: Start")
    print(f"  Layout: {current_layout}")
    print(f"  Score:  {initial_score:.4f}\n")

    weights_tensor = torch.tensor([weights_dict.get(k, 0.0) for k in Config.WEIGHT_KEYS], dtype=torch.float32).to(device)

    for i in trange(1, max_swaps + 1, desc="Optimizing Layout"):
        steps_left = max_swaps - (i - 1)

        graph_data = prepare_graph_data_v4(current_layout, weights_tensor, steps_left)
        graph_batch = Batch.from_data_list([graph_data]).to(device)

        with torch.no_grad():
            policy_logits, _ = model(graph_batch)

            action_probs = F.softmax(policy_logits, dim=1).squeeze()
            sorted_actions = torch.argsort(action_probs, descending=True)

            chosen_action = -1
            for action_candidate in sorted_actions:
                action_candidate = action_candidate.item()

                if action_candidate == Config.NO_OP_ACTION_INDEX:
                    chosen_action = action_candidate
                    break

                potential_layout = apply_action(current_layout, action_candidate)

                if potential_layout not in seen_layouts:
                    chosen_action = action_candidate
                    break

            if chosen_action == -1:
                print(f"Step {i}: All possible moves lead to previous states. Halting.")
                chosen_action = Config.NO_OP_ACTION_INDEX

        if chosen_action == Config.NO_OP_ACTION_INDEX:
            print(f"\nStep {i}: Model chose to stop (NO_OP). Halting optimization.")
            break

        key_indices = ACTION_TO_SWAP[chosen_action]
        char1 = current_layout[key_indices[0]]
        char2 = current_layout[key_indices[1]]

        new_layout = apply_action(current_layout, chosen_action)

        new_components = env.get_score_components(new_layout, weights_dict)
        new_score = get_total_score(new_components, weights_dict)
        score_history.append(new_score)

        seen_layouts.add(new_layout)

        print(f"\nStep {i}: Swap '{char1}' <-> '{char2}'")
        print(f"  Layout: {new_layout}")
        print(f"  Score:  {new_score:.4f} ({new_score - score_history[-2]:+.4f})")

        current_layout = new_layout

    print("\n" + "="*50)
    print("Inference Complete.")
    print(f"Final Layout: {current_layout}")
    print(f"Initial Score: {score_history[0]:.4f}")
    print(f"Final Score:   {score_history[-1]:.4f}")
    print(f"Total Improvement: {score_history[-1] - score_history[0]:.4f}")
    print("="*50)

    return score_history


def main():
    parser = argparse.ArgumentParser(
        description="Run and visualize inference for keyboard layout optimization models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model', type=str, required=True, choices=['greedy', 'rl'], help="Which trained model to use.")
    parser.add_argument('--layout', type=str, required=True, choices=['qwerty', 'random'], help="Which starting layout to use.")
    parser.add_argument('--max_swaps', type=int, default=30, help="Maximum number of swaps the model can perform.")
    for key in Config.WEIGHT_KEYS:
        parser.add_argument(f'--{key}', type=float, default=1.0, help=f"Weight for the '{key}' score component.")
    args = parser.parse_args()

    set_seed(Config.RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = load_model(args.model, device)
    start_layout = "qwertyuiopasdfghjkl;zxcvbnm,.'" if args.layout == 'qwerty' else generate_random_layout()
    weights_dict = {key: getattr(args, key) for key in Config.WEIGHT_KEYS}
    score_history = run_inference(model, args.model, start_layout, weights_dict, args.max_swaps, device)

    if len(score_history) > 1:
        plot_inference_run(score_history, args.model, args.layout, weights_dict)
    else:
        print("\nNo swaps were made. Skipping plot generation.")

if __name__ == '__main__':
    main()
