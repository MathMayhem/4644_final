# greedy_policy_trainer.py
# A standalone script to train a dedicated "Greedy Policy" model.
#
# HOW TO RUN:
#   - To train the model:
#     python greedy_policy_trainer.py --mode train
#
#   - To generate visualizations from the logs:
#     python greedy_policy_trainer.py --mode visualize

import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from tqdm import tqdm
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# --- Import from the original project files ---
from config import Config
from utils import (
    set_seed, generate_random_layout, generate_random_weights,
    prepare_graph_data_v4, apply_action
)
from api import KeyboardEnvironment
from model import PolicyValueNetV4
from torch_geometric.data import Batch

# ==============================================================================
# 1. CUSTOM CONFIGURATION AND MODEL FOR THIS SCRIPT
# ==============================================================================

class GreedyConfig:
    """Configuration specific to the greedy policy trainer."""
    # --- Project Structure ---
    PROJECT_DIR = "greedy_optimizer_project"
    CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints")
    LOGS_DIR = os.path.join(PROJECT_DIR, "logs")
    VISUALIZATIONS_DIR = os.path.join(PROJECT_DIR, "visualizations")

    # --- Training Hyperparameters ---
    TRAINING_STEPS = 15_000
    REPLAY_BUFFER_SIZE = 15_000
    BATCH_SIZE = 256
    LEARNING_RATE = 3e-4
    MAX_STEPS = 29

    ORACLE_SIMULATIONS = 64

    # Hack to prevent constant repetitions
    NO_OP_PENALTY = 0.25
    REPEAT_STATE_PENALTY = -1e9

    # --- Curriculum Schedule ---
    CURRICULUM_SCHEDULE = {
        0: 1, 6000: 3, 8000: 5, 10000: 10, 12000: 20, 13000: MAX_STEPS
    }

    # --- File Paths ---
    LOG_PATH = os.path.join(LOGS_DIR, "greedy_policy_train_log.json")
    FINAL_MODEL_PATH = os.path.join(PROJECT_DIR, "greedy_policy_model.pth")
    VISUALIZATION_PATH = os.path.join(VISUALIZATIONS_DIR, "greedy_training_curve.png")

class GreedyPolicyNet(PolicyValueNetV4):
    """Modified V4 network without the value head for efficiency."""
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

# ==============================================================================
# 2. STATEFUL ORACLE MCTS WITH ANTI-LOOPING LOGIC
# ==============================================================================

class StatefulOracleMCTS:
    """
    An enhanced MCTS that penalizes revisiting states to prevent learning loops.
    This class is self-contained within this script to avoid modifying project files.
    """
    def __init__(self, model: torch.nn.Module, env: KeyboardEnvironment):
        self.model = model
        self.env = env

    def run(self, root_layout_str: str, root_weights_dict: dict, root_weights_tensor: torch.Tensor, steps_left: int) -> np.ndarray:
        class MCTSNode:
            def __init__(self, parent, prior_p, layout_str):
                self.parent, self.children, self.visit_count = parent, {}, 0
                self.total_action_value = 0.0
                self.prior_p, self.layout_str = prior_p, layout_str
            def get_value(self):
                return self.total_action_value / self.visit_count if self.visit_count > 0 else 0.0
            def expand(self, action_priors: np.ndarray):
                for action, prior_p in enumerate(action_priors):
                    if action not in self.children and prior_p > 0:
                        child_layout = apply_action(self.layout_str, action)
                        self.children[action] = MCTSNode(parent=self, prior_p=prior_p, layout_str=child_layout)
            def select_child(self) -> tuple:
                best_score, best_action, best_child = -np.inf, -1, None
                for action, child in self.children.items():
                    ucb_score = child.get_value() + Config.CPUCT * child.prior_p * math.sqrt(self.visit_count) / (1 + child.visit_count)
                    if ucb_score > best_score:
                        best_score, best_action, best_child = ucb_score, action, child
                return best_action, best_child
            def backpropagate(self, value: float):
                self.visit_count += 1
                self.total_action_value += value
                if self.parent:
                    self.parent.backpropagate(value)
            def is_leaf(self) -> bool:
                return not self.children

        self.model.eval()
        device = next(self.model.parameters()).device
        root_node = MCTSNode(parent=None, prior_p=1.0, layout_str=root_layout_str)

        for _ in range(GreedyConfig.ORACLE_SIMULATIONS):
            node = root_node
            path_history = {root_node.layout_str}

            # --- Selection ---
            while not node.is_leaf():
                _, node = node.select_child()
                path_history.add(node.layout_str)

            # --- Evaluation ---
            if node.layout_str in path_history and node is not root_node:
                value = GreedyConfig.REPEAT_STATE_PENALTY
            else:
                score_components = self.env.get_score_components(node.layout_str, root_weights_dict)
                value = sum(score_components.values())

            # --- Expansion ---
            with torch.no_grad():
                graph_data = prepare_graph_data_v4(node.layout_str, root_weights_tensor, steps_left)
                graph_batch = Batch.from_data_list([graph_data]).to(device)
                policy_logits, _ = self.model(graph_batch)
                action_priors = F.softmax(policy_logits, dim=1).squeeze().cpu().numpy()

            if node.parent is None:
                noise = np.random.dirichlet([0.3] * Config.NUM_ACTIONS)
                action_priors = 0.75 * action_priors + 0.25 * noise

            node.expand(action_priors)

            # --- Backpropagation ---
            node.backpropagate(value)

        # --- Final Policy Calculation ---
        visit_counts = np.zeros(Config.NUM_ACTIONS, dtype=np.float32)
        for action, child in root_node.children.items():
            visit_counts[action] = child.visit_count

        if visit_counts[Config.NO_OP_ACTION_INDEX] > 0:
            visit_counts[Config.NO_OP_ACTION_INDEX] *= (1.0 - GreedyConfig.NO_OP_PENALTY)

        # Normalize to get final target policy
        total_visits = np.sum(visit_counts)
        if total_visits > 0:
            return visit_counts / total_visits
        else:
            # Fallback for the rare case of no visits
            return np.ones(Config.NUM_ACTIONS) / Config.NUM_ACTIONS

# ==============================================================================
# 3. TRAINING PHASE
# ==============================================================================
def train():
    """Trains the GreedyPolicyNet using the stateful imitation learning."""
    print("--- TRAINING GREEDY POLICY MODEL (WITH ANTI-LOOPING) ---")
    set_seed(Config.RANDOM_SEED)
    device = Config.DEVICE

    model = GreedyPolicyNet().to(device)
    print("Initialized GreedyPolicyNet model.")

    env = KeyboardEnvironment()
    oracle_mcts = StatefulOracleMCTS(model, env)

    optimizer = torch.optim.Adam(model.parameters(), lr=GreedyConfig.LEARNING_RATE)
    replay_buffer = deque(maxlen=GreedyConfig.REPLAY_BUFFER_SIZE)
    training_logs = []
    progress_bar = tqdm(range(GreedyConfig.TRAINING_STEPS), desc="Greedy Policy Training")

    current_max_steps = 1
    for step in progress_bar:
        # Update curriculum
        for threshold, steps_val in GreedyConfig.CURRICULUM_SCHEDULE.items():
            if step >= threshold:
                current_max_steps = steps_val

        # --- Generate data ---
        layout = generate_random_layout()
        weights_dict = generate_random_weights()
        weights_tensor = torch.tensor([weights_dict.get(k, 0.0) for k in Config.WEIGHT_KEYS], dtype=torch.float32).to(device)
        steps_left = random.randint(1, current_max_steps)

        target_policy = oracle_mcts.run(layout, weights_dict, weights_tensor, steps_left)
        replay_buffer.append({
            "layout": layout, "weights": weights_tensor.cpu(),
            "steps_left": steps_left, "target_policy": target_policy
        })

        if len(replay_buffer) < GreedyConfig.BATCH_SIZE:
            continue

        # --- Training step ---
        model.train()
        batch_items = random.sample(replay_buffer, GreedyConfig.BATCH_SIZE)

        graph_list = [prepare_graph_data_v4(item['layout'], item['weights'].to(device), item['steps_left']) for item in batch_items]
        graph_batch = Batch.from_data_list(graph_list).to(device)
        target_p = torch.tensor(np.array([item['target_policy'] for item in batch_items]), dtype=torch.float32).to(device)

        optimizer.zero_grad()
        pred_p, _ = model(graph_batch)
        loss = F.cross_entropy(pred_p, target_p)
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix({"p_loss": loss.item(), "curriculum": f"{current_max_steps} steps"})

        if step % 100 == 0:
            training_logs.append({'step': step, 'p_loss': loss.item(), 'curriculum_step': current_max_steps})

        if step % 2000 == 0 and step > 0:
            chkpt_path = os.path.join(GreedyConfig.CHECKPOINT_DIR, f"greedy_chkpt_{step}.pth")
            torch.save({'step': step, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, chkpt_path)
            with open(GreedyConfig.LOG_PATH, 'w') as f: json.dump(training_logs, f, indent=2)

    print(f"\nGreedy policy training complete. Final p_loss: {loss.item():.4f}")
    print(f"Saving final model to {GreedyConfig.FINAL_MODEL_PATH}...")
    torch.save(model.state_dict(), GreedyConfig.FINAL_MODEL_PATH)
    with open(GreedyConfig.LOG_PATH, 'w') as f: json.dump(training_logs, f, indent=2)

# ==============================================================================
# 4. VISUALIZATION AND MAIN ENTRYPOINT
# ==============================================================================
def visualize():
    """Generates a plot of the training performance from the log file."""
    print(f"--- GENERATING VISUALIZATION ---")
    sns.set_theme(style="whitegrid")
    try:
        df = pd.read_json(GreedyConfig.LOG_PATH)
    except FileNotFoundError:
        print(f"Log file not found at {GreedyConfig.LOG_PATH}. Run training first.")
        return
    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax1.set_xlabel('Training Step'); ax1.set_ylabel('Policy Loss (Cross-Entropy)', color='tab:blue')
    sns.lineplot(data=df, x='step', y='p_loss', ax=ax1, label='Policy Loss', color='tab:blue', alpha=0.8)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    random_baseline = -np.log(1.0 / Config.NUM_ACTIONS)
    ax1.axhline(y=random_baseline, color='gray', linestyle='--', label=f'Random Guessing Baseline (~{random_baseline:.2f})')
    ax1.legend(loc='upper left')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Max Steps in Curriculum', color='tab:red')
    sns.lineplot(data=df, x='step', y='curriculum_step', ax=ax2, label='Curriculum Steps', color='tab:red', linestyle='--', drawstyle='steps-post')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, GreedyConfig.MAX_STEPS + 2)
    fig.suptitle('Greedy Policy Model Training Performance (with Anti-Looping)', fontsize=16)
    fig.tight_layout()
    plt.savefig(GreedyConfig.VISUALIZATION_PATH)
    print(f"Generated training plot: {GreedyConfig.VISUALIZATION_PATH}")
    plt.show()

def main():
    """Main entry point to run training or visualization."""
    for dir_path in [GreedyConfig.PROJECT_DIR, GreedyConfig.CHECKPOINT_DIR, GreedyConfig.LOGS_DIR, GreedyConfig.VISUALIZATIONS_DIR]:
        if not os.path.exists(dir_path): os.makedirs(dir_path)
    parser = argparse.ArgumentParser(description="Train a greedy policy model for keyboard layout optimization.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'visualize'], help="train: Start training.\nvisualize: Generate plots from logs.")
    args = parser.parse_args()
    if args.mode == 'train':
        train()
        visualize()
    elif args.mode == 'visualize':
        visualize()

if __name__ == '__main__':
    main()
