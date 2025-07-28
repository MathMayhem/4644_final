# mcts.py

import numpy as np
import math

from config import Config
from utils import apply_action, prepare_graph_data_v4
from api import KeyboardEnvironment

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

class OracleMCTS:
    """
    A sequential MCTS that uses the ground-truth API for value estimates.
    Used exclusively for Phase 1 (Policy Pre-training).
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

        for _ in range(Config.POLICY_ORACLE_SIMULATIONS):
            node = root_node
            while not node.is_leaf():
                _, node = node.select_child()

            score_components = self.env.get_score_components(node.layout_str, root_weights_dict)
            value = sum(score_components.values())

            with torch.no_grad():
                graph_data = prepare_graph_data_v4(node.layout_str, root_weights_tensor, steps_left)
                graph_batch = Batch.from_data_list([graph_data]).to(device)
                policy_logits, _ = self.model(graph_batch)
                action_priors = F.softmax(policy_logits, dim=1).squeeze().cpu().numpy()

            if node.parent is None:
                noise = np.random.dirichlet([0.3] * Config.NUM_ACTIONS)
                action_priors = 0.75 * action_priors + 0.25 * noise

            node.expand(action_priors)
            node.backpropagate(value)

        visit_counts = np.zeros(Config.NUM_ACTIONS, dtype=np.float32)
        for action, child in root_node.children.items():
            visit_counts[action] = child.visit_count

        return visit_counts / np.sum(visit_counts) if np.sum(visit_counts) > 0 else np.ones(Config.NUM_ACTIONS) / Config.NUM_ACTIONS

class MCTS:
    """
    The standard, batched MCTS for high-performance reinforcement learning.
    Used exclusively for Phase 3 (RL Fine-tuning).
    """
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def run(self, root_layout_str: str, root_weights_tensor: torch.Tensor, steps_left: int) -> np.ndarray:
        class MCTSNode:
            def __init__(self, parent, prior_p, layout_str):
                self.parent, self.children, self.visit_count = parent, {}, 0
                self.total_action_value = np.zeros(Config.NUM_WEIGHTS, dtype=np.float32)
                self.prior_p, self.layout_str = prior_p, layout_str
            def get_value(self) -> float:
                return np.mean(self.total_action_value / self.visit_count) if self.visit_count > 0 else 0.0
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
            def backpropagate(self, value_vector: np.ndarray):
                self.visit_count += 1
                self.total_action_value += value_vector
                if self.parent:
                    self.parent.backpropagate(value_vector)
            def is_leaf(self) -> bool:
                return not self.children

        self.model.eval()
        device = next(self.model.parameters()).device
        root_node = MCTSNode(parent=None, prior_p=1.0, layout_str=root_layout_str)

        leaf_nodes_to_evaluate = []
        for _ in range(Config.MCTS_SIMULATIONS):
            node = root_node
            while not node.is_leaf():
                _, node = node.select_child()
            leaf_nodes_to_evaluate.append(node)

        if leaf_nodes_to_evaluate:
            unique_leaves = {node.layout_str: node for node in leaf_nodes_to_evaluate}
            layout_to_idx = {layout: i for i, layout in enumerate(unique_leaves.keys())}

            graph_data_list = [prepare_graph_data_v4(l, root_weights_tensor, steps_left) for l in unique_leaves.keys()]
            graph_batch = Batch.from_data_list(graph_data_list).to(device)

            with torch.no_grad():
                policy_logits, value_vectors = self.model(graph_batch)
                action_priors_batch = F.softmax(policy_logits, dim=1).cpu().numpy()
                value_vectors = value_vectors.cpu().numpy()

            for node in leaf_nodes_to_evaluate:
                batch_idx = layout_to_idx[node.layout_str]
                action_priors = action_priors_batch[batch_idx]
                value_vector = value_vectors[batch_idx]

                if node.parent is None:
                    noise = np.random.dirichlet([0.3] * Config.NUM_ACTIONS)
                    action_priors = 0.75 * action_priors + 0.25 * noise

                node.expand(action_priors)
                node.backpropagate(value_vector)

        visit_counts = np.zeros(Config.NUM_ACTIONS, dtype=np.float32)
        for action, child in root_node.children.items():
            visit_counts[action] = child.visit_count

        return visit_counts / np.sum(visit_counts) if np.sum(visit_counts) > 0 else np.ones(Config.NUM_ACTIONS) / Config.NUM_ACTIONS
