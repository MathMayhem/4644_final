# utils.py

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import numpy as np
import random
import string

import os
import json
from tqdm import tqdm
from api import KeyboardEnvironment

from config import Config

def set_seed(seed: int):
    """
    Sets the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

ACTION_TO_SWAP = {
    idx: swap for idx, swap in enumerate(
        (i, j) for i in range(Config.NUM_KEYS) for j in range(i + 1, Config.NUM_KEYS)
    )
}

KEY_TO_HAND = {i: 'L' if i in [0,1,2,3,4, 10,11,12,13,14, 20,21,22,23,24] else 'R' for i in range(Config.NUM_KEYS)}
FINGER_ASSIGNMENTS = [
    [0, 10, 20], [1, 11, 21], [2, 12, 22], [3, 13, 23, 4, 14, 24], # Left
    [5, 15, 25, 6, 16, 26], [7, 17, 27], [8, 18, 28], [9, 19, 29]  # Right
]
KEY_TO_FINGER = {key_idx: finger_idx for finger_idx, keys in enumerate(FINGER_ASSIGNMENTS) for key_idx in keys}
KEY_TO_ROW = {i: (0 if i < 10 else (1 if i < 20 else 2)) for i in range(Config.NUM_KEYS)}

def _create_mirror_map() -> dict:
    key_mirror = {
        0:9, 1:8, 2:7, 3:6, 4:5, 10:19, 11:18, 12:17, 13:16, 14:15,
        20:29, 21:28, 22:27, 23:26, 24:25
    }
    key_mirror.update({v: k for k, v in key_mirror.items()})
    action_map = {}
    swap_to_action = {v: k for k, v in ACTION_TO_SWAP.items()}

    for action, (k1, k2) in ACTION_TO_SWAP.items():
        m_k1 = key_mirror.get(k1, k1)
        m_k2 = key_mirror.get(k2, k2)
        if m_k1 > m_k2: m_k1, m_k2 = m_k2, m_k1
        mirrored_action = swap_to_action.get((m_k1, m_k2), -1)
        action_map[action] = mirrored_action

    action_map[Config.NO_OP_ACTION_INDEX] = Config.NO_OP_ACTION_INDEX
    return action_map

ACTION_MIRROR_MAP = _create_mirror_map()

def generate_random_layout() -> str:
    layout_list = list(Config.ALL_CHARS)
    random.shuffle(layout_list)
    return "".join(layout_list)

def generate_random_weights() -> dict:
    weights = {key: random.uniform(-2.0, 2.0) for key in Config.WEIGHT_KEYS}
    if all(v == 0 for v in weights.values()):
        weights[random.choice(Config.WEIGHT_KEYS)] = 1.0
    return weights

def apply_action(layout_str: str, action_index: int) -> str:
    if action_index == Config.NO_OP_ACTION_INDEX:
        return layout_str

    key_indices_to_swap = ACTION_TO_SWAP[action_index]
    layout_list = list(layout_str)
    char1, char2 = layout_list[key_indices_to_swap[0]], layout_list[key_indices_to_swap[1]]
    layout_list[key_indices_to_swap[0]], layout_list[key_indices_to_swap[1]] = char2, char1
    return "".join(layout_list)

def normalize_score_vector(score_vector: np.ndarray) -> np.ndarray:
    min_val, max_val = Config.SCORE_COMP_MIN, Config.SCORE_COMP_MAX
    clamped = np.clip(score_vector, min_val, max_val)
    if max_val == min_val: return np.zeros_like(clamped)
    normalized_zero_one = (clamped - min_val) / (max_val - min_val)
    return normalized_zero_one * 2.0 - 1.0

def prepare_graph_data_v4(layout_str: str, weights_tensor: torch.Tensor, steps_left: int) -> Data:
    target_device = weights_tensor.device

    char_indices = [Config.CHAR_TO_INT[char] for char in layout_str]
    char_indices_tensor = torch.tensor(char_indices, device=target_device)
    one_hot_chars = F.one_hot(char_indices_tensor, num_classes=Config.NUM_KEYS).float()

    weights_expanded = weights_tensor.unsqueeze(0).expand(Config.NUM_KEYS, -1)

    hand_tensor = torch.tensor([0 if KEY_TO_HAND[i] == 'L' else 1 for i in range(Config.NUM_KEYS)], device=target_device)
    hand_feats = F.one_hot(hand_tensor, num_classes=Config.NUM_FEAT_HAND).float()
    finger_tensor = torch.tensor([KEY_TO_FINGER[i] for i in range(Config.NUM_KEYS)], device=target_device)
    finger_feats = F.one_hot(finger_tensor, num_classes=Config.NUM_FEAT_FINGER).float()
    row_tensor = torch.tensor([KEY_TO_ROW[i] for i in range(Config.NUM_KEYS)], device=target_device)
    row_feats = F.one_hot(row_tensor, num_classes=Config.NUM_FEAT_ROW).float()

    steps_left_normalized = float(steps_left) / float(Config.RL_MAX_STEPS)
    steps_left_feat = torch.full((Config.NUM_KEYS, 1), steps_left_normalized, device=target_device)

    node_features = torch.cat([
        one_hot_chars, weights_expanded, hand_feats, finger_feats, row_feats, steps_left_feat
    ], dim=1)

    edge_index = torch.tensor(
        [[i, j] for i in range(Config.NUM_KEYS) for j in range(Config.NUM_KEYS) if i != j],
        dtype=torch.long, device=target_device
    ).t().contiguous()

    return Data(x=node_features, edge_index=edge_index)

def mirror_policy(policy_vector: np.ndarray) -> np.ndarray:
    mirrored_p = np.zeros_like(policy_vector)
    for action_idx, mirrored_action_idx in ACTION_MIRROR_MAP.items():
        if mirrored_action_idx != -1:
            mirrored_p[mirrored_action_idx] = policy_vector[action_idx]
    return mirrored_p

def mirror_layout(layout_str: str) -> str:
    layout_list = list(layout_str)
    for i in range(5): layout_list[i], layout_list[9-i] = layout_list[9-i], layout_list[i]
    for i in range(5): layout_list[10+i], layout_list[19-i] = layout_list[19-i], layout_list[10+i]
    for i in range(5): layout_list[20+i], layout_list[29-i] = layout_list[29-i], layout_list[20+i]
    return "".join(layout_list)

def generate_data():
    print(f"--- GENERATING SUPERVISED DATA FOR VALUE HEAD ---")

    set_seed(Config.RANDOM_SEED)

    if not os.path.exists(Config.DATA_DIR):
        os.makedirs(Config.DATA_DIR)

    env = KeyboardEnvironment()
    dataset = []

    progress_bar = tqdm(total=Config.NUM_SUPERVISED_SAMPLES, desc="Generating Value Data")
    while len(dataset) < Config.NUM_SUPERVISED_SAMPLES:
        num_to_generate = min(Config.API_BATCH_SIZE, Config.NUM_SUPERVISED_SAMPLES - len(dataset))

        batch_to_process = []
        for _ in range(num_to_generate):
            layout = generate_random_layout()
            weights = generate_random_weights()
            batch_to_process.append((layout, weights))

        results_batch = env.get_score_components_batched(batch_to_process)

        for (layout, weights), result_components in zip(batch_to_process, results_batch):
            score_vector = [result_components.get(k, 0.0) for k in Config.WEIGHT_KEYS]
            weights_vector = [weights.get(k, 0.0) for k in Config.WEIGHT_KEYS]
            dataset.append({
                "layout": layout,
                "weights": weights_vector,
                "score_vector": score_vector
            })

        progress_bar.update(len(results_batch))

    progress_bar.close()
    print(f"\nGenerated {len(dataset)} samples. Saving to {Config.SUPERVISED_DATA_PATH}...")
    with open(Config.SUPERVISED_DATA_PATH, 'w') as f:
        json.dump(dataset, f)
    print("Data generation complete.")
