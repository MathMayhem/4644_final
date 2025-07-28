import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from tqdm import tqdm
import textwrap
import argparse
import math
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from torch.amp import GradScaler, autocast

from config import Config
from utils import (
    set_seed, generate_random_layout, generate_random_weights,
    prepare_graph_data_v4, normalize_score_vector, mirror_layout,
    mirror_policy, apply_action, generate_data
)
from api import KeyboardEnvironment
from model import PolicyValueNetV4
from mcts import MCTS, OracleMCTS
from torch_geometric.data import Batch

# ==============================================================================
# PHASE 1: POLICY HEAD PRE-TRAINING
# ==============================================================================
def pretrain_policy():
    """
    Phase 1: Pre-trains the Policy Head from a random GNN.
    """
    print(f"--- PHASE 1: POLICY HEAD DISTILLATION ---")
    set_seed(Config.RANDOM_SEED)
    device = Config.DEVICE

    model = PolicyValueNetV4().to(device)
    print("Initialized model with random weights.")

    env = KeyboardEnvironment()
    oracle_mcts = OracleMCTS(model, env)

    optimizer = torch.optim.Adam(model.policy_head.parameters(), lr=Config.POLICY_PRETRAIN_LR)
    scaler = GradScaler(enabled=(device == "cuda"))
    replay_buffer = deque(maxlen=Config.POLICY_PRETRAIN_BUFFER_SIZE)
    training_logs = []

    progress_bar = tqdm(range(Config.POLICY_PRETRAIN_STEPS), desc=f"Policy Pre-Train")

    for step in progress_bar:
        layout = generate_random_layout()
        weights_dict = generate_random_weights()
        weights_tensor = torch.tensor([weights_dict.get(k, 0.0) for k in Config.WEIGHT_KEYS], dtype=torch.float32).to(device)

        target_policy = oracle_mcts.run(layout, weights_dict, weights_tensor, steps_left=1)
        replay_buffer.append({"layout": layout, "weights": weights_tensor.cpu(), "target_policy": target_policy})

        if len(replay_buffer) < Config.POLICY_PRETRAIN_BATCH_SIZE:
            continue

        model.train()
        for name, param in model.named_parameters():
            if 'policy_head' not in name:
                param.requires_grad = False

        batch_items = random.sample(replay_buffer, Config.POLICY_PRETRAIN_BATCH_SIZE)

        graph_list = [prepare_graph_data_v4(item['layout'], item['weights'].to(device), steps_left=1) for item in batch_items]
        graph_batch = Batch.from_data_list(graph_list).to(device)
        target_p = torch.tensor(np.array([item['target_policy'] for item in batch_items]), dtype=torch.float32).to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
            pred_p, _ = model(graph_batch)
            loss = F.cross_entropy(pred_p, target_p)
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()

        progress_bar.set_postfix({"p_loss": loss.item()})
        if step % 50 == 0:
            training_logs.append({'phase': 'policy_pretrain', 'step': step, 'p_loss': loss.item()})

    print(f"\nPolicy-first pre-training replica complete. Final p_loss: {loss.item():.4f}")
    if loss.item() >= 6.0:
        print("WARNING: Policy pre-training failed to improve.")

    print(f"Saving policy-trained model to {Config.PRETRAINED_POLICY_MODEL_PATH}...")
    torch.save(model.state_dict(), Config.PRETRAINED_POLICY_MODEL_PATH)
    with open(Config.POLICY_PRETRAIN_LOG_PATH, 'w') as f:
        json.dump(training_logs, f, indent=2)

    return model

# ==============================================================================
# PHASE 2: VALUE HEAD & GNN PRE-TRAINING
# ==============================================================================
def pretrain_value(model):
    """
    Phase 2: Pre-trains the Value Head and GNN body of a model that already
    has a pre-trained policy head.
    """
    print(f"--- PHASE 2: SUPERVISED PRE-TRAINING OF VALUE HEAD & GNN ---")
    set_seed(Config.RANDOM_SEED)

    try:
        with open(Config.SUPERVISED_DATA_PATH, 'r') as f: dataset = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Supervised data not found. Run '--mode generate_value_data' first."); return

    random.shuffle(dataset)
    split_idx = int(len(dataset) * 0.9)
    train_dataset, val_dataset = dataset[:split_idx], dataset[split_idx:]
    print(f"Data split: {len(train_dataset)} train, {len(val_dataset)} validation.")

    device = Config.DEVICE
    model.to(device)

    for param in model.parameters():
        param.requires_grad = True

    for name, param in model.named_parameters():
        if 'policy_head' in name:
            param.requires_grad = False

    policy_params_ids = {id(p) for p in model.policy_head.parameters()}
    trainable_params = [p for p in model.parameters() if id(p) not in policy_params_ids]
    optimizer = torch.optim.Adam(trainable_params, lr=Config.VALUE_PRETRAIN_LR)
    scaler = GradScaler(enabled=(device == "cuda"))
    training_logs = []

    best_val_loss = float('inf'); epochs_no_improve = 0; best_model_state = model.state_dict()
    print(f"Starting pre-training for up to {Config.VALUE_PRETRAIN_EPOCHS} epochs with patience={Config.VALUE_PRETRAIN_PATIENCE}.")

    for epoch in range(Config.VALUE_PRETRAIN_EPOCHS):
        model.train()

        train_pbar = tqdm(range(0, len(train_dataset), Config.VALUE_PRETRAIN_BATCH_SIZE), desc=f"Epoch {epoch+1} [Value Train]")
        for i in train_pbar:
            batch_items = train_dataset[i:i+Config.VALUE_PRETRAIN_BATCH_SIZE]
            if not batch_items: continue
            graph_list, targets = [], []
            for item in batch_items:
                weights = torch.tensor(item['weights'], dtype=torch.float32).to(device)
                graph_list.append(prepare_graph_data_v4(item['layout'], weights, steps_left=0))
                targets.append(normalize_score_vector(np.array(item['score_vector'])))
            graph_batch = Batch.from_data_list(graph_list).to(device)
            target_v = torch.tensor(np.array(targets), dtype=torch.float32).to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
                _, pred_v = model(graph_batch)
                loss = F.mse_loss(pred_v, target_v)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            train_pbar.set_postfix({"train_v_loss": loss.item()})

        # Validation loop
        model.eval()
        val_epoch_loss = 0
        with torch.no_grad():
            for i in range(0, len(val_dataset), Config.VALUE_PRETRAIN_BATCH_SIZE):
                batch_items = val_dataset[i:i+Config.VALUE_PRETRAIN_BATCH_SIZE]
                if not batch_items: continue
                graph_list, targets = [], []
                for item in batch_items:
                    weights = torch.tensor(item['weights'], dtype=torch.float32).to(device)
                    graph_list.append(prepare_graph_data_v4(item['layout'], weights, steps_left=0))
                    targets.append(normalize_score_vector(np.array(item['score_vector'])))
                graph_batch = Batch.from_data_list(graph_list).to(device)
                target_v = torch.tensor(np.array(targets), dtype=torch.float32).to(device)
                with autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
                    _, pred_v = model(graph_batch)
                    val_epoch_loss += F.mse_loss(pred_v, target_v).item()
        avg_val_loss = val_epoch_loss / (len(val_dataset) / Config.VALUE_PRETRAIN_BATCH_SIZE)

        print(f"Epoch {epoch+1} | Avg Val Loss: {avg_val_loss:.6f}")
        training_logs.append({'phase': 'value_pretrain', 'epoch': epoch+1, 'val_loss': avg_val_loss})
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss; best_model_state = model.state_dict(); epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= Config.VALUE_PRETRAIN_PATIENCE:
            print("Early stopping triggered."); break

    print(f"Saving best model (Val Loss: {best_val_loss:.6f}) to {Config.PRETRAINED_FULL_MODEL_PATH}...")
    torch.save(best_model_state, Config.PRETRAINED_FULL_MODEL_PATH)
    with open(Config.VALUE_PRETRAIN_LOG_PATH, 'w') as f:
        json.dump(training_logs, f, indent=2)

    model.load_state_dict(best_model_state)
    return model

# ==============================================================================
# PHASE 3: FULL REINFORCEMENT LEARNING
# ==============================================================================
def train_rl(model):
    print(f"--- PHASE 3: FULL REINFORCEMENT LEARNING ---")
    set_seed(Config.RANDOM_SEED)

    device = Config.DEVICE
    model.to(device)

    policy_params = list(model.policy_head.parameters())
    policy_param_ids = {id(p) for p in policy_params}
    base_and_value_params = [p for p in model.parameters() if id(p) not in policy_param_ids]

    optimizer = torch.optim.Adam([
        {'params': policy_params, 'lr': Config.POLICY_HEAD_LR},
        {'params': base_and_value_params, 'lr': Config.VALUE_BODY_LR}
    ])

    total_steps = Config.RL_TRAINING_EPISODES * 2
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[Config.POLICY_HEAD_LR, Config.VALUE_BODY_LR], total_steps=total_steps)
    scaler = GradScaler(enabled=(device == "cuda"))

    replay_buffer = deque(maxlen=Config.RL_BUFFER_SIZE)
    training_logs = []
    start_episode = 1

    checkpoint_files = glob.glob(os.path.join(Config.CHECKPOINT_DIR, 'rl_checkpoint_ep_*.pth'))
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda p: int(p.split('_')[-1].split('.')[0]))
        print(f"Resuming RL training from checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        replay_buffer = deque(checkpoint['replay_buffer'], maxlen=Config.RL_BUFFER_SIZE)
        start_episode = checkpoint['episode'] + 1
        if os.path.exists(Config.RL_LOG_PATH):
            with open(Config.RL_LOG_PATH, 'r') as f:
                training_logs = json.load(f)

    mcts = MCTS(model); env = KeyboardEnvironment()
    progress_bar = tqdm(range(start_episode, Config.RL_TRAINING_EPISODES + 1),
                        desc="RL Training", initial=start_episode-1, total=Config.RL_TRAINING_EPISODES)

    for episode in progress_bar:
        current_max_steps = Config.RL_MAX_STEPS
        for ep_threshold, steps in Config.RL_CURRICULUM_SCHEDULE.items():
            if episode >= ep_threshold:
                current_max_steps = steps

        episode_memory = []
        current_layout = generate_random_layout() if current_max_steps >= 10 else "qwertyuiopasdfghjkl;zxcvbnm,.'"
        user_weights_dict = generate_random_weights()
        user_weights_tensor = torch.tensor([user_weights_dict.get(k, 0.0) for k in Config.WEIGHT_KEYS], dtype=torch.float32).to(device)

        for step in range(current_max_steps):
            steps_left = current_max_steps - step
            improved_policy = mcts.run(current_layout, user_weights_tensor, steps_left)
            episode_memory.append([current_layout, user_weights_tensor.cpu(), improved_policy, None, steps_left])
            action = np.random.choice(range(Config.NUM_ACTIONS), p=improved_policy)
            current_layout = apply_action(current_layout, action)
            if action == Config.NO_OP_ACTION_INDEX and step > 0:
                break

        final_score_components = np.array([v for k, v in sorted(env.get_score_components(current_layout, user_weights_dict).items())])
        final_value_target = normalize_score_vector(final_score_components)
        for i in range(len(episode_memory)):
            episode_memory[i][3] = final_value_target

        replay_buffer.extend(episode_memory)

        if len(replay_buffer) < Config.RL_BATCH_SIZE * 10:
            continue

        model.train()

        if episode < Config.RL_WARMUP_EPISODES:
            for name, param in model.named_parameters():
                if 'policy_head' not in name:
                    param.requires_grad = False
        else:
            for param in model.parameters():
                param.requires_grad = True

        for _ in range(2):
            batch_data = random.sample(replay_buffer, Config.RL_BATCH_SIZE)
            layouts, weights, policies, values, steps_left_list = zip(*batch_data)

            graph_list = [prepare_graph_data_v4(l, w.to(device), s) for l,w,s in zip(layouts, weights, steps_left_list)]
            graph_batch = Batch.from_data_list(graph_list).to(device)
            target_p = torch.tensor(np.array(policies), dtype=torch.float32).to(device)
            target_v = torch.tensor(np.array(values), dtype=torch.float32).to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
                pred_p, pred_v = model(graph_batch)
                p_loss = F.cross_entropy(pred_p, target_p, label_smoothing=0.1)

                if episode < Config.RL_WARMUP_EPISODES:
                    total_loss = p_loss
                else:
                    v_loss = F.mse_loss(pred_v, target_v)
                    total_loss = p_loss + v_loss

            scaler.scale(total_loss).backward()

            if episode < Config.RL_WARMUP_EPISODES:
                scaler.step(optimizer)
                optimizer.param_groups[0]['lr'] = scheduler.get_last_lr()[0]
            else:
                scaler.step(optimizer)
                scheduler.step()

            scaler.update()

        if episode % 50 == 0:
            lrs = scheduler.get_last_lr()
            log_entry = {
                'phase': 'rl_train',
                'episode': episode,
                'curriculum_step': current_max_steps,
                'p_loss': p_loss.item(),
                'v_loss': v_loss.item() if episode >= Config.RL_WARMUP_EPISODES else 0,
                'policy_lr': lrs[0],
                'base_lr': lrs[1]
            }
            training_logs.append(log_entry)
            progress_bar.set_postfix({
                "p_loss": f"{p_loss.item():.4f}",
                "v_loss": f"{v_loss.item():.4f}" if episode >= Config.RL_WARMUP_EPISODES else "FROZEN"
            })

        if episode % 250 == 0:
            with open(Config.RL_LOG_PATH, 'w') as f:
                json.dump(training_logs, f, indent=2)

            checkpoint_data = {
                'episode': episode,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'replay_buffer': list(replay_buffer)
            }
            torch.save(checkpoint_data, os.path.join(Config.CHECKPOINT_DIR, f"rl_checkpoint_ep_{episode}.pth"))

    print("\nRL Training finished.")
    torch.save(model.state_dict(), Config.FINAL_MODEL_PATH)

# ==============================================================================
# PHASE 4: VISUALIZATION
# ==============================================================================
def visualize():
    """Phase 4: Generates publication-quality plots from the training logs."""
    print(f"--- PHASE 4: GENERATING VISUALIZATIONS ---")
    sns.set_theme(style="whitegrid")
    if not os.path.exists(Config.VISUALIZATIONS_DIR):
        os.makedirs(Config.VISUALIZATIONS_DIR)

    # Plot 1: Policy Pre-training
    try:
        df_p = pd.read_json(Config.POLICY_PRETRAIN_LOG_PATH)
        plt.figure(figsize=(12, 7))
        sns.lineplot(data=df_p, x='step', y='p_loss')
        plt.title('Phase 1: Policy Head Distillation Performance', fontsize=16)
        plt.xlabel('Training Step'); plt.ylabel('Policy Loss (Cross-Entropy)')
        plt.axhline(y=math.log(Config.NUM_ACTIONS), color='r', linestyle='--', label='Random Guessing Baseline')
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(Config.VISUALIZATIONS_DIR, "policy_pretrain_curve.png"))
        print("Generated policy pre-training plot.")
    except FileNotFoundError:
        print("Policy pre-training log not found, skipping plot.")

    # Plot 2: Value Pre-training
    try:
        df_v = pd.read_json(Config.VALUE_PRETRAIN_LOG_PATH)
        plt.figure(figsize=(12, 7))
        sns.lineplot(data=df_v, x='epoch', y='val_loss', label='Validation Loss')
        plt.title('Phase 2: Value Head Pre-training Performance', fontsize=16)
        plt.xlabel('Epoch'); plt.ylabel('Value Loss (MSE)')
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(Config.VISUALIZATIONS_DIR, "value_pretrain_curve.png"))
        print("Generated value pre-training plot.")
    except FileNotFoundError:
        print("Value pre-training log not found, skipping plot.")

    # Plot 3: RL Fine-tuning
    try:
        df_rl = pd.read_json(Config.RL_LOG_PATH)
        fig, ax1 = plt.subplots(figsize=(14, 8))
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Loss', color='tab:blue')
        sns.lineplot(data=df_rl, x='episode', y='p_loss', ax=ax1, label='Policy Loss', color='tab:blue', alpha=0.8)
        sns.lineplot(data=df_rl, x='episode', y='v_loss', ax=ax1, label='Value Loss', color='tab:cyan', alpha=0.8)
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.legend(loc='upper left')
        ax2 = ax1.twinx()
        ax2.set_ylabel('Curriculum Steps', color='tab:red')
        sns.lineplot(data=df_rl, x='episode', y='curriculum_step', ax=ax2, label='Curriculum Steps', color='tab:red', linestyle='--')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax2.legend(loc='upper right')
        fig.suptitle('Phase 3: RL Fine-Tuning Performance', fontsize=16)
        fig.tight_layout()
        plt.savefig(os.path.join(Config.VISUALIZATIONS_DIR, "rl_finetuning_curve.png"))
        print("Generated RL fine-tuning plot.")
    except FileNotFoundError:
        print("RL training log not found, skipping plot.")

    plt.show()

# ==============================================================================
# SCRIPT ENTRYPOINT
# ==============================================================================
def main():
    for dir_path in [Config.CHECKPOINT_DIR, Config.LOGS_DIR, Config.DATA_DIR, Config.VISUALIZATIONS_DIR]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    parser = argparse.ArgumentParser(description="Keyboard Layout Optimizer.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--mode', type=str, required=True,
        choices=['pretrain_policy', 'generate_value_data', 'pretrain_value', 'train_rl', 'visualize', 'all'],
        help=textwrap.dedent('''\
        (P1) pretrain_policy:      Train policy head from a random GNN.
        (P2a) generate_value_data: Create the supervised dataset for the value head.
        (P2b) pretrain_value:      Load policy model, then train value head + GNN.
        (P3) train_rl:             Run the final RL fine-tuning.
        (P4) visualize:            Generate plots from all training logs.
        all:                       Run all three training phases sequentially.
        '''))
    args = parser.parse_args()

    if args.mode == 'pretrain_policy':
        pretrain_policy()
    elif args.mode == 'generate_value_data':
        generate_data()
    elif args.mode == 'pretrain_value':
        try:
            policy_model = PolicyValueNetV4()
            policy_model.load_state_dict(torch.load(Config.PRETRAINED_POLICY_MODEL_PATH))
            print("Loaded policy-trained model to begin value training.")
            pretrain_value(policy_model)
        except FileNotFoundError:
            print(f"ERROR: Policy-trained model not found at {Config.PRETRAINED_POLICY_MODEL_PATH}. Run 'pretrain_policy' first.")
    elif args.mode == 'train_rl':
        try:
            full_model = PolicyValueNetV4()
            full_model.load_state_dict(torch.load(Config.PRETRAINED_FULL_MODEL_PATH))
            train_rl(full_model)
        except FileNotFoundError:
            print(f"ERROR: Doubly pre-trained model not found at {Config.PRETRAINED_FULL_MODEL_PATH}. Run all pre-training phases first.")
    elif args.mode == 'visualize':
        visualize()
    elif args.mode == 'all':
        print("--- RUNNING ALL TRAINING PHASES SEQUENTIALLY ---")
        policy_model = pretrain_policy()
        generate_data()
        full_model = pretrain_value(policy_model)
        train_rl(full_model)
        print("\n--- GENERATING FINAL VISUALIZATIONS ---")
        visualize()

if __name__ == '__main__':
    main()
