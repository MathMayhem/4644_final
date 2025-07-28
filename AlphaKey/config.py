import torch
import os

class Config:
    # ==========================================================================
    # 1. CORE & ENVIRONMENT CONSTANTS
    # ==========================================================================

    # --- Reproducibility ---
    # A fixed seed ensures that all random operations are deterministic.
    RANDOM_SEED = 42

    # --- Environment ---
    ALL_CHARS = "qwertyuiopasdfghjkl;zxcvbnm,.'"
    CHAR_TO_INT = {char: i for i, char in enumerate(ALL_CHARS)}
    NUM_KEYS = len(ALL_CHARS)
    WEIGHT_KEYS = ["sfb", "sfs", "lsb", "alt", "rolls"] # The order must be consistent
    NUM_WEIGHTS = len(WEIGHT_KEYS)
    API_URL = "http://localhost:8888/"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Action Space ---
    NUM_SWAPS = (NUM_KEYS * (NUM_KEYS - 1)) // 2
    NO_OP_ACTION_INDEX = NUM_SWAPS
    NUM_ACTIONS = NUM_SWAPS + 1   # Total action space size (swaps + No-Op)

    # ==========================================================================
    # 2. MODEL ARCHITECTURE (V4.1)
    # ==========================================================================

    # --- Feature Engineering Dimensions ---
    NUM_FEAT_HAND = 2
    NUM_FEAT_FINGER = 8
    NUM_FEAT_ROW = 3
    NUM_FEAT_STEPS_LEFT = 1 # A single scalar for the remaining steps

    # --- Input and Layer Dimensions ---
    NODE_INPUT_DIM = (NUM_KEYS + NUM_WEIGHTS + NUM_FEAT_HAND + NUM_FEAT_FINGER + NUM_FEAT_ROW + NUM_FEAT_STEPS_LEFT)
    GNN_EMBEDDING_DIM = 96
    GAT_LAYERS = 6
    NUM_ATTENTION_HEADS = 4

    # Hidden dimension for the gating network in GlobalAttention.
    ATTENTION_GATE_DIM = 128

    # The output of the pooling layer is the size of the GNN embedding.
    FINAL_EMBEDDING_DIM = GNN_EMBEDDING_DIM
    HEAD_HIDDEN_DIM = 256

    # ==========================================================================
    # 3. FILE PATHS & DIRECTORIES
    # ==========================================================================

    CHECKPOINT_DIR = "checkpoints"
    LOGS_DIR = "logs"
    DATA_DIR = "data"
    VISUALIZATIONS_DIR = "visualizations"

    # Phase 1 (Policy Pre-train) Output
    POLICY_PRETRAIN_LOG_PATH = os.path.join(LOGS_DIR, "policy_pretrain_log.json")
    PRETRAINED_POLICY_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "pretrained_policy_model.pth")

    # Phase 2 (Value Pre-train) Data and Outputs
    SUPERVISED_DATA_PATH = os.path.join(DATA_DIR, "supervised_layout_data.json")
    VALUE_PRETRAIN_LOG_PATH = os.path.join(LOGS_DIR, "value_pretrain_log.json")
    PRETRAINED_FULL_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "pretrained_full_model.pth")

    # Phase 3 (RL Fine-tune) Outputs
    RL_LOG_PATH = os.path.join(LOGS_DIR, "rl_training_log.json")
    FINAL_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "final_rl_model.pth")

    # ==========================================================================
    # 4. TRAINING PHASE HYPERPARAMETERS
    # ==========================================================================

    # --- PHASE 1: POLICY HEAD PRE-TRAINING (THE REPLICA) ---
    POLICY_PRETRAIN_STEPS = 5000  # Number of on-the-fly training steps.
    POLICY_PRETRAIN_BUFFER_SIZE = 20000 # A replay buffer for the online learning.
    POLICY_PRETRAIN_BATCH_SIZE = 512
    POLICY_PRETRAIN_LR = 3e-4
    # How many simulations the slow, sequential Oracle uses to generate a "perfect" policy.
    POLICY_ORACLE_SIMULATIONS = 64

    # --- PHASE 2: VALUE HEAD & GNN PRE-TRAINING ---
    # Data Generation for Value Head
    NUM_SUPERVISED_SAMPLES = 500_000
    API_BATCH_SIZE = 512
    # Supervised Training
    SCORE_COMP_MIN = -100.0
    SCORE_COMP_MAX = 100.0
    VALUE_PRETRAIN_EPOCHS = 8
    VALUE_PRETRAIN_BATCH_SIZE = 512
    VALUE_PRETRAIN_LR = 1e-3
    VALUE_PRETRAIN_PATIENCE = 3 # Early stopping patience

    # --- PHASE 3: REINFORCEMENT LEARNING (FINE-TUNING) ---
    RL_TRAINING_EPISODES = 25_000
    RL_BUFFER_SIZE = 100_000
    RL_BATCH_SIZE = 256
    RL_WARMUP_EPISODES = 5000

    # The maximum number of swaps in a single RL episode. Used for normalization.
    RL_MAX_STEPS = 29

    # Curriculum to gradually increase the RL task's difficulty.
    RL_CURRICULUM_SCHEDULE = {
        0: 1,
        10000: 3,
        13000: 5,
        16000: 10,
        19000: 20,
        22000: RL_MAX_STEPS
    }

    # Differential learning rates for fine-tuning.
    POLICY_HEAD_LR = 5e-5
    VALUE_BODY_LR = 1e-5

    # MCTS search parameters for the fast, batched MCTS used in RL.
    MCTS_SIMULATIONS = 512
    CPUCT = 1.25 # Exploration vs. exploitation constant.
