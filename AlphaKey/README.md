# Keyboard Layout Optimizer

A deep learning project to optimize keyboard layouts using Graph Neural Networks and Reinforcement Learning.

## 1. Prerequisites

- **Python 3.8+**
- **API Server**: Before running any script, you must have the external scoring API server running locally at `http://localhost:8888/`.

## 2. API Server Setup (SVOBODA)

This project requires an external API server for layout scoring. Before running any Python script, you must clone, build, and run the SVOBODA server.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/RusDoomer/SVOBODA.git
    ```
2.  **Install Dependencies:**
    ```bash
    sudo apt install libmicrohttpd-dev libjson-c-dev
    ```

2.  **Run the server:**
    Navigate into the directory and use Cargo to run the server in release mode.
    ```bash
    cd SVOBODA
    wget https://colemak.com/pub/corpus/iweb-corpus-samples-cleaned.txt.xz
    unxz iweb-corpus-samples-cleaned.txt.xz
    mv iweb-corpus-samples-cleaned.txt ./data/english/corpora/shai.txt
    make && ./svoboda
    ```
    The server must be running and accessible at `http://localhost:8888/`. Keep this terminal open while you run the training or inference scripts.


## 3. Installation

Install the required Python packages using pip.

1.  **Install PyTorch:**

2.  **Install PyTorch Geometric:**

3.  **Install other dependencies:**
    ```bash
    pip install requests numpy tqdm pandas matplotlib seaborn
    ```

## 4. How to Run

### Main RL Model Training (`main.py`)

This is the primary training pipeline. The simplest way to train the full model is to run all phases sequentially.

*   **Run all training phases:**
    ```bash
    python main.py --mode all
    ```

*   **Run a specific phase:**
    ```bash
    # Phase 1: Policy Pre-training
    python main.py --mode pretrain_policy

    # Phase 2a: Generate supervised data for value head
    python main.py --mode generate_value_data

    # Phase 2b: Value Head Pre-training
    python main.py --mode pretrain_value

    # Phase 3: Reinforcement Learning
    python main.py --mode train_rl

    # Phase 4: Generate training visualizations
    python main.py --mode visualize
    ```

### Standalone Greedy Model (`greedy_policy_trainer.py`)

This trains a separate, simpler "greedy" policy model.

*   **Train the model:**
    ```bash
    python greedy_policy_trainer.py --mode train
    ```
*   **Generate visualizations from the training log:**
    ```bash
    python greedy_policy_trainer.py --mode visualize
    ```

### Inference (`inference.py`)

Use a trained model to optimize a keyboard layout.

*   **Run the final RL model on the QWERTY layout:**
    ```bash
    python inference.py --model rl --layout qwerty --max_swaps 29
    ```
*   **Run the greedy model on a random layout:**
    ```bash
    python inference.py --model greedy --layout random
    ```
*   **Run with custom weights:**
    ```bash
    python inference.py --model rl --layout qwerty --sfb 1.5 --alt -0.8
    ```
