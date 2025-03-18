# Advanced ML and Autonomous Agents - Project

## Comparing Tree Search and Reinforcement Learning Approaches for the King and Courtesan Game

This repository explores and compares classical tree search algorithms with modern reinforcement learning techniques in the context of the King and Courtesan game. The project demonstrates how different AI approaches can be applied to strategic game environments, highlighting their relative strengths and limitations.

## Project Overview
The main notebook `Reinforcement_Learning_Agents.ipynb` implements various reinforcement learning agents to solve the King and Courtesan game, comparing their performance against traditional tree search methods like Alpha-Beta pruning.

## Key Components
- Reinforcement learning implementations (DQN, REINFORCE)
- Alpha-Beta pruning algorithm implementation
- King and Courtesan game environment (in both Python and Java)
- Training and evaluation frameworks
- Performance comparison metrics and visualizations

## Repository Structure
```
├── Reinforcement_Learning_Agents.ipynb       # Main notebook with RL implementations
├── Complete_Outputs_Reinforcement_Learning_Agents.ipynb  # Notebook with execution results
├── KingAndCourtesanEnv.py                    # Python environment for the game
├── IDAlphaBetaClient.py                      # Iterative Deepening Alpha-Beta implementation
├── King_and_Courtesan_Game_Java/             # Java implementation of the game
├── models/RL_Agents/                         # Trained RL agent models
├── checkpoints/reinforce/                    # Training checkpoints
├── reinforce_train_stats                     # Statistics from training runs
├── test_functions.py                         # Testing utilities
├── RL_KAC.jar                                # Java executable for RL in the game
├── requirements.txt                          # Project dependencies
└── README.md                                 # This documentation file
```

## Technologies Used
- **Python**: Core implementation of RL algorithms
- **Java**: Alternative implementation of the game environment
- **Jupyter Notebook**: Interactive development and visualization
- **Libraries**: TensorFlow/PyTorch (implied for RL implementations)

## Installation
```bash
# Clone the repository
git clone https://github.com/GloireLINVANI/Advanced_ML_and_Autonomous_Agents-Project.git
cd Advanced_ML_and_Autonomous_Agents-Project

# Install requirements
pip install -r requirements.txt
```

## Usage
Launch the main notebook to explore the implementations and experiments:
```bash
jupyter notebook Reinforcement_Learning_Agents.ipynb
```

The notebook guides you through:
1. Understanding the King and Courtesan game rules and dynamics
2. Implementing various RL agents for the game
3. Training and evaluating the agents
4. Comparing performance with tree search approaches
5. Analyzing results and insights

## Methodology
This project implements a comparative approach between:
- **Tree Search Methods**: Alpha-Beta pruning with iterative deepening
- **Reinforcement Learning**: Various algorithms including policy-based and value-based methods

Results are analyzed in terms of win rates, decision quality, and computational efficiency.

## Contributing
Contributions are welcome. Please feel free to submit a pull request or open an issue.

## License
[License information not specified in the repository]
