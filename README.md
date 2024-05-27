# Tetris-RL
The code implements a Deep Q-Network (DQN) agent to learn and play Tetris through reinforcement learning. It defines the Tetris environment and trains the DQN agent using experience replay and a neural network to approximate Q-values. The agent's performance is visualized, and its parameters can be saved.


 

https://github.com/MrSaeidSeyfi/Tetris-RL/assets/87762511/6f3babcf-7713-4bdc-a52c-e276219e4889




## Usage

To train the DQN agent, run:
```bash
python tetris_dqn/dqn.py
```
To test the Tetris environment, run:
```bash
python tetris_dqn/tetris_env.py
```
## Improving RL Reward Methods

A key focus of this project is the ongoing improvement of the reinforcement learning reward methods. The current approach includes a combination of line-clear rewards, height penalties, and hole penalties. Future improvements aim to fine-tune these rewards to enhance the agent's learning efficiency and overall performance.
Current Reward Calculation

    Line Clear Reward: Rewards for clearing lines, with higher rewards for clearing multiple lines at once.
    Height Penalty: Penalizes the agent for increasing the height of the blocks, encouraging a flatter playing field.
    Holes Penalty: Penalizes the agent for creating holes beneath blocks, encouraging a more solid block structure.

## Future Improvements

    Adaptive Rewards: Implementing rewards that adapt based on the agent's current performance and learning stage.
    Complexity-based Penalties: Introducing penalties for complex block structures that are harder to clear.
    Exploration Incentives: Adding incentives for the agent to explore different strategies during the learning phase.

## Contributing

We welcome contributions to improve the Tetris DQN project. To contribute:

    Fork the repository.
    Create a new branch (git checkout -b feature-branch).
    Make your changes.
    Commit your changes (git commit -am 'Add new feature').
    Push to the branch (git push origin feature-branch).
    Create a new Pull Request.
