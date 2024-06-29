cartpole_rl/
├── main.py
├── model.py
├── agent.py
├── train.py
└── requirements.txt

markdown
Copy code

- `main.py`: Loads the trained model and evaluates the agent in the environment.
- `model.py`: Defines the Q-Network architecture.
- `agent.py`: Implements the DQN agent, including action selection and experience replay.
- `train.py`: Handles the training process.
- `requirements.txt`: Lists the dependencies required to run the project.

## Dependencies

- Python 3.7+
- TensorFlow 2.8.0
- Gym 0.21.0
- Numpy 1.22.0

Install the dependencies using the following command:

```bash
pip install -r requirements.txt
Training the Agent
To train the agent, run the following command:

bash
Copy code
python train.py
This will train the DQN agent in the CartPole environment and save the trained model every 50 episodes to the cartpole_model/ directory.

Evaluating the Agent
To evaluate the trained agent, run the following command:

bash
Copy code
python main.py
This will load the trained model and render the agent's performance in the CartPole environment for 10 episodes.

Q-Network Architecture
The Q-Network is a neural network with the following architecture:

Input layer: State size (4 for CartPole)
Hidden layer 1: Dense layer with 24 units and ReLU activation
Hidden layer 2: Dense layer with 24 units and ReLU activation
Output layer: Action size (2 for CartPole)

DQN Agent
The DQN agent is implemented with the following features:
Experience replay: Stores experiences in a replay memory and samples mini-batches for training.
Epsilon-greedy policy: Balances exploration and exploitation by choosing random actions with probability epsilon.
Target network: Uses the current Q-Network for action selection and target calculation.
Hyperparameters
The agent is trained with the following hyperparameters:

Batch size: 64
Gamma (discount rate): 0.99
Epsilon (exploration rate): 1.0 (decayed over time)
Epsilon min: 0.01
Epsilon decay: 0.995
Learning rate: 0.001
References
OpenAI Gym
TensorFlow
License
This project is licensed under the MIT License.
