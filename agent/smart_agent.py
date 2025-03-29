import random
from collections import deque
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from env import core
from env.action import RawAction
from env.base_agent import BaseAgent
from env.full_state import FullState
from env.node_data import NodeData

class NodeFeatureExtractor(nn.Module):
    """
    Neural network module that processes the node data array and extracts features.
    This network handles a varying number of nodes by using a node-wise processing approach
    followed by global pooling.
    """

    def __init__(self, input_dim: int = 7, hidden_dim: int = 128, output_dim: int = 256):
        """
        Initialize the feature extractor network.

        Args:
            input_dim: Dimension of each node feature vector (excluding the node id)
            hidden_dim: Size of hidden layers
            output_dim: Size of output feature vector
        """
        super(NodeFeatureExtractor, self).__init__()

        # Node-wise feature extraction
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Global feature extraction
        self.global_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process a batch of node data.

        Args:
            x: Node feature tensor of shape (batch_size, num_nodes, input_dim)
              where num_nodes can vary between samples

        Returns:
            Global feature vector of shape (batch_size, output_dim)
        """
        # Process each node independently
        # Shape: (batch_size, num_nodes, hidden_dim)
        node_features: torch.Tensor = self.node_encoder(x)

        # Global pooling (mean across nodes)
        # Shape: (batch_size, hidden_dim)
        global_features = node_features.mean(dim=1)

        # Final feature extraction
        # Shape: (batch_size, output_dim)
        output = self.global_encoder(global_features)

        return output


class ActionPredictor(nn.Module):
    """
    Neural network module that predicts action indices and arguments.
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 128, action_space_size: int = 10):
        """
        Initialize the action predictor network.

        Args:
            input_dim: Dimension of input feature vector
            hidden_dim: Size of hidden layers
            action_space_size: Number of possible action indices
        """
        super(ActionPredictor, self).__init__()

        self.action_space_size = action_space_size

        # Common feature processing
        self.common = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )

        # Action index head
        self.action_head = nn.Linear(hidden_dim, action_space_size)

        # Argument heads (3 arguments, each producing an integer)
        self.arg1_head = nn.Linear(hidden_dim, 1)
        self.arg2_head = nn.Linear(hidden_dim, 1)
        self.arg3_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Predict action index and arguments.

        Args:
            x: Input feature vector of shape (batch_size, input_dim)

        Returns:
            Tuple of (action_logits, arg1, arg2, arg3) where:
                action_logits: Action probabilities of shape (batch_size, action_space_size)
                arg1, arg2, arg3: Argument values of shape (batch_size, 1)
        """
        common_features = self.common(x)

        action_logits = self.action_head(common_features)
        arg1 = self.arg1_head(common_features)
        arg2 = self.arg2_head(common_features)
        arg3 = self.arg3_head(common_features)

        return action_logits, arg1, arg2, arg3


class DQN(nn.Module):
    """
    Deep Q-Network for the agent.
    Combines the feature extractor and action predictor.
    """

    def __init__(
        self,
        input_dim: int = 7,
        feature_dim: int = 256,
        hidden_dim: int = 128,
        action_space_size: int = 10
    ):
        """
        Initialize the DQN.

        Args:
            input_dim: Dimension of each node feature vector (excluding the node id)
            feature_dim: Dimension of feature vector produced by node feature extractor
            hidden_dim: Size of hidden layers in action predictor
            action_space_size: Number of possible action indices
        """
        super(DQN, self).__init__()

        self.feature_extractor = NodeFeatureExtractor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=feature_dim
        )

        self.action_predictor = ActionPredictor(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            action_space_size=action_space_size
        )

    def forward(self, x: torch.Tensor) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Process node data and predict action.

        Args:
            x: Node feature tensor of shape (batch_size, num_nodes, input_dim)

        Returns:
            Tuple of (action_logits, arg1, arg2, arg3)
        """
        features = self.feature_extractor(x)
        return self.action_predictor(features)


class ExperienceReplayBuffer:
    """
    Buffer to store experience tuples for experience replay in DQN.
    """

    def __init__(self, capacity: int = 10000):
        """
        Initialize the buffer.

        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer: deque[tuple] = deque(maxlen=capacity)

    def add(
        self,
        state: np.ndarray,
        action: list[int],
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Add an experience to the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list[tuple]:
        """
        Sample a batch of experiences.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            List of experiences
        """
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

    def __len__(self) -> int:
        return len(self.buffer)


class SmartAgent(BaseAgent):
    """
    A reinforcement learning agent that uses a DQN to select actions.
    """

    def __init__(
        self,
        action_space_size: int = 10,
        feature_dim: int = 256,
        hidden_dim: int = 128,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: float = 0.995,
        replay_buffer_capacity: int = 10000,
        batch_size: int = 64,
        target_update_frequency: int = 100,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the SmartAgent.

        Args:
            action_space_size: Number of possible action indices
            feature_dim: Dimension of feature vector produced by node feature extractor
            hidden_dim: Size of hidden layers
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon_start: Starting value for epsilon (exploration probability)
            epsilon_end: Minimum value for epsilon
            epsilon_decay: Decay rate for epsilon
            replay_buffer_capacity: Capacity of experience replay buffer
            batch_size: Batch size for training
            target_update_frequency: How often to update target network
            device: Device to use for tensor operations
        """
        self.action_space_size = action_space_size
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency

        # Use GPU if available
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # Create policy and target networks
        self.policy_net = DQN(
            input_dim=7,  # We use 7 features from the node data array (excluding node id)
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            action_space_size=action_space_size
        ).to(self.device)

        self.target_net = DQN(
            input_dim=7,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            action_space_size=action_space_size
        ).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only used for evaluation

        # Create optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Create experience replay buffer
        self.replay_buffer = ExperienceReplayBuffer(capacity=replay_buffer_capacity)

        # Training stats
        self.steps_done = 0

    def _preprocess_state(self, state: FullState) -> torch.Tensor:
        """
        Convert a FullState to a tensor for the neural network.

        Args:
            state: The current environment state

        Returns:
            Tensor representation of the state
        """
        # Convert tree structure to 2D array using NodeData
        node_data = NodeData(node=state, node_types=state.node_types())
        data_array = node_data.to_data_array()
        features = data_array.astype(np.float32)

        # Convert to tensor and add batch dimension
        tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        return tensor

    def select_action(self, state: FullState) -> RawAction:
        """
        Select an action based on the current state using ε-greedy policy.

        Args:
            state: Current environment state

        Returns:
            Selected action to perform
        """
        # Preprocess state
        state_tensor = self._preprocess_state(state)

        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            # Exploit: use the model
            with torch.no_grad():
                action_logits, arg1, arg2, arg3 = self.policy_net(state_tensor)
                # +1 because action indices start at 1
                action_idx = action_logits.argmax(dim=1).item() + 1
                arg1_val = int(arg1.item())
                arg2_val = int(arg2.item())
                arg3_val = int(arg3.item())
        else:
            # Explore: random action
            action_idx = random.randint(1, self.action_space_size)
            arg1_val = random.randint(0, 100)  # Using reasonable ranges for arguments
            arg2_val = random.randint(0, 100)
            arg3_val = random.randint(0, 100)

        # Create and return RawAction
        raw_action = RawAction.with_raw_args(
            action_index=action_idx,
            arg1=arg1_val,
            arg2=arg2_val,
            arg3=arg3_val
        )

        return raw_action

    def train(
        self,
        state: FullState,
        action: RawAction,
        reward: float,
        next_state: FullState,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """
        Train the agent using an experience tuple.

        Args:
            state: State before action
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            terminated: Whether episode ended naturally (e.g. goal achieved)
            truncated: Whether episode was cut short (e.g. max steps reached)
        """
        # Preprocess states
        state_tensor = self._preprocess_state(state)
        next_state_tensor = self._preprocess_state(next_state)

        # Extract action parameters
        # -1 because our model outputs 0-indexed actions
        action_idx = action.action_index.apply().real(core.IInt).as_int - 1
        arg1_val = action.arg1.apply().real(core.IInt).as_int
        arg2_val = action.arg2.apply().real(core.IInt).as_int
        arg3_val = action.arg3.apply().real(core.IInt).as_int

        # Convert state tensors to numpy arrays for the replay buffer
        state_np = state_tensor.cpu().numpy().squeeze(0)
        next_state_np = next_state_tensor.cpu().numpy().squeeze(0)

        # Add experience to replay buffer
        self.replay_buffer.add(
            state_np,
            [action_idx, arg1_val, arg2_val, arg3_val],
            reward,
            next_state_np,
            terminated or truncated
        )

        # Only train if we have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample a batch from the replay buffer
        batch = self.replay_buffer.sample(self.batch_size)

        # Extract batch components and convert to tensors
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []

        for b_state, b_action, b_reward, b_next_state, b_done in batch:
            batch_states.append(b_state)
            batch_actions.append(b_action)
            batch_rewards.append(b_reward)
            batch_next_states.append(b_next_state)
            batch_dones.append(b_done)

        # Handle variable-length sequences by padding
        # Find the maximum number of nodes in the batch
        max_nodes = max(len(s) for s in batch_states)

        # Pad states and next_states to have the same number of nodes
        padded_states = []
        padded_next_states = []

        for s in batch_states:
            padded = np.zeros((max_nodes, s.shape[1]))
            padded[:len(s)] = s
            padded_states.append(padded)

        for s in batch_next_states:
            padded = np.zeros((max_nodes, s.shape[1]))
            padded[:len(s)] = s
            padded_next_states.append(padded)

        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(padded_states)).to(self.device)
        action_batch = torch.LongTensor(np.array(batch_actions)).to(self.device)
        reward_batch = torch.FloatTensor(np.array(batch_rewards)).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(padded_next_states)).to(self.device)
        done_batch = torch.FloatTensor(np.array(batch_dones)).to(self.device)

        # Compute Q values
        action_logits, arg1, arg2, arg3 = self.policy_net(state_batch)

        # Get Q values for the actions taken
        q_values = action_logits.gather(1, action_batch[:, 0].unsqueeze(1)).squeeze(1)

        # Compute target Q values
        with torch.no_grad():
            next_action_logits, _, _, _ = self.target_net(next_state_batch)
            next_q_values = next_action_logits.max(1)[0]
            target_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)

        # Compute loss
        loss = F.smooth_l1_loss(q_values, target_q_values)

        # Add losses for arguments
        arg1_loss = F.smooth_l1_loss(arg1.squeeze(1), action_batch[:, 1].float())
        arg2_loss = F.smooth_l1_loss(arg2.squeeze(1), action_batch[:, 2].float())
        arg3_loss = F.smooth_l1_loss(arg3.squeeze(1), action_batch[:, 3].float())

        total_loss = loss + arg1_loss + arg2_loss + arg3_loss

        # Optimize the model
        self.optimizer.zero_grad()
        total_loss.backward()
        # Clip gradients to prevent exploding gradients
        for param in self.policy_net.parameters():
            assert param.grad is not None
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Update target network
        self.steps_done += 1
        if self.steps_done % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def reset(self) -> None:
        """
        Reset the agent's episode-specific variables.
        """
        # Nothing to reset for now, epsilon decay continues between episodes

    def save(self, path: str) -> None:
        """
        Save the agent's model to disk.

        Args:
            path: Path to save the model
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'epsilon': self.epsilon,
        }, path)

    def load(self, path: str) -> None:
        """
        Load the agent's model from disk.

        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']
        self.epsilon = checkpoint['epsilon']
