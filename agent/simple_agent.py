import time
from collections import deque
import numpy as np
from numpy.random import MT19937, Generator

from env import core
from env.action import RawAction
from env.base_agent import BaseAgent
from env.full_state import FullState
from env.node_data import NodeData


def random_generator(seed: int | None = None) -> Generator:
    bg = MT19937(seed)
    rg = Generator(bg)
    return rg


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values for array x."""
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x)


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation function."""
    return np.maximum(0, x)


def cross_entropy_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute cross-entropy loss."""
    # Clip predictions to avoid log(0)
    predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
    return -np.sum(targets * np.log(predictions))


class SimpleNetwork:
    """
    A simple neural network implemented with only NumPy that predicts actions and arguments.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        action_space_size: int,
        max_arg_value: int = 100,
        learning_rate: float = 0.001,
        seed: int | None = None,
    ):
        """
        Initialize the simple network.

        Args:
            input_dim: Dimension of input feature vector
            hidden_dim: Size of hidden layers
            action_space_size: Number of possible action indices
            max_arg_value: Maximum value for arguments (used for classification)
            learning_rate: Learning rate for gradient descent
            seed: Random seed for weight initialization
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_space_size = action_space_size
        self.max_arg_value = max_arg_value
        self.learning_rate = learning_rate

        # Set random seed for reproducible weight initialization
        if seed is not None:
            np.random.seed(seed)

        # Initialize weights using Xavier initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)

        self.W2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(hidden_dim)

        self.W3 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b3 = np.zeros(hidden_dim)

        # Output heads
        self.W_action = np.random.randn(hidden_dim, action_space_size) * np.sqrt(2.0 / hidden_dim)
        self.b_action = np.zeros(action_space_size)

        self.W_arg1 = np.random.randn(hidden_dim, max_arg_value + 1) * np.sqrt(2.0 / hidden_dim)
        self.b_arg1 = np.zeros(max_arg_value + 1)

        self.W_arg2 = np.random.randn(hidden_dim, max_arg_value + 1) * np.sqrt(2.0 / hidden_dim)
        self.b_arg2 = np.zeros(max_arg_value + 1)

        self.W_arg3 = np.random.randn(hidden_dim, max_arg_value + 1) * np.sqrt(2.0 / hidden_dim)
        self.b_arg3 = np.zeros(max_arg_value + 1)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward pass through the network.

        Args:
            x: Input array of shape (batch_size, input_dim) or (input_dim,)

        Returns:
            tuple of (action_logits, arg1_logits, arg2_logits, arg3_logits)
        """
        # Ensure input is 2D
        if x.ndim == 1:
            x = x.reshape(1, -1)

        assert x.ndim == 2, f"Input must be 2D array, got {x.ndim}D"

        # Forward pass through hidden layers
        h1 = relu(np.dot(x, self.W1) + self.b1)
        h2 = relu(np.dot(h1, self.W2) + self.b2)
        h3 = relu(np.dot(h2, self.W3) + self.b3)

        # Output predictions
        action_logits = np.dot(h3, self.W_action) + self.b_action
        arg1_logits = np.dot(h3, self.W_arg1) + self.b_arg1
        arg2_logits = np.dot(h3, self.W_arg2) + self.b_arg2
        arg3_logits = np.dot(h3, self.W_arg3) + self.b_arg3

        return action_logits, arg1_logits, arg2_logits, arg3_logits

    def predict(self, x: np.ndarray) -> tuple[int, int, int, int]:
        """
        Make predictions and return the most likely values.

        Args:
            x: Input array

        Returns:
            tuple of (action_idx, arg1_val, arg2_val, arg3_val)
        """
        action_logits, arg1_logits, arg2_logits, arg3_logits = self.forward(x)

        action_idx = int(np.argmax(action_logits[0]))
        arg1_val = int(np.argmax(arg1_logits[0]))
        arg2_val = int(np.argmax(arg2_logits[0]))
        arg3_val = int(np.argmax(arg3_logits[0]))

        return action_idx, arg1_val, arg2_val, arg3_val

    def train_step(
        self,
        x: np.ndarray,
        action_targets: np.ndarray,
        arg1_targets: np.ndarray,
        arg2_targets: np.ndarray,
        arg3_targets: np.ndarray,
        weights: np.ndarray | None = None,
    ):
        """
        Perform one training step using backpropagation.

        Args:
            x: Input data
            action_targets: Target action indices
            arg1_targets: Target arg1 values
            arg2_targets: Target arg2 values
            arg3_targets: Target arg3 values
            weights: Sample weights (optional)
        """
        batch_size = x.shape[0] if x.ndim > 1 else 1
        if weights is None:
            weights = np.ones(batch_size)

        # Forward pass
        action_logits, arg1_logits, arg2_logits, arg3_logits = self.forward(x)

        # Compute softmax probabilities
        action_probs = softmax(action_logits)
        arg1_probs = softmax(arg1_logits)
        arg2_probs = softmax(arg2_logits)
        arg3_probs = softmax(arg3_logits)

        # Create one-hot targets
        action_one_hot = np.zeros_like(action_probs)
        arg1_one_hot = np.zeros_like(arg1_probs)
        arg2_one_hot = np.zeros_like(arg2_probs)
        arg3_one_hot = np.zeros_like(arg3_probs)

        for i in range(batch_size):
            action_one_hot[i, action_targets[i]] = 1
            arg1_one_hot[i, arg1_targets[i]] = 1
            arg2_one_hot[i, arg2_targets[i]] = 1
            arg3_one_hot[i, arg3_targets[i]] = 1

        # Compute gradients
        # Output layer gradients
        action_grad = (action_probs - action_one_hot) * weights.reshape(-1, 1)
        arg1_grad = (arg1_probs - arg1_one_hot) * weights.reshape(-1, 1)
        arg2_grad = (arg2_probs - arg2_one_hot) * weights.reshape(-1, 1)
        arg3_grad = (arg3_probs - arg3_one_hot) * weights.reshape(-1, 1)

        # Backpropagate through the network
        # We'll use a simplified approach and just update based on the combined gradients

        # Get hidden activations for gradient computation
        if x.ndim == 1:
            x = x.reshape(1, -1)

        assert x.ndim == 2, f"Input must be 2D array, got {x.ndim}D"

        h1 = relu(np.dot(x, self.W1) + self.b1)
        h2 = relu(np.dot(h1, self.W2) + self.b2)
        h3 = relu(np.dot(h2, self.W3) + self.b3)

        # Update output layer weights
        self.W_action -= self.learning_rate * np.dot(h3.T, action_grad) / batch_size
        self.b_action -= self.learning_rate * np.mean(action_grad, axis=0)

        self.W_arg1 -= self.learning_rate * np.dot(h3.T, arg1_grad) / batch_size
        self.b_arg1 -= self.learning_rate * np.mean(arg1_grad, axis=0)

        self.W_arg2 -= self.learning_rate * np.dot(h3.T, arg2_grad) / batch_size
        self.b_arg2 -= self.learning_rate * np.mean(arg2_grad, axis=0)

        self.W_arg3 -= self.learning_rate * np.dot(h3.T, arg3_grad) / batch_size
        self.b_arg3 -= self.learning_rate * np.mean(arg3_grad, axis=0)

        # Simplified backprop for hidden layers (using combined gradient signal)
        combined_grad = (
            action_grad @ self.W_action.T +
            arg1_grad @ self.W_arg1.T +
            arg2_grad @ self.W_arg2.T +
            arg3_grad @ self.W_arg3.T
        )

        # h3 gradient
        h3_grad = combined_grad * (h3 > 0)  # ReLU derivative

        self.W3 -= self.learning_rate * np.dot(h2.T, h3_grad) / batch_size
        self.b3 -= self.learning_rate * np.mean(h3_grad, axis=0)

        # h2 gradient
        h2_grad = (h3_grad @ self.W3.T) * (h2 > 0)

        self.W2 -= self.learning_rate * np.dot(h1.T, h2_grad) / batch_size
        self.b2 -= self.learning_rate * np.mean(h2_grad, axis=0)

        # h1 gradient
        h1_grad = (h2_grad @ self.W2.T) * (h1 > 0)

        self.W1 -= self.learning_rate * np.dot(x.T, h1_grad) / batch_size
        self.b1 -= self.learning_rate * np.mean(h1_grad, axis=0)


class SimpleExperienceBuffer:
    """
    Simple experience buffer that stores state-action-reward tuples.
    """

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.experiences: deque[tuple[np.ndarray, list[int], float, bool]] = deque(maxlen=capacity)

    def add(self, state: np.ndarray, action: list[int], reward: float, success: bool):
        """Add an experience to the buffer."""
        self.experiences.append((state, action, reward, success))

    def get_all(self) -> list[tuple[np.ndarray, list[int], float, bool]]:
        """Get all experiences."""
        return list(self.experiences)

    def clear(self):
        """Clear all experiences."""
        self.experiences.clear()

    def __len__(self) -> int:
        return len(self.experiences)


class SimpleAgent(BaseAgent):
    """
    A simple agent that uses cross-entropy loss to learn from successful actions.
    Uses only NumPy and standard Python libraries.
    """

    def __init__(
        self,
        action_space_size: int,
        input_dim: int,
        hidden_dim: int = 128,
        learning_rate: float = 0.001,
        epsilon_start: float = 0.9,
        epsilon_end: float = 0.1,
        epsilon_decay: float = 0.995,
        max_arg_value: int = 50,
        seed: int | None = None,
    ):
        """
        Initialize the SimpleAgent.

        Args:
            action_space_size: Number of possible action indices
            input_dim: Dimension of input state vector
            hidden_dim: Size of hidden layers
            learning_rate: Learning rate for optimizer
            epsilon_start: Starting exploration probability
            epsilon_end: Minimum exploration probability
            epsilon_decay: Decay rate for exploration
            max_arg_value: Maximum value for arguments
            seed: Random seed for reproducibility
        """
        self.action_space_size = action_space_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.max_arg_value = max_arg_value
        self.rg = random_generator(seed)

        # Create network
        self.network = SimpleNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            action_space_size=action_space_size,
            max_arg_value=max_arg_value,
            learning_rate=learning_rate,
            seed=seed,
        )

        # Experience buffer
        self.experience_buffer = SimpleExperienceBuffer(capacity=2000)

        # Track statistics
        self.steps_done = 0

    def _preprocess_state(self, state: FullState) -> np.ndarray:
        """
        Convert a FullState to a NumPy array for the neural network.

        Args:
            state: The current environment state

        Returns:
            NumPy array representation of the state
        """
        # Convert tree structure to 2D array using NodeData
        node_data = NodeData(node=state, node_types=state.node_types())
        data_array = node_data.to_data_array()
        features = data_array.astype(np.float32)

        return features

    def select_action(self, state: FullState) -> RawAction:
        """
        Select an action using epsilon-greedy with the network predictions.

        Args:
            state: Current environment state

        Returns:
            Selected action to perform
        """
        # Preprocess state
        state_array = self._preprocess_state(state)

        # Epsilon-greedy action selection
        if self.rg.random() > self.epsilon:
            start = time.time()
            action_idx, arg1_val, arg2_val, arg3_val = self.network.predict(state_array)
            action_idx += 1  # Convert to 1-indexed

            end = time.time()
            print(
                time.strftime('%H:%M:%S'),
                f"> [Exploit - Time: {end - start:.4f}s]",
                f"Action: {action_idx}, Args: ({arg1_val}, {arg2_val}, {arg3_val})",
                f"State size: {state_array.shape}",
            )
        else:
            # Explore: random action with bias toward successful patterns
            action_idx = self.rg.integers(1, self.action_space_size + 1, dtype=int)

            # Completely random exploration
            arg1_val = max(0, int(round(self.rg.normal(loc=1.0, scale=3.0))))
            arg2_val = max(0, int(round(self.rg.normal(loc=1.0, scale=3.0))))
            arg3_val = max(0, int(round(self.rg.normal(loc=1.0, scale=3.0))))

            # Clamp to max value
            arg1_val = min(arg1_val, self.max_arg_value)
            arg2_val = min(arg2_val, self.max_arg_value)
            arg3_val = min(arg3_val, self.max_arg_value)

            # Sometimes zero out args 2 and 3
            if self.rg.random() < 0.3:
                arg3_val = 0
                if self.rg.random() < 0.3:
                    arg2_val = 0

            print(
                time.strftime('%H:%M:%S'),
                f"> [Explore] Action: {action_idx}, Args: ({arg1_val}, {arg2_val}, {arg3_val})",
                f"State size: {state_array.shape}",
            )

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
        Train the agent using cross-entropy loss on successful actions.

        Args:
            state: State before action
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            terminated: Whether episode ended naturally
            truncated: Whether episode was cut short
        """
        start = time.time()

        # Preprocess state
        state_array = self._preprocess_state(state)

        # Extract action parameters
        action_idx = action.action_index.apply().real(core.IInt).as_int - 1  # -1 for 0-indexing
        arg1_val = action.arg1.apply().real(core.IInt).as_int
        arg2_val = action.arg2.apply().real(core.IInt).as_int
        arg3_val = action.arg3.apply().real(core.IInt).as_int

        # Determine if this was a successful action (positive reward or goal achieved)
        success = reward > 0 or terminated

        # Add experience to buffer
        self.experience_buffer.add(
            state_array,
            [action_idx, arg1_val, arg2_val, arg3_val],
            reward,
            success
        )

        # Train on recent experiences every 5 steps, but only if we have successful examples
        if self.steps_done % 5 == 0 and len(self.experience_buffer) >= 10:
            self._train_network()

        # Update exploration rate
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.steps_done += 1

        end = time.time()
        print(
            time.strftime('%H:%M:%S'),
            f"[Train - Time: {end - start:.4f}s]",
            f"Reward: {reward:.3f}",
            f"Success: {success}",
            f"Epsilon: {self.epsilon:.3f}",
        )

    def _train_network(self):
        """Train the network using cross-entropy loss on successful actions."""
        experiences = self.experience_buffer.get_all()

        # Filter for successful experiences (positive reward or high reward)
        successful_experiences = [exp for exp in experiences if exp[2] > 0 or exp[3]]

        if len(successful_experiences) < 5:
            return  # Not enough successful examples to learn from

        # Prepare training data
        states = []
        actions = []
        arg1s = []
        arg2s = []
        arg3s = []
        weights = []

        for state, action, reward, _ in successful_experiences:
            states.append(state)
            actions.append(action[0])  # action index
            arg1s.append(action[1])
            arg2s.append(action[2])
            arg3s.append(action[3])
            # Weight by reward (successful actions with higher rewards get more weight)
            weights.append(max(0.1, reward))

        # Convert to NumPy arrays
        state_batch = np.array(states)
        action_batch = np.array(actions)
        arg1_batch = np.array(arg1s)
        arg2_batch = np.array(arg2s)
        arg3_batch = np.array(arg3s)
        weight_batch = np.array(weights)

        # Train the network
        self.network.train_step(
            state_batch, action_batch, arg1_batch, arg2_batch, arg3_batch, weight_batch
        )

        print(
            time.strftime('%H:%M:%S'),
            f"[Network Train] Successful experiences: {len(successful_experiences)}",
            f"Avg weight: {np.mean(weight_batch):.3f}",
        )

    def reset(self) -> None:
        """Reset the agent's episode-specific variables."""
        # Clear experience buffer at the end of episodes to focus on recent learning
        if len(self.experience_buffer) > 500:
            # Keep only the most recent 200 experiences
            recent_experiences = list(self.experience_buffer.experiences)[-200:]
            self.experience_buffer.clear()
            for exp in recent_experiences:
                self.experience_buffer.experiences.append(exp)

    def save(self, path: str) -> None:
        """Save the agent's model to disk."""
        # Save network weights using np.savez
        np.savez(
            path,
            # Network weights
            W1=self.network.W1,
            b1=self.network.b1,
            W2=self.network.W2,
            b2=self.network.b2,
            W3=self.network.W3,
            b3=self.network.b3,
            W_action=self.network.W_action,
            b_action=self.network.b_action,
            W_arg1=self.network.W_arg1,
            b_arg1=self.network.b_arg1,
            W_arg2=self.network.W_arg2,
            b_arg2=self.network.b_arg2,
            W_arg3=self.network.W_arg3,
            b_arg3=self.network.b_arg3,
            # Agent state (simple types only)
            steps_done=np.array(self.steps_done),
            epsilon=np.array(self.epsilon),
        )

    def load(self, path: str) -> None:
        """Load the agent's model from disk."""
        data = np.load(path, allow_pickle=True)

        # Load network weights directly from the flattened structure
        self.network.W1 = data['W1']
        self.network.b1 = data['b1']
        self.network.W2 = data['W2']
        self.network.b2 = data['b2']
        self.network.W3 = data['W3']
        self.network.b3 = data['b3']
        self.network.W_action = data['W_action']
        self.network.b_action = data['b_action']
        self.network.W_arg1 = data['W_arg1']
        self.network.b_arg1 = data['b_arg1']
        self.network.W_arg2 = data['W_arg2']
        self.network.b_arg2 = data['b_arg2']
        self.network.W_arg3 = data['W_arg3']
        self.network.b_arg3 = data['b_arg3']

        # Load agent state
        self.steps_done = int(data['steps_done'].item())
        self.epsilon = float(data['epsilon'].item())
