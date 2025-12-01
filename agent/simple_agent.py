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
        # Ensure input is 3D
        if x.ndim == 2:
            x = x.reshape(1, x.shape[0], x.shape[1])

        assert x.ndim == 3, f"Input must be 3D array, got {x.ndim}D"

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

    def predict_q_values(self, x: np.ndarray) -> np.ndarray:
        """
        Predict Q-values for all actions given state input.

        Args:
            x: Input array representing state

        Returns:
            Array of Q-values for each action
        """
        # Ensure input is 3D
        if x.ndim == 2:
            x = x.reshape(1, x.shape[0], x.shape[1])

        assert x.ndim == 3, f"Input must be 3D array, got {x.ndim}D"

        # Forward pass through hidden layers
        h1 = relu(np.dot(x, self.W1) + self.b1)
        h2 = relu(np.dot(h1, self.W2) + self.b2)
        h3 = relu(np.dot(h2, self.W3) + self.b3)

        # Output Q-values (we repurpose the action output head)
        q_values = np.dot(h3, self.W_action) + self.b_action

        return q_values

    def predict_decomposed_q_values(
        self,
        x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict decomposed Q-values for action and each argument.

        Args:
            x: Input array representing state

        Returns:
            Tuple of (action_q_values, arg1_q_values, arg2_q_values, arg3_q_values)
        """
        # Ensure input is 3D
        if x.ndim == 2:
            x = x.reshape(1, x.shape[0], x.shape[1])

        assert x.ndim == 3, f"Input must be 3D array, got {x.ndim}D"

        # Forward pass through hidden layers
        h1 = relu(np.dot(x, self.W1) + self.b1)
        h2 = relu(np.dot(h1, self.W2) + self.b2)
        h3 = relu(np.dot(h2, self.W3) + self.b3)

        print('shape', x.shape, 'h3.shape', h3.shape)

        # Output Q-values for each component
        action_q_values = np.dot(h3, self.W_action) + self.b_action
        arg1_q_values = np.dot(h3, self.W_arg1) + self.b_arg1
        arg2_q_values = np.dot(h3, self.W_arg2) + self.b_arg2
        arg3_q_values = np.dot(h3, self.W_arg3) + self.b_arg3

        return action_q_values, arg1_q_values, arg2_q_values, arg3_q_values

    def combine_q_values(
        self,
        action_q: np.ndarray,
        arg1_q: np.ndarray,
        arg2_q: np.ndarray,
        arg3_q: np.ndarray,
        action_idx: np.ndarray | None = None,
        arg1_idx: np.ndarray | None = None,
        arg2_idx: np.ndarray | None = None,
        arg3_idx: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Combine decomposed Q-values into a single Q-value.

        If action indices are provided, returns Q-values for those specific actions,
        otherwise returns the combined Q-values for all action combinations.

        Args:
            action_q: Q-values for actions
            arg1_q: Q-values for argument 1
            arg2_q: Q-values for argument 2
            arg3_q: Q-values for argument 3
            action_idx: Optional indices of actions to evaluate
            arg1_idx: Optional indices of arg1 values to evaluate
            arg2_idx: Optional indices of arg2 values to evaluate
            arg3_idx: Optional indices of arg3 values to evaluate

        Returns:
            Combined Q-values
        """
        batch_size = action_q.shape[0]

        if action_idx is not None:
            # Return Q-values for specific actions
            combined_q = np.zeros(batch_size)
            for i in range(batch_size):
                a_idx = action_idx[i]
                a1_idx = arg1_idx[i] if arg1_idx is not None else np.argmax(arg1_q[i])
                a2_idx = arg2_idx[i] if arg2_idx is not None else np.argmax(arg2_q[i])
                a3_idx = arg3_idx[i] if arg3_idx is not None else np.argmax(arg3_q[i])

                print(f"Combining Q-values for batch {i}: \n"
                      f"action {a_idx.shape}\n"
                      f"arg1 {a1_idx.shape}\n"
                      f"arg2 {a2_idx.shape}\n"
                      f"arg3 {a3_idx.shape}\n"
                      f"action_q: {action_q[i, a_idx].shape}\n"
                      f"arg1_q: {arg1_q[i, a1_idx].shape}\n"
                      f"arg2_q: {arg2_q[i, a2_idx].shape}\n"
                      f"arg3_q: {arg3_q[i, a3_idx].shape}\n")

                combined_q[i] = (
                    action_q[i, a_idx] +
                    arg1_q[i, a1_idx] +
                    arg2_q[i, a2_idx] +
                    arg3_q[i, a3_idx]
                ) / 4.0

            return combined_q
        else:
            # Compute max Q-value across all possible action combinations
            max_q_values = np.zeros(batch_size)

            for i in range(batch_size):
                max_q = -float('inf')
                for a_idx in range(action_q.shape[1]):
                    for a1_idx in range(arg1_q.shape[1]):
                        for a2_idx in range(arg2_q.shape[1]):
                            for a3_idx in range(arg3_q.shape[1]):
                                q_val = (
                                    action_q[i, a_idx] +
                                    arg1_q[i, a1_idx] +
                                    arg2_q[i, a2_idx] +
                                    arg3_q[i, a3_idx]
                                ) / 4.0
                                if q_val > max_q:
                                    max_q = q_val

                max_q_values[i] = max_q

            return max_q_values

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
        # Get hidden activations for gradient computation
        if x.ndim == 1:
            x = x.reshape(1, -1)

        assert x.ndim == 2, f"Input must be 2D array, got {x.ndim}D"

        batch_size = x.shape[0]

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
        action_grad = action_probs - action_one_hot
        arg1_grad = arg1_probs - arg1_one_hot
        arg2_grad = arg2_probs - arg2_one_hot
        arg3_grad = arg3_probs - arg3_one_hot

        if weights is not None:
            action_grad = action_grad * weights.reshape(-1, 1)
            arg1_grad = arg1_grad * weights.reshape(-1, 1)
            arg2_grad = arg2_grad * weights.reshape(-1, 1)
            arg3_grad = arg3_grad * weights.reshape(-1, 1)

        # Backpropagate through the network
        # We'll use a simplified approach and just update based on the combined gradients

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

    def q_learning_update(self, states: np.ndarray, actions: np.ndarray, targets: np.ndarray):
        """
        Update network weights using Q-learning with MSE loss.

        Args:
            states: Batch of states (batch_size, input_dim)
            actions: Actions taken for each state (batch_size,)
            targets: Target Q-values for the actions (batch_size,)
        """
        batch_size = states.shape[0]

        # Forward pass
        h1 = relu(np.dot(states, self.W1) + self.b1)
        h2 = relu(np.dot(h1, self.W2) + self.b2)
        h3 = relu(np.dot(h2, self.W3) + self.b3)

        # Output Q-values
        q_values = np.dot(h3, self.W_action) + self.b_action

        # Create target vector - copy current predictions
        target_vector = np.copy(q_values)

        # Update only the Q-values for the actions that were taken
        for i, action_idx in enumerate(actions):
            target_vector[i, action_idx] = targets[i]

        # Compute gradient of MSE loss
        q_grad = 2.0 * (q_values - target_vector) / batch_size

        # Backpropagate and update weights
        # Update output layer weights
        self.W_action -= self.learning_rate * np.dot(h3.T, q_grad)
        self.b_action -= self.learning_rate * np.mean(q_grad, axis=0)

        # Backpropagate through hidden layers
        h3_grad = np.dot(q_grad, self.W_action.T) * (h3 > 0)  # ReLU derivative

        self.W3 -= self.learning_rate * np.dot(h2.T, h3_grad)
        self.b3 -= self.learning_rate * np.mean(h3_grad, axis=0)

        h2_grad = np.dot(h3_grad, self.W3.T) * (h2 > 0)

        self.W2 -= self.learning_rate * np.dot(h1.T, h2_grad)
        self.b2 -= self.learning_rate * np.mean(h2_grad, axis=0)

        h1_grad = np.dot(h2_grad, self.W2.T) * (h1 > 0)

        self.W1 -= self.learning_rate * np.dot(states.T, h1_grad)
        self.b1 -= self.learning_rate * np.mean(h1_grad, axis=0)

        # Calculate loss for reporting
        loss = np.mean(np.square(target_vector - q_values))
        return loss

    def q_decomposition_update(
        self,
        states: np.ndarray,
        actions: list[np.ndarray],
        target_q: np.ndarray,
    ):
        """
        Update network weights using Q-decomposition approach.

        Args:
            states: Batch of states (batch_size, input_dim)
            actions: List of action component indices [action_idx, arg1_idx, arg2_idx, arg3_idx]
            target_q: Target Q-values (batch_size,)

        Returns:
            Loss value
        """
        action_idx, arg1_idx, arg2_idx, arg3_idx = actions
        batch_size = states.shape[0]

        # Forward pass
        h1 = relu(np.dot(states, self.W1) + self.b1)
        h2 = relu(np.dot(h1, self.W2) + self.b2)
        h3 = relu(np.dot(h2, self.W3) + self.b3)

        # Get current Q-values for each component
        action_q = np.dot(h3, self.W_action) + self.b_action
        arg1_q = np.dot(h3, self.W_arg1) + self.b_arg1
        arg2_q = np.dot(h3, self.W_arg2) + self.b_arg2
        arg3_q = np.dot(h3, self.W_arg3) + self.b_arg3

        # Create target vectors for each component
        # We'll use the same target value for each component to ensure they all contribute equally
        action_target = np.copy(action_q)
        arg1_target = np.copy(arg1_q)
        arg2_target = np.copy(arg2_q)
        arg3_target = np.copy(arg3_q)

        # Update only the values for the actions that were taken
        for i in range(batch_size):
            a_idx = action_idx[i]
            a1_idx = arg1_idx[i]
            a2_idx = arg2_idx[i]
            a3_idx = arg3_idx[i]

            # Distribute the target Q-value to each component
            action_target[i, a_idx] = target_q[i]
            arg1_target[i, a1_idx] = target_q[i]
            arg2_target[i, a2_idx] = target_q[i]
            arg3_target[i, a3_idx] = target_q[i]

        # Compute gradients
        action_grad = 2.0 * (action_q - action_target) / batch_size
        arg1_grad = 2.0 * (arg1_q - arg1_target) / batch_size
        arg2_grad = 2.0 * (arg2_q - arg2_target) / batch_size
        arg3_grad = 2.0 * (arg3_q - arg3_target) / batch_size

        # Update output layer weights
        self.W_action -= self.learning_rate * np.dot(h3.T, action_grad)
        self.b_action -= self.learning_rate * np.mean(action_grad, axis=0)

        self.W_arg1 -= self.learning_rate * np.dot(h3.T, arg1_grad)
        self.b_arg1 -= self.learning_rate * np.mean(arg1_grad, axis=0)

        self.W_arg2 -= self.learning_rate * np.dot(h3.T, arg2_grad)
        self.b_arg2 -= self.learning_rate * np.mean(arg2_grad, axis=0)

        self.W_arg3 -= self.learning_rate * np.dot(h3.T, arg3_grad)
        self.b_arg3 -= self.learning_rate * np.mean(arg3_grad, axis=0)

        # Backpropagate through hidden layers
        # Combine gradients from all output heads
        combined_grad = (
            action_grad @ self.W_action.T +
            arg1_grad @ self.W_arg1.T +
            arg2_grad @ self.W_arg2.T +
            arg3_grad @ self.W_arg3.T
        )

        # h3 gradient
        h3_grad = combined_grad * (h3 > 0)  # ReLU derivative

        self.W3 -= self.learning_rate * np.dot(h2.T, h3_grad)
        self.b3 -= self.learning_rate * np.mean(h3_grad, axis=0)

        # h2 gradient
        h2_grad = np.dot(h3_grad, self.W3.T) * (h2 > 0)

        self.W2 -= self.learning_rate * np.dot(h1.T, h2_grad)
        self.b2 -= self.learning_rate * np.mean(h2_grad, axis=0)

        # h1 gradient
        h1_grad = np.dot(h2_grad, self.W2.T) * (h1 > 0)

        self.W1 -= self.learning_rate * np.dot(states.T, h1_grad)
        self.b1 -= self.learning_rate * np.mean(h1_grad, axis=0)

        # Calculate loss for reporting (average of the individual component losses)
        action_loss = np.mean(np.square(action_target - action_q))
        arg1_loss = np.mean(np.square(arg1_target - arg1_q))
        arg2_loss = np.mean(np.square(arg2_target - arg2_q))
        arg3_loss = np.mean(np.square(arg3_target - arg3_q))

        return (action_loss + arg1_loss + arg2_loss + arg3_loss) / 4.0

class SimpleExperienceBuffer:
    """
    Simple experience buffer that stores state-action-reward-next_state tuples for Q-learning.
    """

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.experiences: deque[
            tuple[np.ndarray, list[int], float, np.ndarray, bool]
        ] = deque(maxlen=capacity)

    def add(
        self,
        state: np.ndarray,
        action: list[int],
        reward: float,
        next_state: np.ndarray,
        terminated: bool
    ):
        """Add an experience to the buffer."""
        self.experiences.append((state, action, reward, next_state, terminated))

    def get_all(self) -> list[tuple[np.ndarray, list[int], float, np.ndarray, bool]]:
        """Get all experiences."""
        return list(self.experiences)

    def clear(self):
        """Clear all experiences."""
        self.experiences.clear()

    def __len__(self) -> int:
        return len(self.experiences)


class SimpleAgent(BaseAgent):
    """
    A simple agent that uses Q-learning to learn optimal actions.
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
        gamma: float = 0.99,  # Discount factor for future rewards
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
            gamma: Discount factor for future rewards in Q-learning
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
        self.gamma = gamma
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
        print('features', features.shape)

        return features

    def select_action(self, state: FullState) -> RawAction:
        """
        Select an action using epsilon-greedy with the network predictions.
        Uses Q-decomposition to select the best action components.

        Args:
            state: Current environment state

        Returns:
            Selected action to perform
        """
        # Preprocess state
        state_array = self._preprocess_state(state)
        print('state_array', state_array.shape)

        start = time.time()
        exploit = self.rg.random() > self.epsilon
        semi_exploit = (not exploit) and (self.rg.random() > self.epsilon)
        random = (not exploit) and (not semi_exploit)

        # Epsilon-greedy action selection
        if random:
            # Explore: random action with bias toward successful patterns
            action_idx = self.rg.integers(1, self.action_space_size + 1, dtype=int)

            # Random exploration
            arg1_val = self.rg.integers(0, self.max_arg_value + 1, dtype=int)
            arg2_val = self.rg.integers(0, self.max_arg_value + 1, dtype=int)
            arg3_val = self.rg.integers(0, self.max_arg_value + 1, dtype=int)

            # Sometimes zero out args 2 and 3
            if self.rg.random() < 0.3:
                arg3_val = 0
                if self.rg.random() < 0.3:
                    arg2_val = 0
        else:
            # Use decomposed Q-values to select the best action
            action_q, arg1_q, arg2_q, arg3_q = self.network.predict_decomposed_q_values(state_array)

            # Select action components with highest individual Q-values
            action_idx = int(np.argmax(action_q[0])) + 1  # Convert to 1-indexed
            arg1_val = int(np.argmax(arg1_q[0]))
            arg2_val = int(np.argmax(arg2_q[0]))
            arg3_val = int(np.argmax(arg3_q[0]))

            if semi_exploit:
                # Semi-random exploration
                scale = self.max_arg_value / 8
                arg1_val = int(round(self.rg.normal(loc=arg1_val, scale=scale)))
                arg2_val = int(round(self.rg.normal(loc=arg2_val, scale=scale)))
                arg3_val = int(round(self.rg.normal(loc=arg3_val, scale=scale)))

                # Clamp to max value
                arg1_val = max(0, min(arg1_val, self.max_arg_value))
                arg2_val = max(0, min(arg2_val, self.max_arg_value))
                arg3_val = max(0, min(arg3_val, self.max_arg_value))

        end = time.time()

        name = (
            'Explore'
            if random
            else (
                'Semi-Exploit'
                if semi_exploit
                else 'Exploit'
            )
        )
        print(
            time.strftime('%H:%M:%S'),
            f"> [{name} - Time: {end - start:.4f}s]",
            f"Action: {action_idx}, Args: ({arg1_val}, {arg2_val}, {arg3_val})",
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
        Train the agent using Q-learning.

        Args:
            state: State before action
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            terminated: Whether episode ended naturally
            truncated: Whether episode was cut short
        """
        start = time.time()

        # Preprocess states
        state_array = self._preprocess_state(state)
        next_state_array = self._preprocess_state(next_state)

        # Extract action parameters
        action_idx = action.action_index.apply().real(core.IInt).as_int - 1  # -1 for 0-indexing
        arg1_val = action.arg1.apply().real(core.IInt).as_int
        arg2_val = action.arg2.apply().real(core.IInt).as_int
        arg3_val = action.arg3.apply().real(core.IInt).as_int

        # Add experience to buffer
        self.experience_buffer.add(
            state_array,
            [action_idx, arg1_val, arg2_val, arg3_val],
            reward,
            next_state_array,
            terminated
        )

        # Train on experiences every few steps
        if self.steps_done % 5 == 0 and len(self.experience_buffer) >= 10:
            self._train_q_network()

        # Update exploration rate
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.steps_done += 1

        end = time.time()
        print(            time.strftime('%H:%M:%S'),
            f"[Train - Time: {end - start:.4f}s]",
            f"Reward: {reward:.3f}",
            f"Terminated: {terminated}",
            f"Epsilon: {self.epsilon:.3f}",
        )

    def _normalize_states(self, states: list[np.ndarray]) -> np.ndarray:
        max_nodes = max(len(s) for s in states)
        normalized_states: list[np.ndarray] = []
        for state in states:
            nodes = len(state)
            if nodes < max_nodes:
                # Pad with zeros to match max nodes
                padding = np.zeros((max_nodes - nodes, state.shape[1]), dtype=int)
                normalized_state = np.vstack((state, padding))
            else:
                normalized_state = state
            normalized_states.append(normalized_state)
        return np.array(normalized_states)

    def _train_q_network(self):
        """Train the network using Q-decomposition approach."""
        experiences = self.experience_buffer.get_all()

        if len(experiences) < 10:
            return  # Not enough examples to learn from

        # Prepare training data
        states = []
        action_indices = []
        arg1_indices = []
        arg2_indices = []
        arg3_indices = []
        rewards = []
        next_states = []
        terminateds = []

        for state, full_action, reward, next_state, terminated in experiences:
            action_idx, arg1, arg2, arg3 = full_action

            states.append(state)
            action_indices.append(action_idx)
            arg1_indices.append(arg1)
            arg2_indices.append(arg2)
            arg3_indices.append(arg3)
            rewards.append(reward)
            next_states.append(next_state)
            terminateds.append(terminated)

        states = self._normalize_states(states)
        next_states = self._normalize_states(next_states)

        # Convert to NumPy arrays
        state_batch = np.array(states)
        action_batch = np.array(action_indices)
        arg1_batch = np.array(arg1_indices)
        arg2_batch = np.array(arg2_indices)
        arg3_batch = np.array(arg3_indices)
        reward_batch = np.array(rewards)
        next_state_batch = np.array(next_states)
        terminated_batch = np.array(terminateds, dtype=np.float32)

        # Get next state decomposed Q-values
        prediction = self.network.predict_decomposed_q_values(next_state_batch)
        next_action_q, next_arg1_q, next_arg2_q, next_arg3_q = prediction

        # For each next state, find the best action components
        next_action_idx = np.argmax(next_action_q, axis=1)
        next_arg1_idx = np.argmax(next_arg1_q, axis=1)
        next_arg2_idx = np.argmax(next_arg2_q, axis=1)
        next_arg3_idx = np.argmax(next_arg3_q, axis=1)

        # Calculate combined Q-values for next states
        max_next_q_values = self.network.combine_q_values(
            action_q=next_action_q,
            arg1_q=next_arg1_q,
            arg2_q=next_arg2_q,
            arg3_q=next_arg3_q,
            action_idx=next_action_idx,
            arg1_idx=next_arg1_idx,
            arg2_idx=next_arg2_idx,
            arg3_idx=next_arg3_idx
        )

        # Calculate target Q-values using Q-learning formula:
        # Q(s,a) = r + gamma * max(Q(s',a')) * (1 - terminated)
        target_q_values = reward_batch + self.gamma * max_next_q_values * (1 - terminated_batch)

        # Update network weights using Q-decomposition
        loss = self.network.q_decomposition_update(
            state_batch,
            [action_batch, arg1_batch, arg2_batch, arg3_batch],
            target_q_values
        )

        print(
            time.strftime('%H:%M:%S'),
            f"[Q-Decomposition Train] Experiences: {len(experiences)}",
            f"Loss: {loss:.5f}",
            f"Avg Reward: {np.mean(reward_batch):.3f}",
            f"Avg Target Q: {np.mean(target_q_values):.3f}",
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
