import typing
import torch

INPUT_DIM: int = 8
FEATURE_DIM: int = 256
HIDDEN_DIM: int = 128
LEARNING_RATE: float = 0.001
GAMMA: float = 0.99
EPSILON_START: float = 1.0
EPSILON_END: float = 0.05
EPSILON_DECAY: float = 0.99
REPLAY_BUFFER_CAPACITY: int = 10000
BATCH_SIZE: int = 64
TARGET_UPDATE_FREQUENCY: int = 100
EPISODES_PER_STATE: int = 10
MAX_STEPS_PER_EPISODE: int = 10000
SAVE_INTERVAL: int = 100
MODEL_PATH: str = "tmp/trained_model.pt"
DEVICE: typing.Optional[torch.device] = None
