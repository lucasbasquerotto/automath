import typing
import torch

INPUT_DIM: int = 8
FEATURE_DIM: int = 2**8
HIDDEN_DIM: int = 2**7
HIDDEN_AMOUNT: int = 9
LEARNING_RATE: float = 0.001
GAMMA: float = 0.99
EPSILON_START: float = 0.9
EPSILON_END: float = 0.005
EPSILON_DECAY: float = 0.99
REPLAY_BUFFER_CAPACITY: int = 10000
BATCH_SIZE: int = 64
TARGET_UPDATE_FREQUENCY: int = 100
EPISODES_PER_STATE: int = 1
MAX_STEPS_PER_EPISODE: int = 100
SAVE_INTERVAL: int = 1
MODEL_PATH: str = "tmp/trained_model.pt"
DEVICE: typing.Optional[torch.device] = None
