import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from loguru import logger as logging
from collections import namedtuple
from torch.utils.tensorboard import SummaryWriter
from src.agent.actions import Action
from src.agent.host import StateEncoder


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class BasePolicy():
    def __init__(self, logger: SummaryWriter, use_wandb):
        self.use_wandb = use_wandb
        self.tf_logger = logger
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = StateEncoder.state_space
        self.action_dim = Action.action_space
        

def build_net(input_dim: int,
              output_dim: int,
              hidden_shape: list,
              use_layer_norm=False,
              use_batchnorm=False,
              hid_activation="relu",
              use_orthogonal_init=False,
              output_activation=nn.Identity()):
    '''build net with for loop'''
    if hid_activation == "relu":
        hid_activation_ = nn.ReLU
    elif hid_activation == "leaky_relu":
        hid_activation_ = nn.LeakyReLU
    elif hid_activation == "tanh":
        hid_activation_ = nn.Tanh
    elif hid_activation == "softsign":
        hid_activation_ = nn.Softsign
    elif hid_activation == "tanhshrink":
        hid_activation_ = nn.Tanhshrink
    elif hid_activation == "elu":
        hid_activation_ = nn.ELU
    else:
        logging.error("activate_func error")
        hid_activation_ = nn.ReLU

    layers = []
    input_layer = nn.Linear(input_dim, hidden_shape[0])
    output_layer = nn.Linear(hidden_shape[-1], output_dim)
    hidden_layers = []
    for l in range(len(hidden_shape) - 1):
        hidden_layers.append(nn.Linear(hidden_shape[l], hidden_shape[l + 1]))

    layers_ = [input_layer] + hidden_layers + [output_layer]

    for l in range(len(layers_)):
        layers.append(layers_[l])
        if l < len(layers_) - 1:
            if use_layer_norm:
                layers.append(nn.LayerNorm([hidden_shape[l]]))
            layers.append(hid_activation_())
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_shape[l]))
    layers.append(output_activation)

    return nn.Sequential(*layers)


# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

default_Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class Memory(object):
    def __init__(self, Transition: namedtuple = default_Transition):
        self.memory = []
        self.Transition = Transition

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(self.Transition(*args))

    def sample(self, batch_size=None):
        if batch_size is None:
            return self.Transition(*zip(*self.memory))
        else:
            random_batch = random.sample(self.memory, batch_size)
            return self.Transition(*zip(*random_batch))

    def append(self, new_memory):
        self.memory += new_memory.memory

    def __len__(self):
        return len(self.memory)

    def save(self, file):
        import pickle
        with open(file, 'wb') as f:
            data = pickle.dump(self.memory, f)

    def load(self, file):
        import pickle
        with open(file, 'rb') as f:
            self.memory = pickle.load(f)


class ReplayBuffer_PPO:
    def __init__(self, batch_size, state_dim):
        self.s = np.zeros((batch_size, state_dim))
        self.a = np.zeros((batch_size, 1))
        self.a_logprob = np.zeros((batch_size, 1))
        self.r = np.zeros((batch_size, 1))
        self.s_ = np.zeros((batch_size, state_dim))
        self.dw = np.zeros((batch_size, 1))
        self.done = np.zeros((batch_size, 1))
        self.count = 0

    def store(self, s, a, a_logprob, r, s_, dw, done):
        self.s[self.count] = s
        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1

    def numpy_to_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float)
        # In discrete action space, 'a' needs to be torch.long
        a = torch.tensor(self.a, dtype=torch.long)
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float)
        r = torch.tensor(self.r, dtype=torch.float)
        s_ = torch.tensor(self.s_, dtype=torch.float)
        dw = torch.tensor(self.dw, dtype=torch.float)
        done = torch.tensor(self.done, dtype=torch.float)

        return s, a, a_logprob, r, s_, dw, done


class RunningMeanStd:
    """Dynamically calculate mean and std (Welford's algorithm).

    Supports ``reset()`` and ``warmup()`` for domain-transfer scenarios
    (Strategy C §5.3).
    """

    def __init__(self, shape):  # shape:the dimension of input data
        self.shape = shape
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)

    def reset(self):
        """Reset running statistics for domain transfer (Strategy C §5.3 R3)."""
        self.n = 0
        self.mean = np.zeros(self.shape)
        self.S = np.zeros(self.shape)
        self.std = np.sqrt(self.S)

    def warmup(self, states: np.ndarray):
        """Batch-update statistics from collected states.

        Args:
            states: Array of shape ``(N, dim)`` — e.g. collected during
                    random rollouts on the target domain.
        """
        for s in states:
            self.update(s)


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x.astype(np.float32)

    def reset(self):
        """Reset running statistics for domain transfer."""
        self.running_ms.reset()

    def warmup(self, states: np.ndarray):
        """Batch-update statistics from collected states.

        Args:
            states: Array of shape ``(N, dim)``.
        """
        self.running_ms.warmup(states)


class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)

