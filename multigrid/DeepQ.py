import torch # type: ignore
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional
from collections import deque

from multigrid.core.agent import Agent


class ReplayBuffer:
    def __init__(                                                    # Why was this init and not __init__
        self,
        buffer_size: int,
        state_size: int,
        batch_size: int,
        priority_buffer: bool = False,
        priority_scale: Optional[float] = None,
    ):
        self.buffer_size = buffer_size
        self.state_size = state_size
        self.batch_size = batch_size
        self.priority_buffer = priority_buffer

        if self.priority_buffer:
            self.priority_scale = priority_scale
            self.storage = LazyTensorStorage(self.buffer_size)
            self.replay_buffer = PrioritizedReplayBuffer(                   #IGNORE????????????
                batch_size=self.batch_size,
                alpha=self.priority_scale,
                storage=self.storage,
                beta=1,
                eps=0.1,
            )
        else:
            self.replay_buffer = deque(maxlen=self.buffer_size)

    def __len__(self):                                               # Length of replay buffer
        return len(self.replay_buffer)

    def add(self, experience: tuple):                                # Add to replay buffer

        if self.priority_buffer:                                                    # Priority Buffer Ignored
            experience = TensorDict(                                                    # |
                {                                                                       # |
                    "state": experience[0].view(1, self.state_size),                    # |
                    "action": torch.tensor(experience[1]).view(1, 1),                   # |
                    "reward": torch.tensor(experience[2]).view(1, 1),                   # |
                    "next_state": experience[3].view(1, self.state_size),               # |
                    "done": torch.tensor(experience[4], dtype=float).view(1, 1),        # |
                },                                                                      # |
                batch_size=[1],                                                         # |
            )                                                                           # |
                                                                                        # |
            self.replay_buffer.add(experience)                                      # Priority Buffer Ignored
        else:
            self.replay_buffer.append(experience)                                   # Add Experience

    def sample(self):
        if self.priority_buffer:                                                    # Priority Buffer Ignored
            return self.replay_buffer.sample(self.batch_size, return_info=True)     # Priority Buffer Ignored

        batch = random.sample(self.replay_buffer, self.batch_size)                  # Get Random batch of expirences
        return batch, None

    def set_priorities(self, indices, errors):
        if self.priority_buffer:
            self.replay_buffer.update_priority(indices, errors)


class DeepQ(nn.Module):

    def __init__(
        self,
        num_inputs,
        num_actions,
        discount: Optional[float] = None,
        replay_memory_size: Optional[int] = None,
        minimum_replay_memory_size: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        update_target_every: Optional[int] = None,
        priority_scale: Optional[float] = None,
        priority_buffer=False,
        double_network=False,
    ):

        super(DeepQ, self).__init__()
        self.num_inputs = num_inputs
        self.discount = discount
        self.replay_memory_size = replay_memory_size
        self.minimum_replay_memory_size = minimum_replay_memory_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.update_target_every = update_target_every
        self.double_network = double_network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.main_model = nn.Sequential(                                # Network Architecture
            nn.LayerNorm(num_inputs),
            nn.Linear(num_inputs, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        ).to(self.device)                                               # Network Architecture

        self.target_model = nn.Sequential(                              # Target Model Architecture
            nn.LayerNorm(num_inputs),                                   # Updated less frequently
            nn.Linear(num_inputs, 64),                                  # Used to stabalize model
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        ).to(self.device)                                               # Target Model Architecture

        self.main_model = self.main_model.float()
        self.target_model = self.target_model.float()

        # load main model parameters into target model
        self.target_model.load_state_dict(self.main_model.state_dict())

        if self.learning_rate is not None:
            self.main_optimizer = optim.Adam(                           # Initializing Adam Optimizer
                self.main_model.parameters(), lr=self.learning_rate     # Fetching parameters
            )

        # replay memory
        if self.replay_memory_size is not None:
            self.replay_memory = ReplayBuffer(                          # Set replay memory as Replay Buffer
                buffer_size=self.replay_memory_size,
                state_size=num_inputs,
                batch_size=self.batch_size,
                priority_buffer=priority_buffer,
                priority_scale=priority_scale,
            )

        # used to count when to update target model with main model weights
        self.target_update_counter = 0

    def forward_main(self, input_tensor):
        input_tensor = input_tensor.to(self.device).float()
        return self.main_model(input_tensor)

    def forward_target(self, input_tensor):
        input_tensor = input_tensor.to(self.device).float()
        return self.target_model(input_tensor)


    # update replay memory after every step
    # transition: (observation, action, reward, new observation, done)
    def update_replay_memory(self, transition):
        self.replay_memory.add(transition)

    def train(self, terminal_state):
        # Start training only if enough samples in replay memory
        if len(self.replay_memory) < self.minimum_replay_memory_size:           # Only starts training once enough experiences have been collected
            return None

        # get batch from replay memory
        importance_weights = None
        sample_indices = None
        batch, info = self.replay_memory.sample()       # retrieves sampled experiences, info is for prioritized replay
        if info is not None:                                                                    # Prioritized Replay
            importance_weights = info["_weight"]                                                        # |
            sample_indices = info["index"]
            current_states = (
                batch["state"].reshape(self.batch_size, self.num_inputs).to(self.device)
            )
            actions = batch["action"].reshape(self.batch_size).to(self.device)
            rewards = batch["reward"].reshape(self.batch_size).to(self.device)
            next_states = (
                batch["next_state"]
                .reshape(self.batch_size, self.num_inputs)
                .to(self.device)
            )
            dones = batch["done"].reshape(self.batch_size).to(self.device)                      # Prioritized Replay

        else:
            current_states = torch.stack([torch.tensor(transition[0], dtype=torch.float32) for transition in batch]).to(self.device)

            actions = torch.tensor([transition[1] for transition in batch], dtype=torch.long).to(self.device)
            
            rewards = (
                torch.tensor([transition[2] for transition in batch], dtype=torch.float32)
                .float()
                .to(self.device)
            )
            next_states = torch.stack([torch.tensor(transition[3], dtype=torch.float32) for transition in batch]).to(self.device)
            dones = torch.tensor(
                [float(transition[4]) for transition in batch],
                dtype=torch.float32
            ).to(self.device)

        # get current states from the batch and query into main model
        current_qs_for_action = (
            self.forward_main(current_states).gather(1, actions.unsqueeze(1)).squeeze(1)        # Predict Q-Values for selected actions given states in batch
        )

        # get future states from minibatch and query from target model
        with torch.no_grad():                                                                   # Temporarily disables gradient computation
            if self.double_network:
                # get future states from minibatch and action indices from main model
                main_future_best_qs_action_indices = self.forward_main(                         # Find action with highest Q-Value in Next state
                    next_states
                ).argmax(dim=1, keepdim=True)
                target_q_values = self.forward_target(next_states)                              # Uses target network to find Q-Value of actions found from above
                target_selected_q_values = torch.gather(
                    input=target_q_values,
                    dim=1,
                    index=main_future_best_qs_action_indices,
                ).squeeze()
                targets = rewards + (                                                           # Updates Q-Value using Bellman Equation
                    self.discount * target_selected_q_values * (1 - dones)
                )
            else:
                future_best_qs = self.forward_target(next_states).max(1)[
                    0
                ]  # gets max q value from new state
                targets = rewards + (self.discount * future_best_qs * (1 - dones))

        errors = self.get_errors(                                                               
            online_output=current_qs_for_action, target_output=targets      # current_qs_for_action:  
        )                                                                       # estimated earlier by main network (the agent's current estimat of Q-values)
                                                                            # target_outputs:
        if importance_weights is not None:                                      # new Q-Values from Bellman
            loss = torch.mean((errors * importance_weights) ** 2)
        else:
            loss = torch.mean(errors**2)                                    # Calculates Loss

        self.replay_memory.set_priorities(sample_indices, errors.detach())  # Upate Priority
        self.main_optimizer.zero_grad()
        loss.backward()                                                     # Backpropogation
        self.main_optimizer.step()                                          # Backpropogation

        # Update target network counter every episode
        if terminal_state:                                                  # Counts number of episodes
            self.target_update_counter += 1

        # If counter reaches set value,
        # update target network with weights of main network
        if self.target_update_counter > self.update_target_every:           # Update target network to match the main network
            self.target_model.load_state_dict(self.main_model.state_dict())
            self.target_update_counter = 0

        return loss.detach()

    def get_errors(self, online_output, target_output):
        errors = target_output - online_output
        return errors.float()

    def get_qs(self, state):
        with torch.no_grad():
            state = state.to(self.device)
            return self.main_model(state)
        
class DQNAgent(Agent):
    def __init__(self, index, state_size, action_size, **dqn_params):
        """
        Initialize the DQNAgent with its own DeepQ model and replay buffer.
        """
        super().__init__(index=index)  # Initialize the parent Agent class
        self.state_size = state_size
        self.action_size = action_size
        self.dqn = DeepQ(
            num_inputs=state_size,
            num_actions=action_size,
            discount=dqn_params.get("discount", 0.99),
            replay_memory_size=dqn_params.get("replay_memory_size", 50000),
            minimum_replay_memory_size=dqn_params.get("minimum_replay_memory_size", 1000),
            batch_size=dqn_params.get("batch_size", 64),
            learning_rate=dqn_params.get("learning_rate", 0.001),
            update_target_every=dqn_params.get("update_target_every", 50),
            priority_scale=dqn_params.get("priority_scale", 0.6),
            priority_buffer=dqn_params.get("priority_buffer", False),
            double_network=dqn_params.get("double_network", True),
        )
        self.epsilon = .99  # Exploration rate
        self.epsilon_decay = 0.99
        self.min_epsilon = 0.00

    def select_action(self, state):
        """
        Select an action using epsilon-greedy strategy.
        """
        if np.random.rand() < self.epsilon:
            #print("                         RANDOM")
            return np.random.choice(self.action_size)  # Explore
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)                        # Converts state to tensor
        q_values = self.dqn.get_qs(state_tensor)                                                    # Computes Q-Values using main network
        return torch.argmax(q_values).item()  # Exploit

    def decay_epsilon(self):
        """
        Reduce the exploration rate over time.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
