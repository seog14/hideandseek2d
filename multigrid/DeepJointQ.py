import torch # type: ignore
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional
from collections import deque

from multigrid.core.agent import Agent

import itertools

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


class DeepQ_Joint_Action(nn.Module):
    def __init__(
        self,
        observation_size,
        num_actions,
        num_sellers,
        seller_index,
        discount: Optional[float] = None,
        replay_memory_size: Optional[int] = None,
        minimum_replay_memory_size: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        update_target_every: Optional[int] = None,
        double_network: Optional[bool] = False,
    ):
        super(DeepQ_Joint_Action, self).__init__()
        self.observation_size = observation_size
        self.num_actions = num_actions
        self.num_sellers = num_sellers
        self.seller_index = seller_index
        self.discount = discount
        self.minimum_replay_memory_size = minimum_replay_memory_size
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.double_network = double_network

        # initialize value function network
        value_network_num_inputs = self.observation_size + self.num_sellers - 1
        self.value_network = nn.Sequential(
            nn.LayerNorm(value_network_num_inputs),
            nn.Linear(value_network_num_inputs, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions),
        )
        # initialize policy prediction network
        policy_prediction_network_num_inputs = self.observation_size
        policy_prediction_network_num_outputs = self.num_actions * (
            self.num_sellers - 1
        )
        self.policy_prediction_network = nn.Sequential(
            nn.LayerNorm(policy_prediction_network_num_inputs),
            nn.Linear(policy_prediction_network_num_inputs, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, policy_prediction_network_num_outputs),
        )
        self.policy_prediction_criterion = nn.CrossEntropyLoss()

        # initialize target network
        target_network_num_inputs = value_network_num_inputs
        self.target_network = nn.Sequential(
            nn.LayerNorm(target_network_num_inputs),
            nn.Linear(target_network_num_inputs, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions),
        )

        # load value network parameters into target network
        self.target_network.load_state_dict(self.value_network.state_dict())

        if learning_rate is not None:
            self.value_network_optimizer = optim.Adam(
                self.value_network.parameters(), lr=learning_rate
            )
            self.policy_prediction_network_optimizer = optim.Adam(
                self.policy_prediction_network.parameters(), lr=learning_rate
            )

        # initialize replay memory
        if replay_memory_size is not None:
            self.replay_memory = ReplayBuffer(
                buffer_size=replay_memory_size,
                state_size=observation_size,
                batch_size=batch_size,
                priority_buffer=False,
            )

        # set batch size to 1 if None (for multiprocessing)
        if self.batch_size is None:
            self.batch_size = 1

        # used to count when to update target network with value network weights
        self.target_update_counter = 0

    def forward_value_network(self, input_tensor):
        return self.value_network(input_tensor)

    def forward_policy_prediction_network_training(self, input_tensor):
        output = self.policy_prediction_network(input_tensor)
        return output.view(-1, self.num_sellers - 1, self.num_actions)

    def forward_policy_prediction_network(self, input_tensor):
        output = self.policy_prediction_network(input_tensor)
        output = output.view(-1, self.num_sellers - 1, self.num_actions)
        output = nn.Softmax(dim=2)(output)
        return output.view(-1, self.num_actions * (self.num_sellers - 1))

    def forward_value_network_for_double(self, next_states):
        with torch.no_grad():
            # return shape should be (batch_size, num_actions)
            # for every joint action
            # find the probability of that happening
            # run the next_states + joint_action through target network
            # Multiply the values by the probability of thtat happening
            # add that to the final return
            expected_target_values = torch.zeros(
                size=(self.batch_size, self.num_actions)
            )
            predicted_policies = self.forward_policy_prediction_network(next_states)

            # generate all possible joint-actions
            joint_actions_iterator = itertools.product(
                range(self.num_actions), repeat=self.num_sellers - 1
            )
            for joint_action in joint_actions_iterator:
                joint_action_tensor = torch.tensor(joint_action)
                joint_action_tensor_expanded = joint_action_tensor.expand(
                    self.batch_size, self.num_sellers - 1
                )
                # for multiprocessing
                if self.batch_size == 1:
                    dim = 0
                    joint_action_tensor_expanded = joint_action_tensor_expanded.view(-1)
                else:
                    dim = 1
                target_input = torch.cat(
                    (next_states, joint_action_tensor_expanded), dim=dim
                )
                joint_action_target_values = self.value_network(target_input)
                joint_action_probability = torch.ones(size=(self.batch_size,))
                for seller_index, action in enumerate(joint_action):
                    joint_action_probability *= predicted_policies[
                        :, action + self.num_actions * seller_index
                    ]
                expected_target_values += (
                    joint_action_target_values * joint_action_probability.view(-1, 1)
                )

            return expected_target_values

    def forward_target_network(self, next_states):
        with torch.no_grad():
            # return shape should be (batch_size, num_actions)
            # for every joint action
            # find the probability of that happening
            # run the next_states + joint_action through target network
            # Multiply the values by the probability of thtat happening
            # add that to the final return
            expected_target_values = torch.zeros(
                size=(self.batch_size, self.num_actions)
            )
            predicted_policies = self.forward_policy_prediction_network(next_states)

            # generate all possible joint-actions
            joint_actions_iterator = itertools.product(
                range(self.num_actions), repeat=self.num_sellers - 1
            )
            for joint_action in joint_actions_iterator:
                joint_action_tensor = torch.tensor(joint_action)
                joint_action_tensor_expanded = joint_action_tensor.expand(
                    self.batch_size, self.num_sellers - 1
                )
                # for multiprocessing
                if self.batch_size == 1:
                    dim = 0
                    joint_action_tensor_expanded = joint_action_tensor_expanded.view(-1)
                else:
                    dim = 1
                target_input = torch.cat(
                    (next_states, joint_action_tensor_expanded), dim=dim
                )
                joint_action_target_values = self.target_network(target_input)
                joint_action_probability = torch.ones(size=(self.batch_size,))
                for seller_index, action in enumerate(joint_action):
                    joint_action_probability *= predicted_policies[
                        :, action + self.num_actions * seller_index
                    ]
                expected_target_values += (
                    joint_action_target_values * joint_action_probability.view(-1, 1)
                )

            return expected_target_values

    # update replay memory after every step
    # transition: (observation, action, other_action, reward, new observation, done)
    def update_replay_memory(self, transition):
        self.replay_memory.add(transition)

    def train(self, terminal_state):
        # Start training only if enough samples in replay memory
        if len(self.replay_memory) < self.minimum_replay_memory_size:
            return None, None
        # get batch from replay memory
        batch, _ = self.replay_memory.sample()

        current_states = torch.stack([transition[0] for transition in batch])
        actions = torch.tensor([transition[1] for transition in batch])
        other_actions = torch.stack([transition[2] for transition in batch]).to(
            torch.int64
        )  # added as tensor
        rewards = torch.tensor([transition[3] for transition in batch]).float()
        next_states = torch.stack([transition[4] for transition in batch])
        dones = torch.tensor([transition[5] for transition in batch]).float()

        # concatenate current_state as well as actions taken by the other agents
        value_network_input = torch.cat((current_states, other_actions), dim=1)
        # get current states from the batch and query into value network
        current_values_for_action = (
            self.forward_value_network(value_network_input)
            .gather(1, actions.unsqueeze(1))
            .squeeze(1)
        )

        # get future states from minibatch and query from target network
        with torch.no_grad():
            if self.double_network:
                value_future_best_action_value_indices = (
                    self.forward_value_network_for_double(next_states).argmax(
                        dim=1, keepdim=True
                    )
                )
                target_expected_action_values = self.forward_target_network(next_states)
                target_selected_action_values = torch.gather(
                    input=target_expected_action_values,
                    dim=1,
                    index=value_future_best_action_value_indices,
                ).squeeze()
                targets = rewards + (
                    self.discount * target_selected_action_values * (1 - dones)
                )
            else:
                future_best_qs = self.forward_target_network(next_states).max(1)[
                    0
                ]  # gets max q value from new state
                targets = rewards + (self.discount * future_best_qs * (1 - dones))

        # Compute loss for value network
        errors = self.get_errors(
            online_output=current_values_for_action, target_output=targets
        )
        value_network_loss = torch.mean(errors**2)

        self.value_network_optimizer.zero_grad()
        value_network_loss.backward()
        self.value_network_optimizer.step()

        # Compute loss for policy prediction network

        # get predictions from current_state
        other_action_predictions = self.forward_policy_prediction_network_training(
            current_states
        )
        # initialize loss
        policy_prediction_network_loss = torch.tensor(0.0, dtype=torch.float32)
        for seller_index in range(self.num_sellers - 1):
            other_actions_by_seller = other_actions[:, seller_index]
            policy_prediction_network_loss += self.policy_prediction_criterion(
                other_action_predictions[:, seller_index], other_actions_by_seller
            )

        self.policy_prediction_network_optimizer.zero_grad()
        policy_prediction_network_loss.backward()
        self.policy_prediction_network_optimizer.step()

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value,
        #  update target network with weights of main network
        if self.target_update_counter > self.update_target_every:
            self.target_network.load_state_dict(self.value_network.state_dict())
            self.target_update_counter = 0

        return value_network_loss.detach(), policy_prediction_network_loss.detach()

    def get_errors(self, online_output, target_output):
        errors = target_output - online_output
        return errors.float()

    def get_qs(self, state):
        with torch.no_grad():
            expected_q_values = torch.zeros(size=(self.num_actions,))
            predicted_policies = self.forward_policy_prediction_network(state).view(-1)
            # generate all possible joint-actions
            joint_actions_iterator = itertools.product(
                range(self.num_actions), repeat=self.num_sellers - 1
            )
            for joint_action in joint_actions_iterator:
                joint_action_tensor = torch.tensor(joint_action).unsqueeze(0)                       # Will Change
                value_input = torch.cat((state, joint_action_tensor), dim=1)
                joint_action_values = self.value_network(value_input).squeeze()                     # Will Change
                joint_action_probability = torch.ones(size=(1,))
                for seller_index, action in enumerate(joint_action):
                    joint_action_probability *= predicted_policies[
                        action + self.num_actions * seller_index
                    ]
                expected_q_values += joint_action_values * joint_action_probability.squeeze()       # Will Change

            return expected_q_values
        

class DeepJointQNAgent(Agent):
    def __init__(self, index, state_size, action_size, num_agents, agent_indexes, epsilon_decay,**dqn_params):
        """
        Initialize the DQNAgent with its own DeepQ model and replay buffer.
        """
        super().__init__(index=index)  # Initialize the parent Agent class
        self.state_size = state_size
        self.action_size = action_size
        self.dqn = DeepQ_Joint_Action(
            observation_size=state_size,
            num_actions=action_size,
            discount=dqn_params.get("discount", 0.99),
            replay_memory_size=dqn_params.get("replay_memory_size", 50000),
            minimum_replay_memory_size=dqn_params.get("minimum_replay_memory_size", 1000),
            batch_size=dqn_params.get("batch_size", 64),
            learning_rate=dqn_params.get("learning_rate", 0.001),
            update_target_every=dqn_params.get("update_target_every", 50),
            double_network=dqn_params.get("double_network", True),
            num_sellers= num_agents,
            seller_index = agent_indexes,
        )
        self.epsilon = .99  # Exploration rate
        self.epsilon_decay = epsilon_decay
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
