from multigrid.base import MultiGridEnv
from multigrid.core.agent import Agent
from multigrid.core.grid import Grid
from multigrid.core.world_object import Wall, Goal

import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional
from collections import deque

class CustomEnv(MultiGridEnv):
    actions = ["up", "down", "left", "right", "stay"]  # Define available actions here

    def __init__(self, render_mode="human", **kwargs):
        super().__init__(render_mode=render_mode, **kwargs)

    def _gen_grid(self, width: int, height: int):
        """
        Generate the grid layout and initialize the environment.
        """
        # Initialize the grid with the specified dimensions
        self.grid = Grid(width, height)

        # Place walls around the perimeter of the grid
        for x in range(width):
            self.grid.set(x, 0, Wall())
            self.grid.set(x, height - 1, Wall())
        for y in range(height):
            self.grid.set(0, y, Wall())
            self.grid.set(width - 1, y, Wall())

        # Place a goal object at a specific position
        goal_pos = (width - 2, height - 2)
        self.put_obj(Goal(), *goal_pos)

        # Place agents at the top-left corner
        for agent in self.agents:
            self.place_agent(agent, top=(1, 1), size=(2, 2))



class ReplayBuffer:
    def __init__(                                                                   #Why was this init and not __init__
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

    def __len__(self):
        return len(self.replay_buffer)

    def add(self, experience: tuple):

        if self.priority_buffer:
            experience = TensorDict(
                {
                    "state": experience[0].view(1, self.state_size),
                    "action": torch.tensor(experience[1]).view(1, 1),
                    "reward": torch.tensor(experience[2]).view(1, 1),
                    "next_state": experience[3].view(1, self.state_size),
                    "done": torch.tensor(experience[4], dtype=float).view(1, 1),
                },
                batch_size=[1],
            )

            self.replay_buffer.add(experience)
        else:
            self.replay_buffer.append(experience)

    def sample(self):
        if self.priority_buffer:
            return self.replay_buffer.sample(self.batch_size, return_info=True)

        batch = random.sample(self.replay_buffer, self.batch_size)
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

        self.main_model = nn.Sequential(
            nn.LayerNorm(num_inputs),
            nn.Linear(num_inputs, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        ).to(self.device)

        self.target_model = nn.Sequential(
            nn.LayerNorm(num_inputs),
            nn.Linear(num_inputs, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        ).to(self.device)

        # load main model parameters into target model
        self.target_model.load_state_dict(self.main_model.state_dict())

        if self.learning_rate is not None:
            self.main_optimizer = optim.Adam(
                self.main_model.parameters(), lr=self.learning_rate
            )

        # replay memory
        if self.replay_memory_size is not None:
            self.replay_memory = ReplayBuffer(
                buffer_size=self.replay_memory_size,
                state_size=num_inputs,
                batch_size=self.batch_size,
                priority_buffer=priority_buffer,
                priority_scale=priority_scale,
            )

        # used to count when to update target model with main model weights
        self.target_update_counter = 0

    def forward_main(self, input_tensor):
        input_tensor = input_tensor.to(self.device)
        return self.main_model(input_tensor)

    def forward_target(self, input_tensor):
        input_tensor = input_tensor.to(self.device)
        return self.target_model(input_tensor)

    # update replay memory after every step
    # transition: (observation, action, reward, new observation, done)
    def update_replay_memory(self, transition):
        self.replay_memory.add(transition)

    def train(self, terminal_state):
        # Start training only if enough samples in replay memory
        if len(self.replay_memory) < self.minimum_replay_memory_size:
            return None

        # get batch from replay memory
        importance_weights = None
        sample_indices = None
        batch, info = self.replay_memory.sample()
        if info is not None:
            importance_weights = info["_weight"]
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
            dones = batch["done"].reshape(self.batch_size).to(self.device)

        else:
            current_states = torch.stack([transition[0] for transition in batch]).to(
                self.device
            )
            actions = torch.tensor([transition[1] for transition in batch]).to(
                self.device
            )
            rewards = (
                torch.tensor([transition[2] for transition in batch])
                .float()
                .to(self.device)
            )
            next_states = torch.stack([transition[3] for transition in batch]).to(
                self.device
            )
            dones = (
                torch.tensor([transition[4] for transition in batch])
                .float()
                .to(self.device)
            )

        # get current states from the batch and query into main model
        current_qs_for_action = (
            self.forward_main(current_states).gather(1, actions.unsqueeze(1)).squeeze(1)
        )

        # get future states from minibatch and query from target model
        with torch.no_grad():
            if self.double_network:
                # get future states from minibatch and action indices from main model
                main_future_best_qs_action_indices = self.forward_main(
                    next_states
                ).argmax(dim=1, keepdim=True)
                target_q_values = self.forward_target(next_states)
                target_selected_q_values = torch.gather(
                    input=target_q_values,
                    dim=1,
                    index=main_future_best_qs_action_indices,
                ).squeeze()
                targets = rewards + (
                    self.discount * target_selected_q_values * (1 - dones)
                )
            else:
                future_best_qs = self.forward_target(next_states).max(1)[
                    0
                ]  # gets max q value from new state
                targets = rewards + (self.discount * future_best_qs * (1 - dones))

        errors = self.get_errors(
            online_output=current_qs_for_action, target_output=targets
        )

        if importance_weights is not None:
            loss = torch.mean((errors * importance_weights) ** 2)
        else:
            loss = torch.mean(errors**2)

        self.replay_memory.set_priorities(sample_indices, errors.detach())
        self.main_optimizer.zero_grad()
        loss.backward()
        self.main_optimizer.step()

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value,
        # update target network with weights of main network
        if self.target_update_counter > self.update_target_every:
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
            update_target_every=dqn_params.get("update_target_every", 5),
            priority_scale=dqn_params.get("priority_scale", 0.6),
            priority_buffer=dqn_params.get("priority_buffer", False),
            double_network=dqn_params.get("double_network", True),
        )
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01

    def select_action(self, state):
        """
        Select an action using epsilon-greedy strategy.
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)  # Explore
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = self.dqn.get_qs(state_tensor)
        return torch.argmax(q_values).item()  # Exploit

    def decay_epsilon(self):
        """
        Reduce the exploration rate over time.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


# Define a random policy for the agents
def random_policy(obs):
    return random.choice(list(env.actions))  # Randomly choose an action from available actions

discount = 0.99
replay_memory_size = 50000
minimum_replay_memory_size = 1000
batch_size = 64
learning_rate = 0.001
update_target_every = 5
priority_scale = 0.6
priority_buffer = False
double_network = True



# # Create and test the environment
# if __name__ == "__main__":
#     import time

#     # Create agents
#     agents = [Agent(index=i) for i in range(2)]

#     # Create the environment with rendering enabled
#     env = CustomEnv(grid_size=10, agents=agents, render_mode="human")

#     # Reset the environment
#     obs, infos = env.reset()

#     # Render the initial environment
#     env.render()
#     time.sleep(25)  # Pause to view the initial state

#     # Step through the environment with actions
#     for step in range(5):
#         actions = {agent.index: random_policy(obs[agent.index]) for agent in env.agents}
#         obs, rewards, terminations, truncations, infos = env.step(actions)
#         # Render the environment after each step
#         env.render()
#         time.sleep(1)  # Pause to see each step

#     # Close the environment
#     env.close()

if __name__ == "__main__":
    import time

    # Define environment parameters
    grid_size = 10
    num_agents = 2
    state_size = grid_size * grid_size * 3  # Assuming 3 channels per grid cell
    action_size = len(CustomEnv.actions)  # Use predefined actions here

    # Initialize agents with DeepQ
    agents = [
        DQNAgent(index=i, state_size=state_size, action_size=action_size, discount=0.99)
        for i in range(num_agents)
    ]

    # Create the environment
    env = CustomEnv(grid_size=grid_size, agents=agents, render_mode="human")

    # Reset the environment
    obs, infos = env.reset()

    # Render the initial environment
    env.render()
    time.sleep(1)  # Pause to view the initial state

    # Training loop
    episodes = 100
    for episode in range(episodes):
        obs, infos = env.reset()
        total_rewards = [0] * num_agents
        done = {agent.index: False for agent in agents}

        while not all(done.values()):
            actions = {}
            for agent in agents:
                # Convert observation to 1D array for the agent
                agent_state = obs[agent.index]["image"].flatten()
                actions[agent.index] = agent.select_action(agent_state)

            # Step the environment
            obs, rewards, terminations, truncations, infos = env.step(actions)

            # Update replay memory and train each agent
            for agent in agents:
                agent_state = obs[agent.index]["image"].flatten()
                next_state = obs[agent.index]["image"].flatten()
                agent.dqn.update_replay_memory(
                    (agent_state, actions[agent.index], rewards[agent.index], next_state, terminations[agent.index])
                )
                loss = agent.dqn.train(terminal_state=terminations[agent.index])
                total_rewards[agent.index] += rewards[agent.index]

            # Render environment (optional)
            env.render()
            time.sleep(0.1)

        # Decay exploration rate for each agent
        for agent in agents:
            agent.decay_epsilon()

        print(f"Episode {episode + 1}: Total Rewards = {total_rewards}")

    # Close the environment
    env.close()
