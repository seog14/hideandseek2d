from multigrid.base import MultiGridEnv
from multigrid.core.grid import Grid
from multigrid.core.world_object import Wall, Goal

from multigrid.wrappers import FullyObsWrapper

import torch # type: ignore
import numpy as np

import matplotlib.pyplot as plt # type: ignore


from DeepQ import DQNAgent
from DeepJointQ import DeepJointQNAgent

class CustomEnv(MultiGridEnv):

    def __init__(self, render_mode="human", **kwargs):
        self,
        goal_pos: int
        super().__init__(render_mode=render_mode, success_termination_mode=all, **kwargs)

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
        self.goal_pos = goal_pos

        # Place agents at the top-left corner
        for agent in self.agents:
            self.place_agent(agent, top=(1, 1), size=(2, 2))
    
    def _reward(self) -> float:
        return 1 - 0.9 * (self.step_count / self.max_steps)
    
    def step(self, actions):
        obs, rewards, terminated, truncated, info = super().step(actions)
        for agent in agents:
            if terminations[agent.index] == True:
                rewards[agent.index] = 2*grid_size+10
            else:
                rewards[agent.index] = -1
        return obs, rewards, terminated, truncated, info


def one_hot_encode_direction(direction, num_directions=4):
    one_hot = np.zeros(num_directions)
    one_hot[direction] = 1
    return one_hot

discount = 0.99
replay_memory_size = 50000
minimum_replay_memory_size = 1000
batch_size = 64
learning_rate = 0.001
update_target_every = 500
priority_scale = 0.6
priority_buffer = False
double_network = True

if __name__ == "__main__":
    import time
    print("Hi")
    # Define environment parameters
    grid_size = 5
    num_agents = 2
    state_size = grid_size * grid_size * 3 + 2 # 3 channels per grid cell, 1 direction and 1 mission
    action_size = 3                                                                         # Using predefined actions here
    max_eval_steps = 50                        # Maximum steps for evaluation episodes

    # Initialize agents with DeepQ
    # agents = [
    #     DQNAgent(index=i, state_size=state_size, action_size=action_size, discount=0.99)
    #     for i in range(num_agents)
    # ]

    # Initialize agents with DeepJointQ
    agents = [
        DeepJointQNAgent(index=i, state_size=state_size, action_size=action_size, num_agents=num_agents, agent_indexes = [j for j in range(num_agents) if j != i], discount=0.99)
        for i in range(num_agents)
    ]
    
    # Create the environment
    non_obs_env = CustomEnv(grid_size=grid_size, agents=agents, render_mode="human")
    full_obs_env = FullyObsWrapper(non_obs_env)

    #print("Grid size: ", full_obs_env.env.height, full_obs_env.env.width)
    #print("Encoded grid shape: ", full_obs_env.env.grid.encode().shape)


    # Training loop
    episodes = 500                                       # Number of Episodes to do
    
    all_total_rewards = [[] for _ in range(num_agents)]  # Separate list for each agent
    all_episodes = []
    evaluation_rewards = [[] for _ in range(num_agents)]

    for episode in range(episodes):
        # Initialize variables for the episode
        steps = 0
        obs, infos = full_obs_env.reset()
        total_rewards = [0] * num_agents
        terminations = {agent.index: False for agent in agents}

        while not all(terminations.values()):            # Main training loop
            actions = {}
            for agent in agents:
                if not terminations[agent.index]:
                    agent_image = obs[agent.index]["image"].flatten()  # Flattened image observation

                    agent_direction = np.array([obs[agent.index]["direction"]])   # Direction as an integer

                    agent_mission = np.array([obs[agent.index]["mission"]])
                    # Concatenate the flattened image with the one-hot direction
                    agent_state = np.concatenate([agent_image, agent_direction, agent_mission])
                    actions[agent.index] = agent.select_action(agent_state)  # Action selection
                else:
                    actions[agent.index] = None

            # Step the environment
            obs, rewards, terminations, truncations, infos = full_obs_env.step(actions)
            steps += 1
            # Update replay memory and train each agent
            for agent in agents:
                if not terminations[agent.index]:
                    next_agent_image = obs[agent.index]["image"].flatten()  # Flattened image observation
                    next_agent_direction = np.array([obs[agent.index]["direction"]])   # Direction as an integer

                    next_agent_mission = np.array([obs[agent.index]["mission"]])

                    # Concatenate the flattened image with the one-hot direction
                    next_state = np.concatenate([next_agent_image, next_agent_direction, next_agent_mission])
                    agent.dqn.update_replay_memory(
                        (agent_state, actions[agent.index], rewards[agent.index], next_state, terminations[agent.index])
                    )
                    loss = agent.dqn.train(terminal_state=torch.tensor(terminations[agent.index], dtype=torch.float32).unsqueeze(0))
                    total_rewards[agent.index] += rewards[agent.index]
            # Render environment (optional)
            full_obs_env.render()

        # Decay exploration rate for each agent
        for agent in agents:
            agent.decay_epsilon()
            
        # Print training episode results
        print(f"Episode {episode + 1}: Total Rewards = {total_rewards} Epsilon = {agent.epsilon}")
            
        for i, reward in enumerate(total_rewards):
            all_total_rewards[i].append(reward)

        all_episodes.append(episode + 1)

        #Evaluation run every 10 episodes
        # if (episode + 1) % 10 == 0:
        #     for agent in agents:
        #         agent.epsilon = 0  # Set epsilon to 0 for pure exploitation
            
        #     eval_rewards = []
        #     eval_steps = 0
        #     obs, infos = env.reset()
        #     terminations = {agent.index: False for agent in agents}
        #     while not all(terminations.values()) and eval_steps < max_eval_steps:
        #         actions = {}
        #         for agent in agents:
        #             if not terminations[agent.index]:
        #                 agent_image = obs[agent.index]["image"].flatten()  # Flattened image observation
        #                 agent_direction = obs[agent.index]["direction"]   # Direction as an integer
        #                 one_hot_direction = one_hot_encode_direction(agent_direction)  # One-hot encode the direction

        #                 # Concatenate the flattened image with the one-hot direction
        #                 agent_state = np.concatenate([agent_image, one_hot_direction])
        #                 actions[agent.index] = agent.select_action(agent_state)
        #             else:
        #                 actions[agent.index] = None

        #         obs, rewards, terminations, truncations, infos = env.step(actions)
        #         eval_steps += 1
        #         for i, reward in rewards.items():
        #             eval_rewards[i] += reward

            # # Append evaluation rewards
            # for i in range(num_agents):
            #     evaluation_rewards[i].append(eval_rewards[i])

            # print(f"Evaluation after Episode {episode + 1}: Total Rewards = {eval_rewards}, Steps = {eval_steps}")

        # Plot training rewards
        plt.figure(figsize=(10, 6))
        for i in range(num_agents):
            plt.plot(all_episodes, all_total_rewards[i], label=f"Agent {i} Training Rewards")
            if len(evaluation_rewards[i]) > 0:
                eval_episodes = list(range(10, len(evaluation_rewards[i]) * 10 + 1, 10))
                plt.plot(eval_episodes, evaluation_rewards[i], linestyle='--', label=f"Agent {i} Evaluation Rewards")
        
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Reward vs Episode for Each Agent")
        plt.legend()
        plt.grid()
        plt.savefig("reward_vs_episode_agents.png")

    # Close the environment
    full_obs_env.close() 






