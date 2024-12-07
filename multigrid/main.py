import multigrid.envs
from multigrid.envs.hideandseek import HideAndSeekEnv
from multigrid.wrappers import FullyObsWrapper
from multigrid.core.grid import Grid
from multigrid.core.actions import Action
from scripts.utils import generate_dir_tree
import scripts.utils as u

import os

from absl import flags
from absl import app
import wandb

from DeepQ import DQNAgent
from DeepJointQ import DeepJointQNAgent

import numpy as np
import torch

FLAGS = flags.FLAGS

# environment options 
flags.DEFINE_string("render_mode", None, "None for nothing, human for render")
flags.DEFINE_integer("max_steps", 100, "timesteps for one episode")
flags.DEFINE_integer("hide_steps", 75, "timesteps to hide")
flags.DEFINE_integer("seek_steps", 25, "timesteps to seek")

# tining options 
flags.DEFINE_bool("train_independent", True, "train independent deep q")
flags.DEFINE_bool("train_joint", False, "train joint action deep q")
flags.DEFINE_bool("test", False, "test loaded model")

# rl parameter
flags.DEFINE_integer("h", 64, "number of hidden layer neurons")
flags.DEFINE_float("gamma", 0.9, "discount factor for reward")
flags.DEFINE_integer("batch_size", 32, "batch size")
flags.DEFINE_float("learning_rate", 0.0001, "learning rate")
flags.DEFINE_integer("replay_memory_size", 200_000, "number of last steps to keep for training")
flags.DEFINE_integer("minimum_replay_memory_size", 10_000, "minimum number of steps in memory to start training")
flags.DEFINE_integer("update_target_every", 500, "number of episodes to take to train target model")
flags.DEFINE_float("epsilon_decay", 0.9999, "epsilon decay per timestep")
flags.DEFINE_float("minimum_epsilon", 0.1, "minimum epsilon - cannot decay further")
flags.DEFINE_bool("double_network", True, "Use double q network")

# training setting
flags.DEFINE_bool("resume", False, "whether resume from previous checkpoint")
flags.DEFINE_bool("wb_log", True, "use wb to log")
flags.DEFINE_integer("wb_log_interval", 100, "number of episodes to log wb")
flags.DEFINE_integer("torch_seed", 1, "seed for randomness controlling learning")
flags.DEFINE_integer("total_eps", 100000, "total training eps")

# test setting
flags.DEFINE_integer("test_env_seed", 2, "test seed for randomness controlling simulator")
flags.DEFINE_string("test_model_folder", 
                    '',
                    "folder that contains model.pth to test")
flags.DEFINE_bool("learn_independent", True, "learn independent multiseller product")
flags.DEFINE_bool("learn_joint_action", False, "learn joint action multiseller product")
flags.DEFINE_bool("plot_epoch", True, "whether plot agent activity per epoch")

def main(argv): 
    assert FLAGS.hide_steps + FLAGS.seek_steps == FLAGS.max_steps

    u.init_globvar()
    u.MAX_STEPS = FLAGS.max_steps
    u.HIDE_STEPS = FLAGS.hide_steps
    u.SEEK_STEPS = FLAGS.seek_steps
    u.H = FLAGS.h
    u.GAMMA = FLAGS.gamma
    u.BATCH_SIZE = FLAGS.batch_size
    u.LR = FLAGS.learning_rate
    u.REPLAY_MEM_SIZE = FLAGS.replay_memory_size
    u.MIN_REPLAY_MEM_SIZE = FLAGS.minimum_replay_memory_size
    u.UPDATE_TARGET_EVERY = FLAGS.update_target_every
    u.EPSILON_DECAY = FLAGS.epsilon_decay
    u.MIN_EPSILON = FLAGS.minimum_epsilon
    u.DOUBLE_NETWORK  = FLAGS.double_network

    env_config = {
        'render_mode': FLAGS.render_mode,
        'max_steps': FLAGS.max_steps, 
        'hide_steps': FLAGS.hide_steps, 
        'seek_steps': FLAGS.seek_steps
    }

    # save flags/config to folder
    root_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models')

    if FLAGS.test:
        test(env_config)
    
    if FLAGS.train_independent: 
        root_folder = os.path.join(root_folder, 'independent')
        model_dir = generate_dir_tree(root_folder, 'independent_lr_{}_epsilon_decay_{}_batch_{}' + 
                                      '_discount_{}_replay_mem_{}_update_target_{}_episode_len_{}'.format(
                                          u.LR, 
                                          u.EPSILON_DECAY, 
                                          u.BATCH_SIZE, 
                                          u.GAMMA, 
                                          u.REPLAY_MEM_SIZE, 
                                          u.UPDATE_TARGET_EVERY,
                                          u.MAX_STEPS
                                      ))
        train_independent(env_config, model_dir)

    if FLAGS.train_joint: 
        root_folder = os.path.join(root_folder, 'joint')
        model_dir = generate_dir_tree(root_folder, 'joint_lr_{}_epsilon_decay_{}_batch_{}' + 
                                      '_discount_{}_replay_mem_{}_update_target_{}_episode_len_{}'.format(
                                          u.LR, 
                                          u.EPSILON_DECAY, 
                                          u.BATCH_SIZE, 
                                          u.GAMMA, 
                                          u.REPLAY_MEM_SIZE, 
                                          u.UPDATE_TARGET_EVERY,
                                          u.MAX_STEPS
                                      ))
        train_joint(env_config, model_dir)

def train_independent(env_config, model_dir): 
    pass

def train_joint(env_config, model_dir):
    if __name__ == "__main__":
        # Define environment parameters
        grid_size = 7                                                                           # PREDEFINED ENV SIZE
        num_agents = 2
        state_size = grid_size * grid_size * 3 + 8 # 3 channels per grid cell, 1 directions, 1 mission, 2 seeker pos, 2 agent pos, 2 other agent pos 
        action_size = 3                                                                         # PREDEFINED ACTION SIZE
        # Training loop
        episodes = 500                                       # Number of Episodes to do


        # Initialize agents with DeepJointQ
        agents = [
            DeepJointQNAgent(index=i, state_size=state_size, action_size=action_size, num_agents=num_agents, agent_indexes = [j for j in range(num_agents) if j != i], discount=0.99)
            for i in range(num_agents)
        ]
        
        # Create the environment
        non_obs_env = HideAndSeekEnv(grid_size=grid_size, agents=agents, render_mode=env_config['render_mode'])
        full_obs_env = FullyObsWrapper(non_obs_env)
        
        all_total_rewards = [[] for _ in range(num_agents)]  # Separate list for each agent
        all_episodes = []

        for episode in range(episodes):
            # Initialize variables for the episode
            steps = 0
            obs, infos = full_obs_env.reset()
            total_rewards = [0] * num_agents
            terminations = {agent.index: False for agent in agents}

            while not full_obs_env.env.is_done():            # Main training loop                             #HEY LOOK HERE#
                actions = {}
                for agent in agents:
                    if not terminations[agent.index]:
                        agent_image = obs[agent.index]["image"].flatten()  # Flattened image observation

                        agent_direction = np.array([obs[agent.index]["direction"]])   # Direction as an integer

                        agent_mission = np.array([obs[agent.index]["mission"]])

                        seeker_pos = obs[agent.index]["seeker"].flatten()

                        agent_pos = obs[agent.index]["curr_pos"].flatten()
                        other_agent_pos = obs[agent.index]["other_pos"].flatten()

                        # Concatenate the flattened image with the one-hot direction
                        agent_state = np.concatenate([agent_image, agent_direction, agent_mission, seeker_pos, agent_pos, other_agent_pos])
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

                        next_agent_mission = np.array([obs[agent.index]['mission']])

                        next_seeker_pos = obs[agent.index]["seeker"]

                        next_agent_pos = obs[agent.index]["curr_pos"]
                        next_other_agent_pos = obs[agent.index]["other_pos"]

                        # Concatenate the flattened image with the one-hot direction
                        next_state = np.concatenate([next_agent_image, next_agent_direction, next_agent_mission, next_seeker_pos, next_agent_pos, next_other_agent_pos])
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

        # Close the environment
        full_obs_env.close() 

def test(env_config): 
    pass

if __name__ == '__main__':
    app.run(main)