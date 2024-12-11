import multigrid.envs
from multigrid.envs.hideandseek import HideAndSeekEnv
from multigrid.wrappers import FullyObsWrapper
from multigrid.core.grid import Grid
from multigrid.core.actions import Action
from scripts.utils import generate_dir_tree
import scripts.utils as u

import os
import time
import re

from absl import flags
from absl import app
import wandb

from DeepQ import DQNAgent, DQNAgentConv
from DeepJointQ import DeepJointQNAgent

import numpy as np
import torch

FLAGS = flags.FLAGS

# environment options 
flags.DEFINE_string("render_mode", 'human', "None for nothing, human for render")
flags.DEFINE_integer("max_steps", 100, "timesteps for one episode")
flags.DEFINE_integer("hide_steps", 75, "timesteps to hide")
flags.DEFINE_integer("seek_steps", 25, "timesteps to seek")
flags.DEFINE_integer("num_hiders", 2, "number of hiders")
flags.DEFINE_integer("grid_size", 15, "grid width")

# tining options 
flags.DEFINE_bool("train_independent", True, "train independent deep q")
flags.DEFINE_bool("train_joint", False, "train joint action deep q")
flags.DEFINE_bool("test", False, "test loaded model")

# rl parameter
flags.DEFINE_integer("h", 64, "number of hidden layer neurons")
flags.DEFINE_float("gamma", 1, "discount factor for reward")
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
flags.DEFINE_bool("wb_log", False, "use wb to log")
flags.DEFINE_integer("wb_log_interval", 100, "number of episodes to log wb")
flags.DEFINE_integer("torch_seed", 1, "seed for randomness controlling learning")
flags.DEFINE_integer("total_eps", 100000, "total training eps")

# test setting
flags.DEFINE_integer("test_env_seed", 2, "test seed for randomness controlling simulator")
flags.DEFINE_string("test_model_folder", 
                    'models/independent/12_10_16_22_37_independent_lr_{}_epsilon_decay_{}_batch_{}_discount_0.0001_replay_mem_0.9999_update_target_32_episode_len_0.9/checkpoint_agent_ep_4200_32',
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
    # set up device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set up env 
    env = HideAndSeekEnv(**env_config)
    full_obs_env = FullyObsWrapper(env)

    obs, _ = full_obs_env.reset()

    grid_state, extra_state = u.preprocess_agent_observations(obs)[0]
    
    extra_state_size = len(extra_state)
    
    # set up actions 
    valid_actions = [Action.left, Action.right, Action.forward, Action.none]
    num_actions = len(valid_actions)

    # Initialize agents with DeepQ
    agent_models = {}
    for i in range(FLAGS.num_hiders):
        agent_models[i] = DQNAgentConv(index=i, 
                                       grid_size=FLAGS.grid_size, 
                                       extra_state_size=extra_state_size, 
                                       num_hidden_layers=FLAGS.h, 
                                       action_size=num_actions,
                                       minimum_epsilon=u.MIN_EPSILON ,  
                                       epsilon_decay = u.EPSILON_DECAY,
                                       batch_size = u.BATCH_SIZE, 
                                       update_target_every = u.UPDATE_TARGET_EVERY, 
                                       discount=u.GAMMA, 
                                       replay_memory_size = u.REPLAY_MEM_SIZE)
        agent_models[i].dqn.to(device)

    if FLAGS.wb_log:
        wandb.init(name=("independent_num_hidden_{}_lr_{}_epsilon_decay_{}_batch_{}" +
                        "_discount_{}_replay_mem_{}_update_target_{}_episode_len_{}").format(
                        FLAGS.h, 
                        FLAGS.learning_rate, 
                        FLAGS.epsilon_decay, 
                        FLAGS.batch_size, 
                        FLAGS.gamma, 
                        FLAGS.replay_memory_size, 
                        FLAGS.update_target_every, 
                        FLAGS.max_steps
                        ), project='hideandseek2d')
        wandb.define_metric("episodes")
        wandb.define_metric("*", step_metric="episodes")
        for agent_id, deep_q_model in agent_models.items():
            wandb.watch(deep_q_model.dqn, idx=agent_id+1)

    epsilon = 1 
    episode_count = 0
    agent_episode_rewards = []
    episode_losses = []
    start_time = time.time()

    # main loop for model inference and update
    while episode_count < FLAGS.total_eps:
        # restart episode 
        agent_episode_reward = np.zeros(FLAGS.num_hiders)
        episode_loss = np.zeros(FLAGS.num_hiders)
        
        #reset environment and get initial state for seller
        obs, _ = full_obs_env.reset()

        current_states_dict = u.preprocess_agent_observations(obs)

        actions = {}
        first_terminated = {i: False for i in range(FLAGS.num_hiders)}

        #Start iterating until episode ends 
        done = False 
        while not done: 
            # set up actions dictionary
            for agent in current_states_dict: 
                action = agent_models[agent].select_action(current_states_dict[agent][0], current_states_dict[agent][1])
                actions[agent] = action

            new_state, rewards_dict, terminated_dict, _, _ = full_obs_env.step(actions)
            
            new_states_dict = u.preprocess_agent_observations(new_state)

            # Update seller and platform episode rewards and replay memory for training
            for agent in rewards_dict: 
                agent_episode_reward[agent] += rewards_dict[agent]

            done = all(terminated_dict.values())

            # Update replay memory and train main network
            for agent in agent_models:
                if not terminated_dict[agent] or not first_terminated[agent]:

                    if terminated_dict[agent]:
                        first_terminated[agent] = True

                    current_grid_state, current_extra_state = current_states_dict[agent]
                    new_grid_state, new_extra_state = new_states_dict[agent]
                    agent_models[agent].dqn.update_replay_memory(((current_grid_state, current_extra_state),
                                                                  actions[agent],
                                                                  rewards_dict[agent], 
                                                                  (new_grid_state, new_extra_state), terminated_dict[agent]))
                    
                loss = agent_models[agent].dqn.train(terminated_dict[agent])
                if loss is not None: 
                    episode_loss[agent] += loss    

            current_grid_state, current_extra_state = new_grid_state, new_extra_state
        
        # Add episode rewards to a list and log stats 
        agent_episode_rewards.append(agent_episode_reward)
        episode_losses.append(episode_loss)
        if FLAGS.wb_log and episode_count % FLAGS.wb_log_interval == 0 and episode_count >= FLAGS.wb_log_interval: 
            log_dict = {
                "Episode average time": (time.time()-start_time)/FLAGS.wb_log_interval,
                "episodes": episode_count
            }

            # add seller rewards and seller model loss to log_dict
            last_interval_agent_episode_rewards =  agent_episode_rewards[-FLAGS.wb_log_interval:]
            last_interval_episode_losses = episode_losses[-FLAGS.wb_log_interval:]
            for agent_index in range(FLAGS.num_hiders):
                agent_rewards = [array[agent_index] for array in last_interval_agent_episode_rewards]
                agent_model_losses = [array[agent_index] for array in last_interval_episode_losses]
                print(np.mean(agent_rewards))
                log_dict[f"Episode average agent {agent_index+1} reward"] = np.mean(agent_rewards)
                log_dict[f"Episode average loss for agent {agent_index+1}"] = (np.mean(agent_model_losses) if 
                    episode_count * FLAGS.max_steps > FLAGS.minimum_replay_memory_size else None)
            wandb.log(log_dict)
            start_time = time.time()

        # save models
        num_save = 2000 // FLAGS.wb_log_interval + 1 
        if episode_count % (num_save * FLAGS.wb_log_interval) == 0 and episode_count > 10 * FLAGS.wb_log_interval:
            for agent in agent_models:
                checkpoint = {'state_dict': agent_models[agent].dqn.main_model.state_dict(), 
                              'optimizer': agent_models[agent].dqn.main_optimizer.state_dict()}
                checkpoint_folder_name = 'checkpoint_agent_ep_{}_{}'.format(episode_count, 
                                                                            FLAGS.batch_size, 
                                                                            FLAGS.learning_rate)
                checkpoint_folder = os.path.join(model_dir, checkpoint_folder_name)
                if not os.path.isdir(checkpoint_folder):
                    os.makedirs(checkpoint_folder)
                torch.save(checkpoint, os.path.join(checkpoint_folder, f'agent_{int(agent)+1}.pth'))

        # Decay epsilon 
        if epsilon > FLAGS.minimum_epsilon:
            epsilon *= FLAGS.epsilon_decay
            epsilon = max(FLAGS.minimum_epsilon, epsilon)

        episode_count+=1
        print(episode_count) if episode_count % 100 == 0 else None


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
            DeepJointQNAgent(index=i, state_size=state_size, action_size=action_size, num_agents=num_agents, 
                             agent_indexes = [j for j in range(num_agents) if j != i], 
                             epsilon_decay = u.EPSILON_DECAY, batch_size = u.BATCH_SIZE, update_target_every = u.UPDATE_TARGET_EVERY, discount=u.GAMMA, replay_memory_size = u.REPLAY_MEMORY_SIZE)
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
    if FLAGS.learn_independent:
        test_independent(env_config)

def test_independent(env_config):
    # set up env 
    env_config['render_mode'] = 'human'
    env = HideAndSeekEnv(**env_config)
    full_obs_env = FullyObsWrapper(env)
    obs, _ = full_obs_env.reset(seed=FLAGS.test_env_seed)

    # set up device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    grid_state, extra_state = u.preprocess_agent_observations(obs)[0]
    
    extra_state_size = len(extra_state)
    
    # set up actions 
    valid_actions = [Action.left, Action.right, Action.forward, Action.none]
    num_actions = len(valid_actions)

    # set up agent models
    agent_models = {}
    for filename in os.listdir(FLAGS.test_model_folder):
        # extract agent number 
        agent_number = re.search(r'agent_(\d+)\.pth', filename)
        if agent_number:
            agent_index = (int(agent_number.group(1)) - 1)
            # construct full file path 
            file_path = os.path.join(FLAGS.test_model_folder, filename)
            checkpoint = torch.load(file_path)
            model_state_dict = checkpoint['state_dict']
            # initial model and load it
            agent_models[agent_index] = DQNAgentConv(index=agent_index, 
                                                     grid_size=FLAGS.grid_size, 
                                                     extra_state_size=extra_state_size, 
                                                     num_hidden_layers=FLAGS.h, 
                                                     action_size=num_actions)
            agent_models[agent_index].dqn.to(device)
            agent_models[agent_index].dqn.main_model.load_state_dict(model_state_dict)

    # initialize epoch 
    done = False 
    actions = {}

    current_states_dict = u.preprocess_agent_observations(obs)

    step = 0 
    while not done: 
        for agent in current_states_dict: 
            action = agent_models[agent].select_action(current_states_dict[agent][0], current_states_dict[agent][1])
            # add to actions dictionary
            actions[agent] = action

        new_state, rewards_dict, terminated_dict, _ , _ = env.step(actions)

        current_states_dict = u.preprocess_agent_observations(new_state)
        done = all(terminated_dict.values())
        step += 1

    full_obs_env.close()

if __name__ == '__main__':
    app.run(main)