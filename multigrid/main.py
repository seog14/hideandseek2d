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
flags.DEFINE_string("render_mode", None, "None for nothing, human for render")
flags.DEFINE_integer("max_steps", 100, "timesteps for one episode")
flags.DEFINE_integer("hide_steps", 75, "timesteps to hide")
flags.DEFINE_integer("seek_steps", 25, "timesteps to seek")
flags.DEFINE_integer("num_hiders", 2, "number of hiders")
flags.DEFINE_integer("grid_size", 15, "grid width")

# tining options 
flags.DEFINE_bool("train_independent", False, "train independent deep q")    
flags.DEFINE_bool("train_joint", False, "train joint action deep q")
flags.DEFINE_bool("test", False, "test loaded model")
flags.DEFINE_bool("metrics_independent", False, "graph metrics for independent model")
flags.DEFINE_bool("metrics_joint", True, "graph metrics for joint action model")

# rl parameter
flags.DEFINE_integer("h", 64, "number of hidden layer neurons")
flags.DEFINE_float("gamma", 1, "discount factor for reward")
flags.DEFINE_integer("batch_size", 32, "batch size")
flags.DEFINE_float("learning_rate", 0.0001, "learning rate")
flags.DEFINE_integer("replay_memory_size", 200_000, "number of last steps to keep for training")
flags.DEFINE_integer("minimum_replay_memory_size", 10_000, "minimum number of steps in memory to start training")
flags.DEFINE_integer("update_target_every", 500, "number of episodes to take to train target model")
flags.DEFINE_float("epsilon_decay", 0.999995, "epsilon decay per timestep")
flags.DEFINE_float("minimum_epsilon", 0.1, "minimum epsilon - cannot decay further")
flags.DEFINE_bool("double_network", True, "Use double q network")

# training setting
flags.DEFINE_bool("resume", True, "whether resume from previous checkpoint")
flags.DEFINE_string("train_model_folder",
                    'models/joint/12_17_19_47_05_joint_lr_0.0001_epsilon_decay_0.999995_batch_32_discount_1.0_replay_mem_200000_update_target_500_episode_len_100/checkpoint_agent_ep_27000_32',
                    "folder that contains model.pth to test")
flags.DEFINE_bool("wb_log", True, "use wb to log")                          # Sets whether waits and biases is on
flags.DEFINE_integer("wb_log_interval", 250, "number of episodes to log wb")
flags.DEFINE_integer("torch_seed", 1, "seed for randomness controlling learning")
flags.DEFINE_integer("total_eps", 2000000, "total training eps")

# test setting
flags.DEFINE_integer("test_env_seed", 5, "test seed for randomness controlling simulator")
flags.DEFINE_string("test_model_folder", 
                    'models/joint/12_17_20_36_00_joint_lr_0.0001_epsilon_decay_0.999995_batch_32_discount_1.0_replay_mem_200000_update_target_500_episode_len_100/checkpoint_agent_ep_135750_32',
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
        model_dir = generate_dir_tree(root_folder, ('independent_lr_{}_epsilon_decay_{}_batch_{}' + 
                                      '_discount_{}_replay_mem_{}_update_target_{}_episode_len_{}').format(
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
        model_dir = generate_dir_tree(root_folder, ('joint_lr_{}_epsilon_decay_{}_batch_{}' + 
                                      '_discount_{}_replay_mem_{}_update_target_{}_episode_len_{}').format(
                                          u.LR, 
                                          u.EPSILON_DECAY, 
                                          u.BATCH_SIZE, 
                                          u.GAMMA, 
                                          u.REPLAY_MEM_SIZE, 
                                          u.UPDATE_TARGET_EVERY,
                                          u.MAX_STEPS
                                      ))
        train_joint(env_config, model_dir)

    if FLAGS.metrics_independent: 
        metrics_independent(env_config)
    
    if FLAGS.metrics_joint:
        metrics_joint(env_config)

def train_independent(env_config, model_dir): 
    # set up device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set up env 
    env = HideAndSeekEnv(**env_config)
    full_obs_env = FullyObsWrapper(env)

    obs, _ = full_obs_env.reset()

    grid_state, extra_state = u.preprocess_agent_observations(obs, 
                                                              full_obs_env.env.step_count)[0]
    
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

        current_states_dict = u.preprocess_agent_observations(obs, 
                                                              full_obs_env.env.step_count)

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
            
            new_states_dict = u.preprocess_agent_observations(new_state, 
                                                              full_obs_env.env.step_count)

            # Update seller and platform episode rewards and replay memory for training
            for agent in rewards_dict: 
                agent_episode_reward[agent] += rewards_dict[agent]

            done = all(terminated_dict.values())

            # Update replay memory and train main network
            for agent in agent_models:
                loss = None 
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

            current_states_dict = new_states_dict

        # Decay exploration rate for each agent
        for agent in agent_models:
            agent_models[agent].decay_epsilon()
        # print(agent_episode_reward)
        # input()
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
        num_save = 500 // FLAGS.wb_log_interval + 1 
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

        episode_count+=1
        print(episode_count) if episode_count % 100 == 0 else None


def train_joint(env_config, model_dir):
    # set up device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # Extract parameters from env_config or FLAGS
    #grid_size = env_config.get('grid_size', 7)
    #num_agents = env_config.get('num_hiders', 2)

    env = HideAndSeekEnv(**env_config)
    full_obs_env = FullyObsWrapper(env)
    
    obs, _ = full_obs_env.reset()
    initial_state = u.preprocess_agent_observations_as_vector(obs, 
                                                          full_obs_env.env.step_count)[0]
    
    state_size = len(initial_state)
    
    # set up actions 
    valid_actions = [Action.left, Action.right, Action.forward, Action.none]
    num_actions = len(valid_actions)

    # Initialize agents
    agent_models = {}
    for i in range(FLAGS.num_hiders):
        agent_models[i] = DeepJointQNAgent(
            index=i,
            state_size=state_size,
            action_size=num_actions,
            num_agents=FLAGS.num_hiders,
            agent_indexes=[j for j in range(FLAGS.num_hiders) if j != i],
            epsilon_decay=u.EPSILON_DECAY,
            discount=u.GAMMA,
            replay_memory_size=u.REPLAY_MEM_SIZE,
            minimum_replay_memory_size=u.MIN_REPLAY_MEM_SIZE,
            batch_size=u.BATCH_SIZE,
            learning_rate=u.LR,
            update_target_every=u.UPDATE_TARGET_EVERY,
            double_network=True,
        )
        agent_models[i].dqn.to(device)

    # If resuming, load checkpoints
    if FLAGS.resume:
        start_epsilon = 0.87
        for filename in os.listdir(FLAGS.train_model_folder):
            agent_number = re.search(r'agent_(\d+)\.pth', filename)
            if agent_number:
                agent_index = int(agent_number.group(1)) - 1
                file_path = os.path.join(FLAGS.train_model_folder, filename)
                checkpoint = torch.load(file_path)
                value_network_state_dict = checkpoint['value_network_state_dict']
                policy_network_state_dict = checkpoint['policy_network_state_dict']

                # Just load the state into the already defined agent
                agent_models[agent_index].dqn.value_network.load_state_dict(value_network_state_dict)
                agent_models[agent_index].dqn.policy_prediction_network.load_state_dict(policy_network_state_dict)

                if 'optimizer' in checkpoint:
                    agent_models[agent_index].dqn.value_network_optimizer.load_state_dict(checkpoint['optimizer'])
                if 'policy_optimizer' in checkpoint:
                    agent_models[agent_index].dqn.policy_prediction_network_optimizer.load_state_dict(checkpoint['policy_optimizer'])

                agent_models[agent_index].dqn.to(device)
                agent_models[agent_index].set_epsilon(start_epsilon)

    # Initialize wandb logging
    if FLAGS.wb_log:
        if FLAGS.resume:
            wandb.init(
                project='hideandseek2d',
                id='w1k55och',  
                resume='must'
            )
        else:
            wandb.init(
                name=("joint_num_hidden_{}_lr_{}_epsilon_decay_{}_batch_{}" +
                      "_discount_{}_replay_mem_{}_update_target_{}_episode_len_{}").format(
                      FLAGS.h, FLAGS.learning_rate, FLAGS.epsilon_decay, FLAGS.batch_size,
                      FLAGS.gamma, FLAGS.replay_memory_size, FLAGS.update_target_every,
                      FLAGS.max_steps),
                project='hideandseek2d')

        wandb.define_metric("episodes")
        wandb.define_metric("*", step_metric="episodes")
        for agent_id, deep_q_model in agent_models.items():
            wandb.watch(deep_q_model.dqn, idx=agent_id+1)

    episode_count = 27000
    agent_episode_rewards = []
    episode_losses = []
    start_time = time.time()
    # all_total_rewards = [[] for _ in range(FLAGS.num_hiders)]
    # start_time = time.time()

    # main loop for model inference and update
    while episode_count < FLAGS.total_eps:
        #print("Start Episode")
        agent_episode_reward = np.zeros(FLAGS.num_hiders)
        episode_loss = np.zeros(FLAGS.num_hiders)
        
        obs, _ = full_obs_env.reset()
        
        current_states_dict = u.preprocess_agent_observations_as_vector(obs, 
                                                              full_obs_env.env.step_count)
        #total_rewards = [0]*FLAGS.num_hiders
        actions = {}
        first_terminated = {i: False for i in range(FLAGS.num_hiders)}
        #terminations = {agent.index: False for agent in agent_models}

        done = False
        while not done:
            #print("STEP")
            # Select actions for each agent
            for agent in current_states_dict:
                if not first_terminated[agent]:                           # Independent isn't using this
                    action = agent_models[agent].select_action(current_states_dict[agent]) # , current_states_dict[agent][1])
                    actions[agent] = action                                                 # NEED TO CHANGE PREPROCESSING???^^^
                else:
                    actions[agent] = None

            new_obs, rewards_dict, terminated_dict, _, _ = full_obs_env.step(actions)
            new_states_dict = u.preprocess_agent_observations_as_vector(new_obs, 
                                                              full_obs_env.env.step_count)

            # Update seller and platform episode rewards and replay memory for training
            for agent in rewards_dict: 
                agent_episode_reward[agent] += rewards_dict[agent]
                
            done = all(terminated_dict.values())

            # Update replay memory and train
            for agent in agent_models:
                loss = None
                if not terminated_dict[agent] or not first_terminated[agent]:
                    
                    if terminated_dict[agent]:
                        first_terminated[agent] = True

                    current_state = current_states_dict[agent]
                    new_state = new_states_dict[agent]

                    # Construct joint transition:
                    # We need other agent actions (excluding current agent)
                    other_actions = []
                    for other_idx in range(FLAGS.num_hiders):
                        if other_idx != agent:
                            # If the other agent terminated, their action is None,
                            # we can treat it as a no-op or a placeholder.
                            # For joint action framework, let's default to 0 if None:
                            other_actions.append(actions[other_idx] if actions[other_idx] is not None else 0)
                    other_actions = np.array(other_actions, dtype=np.int64)

                    agent_models[agent].dqn.update_replay_memory(
                        (current_state,
                         actions[agent],
                         other_actions,
                         rewards_dict[agent],
                         new_state,
                         terminated_dict[agent])
                    )

                    loss, loss_policy = agent_models[agent].dqn.train(terminal_state=terminated_dict[agent])
                    if loss is not None:
                        episode_loss[agent] += loss
                        
                    #total_rewards[agent.index] += rewards_dict[agent.index]

            current_states_dict = new_states_dict

        # Decay epsilon
        for agent in agent_models:
            agent_models[agent].decay_epsilon()

        # Log results if desired
        #print(f"Episode {episode+1}: Total Rewards = {total_rewards}")

        # for i, reward in enumerate(total_rewards):
        #     all_total_rewards[i].append(reward)
        
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
        num_save = 500 // FLAGS.wb_log_interval + 1 
        if episode_count % (num_save * FLAGS.wb_log_interval) == 0 and episode_count > 10 * FLAGS.wb_log_interval:
            for agent in agent_models:
                checkpoint = {'value_network_state_dict': agent_models[agent].dqn.value_network.state_dict(), 
                              'policy_network_state_dict': agent_models[agent].dqn.policy_prediction_network.state_dict(),
                                'optimizer': agent_models[agent].dqn.value_network_optimizer.state_dict(), 
                                'policy_optimizer': agent_models[agent].dqn.policy_prediction_network_optimizer.state_dict()}
                checkpoint_folder_name = 'checkpoint_agent_ep_{}_{}'.format(episode_count, 
                                                                            FLAGS.batch_size, 
                                                                            FLAGS.learning_rate)
                checkpoint_folder = os.path.join(model_dir, checkpoint_folder_name)
                if not os.path.isdir(checkpoint_folder):
                    os.makedirs(checkpoint_folder)
                torch.save(checkpoint, os.path.join(checkpoint_folder, f'agent_{int(agent)+1}.pth'))

        episode_count+=1
        print(episode_count) if episode_count % 100 == 0 else None

    #full_obs_env.close() 

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

    grid_state, extra_state = u.preprocess_agent_observations(obs, full_obs_env.env.step_count)[0]
    
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

    current_states_dict = u.preprocess_agent_observations(obs, full_obs_env.env.step_count)
    agent_rewards = np.zeros(FLAGS.num_hiders)
    step = 0 
    while not done: 
        for agent in current_states_dict: 
            action = agent_models[agent].select_action(current_states_dict[agent][0], 
                                                       current_states_dict[agent][1], 
                                                       explore=False,
                                                       debug=True)
            # add to actions dictionary
            actions[agent] = action

        new_state, rewards_dict, terminated_dict, _ , _ = full_obs_env.step(actions)

        for agent in rewards_dict: 
            agent_rewards[agent] += rewards_dict[agent]

        if step == 90:
            input()
        current_states_dict = u.preprocess_agent_observations(new_state, full_obs_env.env.step_count)
        done = all(terminated_dict.values())
        step += 1

    for agent in range(FLAGS.num_hiders):
        print(f"agent {agent +1} has reward: {agent_rewards[agent]}")
    full_obs_env.close()

def metrics_independent(env_config): 
    # set up env 
    env_config['render_mode'] = 'human'

    env = HideAndSeekEnv(**env_config)
    full_obs_env = FullyObsWrapper(env)
    obs, _ = full_obs_env.reset(seed=FLAGS.test_env_seed)
    random = 0.2 

    # set up device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    grid_state, extra_state = u.preprocess_agent_observations(obs, full_obs_env.env.step_count)[0]
    
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
            agent_models[agent_index].set_epsilon(random)

    num_simulations = 1

    agent_rewards_list = np.empty(shape=(num_simulations, FLAGS.num_hiders))
    agent_time_hidden_list = np.empty(shape=(num_simulations, FLAGS.num_hiders))
    pressed_plate_list = np.empty(shape=num_simulations)
    chopped_trees_list = np.empty(shape=num_simulations)
    spent_together_list = np.empty(shape=num_simulations)

    for index in range(num_simulations):
        obs, _ = full_obs_env.reset()
        done = False 
        actions = {}

        current_states_dict = u.preprocess_agent_observations(obs, full_obs_env.env.step_count)

        # initialize metrics
        time_hidden = np.zeros(FLAGS.num_hiders)
        pressure_plate = 0 
        chopped_trees = 0 
        spent_together = 0 
        agent_reward = np.zeros(FLAGS.num_hiders)

        while not done: 
            for agent in current_states_dict: 
                action = agent_models[agent].select_action(current_states_dict[agent][0], 
                                             current_states_dict[agent][1], 
                                             explore=False,
                                             )
                actions[agent] = action

            new_state, rewards_dict, done_dict, _, _ = full_obs_env.step(actions)

            # update reward metric
            for agent in rewards_dict: 
                agent_reward[agent] += rewards_dict[agent]

            # update time steps alive metric
            if full_obs_env.env.step_count >= FLAGS.hide_steps:
                for agent in done_dict: 
                    if not done_dict[agent]:
                        time_hidden[agent] += 1 

            
            # check if together
            if full_obs_env.env.agents_together():
                spent_together += 1 

            # check if pressure plate pressed
            pressure_plate = int(full_obs_env.env.plate_pressed())

            # check number of trees chopped
            chopped_trees = full_obs_env.env.num_trees_chopped()

            current_states_dict = u.preprocess_agent_observations(new_state, full_obs_env.env.step_count)
            done = all(done_dict.values())
        
        agent_rewards_list[index] = agent_reward
        agent_time_hidden_list[index] = time_hidden 
        pressed_plate_list[index] = pressure_plate
        chopped_trees_list[index] = chopped_trees
        spent_together_list[index] = spent_together
    
    # Get Metrics 
    mean_agent_rewards = np.mean(agent_rewards_list, axis=0)
    mean_agent_time_hidden = np.mean(agent_time_hidden_list, axis=0)
    mean_pressed_plate = np.mean(pressed_plate_list)
    mean_chopped_trees = np.mean(chopped_trees_list)
    mean_spent_together = np.mean(spent_together)

    print(mean_agent_rewards, mean_agent_time_hidden, mean_pressed_plate, mean_chopped_trees, mean_spent_together)

def metrics_joint(env_config): 
    # set up env 

    env = HideAndSeekEnv(**env_config)
    full_obs_env = FullyObsWrapper(env)
    obs, _ = full_obs_env.reset(seed=FLAGS.test_env_seed)
    random = 0.2 

    # set up device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    initial_state = u.preprocess_agent_observations_as_vector(obs, 
                                                          full_obs_env.env.step_count)[0]    
    state_size = len(initial_state)

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
            value_network_state_dict = checkpoint['value_network_state_dict']
            policy_network_state_dict = checkpoint['policy_network_state_dict']

            # initial model and load it
            agent_models[agent_index] = DeepJointQNAgent(index=agent_index, 
                                                         state_size=state_size,num_hidden_layers=FLAGS.h, action_size=num_actions, 
                                                         num_agents=FLAGS.num_hiders,
                                                         agent_indexes=[j for j in range(FLAGS.num_hiders) if j != agent_index]
                                                         )
            agent_models[agent_index].dqn.value_network.load_state_dict(value_network_state_dict)
            agent_models[agent_index].dqn.policy_prediction_network.load_state_dict(policy_network_state_dict)
    
    num_simulations = 1

    agent_rewards_list = np.empty(shape=(num_simulations, FLAGS.num_hiders))
    agent_time_hidden_list = np.empty(shape=(num_simulations, FLAGS.num_hiders))
    pressed_plate_list = np.empty(shape=num_simulations)
    chopped_trees_list = np.empty(shape=num_simulations)
    spent_together_list = np.empty(shape=num_simulations)

    for index in range(num_simulations):
        obs, _ = full_obs_env.reset(seed=FLAGS.test_env_seed)
        done = False 
        actions = {}

        current_states_dict = u.preprocess_agent_observations_as_vector(obs, full_obs_env.env.step_count)

        # initialize metrics
        time_hidden = np.zeros(FLAGS.num_hiders)
        pressure_plate = 0 
        chopped_trees = 0 
        spent_together = 0 
        agent_reward = np.zeros(FLAGS.num_hiders)

        while not done: 
            for agent in current_states_dict: 
                action = agent_models[agent].select_action(current_states_dict[agent],
                                             explore=False,
                                             )
                actions[agent] = action

            new_state, rewards_dict, done_dict, _, _ = full_obs_env.step(actions)

            # update reward metric
            for agent in rewards_dict: 
                agent_reward[agent] += rewards_dict[agent]

            # update time steps alive metric
            if full_obs_env.env.step_count >= FLAGS.hide_steps:
                for agent in done_dict: 
                    if not done_dict[agent]:
                        time_hidden[agent] += 1 
            
            # check if together
            if full_obs_env.env.agents_together():
                spent_together += 1 

            # check if pressure plate pressed
            pressure_plate = int(full_obs_env.env.plate_pressed())

            # check number of trees chopped
            chopped_trees = full_obs_env.env.num_trees_chopped()

            current_states_dict = u.preprocess_agent_observations_as_vector(new_state, full_obs_env.env.step_count)
            done = all(done_dict.values())
        
        agent_rewards_list[index] = agent_reward
        agent_time_hidden_list[index] = time_hidden 
        pressed_plate_list[index] = pressure_plate
        chopped_trees_list[index] = chopped_trees
        spent_together_list[index] = spent_together
    
    # Get Metrics 
    mean_agent_rewards = np.mean(agent_rewards_list, axis=0)
    mean_agent_time_hidden = np.mean(agent_time_hidden_list, axis=0)
    mean_pressed_plate = np.mean(pressed_plate_list)
    mean_chopped_trees = np.mean(chopped_trees_list)
    mean_spent_together = np.mean(spent_together)

    print(mean_agent_rewards, mean_agent_time_hidden, mean_pressed_plate, mean_chopped_trees, mean_spent_together)

if __name__ == '__main__':
    app.run(main)