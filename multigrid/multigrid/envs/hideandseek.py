from __future__ import annotations
import numpy as np
import random
import pygame
import pygame.freetype

from itertools import repeat
from collections import defaultdict
from typing import Any, Callable, Iterable, Literal, SupportsFloat

from multigrid.envs import gen_env
from multigrid import MultiGridEnv
from multigrid.core import Action, Grid, MissionSpace
from multigrid.core.constants import Direction
from multigrid.core.constants import *
from multigrid.core.world_object import *

"""
Square Grid 

    - default size 15 (?)
    - defaul max steps 100
    - agents 4 (2 hiders, 2 seekers)
    - agent_view_size = 1 NOTE: seekers should see everything though ...  
"""

AgentID = int


class HideAndSeekEnv(MultiGridEnv): 

    def __init__(
        self,
        size: int = 15,
        max_steps: int = 100,
        num_agents: int = 2,
        percent_trees: float = .1,
        hide_steps: int = 75, 
        seek_steps: int = 25,
        shelter_time: int = 50,
        **kwargs):
        
        self.size = size
        self.max_steps = max_steps
        self.hide_steps = hide_steps
        self.seek_steps = seek_steps
        self.shelter_time = shelter_time
        self.seeker = [np.int_(size // 2), np.int_(size // 2)]  # starts out in center!
        self.grid_template = gen_env.gen_environment(self.size, percent_trees)
        self.trees_chopped = 0 
        super().__init__(
            agents=num_agents,
            agent_view_size=self.size,
            width=size,
            height=size,
            max_steps=max_steps,
            allow_agent_overlap=True,
            joint_reward=False,
            success_termination_mode='any', 
            failure_termination_mode='all',
            **kwargs,
        )


    def _gen_grid(self, width, height):

        # Create empty grid
        self.grid = Grid(self.size, self.size)
        
        for i in range(len(self.grid_template)): 
            for j in range(len(self.grid_template[i])): 
                if self.grid_template[i][j] == 1: 
                    self.grid.set(i, j, Wall())
                elif self.grid_template[i][j] == 2: 
                    self.grid.set(i, j, Tree())
                elif self.grid_template[i][j] == 3: 
                    self.hiding_spot = HidingSpot()
                    self.grid.set(i, j, self.hiding_spot)
                elif self.grid_template[i][j] == 4: 
                    self.pressure_plate = PressurePlate()
                    self.grid.set(i, j, self.pressure_plate)

        # Reset shelter and seeker
        self.shelter_time = 50
        self.seeker = [np.int_(self.size // 2), np.int_(self.size // 2)]  

        # Place agents in the center
        for agent in self.agents:
            agent.state.pos = self.seeker
            agent.state.dir = Direction.up

        self.trees_chopped = 0 



    def gen_obs(self):
        observations = super().gen_obs()
        for i in range(self.num_agents):
            observations[i]['seeker'] = self.seeker
            observations[i]['curr_pos'] = self.agents[i].pos
            observations[i]['other_pos'] = self.agents[1 - i].pos

        return observations
    
    def calc_manhattan_distance_seeker(self, pos): 
        return abs(pos[0] - self.seeker[0]) + abs(pos[1] - self.seeker[1])

    # Handles actions that will give a reward
    def handle_actions(
        self, actions: dict[AgentID, Action]) -> dict[AgentID, SupportsFloat]:
        """
        Handle actions taken by agents.

        Parameters
        ----------
        actions : dict[AgentID, Action]
            Action for each agent acting at this timestep

        Returns
        -------
        rewards : dict[AgentID, SupportsFloat]
            Reward for each agent
        """
        rewards = {agent_index: 0 for agent_index in range(self.num_agents)}

        # Randomize agent action order
        if self.num_agents == 1:
            order = (0,)
        else:
            order = self.np_random.random(size=self.num_agents).argsort()

        # Update agent states, grid states, and reward from actions
        if self.step_count < self.hide_steps: 
            for i in order:
                if i not in actions:
                    continue

                agent, action = self.agents[i], actions[i]

                if agent.state.terminated:
                    continue

                assert action == Action.left or action == Action.right or action == Action.forward or action == Action.none

                # Rotate left
                if action == Action.left:
                    agent.state.dir = (agent.state.dir - 1) % 4

                # Rotate right
                elif action == Action.right:
                    agent.state.dir = (agent.state.dir + 1) % 4

                # Move forward
                elif action == Action.forward:
                    
                    fwd_pos = agent.front_pos
                    fwd_obj = self.grid.get(*fwd_pos)

                    # If object is not overlap-able then continues
                    if fwd_obj is None or fwd_obj.can_overlap():
                        if not self.allow_agent_overlap:
                            agent_present = np.bitwise_and.reduce(
                                self.agent_states.pos == fwd_pos, axis=1).any()
                            if agent_present:
                                continue

                        agent.state.pos = fwd_pos 
                    
                
                elif action == Action.none: 
                    continue

                else:
                    raise ValueError(f"Unknown action: {action}")
        else: 
            for i in order:

                if i not in actions:
                    continue

                agent, action = self.agents[i], actions[i]

                if agent.state.terminated:
                    continue

                assert action == Action.left or action == Action.right or action == Action.forward or action == Action.none


                # Rotate left
                if action == Action.left:
                    agent.state.dir = (agent.state.dir - 1) % 4

                # Rotate right
                elif action == Action.right:
                    agent.state.dir = (agent.state.dir + 1) % 4

                # Move forward
                elif action == Action.forward:
                    
                    fwd_pos = agent.front_pos
                    fwd_obj = self.grid.get(*fwd_pos)
                    
                     # If object is not overlap-able then continues
                    if fwd_obj is None or fwd_obj.can_overlap():
                        if not self.allow_agent_overlap:
                            agent_present = np.bitwise_and.reduce(
                                self.agent_states.pos == fwd_pos, axis=1).any()
                            if agent_present:
                                continue
                
                        agent.state.pos = fwd_pos

                    # If agent steps onto seeker
                    if agent.state.pos == self.seeker:
                        agent.state.terminated = True # terminate this agent only
                        rewards[i] = -1*self.seek_steps
                
                    else: 
                        if self.step_count == self.max_steps: 
                            rewards[i] = self.calc_manhattan_distance_seeker(agent.state.pos)
                            continue

                elif action == Action.none: 
                    continue

                else:
                    raise ValueError(f"Unknown action: {action}")

                if not agent.state.terminated:
                    rewards[i] += 2

        return rewards
    
    def is_adj(self, agent_pos, other_pos): 
        return abs(agent_pos[0] - other_pos[0]) + abs(agent_pos[1] - other_pos[1]) <= 1
    
    # if agent_pos equals seeker, seeker does not move
    def move_seeker(self, agent_pos): 
        if agent_pos == self.hiding_spot.cur_pos and self.shelter_time > 0: 
            if self.adj(agent_pos, self.seeker): 
                return 

        new_pos = None

        # Same x value
        if agent_pos[0] == self.seeker[0]: 
            if self.seeker[1] > agent_pos[1]: 
                new_pos = (self.seeker[0], self.seeker[1] - 1)
            elif self.seeker[1] < agent_pos[1]: 
                new_pos = (self.seeker[0], self.seeker[1] + 1)

        # Same y value
        elif agent_pos[1] == self.seeker[1]: 
            if self.seeker[0] > agent_pos[0]: 
                new_pos = (self.seeker[0] - 1, self.seeker[1])
            elif self.seeker[0] < agent_pos[0]: 
                new_pos = (self.seeker[0] + 1, self.seeker[1])
        
        # Both coords different
        else: 
            # Above
            if agent_pos[1] > self.seeker[1]: 
                new_pos = (self.seeker[0], self.seeker[1]+1)
            # Below
            elif agent_pos[1] < self.seeker[1]: 
                new_pos = (self.seeker[0], self.seeker[1]-1)
        #TODO: don't let seeker enter hididng spot regardless
        # Something weird is happenign here where new_pos is none, too lazy to find out why
        if new_pos is not None:
            self.seeker = new_pos

    def step(self, actions: dict[AgentID, Action]): 
        observations, rewards, terminations, truncations, infos = super().step(actions)

        # Check if both adjacent to a tree 
        if self.agents[0].state.pos == self.agents[1].state.pos: 
            if self.agents[0].state.pos == self.pressure_plate.cur_pos:
                self.pressure_plate.is_pressed(True)
                self.hiding_spot.is_open = True
                rewards[0] += 2
                rewards[1] += 2 

            positions = [(0, -1), (-1, 0), (1, 0), (0, 1)]
            
            # If an adjacent cell is a tree 
            for i in positions: 
                new_pos = tuple(map(sum, zip(i, self.agents[0].state.pos)))
                obj = self.grid.get(new_pos[0], new_pos[1])
                if obj is not None and obj.type == Type.tree and not obj.is_open:  
                    obj.is_open = True
                    self.trees_chopped += 1

        # Seeking Phase
        if self.step_count > self.hide_steps:
            agent1_dist = float('inf')
            agent2_dist = float('inf')

            if not self.agents[0].state.terminated and not (self.agents[0].state.pos == self.hiding_spot.cur_pos and self.shelter_time > 0): 
                agent1_dist = self.calc_manhattan_distance_seeker(self.agents[0].state.pos)
            if not self.agents[1].state.terminated and not (self.agents[1].state.pos == self.hiding_spot.cur_pos and self.shelter_time > 0): 
                agent2_dist = self.calc_manhattan_distance_seeker(self.agents[1].state.pos)

            if self.agents[0].state.terminated and self.agents[1].state.terminated: 
                pass
            elif agent1_dist < agent2_dist: 
                prey = self.agents[0]
                self.move_seeker(prey.state.pos)
            else: 
                prey = self.agents[1]
                self.move_seeker(prey.state.pos)

            if tuple(self.seeker) == self.agents[0].state.pos and not self.agents[0].state.terminated: 
                self.agents[0].state.terminated = True # terminate this agent only
                rewards[0] -= 2 + self.seek_steps

            if tuple(self.seeker) == self.agents[1].state.pos and not self.agents[1].state.terminated: 
                self.agents[1].state.terminated = True # terminate this agent only
                rewards[1] -= 2 + self.seek_steps

        
        if self.hiding_spot.is_open: 
            self.shelter_time -= 1

        terminations = dict(enumerate(self.agent_states.terminated))
        truncated = self.step_count >= self.max_steps
        truncations = dict(enumerate(repeat(truncated, self.num_agents)))

        observations = self.gen_obs()

        if truncated:
            terminations = {agent_id: True for agent_id in truncations}
        return observations, rewards, terminations, truncations, defaultdict(dict)
   
    def render(self):
        """
        Render the environment and modify one cell's color in the subclass.
        """
        """
        Render the environment.
        """
        img = self.get_frame(self.highlight, self.tile_size)

        if self.render_mode == 'human':
            img = np.transpose(img, axes=(1, 0, 2))
            cell_y, cell_x = self.seeker  # Coordinates of the cell
            cell_size = self.tile_size
            start_x = cell_x * cell_size
            end_x = start_x + cell_size
            start_y = cell_y * cell_size
            end_y = start_y + cell_size
            img[start_y:end_y, start_x:end_x] = [255, 0, 0]  # Make the cell red

            screen_size = (
                self.screen_size * min(img.shape[0] / img.shape[1], 1.0),
                self.screen_size * min(img.shape[1] / img.shape[0], 1.0),
            )
            if self.render_size is None:
                self.render_size = img.shape[:2]
            if self.window is None:
                pygame.init()
                pygame.display.init()
                pygame.display.set_caption(f'multigrid - {self.__class__.__name__}')
                self.window = pygame.display.set_mode(screen_size)
            if self.clock is None:
                self.clock = pygame.time.Clock()
            surf = pygame.surfarray.make_surface(img)

            # Create background with mission description
            offset = surf.get_size()[0] * 0.1
            # offset = 32 if self.agent_pov else 64
            bg = pygame.Surface(
                (int(surf.get_size()[0] + offset), int(surf.get_size()[1] + offset))
            )
            bg.convert()
            bg.fill((255, 255, 255))
            bg.blit(surf, (offset / 2, 0))

            bg = pygame.transform.smoothscale(bg, screen_size)

            font_size = 22
            text = str(self.mission)
            font = pygame.freetype.SysFont(pygame.font.get_default_font(), font_size)
            text_rect = font.get_rect(text, size=font_size)
            text_rect.center = bg.get_rect().center
            text_rect.y = bg.get_height() - font_size * 1.5
            font.render_to(bg, text_rect, text, size=font_size)

            self.window.blit(bg, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata['render_fps'])
            pygame.display.flip()

        elif self.render_mode == 'rgb_array':
            return img
    
    def agents_together(self):
        position = None 

        for agent in self.agents: 
            if position is not None: 
                if agent.state.pos != position: 
                    return False 
            else: 
                position = agent.state.pos

        return True

    def plate_pressed(self):
        return self.pressure_plate.is_pressed
    
    def num_trees_chopped(self):
        return self.trees_chopped