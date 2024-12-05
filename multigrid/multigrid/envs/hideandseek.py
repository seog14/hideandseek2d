from __future__ import annotations
import numpy as np
import random

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

class HideAndSeekEnv(MultiGridEnv): 

    def __init__(
        self,
        size: int = 15,
        max_steps: int = 100,
        num_agents: int = 2,
        percent_trees: float = .1,
        **kwargs):
        
        self.size = size
        self.max_steps = max_steps
        self.seeker = (size // 2, size // 2)  # starts out in center!
        self.grid_template = gen_env.gen_environment(self.size, percent_trees)
       
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
        
        # Place agents in the center
        for agent in self.agents:
            agent.state.pos = self.seeker
            agent.state.dir = Direction.up
