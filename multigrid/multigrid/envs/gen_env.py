import numpy as np
import random

def gen_environment(size: int 15, percent_trees: int 25) {
    # Semi-randomly place walls 1 = wall 0 = open
    grid_template = np.ones([size, size])


    np.random.seed(324)
    random.seed(324)
    center = size // 2

    grid_template[center][center] = 0

    op = {(center, center)}
    d = size


    def find_open(i, j):
        num_open = 0
        if j < d-1 and grid_template[i][j+1] == 0:  # right
            num_open = num_open + 1
        if j > 0 and grid_template[i][j-1] == 0: # left
            num_open = num_open + 1
        if i > 0 and grid_template[i-1][j] == 0: # up
            num_open = num_open + 1
        if i < d-1 and grid_template[i+1][j] == 0: # down
            num_open = num_open + 1

        return num_open

    def get_closed(i, j):
        list = []
        if j < d-1 and grid_template[i][j+1] == 1:  # right
            list.append((i, j+1))
        if j > 0 and grid_template[i][j-1] == 1: # left
            list.append((i, j-1))
        if i > 0 and grid_template[i-1][j] == 1: # up
            list.append((i-1, j))
        if i < d-1 and grid_template[i+1][j] == 1: # down
            list.append((i+1, j))

        return list


    while(True):
        # Identify all blocked cells that have exactly one open neigbor
        cells_to_open = {}

        for i in range(d):
            for j in range(d):
                if grid_template[i][j] == 1: # cell is blocked
                    if find_open(i, j) == 1:
                        cells_to_open[(i, j)] = 0

        # If there are no blocked cells with exactly 1 open neighbor, exit
        if len(cells_to_open) == 0:
            break

        # Pick one at random, open it
        d1, d2 = random.choice(list(cells_to_open.keys()))
        grid_template[d1][d2] = 0
        op.add((d1, d2))

    # Open a neighbor for half of dead end cells
    for cell in op:
        if find_open(cell[0], cell[1]) == 1:
            if np.random.randint(1,3) == 2:
                a, b = random.choice(get_closed(cell[0], cell[1]))
                grid_template[a][b] = 0
                op.add((a, b))

    assert (center, center) in op

    for i in range(self.size): 
        if (0, i) in op: 
            op.remove((0, i))
            
        if (i, 0) in op: 
            op.remove((i, 0))
        
        if (i, self.size-1) in op: 
            op.remove((i, self.size-1))
        
        if (self.size-1, i) in op: 
            op.remove((self.size-1, i))

    closed = {}
    trees = {}

    # Randomly make percent_trees % of the walls into trees

    for i in range(self.size): 
        for j in range(self.size): 
            if (i, j) not in op: 
                if(random.random() <= percent_trees): 
                    grid_template[i][j] = 2

    # Choose a hiding spot that is not the starting point
    hiding_spot = random.choice(op.difference({(center, center)}))
    key = random.choice(op.difference({(center, center), hiding_spot}))

    grid_template[hiding_spot] = 3
    grid_template[key] = 4


    return grid_template

    

}
