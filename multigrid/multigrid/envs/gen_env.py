import numpy as np
import random

def gen_environment(size: int = 15, percent_trees: int = .1): 
    # Semi-randomly place walls 1 = wall 0 = open
    grid_template = np.ones([size, size])
    # seed 420 is lit
    np.random.seed(420)
    random.seed(420)
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
    
    curr_op = op.copy()

    # Open a neighbor for half of dead end cells
    for cell in curr_op:
        if find_open(cell[0], cell[1]) == 1:
            if np.random.randint(1,3) == 2:
                a, b = random.choice(get_closed(cell[0], cell[1]))
                grid_template[a][b] = 0
                op.add((a, b))

    assert (center, center) in op

    # Randomly make percent_trees % of the walls into trees

    for i in range(size): 
        for j in range(size): 
            if (i, j) in op: 
                if(random.random() <= percent_trees): 
                    grid_template[i][j] = 2

    for i in range(size): 
        if (0, i) in op: 
            op.remove((0, i))
        grid_template[0, i] = 1

            
        if (i, 0) in op: 
            op.remove((i, 0))
        grid_template[i, 0] = 1

        if (i, size-1) in op: 
            op.remove((i, size-1))
        grid_template[i][size-1] = 1

        
        if (size-1, i) in op: 
            op.remove((size-1, i))
        grid_template[size-1, i] = 1


    # Choose a hiding spot that is not the starting point
    hiding_spot = random.choice(tuple(op.difference({(center, center)})))
    key = random.choice(tuple(op.difference({(center, center), hiding_spot})))

    grid_template[hiding_spot] = 3
    grid_template[key] = 4

    return grid_template
