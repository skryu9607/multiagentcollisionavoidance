import numpy as np

class Params():
    def __init__(self) -> None:
        self.roi_x = np.r_[0, 100]
        self.roi_y = np.r_[0, 100]

        self.num_task = 5       # of total tasks
        self.num_agent = 2      # of agents
        self.pos_agents = [[0, 0], [0, 10]]
        self.num_max_task_in_bundle = 5       # of task for 1 agents
        self.llambda = 0.01 # lambda value


p_ = Params()