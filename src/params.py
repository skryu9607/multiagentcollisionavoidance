import numpy as np

class Params():
    def __init__(self) -> None:
        self.N_agent =50
        self.N_obs = 0
        self.rep_coef = 0.022 # repulsive 반영 정도
        self.dist = 0.01
        # self.shape = "circle"
        self.shape = "random"


params_ = Params()