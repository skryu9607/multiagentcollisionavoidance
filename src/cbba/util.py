import numpy as np

class Util():
    def __init__(self) -> None:
        pass

    def calc_dist(self, pos_0, pos_1):
        return np.linalg.norm(pos_1 - pos_0)