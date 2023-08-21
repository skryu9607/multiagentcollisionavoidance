import numpy as np

class Util():
    def __init__(self) -> None:
        pass
    # 이제 이 부분이 prm으로 바뀔 예정
    def calc_dist(self, pos_0, pos_1):
        return np.linalg.norm(pos_1 - pos_0)
정정
