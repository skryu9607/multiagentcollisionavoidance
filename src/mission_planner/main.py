import numpy as np

from mission_planner.cbba import CBBA
from mission_planner.params import p_
from common.timer import Timer
from plotter import Plotter

def main():

    np.random.seed(1)
    
    cbba = CBBA()
    plot_handler = Plotter()

    with Timer('CBBA'):
        pos_agents, pos_tasks, p_path = cbba.process(
            p_.num_task,
            p_.num_max_task_in_bundle, 
            p_.llambda
        )

    plot_handler.init_plots()
    plot_handler.plot_cbba_result(pos_agents, pos_tasks, p_path, p_.roi_x, p_.roi_y)

if __name__ == '__main__':
    main()