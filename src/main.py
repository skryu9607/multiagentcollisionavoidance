import numpy as np

from mission_planner.cbba import CBBA
from common.timer import Timer
from simulator.params import p_
from mission_planner.plotter import Plotter

def main():

    np.random.seed(10)
    
    cbba = CBBA()
    plot_handler = Plotter()    

    with Timer('CBBA'):

        pos_tasks = cbba.generate_points(20, p_.roi_x, p_.roi_y)  # task position setting
        pos_agents = cbba.generate_points(4, p_.roi_x, p_.roi_y)  # agent position setting

        pos_agents, pos_tasks, p_path = cbba.core_bidding(
            pos_tasks, 
            pos_agents, 
            p_.num_max_task_in_bundle, 
            p_.llambda
        )

    plot_handler.init_plots()
    plot_handler.plot_cbba_result(pos_agents, pos_tasks, p_path, p_.roi_x, p_.roi_y)
    xx = 9


if __name__ == '__main__':
    main()