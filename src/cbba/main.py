import numpy as np

from cbba import CBBA
from timer import Timer
from params import p_
from plotter import Plotter

def main():

    np.random.seed(1)
    
    cbba = CBBA()
    plot_handler = Plotter()

    with Timer('CBBA'):
        pos_agents, pos_tasks, p_path = cbba.process(
            p_.num_task,
            p_.num_agent, 
            p_.num_max_task_in_bundle, 
            p_.llambda
        )

    plot_handler.init_plots()
    plot_handler.plot_cbba_result(pos_agents, pos_tasks, p_path)

if __name__ == '__main__':
    main()