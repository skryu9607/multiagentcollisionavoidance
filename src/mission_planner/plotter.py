import matplotlib.pyplot as plt
import numpy as np
from mission_planner.params import p_

class Plotter():

    def __init__(self) -> None:
        self.flag_cbba_result = True

    def init_plots(self):
        if self.flag_cbba_result:
            self.fig_cbba_result = plt.figure(figsize=(8,8))
    
    def plot_cbba_result(self, pos_agents, pos_tasks, p_path, roi_x, roi_y):

        num_agent = pos_agents.shape[0]
        num_task = pos_agents.shape[0]
        num_max_task_in_bundle = p_path.shape[1]
        
        route_x = []
        route_y = []
        routes = []
        for i in range(num_agent):
            route = []
            route_x.append([])
            route_y.append([])
            route_x[i].append(pos_agents[i,0])
            route_y[i].append(pos_agents[i,1])
            route.append(pos_agents[i])
            for j in range(num_max_task_in_bundle):
                if p_path[i,j] != -1:
                    route.append(pos_tasks[int(p_path[i,j])])
                    route_x[i].append(pos_tasks[int(p_path[i,j]),0])
                    route_y[i].append(pos_tasks[int(p_path[i,j]),1])

            route = np.array(route)
            routes.append(route)

        ax = self.fig_cbba_result.add_subplot(111, autoscale_on=False, xlim=roi_x, ylim=roi_y)
        ax.set_aspect('equal')
        ax.grid()
        dot, = ax.plot(pos_tasks[:,0], pos_tasks[:,1], 'o', lw=2)
        dot, = ax.plot(pos_agents[0,0], pos_agents[0,1], 'o', lw=2)
        for i in range(num_task):
            time_text = plt.text(pos_tasks[i,0], pos_tasks[i,1], i)

        for i in range(num_agent):
            route = routes[i]
            line = ax.plot(route[:,0],route[:,1],'o-',lw = 2)

        plt.pause(.1)
        print(9)


