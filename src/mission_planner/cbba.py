import numpy as np
import math

from common.util import Util
from mission_planner.params import p_

class CBBA(Util):
    def __init__(self) -> None:
        pass

    def check_bundle(self, B, j):
        """
        Checks if a task (j) is in a bundle (B).

        Args:
            B (list): A list representing a bundle of tasks.
            j (int): The task to check for.

        Returns:
            bool: Returns True if task j is in bundle B, otherwise returns False.
        """
        return (j in B)

    def calc_time(self, Path, Task_Pos, Agent_Pos):
        """
        Calculates the total distance between the Agent position and the tasks in the Path.

        Args:
            Path (list): A list of tasks that the agent has to perform.
            Task_Pos (list): A list of positions of all tasks.
            Agent_Pos (list): The position of the agent.

        Returns:
            float: The total distance from the agent position to all tasks in the Path.
        """
        dist = 0
        for i in range(len(Path)):
            if i == 0:
                dist = self.calc_dist(Agent_Pos, Task_Pos[Path[i]])
            else:
                dist += self.calc_dist(Task_Pos[Path[i - 1]], Task_Pos[Path[i]])
        return dist

    def calc_score(self, path, pos_task, pos_agent):
        """
        Calculates the score for a given path. The score is the total reward minus the total distance.

        Args:
            path (list): A list of tasks that the agent has to perform.
            pos_task (list): A list of positions of all tasks.
            pos_agent (list): The position of the agent.

        Returns:
            float: The total score for the given path.
        """
        dist = 0
        reward = 0
        for i in range(len(path)):
            if i == 0:
                dist = self.calc_dist(pos_agent, pos_task[path[i]])
            else:
                dist += self.calc_dist(pos_task[path[i - 1]], pos_task[path[i]])
            reward += 100
        return reward - dist

    def generate_mat(self, path, new_task, pos):
        """
        Inserts a new task into a given position in the path and returns the new path.

        Args:
            path (list): The original path.
            new_task (int): The new task to insert.
            pos (int): The position to insert the new task.

        Returns:
            list: The new path with the new task inserted at the given position.
        """
        path_new = path[:pos] + [new_task] + path[pos:]
        return path_new

    def calc_max_score(self, paths, pos_tasks, pos_agents, new_task):
        """
        Calculates the maximum score that can be achieved by inserting a new task into the paths.

        Args:
            paths (list): A list of current paths.
            pos_tasks (list): A list of positions of all tasks.
            pos_agents (list): The positions of all agents.
            new_task (int): The new task to insert.

        Returns:
            tuple: A tuple where the first element is the maximum score and the second element is the index of the path with the maximum score.
        """
        scores = [self.calc_score(self.generate_mat(paths, new_task, i), pos_tasks, pos_agents) for i in range(len(paths) + 1)]
        idx_max = np.argmax(scores)
        return max(scores), idx_max
    
    def generate_points(self, n, roi_x, roi_y):
        """
        Generates a 2D array with n rows and 2 columns. 
        Each row represents the x and y coordinates of a task.

        Args:
            n (int): The number of tasks.

        Returns:
            np.ndarray: A 2D array with n rows and 2 columns. 
            Each row represents the x and y coordinates of a task.
        """
        pos_tasks = np.random.random(size=(n, 2))
        pos_tasks[:, 0] = pos_tasks[:, 0] * (roi_x[1] - roi_x[0]) - roi_x[0]
        pos_tasks[:, 1] = pos_tasks[:, 1] * (roi_y[1] - roi_y[0]) - roi_y[0]
        return pos_tasks
    
    def core_bidding(self, pos_tasks, pos_agents, num_lt, llambda):
        num_task = pos_tasks.shape[0]
        num_agent = pos_agents.shape[0]

        y_winning_bid = np.zeros((num_agent, num_task))
        z_winning_agent = -np.ones((num_agent, num_task))
        b_bundle = -np.ones((num_agent, num_lt))

        p_path = -np.ones((num_agent, num_lt))

        HH = np.zeros((num_agent, num_task))
        HC = np.zeros((num_agent, num_task))
        NN = -np.ones((num_agent, num_task))
        JJ = -np.ones(num_agent)
        cost = np.zeros((num_agent, num_task))

        tp1 = 0
        tp2 = 0
        flag_continue_consensus = 1
        iter_consensus = 0
        while flag_continue_consensus:
            # Phase I - Build Bundle, select tasks with highest reward in mission
            flag_continue_phase1 = 1
            iter_consensus = iter_consensus + 1
            iter_phase1 = 0
            while flag_continue_phase1:
                iter_phase1 = iter_phase1 + 1
                if iter_phase1 > 100:
                    flag_continue_phase1 = False
                for idx_a1 in range(num_agent):
                    bundle = []
                    Path = []
                    for idx_task in range(num_lt):
                        if b_bundle[idx_a1,idx_task] != -1:
                            bundle.append(int(b_bundle[idx_a1,idx_task]))
                        if p_path[idx_a1,idx_task] != -1:
                            Path.append(int(p_path[idx_a1,idx_task]))
                    for idx_task in range(num_task):
                        if not self.check_bundle(bundle, idx_task):
                            tp1, tp2 = self.calc_max_score(Path, pos_tasks, pos_agents[idx_a1], idx_task)
                            cost[idx_a1,idx_task] = tp1 - self.calc_score(Path, pos_tasks, pos_agents[idx_a1])
                            cost[idx_a1,idx_task] = cost[idx_a1,idx_task] * math.exp(-llambda * self.calc_time(Path, pos_tasks, pos_agents[idx_a1]))

                for idx_a1 in range(num_agent):
                    bundle = []
                    Path = []
                    for idx_task in range(num_lt):
                        if b_bundle[idx_a1,idx_task] != -1:
                            bundle.append(int(b_bundle[idx_a1,idx_task]))
                        if p_path[idx_a1,idx_task] != -1:
                            Path.append(int(p_path[idx_a1,idx_task]))

                    for idx_task in range(num_task):
                        HH[idx_a1,idx_task] = int(cost[idx_a1,idx_task] > (y_winning_bid[idx_a1,idx_task] + 1e-6))
                        HC[idx_a1,idx_task] = HH[idx_a1,idx_task] * cost[idx_a1,idx_task]

                    JJ[idx_a1] = np.argmax(HC[idx_a1])
                    tp1, tp2 = self.calc_max_score(Path, pos_tasks, pos_agents[idx_a1], int(JJ[idx_a1]))
                    bundle = self.generate_mat(bundle, int(JJ[idx_a1]), len(bundle))
                    Path = self.generate_mat(Path, int(JJ[idx_a1]), tp2)
                    if len(bundle) <= num_lt:
                        if sum(HH[idx_a1]) > 0:
                            for idx_task in range(len(bundle)):
                                b_bundle[idx_a1,idx_task] = bundle[idx_task]
                                # print(Bundle, Path)
                            for idx_task in range(len(Path)):
                                p_path[idx_a1,idx_task] = Path[idx_task]
                            for idx_task in range(num_lt - len(Path)):
                                p_path[idx_a1,idx_task+len(Path)] = -1
                            y_winning_bid[idx_a1,int(JJ[idx_a1])] = cost[idx_a1,int(JJ[idx_a1])]
                            z_winning_agent[idx_a1,int(JJ[idx_a1])] = idx_a1

                flag_n = 0
                for idx_a1 in range(num_agent):
                    if b_bundle[idx_a1,-1] != -1:
                        flag_n = flag_n + 1
                if flag_n == num_agent:
                    flag_continue_phase1 = False
                    # print("------end full---------")
                if sum(sum(HH[0:num_task])) == 0:
                    flag_continue_phase1 = False
                    # print("------end H---------")
                    
            # Phase II - Communication
            print(f'iter - {iter_consensus}')
            for idx_agent, path_agent in enumerate(p_path):
                print(f'agent{idx_agent:2d} - {path_agent}')
            print()

            old_y = y_winning_bid
            old_z = z_winning_agent
            #
            for idx_a1 in range(num_agent):
                for idx_a2 in range(num_agent):
                    if idx_a1 != idx_a2:
                        for idx_task in range(num_task):
                            if old_z[idx_a2,idx_task] == idx_a2:
                                if z_winning_agent[idx_a1,idx_task] == idx_a1:
                                    # print("Entry 1")
                                    if old_y[idx_a2,idx_task] - y_winning_bid[idx_a1,idx_task] > 10e-6:
                                        z_winning_agent[idx_a1,idx_task] = old_z[idx_a2,idx_task]
                                        y_winning_bid[idx_a1,idx_task] = old_y[idx_a2,idx_task]
                                    elif abs(old_y[idx_a2,idx_task] - y_winning_bid[idx_a1,idx_task]) <= 10e-6:
                                        if z_winning_agent[idx_a1,idx_task] > old_z[idx_a2,idx_task]: # Tie -break based on smaller index
                                            z_winning_agent[idx_a1,idx_task] = old_z[idx_a2,idx_task]
                                            y_winning_bid[idx_a1,idx_task] = old_y[idx_a2,idx_task]
                                    # print(z_winningagent)
                                    # print(y_winningbid)
                                elif z_winning_agent[idx_a1,idx_task] == idx_a2:
                                    # print("Entry 2")
                                    z_winning_agent[idx_a1,idx_task] = old_z[idx_a2,idx_task]
                                    y_winning_bid[idx_a1,idx_task] = old_y[idx_a2,idx_task]
                                elif z_winning_agent[idx_a1,idx_task] > -1:
                                    # print("Entry 3")
                                    if old_y[idx_a2,idx_task] > y_winning_bid[idx_a1,idx_task]:
                                        z_winning_agent[idx_a1,idx_task] = old_z[idx_a2,idx_task]
                                        y_winning_bid[idx_a1,idx_task] = old_y[idx_a2,idx_task]
                                elif z_winning_agent[idx_a1,idx_task] == -1:
                                    # print("Entry 4")
                                    z_winning_agent[idx_a1,idx_task] = old_z[idx_a2,idx_task]
                                    y_winning_bid[idx_a1,idx_task] = old_y[idx_a2,idx_task]
                                else:
                                    print("Error")
                            elif old_z[idx_a2,idx_task] == idx_a1:
                                if z_winning_agent[idx_a1,idx_task] == idx_a1:
                                    # print("Entry 5")
                                    old_z[idx_a2,idx_task] = old_z[idx_a2,idx_task] # do nothing 인데 비어 있으면 에러 떠서 의미 없는 코드 한 줄 삽입함
                                elif z_winning_agent[idx_a1,idx_task] == idx_a2:
                                    # print("Entry 6")
                                    z_winning_agent[idx_a1,idx_task] = -1
                                    y_winning_bid[idx_a1,idx_task] = 0
                                elif z_winning_agent[idx_a1,idx_task] > -1:
                                    # print("Entry 7")
                                    z_winning_agent[idx_a1,idx_task] = -1
                                    y_winning_bid[idx_a1,idx_task] = 0
                                elif z_winning_agent[idx_a1,idx_task] == -1:
                                    # print("Entry 8")
                                    old_z[idx_a2,idx_task] = old_z[idx_a2,idx_task] # do nothing 인데 비어 있으면 에러 떠서 의미 없는 코드 한 줄 삽입함
                                else:
                                    print("Error")
                            elif old_z[idx_a2,idx_task] > -1:
                                if z_winning_agent[idx_a1,idx_task] == idx_a1:
                                    # print("Entry 9")
                                    if old_y[idx_a2,idx_task] > y_winning_bid[idx_a1,idx_task]:
                                        z_winning_agent[idx_a1,idx_task] = old_z[idx_a2,idx_task]
                                        y_winning_bid[idx_a1,idx_task] = old_y[idx_a2,idx_task]
                                elif z_winning_agent[idx_a1,idx_task] == idx_a2:
                                    # print("Entry 10")
                                    z_winning_agent[idx_a1,idx_task] = old_z[idx_a2,idx_task]
                                    y_winning_bid[idx_a1,idx_task] = old_y[idx_a2,idx_task]
                                elif z_winning_agent[idx_a1,idx_task] == old_z[idx_a2,idx_task]:
                                    z_winning_agent[idx_a1,idx_task] = old_z[idx_a2,idx_task]
                                    y_winning_bid[idx_a1,idx_task] = old_y[idx_a2,idx_task]
                                elif z_winning_agent[idx_a1,idx_task] > -1:
                                    if old_y[idx_a2,idx_task] > y_winning_bid[idx_a1,idx_task]:
                                        z_winning_agent[idx_a1,idx_task] = old_z[idx_a2,idx_task]
                                        y_winning_bid[idx_a1,idx_task] = old_y[idx_a2,idx_task]
                                elif z_winning_agent[idx_a1,idx_task] == -1:
                                    z_winning_agent[idx_a1,idx_task] = old_z[idx_a2,idx_task]
                                    y_winning_bid[idx_a1,idx_task] = old_y[idx_a2,idx_task]
                                else:
                                    print("Error")
                            elif old_z[idx_a2,idx_task] == -1:
                                if z_winning_agent[idx_a1,idx_task] == idx_a1:
                                    old_z[idx_a2,idx_task] = old_z[idx_a2,idx_task] # do nothing 인데 비어 있으면 에러 떠서 의미 없는 코드 한 줄 삽입함
                                elif z_winning_agent[idx_a1,idx_task] == idx_a2:
                                    z_winning_agent[idx_a1,idx_task] = -1
                                    y_winning_bid[idx_a1,idx_task] = 0
                                elif z_winning_agent[idx_a1,idx_task] > -1:
                                    z_winning_agent[idx_a1,idx_task] = -1
                                    y_winning_bid[idx_a1,idx_task] = 0
                                elif z_winning_agent[idx_a1,idx_task] == -1:
                                    old_z[idx_a2,idx_task] = old_z[idx_a2,idx_task] # do nothing 인데 비어 있으면 에러 떠서 의미 없는 코드 한 줄 삽입함
                                else:
                                    print("Error")
                            else:
                                print("Error")
            # print("z",z_winningagent)
            # phase 2-1 bundle remove

            old_b = []
            for idx_a1 in range(len(b_bundle)):
                old_b.append([])
                for idx_task in range(len(b_bundle[idx_a1])):
                    old_b[idx_a1].append(b_bundle[idx_a1,idx_task])
            # print('old_b',old_b)
            for idx_a1 in range(num_agent):
                bundle = []
                flag_bundle = -1
                for idx_task in range(num_lt):
                    if b_bundle[idx_a1,idx_task] != -1:
                        bundle.append(int(b_bundle[idx_a1,idx_task]))
                for idx_task in range(len(bundle)):
                    if flag_bundle == -1:
                        if z_winning_agent[idx_a1,bundle[idx_task]] != idx_a1:
                            flag_bundle = idx_task
                # print(flag_bundle)
                # print("bundle",i,b_bundle[i])
                if flag_bundle > -1:
                    for idx_task in range(len(bundle) - flag_bundle):
                        for idx_a2 in range(len(p_path[idx_a1])):
                            if p_path[idx_a1,idx_a2] == b_bundle[idx_a1,idx_task+flag_bundle]:
                                if p_path[idx_a1,idx_a2] != -1:
                                    p_path[idx_a1,idx_a2] = -1
                        b_bundle[idx_a1,idx_task+flag_bundle] = -1

                    for idx_task in range(len(bundle) - flag_bundle - 1):
                        z_winning_agent[idx_a1,bundle[idx_task+flag_bundle+1]] = -1
                        y_winning_bid[idx_a1,bundle[idx_task+flag_bundle+1]] = 0
            # print("b_bundle",b_bundle,old_b)
            flags = []
            for idx_a1 in range(num_agent):
                if np.array_equal(old_b[idx_a1],b_bundle[idx_a1]):
                    flags.append(1)

            if len(flags) == num_agent:
                flag_continue_consensus = False
                print("converged", iter_consensus)

            if iter_consensus > 100:
                flag_continue_consensus = False
                print("not converged")

        return pos_agents, pos_tasks, p_path

    def process(self, num_task, num_lt, llambda):
        pos_tasks = self.generate_points(num_task, p_.roi_x, p_.roi_y)  # task position setting
        pos_agents = np.array(p_.pos_agents)
        pos_agents, pos_tasks, p_path = \
            self.core_bidding(pos_tasks, pos_agents, num_lt, llambda)
        
        return pos_agents, pos_tasks, p_path


