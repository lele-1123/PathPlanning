"""
RRT_star 2D
@author: lele li
    更改说明：在rrt_star2.py后，更改node_neighbor的碰撞检测位置，试图进一步减小计算时间
"""

import math
import numpy as np
import time

from Sampling_based_Planning.rrt_2D import env, plotting, utils


class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None
        self.cost = 0


class RrtStar:
    def __init__(self, x_start, x_goal, step_len,
                 goal_sample_rate, search_radius, iter_max):
        self.s_start = Node(x_start)
        self.s_goal = Node(x_goal)
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self.iter_max = iter_max
        self.vertex = [self.s_start]
        self.path = []

        self.env = env.Env()
        self.plotting = plotting.Plotting(x_start, x_goal)
        self.utils = utils.Utils()

        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary

    def planning(self):
        t1 = time.perf_counter()
        for k in range(self.iter_max):
            node_rand = self.generate_random_node(self.goal_sample_rate)    # 产生随机点 =3(伪代码中的行数)
            node_near = self.nearest_neighbor(self.vertex, node_rand)       # 在树中选择距离随机点最近的点 =4
            node_new = self.new_state(node_near, node_rand)                 # 确定输入，产生由node_near到node_rand的新点 =5

            # if k % 500 == 0:
            #     print(k)

            if node_new and not self.utils.is_collision(node_near, node_new):  # node_near到node_new是否与障碍物碰撞 =6
                neighbor_index = self.find_near_neighbor(node_new)          # 选取node_new附近的点集, 且不会发生碰撞 =7
                self.vertex.append(node_new)                                # 添加新点到树的顶点(图的点集合)中 =8
                node_new.parent = node_near
                node_new.cost = self.get_new_cost(node_near, node_new)      # 先设定node_near为node_new的父节点 =9

                if neighbor_index:
                    self.choose_parent(node_new, neighbor_index)            # connect along a minimum-cost path
                    self.rewire(node_new, neighbor_index)                   # rewire the tree

        index = self.search_goal_parent()
        self.s_goal.parent = self.vertex[index]
        self.s_goal.cost = self.get_new_cost(self.s_goal.parent, self.s_goal)
        self.path = self.extract_path()
        t2 = time.perf_counter()
        # print("the cost of path is " + str(self.s_goal.cost))
        t = int(t2 - t1)

        self.plotting.animation(self.vertex, self.path,
                                "RRT*, r = " + str(self.search_radius) + ", t = " + str(t) + ", c = " + str(int(self.s_goal.cost)))  # 绘制图像

    def new_state(self, node_start, node_goal):
        dist, theta = self.get_distance_and_angle(node_start, node_goal)

        dist = min(self.step_len, dist)
        node_new = Node((node_start.x + dist * math.cos(theta),
                         node_start.y + dist * math.sin(theta)))

        node_new.parent = node_start

        return node_new

    def choose_parent(self, node_new, neighbor_index):
        for i in neighbor_index:
            node_neighbor = self.vertex[i]
            c_new = self.get_new_cost(node_neighbor, node_new)
            if c_new < node_new.cost:
                if self.utils.is_collision(node_neighbor, node_new):
                    neighbor_index.remove(i)
                else:
                    node_new.parent = node_neighbor
                    node_new.cost = c_new

    def rewire(self, node_new, neighbor_index):
        for i in neighbor_index:
            node_neighbor = self.vertex[i]
            cost_rewire = self.get_new_cost(node_new, node_neighbor)

            if node_neighbor.cost > cost_rewire and not self.utils.is_collision(node_neighbor, node_new):
                node_neighbor.parent = node_new
                node_neighbor.cost = cost_rewire

    def search_goal_parent(self):
        dist_list = [math.hypot(n.x - self.s_goal.x, n.y - self.s_goal.y) for n in self.vertex]
        node_index = [i for i in range(len(dist_list))
                      if dist_list[i] <= self.step_len and not self.utils.is_collision(self.vertex[i], self.s_goal)]

        if len(node_index) > 0:
            cost_list = [dist_list[i] + self.vertex[i].cost for i in node_index]
            return node_index[int(np.argmin(cost_list))]

        return len(self.vertex) - 1

    def get_new_cost(self, node_start, node_end):
        dist = math.hypot(node_end.x - node_start.x, node_end.y - node_start.y)

        if node_start == 0:
            return self.cost_node(node_start) + dist

        return node_start.cost + dist

    def generate_random_node(self, goal_sample_rate):
        delta = self.utils.delta

        if np.random.random() > goal_sample_rate:
            return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                         np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))

        return self.s_goal

    def find_near_neighbor(self, node_new):
        n = len(self.vertex) + 1
        r = min(self.search_radius * math.sqrt((math.log(n) / n)), self.step_len)

        dist_table = [math.hypot(nd.x - node_new.x, nd.y - node_new.y) for nd in self.vertex]
        dist_table_index = [ind for ind in range(n - 1) if dist_table[ind] <= r]

        return dist_table_index

    @staticmethod
    def nearest_neighbor(node_list, n):
        return node_list[int(np.argmin([math.hypot(nd.x - n.x, nd.y - n.y)
                                        for nd in node_list]))]

    @staticmethod
    def cost_node(node_p):
        node = node_p
        cost = 0.0

        while node.parent:
            cost += math.hypot(node.x - node.parent.x, node.y - node.parent.y)
            node = node.parent

        return cost

    # def update_cost(self, parent_node):
    #     OPEN = queue.QueueFIFO()
    #     OPEN.put(parent_node)
    #
    #     while not OPEN.empty():
    #         node = OPEN.get()
    #
    #         if len(node.child) == 0:
    #             continue
    #
    #         for node_c in node.child:
    #             node_c.Cost = self.get_new_cost(node, node_c)
    #             OPEN.put(node_c)

    def extract_path(self):  # 提取路径
        path = [(self.s_goal.x, self.s_goal.y)]
        node_now = self.s_goal

        while node_now.parent is not None:
            node_now = node_now.parent
            path.append((node_now.x, node_now.y))

        return path

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)


def main():
    x_start = (18, 8)  # Starting node
    x_goal = (39, 18)  # Goal node

    rrt_star = RrtStar(x_start, x_goal, 15, 0.10, 30, 8000)
    rrt_star.planning()


if __name__ == '__main__':
    main()
