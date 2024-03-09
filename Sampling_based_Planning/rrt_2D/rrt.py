"""
RRT_2D
@author: huiming zhou
"""

import os
import sys
import math
import numpy as np

# sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
#                 "/../../Sampling_based_Planning/")

from Sampling_based_Planning.rrt_2D import env, plotting, utils


class Node:
    def __init__(self, n):
        self.x = n[0]  # 点的x坐标
        self.y = n[1]  # 点的y坐标
        self.parent = None  # 点在树中的父节点


class Rrt:
    def __init__(self, s_start, s_goal, step_len, goal_sample_rate, iter_max):
        self.s_start = Node(s_start)    # 起始点
        self.s_goal = Node(s_goal)      # 目的地
        self.step_len = step_len        # 步长
        self.goal_sample_rate = goal_sample_rate  # 采样速率：随机点以此概率朝向目的地
        self.iter_max = iter_max        # 最大迭代数
        self.vertex = [self.s_start]    # 树的顶点

        self.env = env.Env()
        self.plotting = plotting.Plotting(s_start, s_goal)
        self.utils = utils.Utils()

        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary

    def planning(self):
        for i in range(self.iter_max):
            node_rand = self.generate_random_node(self.goal_sample_rate)        # 产生随机点
            node_near = self.nearest_neighbor(self.vertex, node_rand)           # 在树中选择距离随机点最近的点
            node_new = self.new_state(node_near, node_rand)                     # 确定输入，产生由node_near到node_rand的新点

            if node_new and not self.utils.is_collision(node_near, node_new):   # node_near到node_new是否与障碍物碰撞
                self.vertex.append(node_new)  # 不碰撞，添加新点到树的顶点中
                dist, _ = self.get_distance_and_angle(node_new, self.s_goal)  # 计算新点到终点的距离

                if dist <= self.step_len and not self.utils.is_collision(node_new, self.s_goal):  # 判断下一步是否可以直接到终点
                    # self.new_state(node_new, self.s_goal)
                    # return self.extract_path(node_new)  # 提取路径
                    self.s_goal.parent = node_new  # 可以取代上两行
                    return self.extract_path(self.s_goal)  # 提取路径

        return None

    def generate_random_node(self, goal_sample_rate):  # 产生随机点
        delta = self.utils.delta

        if np.random.random() > goal_sample_rate:  # 随机数大于rate，随机选点；否则选择终点。更快地完成规划
            return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                         np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))

        return self.s_goal

    @staticmethod
    def nearest_neighbor(node_list, n):  # 输出node_list中距离n最近的点
        return node_list[int(np.argmin([math.hypot(nd.x - n.x, nd.y - n.y)
                                        for nd in node_list]))]

    def new_state(self, node_start, node_end):  # 确定新点
        dist, theta = self.get_distance_and_angle(node_start, node_end)  # 计算start到end的距离和角度

        dist = min(self.step_len, dist)  # 步长和距离中的最小值
        node_new = Node((node_start.x + dist * math.cos(theta),
                         node_start.y + dist * math.sin(theta)))  # 确定新点
        node_new.parent = node_start  # 新点的父点

        return node_new

    def extract_path(self, node_end):  # 提取路径
        path = [(self.s_goal.x, self.s_goal.y)]
        node_now = node_end

        while node_now.parent is not None:
            node_now = node_now.parent
            path.append((node_now.x, node_now.y))

        return path

    @staticmethod
    def get_distance_and_angle(node_start, node_end):  # 计算start到end的距离和角度
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)


def main():
    x_start = (2, 2)  # Starting node
    x_goal = (45, 24)  # Goal node

    rrt = Rrt(x_start, x_goal, 0.5, 0.05, 10000)
    path = rrt.planning()  # 规划路径

    if path:
        rrt.plotting.animation(rrt.vertex, path, "RRT", True)  # 绘制图像
    else:
        print("No Path Found!")


if __name__ == '__main__':
    main()
