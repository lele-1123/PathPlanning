"""
Environment for rrt_2D
@author: huiming zhou
"""


class Env:
    def __init__(self):
        self.x_range = (0, 50)  # 环境的x范围
        self.y_range = (0, 30)  # 环境的y范围
        self.obs_boundary = self.obs_boundary()     # 边界墙
        self.obs_circle = self.obs_circle()         # 圆形障碍物
        self.obs_rectangle = self.obs_rectangle()   # 矩形障碍物

    @staticmethod
    def obs_boundary():
        obs_boundary = [
            [0, 0, 1, 30],  # [左下角起始点x, y, 边界墙的宽x, 高y]
            [0, 30, 50, 1],
            [1, 0, 50, 1],
            [50, 1, 1, 30]
        ]
        return obs_boundary

    @staticmethod
    def obs_rectangle():
        obs_rectangle = [
            [14, 12, 8, 2],  # [矩形左下角x, y, 矩形的宽x, 高y]
            [18, 22, 8, 3],
            [26, 7, 2, 12],
            [32, 14, 10, 2]
        ]
        return obs_rectangle

    @staticmethod
    def obs_circle():
        obs_cir = [
            [7, 12, 3],  # [圆心x, 圆心y, 半径r]
            [46, 20, 2],
            [15, 5, 2],
            [37, 7, 3],
            [37, 23, 3]
        ]

        return obs_cir
