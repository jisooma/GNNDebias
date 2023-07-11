#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/13 15:24
# @Author  : zhixiuma
# @File    : test.py
# @Project : Test
# @Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np

# x = np.random.rand(50)
# y = np.random.rand(50)
# colors = np.random.rand(50)
# markers = ["o", "s", "D", "^", "v", "P", "*", "X", "H", "+"]
# selected_markers = [np.random.choice(markers) for _ in range(50)]
#
# fig, ax = plt.subplots()
# scatter = ax.scatter(x, y, c=colors, marker=selected_markers)
#
# plt.show()


import numpy as np
import matplotlib.pyplot as plt

# 生成一些示例数据
x = np.random.normal(size=50)
y = np.random.normal(size=50)
colors = np.random.rand(50)

# 绘制 scatter 图形，并设置颜色
plt.scatter(x, y, c='red', alpha=0.5)  # 使用单个字符串值设置颜色
plt.scatter(x, y+1, c=colors, cmap='viridis', alpha=0.5)  # 使用 colormap 设置颜色
plt.scatter(x, y+2, c=np.random.rand(50, 3), alpha=0.5)  # 使用 RGB 值设置颜色

# 显示图形
plt.show()
