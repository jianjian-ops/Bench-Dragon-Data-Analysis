import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# 指定字体为支持中文的字体，例如：'SimHei'
import pandas as pd

matplotlib.rcParams['font.family'] = 'SimHei'

# 设置等距螺线的参数
a = 0
b = 55 / (2 * np.pi)  # 可以调整b的值来改变螺线的紧密程度
X = []
Y = []
i = 0
f = pd.read_excel('龙头前端位置.xlsx')
# 已知端点的信息
theta = f['角度'].to_numpy()
for theta1 in theta:
    # print(theta1)
    r1 = b*theta1
    x1 = r1 * np.cos(theta1)
    y1 = r1 * np.sin(theta1)

    # 弦的长度
    def time_f(theta, theta0):
        r = b * theta
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        r0 = b * theta0
        x0 = r0 * np.cos(theta0)
        y0 = r0 * np.sin(theta0)
        return math.sqrt((x - x0)**2 + (y - y0)**2)

    theta_1 = []
    for i in range(224):
        b1 = 341 - 2 * 27.5
        b2 =220 - 2 * 27.5
        if i == 0:
            l = b1
        else:
            l = b2
        e = 1e-10
        theta_a = theta1
        theta_b = theta1 + np.pi
        while (theta_b - theta_a) > e:
            if (time_f(theta_a, theta1) - l) * (time_f(theta_b, theta1) - l) >= 0:
                theta_a = theta_a + np.pi
                theta_b = theta_b + np.pi
            else:
                temp = (theta_a + theta_b) / 2
                if math.isclose(time_f(temp, theta1), l, abs_tol=e):
                    theta2 = temp
                    break
                elif (time_f(theta_a, theta1) - l) * (time_f(temp, theta1) - l) < 0:
                    theta_b = temp
                else:
                    theta_a = temp
        theta1 =(theta_a + theta_b) / 2
        theta_1.append(theta1)

    theta_solution = np.array(theta_1)
    # 计算另一个端点的坐标
    r2 = a + b * theta_solution
    x2 = r2 * np.cos(theta_solution)
    y2 = r2 * np.sin(theta_solution)
    X.append(np.insert(x2,0,x1))
    Y.append(np.insert(y2,0,y1))
df_x = pd.DataFrame()
df_y = pd.DataFrame()
for i, arr in enumerate(X):
    name = f'时间{i}s'
    df_x[name] = arr
df_x.index.name = '节点'
for j in range(len(X[0])):
    df_x.rename(index={j: f'第{j}个'},inplace=True)
df_x.to_excel("前端x的坐标.xlsx")

for i, arr in enumerate(Y):
    name = f'时间{i}s'
    df_y[name] = arr
df_y.index.name = '节点'
for j in range(len(Y[0])):
    df_y.rename(index={j: f'第{j}个'},inplace=True)
df_y.to_excel("前端y的坐标.xlsx")



