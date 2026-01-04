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

f = pd.read_excel('龙头前端位置.xlsx')
# 已知端点的信息
r1 = 880
theta1 = 32*np.pi  # 将角度转换为弧度
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
    print(math.sqrt((x - x0)**2 + (y - y0)**2))
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



# 生成等距螺线的点
theta_range = np.linspace(0, 2 * np.pi * 16, 2000)  # 调整范围以包含解
r = a + b * theta_range
x = r * np.cos(theta_range)
y = r * np.sin(theta_range)

# 绘制等距螺线
plt.figure(figsize=(8, 8))
plt.plot(x, y)

# 标记已知端点
plt.plot(x1, y1, 'ro', label='A')

# 标记计算得到的端点
plt.plot(x2, y2, 'bo')

plt.plot([x1,x2[0]], [y1,y2[0]],label='龙头')
for i in range(len(x2) - 1):
    plt.plot([x2[i], x2[i+1]], [y2[i], y2[i+1]], 'r--')

# 添加图例
plt.legend()

# 设置图表标题和轴标签
plt.title('盘龙队伍初始位置表')
plt.xlabel('X')
plt.ylabel('Y')

# 显示图表
plt.axis('equal')  # 确保x轴和y轴的缩放比例相同
plt.grid(True)
plt.show()