import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotlib
import math
import pandas as pd

matplotlib.rcParams['font.family'] = 'SimHei'

def an(point_x,point_y,center_x,center_y,theta):

    # 半径
    radius = np.sqrt((point_x - center_x)**2 + (point_y - center_y)**2)

    # 从圆上一点对应的角度开始到 180 度（π 弧度）顺时针方向的角度范围
    start_angle = np.arctan2(point_y - center_y, point_x - center_x)
    print(start_angle)
    theta = np.linspace(start_angle, start_angle +theta, 100)

    # 计算圆上点的坐标
    x = center_x + radius * np.cos(theta)
    y = center_y + radius * np.sin(theta)
    return x,y

def se(point1_x, point1_y, point2_x, point2_y ):
    # 两点坐标
    point1 = (point1_x, point1_y)
    point2 = (point2_x, point2_y)

    # 计算两点间的距离
    distance = math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

    # 确定要选取的线段长度
    segment_length = distance*2

    # 计算单位向量
    dx = (point2[0] - point1[0]) / distance
    dy = (point2[1] - point1[1]) / distance

    # 六等分点坐标
    points_sixth = []
    for i in range(1, 6):
        new_point = (point1[0] + dx * segment_length * i / 6, point1[1] + dy * segment_length * i / 6)
        points_sixth.append(new_point)

    x1, y1 = an(point1[0],point1[1],points_sixth[1][0],points_sixth[1][1],-np.pi)
    x2, y2 = an(points_sixth[3][0], points_sixth[3][1], points_sixth[4][0], points_sixth[4][1], np.pi)
    x = np.concatenate((x1, x2))
    y = np.concatenate((y1, y2))
    return x, y


# 固定的径向增量，近似螺距概念
pitch = 170  # 单位：cm
theta_min = 0  # 角度最小值
theta_max = 10 * np.pi  # 角度最大值
X = []
Y = []
# 生成角度向量，从0到10π
theta = np.linspace(theta_min, 20*np.pi, 2000)

# 近似固定螺距的等距螺线方程
# r = a + b * theta, 其中b为每转的径向增量
b = pitch / (2 * np.pi)
# 为了近似固定螺距，我们设定每转径向增量为pitch / (2 * np.pi)
r = b * theta
# 转换为笛卡尔坐标
x = r * np.cos(theta)
y = r * np.sin(theta)
x1 = r * -np.cos(theta)
y1 = r * np.sin(-theta)

# 螺旋线r(theta)
def R(theta):
    return pitch / (2 * np.pi) * theta

# dx/dtheta 和 dy/dtheta
def dx_dtheta(theta):
    return (R(theta) * np.cos(theta) - pitch / (2 * np.pi) * np.sin(theta))

def dy_dtheta(theta):
    return (R(theta) * np.sin(theta) + pitch / (2 * np.pi) * np.cos(theta))

# 弧长积分
def arc_length(theta_min, theta_max):
    l,_ = quad(lambda theta: np.sqrt(dx_dtheta(theta)**2 + dy_dtheta(theta)**2), theta_min, theta_max)
    return l

def time_f1(t):
        return arc_length(t, theta_max)


thetahead = []
for time in range(101):
    print(time)
    e = 1e-10
    a = theta_min
    b = theta_max
    if (time_f1(a)-time*100) * (time_f1(b)-time*100) >= 0:
        print("erro")
    while(b - a) > e:
        temp = (b + a) / 2
        if time_f1(temp) - time*100 == 0:
            theta_t = temp
        elif (time_f1(a)-time*100) * (time_f1(temp)-time*100)  < 0:
            b = temp
        else:
            a = temp
    theta_t = (a+b)/2
    thetahead.append(theta_t)
    d = pitch / (2 * np.pi)
    r_head = d * theta_t
    x_head = r_head * np.cos(theta_t)
    y_head = r_head * np.sin(theta_t)

    # 弦的长度
    def time_f(theta, theta0):
        r = b * theta
        x = r * -np.cos(theta)
        y = r * -np.sin(theta)
        r0 = b * theta0
        x0 = r0 * -np.cos(theta0)
        y0 = r0 * -np.sin(theta0)
        return math.sqrt((x - x0) ** 2 + (y - y0) ** 2)


    theta1 = theta_t
    theta_1 = []
    for i in range(224):
        b1 = 341 - 2 * 27.5
        b2 = 220 - 2 * 27.5
        b = pitch/(2*np.pi)
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
        theta1 = (theta_a + theta_b) / 2
        theta_1.append(theta1)
    theta_solution = np.array(theta_1)

    # 计算另一个端点的坐标
    r_b = b * theta_solution
    x_b = r_b* np.cos(theta_solution)
    y_b = r_b* np.sin(theta_solution)

    X.append(np.insert(x_b, 0, x_head))
    Y.append(np.insert(y_b, 0, y_head))
r_head_np = np.array(b) *thetahead
n = len(X)
x_z,y_z = se(X[n-1][0],Y[n-1][0],0,0)
plt.figure(figsize=(8, 8))
plt.plot(x1, y1)
plt.plot(x, y)
plt.plot(x_z,y_z)
plt.plot(X[n-1],Y[n-1],'--',label = '盘龙队伍')
plt.legend()  # 显示图例
plt.title(f'盘龙队伍0时刻的状态')
# plt.title('盘龙队伍初始状态板块的模型')
plt.xlabel('X(单位/cm）')
plt.ylabel('Y(单位/cm)')
plt.axis('equal')
plt.grid(True)
plt.show()

df_x = pd.DataFrame()
df_y = pd.DataFrame()
for i, arr in enumerate(X):
    name = f'时间{i}s'
    df_x[name] = arr
df_x.index.name = '节点'
for j in range(len(X[0])):
    df_x.rename(index={j: f'第{j}个'},inplace=True)
df_x.to_excel("X问题四前端x的坐标.xlsx")

for i, arr in enumerate(Y):
    name = f'时间{i}s'
    df_y[name] = arr
df_y.index.name = '节点'
for j in range(len(Y[0])):
    df_y.rename(index={j: f'第{j}个'},inplace=True)
df_y.to_excel("Y问题四前端y的坐标.xlsx")

# 转换df_x的列到数值类型，无法转换的设置为NaN
for col in df_x.columns:
    df_x[col] = pd.to_numeric(df_x[col], errors='coerce')

# 初始化r_x来存储差值结果
r_x = pd.DataFrame(columns=df_x.columns[:-1], index=df_x.index)

# 遍历df_x的列，除了最后一列
for i in range(len(df_x.columns) - 1):
    col = df_x.columns[i]
    next_col = df_x.columns[i + 1]
    # 计算当前列与下一列之间的差值，并将结果存储在r_x中
    r_x[col] = df_x[col] - df_x[next_col]

# 同样处理df_y（注意这里应该使用df_y的索引和列）
for col in df_y.columns:
    df_y[col] = pd.to_numeric(df_y[col], errors='coerce')

r_y = pd.DataFrame(columns=df_y.columns[:-1], index=df_y.index)

# 遍历df_y的列，除了最后一列
for i in range(len(df_y.columns) - 1):
    col = df_y.columns[i]
    next_col = df_y.columns[i + 1]
    # 计算当前列与下一列之间的差值，并将结果存储在r_y中
    r_y[col] = df_y[col] - df_y[next_col]

# 创建一个新的DataFrame来存储结果
r_df = pd.DataFrame(index=r_x.index)

# 计算每个位置的距离
for col in r_x.columns[1:]:
    if col in r_y.columns:  # 确保两个DataFrame都有这个列
        r_df[col] = np.sqrt(r_x[col] ** 2 + r_y[col] ** 2)
r_df.to_excel("问题四各个节点速度.xlsx")
    # 如果需要，可以在这里添加第一列作为索引或其他非数据列
# 例如，如果原始数据有时间戳或ID作为第一列，可以保留它











