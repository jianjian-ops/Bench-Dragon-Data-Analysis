import numpy as np
from scipy.integrate import quad
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'SimHei'
# 固定的径向增量，近似螺距概念
pitch = 55  # 单位：cm
theta_min = 0  # 角度最小值
theta_max = 32 * np.pi  # 角度最大值
# 生成角度向量，从0到10π
theta = np.linspace(theta_min, theta_max, 2000)

# 近似固定螺距的等距螺线方程
# r = a + b * theta, 其中b为每转的径向增量
b = pitch / (2 * np.pi)
# 为了近似固定螺距，我们设定每转径向增量为pitch / (2 * np.pi)
r = b * theta

# 转换为笛卡尔坐标
x = r * np.cos(theta)
y = r * np.sin(theta)

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
# 计算弧长
length= arc_length(theta_min, theta_max)
print(f"The length of the spiral from theta = {theta_min} to theta = {theta_max} is approximately {length:.2f} cm.")

r0 = 16 * 55
def time_f(t):
        return arc_length(t, theta_max)
theta_t = []
for time in range(0,440):
    print(time)
    e = 1e-10
    a = theta_min
    b = theta_max
    if (time_f(a)-time*100) * (time_f(b)-time*100) >= 0:
        print("erro")
    while(b - a) > e:
        temp = (b + a) / 2
        if time_f(temp) - time*100 == 0:
            theta_t.append(temp)
        elif (time_f(a)-time*100) * (time_f(temp)-time*100)  < 0:
            b = temp
        else:
            a = temp
    theta_t.append((a+b)/2)

b = pitch/(2*np.pi)
theta_t = np.array(theta_t)
r_head  = b * theta_t
x_head = r_head * np.cos(theta_t)
y_head = r_head * np.sin(theta_t)

plt.figure(figsize=(8, 8))
plt.plot(x, y)
plt.plot(x_head, y_head,'bo',markersize = '3',label='各时刻龙头前端位置')
for i in range(len(x_head) - 1):
    plt.plot([x_head[i], x_head[i+1]], [y_head[i], y_head[i+1]], 'r--')
plt.title('初始时刻到300秒为止龙头前端的位置图')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')  # 保持坐标轴比例相同
plt.grid(True)
plt.show()

s1 = pd.Series(theta_t)
s2 = pd.Series(r_head)
df = pd.DataFrame({'角度':s1,'半径':s2})
df.to_excel('龙头前端位置.xlsx',index=False)
