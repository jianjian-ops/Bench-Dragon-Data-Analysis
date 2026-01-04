import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotlib
import math

matplotlib.rcParams['font.family'] = 'SimHei'
# 固定的径向增量，近似螺距概念
pitch = 50.4700  # 单位：cm
theta_min = 0  # 角度最小值
theta_max = 20 * np.pi  # 角度最大值
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

def cross_product(O, A, B):
    """计算向量OA和OB的叉积（二维中为标量）"""
    return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0])


def segments_intersect(A, B, C, D):
    """判断线段AB和CD是否相交"""
    # 快速排斥实验
    if max(A[0], B[0]) < min(C[0], D[0]) or \
            min(A[0], B[0]) > max(C[0], D[0]) or \
            max(A[1], B[1]) < min(C[1], D[1]) or \
            min(A[1], B[1]) > max(C[1], D[1]):
        return False

        # 跨立实验
    d1 = cross_product(A, C, D)
    d2 = cross_product(B, C, D)
    d3 = cross_product(C, A, B)
    d4 = cross_product(D, A, B)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
            ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


r0 = 16 * 55
w = 22
flat = False
def time_f1(t):
        return arc_length(t, theta_max)



thetahead = []
for time in range(440):
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
    x_head_o = (r_head+w) * np.cos(theta_t)
    x_head_i = (r_head-w) * np.cos(theta_t)
    y_head_o = (r_head + w) * np.sin(theta_t)
    y_head_i = (r_head - w) * np.sin(theta_t)

    # 弦的长度
    def time_f(theta, theta0):
        r = b * theta
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        r0 = b * theta0
        x0 = r0 * np.cos(theta0)
        y0 = r0 * np.sin(theta0)
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
    x_o = (r_b+w) * np.cos(theta_solution)
    y_o = (r_b+w) * np.sin(theta_solution)
    x_i = (r_b - w) * np.cos(theta_solution)
    y_i = (r_b - w) * np.sin(theta_solution)
    xian = []
    xian.append((x_head_o,y_head_o))
    xian.append((x_o[0],y_o[0]))
    for i in range(len(x_i)):
        xian.append((x_i[i],y_i[i]))
    for i in range(3,20):
        if segments_intersect(xian[0],xian[1],xian[i],xian[i+1]):
            xo = np.insert(x_o, 0, x_head_o)
            yo = np.insert(y_o, 0, y_head_o)
            xi = np.insert(x_i, 0, x_head_i)
            yi = np.insert(y_i, 0, y_head_i)

            plt.figure(figsize=(8, 8))
            plt.plot(xo, yo,'ro',markersize='2' )  # 添加标签
            plt.plot(xi, yi,'bo',markersize='2')  # 添加标签

            for i in range(len(xo) - 1):
                if i == 0:
                    plt.plot([xo[i], xo[i + 1]], [yo[i], yo[i + 1]], 'r-',label = '板凳龙内测边缘')
                plt.plot([xo[i], xo[i + 1]], [yo[i], yo[i + 1]], 'r-')
            for i in range(len(xi) - 1):
                if i == 0:
                    plt.plot([xi[i], xi[i + 1]], [yi[i], yi[i + 1]], 'b-',label = '板凳龙外测边缘')
                plt.plot([xi[i], xi[i + 1]], [yi[i], yi[i + 1]], 'b-')

            plt.legend()  # 显示图例
            plt.title(f'螺距为{pitch}cm时发生首次碰撞')
            # plt.title('盘龙队伍初始状态板块的模型')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.axis('equal')
            plt.grid(True)
            plt.show()
            flat = True
            if flat:
                break
        if flat:
            break
    if flat:
        break















