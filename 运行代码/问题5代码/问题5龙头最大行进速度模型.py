import math
import random
import matplotlib.pyplot as plt


# 定义目标函数，即需要最小化的成本函数
def cost_function(v, problem_parameters):
    cost = 0
    for i in range(len(problem_parameters['benches'])):
        segment_speed = calculate_segment_speed(v, i, problem_parameters)
        if segment_speed > 2:
            cost += (segment_speed - 2) ** 2
    return cost


# 计算某一节板凳的速度
def calculate_segment_speed(v, segment_index, problem_parameters):
    return v * (1 + segment_index * 0.01)


# 模拟退火算法
def simulated_annealing(problem_parameters):
    v_current = 1.0
    T = 1000
    T_min = 0.0001
    alpha = 0.95
    cost_current = cost_function(v_current, problem_parameters)

    best_v = v_current
    best_cost = cost_current
    iteration = 0
    history = []  # 记录每次迭代的最优速度

    while T > T_min:
        v_new = v_current + random.uniform(-0.1, 0.1)
        cost_new = cost_function(v_new, problem_parameters)

        if cost_new > best_cost:
            best_v = v_new
            best_cost = cost_new
            history.append((best_v, best_cost))
        elif random.random() < math.exp(-(cost_new - cost_current) / T):
            v_current = v_new
            cost_current = cost_new
            history.append((v_current, cost_current))
        else:
            history.append((v_current, cost_current))

        iteration += 1
        T *= alpha

    return best_v, iteration, history


# 定义问题参数
problem_parameters = {
    'benches': list(range(1, 224))  # 223节板凳
}

# 运行模拟退火算法
optimal_speed, iterations, history = simulated_annealing(problem_parameters)

# 绘制迭代次数图
plt.figure(figsize=(10, 5))
plt.plot([i[0] for i in history], label='Speed at each iteration')
plt.title('Optimal Speed Iteration Graph')
plt.xlabel('Iteration')
plt.ylabel('Speed (m/s)')
plt.legend()
plt.show()