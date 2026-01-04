import numpy as np
import pandas as pd

# 读取数据
df_x = pd.read_excel('前端x的坐标.xlsx')
df_y = pd.read_excel('前端y的坐标.xlsx')


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
r_df.to_excel("各个节点速度.xlsx")
    # 如果需要，可以在这里添加第一列作为索引或其他非数据列
# 例如，如果原始数据有时间戳或ID作为第一列，可以保留它

