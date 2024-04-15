import numpy as np

# 假设这是你的多维数组
data = np.array([
    [4, 9, 2],
    [1, 5, 6],
    [7, 3, 8],
    [2, 8, 4]
])

# 使用argsort()函数获取第一列的排序索引
sorted_indices = np.argsort(data[:, 0])

# 使用索引数组对整个数组进行排序
sorted_data = data[sorted_indices]

# 输出排序后的数组
print("原始数组:")
print(data)
print("排序后的数组:")
print(sorted_data)