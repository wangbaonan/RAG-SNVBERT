import numpy as np

# 读取.npy文件
data = np.load('data/Freq.npy')

# 查看基本信息
print(f"数组形状：{data.shape}")  # 维度信息
print(f"数据类型：{data.dtype}")  # 元素类型
print(f"数组维度：{data.ndim}")    # 维度数

# 查看数据示例（前5个元素）
print("前5个元素：", data[:5] if data.ndim == 1 else data[0, :5])

# 完整数据预览（适合小数组）
#print("完整数据：\n", data)