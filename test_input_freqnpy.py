import pickle
import numpy as np
import os

def analyze_array_stats(arr, name):
    """详细分析数组的统计特征和NaN情况"""
    print(f"\n▌ 详细分析数组 [{name}]")
    
    # 基础信息
    print(f"形状: {arr.shape}")
    print(f"数据类型: {arr.dtype}")
    print(f"总元素数: {arr.size:,}")
    
    # 处理全NaN数组的特殊情况
    if np.all(np.isnan(arr)):
        print("⚠️ 警告: 数组所有元素均为NaN!")
        return
    
    # 基本统计量（自动处理NaN）
    print("\n[基础统计]")
    print(f"最小值: {np.nanmin(arr):.6f}")
    print(f"最大值: {np.nanmax(arr):.6f}")
    print(f"平均值: {np.nanmean(arr):.6f}")
    print(f"标准差: {np.nanstd(arr):.6f}")
    print(f"中位数: {np.nanmedian(arr):.6f}")
    
    # 分位数统计
    print("\n[分位数统计]")
    for q in [0.25, 0.5, 0.75, 0.95]:
        print(f"{q*100:.0f}%分位数: {np.nanquantile(arr, q):.6f}")
    
    # 零值分析
    if np.issubdtype(arr.dtype, np.number):
        zero_count = np.sum(arr == 0)
        print(f"\n零值数量: {zero_count} ({zero_count/arr.size:.2%})")
    
    # NaN分析
    nan_count = np.sum(np.isnan(arr))
    inf_count = np.sum(np.isinf(arr))
    print(f"\n[异常值统计]")
    print(f"NaN元素数: {nan_count} ({nan_count/arr.size:.2%})")
    print(f"Inf元素数: {inf_count} ({inf_count/arr.size:.2%})")
    
    # 特殊值分布示例
    if nan_count > 0:
        print("\nNaN位置示例:")
        nan_indices = np.argwhere(np.isnan(arr))
        for idx in nan_indices[:3]:
            print(f"索引 {tuple(idx)}")

def load_and_compare(file1, file2, data_type):
    """加载并比较两个文件，增加统计分析"""
    print(f"\n{'='*60}")
    print(f"▶ 文件比较：")
    print(f"  组1路径: {file1}")
    print(f"  组2路径: {file2}")

    # 检查文件存在性
    for f in [file1, file2]:
        if not os.path.exists(f):
            print(f"❌ 错误：文件不存在 - {f}")
            return

    try:
        if data_type == "numpy":
            # 加载数据
            data1 = np.load(file1)
            data2 = np.load(file2)
            
            # 对每个文件进行独立分析
            print("\n" + "="*60)
            analyze_array_stats(data1, os.path.basename(file1))
            print("\n" + "="*60)
            analyze_array_stats(data2, os.path.basename(file2))
            
            # 执行比较
            print("\n" + "="*60)
            compare_arrays(data1, data2, os.path.basename(file1))
        else:
            with open(file1, "rb") as f1, open(file2, "rb") as f2:
                data1 = pickle.load(f1)
                data2 = pickle.load(f2)
                compare_dicts(data1, data2, os.path.basename(file1))
    except Exception as e:
        print(f"❌ 加载文件时出错: {str(e)}")
        import traceback
        traceback.print_exc()

# 文件路径配置（根据实际情况修改）
group1 = {
    "freq": "data/Freq.npy"  # 原始文件路径
}

group2 = {
    "freq": "/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/Freq/Freq.npy"  # 新文件路径
}

# 执行分析比较
load_and_compare(group1["freq"], group2["freq"], "numpy")
