import pickle
import numpy as np
import os

def print_array_samples(arr, name, group_name, max_samples=3):
    """打印数组的示例内容"""
    print(f"\n▌ {group_name} 数组样本 [{name}]：")
    
    # 处理不同维度数组的展示
    if arr.ndim == 1:
        samples = arr[:max_samples]
        for i, val in enumerate(samples):
            print(f"索引 [{i}] → 值: {val:.6f}")
    elif arr.ndim == 2:
        samples = arr[:max_samples, :max_samples]
        for i in range(samples.shape[0]):
            for j in range(samples.shape[1]):
                print(f"索引 [{i},{j}] → 值: {samples[i,j]:.6f}")
    else:
        # 高维数组特殊处理
        flat_samples = arr.flat[:max_samples]
        print(f"前{max_samples}个元素值:")
        for idx, val in enumerate(flat_samples):
            print(f"元素 {idx} → 值: {val:.6f}")

def compare_dicts(d1, d2, name, max_examples=3):
    """比较两个字典的结构和内容，显示详细示例"""
    print(f"\n▌ 比较字典 [{name}]：")
    
    # 比较字典长度
    len1, len2 = len(d1), len(d2)
    print(f"键数量: 组1 ({len1}) | 组2 ({len2})")
    
    # 比较键集合差异
    keys1 = set(d1.keys())
    keys2 = set(d2.keys())
    common_keys = keys1 & keys2
    only_in_group1 = keys1 - keys2
    only_in_group2 = keys2 - keys1

    # 显示独有键示例
    if only_in_group1:
        print(f"\n→ 仅组1包含的键 ({len(only_in_group1)}个):")
        for k in list(only_in_group1)[:max_examples]:
            print(f"  键: {k} → 值: {d1[k]}")
    
    if only_in_group2:
        print(f"\n→ 仅组2包含的键 ({len(only_in_group2)}个):")
        for k in list(only_in_group2)[:max_examples]:
            print(f"  键: {k} → 值: {d2[k]}")

    # 比较共同键的值差异
    diff_keys = [k for k in common_keys if d1[k] != d2[k]]
    if diff_keys:
        print(f"\n→ 共同键的值差异 ({len(diff_keys)}个):")
        for k in diff_keys[:max_examples]:
            print(f"  键: {k}")
            print(f"  组1值: {d1[k]} | 组2值: {d2[k]}")

def compare_arrays(a1, a2, name, max_examples=3):
    """比较两个numpy数组，显示详细数值示例"""
    print(f"\n▌ 比较数组 [{name}]：")
    
    # 形状比较
    if a1.shape != a2.shape:
        print(f"形状不同 → 组1: {a1.shape} | 组2: {a2.shape}")
        return
    
    print(f"形状相同 ({a1.shape})")
    
    # 内容比较
    if np.array_equal(a1, a2):
        print("数据内容完全相同")
        return
    
    # 数值差异分析
    diff_mask = a1 != a2
    diff_count = np.sum(diff_mask)
    print(f"\n→ 发现差异元素 ({diff_count}/{a1.size}，{diff_count/a1.size:.2%})")
    
    # 显示数值差异示例
    diff_indices = np.argwhere(diff_mask)
    print(f"\n数值差异示例（最多显示{max_examples}处）：")
    
    for idx in diff_indices[:max_examples]:
        # 处理高维数组的索引
        idx_tuple = tuple(idx)
        val1 = a1[idx_tuple]
        val2 = a2[idx_tuple]
        diff = abs(val1 - val2)
        
        print(f"索引: {idx}")
        print(f"组1值: {val1:.6f} | 组2值: {val2:.6f}")
        print(f"绝对差值: {diff:.6f}\n")

def compare_arrays(a1, a2, name, max_examples=3):
    """增强版数组比较，包含内容展示"""
    print(f"\n{'='*60}")
    print(f"▌ 深度分析数组 [{name}]")
    
    # 先展示原始数据样本
    print_array_samples(a1, name, "组1原始数据")
    print_array_samples(a2, name, "组2原始数据")
    
    # 形状比较
    if a1.shape != a2.shape:
        print(f"\n❌ 形状差异 → 组1: {a1.shape} | 组2: {a2.shape}")
        return
    
    print(f"\n✅ 形状验证通过 ({a1.shape})")
    
    # 内容比较
    if np.array_equal(a1, a2):
        print("✅ 数据内容完全一致")
        return
    
    # 数值差异分析
    diff_mask = a1 != a2
    diff_count = np.sum(diff_mask)
    diff_percent = diff_count / a1.size * 100
    
    print(f"\n❌ 发现差异元素: {diff_count}/{a1.size} ({diff_percent:.2f}%)")
    
    # 差异示例展示
    print("\n▌ 差异位置对比：")
    diff_indices = np.argwhere(diff_mask)
    
    for idx in diff_indices[:max_examples]:
        idx_tuple = tuple(idx)
        val1 = a1[idx_tuple]
        val2 = a2[idx_tuple]
        
        print(f"\n→ 索引: {idx}")
        print(f"  组1值: {val1:.6f} ({val1.dtype})")
        print(f"  组2值: {val2:.6f} ({val2.dtype})")
        print(f"  绝对差值: {abs(val1 - val2):.6f}")
        if val1 != 0:
            print(f"  相对差值: {abs((val1 - val2)/val1)*100:.2f}%")
        else:
            print("  相对差值: N/A (分母为零)")

def inspect_pickle_file(file_path, max_items=10):
    """查看pickle文件内容"""
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            
            print(f"\n{'='*40}")
            print(f"文件内容检查: {file_path}")
            
            # 显示基础信息
            print(f"\n数据类型: {type(data).__name__}")
            if isinstance(data, dict):
                print(f"条目数量: {len(data)}")
                print("\n前{}个键值对示例:".format(max_items))
                
                # 打印示例条目
                for idx, (k, v) in enumerate(data.items()):
                    if idx >= max_items:
                        break
                    print(f"[{k}] → {v}")
                    
            elif isinstance(data, np.ndarray):
                print(f"数组形状: {data.shape}")
                print(f"数据类型: {data.dtype}")
                print("\n前10个元素值:")
                print(data.flat[:10])
                
            else:
                print("\n文件内容:")
                print(data)

    except Exception as e:
        print(f"\n错误: {str(e)}")
        print("提示: 该文件可能不是pickle格式，或使用了其他序列化方式")

def load_and_compare(file1, file2, data_type):
    """加载并比较两个文件"""
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
            data1 = np.load(file1)
            data2 = np.load(file2)
            compare_arrays(data1, data2, os.path.basename(file1))
        else:
            with open(file1, "rb") as f1, open(file2, "rb") as f2:
                data1 = pickle.load(f1)
                data2 = pickle.load(f2)
                compare_dicts(data1, data2, os.path.basename(file1))
    except Exception as e:
        print(f"❌ 加载文件时出错: {str(e)}")

if __name__ == "__main__":
    # 配置数据路径（请根据实际情况修改）
    group1 = {
        "type": "data/type_to_idx.bin",
        "pop": "data/pop_to_idx.bin",
        "pos": "data/pos_to_idx.bin",
        "freq": "data/Freq.npy"
    }

    group2 = {
        "type": "data/type_to_idx.bin",  # 保持相同
        "pop": "/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/pop_to_idx.bin",  # 替换实际路径
        "pos": "/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/pos_to_idx.bin",  # 替换实际路径
        "freq": "/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/Freq/Freq.npy"       # 替换实际路径
    }

    # 执行比较
    load_and_compare(group1["pop"], group2["pop"], "pickle")
    load_and_compare(group1["pos"], group2["pos"], "pickle")
    load_and_compare(group1["freq"], group2["freq"], "numpy")

    file_path = "data/type_to_idx.bin"
    
    # 安全检查
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 - {file_path}")
    else:
        # 执行检查（显示前10个条目）
        inspect_pickle_file(file_path)
        
        # 可选：验证是否是字典类型
        print("\n验证建议:")
        print("1. 如果输出显示数据类型为dict，则表示这是标准的字典文件")
        print("2. 若显示numpy.ndarray，请检查文件名是否混淆")
        print("3. 出现解码错误可能需要检查文件来源")
