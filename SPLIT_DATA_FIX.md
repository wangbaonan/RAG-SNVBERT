# Split Data修复说明

## 🐛 问题

运行`run_v12_split_val.sh`后出现错误：

```
KeyError: 'Unable to open object (component not found)'
```

在尝试读取`variants/POS`时失败。

---

## 🔍 原因

`split_data.py`脚本在读取原始H5文件的元数据时，没有正确处理`variants/`组中的数据（如`variants/POS`、`variants/CHROM`等）。

原代码：
```python
for key in f.keys():
    if key != 'calldata':
        try:
            metadata[key] = f[key][:]  # 这只读取顶层，不读取组内的数据集
```

**问题**：`variants`是一个组（group），而不是数据集（dataset）。需要遍历组内的数据集。

---

## ✅ 修复

已更新`scripts/split_data.py`，正确读取H5文件的所有组和数据集：

```python
# 读取variants组（包含POS等）
metadata = {}
if 'variants' in f:
    for key in f['variants'].keys():
        try:
            metadata[f'variants/{key}'] = f[f'variants/{key}'][:]
            print(f"  - Read variants/{key}: shape={f[f'variants/{key}'].shape}")
        except Exception as e:
            print(f"  - Warning: Could not read variants/{key}: {e}")
```

---

## 🚀 重新运行

### 步骤1：拉取最新代码

```bash
cd /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_RAG-SNVBERT-packup
git pull origin main
```

---

### 步骤2：删除之前失败的划分数据

```bash
rm -rf data/train_val_split
```

---

### 步骤3：重新划分数据

```bash
python scripts/split_data.py \
    --input_h5 /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/maf_data/KGP.chr21.Train.maf01.vcf.h5 \
    --input_panel /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/VCF/Train/train.980.sample.panel \
    --output_dir data/train_val_split \
    --val_ratio 0.15
```

**现在你应该看到**：
```
Loading data...
  - Read variants/POS: shape=(150508,)
  - Read variants/CHROM: shape=(150508,)
  - Read variants/REF: shape=(150508,)
  - Read variants/ALT: shape=(150508, 3)
✓ Data loaded:
  - Variants: 150508
  - Samples: 980
  - Ploidy: 2
```

---

### 步骤4：验证划分后的文件

检查`variants/POS`是否存在：

```bash
python -c "
import h5py
with h5py.File('data/train_val_split/train_split.h5', 'r') as f:
    print('Keys in file:', list(f.keys()))
    if 'variants' in f:
        print('Keys in variants:', list(f['variants'].keys()))
        if 'POS' in f['variants']:
            print('✓ variants/POS exists, shape:', f['variants/POS'].shape)
        else:
            print('✗ variants/POS not found!')
    else:
        print('✗ variants group not found!')
"
```

**预期输出**：
```
Keys in file: ['calldata', 'variants']
Keys in variants: ['POS', 'CHROM', 'REF', 'ALT']
✓ variants/POS exists, shape: (150508,)
```

---

### 步骤5：开始训练

```bash
bash run_v12_split_val.sh
```

**现在应该能正常运行了！**

---

## 📊 完整的H5文件结构

正确的H5文件应该包含：

```
train_split.h5 (or val_split.h5)
├── calldata/
│   └── GT              # (n_variants, n_samples, 2)
└── variants/
    ├── POS             # (n_variants,) - 位点位置
    ├── CHROM           # (n_variants,) - 染色体
    ├── REF             # (n_variants,) - 参考等位基因
    └── ALT             # (n_variants, 3) - 替代等位基因
```

`TrainDataset.from_file()`需要：
- `calldata/GT` - 基因型数据
- `variants/POS` - 位点位置（用于窗口划分和RAG检索）

---

## ⚠️ 如果还是失败

### 检查原始H5文件结构

```bash
python -c "
import h5py
with h5py.File('/cpfs01/.../maf_data/KGP.chr21.Train.maf01.vcf.h5', 'r') as f:
    print('Top-level keys:', list(f.keys()))

    if 'calldata' in f:
        print('calldata keys:', list(f['calldata'].keys()))

    if 'variants' in f:
        print('variants keys:', list(f['variants'].keys()))
    else:
        print('⚠ No variants group found!')
        print('Checking for POS at top level...')
        if 'POS' in f:
            print('Found POS at top level')
"
```

如果原始文件没有`variants/`组，可能需要：

1. **情况A**：POS在顶层
   - 修改`dataset.py`读取`pos = vcf_h5['POS']`

2. **情况B**：文件格式不标准
   - 重新生成H5文件，确保包含variants组

---

## 🎯 总结

**问题**：`split_data.py`没有正确读取`variants/`组中的数据集

**修复**：更新脚本显式读取`variants/`组内的所有数据集

**操作**：
1. `git pull`拉取修复
2. 删除旧的划分数据
3. 重新运行`split_data.py`
4. 验证`variants/POS`存在
5. 开始训练

修复已提交到GitHub！🚀
