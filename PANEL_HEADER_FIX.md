# Panel Header错误修复

## 🐛 问题

训练时出现KeyError：

```
KeyError: 'super_pop'
```

在`dataset.py:524`尝试访问`self.pop_to_idx[pop]`时失败。

---

## 🔍 原因

**问题1：Panel文件包含header**

你的原始panel文件格式：
```
sample    pop    super_pop
NA12878   EUR    super_pop
NA12879   EUR    super_pop
...
```

第一行是header（列名），不是实际数据。

**问题2：split_data.py的header检测逻辑错误**

原代码：
```python
has_header = not lines[0].split()[0].startswith('sample') and not lines[0][0].isdigit()
```

这个逻辑是**反的**！如果第一列是'sample'，应该判断为**有**header，但代码判断为**无**header。

**结果**：
1. Header行被当作数据保存到了`train_panel.txt`
2. 训练时读取panel，把`super_pop`当作population ID
3. `pop_to_idx`映射文件中没有`super_pop`这个键
4. KeyError！

---

## ✅ 修复

已更新`scripts/split_data.py`，修复header检测逻辑：

```python
# 检测第一行是否是header
first_line_lower = lines[0].lower().strip()
has_header = ('sample' in first_line_lower or
              'pop' in first_line_lower or
              'super_pop' in first_line_lower or
              lines[0].startswith('#'))

if has_header:
    header = lines[0]
    samples = lines[1:]         # 跳过header
    print(f"  - Detected header: {header.strip()}")
else:
    header = None
    samples = lines             # 所有行都是数据
    print(f"  - No header detected")
```

**修复逻辑**：
- 检查第一行是否包含关键词（'sample', 'pop', 'super_pop', '#'）
- 如果包含 → 是header → 跳过第一行
- 如果不包含 → 不是header → 使用所有行

---

## 🚀 重新运行

### 步骤1：拉取最新修复

```bash
cd /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_RAG-SNVBERT-packup

git pull origin main
```

---

### 步骤2：删除错误的划分数据

```bash
rm -rf data/train_val_split
```

---

### 步骤3：重新划分

```bash
python scripts/split_data.py \
    --input_h5 /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/maf_data/KGP.chr21.Train.maf01.vcf.h5 \
    --input_panel /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/VCF/Train/train.980.sample.panel \
    --output_dir data/train_val_split \
    --val_ratio 0.15
```

**现在你会看到**：
```
Splitting panel file...
  - Detected header: sample    pop    super_pop   ← 检测到header了！
  - Total samples in panel: 980                   ← 不包含header行
✓ Train panel saved: data/train_val_split/train_panel.txt (833 samples)
✓ Val panel saved: data/train_val_split/val_panel.txt (147 samples)
```

---

### 步骤4：验证panel文件

检查生成的panel文件第一行：

```bash
head -3 data/train_val_split/train_panel.txt
```

**正确输出**（包含header）：
```
sample    pop    super_pop
NA12878   EUR    super_pop
NA12879   EUR    super_pop
```

或者如果原始文件没有header，**正确输出**（不包含header）：
```
NA12878   EUR
NA12879   EUR
NA12880   EUR
```

**❌ 错误输出**（之前的bug）：
```
super_pop             ← 这是header的最后一列被当作数据了！
NA12878   EUR    ...
```

---

### 步骤5：开始训练

```bash
bash run_v12_split_val.sh
```

**现在应该能正常运行了！**

---

## 🔍 深入理解

### Panel文件的两种格式

#### 格式1：有header

```
sample      pop    super_pop
HG00096     GBR    EUR
HG00097     GBR    EUR
NA12878     CEU    EUR
```

- 第1行：header（列名）
- 第2行起：数据
- 训练时使用第2列（pop）作为population ID

#### 格式2：无header

```
HG00096     GBR
HG00097     GBR
NA12878     CEU
```

- 所有行都是数据
- 第2列是population ID

### pop_to_idx映射

`pop_to_idx.bin`文件包含从population名称到索引的映射：

```python
{
    'EUR': 0,
    'AFR': 1,
    'EAS': 2,
    'SAS': 3,
    'AMR': 4,
    'GBR': 5,
    'CEU': 6,
    ...
}
```

**不包含**：
- ❌ `'sample'` （header列名）
- ❌ `'pop'` （header列名）
- ❌ `'super_pop'` （header列名）

所以如果把header当作数据，就会KeyError。

---

## 🎯 总结

**问题**：split_data.py错误地把panel header当作数据

**原因**：header检测逻辑写反了

**修复**：正确检测header（检查关键词）

**操作**：
1. `git pull`拉取修复
2. 删除旧的`data/train_val_split`
3. 重新运行`split_data.py`
4. 验证panel文件第一行
5. 开始训练

修复已提交！🚀
