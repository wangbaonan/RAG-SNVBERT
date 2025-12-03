#!/bin/bash

# ============================================
# V17 一步步部署脚本
# 用户可以按照这个脚本一步步执行
# ============================================

echo "============================================"
echo "V17 RAG 正确部署 - 分步执行指南"
echo "============================================"
echo ""
echo "重要发现: V17 RAG的Query mask必须与Index mask一致!"
echo "修复: 训练集必须用静态mask (use_dynamic_mask=False)"
echo ""
echo "请按照以下步骤执行:"
echo "============================================"
echo ""

# ============================================
# Step 1: 进入正确目录
# ============================================
echo "Step 1: 进入项目目录"
echo "----------------------------------------"
echo "命令:"
echo "  cd /e/AI4S/00_SNVBERT/VCF-Bert"
echo ""
read -p "请执行上述命令后按Enter继续..."
echo ""

# ============================================
# Step 2: 检查修复是否已应用
# ============================================
echo "Step 2: 检查修复是否已应用"
echo "----------------------------------------"
echo "命令:"
echo "  grep -n 'use_dynamic_mask' src/train_with_val_optimized.py"
echo ""
echo "预期输出:"
echo "  122:        use_dynamic_mask=False  # 训练集 ← 应该是False!"
echo "  153:        use_dynamic_mask=True   # 验证集 ← 可以是True"
echo ""
read -p "请执行上述命令，确认Line 122是False后按Enter继续..."
echo ""

# ============================================
# Step 3: (可选) 如果修复未应用
# ============================================
echo "Step 3: 如果Line 122不是False，需要手动修改"
echo "----------------------------------------"
echo "命令:"
echo "  # 方法1: 使用sed修改"
echo "  sed -i '122s/use_dynamic_mask=True/use_dynamic_mask=False/' src/train_with_val_optimized.py"
echo ""
echo "  # 或方法2: 手动编辑"
echo "  vi src/train_with_val_optimized.py"
echo "  # 找到Line 122，改为: use_dynamic_mask=False"
echo ""
read -p "如果需要修改，请执行后按Enter继续，否则直接按Enter..."
echo ""

# ============================================
# Step 4: 再次确认修改
# ============================================
echo "Step 4: 再次确认修改"
echo "----------------------------------------"
echo "命令:"
echo "  grep -A 1 -B 1 'use_dynamic_mask=False' src/train_with_val_optimized.py | head -5"
echo ""
echo "预期看到:"
echo "  n_gpu=1,"
echo "  use_dynamic_mask=False  # 重要! 训练集必须用静态mask"
echo "  )"
echo ""
read -p "请执行并确认后按Enter继续..."
echo ""

# ============================================
# Step 5: 检查GPU状态
# ============================================
echo "Step 5: 检查GPU可用性"
echo "----------------------------------------"
echo "命令:"
echo "  nvidia-smi"
echo ""
echo "确认:"
echo "  - 至少有20GB空闲内存"
echo "  - GPU利用率不是100% (没有其他训练)"
echo ""
read -p "请执行并确认后按Enter继续..."
echo ""

# ============================================
# Step 6: (可选) 备份之前的输出
# ============================================
echo "Step 6: (可选) 备份之前的训练输出"
echo "----------------------------------------"
echo "如果之前有崩溃的训练，建议备份:"
echo ""
echo "命令:"
echo "  # 备份输出目录"
echo "  mv /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/output_v17_memfix \\"
echo "     /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/output_v17_memfix_backup"
echo ""
echo "  # 备份日志"
echo "  cp -r logs/v17_extreme_memfix logs/v17_extreme_memfix_backup"
echo ""
read -p "如果需要备份，请执行后按Enter继续，否则直接按Enter..."
echo ""

# ============================================
# Step 7: 运行训练
# ============================================
echo "Step 7: 启动训练"
echo "----------------------------------------"
echo "命令:"
echo "  bash run_v17_extreme_memory_fix.sh"
echo ""
echo "预期:"
echo "  - 开始构建FAISS索引 (~5分钟)"
echo "  - 开始训练"
echo "  - 每个epoch约4-5小时"
echo ""
read -p "准备好后按Enter启动训练..."
echo ""

# 实际启动训练
echo "正在启动训练..."
echo "============================================"
bash run_v17_extreme_memory_fix.sh

# 训练完成后
echo ""
echo "============================================"
echo "训练已结束"
echo "============================================"
echo ""
echo "查看结果:"
echo "  - 日志: tail logs/v17_extreme_memfix/latest.log"
echo "  - 指标: cat metrics/v17_extreme_memfix/latest.csv"
echo "  - 模型: ls output_v17_memfix/"
echo ""
