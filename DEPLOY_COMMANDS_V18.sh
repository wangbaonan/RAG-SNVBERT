#!/bin/bash

# ============================================================
# V18 Mask对齐版本 - 一键部署脚本
# ============================================================

echo "============================================"
echo "V18 Mask对齐版本 - 部署指南"
echo "============================================"
echo ""
echo "请按照以下步骤逐步执行:"
echo "============================================"
echo ""

# ============================================
# Step 1: Pull最新代码
# ============================================
echo "Step 1: Pull最新代码"
echo "----------------------------------------"
echo "命令:"
echo "  cd /path/to/VCF-Bert  # 替换为您的实际路径"
echo "  git status"
echo "  git stash  # 如果有未提交的修改"
echo "  git pull origin main"
echo "  git stash pop  # 恢复之前的修改"
echo ""
read -p "请执行上述命令后按Enter继续..."
echo ""

# ============================================
# Step 2: 验证文件完整性
# ============================================
echo "Step 2: 验证文件完整性"
echo "----------------------------------------"
echo "命令:"
echo "  grep 'ref_embeddings_complete' src/dataset/embedding_rag_dataset.py"
echo "  grep 'regenerate_masks' src/dataset/embedding_rag_dataset.py"
echo "  grep 'refresh_complete_embeddings' src/train_embedding_rag.py"
echo ""
echo "预期: 所有命令都应该找到匹配"
echo ""
read -p "请执行并确认后按Enter继续..."
echo ""

# ============================================
# Step 3: 检查环境
# ============================================
echo "Step 3: 检查环境"
echo "----------------------------------------"
echo "命令:"
echo "  nvidia-smi"
echo ""
echo "确认:"
echo "  - 至少20GB空闲显存"
echo "  - GPU利用率不是100%"
echo ""
read -p "请执行并确认后按Enter继续..."
echo ""

# ============================================
# Step 4: 检查数据文件
# ============================================
echo "Step 4: 检查数据文件"
echo "----------------------------------------"
echo "命令:"
echo "  DATA_DIR=/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data"
echo "  ls -lh \$DATA_DIR/train_split.h5"
echo "  ls -lh \$DATA_DIR/val_split.h5"
echo "  ls -lh \$DATA_DIR/KGP.chr21.Panel.maf01.vcf.gz"
echo "  ls -lh \$DATA_DIR/Freq.npy"
echo ""
echo "确认: 所有文件都存在"
echo ""
read -p "请执行并确认后按Enter继续..."
echo ""

# ============================================
# Step 5: 运行训练 (选择方式)
# ============================================
echo "Step 5: 运行训练"
echo "----------------------------------------"
echo "请选择运行方式:"
echo ""
echo "方式1: 前台运行 (推荐先测试)"
echo "  bash run_v18_embedding_rag.sh"
echo ""
echo "方式2: 后台运行"
echo "  nohup bash run_v18_embedding_rag.sh > v18_mask_aligned.log 2>&1 &"
echo "  echo \$! > v18_train.pid"
echo ""
echo "方式3: 指定GPU"
echo "  CUDA_VISIBLE_DEVICES=0 bash run_v18_embedding_rag.sh"
echo ""
read -p "请选择并执行后按Enter继续..."
echo ""

# ============================================
# Step 6: 监控命令
# ============================================
echo "Step 6: 监控训练"
echo "----------------------------------------"
echo "监控命令:"
echo ""
echo "  # 实时日志"
echo "  tail -f logs/v18_embedding_rag/latest.log"
echo ""
echo "  # GPU监控"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "  # 指标监控"
echo "  watch -n 10 \"tail -10 metrics/v18_embedding_rag/latest.csv\""
echo ""
echo "  # 查看进程"
echo "  ps aux | grep train_embedding_rag"
echo ""
echo "  # 如果是后台运行，查看日志"
echo "  tail -f v18_mask_aligned.log"
echo ""
read -p "按Enter查看关键监控指标..."
echo ""

# ============================================
# Step 7: 关键监控指标
# ============================================
echo "Step 7: 关键监控指标"
echo "----------------------------------------"
echo ""
echo "初始化阶段 (~18分钟):"
echo "  ✓ 预编码完成!"
echo "  - Mask版本号: 0"
echo "  - 存储大小: 1486.4 MB (两套embeddings)"
echo ""
echo "Epoch 1 (~1.5小时):"
echo "  Train F1: 0.9201"
echo "  Val F1: 0.9505"
echo "  ✓ Complete刷新完成! 耗时: 495s"
echo ""
echo "Epoch 2+ (~1.8小时):"
echo "  ▣ 刷新Mask Pattern (版本 1, Seed=2)"
echo "  ✓ Mask刷新完成! 新版本: 1"
echo "  ✓ 索引重建完成! 耗时: 492s"
echo "  [正常训练...]"
echo "  ✓ Complete刷新完成!"
echo ""
echo "异常标志:"
echo "  ❌ Mask版本号不递增 → 检查regenerate_masks是否被调用"
echo "  ❌ 存储大小只有743MB → 只有一套embeddings，代码未更新"
echo "  ❌ OOM → batch size太大，改为8"
echo "  ❌ Train F1虚高(0.978) → mask未刷新，过拟合"
echo ""
read -p "按Enter查看故障排查..."
echo ""

# ============================================
# Step 8: 常见问题排查
# ============================================
echo "Step 8: 常见问题排查"
echo "----------------------------------------"
echo ""
echo "问题1: Pull后代码未更新"
echo "  检查: git log -1"
echo "  解决: git pull --force origin main"
echo ""
echo "问题2: OOM"
echo "  检查: nvidia-smi"
echo "  解决: 编辑run_v18_embedding_rag.sh，改batch_size=8"
echo ""
echo "问题3: Mask版本不递增"
echo "  检查: grep 'regenerate_masks' src/train_embedding_rag.py"
echo "  解决: 确认代码已更新"
echo ""
echo "问题4: 存储大小不对"
echo "  检查: 初始化日志中的'存储大小'"
echo "  预期: 1486.4 MB (两套embeddings)"
echo "  如果是743MB: 代码未更新"
echo ""
echo "问题5: AttributeError: 'ref_embeddings_complete'"
echo "  原因: pull的代码不完整"
echo "  解决: 重新pull或检查分支"
echo ""
read -p "按Enter完成..."
echo ""

echo "============================================"
echo "部署完成！"
echo "============================================"
echo ""
echo "下一步:"
echo "  1. 监控前2个epoch确认正常"
echo "  2. 检查Mask版本号递增"
echo "  3. 检查性能稳定 (Val F1 ~0.95)"
echo ""
echo "预期训练时间: ~32小时 (20 epochs)"
echo ""
echo "祝训练顺利！🚀"
echo ""
