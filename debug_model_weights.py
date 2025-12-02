#!/usr/bin/env python
"""
诊断脚本：检查不同epoch的模型权重是否真的不同
"""

import torch
import sys

def check_model_differences(model_paths):
    """比较多个模型文件的权重"""

    print(f"\n{'='*70}")
    print(f"Model Weight Comparison")
    print(f"{'='*70}\n")

    models = []
    for path in model_paths:
        try:
            print(f"Loading: {path}")
            model = torch.load(path, map_location='cpu')
            models.append((path, model))
            print(f"✓ Loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load: {e}")
            continue

    if len(models) < 2:
        print("\n✗ Need at least 2 models to compare!")
        return

    print(f"\n{'='*70}")
    print(f"Comparing Models")
    print(f"{'='*70}\n")

    # 获取第一个模型的参数作为基准
    base_path, base_model = models[0]
    base_params = list(base_model.parameters())

    print(f"Base model: {base_path}")
    print(f"Total parameters: {len(base_params)}")

    # 比较每个模型与基准模型
    for i, (path, model) in enumerate(models[1:], 1):
        print(f"\n--- Comparing with: {path} ---")

        params = list(model.parameters())

        if len(params) != len(base_params):
            print(f"✗ Different number of parameters! {len(params)} vs {len(base_params)}")
            continue

        # 比较每一层
        all_same = True
        diff_count = 0

        for j, (base_p, p) in enumerate(zip(base_params, params)):
            if not torch.allclose(base_p, p, rtol=1e-6, atol=1e-8):
                all_same = False
                diff_count += 1

                # 计算差异统计
                diff = (p - base_p).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()

                if diff_count <= 3:  # 只显示前3个不同的层
                    print(f"  Layer {j}: DIFFERENT")
                    print(f"    - Shape: {base_p.shape}")
                    print(f"    - Base mean: {base_p.mean().item():.6f}")
                    print(f"    - Current mean: {p.mean().item():.6f}")
                    print(f"    - Max diff: {max_diff:.6e}")
                    print(f"    - Mean diff: {mean_diff:.6e}")

        if all_same:
            print(f"  ⚠️  WARNING: ALL PARAMETERS ARE IDENTICAL!")
            print(f"  This model is exactly the same as the base model!")
        else:
            print(f"\n  ✓ Models are different")
            print(f"  Total layers with differences: {diff_count}/{len(base_params)}")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python debug_model_weights.py <model1.pth> <model2.pth> [model3.pth ...]")
        print("\nExample:")
        print("  python debug_model_weights.py output_rag/bert.model.ep0.pth output_rag/bert.model.ep5.pth")
        sys.exit(1)

    model_paths = sys.argv[1:]
    check_model_differences(model_paths)
