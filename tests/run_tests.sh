#!/bin/bash

# FrozenQwen 测试运行脚本

echo "========================================"
echo "FrozenQwen 单元测试和诊断"
echo "========================================"

# 激活环境
if [ ! -z "$CONDA_DEFAULT_ENV" ]; then
    echo "当前环境: $CONDA_DEFAULT_ENV"
else
    echo "警告：未检测到 conda 环境"
fi

echo ""
echo "步骤 1: 运行快速诊断"
echo "----------------------------------------"
python diagnose_image_grid_thw.py

echo ""
echo "步骤 2: 运行单元测试"
echo "----------------------------------------"
python test_frozen_qwen.py

echo ""
echo "========================================"
echo "测试完成"
echo "========================================"

