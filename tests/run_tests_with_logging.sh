#!/bin/bash

# FrozenQwen 测试运行脚本（日志输出版本）

echo "========================================"
echo "FrozenQwen 测试套件 - 日志输出版本"
echo "========================================"

# 激活环境
if [ ! -z "$CONDA_DEFAULT_ENV" ]; then
    echo "当前环境: $CONDA_DEFAULT_ENV"
else
    echo "警告：未检测到 conda 环境"
fi

echo ""
echo "开始运行测试..."
echo "----------------------------------------"

# 运行测试
python test_frozen_qwen_with_logging.py

# 检查退出状态
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✅ 测试完成"
    echo "========================================"
else
    echo ""
    echo "========================================"
    echo "❌ 测试执行出错"
    echo "========================================"
fi

# 显示日志文件位置
echo ""
echo "💾 日志文件已保存在当前目录"
echo "可以使用以下命令查看："
echo "  ls -lt test_results_*.log | head -1"
echo "  cat \$(ls -t test_results_*.log | head -1)"
echo ""

