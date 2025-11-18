#!/bin/bash

##############################################################################
# F-LMM 训练监控脚本
# 用于实时监控训练进度
# 用法: ./monitor_training.sh [日志文件路径]
##############################################################################

# 如果没有提供日志文件，查找最新的日志
if [ -z "$1" ]; then
    # 从 work_dirs 查找最新的日志
    LATEST_LOG=$(find work_dirs -name "*.log" -type f 2>/dev/null | sort -r | head -1)
    
    # 如果 work_dirs 没有，从 logs 目录找
    if [ -z "$LATEST_LOG" ]; then
        LATEST_LOG=$(find logs -name "*.log" -type f 2>/dev/null | sort -r | head -1)
    fi
    
    # 如果还没有，尝试当前目录
    if [ -z "$LATEST_LOG" ]; then
        LATEST_LOG=$(ls -t *.log 2>/dev/null | head -1)
    fi
    
    if [ -z "$LATEST_LOG" ]; then
        echo "错误: 未找到日志文件"
        echo "用法: $0 <日志文件路径>"
        echo "或者: $0   # 自动查找最新日志"
        exit 1
    fi
    
    LOG_FILE="$LATEST_LOG"
else
    LOG_FILE="$1"
fi

# 检查文件是否存在
if [ ! -f "$LOG_FILE" ]; then
    echo "错误: 日志文件不存在: $LOG_FILE"
    exit 1
fi

echo "=========================================="
echo "F-LMM 训练监控"
echo "=========================================="
echo "监控日志: $LOG_FILE"
echo "按 Ctrl+C 停止监控"
echo "=========================================="
echo ""

# 监控模式选择
echo "选择监控模式:"
echo "  1) 只显示关键信息（损失、学习率、步数等）- 推荐"
echo "  2) 显示完整日志"
echo "  3) 只看错误和警告"
echo "  4) 只看损失变化"
read -p "请选择 [1-4] (默认: 1): " MODE
MODE=${MODE:-1}

case $MODE in
    1)
        echo ""
        echo "监控关键信息..."
        echo "===================="
        tail -f "$LOG_FILE" | grep -E \
            "loss.*:|lr.*:|step|epoch|iter|save|checkpoint|accuracy|IoU|ERROR|WARNING" \
            --line-buffered | \
            sed -E 's/.*\[([0-9]+\/[0-9]+)\].*loss[^:]*: ([0-9.]+).*/Step \1 | Loss: \2/' \
            || true
        ;;
    2)
        echo ""
        echo "监控完整日志..."
        echo "===================="
        tail -f "$LOG_FILE"
        ;;
    3)
        echo ""
        echo "监控错误和警告..."
        echo "===================="
        tail -f "$LOG_FILE" | grep -E "ERROR|WARNING|Exception|Traceback" --line-buffered || true
        ;;
    4)
        echo ""
        echo "监控损失变化..."
        echo "===================="
        tail -f "$LOG_FILE" | grep -E "loss.*:" --line-buffered | \
            sed -E 's/.*loss[^:]*: ([0-9.]+).*/Loss: \1/' || true
        ;;
    *)
        echo "无效选择，使用默认模式（关键信息）"
        tail -f "$LOG_FILE" | grep -E "loss|lr|step|epoch|save" --line-buffered || true
        ;;
esac

