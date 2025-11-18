#!/bin/bash

##############################################################################
# F-LMM 训练脚本
# 用法: ./train.sh [选项]
##############################################################################

set -e  # 遇到错误立即退出

# 默认配置
CONFIG_FILE="configs/qwen/frozen_qwen2_5_vl_3b_instruct_unet_sam_l_refcoco_png.py"
NUM_GPUS=2
DEEPSPEED="deepspeed_zero2"
LOG_DIR="logs"
SHOW_FILTERED=true  # 是否只显示过滤后的关键信息
FULL_LOG=false      # 是否显示完整日志
RESUME=false
RESUME_FROM=""
WORK_DIR_OVERRIDE=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --deepspeed)
            DEEPSPEED="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --work-dir)
            WORK_DIR_OVERRIDE="$2"
            shift 2
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --resume-from)
            RESUME_FROM="$2"
            RESUME=true
            shift 2
            ;;
        --full-log)
            SHOW_FILTERED=false
            FULL_LOG=true
            shift
            ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --config PATH        配置文件路径 (默认: $CONFIG_FILE)"
            echo "  --gpus N             使用的 GPU 数量 (默认: $NUM_GPUS)"
            echo "  --deepspeed STR      DeepSpeed 策略 (默认: $DEEPSPEED)"
            echo "  --log-dir DIR        日志保存目录 (默认: $LOG_DIR)"
            echo "  --work-dir DIR       指定 work_dir（默认: work_dirs/<config_name>）"
            echo "  --resume             开启断点续训，自动寻找最新 checkpoint"
            echo "  --resume-from PATH   指定 checkpoint 文件恢复"
            echo "  --full-log           显示完整日志（不过滤）"
            echo "  --help, -h           显示帮助信息"
            echo ""
            echo "示例:"
            echo "  $0                                                    # 使用默认 Qwen 配置"
            echo "  $0 --gpus 4 --config configs/llava/...py             # 自定义配置与 GPU"
            echo "  $0 --resume --resume-from work_dirs/.../epoch_004.pth # 明确恢复路径"
            echo "  $0 --resume                                           # 自动查找最近 checkpoint"
            echo "  $0 --full-log                                        # 显示完整日志"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 创建日志目录
mkdir -p "$LOG_DIR"

# 生成日志文件名（包含时间戳）
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CONFIG_NAME=$(basename "$CONFIG_FILE" .py)
LOG_FILE="$LOG_DIR/${CONFIG_NAME}_${TIMESTAMP}.log"

# 推断 work_dir
if [ -n "$WORK_DIR_OVERRIDE" ]; then
    WORK_DIR="$WORK_DIR_OVERRIDE"
else
    WORK_DIR="work_dirs/$CONFIG_NAME"
fi

# 自动定位 checkpoint
if [ "$RESUME" = true ] && [ -z "$RESUME_FROM" ]; then
    LAST_CKPT_FILE="$WORK_DIR/last_checkpoint"
    if [ -f "$LAST_CKPT_FILE" ]; then
        RESUME_FROM=$(tr -d '\r\n' < "$LAST_CKPT_FILE")
    fi
    if [ -z "$RESUME_FROM" ]; then
        # 找最新的 epoch 或 iter checkpoint
        CANDIDATE=$(ls -1t "$WORK_DIR"/epoch_*.pth "$WORK_DIR"/iter_*.pth 2>/dev/null | head -n 1)
        if [ -z "$CANDIDATE" ]; then
            LATEST_RUN=$(ls -1dt "$WORK_DIR"/*/ 2>/dev/null | head -n 1)
            if [ -n "$LATEST_RUN" ]; then
                if [ -f "${LATEST_RUN}last_checkpoint" ]; then
                    RESUME_FROM=$(tr -d '\r\n' < "${LATEST_RUN}last_checkpoint")
                else
                    CANDIDATE=$(ls -1t "${LATEST_RUN}"/epoch_*.pth "${LATEST_RUN}"/iter_*.pth 2>/dev/null | head -n 1)
                    RESUME_FROM="$CANDIDATE"
                fi
            fi
        else
            RESUME_FROM="$CANDIDATE"
        fi
    fi
    if [ -z "$RESUME_FROM" ]; then
        echo "WARNING: 未找到可用的 checkpoint，自动续训将被跳过。"
        RESUME=false
    else
        echo "INFO: 自动检测到 checkpoint: $RESUME_FROM"
    fi
fi

# 设置环境变量
export PYTHONPATH=.
export NPROC_PER_NODE=$NUM_GPUS

echo "=========================================="
echo "F-LMM 训练脚本"
echo "=========================================="
echo "配置文件: $CONFIG_FILE"
echo "GPU 数量: $NUM_GPUS"
echo "DeepSpeed: $DEEPSPEED"
echo "日志文件: $LOG_FILE"
if [ "$RESUME" = true ]; then
    echo "断点续训: 已开启"
fi
if [ -n "$RESUME_FROM" ]; then
    echo "恢复路径: $RESUME_FROM"
fi
echo "=========================================="
echo ""

# 检查是否有 GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "警告: 未检测到 nvidia-smi，可能没有 GPU"
else
    echo "检测到的 GPU:"
    nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader | head -n "$NUM_GPUS"
    echo ""
fi

# 提示信息
if [ "$SHOW_FILTERED" = true ]; then
    echo "训练将启动，终端只显示关键信息（损失、学习率、步数等）"
    echo "完整日志保存在: $LOG_FILE"
    echo ""
    echo "在另一个终端可以使用以下命令监控训练:"
    echo "  tail -f $LOG_FILE                              # 查看完整日志"
    echo "  tail -f $LOG_FILE | grep -E 'loss|lr|step'    # 只看关键信息"
    echo ""
else
    echo "训练将启动，显示完整日志"
    echo "日志同时保存在: $LOG_FILE"
    echo ""
fi

# 构建训练命令
CMD=(xtuner train "$CONFIG_FILE" --deepspeed "$DEEPSPEED")
if [ -n "$WORK_DIR_OVERRIDE" ]; then
    CMD+=(--work-dir "$WORK_DIR_OVERRIDE")
fi
if [ "$RESUME" = true ] && [ -n "$RESUME_FROM" ]; then
    CMD+=(--resume "$RESUME_FROM")
fi

# 根据显示模式执行训练
if [ "$SHOW_FILTERED" = true ]; then
    echo "开始训练..."
    echo ""
    "${CMD[@]}" 2>&1 | tee "$LOG_FILE" | grep -E \
        "loss|lr|step|epoch|iter|save|checkpoint|accuracy|IoU|ERROR|WARNING|Starting|Finished" \
        --line-buffered || true
else
    echo "开始训练..."
    echo ""
    "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
fi

echo ""
echo "=========================================="
echo "训练完成或中断"
echo "日志文件: $LOG_FILE"
echo "=========================================="
