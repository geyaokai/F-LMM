#!/bin/bash
# 运行数据管道验证脚本

source ~/miniconda3/etc/profile.d/conda.sh
conda activate flmm-qwen-py310

cd "$(dirname "$0")"
python verify_data_pipeline.py

