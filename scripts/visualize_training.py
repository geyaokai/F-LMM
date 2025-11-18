#!/usr/bin/env python3
"""
训练日志可视化脚本
从训练日志文件中提取指标并生成可视化图表
"""

import re
import argparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import numpy as np
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def parse_log_file(log_file):
    """解析日志文件，提取训练指标"""
    metrics = {
        'iter': [],
        'loss': [],
        'loss_mask': [],
        'loss_dice': [],
        'accuracy': [],
        'aiou': [],
        'sam_loss_mask': [],
        'sam_loss_dice': [],
        'sam_accuracy': [],
        'sam_aiou': [],
        'lr': [],
        'memory': [],
        'time': [],
    }
    
    # 匹配训练迭代行的正则表达式
    pattern = r'Iter\(train\)\s+\[\s*(\d+)/(\d+)\]\s+lr:\s+([\d.]+[eE]?[+-]?\d*)\s+eta:.*?time:\s+([\d.]+)\s+data_time:\s+([\d.]+)\s+memory:\s+(\d+)\s+loss:\s+([\d.]+)\s+loss_mask:\s+([\d.]+)\s+loss_dice:\s+([\d.]+)\s+accuracy:\s+([\d.]+)\s+aiou:\s+([\d.]+)\s+sam_loss_mask:\s+([\d.]+)\s+sam_loss_dice:\s+([\d.]+)\s+sam_accuracy:\s+([\d.]+)\s+sam_aiou:\s+([\d.]+)'
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                iter_num = int(match.group(1))
                total_iters = int(match.group(2))
                lr = float(match.group(3))
                time = float(match.group(4))
                data_time = float(match.group(5))
                memory = int(match.group(6))
                loss = float(match.group(7))
                loss_mask = float(match.group(8))
                loss_dice = float(match.group(9))
                accuracy = float(match.group(10))
                aiou = float(match.group(11))
                sam_loss_mask = float(match.group(12))
                sam_loss_dice = float(match.group(13))
                sam_accuracy = float(match.group(14))
                sam_aiou = float(match.group(15))
                
                metrics['iter'].append(iter_num)
                metrics['loss'].append(loss)
                metrics['loss_mask'].append(loss_mask)
                metrics['loss_dice'].append(loss_dice)
                metrics['accuracy'].append(accuracy)
                metrics['aiou'].append(aiou)
                metrics['sam_loss_mask'].append(sam_loss_mask)
                metrics['sam_loss_dice'].append(sam_loss_dice)
                metrics['sam_accuracy'].append(sam_accuracy)
                metrics['sam_aiou'].append(sam_aiou)
                metrics['lr'].append(lr)
                metrics['memory'].append(memory)
                metrics['time'].append(time)
    
    return metrics


def plot_training_curves(metrics, output_dir):
    """绘制训练曲线"""
    if not metrics['iter']:
        print("警告: 未找到训练数据，请检查日志文件格式")
        return
    
    iterations = np.array(metrics['iter'])
    
    # 创建图表
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Loss 曲线
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(iterations, metrics['loss'], 'b-', label='Total Loss', linewidth=1.5)
    ax1.plot(iterations, metrics['loss_mask'], 'g--', label='Mask Loss', linewidth=1)
    ax1.plot(iterations, metrics['loss_dice'], 'r--', label='Dice Loss', linewidth=1)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. SAM Loss 曲线
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(iterations, metrics['sam_loss_mask'], 'g-', label='SAM Mask Loss', linewidth=1.5)
    ax2.plot(iterations, metrics['sam_loss_dice'], 'r-', label='SAM Dice Loss', linewidth=1.5)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_title('SAM Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Accuracy 曲线
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(iterations, metrics['accuracy'], 'b-', label='Accuracy', linewidth=1.5)
    ax3.plot(iterations, metrics['sam_accuracy'], 'g-', label='SAM Accuracy', linewidth=1.5)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Accuracy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    # 4. IoU 曲线
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(iterations, metrics['aiou'], 'b-', label='IoU', linewidth=1.5)
    ax4.plot(iterations, metrics['sam_aiou'], 'g-', label='SAM IoU', linewidth=1.5)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('IoU')
    ax4.set_title('Intersection over Union (IoU)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])
    
    # 5. Learning Rate 曲线
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(iterations, metrics['lr'], 'purple', linewidth=1.5)
    ax5.set_xlabel('Iteration')
    ax5.set_ylabel('Learning Rate')
    ax5.set_title('Learning Rate Schedule')
    ax5.grid(True, alpha=0.3)
    ax5.set_yscale('log')
    
    # 6. Memory 使用曲线
    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(iterations, np.array(metrics['memory']) / 1024, 'orange', linewidth=1.5)
    ax6.set_xlabel('Iteration')
    ax6.set_ylabel('Memory (GB)')
    ax6.set_title('GPU Memory Usage')
    ax6.grid(True, alpha=0.3)
    
    # 7. Time per Iteration
    ax7 = plt.subplot(3, 3, 7)
    ax7.plot(iterations, metrics['time'], 'brown', linewidth=1.5)
    ax7.set_xlabel('Iteration')
    ax7.set_ylabel('Time (seconds)')
    ax7.set_title('Time per Iteration')
    ax7.grid(True, alpha=0.3)
    
    # 8. Loss 对比（UNet vs SAM）
    ax8 = plt.subplot(3, 3, 8)
    total_unet_loss = np.array(metrics['loss_mask']) + np.array(metrics['loss_dice'])
    total_sam_loss = np.array(metrics['sam_loss_mask']) + np.array(metrics['sam_loss_dice'])
    ax8.plot(iterations, total_unet_loss, 'b-', label='UNet Total Loss', linewidth=1.5)
    ax8.plot(iterations, total_sam_loss, 'g-', label='SAM Total Loss', linewidth=1.5)
    ax8.set_xlabel('Iteration')
    ax8.set_ylabel('Loss')
    ax8.set_title('UNet vs SAM Loss Comparison')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Accuracy & IoU 综合
    ax9 = plt.subplot(3, 3, 9)
    ax9_twin = ax9.twinx()
    line1 = ax9.plot(iterations, metrics['sam_accuracy'], 'b-', label='SAM Accuracy', linewidth=1.5)
    line2 = ax9_twin.plot(iterations, metrics['sam_aiou'], 'r-', label='SAM IoU', linewidth=1.5)
    ax9.set_xlabel('Iteration')
    ax9.set_ylabel('Accuracy', color='b')
    ax9_twin.set_ylabel('IoU', color='r')
    ax9.set_title('SAM Accuracy & IoU')
    ax9.tick_params(axis='y', labelcolor='b')
    ax9_twin.tick_params(axis='y', labelcolor='r')
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax9.legend(lines, labels, loc='upper left')
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    output_file = Path(output_dir) / 'training_curves.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ 训练曲线已保存到: {output_file}")
    
    # 保存统计信息
    if metrics['iter']:
        stats_file = Path(output_dir) / 'training_stats.txt'
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("训练统计信息\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"总迭代次数: {len(metrics['iter'])}\n")
            f.write(f"迭代范围: {min(metrics['iter'])} - {max(metrics['iter'])}\n\n")
            
            f.write("Loss 统计:\n")
            f.write(f"  总 Loss - 最小值: {min(metrics['loss']):.4f}, 最大值: {max(metrics['loss']):.4f}, 平均值: {np.mean(metrics['loss']):.4f}\n")
            f.write(f"  Mask Loss - 最小值: {min(metrics['loss_mask']):.4f}, 最大值: {max(metrics['loss_mask']):.4f}, 平均值: {np.mean(metrics['loss_mask']):.4f}\n")
            f.write(f"  Dice Loss - 最小值: {min(metrics['loss_dice']):.4f}, 最大值: {max(metrics['loss_dice']):.4f}, 平均值: {np.mean(metrics['loss_dice']):.4f}\n\n")
            
            f.write("SAM Loss 统计:\n")
            f.write(f"  SAM Mask Loss - 最小值: {min(metrics['sam_loss_mask']):.4f}, 最大值: {max(metrics['sam_loss_mask']):.4f}, 平均值: {np.mean(metrics['sam_loss_mask']):.4f}\n")
            f.write(f"  SAM Dice Loss - 最小值: {min(metrics['sam_loss_dice']):.4f}, 最大值: {max(metrics['sam_loss_dice']):.4f}, 平均值: {np.mean(metrics['sam_loss_dice']):.4f}\n\n")
            
            f.write("Accuracy 统计:\n")
            f.write(f"  Accuracy - 最小值: {min(metrics['accuracy']):.4f}, 最大值: {max(metrics['accuracy']):.4f}, 平均值: {np.mean(metrics['accuracy']):.4f}\n")
            f.write(f"  SAM Accuracy - 最小值: {min(metrics['sam_accuracy']):.4f}, 最大值: {max(metrics['sam_accuracy']):.4f}, 平均值: {np.mean(metrics['sam_accuracy']):.4f}\n\n")
            
            f.write("IoU 统计:\n")
            f.write(f"  IoU - 最小值: {min(metrics['aiou']):.4f}, 最大值: {max(metrics['aiou']):.4f}, 平均值: {np.mean(metrics['aiou']):.4f}\n")
            f.write(f"  SAM IoU - 最小值: {min(metrics['sam_aiou']):.4f}, 最大值: {max(metrics['sam_aiou']):.4f}, 平均值: {np.mean(metrics['sam_aiou']):.4f}\n\n")
            
            f.write("性能统计:\n")
            f.write(f"  平均每次迭代时间: {np.mean(metrics['time']):.2f} 秒\n")
            f.write(f"  平均 GPU 内存使用: {np.mean(metrics['memory']) / 1024:.2f} GB\n")
            f.write(f"  学习率范围: {min(metrics['lr']):.2e} - {max(metrics['lr']):.2e}\n")
        
        print(f"✅ 统计信息已保存到: {stats_file}")


def main():
    parser = argparse.ArgumentParser(description='可视化训练日志')
    parser.add_argument('log_file', type=str, help='训练日志文件路径')
    parser.add_argument('--output-dir', type=str, default=None, 
                       help='输出目录（默认：日志文件所在目录）')
    
    args = parser.parse_args()
    
    log_file = Path(args.log_file)
    if not log_file.exists():
        print(f"错误: 日志文件不存在: {log_file}")
        return
    
    # 确定输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = log_file.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"正在解析日志文件: {log_file}")
    metrics = parse_log_file(log_file)
    
    if not metrics['iter']:
        print("错误: 未能从日志文件中提取训练数据")
        print("请确保日志文件包含格式为 'Iter(train) [xxx/xxx] ...' 的训练记录")
        return
    
    print(f"✅ 成功提取 {len(metrics['iter'])} 条训练记录")
    print(f"迭代范围: {min(metrics['iter'])} - {max(metrics['iter'])}")
    
    print("正在生成可视化图表...")
    plot_training_curves(metrics, output_dir)
    
    print("\n完成！")


if __name__ == '__main__':
    main()

