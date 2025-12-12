#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最优传输（Optimal Transport）知识分类 - 知识点1：引言与背景

本文件对应知识点1的内容，主要介绍最优传输的背景和基本概念。
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

def gan_art_introduction():
    """
    介绍GAN生成艺术的案例
    - GAN生成的艺术作品在佳士得拍卖
    - 预期价格：$7,000-$10,000
    - 实际价格：$432,500
    """
    print("="*60)
    print("知识点1：引言与背景")
    print("="*60)
    print("\n1. Beauty lies in the eyes of the discriminator")
    print("- GAN生成的艺术作品在佳士得拍卖")
    print("  - 预期价格：$7,000-$10,000")
    print("  - 实际价格：$432,500")
    print("- 来源：Robbie Barrat, Obvious")
    print()

def why_optimal_transport():
    """
    解释为什么我们需要最优传输
    - 最优传输找到将一个数据分布转换为另一个的最小成本
    - 它不仅告诉我们两个分布是否不同，还告诉我们如何从一个转换到另一个
    - 最优传输尊重几何结构
    - 应用领域：计算机视觉、自然语言处理、基因组学等
    """
    print("2. Why we need optimal transport?")
    print("- 最优传输（OT）找到将一个数据分布转换为另一个的最小成本")
    print("- 它不仅告诉我们两个分布是否不同，还告诉我们如何从一个转换到另一个")
    print("- 最优传输尊重几何结构：")
    print("  - 不好的度量：这两堆土不同")
    print("  - 好的度量（OT）：要将堆A变成堆B，我需要移动10公斤的土，平均移动5米")
    print("- 应用领域：")
    print("  - 计算机视觉（比较图像/颜色）")
    print("  - 自然语言处理（比较文档含义）")
    print("  - 基因组学（对齐细胞数据）")
    print("  - 任何数据结构重要的问题")

def visualize_ot_concept():
    """
    可视化最优传输的概念
    展示两个分布之间的转换成本
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # 分布A
    ax1.set_title('分布 A')
    ax1.bar([0, 1, 2, 3], [0.1, 0.3, 0.4, 0.2], color='blue', alpha=0.7)
    ax1.set_ylim(0, 0.5)
    ax1.set_xticks([0, 1, 2, 3])
    ax1.set_xticklabels(['A1', 'A2', 'A3', 'A4'])
    
    # 分布B
    ax2.set_title('分布 B')
    ax2.bar([0, 1, 2, 3], [0.2, 0.2, 0.3, 0.3], color='red', alpha=0.7)
    ax2.set_ylim(0, 0.5)
    ax2.set_xticks([0, 1, 2, 3])
    ax2.set_xticklabels(['B1', 'B2', 'B3', 'B4'])
    
    # 传输计划
    ax3.set_title('最优传输计划')
    
    # 绘制两个分布
    for i in range(4):
        ax3.bar(i, [0.1, 0.3, 0.4, 0.2][i], color='blue', alpha=0.7)
        ax3.bar(i + 5, [0.2, 0.2, 0.3, 0.3][i], color='red', alpha=0.7)
    
    # 绘制传输箭头
    # 简化的传输计划示例
    transfers = [
        (0, 0, 0.1),  # A1 -> B1, 0.1
        (1, 1, 0.2),  # A2 -> B2, 0.2
        (1, 2, 0.1),  # A2 -> B3, 0.1
        (2, 2, 0.2),  # A3 -> B3, 0.2
        (2, 3, 0.2),  # A3 -> B4, 0.2
        (3, 3, 0.1),  # A4 -> B4, 0.1
        (3, 0, 0.1)   # A4 -> B1, 0.1
    ]
    
    for from_idx, to_idx, amount in transfers:
        x1, y1 = from_idx, 0
        x2, y2 = to_idx + 5, 0
        
        # 计算箭头的起点和终点
        start_y = [0.1, 0.3, 0.4, 0.2][from_idx]
        end_y = [0.2, 0.2, 0.3, 0.3][to_idx]
        
        # 绘制箭头
        ax3.arrow(x1, start_y, x2 - x1, 0, 
                  length_includes_head=True, head_width=0.2, head_length=0.3,
                  color='green', alpha=0.5, width=amount * 0.5)
    
    ax3.set_ylim(0, 0.5)
    ax3.set_xticks([0, 1, 2, 3, 5, 6, 7, 8])
    ax3.set_xticklabels(['A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'B4'])
    ax3.set_xlabel('数据点')
    ax3.set_ylabel('概率质量')
    
    plt.tight_layout()
    plt.savefig('./figure/ot_concept.png')
    print("\n已生成OT概念可视化图：./figure/ot_concept.png")
    plt.close()

if __name__ == "__main__":
    gan_art_introduction()
    why_optimal_transport()
    visualize_ot_concept()
