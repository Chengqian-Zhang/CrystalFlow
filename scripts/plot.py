import os
from diffcsp.common.utils import log_hyperparameters, PROJECT_ROOT
import re,sys
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from IPython import embed

# 解析日志文件
def parse_logs(log_text):
    metrics = defaultdict(list)
    
    for line in log_text.split('\n')[3:-2]:
        if '[metrics][INFO]' not in line:
            continue
        
        # 提取JSON数据
        raw_str = re.search(r'\{.*\}', line).group()
        json_str = raw_str.replace("'", '"')  # 关键修改
        data = json.loads(json_str)
        
        # 处理epoch数据
        epoch = data.pop('epoch')
        if epoch >= len(metrics['epoch']):
            metrics['epoch'].append(epoch)
            for key in data:
                metrics[key].append(data[key])

    return metrics

# 绘制损失曲线
def plot_losses(metrics, run_path):
    plt.figure(figsize=(12, 8))
    
    # 训练与验证损失对比
    plt.subplot(2, 2, 1)
    plot_metric(metrics, 'train_loss_epoch', 'training loss', color='blue')
    plot_metric(metrics, 'val_loss', 'validation loss', color='orange')
    plt.yscale('log')
    plt.xscale('log')
    plt.title('Total Loss')
    
    # 晶格损失
    plt.subplot(2, 2, 2)
    plot_metric(metrics, 'lattice_loss_epoch', 'train lattice', color='green')
    plot_metric(metrics, 'val_lattice_loss', 'val lattice', color='red')
    plt.yscale('log')
    plt.xscale('log')
    plt.title('Lattice Loss')
    
    # 坐标损失
    plt.subplot(2, 2, 3)
    plot_metric(metrics, 'coord_loss_epoch', 'train coord', color='purple')
    plot_metric(metrics, 'val_coord_loss', 'val coord', color='brown')
    plt.yscale('log')
    plt.xscale('log')
    plt.title('Coordinate Loss')
    
    # 损失分量分布
    '''
    plt.subplot(2, 2, 4)
    plot_metric(metrics, 'train_loss_step', 'Step Loss', color='gray', alpha=0.3)
    plot_metric(metrics, 'train_loss_epoch', 'Epoch Loss', color='blue')
    plt.title('Training Loss Components')
    '''
    
    plt.tight_layout()
    plt.savefig(f'{run_path}/training_metrics.png', dpi=300)

# 辅助绘图函数
def plot_metric(metrics, key, label, **style):
    if key in metrics:
        values = [np.mean(epoch_data) for epoch_data in metrics[key]]
        plt.plot(metrics['epoch'][3:], values[3:], label=label, **style)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

expname=sys.argv[1]
run_path = os.path.join(PROJECT_ROOT, "hydra/singlerun", expname)
log_file = os.path.join(run_path, "run.metrics.log")

with open(log_file) as f:
    log_data = parse_logs(f.read())

plot_losses(log_data,run_path)
