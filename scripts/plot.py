import os
from diffcsp.common.utils import log_hyperparameters, PROJECT_ROOT
import re,sys
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from IPython import embed

# parser log file
def parse_logs(log_text):
    metrics = defaultdict(list)
    
    for line in log_text.split('\n')[3:-2]:
        if '[metrics][INFO]' not in line:
            continue

        raw_str = re.search(r'\{.*\}', line).group()
        json_str = raw_str.replace("'", '"')
        data = json.loads(json_str)

        epoch = data.pop('epoch')
        if epoch >= len(metrics['epoch']):
            metrics['epoch'].append(epoch)
            for key in data:
                metrics[key].append(data[key])

    return metrics

# plot loss curve
def plot_losses(metrics, run_path, expname):
    
    # compare training and validation loss
    plt.subplot(2, 2, 1)
    plot_metric(metrics, 'train_loss_epoch', f'training loss {expname}')
    plot_metric(metrics, 'val_loss', f'validation loss {expname}')
    plt.yscale('log')
    plt.xscale('log')
    plt.title('Total Loss')
    
    # lattice loss
    plt.subplot(2, 2, 2)
    plot_metric(metrics, 'lattice_loss_epoch', f'train lattice {expname}')
    plot_metric(metrics, 'val_lattice_loss', f'val lattice {expname}')
    plt.yscale('log')
    plt.xscale('log')
    plt.title('Lattice Loss')
    
    # coord loss
    plt.subplot(2, 2, 3)
    plot_metric(metrics, 'coord_loss_epoch', f'train coord {expname}')
    plot_metric(metrics, 'val_coord_loss', f'val coord {expname}')
    plt.yscale('log')
    plt.xscale('log')
    plt.title('Coordinate Loss')

def plot_metric(metrics, key, label, **style):
    if key in metrics:
        values = [np.mean(epoch_data) for epoch_data in metrics[key]]
        plt.plot(metrics['epoch'][3:], values[3:], label=label, **style)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

plt.figure(figsize=(12, 8))
num_exp = len(sys.argv) - 1
for expname in sys.argv[1:]:
    run_path = os.path.join(PROJECT_ROOT, "hydra/singlerun", expname)
    log_file = os.path.join(run_path, "run.metrics.log")
    with open(log_file) as f:
        log_data = parse_logs(f.read())
    plot_losses(log_data,run_path,expname)

plt.tight_layout()
plt.savefig(f'hydra/training_metrics.png', dpi=300)
