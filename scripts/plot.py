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
    
    for line in log_text.split('\n')[30:-2]:
        if '[metrics][INFO]' not in line:
            continue

        raw_str = re.search(r'\{.*\}', line).group()
        json_str = raw_str.replace("'", '"')
        data = json.loads(json_str)

        epoch = data.pop('epoch')
        if (epoch >= len(metrics['epoch'])) and (len(data.keys())==18):
            metrics['epoch'].append(epoch)
            for key in data:
                metrics[key].append(data[key])

    return metrics

# plot loss curve
def plot_losses(metrics, run_path, expname, valid=False, train=True):
    
    # compare training and validation loss
    plt.subplot(2, 2, 1)
    if train:
        plot_metric(metrics, 'train_loss_epoch', f'training loss {expname}')
    if valid:
        plot_metric(metrics, 'val_loss', f'validation loss {expname}')
    plt.yscale('log')
    plt.xscale('log')
    plt.title('Total Loss')
    
    # lattice loss
    plt.subplot(2, 2, 2)
    if train:
        plot_metric(metrics, 'lattice_loss_epoch', f'train lattice {expname}')
    if valid:
        plot_metric(metrics, 'val_lattice_loss', f'val lattice {expname}')
    plt.yscale('log')
    plt.xscale('log')
    plt.title('Lattice Loss')
    
    # coord loss
    plt.subplot(2, 2, 3)
    if train:
        plot_metric(metrics, 'coord_loss_epoch', f'train coord {expname}')
    if valid:
        plot_metric(metrics, 'val_coord_loss', f'val coord {expname}')
    plt.yscale('log')
    plt.xscale('log')
    plt.title('Coordinate Loss')

    # type loss
    plt.subplot(2, 2, 4)
    if train:
        plot_metric(metrics, 'type_loss_epoch', f'train type {expname}')
    if valid:
        plot_metric(metrics, 'val_type_loss', f'val type {expname}')
    plt.yscale('log')
    plt.xscale('log')
    plt.title('Type Loss')

def plot_metric(metrics, key, label, **style):
    if key in metrics:
        values = [np.mean(epoch_data) for epoch_data in metrics[key]]
        plt.plot(metrics['epoch'][3:], values[3:], label=label, **style)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

plt.figure(figsize=(14, 7))
num_exp = len(sys.argv) - 1
plot_valid = True if "true" in sys.argv[-1] else False
plot_train = True if "true" in sys.argv[-2] else False
for expname in sys.argv[1:-2]:
    run_path = os.path.join(PROJECT_ROOT, "hydra/singlerun", expname)
    log_file = os.path.join(run_path, "run.metrics.log")
    with open(log_file) as f:
        log_data = parse_logs(f.read())
    plot_losses(log_data,run_path,expname,valid=plot_valid, train=plot_train)

plt.tight_layout()
plt.savefig(f'hydra/training_metrics.png', dpi=300)
