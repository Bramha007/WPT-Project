import os
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch 
from src.setup import new_config_cls as config 

DEFAULT_CLASS_NAMES = config.AREA_NAMES 

def save_confusion_matrix(y_true, y_pred, out_path="outputs/confmat_area.png", class_names=DEFAULT_CLASS_NAMES):
    """Saves the Confusion Matrix for the Area Classification Task (5x5)."""
    num_classes = len(class_names) 
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes))) 
    
    fig, ax = plt.subplots(figsize=(8,8)) 
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True label (Area)', xlabel='Predicted label (Area)',
           title='Area Classification Confusion Matrix')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close(fig)
    return out_path

def save_sample_grid(images, y_true, y_pred, out_path="outputs/preds_grid_area.png", max_samples=36, class_names=DEFAULT_CLASS_NAMES):
    """Saves a grid of predictions for the Area Classification Task."""
    # de-normalize for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    n = min(len(images), max_samples)
    cols = 6
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.3, rows*2.3))
    axes = np.array(axes).reshape(rows, cols)

    for i in range(rows*cols):
        ax = axes.flat[i]
        ax.axis("off")
        if i >= n: 
            continue
        img = images[i].cpu() * std + mean
        img = img.clamp(0,1).permute(1,2,0).numpy()

        t = int(y_true[i]); p = int(y_pred[i])
        ok = (t == p)
        # Use the Area Class Names
        title = f"T:{class_names[t]} • P:{class_names[p]}" 
        
        ax.imshow(img)
        ax.set_title(title, color=("green" if ok else "red"), fontsize=8) 

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close(fig)
    return out_path