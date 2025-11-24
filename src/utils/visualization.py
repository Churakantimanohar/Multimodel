import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, labels, save_path=None):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return cm


def plot_training_curves(history, save_dir):
    plt.figure(figsize=(6,4))
    for key in ['loss','val_loss']:
        if key in history:
            plt.plot(history[key], label=key)
    plt.legend(); plt.title('Loss Curves'); plt.tight_layout()
    plt.savefig(f"{save_dir}/loss_curves.png")

    plt.figure(figsize=(6,4))
    for key in ['f1','val_f1']:
        if key in history:
            plt.plot(history[key], label=key)
    plt.legend(); plt.title('F1 Curves'); plt.tight_layout()
    plt.savefig(f"{save_dir}/f1_curves.png")
