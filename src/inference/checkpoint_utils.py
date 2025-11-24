import os
import glob
import torch


def find_latest_checkpoint(directory: str = 'outputs'):
    if not os.path.isdir(directory):
        return None
    # Prefer explicit final checkpoint
    final_path = os.path.join(directory, 'ckpt_final.pt')
    if os.path.exists(final_path):
        return final_path
    pattern = os.path.join(directory, 'ckpt_epoch_*.pt')
    files = glob.glob(pattern)
    if not files:
        return None
    # Extract epoch numbers
    def epoch_num(path):
        try:
            base = os.path.basename(path)
            num = base.split('_')[-1].split('.')[0]
            return int(num)
        except Exception:
            return -1
    files.sort(key=epoch_num, reverse=True)
    return files[0]


def load_checkpoint(models: dict, path: str):
    """models: mapping name->module"""
    ckpt = torch.load(path, map_location='cpu')
    loaded = []
    for name, module in models.items():
        key = name if name in ckpt else name  # direct
        if key in ckpt:
            module.load_state_dict(ckpt[key])
            loaded.append(name)
    return ckpt, loaded


def load_latest(models: dict, directory: str = 'outputs'):
    path = find_latest_checkpoint(directory)
    if not path:
        return None, []
    ckpt, loaded = load_checkpoint(models, path)
    return path, loaded
