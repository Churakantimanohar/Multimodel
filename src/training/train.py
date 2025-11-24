import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ..utils.config import get_config
from ..utils.logging_utils import get_logger
from ..models.text_model import TextEncoder
from ..models.audio_model import AudioEncoder
from ..models.video_model import VideoEncoder
from ..models.fusion.attention_fusion import MultiModalAttentionFusion
from ..models.classifier import FusionClassifier
from ..preprocessing.text_preprocess import preprocess_text_batch
from ..preprocessing.audio_preprocess import extract_audio_features
from ..preprocessing.video_preprocess import process_video
from ..utils.visualization import plot_training_curves
from .metrics import compute_metrics

logger = get_logger()
cfg = get_config()

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, length: int = 40):
        self.length = length
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        return {
            'text': 'synthetic sample for mental health detection',
            'audio_path': None,
            'video_path': None,
            'input_ids': torch.randint(0,100,(cfg.max_seq_len,)),
            'attention_mask': torch.ones(cfg.max_seq_len),
            'mfcc': torch.randn(120, cfg.n_mfcc),
            'pitch': torch.randn(60),
            'energy': torch.randn(60),
            'jitter': torch.randn(1),
            'shimmer': torch.randn(1),
            'frames': torch.randn(1,3,cfg.video_frame_size,cfg.video_frame_size),
            'landmarks': torch.randn(32,468,3),
            'label': torch.randint(0,cfg.num_classes,(1,)).item()
        }


def collate(batch_list):
    texts = [b['text'] for b in batch_list]
    labels = torch.tensor([b['label'] for b in batch_list])
    # Pre-tokenize here for manifest/dummy uniformity
    tok = preprocess_text_batch(texts)
    # Audio batch: stack mfcc if present else zeros
    mfcc_list = []
    pitch_list, energy_list, jitter_list, shimmer_list = [], [], [], []
    frames_batch = []
    landmarks_list = []
    for b in batch_list:
        mfcc_list.append(b['mfcc'])
        pitch_list.append(b['pitch'])
        energy_list.append(b['energy'])
        jitter_list.append(b['jitter'])
        shimmer_list.append(b['shimmer'])
        frames_batch.append(b['frames'])
        landmarks_list.append(b['landmarks'])
    mfcc = torch.stack(mfcc_list)  # (B,T,n_mfcc)
    frames = torch.cat(frames_batch, dim=0)
    return tok, mfcc, pitch_list, energy_list, jitter_list, shimmer_list, frames, landmarks_list, labels


def train():
    Path('outputs').mkdir(exist_ok=True)
    manifest_path = 'data/train_manifest.csv'
    if os.path.exists(manifest_path):
        from src.data.manifest_dataset import ManifestMultimodalDataset
        logger.info('Using manifest dataset')
        raw_dataset = ManifestMultimodalDataset(manifest_path)
        # Wrap to produce tensors similar to dummy
        class Wrapped(torch.utils.data.Dataset):
            def __init__(self, base):
                self.base = base
            def __len__(self):
                return len(self.base)
            def __getitem__(self, idx):
                s = self.base[idx]
                # Audio features
                if s['audio_feats']:
                    mfcc = torch.tensor(s['audio_feats']['mfcc'], dtype=torch.float32)
                    pitch = s['audio_feats']['pitch']
                    energy = s['audio_feats']['energy']
                    jitter = s['audio_feats']['jitter']
                    shimmer = s['audio_feats']['shimmer']
                else:
                    mfcc = torch.zeros(64, cfg.n_mfcc)
                    pitch = energy = torch.zeros(10)
                    jitter = torch.zeros(1)
                    shimmer = torch.zeros(1)
                if s['video_feats']:
                    landmarks = torch.tensor(s['video_feats']['landmarks'], dtype=torch.float32)
                else:
                    landmarks = torch.zeros(32,468,3)
                frames = torch.zeros(1,3,cfg.video_frame_size,cfg.video_frame_size)
                return {
                    'text': s['text'],
                    'mfcc': mfcc,
                    'pitch': pitch,
                    'energy': energy,
                    'jitter': jitter,
                    'shimmer': shimmer,
                    'frames': frames,
                    'landmarks': landmarks,
                    'label': s['label']
                }
        dataset = Wrapped(raw_dataset)
    else:
        logger.info('Manifest not found; using dummy dataset')
        dataset = DummyDataset()
    loader = DataLoader(dataset, batch_size=cfg.batch_size, collate_fn=collate, shuffle=True)

    text_enc = TextEncoder().to(cfg.device)
    audio_enc = AudioEncoder().to(cfg.device)
    video_enc = VideoEncoder().to(cfg.device)
    fusion = MultiModalAttentionFusion(dim_text=256, dim_audio=256, dim_video=256, fusion_dim=cfg.fusion_hidden_dim).to(cfg.device)
    classifier = FusionClassifier(seq_dim=cfg.fusion_hidden_dim, num_classes=cfg.num_classes).to(cfg.device)

    params = list(text_enc.parameters()) + list(audio_enc.parameters()) + list(video_enc.parameters()) + list(fusion.parameters()) + list(classifier.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    history = {'loss': [], 'f1': []}

    for epoch in range(cfg.num_epochs):
        text_enc.train(); audio_enc.train(); video_enc.train(); fusion.train(); classifier.train()
        epoch_losses = []
        y_true, y_pred = [], []
        for batch in loader:
            (tok, mfcc, pitch_list, energy_list, jitter_list, shimmer_list, frames, landmarks_list, labels) = batch
            input_ids = tok['input_ids'].to(cfg.device); attention_mask = tok['attention_mask'].to(cfg.device); labels = labels.to(cfg.device)
            text_vec = text_enc(input_ids=input_ids, attention_mask=attention_mask)
            audio_vec = audio_enc(mfcc.to(cfg.device), pitch_list, energy_list, jitter_list, shimmer_list)
            # Prepare landmarks batch
            lm_batch = []
            for lm in landmarks_list:
                lm_batch.append(lm if isinstance(lm, torch.Tensor) else torch.tensor(lm))
            landmarks = torch.stack(lm_batch)
            video_vec = video_enc(frames.to(cfg.device), landmarks.to(cfg.device))

            fused_pooled, fused_seq = fusion(text_vec, audio_vec, video_vec)
            logits = classifier(fused_seq)
            loss = criterion(logits, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_losses.append(loss.item())

            preds = torch.argmax(logits, dim=1).detach().cpu().tolist()
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(preds)

        metrics = compute_metrics(y_true, y_pred)
        avg_loss = sum(epoch_losses)/len(epoch_losses)
        history['loss'].append(avg_loss)
        history['f1'].append(metrics['f1'])
        logger.info(f"Epoch {epoch+1}/{cfg.num_epochs} loss={avg_loss:.4f} f1={metrics['f1']:.4f} acc={metrics['accuracy']:.4f}")

        if (epoch + 1) % 5 == 0:
            ckpt_path = f"outputs/ckpt_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch+1,
                'text_enc': text_enc.state_dict(),
                'audio_enc': audio_enc.state_dict(),
                'video_enc': video_enc.state_dict(),
                'fusion': fusion.state_dict(),
                'classifier': classifier.state_dict(),
                'history': history
            }, ckpt_path)
            logger.info(f"Saved checkpoint {ckpt_path}")

    torch.save(history, 'outputs/history.pt')
    # Always save a final checkpoint for inference convenience
    final_ckpt_path = 'outputs/ckpt_final.pt'
    torch.save({
        'epoch': cfg.num_epochs,
        'text_enc': text_enc.state_dict(),
        'audio_enc': audio_enc.state_dict(),
        'video_enc': video_enc.state_dict(),
        'fusion': fusion.state_dict(),
        'classifier': classifier.state_dict(),
        'history': history
    }, final_ckpt_path)
    logger.info(f"Saved final checkpoint {final_ckpt_path}")
    plot_training_curves(history, 'outputs')
    logger.info("Training complete")

if __name__ == "__main__":
    train()
