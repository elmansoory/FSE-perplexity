from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

class TemporalSkeletonNet(nn.Module):
    def __init__(self, channels: int, classes: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=(9, 1), padding=(4, 0)),
            nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=(9, 1), padding=(4, 0)),
            nn.BatchNorm2d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv2d(hidden, hidden * 2, kernel_size=(9, 1), padding=(4, 0)),
            nn.BatchNorm2d(hidden * 2), nn.ReLU(), nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden * 2, classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(-1)
        x = self.features(x)
        x = x.mean(dim=(2, 3))
        return self.head(x)

def load_npz(path: Path):
    item = np.load(path)
    return torch.from_numpy(item['X']), torch.from_numpy(item['y'])

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    seed = config['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    data_dir = Path(args.data); output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    X_train, y_train = load_npz(data_dir / 'train.npz')
    X_test, y_test = load_npz(data_dir / 'test.npz')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TemporalSkeletonNet(X_train.shape[1], len(config['classes']), config['model']['hidden_channels'], config['model']['dropout']).to(device)
    counts = torch.bincount(y_train, minlength=len(config['classes'])).float().clamp_min(1)
    weights = (counts.sum() / (len(counts) * counts)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['training']['learning_rate']), weight_decay=float(config['training']['weight_decay']))
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=int(config['training']['batch_size']), shuffle=True)
    best_f1, stale = -1.0, 0
    for epoch in range(int(config['training']['epochs'])):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.cross_entropy(model(xb), yb, weight=weights)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            prediction = model(X_test.to(device)).argmax(1).cpu().numpy()
        score = f1_score(y_test.numpy(), prediction, average='macro', zero_division=0)
        if score > best_f1:
            best_f1, stale = score, 0
            torch.save({'state_dict': model.state_dict(), 'config': config, 'channels': int(X_train.shape[1]), 'best_macro_f1': float(best_f1)}, output / 'model.pt')
        else:
            stale += 1
            if stale >= int(config['training']['patience']):
                break
    (output / 'metrics.json').write_text(json.dumps({'best_test_macro_f1': best_f1, 'selection': 'athlete-disjoint holdout', 'training_scope': json.loads((data_dir / 'metadata.json').read_text())['team_id']}, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
