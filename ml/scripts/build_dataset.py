from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

CLASSES = ['jump', 'spin', 'step', 'transition', 'unknown']

def resize_time(sequence: np.ndarray, frames: int) -> np.ndarray:
    if len(sequence) == frames:
        return sequence
    indices = np.linspace(0, len(sequence) - 1, frames).astype(int)
    return sequence[indices]

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', required=True)
    parser.add_argument('--team-id', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--frames', type=int, default=120)
    args = parser.parse_args()
    labels = pd.read_csv(args.labels)
    required = {'clip_id', 'athlete_id', 'team_id', 'label', 'coach_verified', 'landmark_path'}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(f'Missing CSV fields: {sorted(missing)}')
    labels = labels[(labels.team_id.astype(str) == str(args.team_id)) & (labels.coach_verified.astype(str).str.lower() == 'true')]
    labels = labels[labels.label.isin(CLASSES)].copy()
    if labels.empty:
        raise ValueError('No coach-verified rows found for this team.')
    sequences, targets, athletes, clips = [], [], [], []
    for row in labels.itertuples(index=False):
        path = Path(row.landmark_path)
        if not path.exists():
            continue
        item = np.load(path)
        key = 'features' if 'features' in item else 'sequence'
        seq = item[key]
        if seq.ndim != 3 or seq.shape[1] != 33:
            continue
        seq = resize_time(seq, args.frames)
        sequences.append(seq.transpose(2, 0, 1)[..., None])
        targets.append(CLASSES.index(row.label))
        athletes.append(str(row.athlete_id))
        clips.append(str(row.clip_id))
    if len(sequences) < 10:
        raise ValueError('Need at least 10 valid coach-labeled sequences.')
    X = np.stack(sequences).astype(np.float32)
    y = np.asarray(targets, dtype=np.int64)
    groups = np.asarray(athletes)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_index, test_index = next(splitter.split(X, y, groups))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'train.npz', X=X[train_index], y=y[train_index], clips=np.asarray(clips)[train_index], athletes=groups[train_index])
    np.savez_compressed(output / 'test.npz', X=X[test_index], y=y[test_index], clips=np.asarray(clips)[test_index], athletes=groups[test_index])
    metadata = {'team_id': args.team_id, 'classes': CLASSES, 'frames': args.frames, 'shape': list(X.shape), 'coach_verified_only': True, 'athlete_disjoint_split': True}
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
