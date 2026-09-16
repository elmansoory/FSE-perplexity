from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ba = a - b
    bc = c - b
    denom = np.maximum(np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1), 1e-6)
    cosine = np.clip(np.sum(ba * bc, axis=-1) / denom, -1.0, 1.0)
    return np.arccos(cosine) / np.pi

def temporal_features(landmarks: np.ndarray) -> np.ndarray:
    if landmarks.ndim != 3 or landmarks.shape[1:] != (33, 4):
        raise ValueError('Expected landmarks with shape (T, 33, 4).')
    xyz = landmarks[..., :3].astype(np.float32)
    visibility = np.clip(landmarks[..., 3:4].astype(np.float32), 0.0, 1.0)
    hip_center = (xyz[:, LEFT_HIP] + xyz[:, RIGHT_HIP]) / 2.0
    shoulder_center = (xyz[:, LEFT_SHOULDER] + xyz[:, RIGHT_SHOULDER]) / 2.0
    scale = np.linalg.norm(shoulder_center - hip_center, axis=1, keepdims=True)
    scale = np.maximum(scale, 1e-4)
    normalized = (xyz - hip_center[:, None, :]) / scale[:, None, :]
    velocity = np.diff(normalized, axis=0, prepend=normalized[:1])
    acceleration = np.diff(velocity, axis=0, prepend=velocity[:1])
    left_knee = _angle(normalized[:, LEFT_HIP], normalized[:, LEFT_KNEE], normalized[:, LEFT_ANKLE])
    right_knee = _angle(normalized[:, RIGHT_HIP], normalized[:, RIGHT_KNEE], normalized[:, RIGHT_ANKLE])
    trunk = shoulder_center - hip_center
    trunk_lean = np.arctan2(np.linalg.norm(trunk[:, [0, 2]], axis=1), np.maximum(np.abs(trunk[:, 1]), 1e-6)) / np.pi
    global_features = np.stack([left_knee, right_knee, trunk_lean], axis=-1)
    global_features = np.repeat(global_features[:, None, :], 33, axis=1)
    return np.concatenate([normalized, velocity, acceleration, visibility, global_features], axis=-1)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input .npz containing landmarks (T,33,4).')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    item = np.load(args.input)
    key = 'landmarks' if 'landmarks' in item else 'sequence'
    raw = item[key]
    if raw.ndim == 2 and raw.shape[1] == 132:
        raw = raw.reshape(raw.shape[0], 33, 4)
    features = temporal_features(raw)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, features=features, pose_coverage=float((raw[..., 3] >= 0.5).mean()))

if __name__ == '__main__':
    main()
