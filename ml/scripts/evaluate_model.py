from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix

from train_general_classifier import TemporalSkeletonNet

def conservative_decision(probabilities, pose_coverage, config):
    order = np.argsort(probabilities)[::-1]
    top, second = probabilities[order[0]], probabilities[order[1]]
    if pose_coverage < config['data']['minimum_pose_coverage']:
        return 'unknown', 'insufficient_pose_coverage'
    if top < config['decision']['accept_confidence']:
        return 'unknown', 'below_accept_threshold'
    if top - second < config['decision']['minimum_margin']:
        return 'unknown', 'ambiguous_prediction'
    return config['classes'][int(order[0])], 'accepted'

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', default='ml/evaluations/general.json')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    checkpoint = torch.load(args.model, map_location='cpu')
    test = np.load(Path(args.data) / 'test.npz')
    model = TemporalSkeletonNet(checkpoint['channels'], len(config['classes']), config['model']['hidden_channels'], config['model']['dropout'])
    model.load_state_dict(checkpoint['state_dict']); model.eval()
    logits = model(torch.from_numpy(test['X'])).detach().numpy()
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True)); probabilities /= probabilities.sum(axis=1, keepdims=True)
    raw = probabilities.argmax(axis=1)
    labels = config['classes']
    report = classification_report(test['y'], raw, labels=range(len(labels)), target_names=labels, output_dict=True, zero_division=0)
    matrix = confusion_matrix(test['y'], raw, labels=range(len(labels))).tolist()
    decisions = [conservative_decision(p, 1.0, config) for p in probabilities]
    accepted = sum(reason == 'accepted' for _, reason in decisions)
    result = {'classes': labels, 'classification_report': report, 'confusion_matrix': matrix, 'accepted_predictions': accepted, 'unknown_or_review_predictions': len(decisions) - accepted, 'policy': config['decision']}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
