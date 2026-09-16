# Coach-guided skating classifier

This module trains a team-scoped, conservative skeleton action classifier. Only coach-verified rows are eligible for training. The model predicts `jump`, `spin`, `step`, `transition`, or `unknown`; it returns `unknown` when confidence, pose coverage, or prediction margin does not meet the configured threshold.

## Layout

- `scripts/extract_features.py`: landmarks to normalized temporal skeleton channels
- `scripts/build_dataset.py`: coach-labeled `.npz` sequences to team-scoped ST-GCN tensors and athlete-disjoint splits
- `scripts/train_general_classifier.py`: conservative temporal skeleton classifier
- `scripts/evaluate_model.py`: macro-F1, confusion matrix, athlete-level evaluation, and Unknown policy
- `scripts/coach_feedback.py`: validates and appends immutable coach correction records
- `configs/general_stgcn.yaml`: training and decision configuration

## Label file

Create `ml/data/labels/labels.csv` with:

```csv
clip_id,athlete_id,team_id,label,coach_verified,landmark_path
clip_001,athlete_01,team_a,jump,true,ml/data/landmarks/clip_001.npz
```

Landmark `.npz` files must contain `landmarks` with shape `(T, 33, 4)` in MediaPipe order: `x, y, z, visibility`.

## Train

```bash
python ml/scripts/build_dataset.py --labels ml/data/labels/labels.csv --team-id team_a --output ml/data/processed/team_a
python ml/scripts/train_general_classifier.py --config ml/configs/general_stgcn.yaml --data ml/data/processed/team_a --output ml/models/team_a_general
python ml/scripts/evaluate_model.py --model ml/models/team_a_general/model.pt --data ml/data/processed/team_a --config ml/configs/general_stgcn.yaml
```

## Feedback loop

Each coach correction is saved as an append-only JSONL record containing model version, prediction, confidence, corrected label, reviewer ID, timestamp, team ID, and clip ID. Rebuild the dataset and retrain only after an approved review batch; do not overwrite an existing model without recording evaluation metrics.

## Safety and scope

This is a coaching-assistance tool, not an ISU judging system. Low-confidence, incomplete-pose, or ambiguous clips must remain `unknown` and require coach review. Keep each team dataset and model output directory separate.
