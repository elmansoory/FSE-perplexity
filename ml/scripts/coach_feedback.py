from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ALLOWED = {'jump', 'spin', 'step', 'transition', 'unknown'}
REQUIRED = {'clip_id', 'athlete_id', 'team_id', 'coach_id', 'model_version', 'predicted_label', 'prediction_confidence', 'coach_label'}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--feedback', required=True, help='JSON file containing one feedback object.')
    parser.add_argument('--ledger', required=True, help='Append-only JSONL ledger.')
    args = parser.parse_args()
    record = json.loads(Path(args.feedback).read_text(encoding='utf-8'))
    missing = REQUIRED.difference(record)
    if missing:
        raise ValueError(f'Missing feedback fields: {sorted(missing)}')
    if record['predicted_label'] not in ALLOWED or record['coach_label'] not in ALLOWED:
        raise ValueError('Labels must be in the approved general taxonomy.')
    confidence = float(record['prediction_confidence'])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError('prediction_confidence must be between 0 and 1.')
    record['reviewed_at'] = datetime.now(timezone.utc).isoformat()
    record['coach_verified'] = True
    ledger = Path(args.ledger); ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')

if __name__ == '__main__':
    main()
