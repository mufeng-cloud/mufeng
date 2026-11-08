"""Log10 data helpers; training only calls load_development, tests are a separate phase."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from chemprop import data, featurizers, utils
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from models.public_tg_head import encode_digits

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT
SEED = 3003105693


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def load_development():
    provenance = ROOT / 'provenance/public_tg'
    manifest = json.loads((provenance / 'data_manifest.json').read_text())
    path = BASE / 'data/public_tg/training_data.csv'
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['data_sha256']['training_data.csv']:
        raise ValueError('Development CSV hash mismatch')
    frame = pd.read_csv(path)
    if len(frame) != 683:
        raise ValueError('Expected 683 development rows')
    encode_digits(torch.tensor(frame.logTg.to_numpy(), dtype=torch.float64))
    splits = json.loads((provenance / 'folds.json').read_text())
    return frame, splits


def load_test():
    manifest=json.loads((ROOT/'provenance/public_tg/data_manifest.json').read_text())
    path=BASE/'data/public_tg/test_data.csv'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['data_sha256']['test_data.csv']
    frame=pd.read_csv(path)
    assert len(frame)==manifest['n_test']==209
    exclusion=json.loads((ROOT/'provenance/public_tg/test_input_exclusion.json').read_text())
    assert frame.ID.tolist()==exclusion['retained_test_IDs'], 'Unexpected test identities or order'
    assert exclusion['removed_rows']==10 and exclusion['retained_test_rows']==209
    assert manifest['test_exclusion_manifest_sha256']==hashlib.sha256((ROOT/'provenance/public_tg/test_input_exclusion.json').read_bytes()).hexdigest()
    encode_digits(torch.tensor(frame.logTg.to_numpy(),dtype=torch.float64))
    return frame


def make_loader(frame, batch_size, shuffle=False, targets=True):
    points = [data.MoleculeDatapoint(
        utils.make_mol(row.SMILES_clean, keep_h=False, add_h=False),
        np.array([row.logTg]) if targets else None) for row in frame.itertuples()]
    ds = data.MoleculeDataset(points, featurizers.SimpleMoleculeMolGraphFeaturizer())
    return data.build_dataloader(ds, batch_size=batch_size, num_workers=0, shuffle=shuffle)


def metrics(y, p):
    if not np.isfinite(p).all():
        raise ValueError('Nonfinite predictions')
    return {'mae': float(mean_absolute_error(y, p)),
            'rmse': float(np.sqrt(mean_squared_error(y, p))),
            'r2': float(r2_score(y, p))}


def predict_validation(model, frame, trainer, path):
    batches = trainer.predict(model, dataloaders=make_loader(frame, 64, targets=False))
    digits = torch.cat([b['digits'].cpu() for b in batches]).numpy()
    values = torch.cat([b['log10'].cpu() for b in batches]).numpy().reshape(-1).astype(np.float64)
    if len(values) != len(frame) or not np.isfinite(digits).all() or not ((digits >= 0) & (digits <= 9)).all():
        raise ValueError('Invalid validation outputs')
    result = pd.DataFrame({'row_id': frame.index, 'smiles': frame.SMILES_clean,
                           'true_logTg': frame.logTg, 'pred_logTg': values})
    for i, name in enumerate(['thousands', 'hundreds', 'tens']):
        result['raw_' + name] = digits[:, i]
    result.to_csv(path, index=False)
    return metrics(frame.logTg, values)
