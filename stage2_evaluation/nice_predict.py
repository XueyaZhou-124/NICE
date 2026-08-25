import argparse
import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

PHE_COLS = ['Morphology', 'typeMorphology', 'typeMorphology2']
LABEL_COL = 'label'


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Predict with saved NICE final model.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to final_model.joblib')
    parser.add_argument('--data_path', type=str, required=True, help='Path to new data CSV')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save prediction CSV')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for predicted label')
    parser.add_argument('--fill_value', type=float, default=0.0, help='Fill value for missing feature values')
    parser.add_argument(
        '--strict_schema',
        action='store_true',
        help='If set, raise error when input columns do not exactly match training features.',
    )
    return parser.parse_args()


def load_model_bundle(model_path: str) -> Dict:
    bundle = joblib.load(model_path)
    required_keys = ['feature_columns', 'scaler', 'selector', 'pca', 'clf']
    missing_keys = [k for k in required_keys if k not in bundle]
    if missing_keys:
        raise KeyError(f'Invalid model artifact, missing keys: {missing_keys}')
    return bundle


def align_features(
    data: pd.DataFrame,
    feature_columns: List[str],
    fill_value: float,
    strict_schema: bool,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    data = data.copy()
    drop_cols = [c for c in [LABEL_COL] + PHE_COLS if c in data.columns]
    if drop_cols:
        data = data.drop(columns=drop_cols)

    data = data.apply(pd.to_numeric, errors='coerce')
    feature_set = set(feature_columns)
    current_set = set(data.columns)
    missing_cols = sorted(feature_set - current_set)
    extra_cols = sorted(current_set - feature_set)

    if strict_schema and (missing_cols or extra_cols):
        raise ValueError(
            f'Column schema mismatch. Missing: {missing_cols}, Extra: {extra_cols}'
        )

    aligned = data.reindex(columns=feature_columns)
    aligned = aligned.fillna(fill_value)
    return aligned, missing_cols, extra_cols


def predict_scores(bundle: Dict, X: pd.DataFrame, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    X_new = bundle['scaler'].transform(X)
    if bundle['selector'] is not None:
        X_new = bundle['selector'].transform(X_new)
    if bundle['pca'] is not None:
        X_new = bundle['pca'].transform(X_new)

    clf = bundle['clf']
    if hasattr(clf, 'predict_proba'):
        y_score = clf.predict_proba(X_new)[:, -1]
    else:
        y_score = clf.predict(X_new).astype(float)
    y_pred = (y_score >= threshold).astype(int)
    return y_score, y_pred


def main() -> None:
    args = get_args()
    bundle = load_model_bundle(args.model_path)
    data = pd.read_csv(args.data_path, index_col=0)

    feature_columns = list(bundle['feature_columns'])
    X, missing_cols, extra_cols = align_features(
        data,
        feature_columns=feature_columns,
        fill_value=args.fill_value,
        strict_schema=args.strict_schema,
    )
    y_score, y_pred = predict_scores(bundle, X=X, threshold=args.threshold)

    pred_df = pd.DataFrame(
        {
            'pred_score': y_score,
            'pred_label': y_pred,
        },
        index=data.index,
    )

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    pred_df.to_csv(args.save_path)

    print(f'samples: {len(pred_df)}')
    print(f'prediction file saved to: {args.save_path}')
    if missing_cols:
        print(f'missing columns filled with {args.fill_value}: {len(missing_cols)}')
    if extra_cols:
        print(f'extra columns ignored: {len(extra_cols)}')


if __name__ == '__main__':
    main()
