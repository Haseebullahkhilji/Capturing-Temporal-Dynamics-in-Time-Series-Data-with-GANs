"""
Command:
    python preprocess.py --input data/raw/sp500.csv --out_dir data/processed --seq_len 2 --stride 2

"""
import os
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def make_dirs(path):
    os.makedirs(path, exist_ok=True)

def feature_engineer(df):
    # require: date, Open, High, Low, Close, Volume (if available)
    df = df.copy()
    # fill missing price columns if exist
    for c in ['Open','High','Low','Close','Volume']:
        if c not in df.columns:
            df[c] = 0.0
    df['log_close'] = np.log(df['Close'].replace(0, np.nan)).fillna(method='ffill').fillna(0)
    df['returns'] = df['log_close'].diff().fillna(0)
    df['sma_7'] = df['Close'].rolling(7, min_periods=1).mean()
    df = df[['Open','High','Low','Close','Volume','returns','sma_7']]
    return df

def scale_and_window(df, seq_len=60, stride=1, out_dir='data/processed'):
    arr = df.values.astype(float)
    scaler = StandardScaler()
    arr_scaled = scaler.fit_transform(arr)

    seqs = []
    L = seq_len
    for i in range(0, arr_scaled.shape[0] - L + 1, stride):
        seqs.append(arr_scaled[i:i+L])
    seqs = np.stack(seqs) if len(seqs) > 0 else np.zeros((0, L, arr_scaled.shape[1]))

    joblib.dump(scaler, os.path.join(out_dir, 'scaler.pkl'))
    np.save(os.path.join(out_dir, 'sequences.npy'), seqs)

    meta = {
        'n_sequences': int(seqs.shape[0]),
        'seq_len': int(L),
        'n_features': int(arr_scaled.shape[1]),
        'columns': df.columns.tolist()
    }
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    return seqs, scaler, meta

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='CSV file path with date column')
    p.add_argument('--out_dir', default='data/processed')
    p.add_argument('--seq_len', type=int, default=60)
    p.add_argument('--stride', type=int, default=1)
    args = p.parse_args()

    make_dirs(args.out_dir)
    print('Loading', args.input)

    df = pd.read_csv(args.input, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)

    df_fe = feature_engineer(df)
    df_fe = df_fe.fillna(method='ffill').fillna(method='bfill')

    seqs, scaler, meta = scale_and_window(
        df_fe, seq_len=args.seq_len, stride=args.stride, out_dir=args.out_dir
    )

    print('Saved sequences:', seqs.shape)
    print('Meta saved to', args.out_dir + '/meta.json')

if __name__ == '__main__':
    main()
