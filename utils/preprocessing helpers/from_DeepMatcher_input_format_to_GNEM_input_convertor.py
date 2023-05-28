import pandas as pd
from pathlib import Path


datasets = ['train', 'test', 'valid']
for dataset in datasets:
    df = pd.read_csv('data/Compas/' + dataset + '.csv')
    df = df.loc[:, ['left_id', 'right_id', 'label']]
    df = df.rename({'left_id': 'ltable_id', 'right_id': 'rtable_id'}, axis=1)
    Path('data/Compas/GNEM/' + dataset + '.csv').mkdir(parents=True, exist_ok=True)
    df.to_csv('data/Compas/GNEM/' + dataset + '.csv', index=False)
