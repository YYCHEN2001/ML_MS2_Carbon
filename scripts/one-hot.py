import pandas as pd

data = pd.read_csv('../data/processed/dataset_reduced.csv')

# 对 Cation, Anion 进行One-hot编码
data_encoded = pd.get_dummies(data, columns=['Cation', 'Anion'])

data_encoded.to_csv('../data/manual/dataset_manual.csv', index=True)