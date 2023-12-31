# take a csv file and return an npz and json file with the format expected by the mia code
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import json

random_state = 42
test_size = 0.2
inFileName = 'mia/data/network_traffic_data.csv'
outFileName = 'mia/data/Network.npz'
continuousFields = ['No.', 'Time', 'Length']
categoricalFields = ['Source', 'Source Port', 'Destination', 'Destination Port', 'Protocol', 'Info', 'Extra']
problemType = 'multiclass_classification'

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.float):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
data = {"columns":[]}

if inFileName[-4:] != '.csv':
    print(f"Error: inFileName must be .csv file")
    exit(-1)
if outFileName[-4:] != '.npz':
    print(f"Error: outFileName must be .npz file")
    exit(-1)

df = pd.read_csv(inFileName)
new_df = pd.DataFrame()

for col in df.columns:
    if col in continuousFields:
        max = np.max(df[col])
        min = np.min(df[col])
        t = "continuous"
        d = {"max": max, "min": min, "name": col, "type": t}
        data['columns'].append(d)
        new_df[col] = df[col]

    if col in categoricalFields:
        unique = df[col].unique().tolist()
        size = len(unique)
        t = "categorical"
        d = {"i2s": unique, "name": col, "size": size, "type": t}
        data['columns'].append(d)
        new_col = df[col].copy()
        for i in range(size):
            new_col = new_col.replace(unique[i], int(i))
        new_df[col] = new_col

data['problem_type'] = problemType


jsonFilePath = outFileName[:-4] + '.json'
with open(jsonFilePath, "w") as f:
    json.dump(data, f, cls=NpEncoder)

train, test = train_test_split(new_df, test_size=test_size, random_state=random_state)
np.savez(outFileName, train=train, test=test)
print(f'{outFileName} file has been created from {inFileName} with Test Size: {test_size*100}%')
