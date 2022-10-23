import pandas as pd

filepath = 'D:/00_SARDARCHITECTLABS/ucsd_capstone/datasets/399_buildings.csv'
df = pd.read_csv(filepath)
print(df.columns)