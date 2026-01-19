import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert population and other numerical columns to numeric
df['average annual rainfall (mm)'] = pd.to_numeric(df['average annual rainfall (mm)'], errors='coerce')
df['average annual runoff (mm)'] = pd.to_numeric(df['average annual runoff (mm)'], errors='coerce')

# Drop rows with missing values
df = df.dropna(subset=['average annual rainfall (mm)', 'average annual runoff (mm)'])

# Compute mean and std for rainfall and runoff
mean_rainfall = df['average annual rainfall (mm)'].mean()
std_rainfall = df['average annual rainfall (mm)'].std()

mean_runoff = df['average annual runoff (mm)'].mean()
std_runoff = df['average annual runoff (mm)'].std()

# Identify abnormal regions using Z-score threshold (|Z| > 2)
def is_abnormal(row):
    z_rainfall = abs((row['average annual rainfall (mm)'] - mean_rainfall) / std_rainfall)
    z_runoff = abs((row['average annual runoff (mm)'] - mean_runoff) / std_runoff)
    return z_rainfall > 2 or z_runoff > 2

abnormal_regions = df[df.apply(is_abnormal, axis=1)]['administrative region'].tolist()

print(f"Final Answer: {', '.join(abnormal_regions)}")