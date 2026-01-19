import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert all values in the columns to numeric (some might be strings)
df = df.apply(pd.to_numeric, errors='coerce')

# For each term, calculate mean and std
outliers = []

for term in df.columns[1:]:
    values = df[term].dropna()
    mean_val = values.mean()
    std_val = values.std()
    if std_val == 0:
        continue
    # Identify values more than 2 standard deviations from mean
    z_scores = np.abs((values - mean_val) / std_val)
    outlier_indices = z_scores > 2
    for idx in values.index[outlier_indices]:
        outliers.append({
            'term': term,
            'year': df.index[idx],
            'value': values.iloc[idx]
        })

# Print the identified outliers
if outliers:
    for out in outliers:
        print(f"Outlier found: {out['term']} in year {out['year']} with value {out['value']}")
else:
    print("No significant deviations found.")