import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert all values to numeric (some entries are strings)
df = df.apply(pd.to_numeric, errors='coerce')

# Identify anomalies:
# - Values greater than 1000 in any municipality (extreme high)
# - Values less than 1 (extreme low)
# - Any municipality where one language dominates (e.g., >90% of total)

# First, check for extreme values
extreme_high = df[df.gt(1000).any(axis=1)]
extreme_low = df[df.lt(1).any(axis=1)]

# List anomalies
anomalies = []

# Check for extremely high values
for col in df.columns[1:]:
    if df[col].max() > 1000:
        anomalies.append(f"High value in {col}: {df[col].max()}")

# Check for extremely low values
for col in df.columns[1:]:
    if df[col].min() < 1:
        anomalies.append(f"Low value in {col}: {df[col].min()}")

# Check for dominance (e.g., one language has >90% of total in a municipality)
for col in df.columns[1:]:
    row_sums = df[col].fillna(0).astype(float)
    # Normalize per municipality
    total_per_mun = row_sums.sum()
    for idx, val in enumerate(row_sums):
        if total_per_mun > 0:
            # Find max value in this municipality
            max_val = row_sums.max()
            if max_val / total_per_mun > 0.9 and max_val > 10:
                lang_name = df.iloc[idx, 0]
                anomalies.append(f"Dominant language in {col}: {lang_name} ({max_val/total_per_mun:.1%})")

# Print anomalies
print("Final Answer:", ", ".join(anomalies))