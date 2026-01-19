import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert elevation and prominence to numeric (handle any non-numeric issues)
df['elevation (m)'] = pd.to_numeric(df['elevation (m)'], errors='coerce')
df['prominence (m)'] = pd.to_numeric(df['prominence (m)'], errors='coerce')

# Remove rows with NaN after conversion
df = df.dropna(subset=['elevation (m)', 'prominence (m)'])

# Calculate mean and std for elevation and prominence
mean_elev = df['elevation (m)'].mean()
std_elev = df['elevation (m)'].std()
mean_promin = df['prominence (m)'].mean()
std_promin = df['prominence (m)'].std()

# Define thresholds for outliers (2 standard deviations)
upper_elev = mean_elev + 2 * std_elev
lower_elev = mean_elev - 2 * std_elev
upper_promin = mean_promin + 2 * std_promin
lower_promin = mean_promin - 2 * std_promin

# Identify outliers
outliers = []
for idx, row in df.iterrows():
    name = row['peak']
    elev = row['elevation (m)']
    promin = row['prominence (m)']
    
    if (elev > upper_elev or elev < lower_elev) or (promin > upper_promin or promin < lower_promin):
        outliers.append(name)

# Also manually check for negative prominence (should not normally exist)
negative_prominence = df[df['prominence (m)'] < 0]['peak'].tolist()

# Combine both types of abnormalities
final_abnormal = list(set(outliers + negative_prominence))

print(f"Final Answer: {', '.join(final_abnormal)}")