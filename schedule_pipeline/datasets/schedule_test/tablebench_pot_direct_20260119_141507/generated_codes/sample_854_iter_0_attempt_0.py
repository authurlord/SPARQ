import pandas as pd

df = pd.read_csv('table.csv')
# Convert enrollment to numeric and filter out non-numeric or invalid entries
df['enrollment'] = pd.to_numeric(df['enrollment'], errors='coerce')
df = df.dropna(subset=['enrollment'])

# Sort by enrollment to help identify close values
df_sorted = df.sort_values(by='enrollment')

# Compute pairwise differences between consecutive entries
diffs = []
for i in range(len(df_sorted) - 1):
    diff = abs(df_sorted['enrollment'].iloc[i] - df_sorted['enrollment'].iloc[i + 1])
    diffs.append((df_sorted['location'].iloc[i], df_sorted['location'].iloc[i + 1], diff))

# Find the pair with the smallest difference
min_diff = min(diffs, key=lambda x: x[2])
print(f"Final Answer: {min_diff[0]}, {min_diff[1]}, {min_diff[2]}")