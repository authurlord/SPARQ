import pandas as pd

df = pd.read_csv('table.csv')
# Convert enrollment to integer
df['enrollment'] = pd.to_numeric(df['enrollment'])
# Calculate pairwise differences in enrollment
min_diff = float('inf')
best_pair = None
for i in range(len(df)):
    for j in range(i + 1, len(df)):
        diff = abs(df.iloc[i]['enrollment'] - df.iloc[j]['enrollment'])
        if diff < min_diff:
            min_diff = diff
            best_pair = (df.iloc[i]['location'], df.iloc[j]['location'])
print(f"Final Answer: {best_pair[0]}, {best_pair[1]}, {min_diff}")