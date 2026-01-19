import pandas as pd

df = pd.read_csv('table.csv')
# Convert enrollment to integer
df['enrollment'] = pd.to_numeric(df['enrollment'])

# Initialize variables to track the smallest difference and corresponding universities
min_diff = float('inf')
best_pair = None

# Compare each pair of universities
for i in range(len(df)):
    for j in range(i + 1, len(df)):
        diff = abs(df.iloc[i]['enrollment'] - df.iloc[j]['enrollment'])
        if diff < min_diff:
            min_diff = diff
            best_pair = (df.iloc[i]['location'], df.iloc[j]['location'])

print(f"Final Answer: {best_pair[0]}, {best_pair[1]}, {min_diff}")