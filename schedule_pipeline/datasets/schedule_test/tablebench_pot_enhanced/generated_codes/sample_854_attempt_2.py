import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'enrollment' to integer
df['enrollment'] = df['enrollment'].astype(int)

# Initialize variables to track minimum difference and corresponding universities
min_diff = float('inf')
university1 = ""
university2 = ""

# Compare each pair of universities
for i in range(len(df)):
    for j in range(i + 1, len(df)):
        diff = abs(df.iloc[i]['enrollment'] - df.iloc[j]['enrollment'])
        if diff < min_diff:
            min_diff = diff
            university1 = df.iloc[i]['location']
            university2 = df.iloc[j]['location']

print(f"Final Answer: {university1}, {university2}, {min_diff}")