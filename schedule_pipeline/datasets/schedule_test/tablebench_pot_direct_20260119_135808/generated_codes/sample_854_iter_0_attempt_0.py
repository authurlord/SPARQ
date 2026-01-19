import pandas as pd

df = pd.read_csv('table.csv')
# Convert enrollment to integer
df['enrollment'] = pd.to_numeric(df['enrollment'])
# Calculate differences between all pairs
min_diff = float('inf')
closest_unis = ('', '')

for i in range(len(df)):
    for j in range(i + 1, len(df)):
        diff = abs(df.iloc[i]['enrollment'] - df.iloc[j]['enrollment'])
        if diff < min_diff:
            min_diff = diff
            closest_unis = (df.iloc[i]['location'], df.iloc[j]['location'])

print(f"Final Answer: {closest_unis[0]}, {closest_unis[1]}, {min_diff}")