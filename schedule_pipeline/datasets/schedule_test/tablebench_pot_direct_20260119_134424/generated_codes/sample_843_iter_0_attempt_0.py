import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Office started' and 'Office ended' to integers
df['Office started'] = pd.to_numeric(df['Office started'], errors='coerce')
df['Office ended'] = pd.to_numeric(df['Office ended'], errors='coerce')

# Calculate tenure for each bishop
df['tenure'] = df['Office ended'] - df['Office started']

# Find the bishop with the longest tenure
max_tenure = df['tenure'].max()
longest_tenure_bishop = df.loc[df['tenure'] == max_tenure, 'Name'].values[0]

# Calculate average tenure
avg_tenure = df['tenure'].mean()

# Compute the difference
difference = max_tenure - avg_tenure

print(f"Final Answer: {longest_tenure_bishop}, {difference:.1f}")