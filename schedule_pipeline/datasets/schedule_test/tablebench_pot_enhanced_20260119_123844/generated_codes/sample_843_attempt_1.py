import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'Office started' and 'Office ended' to integers, handling 'Incumbent'
df['Office started'] = df['Office started'].astype(int)
df['Office ended'] = df['Office ended'].replace('Incumbent', 2024).astype(int)

# Calculate tenure in years
df['tenure'] = df['Office ended'] - df['Office started']

# Find the bishop with the longest tenure
max_tenure = df['tenure'].max()
longest_serving_bishop = df.loc[df['tenure'] == max_tenure, 'Name'].values[0]

# Calculate average tenure
avg_tenure = df['tenure'].mean()

# Calculate difference
difference = max_tenure - avg_tenure

print(f"Final Answer: {longest_serving_bishop}, {difference:.1f}")