import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Office started' and 'Office ended' to integers
df['Office started'] = pd.to_numeric(df['Office started'], errors='coerce')
df['Office ended'] = pd.to_numeric(df['Office ended'], errors='coerce')

# Calculate tenure
df['tenure'] = df['Office ended'] - df['Office started']

# Find the bishop with the longest tenure
max_tenure = df['tenure'].max()
avg_tenure = df['tenure'].mean()

# Find the name of the bishop with the longest tenure
longest_serving_bishop = df.loc[df['tenure'] == max_tenure, 'Name'].iloc[0]

# Calculate the difference between longest and average tenure
difference = max_tenure - avg_tenure

print(f"Final Answer: {longest_serving_bishop}, {difference:.1f}")