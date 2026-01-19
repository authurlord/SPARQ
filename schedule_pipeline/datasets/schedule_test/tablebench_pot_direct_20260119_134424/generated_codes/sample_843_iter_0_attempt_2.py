import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'Office started' and 'Office ended' to integers
df['Office started'] = pd.to_numeric(df['Office started'], errors='coerce')
df['Office ended'] = pd.to_numeric(df['Office ended'], errors='coerce')

# Calculate tenure for each bishop
df['tenure'] = df['Office ended'] - df['Office started']

# Find the bishop with the longest tenure
max_tenure_bishop = df.loc[df['tenure'].idxmax()]
longest_tenure = max_tenure_bishop['tenure']
average_tenure = df['tenure'].mean()
difference = longest_tenure - average_tenure

print(f"Final Answer: {max_tenure_bishop['Name']}, {difference:.1f}")