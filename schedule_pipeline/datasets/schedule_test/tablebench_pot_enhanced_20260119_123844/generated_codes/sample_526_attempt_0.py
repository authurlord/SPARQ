import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'gold medals' and 'total medals' to numeric
df['gold medals'] = pd.to_numeric(df['gold medals'])
df['total medals'] = pd.to_numeric(df['total medals'])

# Calculate medal ratio
df['medal_ratio'] = df['gold medals'] / df['total medals']

# Find the school with the highest medal ratio
max_ratio_school = df.loc[df['medal_ratio'].idxmax(), 'ensemble']

print(f"Final Answer: {max_ratio_school}")