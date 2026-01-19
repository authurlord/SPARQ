import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'rating' and 'change' columns to numeric
df['rating'] = pd.to_numeric(df['rating'])
df['change'] = pd.to_numeric(df['change'])

# Calculate rating growth as a percentage
df['rating_growth'] = (df['change'] / df['rating']) * 100

# Find the center with the highest rating growth
max_growth_center = df.loc[df['rating_growth'].idxmax(), 'centre']

print(f"Final Answer: {max_growth_center}")