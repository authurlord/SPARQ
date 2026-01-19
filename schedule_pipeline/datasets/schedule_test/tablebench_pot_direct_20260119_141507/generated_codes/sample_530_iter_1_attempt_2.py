import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'change' and 'rating' to numeric, coercing errors to NaN if any
df['change'] = pd.to_numeric(df['change'], errors='coerce')
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

# Calculate rating growth as percentage
df['rating_growth'] = (df['change'] / df['rating']) * 100

# Find the centre with the highest rating growth
max_growth_centre = df.loc[df['rating_growth'].idxmax(), 'centre']

print(f"Final Answer: {max_growth_centre}")