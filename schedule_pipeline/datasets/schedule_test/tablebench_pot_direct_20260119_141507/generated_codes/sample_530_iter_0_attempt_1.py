import pandas as pd

df = pd.read_csv('table.csv')
# Calculate rating growth as (change / rating) * 100
df['rating_growth'] = (df['change'] / df['rating']) * 100
# Find the centre with the highest rating growth
highest_growth_centre = df.loc[df['rating_growth'].idxmax(), 'centre']
print(f"Final Answer: {highest_growth_centre}")