import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'urban , %' column to numeric for proper comparison
df['urban , %'] = pd.to_numeric(df['urban , %'])
# Find the first year where urban percentage exceeds 50
first_year_above_50 = df[df['urban , %'] > 50]['year (january)'].iloc[0]
print(f"Final Answer: {first_year_above_50}")