import pandas as pd

df = pd.read_csv('table.csv')
# Filter top 5 ranked countries
top_5 = df[df['rank'].astype(float) <= 5]

# Convert 2010 and 2011 columns to numeric
top_5['2010'] = pd.to_numeric(top_5['2010'])
top_5['2011'] = pd.to_numeric(top_5['2011'])

# Calculate growth rate for each country
growth_rates = (top_5['2011'] - top_5['2010']) / top_5['2010']

# Average growth rate
avg_growth_rate = growth_rates.mean()

# Convert to percentage
percentage_difference = avg_growth_rate * 100

print(f"Final Answer: {percentage_difference:.2f}")