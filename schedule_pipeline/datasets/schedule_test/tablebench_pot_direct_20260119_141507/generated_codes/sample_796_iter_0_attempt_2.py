import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Filter top 5 countries (rank 1 to 5)
top_5 = df[df['rank'].astype(float).between(1, 5)]

# Convert population columns to integers
top_5['2009'] = pd.to_numeric(top_5['2009'], errors='coerce')
top_5['2010'] = pd.to_numeric(top_5['2010'], errors='coerce')
top_5['2011'] = pd.to_numeric(top_5['2011'], errors='coerce')

# Calculate growth rates
def growth_rate(year1, year2):
    return ((year2 - year1) / year1) * 100

growth_2010_to_2011 = top_5[['2010', '2011']].apply(lambda x: growth_rate(x['2010'], x['2011']), axis=1)
growth_2009_to_2010 = top_5[['2009', '2010']].apply(lambda x: growth_rate(x['2009'], x['2010']), axis=1)

avg_growth_2010_2011 = growth_2010_to_2011.mean()
avg_growth_2009_2010 = growth_2009_to_2010.mean()

# Percentage difference between the two average growth rates
percentage_difference = abs(avg_growth_2010_2011 - avg_growth_2009_2010) / ((avg_growth_2010_2011 + avg_growth_2009_2010) / 2) * 100

print(f"Final Answer: {percentage_difference:.2f}")