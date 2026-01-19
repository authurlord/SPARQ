import pandas as pd

df = pd.read_csv('table.csv')
# Filter top 5 ranked countries
top_5 = df[df['rank'].astype(float) <= 5]
# Convert '2010' and '2011' columns to numeric
top_5['2010'] = pd.to_numeric(top_5['2010'])
top_5['2011'] = pd.to_numeric(top_5['2011'])
# Calculate growth rate from 2010 to 2011
top_5['growth_rate'] = (top_5['2011'] - top_5['2010']) / top_5['2010'] * 100
# Calculate average growth rate
avg_growth_rate = top_5['growth_rate'].mean()
print(f"Final Answer: {avg_growth_rate:.2f}")