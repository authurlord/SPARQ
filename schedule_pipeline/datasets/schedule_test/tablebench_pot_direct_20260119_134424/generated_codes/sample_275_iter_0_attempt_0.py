import pandas as pd

df = pd.read_csv('table.csv')
# Filter top 3 countries
top_3 = df[df['rank'].isin(['1.0', '2.0', '3.0'])]
# Convert columns to integers
top_3['2009'] = top_3['2009'].astype(int)
top_3['2011'] = top_3['2011'].astype(int)
# Calculate increase for each country
top_3['increase'] = top_3['2011'] - top_3['2009']
# Sum the increases
total_increase = top_3['increase'].sum()
print(f"Final Answer: {total_increase}")