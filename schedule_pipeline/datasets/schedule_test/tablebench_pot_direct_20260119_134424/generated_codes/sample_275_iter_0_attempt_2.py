import pandas as pd

df = pd.read_csv('table.csv')
# Filter top 3 countries
top_3 = df[df['rank'].isin(['1.0', '2.0', '3.0'])]
# Convert 2009 and 2011 columns to integers
top_3['2009'] = top_3['2009'].astype(int)
top_3['2011'] = top_3['2011'].astype(int)
# Calculate increase for each country and sum
total_increase = (top_3['2011'] - top_3['2009']).sum()
print(f"Final Answer: {total_increase}")