import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for the years 2005/2006 to 2007/2008
filtered_df = df[(df['year'] == '2005 / 2006') | (df['year'] == '2006 / 2007') | (df['year'] == '2007 / 2008')]
# Sum the 'increase in net assets' for these rows
total_increase = filtered_df['increase in net assets'].sum()
print(f"Final Answer: {total_increase}")