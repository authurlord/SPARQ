import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for the specified years
filtered_df = df[df['year'].isin(['2005 / 2006', '2006 / 2007', '2007 / 2008'])]
# Sum the 'increase in net assets' column
total_increase = filtered_df['increase in net assets'].sum()
print(f"Final Answer: {total_increase}")