import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for seasons 2004 to 2007
filtered_df = df[df['season'].isin(['2004', '2005', '2006', '2007'])]
# Count the number of series (rows)
num_series = len(filtered_df)
print(f"Final Answer: {num_series}")