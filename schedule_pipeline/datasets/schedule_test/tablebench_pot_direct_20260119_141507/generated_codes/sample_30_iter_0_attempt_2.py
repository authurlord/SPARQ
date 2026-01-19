import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for years from 2002 to 2012 inclusive
filtered_df = df[(df['year'] >= 2002) & (df['year'] <= 2012)]
# Sum the 'mintage' column
total_mintage = filtered_df['mintage'].sum()
print(f"Final Answer: {total_mintage}")