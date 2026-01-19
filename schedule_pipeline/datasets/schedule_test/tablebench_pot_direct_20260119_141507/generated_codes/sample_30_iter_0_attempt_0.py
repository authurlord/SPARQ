import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where year is between 2002 and 2012 inclusive
filtered_df = df[(df['year'] >= 2002) & (df['year'] <= 2012)]
# Sum the mintage values
total_mintage = filtered_df['mintage'].sum()
print(f"Final Answer: {total_mintage}")