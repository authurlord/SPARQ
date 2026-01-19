import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for years between 2000 and 2005 inclusive
filtered_df = df[(df['year'] >= 2000) & (df['year'] <= 2005)]
# Sum the mintage
total_mintage = filtered_df['mintage'].sum()
print(f"Final Answer: {total_mintage}")