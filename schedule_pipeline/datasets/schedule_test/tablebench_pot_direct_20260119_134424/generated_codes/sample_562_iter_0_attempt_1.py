import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for years 2002 to 2006
filtered_df = df[df['year'].isin(['2002', '2004', '2005', '2006'])]
# Sum the mintage values
total_mintage = filtered_df['mintage'].sum()
print(f"Final Answer: {total_mintage}")