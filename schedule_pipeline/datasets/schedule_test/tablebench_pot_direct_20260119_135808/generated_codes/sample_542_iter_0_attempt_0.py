import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for year 2004
df_2004 = df[df['year'] == '2004']
# Calculate total mintage for 2004
total_mintage_2004 = df_2004['mintage'].sum()
print(f"Final Answer: {total_mintage_2004}")