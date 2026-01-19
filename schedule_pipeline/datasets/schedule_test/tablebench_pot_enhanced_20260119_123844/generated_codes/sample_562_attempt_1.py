import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for years 2002 to 2006
filtered_df = df[(df['year'] >= '2002') & (df['year'] <= '2006')]
# Calculate total mintage
total_mintage = filtered_df['mintage'].sum()
print(f"Final Answer: {total_mintage}")