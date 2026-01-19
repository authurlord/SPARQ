import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for years 1850 to 1870
filtered_df = df[(df['Year'] >= '1850') & (df['Year'] <= '1870')]
# Convert the value column to numeric (removing commas)
filtered_df['value'] = filtered_df['Year_1'].str.replace(',', '').astype(int)
# Calculate the increase from the previous year
filtered_df = filtered_df.sort_values('Year')
filtered_df['increase'] = filtered_df['value'].diff()
# Find the year with the highest increase
max_increase_year = filtered_df.loc[filtered_df['increase'].idxmax()]
print(f"Final Answer: {max_increase_year['value']}")