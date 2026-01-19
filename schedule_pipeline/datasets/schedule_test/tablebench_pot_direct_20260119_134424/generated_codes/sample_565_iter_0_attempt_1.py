import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for years 1850 to 1870
df_filtered = df[(df['Year'] >= '1850') & (df['Year'] <= '1870')]
# Convert 'Year_1' to numeric (removing commas)
df_filtered['Year_1'] = df_filtered['Year_1'].str.replace(',', '').astype(int)
# Calculate the increase between consecutive years
df_filtered = df_filtered.sort_values('Year')
df_filtered['increase'] = df_filtered['Year_1'].diff()
# Find the row with the maximum increase
max_increase_row = df_filtered.loc[df_filtered['increase'].idxmax()]
# Get the value (Year_1) at that year
final_value = max_increase_row['Year_1']
print(f"Final Answer: {final_value}")