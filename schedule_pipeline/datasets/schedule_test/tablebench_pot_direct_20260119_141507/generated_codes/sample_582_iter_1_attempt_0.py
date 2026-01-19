import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Percentage' to numeric, removing any non-numeric characters if needed
df['Percentage'] = pd.to_numeric(df['Percentage'], errors='coerce')

# Calculate the year-over-year differences in percentage
df['diff'] = df['Percentage'].diff()

# Find the year with the maximum negative difference (largest decrease)
# We exclude the first row since there's no previous year
decrease_df = df[1:].copy()
decrease_df['diff'] = decrease_df['diff'].fillna(0)

# Find the index of the minimum (most negative) difference
min_diff_index = decrease_df['diff'].idxmin()
year_with_max_decrease = decrease_df.loc[min_diff_index, 'year']

print(f"Final Answer: {year_with_max_decrease}")