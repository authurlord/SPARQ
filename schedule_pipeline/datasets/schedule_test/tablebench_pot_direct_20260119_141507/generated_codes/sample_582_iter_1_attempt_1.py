import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Percentage' to numeric
df['Percentage'] = pd.to_numeric(df['Percentage'], errors='coerce')

# Calculate year-over-year differences in percentage
df['diff'] = df['Percentage'].diff()

# Find the year with the largest negative difference (maximum decrease)
# We exclude the first row since there's no previous year
decrease_df = df[1:].copy()
decrease_df['diff'] = decrease_df['diff'].fillna(0)

# Find the row with the minimum (most negative) difference
min_decrease_row = decrease_df.loc[decrease_df['diff'].idxmin()]

# Return the year of that row
final_year = min_decrease_row['year']
print(f"Final Answer: {final_year}")