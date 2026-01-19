import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'bötzow' column to numeric
df['bötzow'] = pd.to_numeric(df['bötzow'], errors='coerce')

# Calculate year-on-year differences in 'bötzow'
df['diff_bötzow'] = df['bötzow'].diff()

# Find the year with the maximum increase (excluding the first row, which has no previous year)
max_increase_row = df[df['diff_bötzow'] == df['diff_bötzow'].max()]
year_with_max_increase = max_increase_row.iloc[0]['year']

print(f"Final Answer: {year_with_max_increase}")