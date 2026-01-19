import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Year' to integer for proper comparison
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# Calculate the difference in Conservative councillors from the previous year
df['conservative_diff'] = df['Conservative councillors'].diff()

# Find the year with the maximum increase (positive difference)
max_increase_row = df[df['conservative_diff'] == df['conservative_diff'].max()]
year_with_max_increase = max_increase_row.iloc[0]['Year']

print(f"Final Answer: {year_with_max_increase}")