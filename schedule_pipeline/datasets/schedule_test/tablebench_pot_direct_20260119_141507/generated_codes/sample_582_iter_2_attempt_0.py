import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Percentage' column to numeric, removing any non-numeric characters
df['Percentage'] = pd.to_numeric(df['Percentage'].str.replace('%', ''), errors='coerce')

# Sort by year to ensure chronological order
df = df.sort_values('year')

# Calculate the year-over-year difference in percentage
df['percentage_diff'] = df['Percentage'].diff()

# Find the year with the maximum decrease (most negative)
max_decrease_row = df[df['percentage_diff'] == df['percentage_diff'].min()]
decrease_year = max_decrease_row.iloc[0]['year']

print(f"Final Answer: {decrease_year}")