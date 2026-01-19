import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Percentage' column to float
df['Percentage'] = pd.to_numeric(df['Percentage'], errors='coerce')

# Calculate year-on-year differences in percentage
df['diff'] = df['Percentage'].diff()

# Find the year with the maximum negative difference (largest decrease)
decrease_index = df[df['diff'] < 0]['diff'].idxmin()
year_with_max_decrease = df.loc[decrease_index, 'year']

print(f"Final Answer: {year_with_max_decrease}")