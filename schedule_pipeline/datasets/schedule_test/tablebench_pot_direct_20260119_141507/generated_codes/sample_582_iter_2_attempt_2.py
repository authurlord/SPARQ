import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Percentage' to numeric
df['Percentage'] = pd.to_numeric(df['Percentage'], errors='coerce')

# Calculate year-over-year differences in percentage
df['diff'] = df['Percentage'].diff()

# Find the year with the largest decrease (most negative difference)
decrease_index = df[df['diff'] < 0].index
if len(decrease_index) > 0:
    max_decrease_row = df.loc[decrease_index, 'year'].iloc[(df['diff'].abs().idxmax())]
else:
    max_decrease_row = None

# Get the year with the maximum negative change
max_decrease_year = df[df['diff'] == df['diff'].min()]['year'].values[0] if df['diff'].min() < 0 else None

# Since we want the year where the percentage decreased the most (largest drop), we find the row with the minimum (most negative) diff
if df['diff'].min() < 0:
    max_decrease_year = df['year'].iloc[df['diff'].idxmin()]
else:
    max_decrease_year = None

print(f"Final Answer: {max_decrease_year}")